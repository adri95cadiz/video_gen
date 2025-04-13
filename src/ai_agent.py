import os
import time
import datetime
import shutil
from dotenv import load_dotenv
from src.services.script_generator import ScriptGenerator
from src.services.voice_generator import VoiceGenerator
from src.services.image_generator import ImageGenerator
from src.services.video_generator import VideoGenerator

class AIVideoAgent:
    def __init__(self, llm_provider=None, image_model=None, voice_model=None, 
                 local_script=False, local_image=False, local_voice=False, music_reference=None, output_dir="videos",
                 transcribe_audio=False, image_dir=None):
        """
        Inicializa el agente de IA para generación de videos.
        
        Args:
            llm_provider: Proveedor de LLM a utilizar
            image_model: Modelo de generación de imágenes
            voice_model: Modelo de generación de voz
            local_script: Si se debe usar un modelo local para el guión
            local_image: Si se debe usar un modelo local para las imágenes
            local_voice: Si se debe usar un modelo local para la voz
            music_reference: Ruta a archivo de música de referencia
            output_dir: Directorio de salida para los videos
            transcribe_audio: Si se debe transcribir audio de entrada
            image_dir: Directorio con imágenes existentes para usar en lugar de generar nuevas
        """
        # Cargar variables de entorno desde el archivo .env
        load_dotenv()
        
        # Configurar directorios
        self.output_dir = output_dir or os.environ.get("OUTPUT_DIRECTORY", "videos")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Inicializar servicios
        if local_script:
            # Usar modelos locales (path del modelo puede especificarse en .env)
            texto_model_path = os.environ.get("LOCAL_TEXT_MODEL_PATH", "tiiuae/falcon-7b-instruct")            
            self.script_generator = ScriptGenerator(use_local_model=True, local_model_path=texto_model_path)
        else:
            # Usar APIs externas (OpenAI, Stability AI)
            self.script_generator = ScriptGenerator()

        if local_image:
            imagen_model_path = os.environ.get("LOCAL_IMAGE_MODEL_PATH", "stabilityai/stable-diffusion-2-1")
            self.image_generator = ImageGenerator(use_local_model=True, local_model_path=imagen_model_path, image_dir=image_dir)
        else:
            self.image_generator = ImageGenerator(image_dir=image_dir)

        if local_voice:
            self.voice_generator = VoiceGenerator(use_local_model=True)
        else:
            self.voice_generator = VoiceGenerator()
        
        # Configuración del video
        width = int(os.environ.get("DEFAULT_VIDEO_WIDTH", 1080))
        height = int(os.environ.get("DEFAULT_VIDEO_HEIGHT", 1920))
        self.video_generator = VideoGenerator(width=width, height=height)
        
        # Crear directorios temporales
        self.temp_dir = os.path.join(os.getcwd(), "temp")
        self.temp_audio_dir = os.path.join(self.temp_dir, "audio")
        self.temp_image_dir = os.path.join(self.temp_dir, "images")
        
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.temp_audio_dir, exist_ok=True)
        os.makedirs(self.temp_image_dir, exist_ok=True)
        
        # Otros parámetros
        self.music_reference = music_reference
        self.transcribe_audio = transcribe_audio
    
    def _clean_temp_directories(self):
        """Limpia los directorios temporales"""
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
            os.makedirs(self.temp_audio_dir, exist_ok=True)
            os.makedirs(self.temp_image_dir, exist_ok=True)
        except Exception as e:
            print(f"Error al limpiar directorios temporales: {str(e)}")
    
    def _generate_audio_for_scenes(self, scenes, output_dir, lang="es"):
        """
        Genera audio para todas las escenas.
        
        Args:
            scenes: Lista de escenas con texto
            output_dir: Directorio donde guardar los archivos de audio
            lang: Código de idioma para la generación de voz
            
        Returns:
            dict: Mapa de scenes a rutas de archivos de audio
        """
        audio_files = {}
        os.makedirs(output_dir, exist_ok=True)
        
        for i, scene in enumerate(scenes):
            output_path = os.path.join(output_dir, f"scene_{i}.mp3")
            try:
                # Usar el nuevo método de voice_generator que incluye el parámetro lang
                self.voice_generator.generate_voice(scene, output_path, lang=lang)
                audio_files[i] = output_path
            except Exception as e:
                print(f"Error al generar audio para escena {i}: {str(e)}")
        
        return audio_files
    
    def generate_video(self, prompt, max_words=200, output_filename=None):
        """
        Genera un video a partir de un prompt de texto
        
        Args:
            prompt: El prompt de texto que describe el contenido del video
            max_words: Número máximo de palabras para el guión
            output_filename: Nombre del archivo de salida
            
        Returns:
            str: Ruta al video generado
        """
        try:
            print("🎬 Iniciando generación de video...")
                
            start_time = time.time()
            
            # Limpiar directorios temporales
            self._clean_temp_directories()
            
            # 1. Generar guión
            print("📝 Generando guión...")
            script = self.script_generator.generate_script(prompt, max_words=max_words)
            print(f"✅ Guión generado ({len(script.split())} palabras)")
            print("-" * 40)
            print(script)
            print("-" * 40)
            
            # 2. Dividir el guión en oraciones
            sentences = self.script_generator.split_script_into_sentences(script)
            print(f"🔄 Dividido en {len(sentences)} oraciones")
            
            # 3. Generar audio para cada oración
            print("🎤 Generando narración de voz...")
            audio_files = self._generate_audio_for_scenes(sentences, self.temp_audio_dir)
            print(f"✅ {len(audio_files)} archivos de audio generados")
            
            # 4. Generar imágenes para cada oración
            print("🖼️ Generando imágenes...")
            image_prompts = [f"{prompt}: {sentence}" for sentence in sentences]
            image_files = []
            
            self.image_generator.context = prompt

            for i, img_prompt in enumerate(image_prompts):
                output_path = os.path.join(self.temp_image_dir, f"image_{i}.png")
                try:
                    self.image_generator.generate_image(img_prompt, output_path)
                    image_files.append(output_path)
                except Exception as e:
                    print(f"Error al generar imagen {i}: {str(e)}")
            
            print(f"✅ {len(image_files)} imágenes generadas")
            
            # 5. Crear el video
            if not output_filename:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"video_{timestamp}.mp4"
                
            output_path = os.path.join(self.output_dir, output_filename)
            
            print("🎥 Creando video...")
            self.video_generator.create_video(image_files, list(audio_files.values()), sentences, output_path)
            print(f"✅ Video generado: {output_path}")
            
            end_time = time.time()
            duration = end_time - start_time
            print(f"⏱️ Tiempo total: {duration:.2f} segundos")
            
            return output_path
            
        except Exception as e:
            print(f"❌ Error al generar el video: {str(e)}")
            raise 