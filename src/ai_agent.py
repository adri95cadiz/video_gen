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
    def __init__(self, output_dir="videos", use_local_models=False):
        """
        Inicializa el agente de IA para generar videos
        
        Args:
            output_dir: Directorio donde se guardarán los videos generados
            use_local_models: Si se deben usar modelos locales en lugar de APIs externas
        """
        # Cargar variables de entorno desde el archivo .env
        load_dotenv()
        
        # Configurar si usar modelos locales
        self.use_local_models = use_local_models
        
        # Configurar directorios
        self.output_dir = output_dir or os.environ.get("OUTPUT_DIRECTORY", "videos")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Inicializar servicios
        if use_local_models:
            print("🔄 Configurando con modelos locales (sin costes de API)...")
            # Usar modelos locales (path del modelo puede especificarse en .env)
            texto_model_path = os.environ.get("LOCAL_TEXT_MODEL_PATH", "tiiuae/falcon-7b-instruct")
            imagen_model_path = os.environ.get("LOCAL_IMAGE_MODEL_PATH", "stabilityai/stable-diffusion-2-1")
            
            self.script_generator = ScriptGenerator(use_local_model=True, local_model_path=texto_model_path)
            self.image_generator = ImageGenerator(use_local_model=True, local_model_path=imagen_model_path)
            # También usar modo local para voz
            self.voice_generator = VoiceGenerator(use_local_model=True)
        else:
            # Usar APIs externas (OpenAI, Stability AI)
            self.script_generator = ScriptGenerator()
            self.image_generator = ImageGenerator()
            # ElevenLabs puede fallar si no hay API key, se manejará internamente
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
    
    def _clean_temp_directories(self):
        """Limpia los directorios temporales"""
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
            os.makedirs(self.temp_audio_dir, exist_ok=True)
            os.makedirs(self.temp_image_dir, exist_ok=True)
        except Exception as e:
            print(f"Error al limpiar directorios temporales: {str(e)}")
    
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
            if self.use_local_models:
                print("💻 Utilizando modelos locales (sin costes de API)")
            else:
                print("🌐 Utilizando APIs externas (OpenAI, Stability AI)")
                
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
            audio_files = self.voice_generator.generate_voice_for_sentences(sentences, self.temp_audio_dir)
            print(f"✅ {len(audio_files)} archivos de audio generados")
            
            # 4. Generar imágenes para cada oración
            print("🖼️ Generando imágenes...")
            image_files = self.image_generator.generate_images_for_sentences(sentences, self.temp_image_dir)
            print(f"✅ {len(image_files)} imágenes generadas")
            
            # 5. Crear el video
            if not output_filename:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"video_{timestamp}.mp4"
                
            output_path = os.path.join(self.output_dir, output_filename)
            
            print("🎥 Creando video...")
            self.video_generator.create_video(image_files, audio_files, sentences, output_path)
            print(f"✅ Video generado: {output_path}")
            
            end_time = time.time()
            duration = end_time - start_time
            print(f"⏱️ Tiempo total: {duration:.2f} segundos")
            
            return output_path
            
        except Exception as e:
            print(f"❌ Error al generar el video: {str(e)}")
            raise 