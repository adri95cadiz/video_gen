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
                 transcribe_audio=False, image_dir=None, voice_reference=None):
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
            voice_reference: Ruta a un archivo de audio para clonar la voz
        """
        # Cargar variables de entorno desde el archivo .env
        load_dotenv()
        
        # Configurar directorios
        self.output_dir = output_dir or os.environ.get("OUTPUT_DIRECTORY", "videos")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Configurar salida del video
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_path = os.path.join(self.output_dir, f"video_{timestamp}.mp4")
        
        # Configurar max_words por defecto
        self.max_words = 200
        
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
            self.voice_generator = VoiceGenerator(use_local_model=True, reference_voice_path=voice_reference)
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
    
    def _create_silent_audio(self, duration, output_path):
        """
        Crea un archivo de audio silencioso
        
        Args:
            duration: Duración en segundos
            output_path: Ruta donde guardar el archivo
            
        Returns:
            str: Ruta al archivo de audio generado
        """
        from pydub import AudioSegment
        
        # Crear un segmento de audio silencioso
        silent_segment = AudioSegment.silent(duration=int(duration * 1000))  # pydub usa milisegundos
        
        # Guardar el archivo
        silent_segment.export(output_path, format="mp3")
        
        return output_path
    
    def _create_text_image(self, text, output_path):
        """
        Crea una imagen con texto para usar como fallback
        
        Args:
            text: Texto a mostrar en la imagen
            output_path: Ruta donde guardar la imagen
            
        Returns:
            str: Ruta a la imagen generada
        """
        try:
            # Intentar usar PIL para crear una imagen con texto
            from PIL import Image, ImageDraw, ImageFont
            import textwrap
            
            # Crear imagen en blanco
            width, height = 1080, 1920  # Tamaño vertical estándar
            background_color = (0, 0, 0)  # Negro
            text_color = (255, 255, 255)  # Blanco
            
            image = Image.new('RGB', (width, height), background_color)
            draw = ImageDraw.Draw(image)
            
            # Usar una fuente por defecto o una fuente del sistema
            try:
                # Intentar cargar una fuente del sistema
                font_path = os.path.join(os.environ.get("SYSTEMROOT", "C:\\Windows"), "Fonts", "arial.ttf")
                font = ImageFont.truetype(font_path, 48)
            except:
                # Si falla, usar una fuente por defecto
                font = ImageFont.load_default()
            
            # Ajustar el texto para que quepa en la imagen
            margin = 100
            wrapper = textwrap.TextWrapper(width=30)  # 30 caracteres por línea
            word_list = wrapper.wrap(text=text)
            
            # Dibujar cada línea de texto centrada
            y_text = height // 3
            for line in word_list:
                # Obtener ancho y alto del texto para centrarlo
                text_width, text_height = draw.textsize(line, font=font)
                position = ((width - text_width) // 2, y_text)
                draw.text(position, line, font=font, fill=text_color)
                y_text += text_height + 10  # Espacio entre líneas
            
            # Guardar la imagen
            image.save(output_path)
            
        except Exception as e:
            # Si falla PIL, intentar crear una imagen simple con OpenCV
            try:
                import cv2
                import numpy as np
                
                # Crear imagen en blanco
                img = np.zeros((1920, 1080, 3), np.uint8)
                
                # Dividir el texto en líneas
                words = text.split()
                lines = []
                current_line = []
                
                for word in words:
                    if len(' '.join(current_line + [word])) <= 30:
                        current_line.append(word)
                    else:
                        lines.append(' '.join(current_line))
                        current_line = [word]
                
                if current_line:
                    lines.append(' '.join(current_line))
                
                # Añadir texto a la imagen
                y_position = 900  # Posición vertical central
                for line in lines:
                    cv2.putText(img, line, (50, y_position), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
                    y_position += 60  # Espacio entre líneas
                
                # Guardar la imagen
                cv2.imwrite(output_path, img)
                
            except Exception as nested_e:
                # Si todo falla, crear una imagen en blanco como último recurso
                try:
                    import numpy as np
                    blank_image = np.zeros((1920, 1080, 3), np.uint8)
                    with open(output_path, 'wb') as f:
                        import cv2
                        cv2.imwrite(output_path, blank_image)
                except:
                    # Si incluso esto falla, crear un archivo en blanco
                    with open(output_path, 'wb') as f:
                        f.write(b'\x00' * 100)
                    print(f"Creada imagen en blanco como fallback último: {output_path}")
        
        return output_path
    
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
    
    def generate_video(self, prompt, max_words=None, output_filename=None, background_music_path=None):
        """
        Genera un video completo a partir de un prompt
        
        Args:
            prompt: Prompt de texto para el contenido del video
            max_words: Número máximo de palabras para el guión (opcional)
            output_filename: Ruta de salida para el video (sobrescribe self.output_path)
            background_music_path: Ruta a un archivo de música para usar como fondo (opcional)
            
        Returns:
            str: Ruta al video generado
        """
        try:
            # Configurar salida
            output_path = output_filename or self.output_path
            
            # Ajustar max_words si se proporcionó
            if max_words:
                self.max_words = max_words
                
            print(f"Generando video para: '{prompt}'")
            
            # 1. Generar guión
            print("Generando guión...")
            script = self.script_generator.generate_script(prompt, self.max_words)
            print(f"Guión generado ({len(script.split())} palabras)")
            
            # 2. Dividir en oraciones para las imágenes y voz
            sentences = self.script_generator.split_script_into_sentences(script)
            print(f"Script dividido en {len(sentences)} segmentos")
            
            # 3. Generar imágenes para cada oración
            print("Generando imágenes...")
            image_paths = []
            for i, sentence in enumerate(sentences):
                try:
                    # Generar imagen
                    image_path = self.image_generator.generate_image(
                        prompt=sentence,
                        output_path=os.path.join(self.temp_dir, f"img_{i}.png")
                    )
                    image_paths.append(image_path)
                    print(f"Imagen {i+1}/{len(sentences)} generada: {os.path.basename(image_path)}")
                except Exception as e:
                    print(f"Error generando imagen para '{sentence}': {str(e)}")
                    # Usar una imagen de placeholder en caso de error
                    image_path = self._create_text_image(sentence, os.path.join(self.temp_dir, f"img_{i}.png"))
                    image_paths.append(image_path)
            
            # 4. Generar audio para cada oración
            print("Generando audio...")
            audio_paths = []
            for i, sentence in enumerate(sentences):
                try:
                    # Generar audio
                    audio_path = self.voice_generator.generate_voice(
                        text=sentence,
                        output_file=os.path.join(self.temp_dir, f"audio_{i}.mp3")
                    )
                    audio_paths.append(audio_path)
                    print(f"Audio {i+1}/{len(sentences)} generado: {os.path.basename(audio_path)}")
                except Exception as e:
                    print(f"Error generando audio para '{sentence}': {str(e)}")
                    # Continuar sin audio para este segmento
                    audio_path = self._create_silent_audio(3.0, os.path.join(self.temp_dir, f"audio_{i}.mp3"))
                    audio_paths.append(audio_path)
                    
            # 5. Crear video final
            print("Creando video final...")
            video_path = self.video_generator.create_video(
                image_paths=image_paths,
                audio_paths=audio_paths,
                sentences=sentences,
                output_path=output_path,
                background_music_path=background_music_path
            )
            
            print(f"Video generado: {video_path}")
            return video_path
            
        except Exception as e:
            print(f"Error generando video: {str(e)}")
            raise e 