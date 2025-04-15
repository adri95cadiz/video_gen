import os
import datetime
import shutil
from dotenv import load_dotenv
from src.services.script_generator import ScriptGenerator
from src.services.voice_generator import VoiceGenerator
from src.services.image_generator import ImageGenerator
from src.services.video_generator import VideoGenerator


class AIVideoAgent:
    def __init__(self, image_model=None,
                 local_script=False, local_image=False, local_voice=False, music_reference=None, output_dir="videos",
                 transcribe_audio=False, image_dir=None, script_model=None, style=None, custom_media=None):
        """
        Inicializa el agente de IA para generación de videos.

        Args:
            image_model: Modelo de generación de imágenes
            voice_model: Modelo de generación de voz
            local_script: Si se debe usar un modelo local para el guión
            local_image: Si se debe usar un modelo local para las imágenes
            local_voice: Si se debe usar un modelo local para la voz
            music_reference: Ruta a archivo de música de referencia
            output_dir: Directorio de salida para los videos
            transcribe_audio: Si se debe transcribir audio de entrada
            image_dir: Directorio con imágenes existentes para usar en lugar de generar nuevas
            script_model: Modelo específico de OpenAI a utilizar para la generación de guiones
            style: Diccionario con información del estilo seleccionado por el usuario
            custom_media: Lista de rutas a archivos multimedia proporcionados por el usuario (opcional)
        """
        # Cargar variables de entorno desde el archivo .env
        load_dotenv()

        # Configurar directorios
        self.output_dir = output_dir or os.environ.get(
            "OUTPUT_DIRECTORY", "videos")
        os.makedirs(self.output_dir, exist_ok=True)

        # Configurar salida del video
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_path = os.path.join(
            self.output_dir, f"video_{timestamp}.mp4")

        # Configurar max_words por defecto
        self.max_words = 200

        # Guardar el modelo de script seleccionado y el estilo
        self.script_model = script_model
        self.style = style
        self.custom_media = custom_media or [] # Guardar la lista de medios personalizados
        self.use_custom_media = bool(self.custom_media)

        # Inicializar servicios
        if local_script:
            # Usar modelos locales (path del modelo puede especificarse en .env)
            texto_model_path = os.environ.get(
                "LOCAL_TEXT_MODEL_PATH", "tiiuae/falcon-7b-instruct")
            self.script_generator = ScriptGenerator(
                use_local_model=True, local_model_path=texto_model_path)
        else:
            # Usar APIs externas (OpenAI, Stability AI)
            self.script_generator = ScriptGenerator()

        # Si se usan medios personalizados, no necesitamos el generador de imágenes
        if self.use_custom_media:
            self.image_generator = None
            print("Usando medios personalizados. El generador de imágenes no se inicializará.")
        elif local_image:
            # Usar el modelo especificado o el de .env como respaldo
            if image_model:
                imagen_model_path = image_model
            else:
                imagen_model_path = os.environ.get(
                    "LOCAL_IMAGE_MODEL_PATH", "stabilityai/stable-diffusion-xl-base-1.0")

            print(f"Usando modelo de imagen: {imagen_model_path}")
            self.image_generator = ImageGenerator(
                use_local_model=True, local_model_path=imagen_model_path, image_dir=image_dir)
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

    def generate_video(self, video_topic, max_words=None, num_scenes=5, progress_callback=None):
        """Genera un video basado en un tema proporcionado"""
        # Si se usan medios personalizados, ajustar el número de escenas
        if self.use_custom_media:
            num_scenes = len(self.custom_media)
            print(f"Usando {num_scenes} escenas basadas en los archivos multimedia proporcionados.")
        else:
            num_scenes = num_scenes or 5
            
        self.max_words = max_words or self.max_words
        self._clean_temp_directories()

        # Paso 1: Generar guión basado en el tema
        print(f"🖋️ Generando guión para: {video_topic}")

        if progress_callback:
            progress_callback.on_script_start()

        # Usar el prompt específico del estilo si está disponible
        script_prompt = None
        if self.style and 'script_prompt' in self.style:
            # El script_prompt ya está completo, no necesita parámetros
            script_prompt = self.style['script_prompt']

        script = self.script_generator.generate_script(
            video_topic,
            max_words=self.max_words,
            model=self.script_model,
            custom_prompt=script_prompt
        )

        if not script:
            raise ValueError("No se pudo generar un guión válido")

        if progress_callback:
            progress_callback.on_script_complete(script)

        # Obtener las escenas para el guión
        scenes = self.script_generator.split_into_scenes(script, num_scenes)
        
        # Adaptar el número de escenas al número de medios personalizados si es necesario
        if self.use_custom_media and len(scenes) != num_scenes:
            print(f"Aviso: El número de escenas generadas ({len(scenes)}) no coincide con el número de archivos multimedia ({num_scenes}). Se ajustará.")
            # Si hay más escenas que medios, truncar escenas
            if len(scenes) > num_scenes:
                scenes = scenes[:num_scenes]
            # Si hay menos escenas que medios, duplicar la última escena (o manejar de otra forma)
            elif len(scenes) < num_scenes:
                last_scene = scenes[-1] if scenes else "Escena adicional."
                scenes.extend([last_scene] * (num_scenes - len(scenes)))
        elif len(scenes) != num_scenes:
            print(f"Aviso: El número de escenas generadas ({len(scenes)}) difiere del solicitado ({num_scenes}).")
            num_scenes = len(scenes)

        # Paso 2: Generar imágenes para cada escena
        media_paths = []
        if self.use_custom_media:
            print("🖼️ Usando archivos multimedia proporcionados por el usuario...")
            media_paths = self.custom_media
            if progress_callback:
                progress_callback.update("Usando medios personalizados", 40) # Actualizar progreso
                for i in range(num_scenes):
                    progress_callback.on_image_complete(i + 1, num_scenes)
        else:
            print(f"🎨 Generando {num_scenes} imágenes...")
            if progress_callback:
                progress_callback.on_images_start()
            for i, scene_text in enumerate(scenes):
                # Usar la plantilla de prompt de imagen del estilo si está disponible
                image_prompt = scene_text
                if self.style and 'image_prompt_template' in self.style:
                    image_prompt = self.style['image_prompt_template'].replace(
                        "{tema}", video_topic) + ". El texto de la escena es: " + scene_text

                if self.image_generator:
                    image_path = self.image_generator.generate_image(
                        prompt=image_prompt,
                        output_path=os.path.join(
                            self.temp_image_dir, f"scene_{i+1}.png")
                    )
                    media_paths.append(image_path)
                else:
                    print("Error: image_generator no está inicializado pero se intentó generar imagen.")
                    # Opcional: generar imagen de fallback o lanzar error
                    media_paths.append(None) # Añadir None o manejar error
                    
                if progress_callback:
                    progress_callback.on_image_complete(i + 1, num_scenes)

        # Paso 3: Generar voces para cada escena
        print(f"🔊 Generando {num_scenes} voces...")
        audio_paths = []

        if progress_callback:
            progress_callback.on_voice_start()

        for i, scene_text in enumerate(scenes):
            audio_path = self.voice_generator.generate_voice(
                scene_text,
                output_file=os.path.join(
                    self.temp_audio_dir, f"scene_{i+1}.mp3"),
                lang="es"
            )
            audio_paths.append(audio_path)

        if progress_callback:
            progress_callback.on_voice_complete()

        # Paso 4: Compilar video
        print("🎬 Compilando video final...")

        if progress_callback:
            progress_callback.on_video_start()

        video_path = self.video_generator.create_video(
            media_paths=media_paths,
            audio_paths=audio_paths,
            output_path=self.output_path,
            sentences=scenes,
            background_music_path=self.music_reference
        )

        print(f"✅ Video generado correctamente: {video_path}")

        if progress_callback:
            progress_callback.on_video_complete(video_path)

        # Crear y devolver un objeto con toda la información del video
        video_info = {
            "topic": video_topic,
            "script": script,
            "scenes": scenes,
            "media_paths": media_paths,
            "audio_path": audio_paths,
            "video_path": video_path,
            "style": self.style['name'] if self.style else None
        }

        return video_info
