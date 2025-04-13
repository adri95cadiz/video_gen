import os
import tempfile
from moviepy import VideoFileClip, AudioFileClip, concatenate_videoclips, ImageClip
from pydub import AudioSegment
import time
import shutil
import torch

class VideoGenerator:
    def __init__(self, width=1080, height=1920):
        """
        Inicializa el generador de videos
        
        Args:
            width: Ancho del video (por defecto 1080px para formato vertical)
            height: Alto del video (por defecto 1920px para formato vertical)
        """
        self.width = width
        self.height = height
        
        # Intentar configurar ffmpeg
        self._setup_ffmpeg()
        
    def _setup_ffmpeg(self):
        """Configura la ruta a FFmpeg"""
        # Intenta usar FFmpeg del PATH
        ffmpeg_path = shutil.which('ffmpeg')
        
        # Si no está en el PATH, intenta con la ubicación común en Windows
        if not ffmpeg_path:
            possible_paths = [
                os.path.expanduser("~\\ffmpeg\\bin\\ffmpeg.exe"),
                "C:\\ffmpeg\\bin\\ffmpeg.exe",
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    # Establecer configuración para moviepy
                    os.environ["IMAGEIO_FFMPEG_EXE"] = path
                    print(f"FFmpeg encontrado en: {path}")
                    break
                    
    def get_audio_duration(self, audio_path):
        """
        Obtiene la duración de un archivo de audio
        
        Args:
            audio_path: Ruta al archivo de audio
            
        Returns:
            float: Duración en segundos
        """
        audio = AudioSegment.from_file(audio_path)
        return len(audio) / 1000.0  # Convertir de ms a segundos
    
    def create_video(self, image_paths, audio_paths, sentences, output_path, bg_color="black"):
        """
        Crea un video combinando imágenes y audio (versión simplificada)
        
        Args:
            image_paths: Lista de rutas a las imágenes
            audio_paths: Lista de rutas a los archivos de audio
            sentences: Lista de oraciones para los subtítulos (no usados en versión simple)
            output_path: Ruta de salida para el video
            bg_color: Color de fondo (no usado en versión simple)
            
        Returns:
            str: Ruta al video generado
        """
        print("Usando generador de video simplificado para modelos locales")
        
        if len(image_paths) != len(audio_paths):
            raise ValueError("Las listas de imágenes y audios deben tener la misma longitud")
        
        # Crear archivos de video temporales a partir de cada par imagen-audio
        temp_videos = []
        
        for i, (img_path, audio_path) in enumerate(zip(image_paths, audio_paths)):
            try:
                # Crear un archivo de video temporal
                temp_video = os.path.join(tempfile.gettempdir(), f"temp_video_{i}.mp4")
                
                # Obtener la duración del audio
                audio_duration = self.get_audio_duration(audio_path)
                
                # Crear comando ffmpeg para combinar imagen y audio
                cmd = (
                    f"ffmpeg -y -loop 1 -i \"{img_path}\" -i \"{audio_path}\" -c:v libx264 "
                    f"-t {audio_duration} -pix_fmt yuv420p -vf \"scale={self.width}:{self.height}:force_original_aspect_ratio=decrease,"
                    f"pad={self.width}:{self.height}:(ow-iw)/2:(oh-ih)/2\" \"{temp_video}\""
                )
                
                # Ejecutar comando
                print(f"Procesando clip {i+1}/{len(image_paths)}...")
                os.system(cmd)
                
                if os.path.exists(temp_video):
                    temp_videos.append(temp_video)
            except Exception as e:
                print(f"Error al procesar clip {i}: {str(e)}")
        
        # Si no se generaron videos temporales, salir
        if not temp_videos:
            raise Exception("No se pudieron generar videos temporales")
        
        # Si solo hay un video, usarlo directamente
        if len(temp_videos) == 1:
            shutil.copy(temp_videos[0], output_path)
            return output_path
        
        # Combinar videos con ffmpeg
        concat_file = os.path.join(tempfile.gettempdir(), "concat_list.txt")
        with open(concat_file, "w") as f:
            for video in temp_videos:
                f.write(f"file '{video}'\n")
        
        # Crear comando de concatenación
        concat_cmd = f"ffmpeg -y -f concat -safe 0 -i \"{concat_file}\" -c copy \"{output_path}\""
        
        # Ejecutar comando
        print("Combinando clips...")
        os.system(concat_cmd)
        
        # Limpiar archivos temporales
        try:
            if os.path.exists(concat_file):
                os.remove(concat_file)
            for video in temp_videos:
                if os.path.exists(video):
                    os.remove(video)
        except Exception as e:
            print(f"Error al limpiar archivos temporales: {str(e)}")
        
        return output_path
    
    def clean_temp_files(self, file_paths):
        """
        Elimina archivos temporales
        
        Args:
            file_paths: Lista de rutas a archivos a eliminar
        """
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error al eliminar archivo temporal {file_path}: {str(e)}") 