import os
import tempfile
from moviepy import *
from moviepy.audio.AudioClip import AudioClip
from moviepy.video.VideoClip import TextClip
import numpy as np
from pydub import AudioSegment
import time
import shutil

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
        Crea un video combinando imágenes, audio y subtítulos
        
        Args:
            image_paths: Lista de rutas a las imágenes
            audio_paths: Lista de rutas a los archivos de audio
            sentences: Lista de oraciones para los subtítulos
            output_path: Ruta de salida para el video
            bg_color: Color de fondo
            
        Returns:
            str: Ruta al video generado
        """
        if len(image_paths) != len(audio_paths) or len(audio_paths) != len(sentences):
            raise ValueError("Las listas de imágenes, audios y oraciones deben tener la misma longitud")
        
        clips = []
        for i, (img_path, audio_path, text) in enumerate(zip(image_paths, audio_paths, sentences)):
            # Obtener duración del audio
            duration = self.get_audio_duration(audio_path)
            
            # Crear clip de imagen
            img_clip = ImageClip(img_path, duration=duration)
            img_clip = img_clip.resize(height=self.height)
            
            # Centrar la imagen horizontalmente
            img_clip = img_clip.set_position('center')
            
            # Crear subtítulo
            fontsize = 40
            txt_clip = TextClip(text, fontsize=fontsize, color='white', bg_color='rgba(0,0,0,0.5)', 
                               size=(self.width-100, None), method='caption')
            txt_clip = txt_clip.set_position(('center', 'bottom')).set_duration(duration)
            
            # Crear clip compuesto
            audio_clip = AudioFileClip(audio_path)
            video_clip = CompositeVideoClip([img_clip, txt_clip], size=(self.width, self.height), bg_color=bg_color)
            video_clip = video_clip.set_audio(audio_clip)
            
            # Añadir transición de fundido (excepto para el primer clip)
            if i > 0:
                video_clip = video_clip.crossfadein(0.5)
                
            clips.append(video_clip)
        
        # Concatenar todos los clips
        final_clip = concatenate_videoclips(clips, method="compose")
        
        # Añadir transiciones de entrada y salida
        final_clip = final_clip.fadein(0.5).fadeout(0.5)
        
        # Exportar video
        try:
            temp_dir = tempfile.gettempdir()
            # Crear directorio para la salida si no existe
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                
            # Usar configuración de baja bitrate para reducir el tamaño
            final_clip.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                bitrate='2000k',
                audio_bitrate='128k',
                fps=24,
                threads=2,
                preset='fast',  # más rápido que 'medium' pero menos compresión
                temp_audiofile=os.path.join(temp_dir, "temp_audio.m4a"),
                remove_temp=True
            )
            
            return output_path
        except Exception as e:
            raise Exception(f"Error al crear el video: {str(e)}")
    
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