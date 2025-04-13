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
            sentences: Lista de oraciones para los subtítulos
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
                
                # Crear un archivo temporal para el subtítulo (formato SRT)
                srt_path = os.path.join(tempfile.gettempdir(), f"subtitle_{i}.srt")
                with open(srt_path, "w", encoding="utf-8") as f:
                    # Formato SRT: número, tiempo inicio --> tiempo fin, texto
                    f.write(f"1\n00:00:00,000 --> 00:{int(audio_duration//60):02d}:{audio_duration%60:06.3f}\n{sentences[i]}\n\n")
                
                # Corregir la ruta para Windows reemplazando barras invertidas y escapando adecuadamente
                img_path_escaped = img_path.replace('\\', '/')
                audio_path_escaped = audio_path.replace('\\', '/')
                temp_video_escaped = temp_video.replace('\\', '/')
                
                # Escapar texto para ffmpeg (reemplazar comillas, etc.)
                texto_subtitulo = sentences[i].replace("'", "\\'").replace('"', '\\"')
                
                # Dividir el texto en líneas más cortas para mejor legibilidad
                lineas_subtitulos = self._dividir_texto_en_lineas(texto_subtitulo, 40)
                
                # Crear múltiples comandos de drawtext para cada línea con diferentes posiciones verticales
                drawtext_commands = []
                line_height = 55  # Altura estimada para cada línea con tamaño de fuente 48
                
                for idx, linea in enumerate(lineas_subtitulos):
                    # Calcular la posición vertical para cada línea
                    y_position = f"h-{(len(lineas_subtitulos)-idx)*line_height + 100}"
                    
                    # Crear el comando drawtext para esta línea
                    drawtext_cmd = (
                        f"drawtext=text='{linea}':x=(w-text_w)/2:y={y_position}:"
                        f"fontsize=48:fontcolor=white:font=bold:box=1:boxcolor=black@0.8:boxborderw=10"
                    )
                    drawtext_commands.append(drawtext_cmd)
                
                # Concatenar los comandos drawtext con comas para aplicar filtros en secuencia
                drawtext_filters = ",".join(drawtext_commands)
                
                # Crear comando ffmpeg para combinar imagen, audio y usar drawtext en lugar de subtitles
                cmd = (
                    f"ffmpeg -y -loop 1 -i \"{img_path_escaped}\" -i \"{audio_path_escaped}\" -c:v libx264 "
                    f"-t {audio_duration} -pix_fmt yuv420p -vf \"scale={self.width}:{self.height}:force_original_aspect_ratio=decrease,"
                    f"pad={self.width}:{self.height}:(ow-iw)/2:(oh-ih)/2,"
                    f"{drawtext_filters}\" "
                    f"\"{temp_video_escaped}\""
                )
                
                # Ejecutar comando
                print(f"Procesando clip {i+1}/{len(image_paths)}...")
                print(f"Comando ffmpeg: {cmd[:150]}...")  # Para depuración (mostrar solo el inicio)
                os.system(cmd)
                
                if os.path.exists(temp_video):
                    temp_videos.append(temp_video)
                    
                # Eliminar archivo temporal de subtítulos
                if os.path.exists(srt_path):
                    os.remove(srt_path)
                    
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
                # Usar formato de rutas compatible con ffmpeg en Windows (barras normales)
                video_path = video.replace('\\', '/')
                f.write(f"file '{video_path}'\n")
        
        # Corregir ruta para Windows
        concat_file_escaped = concat_file.replace('\\', '//')
        output_path_escaped = output_path.replace('\\', '//')
        
        # Crear comando de concatenación
        concat_cmd = f"ffmpeg -y -f concat -safe 0 -i \"{concat_file_escaped}\" -c copy \"{output_path_escaped}\""
        
        # Ejecutar comando
        print("Combinando clips...")
        print(f"Comando de concatenación: {concat_cmd}")  # Para depuración
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
    
    def _dividir_texto_en_lineas(self, texto, max_caracteres_por_linea=40):
        """
        Divide un texto largo en líneas para una mejor visualización de subtítulos
        
        Args:
            texto: Texto a dividir
            max_caracteres_por_linea: Número máximo de caracteres por línea
            
        Returns:
            list: Lista de líneas de texto
        """
        palabras = texto.split()
        lineas = []
        linea_actual = []
        longitud_actual = 0
        
        for palabra in palabras:
            # +1 por el espacio entre palabras
            nueva_longitud = longitud_actual + len(palabra) + (1 if longitud_actual > 0 else 0)
            
            if nueva_longitud > max_caracteres_por_linea and longitud_actual > 0:
                # Agregar la línea actual y empezar una nueva
                lineas.append(" ".join(linea_actual))
                linea_actual = [palabra]
                longitud_actual = len(palabra)
            else:
                # Agregar palabra a la línea actual
                linea_actual.append(palabra)
                longitud_actual = nueva_longitud
        
        # Agregar la última línea
        if linea_actual:
            lineas.append(" ".join(linea_actual))
        
        return lineas
    
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