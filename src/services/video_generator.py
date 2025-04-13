import os
import tempfile
from moviepy import VideoFileClip, AudioFileClip, concatenate_videoclips, ImageClip
from pydub import AudioSegment
import time
import shutil
import torch
import random

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
                
                # Escapar texto para ffmpeg (reemplazar comillas, etc.)
                texto_subtitulo = sentences[i].replace("'", "\\'").replace('"', '\\"')
                
                # Dividir el texto en fragmentos cortos basados en número de caracteres
                palabras = texto_subtitulo.split()
                fragmentos = []
                fragmento_actual = []
                caracteres_actual = 0
                
                # Máximo de caracteres por fragmento (aproximadamente 20-25 caracteres)
                max_caracteres_por_fragmento = 25
                
                for palabra in palabras:
                    # Calcular caracteres con la nueva palabra (incluyendo espacios)
                    nuevos_caracteres = caracteres_actual + len(palabra) + (1 if fragmento_actual else 0)
                    
                    if nuevos_caracteres > max_caracteres_por_fragmento and fragmento_actual:
                        # Guardar el fragmento actual y comenzar uno nuevo
                        fragmentos.append(" ".join(fragmento_actual))
                        fragmento_actual = [palabra]
                        caracteres_actual = len(palabra)
                    else:
                        # Añadir la palabra al fragmento actual
                        fragmento_actual.append(palabra)
                        caracteres_actual = nuevos_caracteres
                
                # Añadir el último fragmento si queda algo
                if fragmento_actual:
                    fragmentos.append(" ".join(fragmento_actual))
                
                # Variables para controlar el tiempo
                duracion_por_fragmento = audio_duration / len(fragmentos)
                
                # Crear múltiples comandos de drawtext para cada fragmento con diferentes posiciones y tiempos
                drawtext_commands = []
                
                for j, fragmento in enumerate(fragmentos):
                    # Calcular el tiempo para cada fragmento en segundos
                    tiempo_inicio = j * duracion_por_fragmento
                    tiempo_fin = (j + 1) * duracion_por_fragmento
                    
                    # Escapar el texto para FFmpeg
                    fragmento_escapado = fragmento.replace("'", "\\'").replace('"', '\\"')
                    
                    # Crear el comando drawtext para este fragmento con tiempo de inicio y fin
                    drawtext_cmd = (
                        f"drawtext=text='{fragmento_escapado}':x=(w-text_w)/2:y=h-th-100:"
                        f"fontsize=60:fontcolor=white:borderw=3:bordercolor=black:"
                        f"box=1:boxcolor=black@0.5:"
                        f"enable='between(t,{tiempo_inicio:.3f},{tiempo_fin:.3f})'"
                    )
                    drawtext_commands.append(drawtext_cmd.replace('\\', '/'))
                
                # Corregir la ruta para Windows reemplazando barras invertidas y escapando adecuadamente
                img_path_escaped = img_path.replace('\\', '/')
                audio_path_escaped = audio_path.replace('\\', '/')
                temp_video_escaped = temp_video.replace('\\', '/')
                
                # Concatenar los comandos drawtext con comas para aplicar filtros en secuencia
                drawtext_filters = ",".join(drawtext_commands)
                
                # Seleccionar una animación aleatoria para la imagen
                animations = [
                    # Zoom lento hacia adentro
                    f"zoompan=z='min(zoom+0.0015,1.3)':d={int(audio_duration*25)}:s={self.width}x{self.height}",
                    # Zoom lento hacia afuera
                    f"zoompan=z='if(lte(on,1),1.3,max(1.3-(on)*0.0015,1))':d={int(audio_duration*25)}:s={self.width}x{self.height}",
                    # Movimiento lento hacia arriba
                    f"zoompan=z=1.1:y='ih-ih*0.1-on*0.0007':d={int(audio_duration*25)}:s={self.width}x{self.height}",
                    # Movimiento lento hacia abajo
                    f"zoompan=z=1.1:y='on*0.0007':d={int(audio_duration*25)}:s={self.width}x{self.height}",
                    # Movimiento diagonal
                    f"zoompan=z=1.1:x='iw*0.05+on*0.0005':y='ih*0.05+on*0.0004':d={int(audio_duration*25)}:s={self.width}x{self.height}",
                ]
                animation = random.choice(animations)
                
                # Añadir fade in al inicio y fade out al final para facilitar transiciones
                if i == 0:
                    # Primer clip: solo fade in
                    fade_filter = "fade=in:0:30"
                elif i == len(image_paths) - 1:
                    # Último clip: solo fade out
                    fade_out_start = max(0, int(audio_duration*25) - 30)
                    fade_filter = f"fade=out:{fade_out_start}:30"
                else:
                    # Clips intermedios: fade in y fade out
                    fade_out_start = max(0, int(audio_duration*25) - 30)
                    fade_filter = f"fade=in:0:30,fade=out:{fade_out_start}:30"
                
                # Crear comando ffmpeg para combinar imagen, audio, animación y subtítulos
                cmd = (
                    f"ffmpeg -y -loop 1 -i \"{img_path_escaped}\" -i \"{audio_path_escaped}\" -c:v libx264 "
                    f"-t {audio_duration} -pix_fmt yuv420p -vf \"{animation},scale={self.width}:{self.height}:force_original_aspect_ratio=decrease,"
                    f"pad={self.width}:{self.height}:(ow-iw)/2:(oh-ih)/2,{fade_filter},"
                    f"{drawtext_filters}\" "
                    f"\"{temp_video_escaped}\""
                )
                
                # Ejecutar comando
                print(f"Procesando clip {i+1}/{len(image_paths)}...")
                print(f"Comando ffmpeg: {cmd[:150]}...")  # Para depuración (mostrar solo el inicio)
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
        
        # Preparar directorios temporales para la concatenación con transiciones
        temp_dir = os.path.join(tempfile.gettempdir(), f"video_concat_{int(time.time())}")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Vamos a crear una concatenación más avanzada usando filtros complejos pero de forma segura
        # Primero, creamos un .txt para el método simple de concatenación
        concat_file = os.path.join(temp_dir, "concat_list.txt")
        with open(concat_file, "w") as f:
            for video in temp_videos:
                # Usar formato de rutas compatible con ffmpeg en Windows (barras normales)
                video_path = video.replace('\\', '/')
                f.write(f"file '{video_path}'\n")
        
        # Corregir ruta para Windows
        concat_file_escaped = concat_file.replace('\\', '//')
        output_path_escaped = output_path.replace('\\', '//')
        
        # Primero generamos una versión sin transiciones para tener una base funcional
        temp_output = os.path.join(temp_dir, "temp_output.mp4")
        temp_output_escaped = temp_output.replace('\\', '//')
        
        # Comando para concatenación simple (sin transiciones sofisticadas pero seguro)
        concat_simple_cmd = f"ffmpeg -y -f concat -safe 0 -i \"{concat_file_escaped}\" -c copy \"{temp_output_escaped}\""
        print("Generando versión base del video...")
        os.system(concat_simple_cmd)
        
        # Si la concatenación simple funciona, proceder con la versión final
        if os.path.exists(temp_output) and os.path.getsize(temp_output) > 0:
            # Crear una versión con transiciones suaves usando crossfade
            final_cmd = f"ffmpeg -y -i \"{temp_output_escaped}\" -vf \"tblend=all_mode=average,framestep=1\" -c:a copy \"{output_path_escaped}\""
            print("Aplicando transiciones suaves...")
            os.system(final_cmd)
        else:
            # Si la versión con transiciones falla, usar la versión simple
            shutil.copy(temp_output, output_path)
        
        # Limpiar archivos temporales
        try:
            # Eliminar el archivo de lista
            if os.path.exists(concat_file):
                os.remove(concat_file)
            
            # Eliminar videos temporales
            for video in temp_videos:
                if os.path.exists(video):
                    os.remove(video)
            
            # Eliminar archivos temporales de concatenación
            if os.path.exists(temp_output):
                os.remove(temp_output)
            
            # Eliminar directorio temporal
            if os.path.exists(temp_dir) and temp_dir.startswith(tempfile.gettempdir()):
                shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Error al limpiar archivos temporales: {str(e)}")
        
        return output_path
    
    def _formatear_tiempo(self, segundos):
        """
        Formatea un tiempo en segundos al formato HH:MM:SS,mmm para subtítulos SRT
        
        Args:
            segundos: Tiempo en segundos
            
        Returns:
            str: Tiempo formateado
        """
        horas = int(segundos // 3600)
        minutos = int((segundos % 3600) // 60)
        segundos_restantes = segundos % 60
        milisegundos = int((segundos_restantes - int(segundos_restantes)) * 1000)
        return f"{horas:02d}:{minutos:02d}:{int(segundos_restantes):02d},{milisegundos:03d}"
    
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