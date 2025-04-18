import os
import tempfile
from pydub import AudioSegment
import time
import shutil
import random
import subprocess # Importar subprocess para mejor control
import mimetypes # Para detectar tipo de archivo


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

    def get_media_duration(self, media_path):
        """Obtiene la duración de un archivo de imagen (predeterminado) o video."""
        mime_type, _ = mimetypes.guess_type(media_path)
        if mime_type and mime_type.startswith('video'):
            try:
                # Usar ffprobe para obtener la duración del video
                cmd = f'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "{media_path}'
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
                return float(result.stdout.strip())
            except Exception as e:
                print(f"Error al obtener duración del video {media_path}: {e}")
                return 5.0 # Duración por defecto si falla ffprobe
        else:
            # Para imágenes, devolver la duración del audio correspondiente (o un valor por defecto)
            return None # Indicador para usar la duración del audio

    def create_video(self, media_paths, audio_paths, sentences, output_path, bg_color="black", background_music_path=None):
        """
        Crea un video combinando imágenes y/o videos con audio.

        Args:
            media_paths: Lista de rutas a las imágenes o videos
            audio_paths: Lista de rutas a los archivos de audio
            sentences: Lista de oraciones para los subtítulos
            output_path: Ruta de salida para el video
            bg_color: Color de fondo (no usado en versión simple)
            background_music_path: Ruta a un archivo de música de fondo (opcional)

        Returns:
            str: Ruta al video generado
        """
        print("Usando generador de video simplificado para modelos locales")

        if len(media_paths) != len(audio_paths):
            raise ValueError(
                "Las listas de medios y audios deben tener la misma longitud")

        # Verificar si se proporcionó música de fondo
        has_background_music = background_music_path and os.path.exists(
            background_music_path)

        # Crear archivos de video temporales a partir de cada par imagen-audio
        temp_videos = []
        total_duration = 0

        for i, (media_path, audio_path) in enumerate(zip(media_paths, audio_paths)):
            try:
                # Crear un archivo de video temporal
                temp_video = os.path.join(
                    tempfile.gettempdir(), f"temp_video_{i}.mp4")

                # Determinar la duración del clip
                media_duration = self.get_media_duration(media_path)
                audio_duration = self.get_audio_duration(audio_path)
                
                # Si es una imagen, la duración la marca el audio
                # Si es un video, la duración es la del propio video (ignorar audio asociado?)
                clip_duration = media_duration if media_duration is not None else audio_duration
                
                # Si es video, podríamos decidir usar su audio original o el generado
                # Por simplicidad, usaremos el audio generado para ambos casos
                use_audio_path = audio_path
                
                total_duration += clip_duration

                # Escapar texto para ffmpeg (reemplazar comillas, etc.)
                texto_subtitulo = sentences[i].replace(
                    "'", "\\'").replace('"', '\\"')

                # Dividir el texto en fragmentos cortos basados en número de caracteres
                palabras = texto_subtitulo.split()
                fragmentos = []
                fragmento_actual = []
                caracteres_actual = 0

                # Máximo de caracteres por fragmento (aproximadamente 20-25 caracteres)
                max_caracteres_por_fragmento = 25

                for palabra in palabras:
                    # Calcular caracteres con la nueva palabra (incluyendo espacios)
                    nuevos_caracteres = caracteres_actual + \
                        len(palabra) + (1 if fragmento_actual else 0)

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

                # Evitar división por cero si no hay fragmentos
                if not fragmentos:
                    fragmentos = [texto_subtitulo] # Usar el texto completo si no se pudo dividir

                # Variables para controlar el tiempo
                duracion_por_fragmento = clip_duration / len(fragmentos)

                # Crear múltiples comandos de drawtext para cada fragmento con diferentes posiciones y tiempos
                drawtext_commands = []

                for j, fragmento in enumerate(fragmentos):
                    # Calcular el tiempo para cada fragmento en segundos
                    tiempo_inicio = j * duracion_por_fragmento
                    tiempo_fin = (j + 1) * duracion_por_fragmento

                    # Escapar el texto para FFmpeg
                    fragmento_escapado = fragmento.replace(
                        "'", "\\'").replace('"', '\\"')

                    # Crear el comando drawtext para este fragmento con tiempo de inicio y fin
                    drawtext_cmd = (
                        f"drawtext=text='{fragmento_escapado}':x=(w-text_w)/2:y=h-th-100:"
                        f"fontsize=60:fontcolor=white:borderw=3:bordercolor=black:"
                        f"box=1:boxcolor=black@0.5:"
                        f"enable='between(t,{tiempo_inicio:.3f},{tiempo_fin:.3f})'"
                    )
                    drawtext_commands.append(drawtext_cmd)

                # Corregir la ruta para Windows reemplazando barras invertidas y escapando adecuadamente
                media_path_escaped = media_path.replace('\\', '/')
                use_audio_path_escaped = use_audio_path.replace('\\', '/')
                temp_video_escaped = temp_video.replace('\\', '/')

                # Concatenar los comandos drawtext con comas para aplicar filtros en secuencia
                drawtext_filters = ",".join(drawtext_commands)

                # Seleccionar una animación aleatoria para la imagen
                animations = [
                    # Zoom lento hacia adentro
                    f"zoompan=z='min(zoom+0.0015,1.3)':d={int(clip_duration*25)}:s={self.width}x{self.height}",
                    # Zoom lento hacia afuera
                    f"zoompan=z='if(lte(on,1),1.3,max(1.3-(on)*0.0015,1))':d={int(clip_duration*25)}:s={self.width}x{self.height}",
                    # Movimiento lento hacia arriba
                    f"zoompan=z=1.1:y='ih-ih*0.1-on*0.0007':d={int(clip_duration*25)}:s={self.width}x{self.height}",
                    # Movimiento lento hacia abajo
                    f"zoompan=z=1.1:y='on*0.0007':d={int(clip_duration*25)}:s={self.width}x{self.height}",
                    # Movimiento diagonal
                    f"zoompan=z=1.1:x='iw*0.05+on*0.0005':y='ih*0.05+on*0.0004':d={int(clip_duration*25)}:s={self.width}x{self.height}",
                ]
                animation = random.choice(animations)

                # Añadir fade in al inicio y fade out al final para facilitar transiciones
                if i == 0:
                    # Primer clip: solo fade in
                    fade_frames = min(30, int(clip_duration*25/2)) # Fade de max 30 frames o mitad de clip
                    fade_filter = f"fade=in:0:{fade_frames}"
                elif i == len(media_paths) - 1:
                    # Último clip: solo fade out
                    fade_frames = min(30, int(clip_duration*25/2))
                    fade_out_start = max(0, int(clip_duration*25) - fade_frames)
                    fade_filter = f"fade=out:{fade_out_start}:{fade_frames}"
                else:
                    # Clips intermedios: fade in y fade out
                    fade_frames = min(30, int(clip_duration*25/2))
                    fade_out_start = max(0, int(clip_duration*25) - fade_frames)
                    fade_filter = f"fade=in:0:{fade_frames},fade=out:{fade_out_start}:{fade_frames}"

                # Determinar si el input es video o imagen
                mime_type, _ = mimetypes.guess_type(media_path)
                is_video_input = mime_type and mime_type.startswith('video')

                # Construir comando FFmpeg
                cmd_parts = ["ffmpeg", "-y"]

                if is_video_input:
                    # Input es video
                    cmd_parts.extend(["-i", f'\"{media_path_escaped}\"'])
                    # Usar audio generado
                    cmd_parts.extend(["-i", f'\"{use_audio_path_escaped}\"'])
                    # Mapear video y audio nuevo, ignorar audio original del video
                    cmd_parts.extend(["-map", "0:v:0", "-map", "1:a:0"])
                    cmd_parts.extend(["-c:v", "libx264"])
                    # Ajustar la duración del video de entrada si es necesario (usar -t)
                    cmd_parts.extend(["-t", str(clip_duration)]) 
                else:
                    # Input es imagen
                    cmd_parts.extend(["-loop", "1", "-i", f'\"{media_path_escaped}\"'])
                    cmd_parts.extend(["-i", f'\"{use_audio_path_escaped}\"'])
                    cmd_parts.extend(["-map", "0:v:0", "-map", "1:a:0"])
                    cmd_parts.extend(["-c:v", "libx264"])
                    cmd_parts.extend(["-t", str(clip_duration)])

                # Filtros comunes (escalado, pad, animación si es imagen, fade, subtítulos)
                filter_complex = []
                # Aplicar animación solo a imágenes
                if not is_video_input:
                    filter_complex.append(animation)
                    
                filter_complex.extend([
                    f"scale={self.width}:{self.height}:force_original_aspect_ratio=decrease",
                    f"pad={self.width}:{self.height}:(ow-iw)/2:(oh-ih)/2",
                    fade_filter,
                    drawtext_filters # Añadir subtítulos al final
                ])
                
                # Unir filtros con comas
                vf_filter = ",".join(filter(None, filter_complex))
                cmd_parts.extend(["-vf", f'\"{vf_filter}\"'])
                
                # Parámetros de codificación y salida
                cmd_parts.extend(["-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", f'\"{temp_video_escaped}\"'])

                # Unir comando
                cmd = " ".join(cmd_parts)

                # Ejecutar comando
                print(f"Procesando clip {i+1}/{len(media_paths)}...")
                # Para depuración (mostrar solo el inicio)
                print(f"Comando ffmpeg: {cmd[:150]}...")
                os.system(cmd)

                if os.path.exists(temp_video):
                    temp_videos.append(temp_video)

            except Exception as e:
                print(f"Error al procesar clip {i}: {str(e)}")

        # Si no se generaron videos temporales, salir
        if not temp_videos:
            raise Exception("No se pudieron generar videos temporales")

        # Si solo hay un video, usarlo directamente
        if len(temp_videos) == 1 and not has_background_music:
            shutil.copy(temp_videos[0], output_path)
            return output_path

        # Preparar directorios temporales para la concatenación con transiciones
        temp_dir = os.path.join(tempfile.gettempdir(),
                                f"video_concat_{int(time.time())}")
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

        # Primero generamos una versión sin transiciones para tener una base funcional
        temp_output = os.path.join(temp_dir, "temp_output.mp4")
        temp_output_escaped = temp_output.replace('\\', '//')

        # Validar que concat_file existe y no está vacío
        if not os.path.exists(concat_file) or os.path.getsize(concat_file) == 0:
            raise Exception(f"El archivo de concatenación {concat_file} está vacío o no existe.")
            
        # Comando para concatenación simple (sin transiciones sofisticadas pero seguro)
        concat_simple_cmd = f"ffmpeg -y -f concat -safe 0 -i \"{concat_file_escaped}\" -c copy \"{temp_output_escaped}\""
        print("Generando versión base del video...")
        os.system(concat_simple_cmd)

        # Si se proporcionó música de fondo, añadirla al video final
        if has_background_music:
            # Preparar el archivo de música de fondo
            bg_music_escaped = background_music_path.replace('\\', '//')
            video_with_music = os.path.join(temp_dir, "video_with_music.mp4")
            video_with_music_escaped = video_with_music.replace('\\', '//')

            # Extraer solo el audio del video para usarlo como "sidechain" para la música
            voice_track = os.path.join(temp_dir, "voice_track.wav")
            voice_track_escaped = voice_track.replace('\\', '//')
            extract_voice_cmd = f"ffmpeg -y -i \"{temp_output_escaped}\" -vn -acodec pcm_s16le \"{voice_track_escaped}\""
            print("Extrayendo pista de voz para procesamiento dinámico...")
            os.system(extract_voice_cmd)

            # Calcular las opciones de mezcla de audio con equilibrio dinámico
            # Usar sidechaincompress para reducir volumen de música cuando hay voz
            audio_mix_cmd = (
                f"ffmpeg -y -i \"{temp_output_escaped}\" -i \"{bg_music_escaped}\" -filter_complex "
                f"\"[1:a]aloop=loop=-1:size=2e+09,apad,volume=0.25[music];"
                f"[0:a]asplit=2[voiceorig][voicegate];"
                f"[voicegate]agate=threshold=0.04:release=50:attack=50[voicedetect];"
                f"[music][voicedetect]sidechaincompress=threshold=0.03:ratio=2.5:release=400:attack=50:makeup=1.1[musicduck];"
                f"[voiceorig][musicduck]amix=inputs=2:duration=first:weights=1.2 0.8[a]\" "
                f"-map 0:v -map \"[a]\" -c:v copy -c:a aac -b:a 192k -shortest "
                f"\"{video_with_music_escaped}\""
            )

            print("Añadiendo música de fondo con equilibrio dinámico suave...")
            # Mostrar solo los primeros 150 caracteres
            print(f"Comando: {audio_mix_cmd[:150]}...")
            os.system(audio_mix_cmd)

            # Comprobar si se generó correctamente
            if os.path.exists(video_with_music) and os.path.getsize(video_with_music) > 0:
                temp_output = video_with_music
                temp_output_escaped = video_with_music_escaped
                print("Música aplicada correctamente con equilibrio dinámico suave")
            else:
                print(
                    "Error al aplicar música con equilibrio dinámico, intentando con método simple...")

                # Método alternativo más simple si el complejo falla
                simple_mix_cmd = (
                    f"ffmpeg -y -i \"{temp_output_escaped}\" -i \"{bg_music_escaped}\" -filter_complex "
                    f"\"[1:a]aloop=loop=-1:size=2e+09,apad,volume=0.15[bg];"
                    f"[0:a][bg]amix=inputs=2:duration=first:weights=1.2 0.8[a]\" "
                    f"-map 0:v -map \"[a]\" -c:v copy -c:a aac -b:a 192k -shortest "
                    f"\"{video_with_music_escaped}\""
                )

                print("Usando método simple de mezcla...")
                os.system(simple_mix_cmd)

                if os.path.exists(video_with_music) and os.path.getsize(video_with_music) > 0:
                    temp_output = video_with_music
                    temp_output_escaped = video_with_music_escaped
                else:
                    print(
                        "Error al añadir música de fondo, se usará el video sin música")

        # Si la concatenación simple funciona, proceder con la versión final
        if os.path.exists(temp_output) and os.path.getsize(temp_output) > 0: 
            shutil.copy(temp_output, output_path)
            print("Video final copiado (sin transiciones complejas)")
        else:
            raise Exception("Falló la concatenación simple del video.")

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

            if has_background_music and os.path.exists(video_with_music):
                os.remove(video_with_music)

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
        milisegundos = int(
            (segundos_restantes - int(segundos_restantes)) * 1000)
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
                print(
                    f"Error al eliminar archivo temporal {file_path}: {str(e)}")
