#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import streamlit as st
from src.ai_agent import AIVideoAgent
import tempfile
import shutil
from pathlib import Path
import time

# Configuración para solucionar el error "no running event loop"
import asyncio
try:
    # Configurar el bucle de eventos para Windows
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
except Exception as e:
    print(f"Error al configurar la política de bucle de eventos: {e}")

# Configuración de la página
st.set_page_config(
    page_title="Generador de Videos IA",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos personalizados
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
    }
    .stProgress .st-bo {
        background-color: #4CAF50;
    }
    .success-text {
        color: #4CAF50;
        font-weight: bold;
    }
    .error-text {
        color: #F44336;
        font-weight: bold;
    }
    h1, h2, h3 {
        color: #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

# Título de la aplicación
st.title("🎬 Generador de Videos con IA")
st.write("Crea videos cortos para YouTube o redes sociales a partir de un prompt de texto.")

# Función para crear directorios temporales
def create_temp_dir():
    """Crea un directorio temporal y devuelve su ruta"""
    temp_dir = tempfile.mkdtemp()
    return temp_dir

# Sidebar con opciones
st.sidebar.title("⚙️ Configuración")

# Prompt de texto
prompt = st.sidebar.text_area(
    "Prompt para el video", 
    value="Un video corto sobre la importancia de aprender programación",
    height=100,
    help="Describe el contenido que deseas para el video"
)

# Opciones avanzadas
with st.sidebar.expander("Opciones avanzadas", expanded=False):
    # Número máximo de palabras
    max_words = st.number_input(
        "Palabras máximas", 
        min_value=50, 
        max_value=500, 
        value=200,
        help="Número máximo de palabras para el guión"
    )
    
    # Opciones para usar modelos locales
    col1, col2, col3 = st.columns(3)
    with col1:
        local_script = st.checkbox("Guión local", value=False, help="Usar modelo local para el guión")
    with col2:
        local_image = st.checkbox("Imágenes local", value=False, help="Usar modelo local para las imágenes")
    with col3:
        local_voice = st.checkbox("Voz local", value=False, help="Usar modelo local para la voz")
    
    # Selección de modelo de imagen
    if local_image:
        image_model = st.selectbox(
            "Modelo de imagen",
            options=[
                "stabilityai/stable-diffusion-xl-base-1.0",
                "runwayml/stable-diffusion-v1-5",
                "CompVis/stable-diffusion-v1-4"
            ],
            index=0,
            help="Modelo de Stable Diffusion a utilizar para generar imágenes"
        )
    else:
        image_model = None
    
    # Directorio de salida
    output_dir = st.text_input(
        "Directorio de salida", 
        value="videos",
        help="Directorio donde se guardarán los videos generados"
    )
    
    # Cargar archivos opcionales
    voice_reference_file = st.file_uploader(
        "Archivo de referencia de voz (opcional)", 
        type=["mp3", "wav"],
        help="Archivo de audio para clonar la voz (requiere modelo local de voz)"
    )
    
    background_music_file = st.file_uploader(
        "Música de fondo (opcional)", 
        type=["mp3", "wav"],
        help="Archivo de música para usar como fondo en el video"
    )
    
    image_dir = st.text_input(
        "Directorio de imágenes (opcional)",
        value="",
        help="Directorio con imágenes existentes para usar en lugar de generar nuevas"
    )

# Función para generar el video
def generate_video(progress_placeholder):
    temp_files = []
    
    try:
        # Validar que el prompt no esté vacío
        if not prompt or len(prompt.strip()) < 3:
            progress_placeholder.error("Por favor ingresa un prompt válido con al menos 3 caracteres")
            return
        
        # Configurar directorios
        os.makedirs(output_dir, exist_ok=True)
        
        # Verificar y guardar archivos temporalmente si se subieron
        voice_reference_path = None
        if voice_reference_file:
            temp_voice = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            temp_voice.write(voice_reference_file.read())
            voice_reference_path = temp_voice.name
            temp_files.append(temp_voice.name)
            temp_voice.close()
            
        background_music_path = None
        if background_music_file:
            temp_music = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            temp_music.write(background_music_file.read())
            background_music_path = temp_music.name
            temp_files.append(temp_music.name)
            temp_music.close()
        
        # Verificar directorio de imágenes
        valid_image_dir = None
        if image_dir and os.path.isdir(image_dir):
            valid_image_dir = image_dir
        
        # Mensajes de progreso
        progress_placeholder.text("Inicializando agente de IA...")
        progress_bar = st.progress(0)
        
        # Crear área para log
        log_container = st.container()
        with log_container:
            st.subheader("Progreso de la generación")
            log_area = st.empty()
        
        logs = []
        
        def add_log(message):
            logs.append(f"[{time.strftime('%H:%M:%S')}] {message}")
            log_area.code("\n".join(logs), language="bash")
        
        add_log("Iniciando proceso de generación...")
        
        # Monkey patch para capturar el progreso del agente
        original_print = print
        
        def print_interceptor(message, *args, **kwargs):
            original_print(message, *args, **kwargs)
            if isinstance(message, str):
                add_log(message)
            return message
        
        import builtins
        builtins.print = print_interceptor
        
        add_log("Inicializando agente de IA...")
        
        # Inicializar el agente
        agent = AIVideoAgent(
            local_script=local_script,
            local_image=local_image,
            local_voice=local_voice,
            output_dir=output_dir,
            image_dir=valid_image_dir,
            voice_reference=voice_reference_path,
            image_model=image_model if local_image else None
        )
        
        add_log("Generando guión a partir del prompt...")
        progress_bar.progress(10)
        
        # Clase para crear un callback que actualice el progreso
        class ProgressCallback:
            def __init__(self):
                self.current_step = "Inicializando"
                self.current_percentage = 0
                
            def update(self, step, percentage):
                self.current_step = step
                self.current_percentage = percentage
                add_log(f"{step} - {percentage}%")
                progress_placeholder.text(f"{step}...")
                progress_bar.progress(percentage)
            
            def on_script_start(self):
                self.update("Generando guión", 10)
                
            def on_script_complete(self, script):
                self.update("Guión completado", 30)
                add_log(f"Guión generado: {len(script.split())} palabras")
                
            def on_images_start(self):
                self.update("Generando imágenes", 40)
                
            def on_image_complete(self, scene_index, total_scenes):
                percentage = 30 + int((scene_index / total_scenes) * 30)
                self.update(f"Imágenes generadas: {scene_index}/{total_scenes}", percentage)
                
            def on_voice_start(self):
                self.update("Generando narración de voz", 70)
                
            def on_voice_complete(self):
                self.update("Narración generada", 80)
                
            def on_video_start(self):
                self.update("Creando video final", 90)
                
            def on_video_complete(self, video_path):
                self.update("¡Video completado!", 100)
        
        progress_callback = ProgressCallback()
        
        # Generar el video
        add_log(f"Generando video con prompt: '{prompt[:50]}...'")
        video_path = agent.generate_video(
            prompt=prompt,
            max_words=max_words,
            background_music_path=background_music_path,
            progress_callback=progress_callback
        )
        
        # Restaurar print original
        builtins.print = original_print
        
        progress_bar.progress(100)
        progress_placeholder.markdown(f'<p class="success-text">¡Video generado con éxito!</p>', unsafe_allow_html=True)
        
        add_log(f"Video generado satisfactoriamente: {video_path}")
        
        # Mostrar el video
        st.subheader("🎥 Video Generado")
        video_file = open(video_path, 'rb')
        video_bytes = video_file.read()
        
        st.video(video_bytes)
        
        # Opción para descargar
        st.download_button(
            label="Descargar video",
            data=video_bytes,
            file_name=os.path.basename(video_path),
            mime="video/mp4"
        )
        
    except Exception as e:
        import traceback
        error_message = str(e)
        error_traceback = traceback.format_exc()
        progress_placeholder.markdown(f'<p class="error-text">Error: {error_message}</p>', unsafe_allow_html=True)
        st.error(f"Error durante la generación del video")
        st.code(error_traceback, language="python")
        
        # Restaurar print original en caso de error
        try:
            builtins.print = original_print
        except:
            pass
    finally:
        # Limpiar archivos temporales
        for file_path in temp_files:
            if os.path.exists(file_path):
                os.unlink(file_path)

# Botón para generar el video
if st.sidebar.button("Generar Video", type="primary", use_container_width=True):
    progress_placeholder = st.empty()
    try:
        # Asegurar que no haya conflictos con eventos asyncio
        generate_video(progress_placeholder)
    except Exception as e:
        st.error(f"Error durante la ejecución: {str(e)}")
        st.code(str(e), language="bash")
        import traceback
        st.code(traceback.format_exc(), language="python")

# Sección informativa
with st.expander("ℹ️ Acerca de esta herramienta"):
    st.markdown("""
    ### ¿Cómo funciona?
    
    Este generador utiliza Inteligencia Artificial para crear videos en tres pasos:
    
    1. **Generación de guión**: Crea un guión narrativo basado en tu prompt
    2. **Creación de imágenes**: Genera imágenes que ilustran cada escena del guión (usando Stable Diffusion 3.5)
    3. **Síntesis de voz**: Convierte el texto a voz para narrar el video
    4. **Edición final**: Combina todo en un video con transiciones y música
    
    ### Opciones avanzadas
    
    - **Modelos locales**: Puedes usar modelos locales para evitar costes de API
    - **Referencia de voz**: Proporciona un archivo de audio para clonar esa voz
    - **Música de fondo**: Añade tu propia música como fondo del video
    
    ### Modelos utilizados
    
    - **Imágenes**: Stable Diffusion 3.5 Large (última versión)
    - **Texto**: OpenAI GPT-4 (API) o Falcon-7B-Instruct (local)
    - **Voz**: ElevenLabs (API) o modelos locales de TTS
    
    ### Requisitos
    
    La herramienta requiere una GPU para usar modelos locales de manera eficiente.
    """) 