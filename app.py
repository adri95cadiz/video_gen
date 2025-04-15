#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import streamlit as st
import json
from src.ai_agent import AIVideoAgent
import tempfile
from pathlib import Path
import time
import shutil
import random
from streamlit_sortables import sort_items

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

# Cargar estilos desde el archivo JSON
def load_styles():
    try:
        with open('src/styles.json', 'r', encoding='utf-8') as f:
            return json.load(f)['styles']
    except Exception as e:
        print(f"Error al cargar estilos: {e}")
        return []

# Función para crear directorios temporales
def create_temp_dir():
    """Crea un directorio temporal y devuelve su ruta"""
    temp_dir = tempfile.mkdtemp()
    return temp_dir

# Sidebar con opciones
st.sidebar.title("⚙️ Configuración")

# Cargar estilos
styles = load_styles()
style_options = {style["name"]: style["id"] for style in styles}
style_descriptions = {style["id"]: style["description"] for style in styles}

# Prompt de texto
prompt = st.sidebar.text_area(
    "Prompt para el video", 
    value="Un video corto sobre la importancia de aprender programación",
    height=100,
    help="Describe el contenido que deseas para el video"
)

# Selector de estilo de video
selected_style_name = st.sidebar.selectbox(
    "Estilo de video",
    options=list(style_options.keys()),
    index=0,
    help="Selecciona el estilo que deseas para tu video"
)
selected_style_id = style_options[selected_style_name]

# Mostrar descripción del estilo seleccionado
st.sidebar.info(style_descriptions[selected_style_id])

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
        local_image = st.checkbox("Imágenes local", value=True, help="Usar modelo local para las imágenes")
    with col3:
        local_voice = st.checkbox("Voz local", value=True, help="Usar modelo local para la voz")
    
    # Selector de modelo de OpenAI para guiones
    if not local_script:
        openai_model = st.selectbox(
            "Modelo de OpenAI para guiones",
            options=[
                "gpt-3.5-turbo", 
                "gpt-3.5-turbo-0125",
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-4-turbo-2024-04-09",
                "gpt-4.1",
                "gpt-4.5-preview"
            ],
            index=0,
            help="Modelo de OpenAI a utilizar para generar el guión. Los modelos más avanzados (GPT-4) generan textos más creativos pero son más costosos."
        )
        
        # Mostrar información sobre el modelo seleccionado
        model_info = {
            "gpt-3.5-turbo": "Modelo estándar con buen equilibrio entre calidad y costo.",
            "gpt-3.5-turbo-0125": "Versión actualizada de GPT-3.5 con mejor formato y estructura.",
            "gpt-4o": "Modelo multimodal de última generación para textos muy creativos y detallados.",
            "gpt-4o-mini": "Versión económica y rápida de GPT-4o.",
            "gpt-4-turbo-2024-04-09": "Modelo avanzado para guiones detallados y estructurados.",
            "gpt-4.1": "El modelo GPT más reciente de OpenAI (2025-04-14).",
            "gpt-4.5-preview": "Versión preliminar con capacidades avanzadas de texto e imagen."
        }
        st.info(model_info[openai_model])
    else:
        openai_model = None
    
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

    num_scenes = st.number_input(
        "Número de escenas",
        min_value=1,
        max_value=10,
        value=5,
        help="Número de escenas para el guión"
    )
    
    # Directorio de salida
    output_dir = st.text_input(
        "Directorio de salida", 
        value="videos",
        help="Directorio donde se guardarán los videos generados"
    )
    
    # Cargar archivos opcionales
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

    # Cargar archivos multimedia personalizados
    st.markdown("---**Medios personalizados**---")
    use_custom_media = st.checkbox("Usar mis propios archivos", value=False,
                                   help="Selecciona esta opción para usar tus propias imágenes o videos en lugar de los generados por IA.")
    
    uploaded_files = None
    if use_custom_media:
        uploaded_files_current = st.file_uploader(
            "Cargar imágenes o videos",
            type=["png", "jpg", "jpeg", "bmp", "gif", "webp", "mp4", "mov", "avi"],
            accept_multiple_files=True,
            help="Sube los archivos que quieres usar para las escenas. El orden de subida determinará el orden en el video."
        )
        
        if 'uploaded_media_state' not in st.session_state:
            st.session_state.uploaded_media_state = []
        
        if uploaded_files_current != st.session_state.get('last_uploaded_files', []):
            st.session_state.uploaded_media_state = uploaded_files_current
            st.session_state.last_uploaded_files = uploaded_files_current
            st.session_state.sort_key_suffix = 0 # Reiniciar key si cambian los archivos
        
        # Inicializar el sufijo de la key si no existe
        if 'sort_key_suffix' not in st.session_state:
            st.session_state.sort_key_suffix = 0
        
        if st.session_state.uploaded_media_state:
            st.write("**Ordenar archivos para escenas (arrastrar y soltar):**")
            # Extraer solo los nombres para mostrar en la lista sortable
            items_to_sort = [f.name for f in st.session_state.uploaded_media_state]
            
            # Usar una key dinámica que cambia al mezclar
            current_key = f"media_sorter_{st.session_state.sort_key_suffix}"
            sorted_item_names = sort_items(items_to_sort, key=current_key, direction="vertical")            
            
            # Reconstruir la lista de archivos subidos en el nuevo orden
            # Crear un mapeo de nombre a objeto archivo para reconstrucción rápida
            file_map = {f.name: f for f in st.session_state.uploaded_media_state}
            
            # Asegurarse de que todos los nombres sorteados existen en el mapeo
            reordered_files = []
            for name in sorted_item_names:
                if name in file_map:
                    reordered_files.append(file_map[name])
                else:
                    # Manejar caso de inconsistencia (poco probable pero seguro)
                    st.warning(f"El archivo '{name}' no se encontró después de reordenar. Puede que haya sido eliminado.")
                    # Intentar encontrar por si acaso cambió el estado
                    current_file_map = {f.name: f for f in uploaded_files_current}
                    if name in current_file_map:
                        reordered_files.append(current_file_map[name])
            
            # Actualizar el estado de la sesión con la lista reordenada
            st.session_state.uploaded_media_state = reordered_files
            
            # Asignar la lista reordenada para pasarla a la función
            uploaded_files = st.session_state.uploaded_media_state
        else:
            uploaded_files = []

# Función para generar el video
def generate_video(progress_placeholder, custom_media_files=None):
    temp_files = []
    custom_media_paths = []
    
    try:
        # Guardar archivos subidos temporalmente
        if custom_media_files:
            if (len(custom_media_files) != num_scenes):
                progress_placeholder.error("El número de escenas y medios personalizados debe coincidir")
                return
            temp_media_dir = tempfile.mkdtemp(prefix="custom_media_")
            for uploaded_file in custom_media_files:
                temp_path = os.path.join(temp_media_dir, uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                custom_media_paths.append(temp_path)
            
            temp_files.append(temp_media_dir) # Para limpiar el directorio después
            print(f"Archivos personalizados guardados en: {temp_media_dir}")
            print(f"Rutas de archivos personalizados: {custom_media_paths}")
            
        # Validar que el prompt no esté vacío (a menos que se usen medios personalizados)
        if not prompt or len(prompt.strip()) < 3:
            progress_placeholder.error("Por favor ingresa un prompt válido con al menos 3 caracteres")
            return
        
        # Configurar directorios
        os.makedirs(output_dir, exist_ok=True)
        
        # Verificar y guardar archivos temporalmente si se subieron            
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
        
        # Obtener el estilo seleccionado
        selected_style = next((style for style in styles if style["id"] == selected_style_id), None)
        
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
        
        add_log(f"Inicializando agente de IA con estilo: {selected_style_name}...")
        
        # Inicializar el agente
        agent = AIVideoAgent(
            local_script=local_script,
            local_image=local_image,
            local_voice=local_voice,
            output_dir=output_dir,
            custom_media=custom_media_paths,
            image_model=image_model if local_image else None,
            script_model=openai_model if not local_script else None,
            music_reference=background_music_path,
            style=selected_style
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
        if custom_media_paths:
            add_log(f"Generando video usando {len(custom_media_paths)} archivos multimedia personalizados...")
        else:
            add_log(f"Generando video con prompt: '{prompt[:50]}...' y estilo: {selected_style_name}")
        
        video_path = agent.generate_video(
            video_topic=prompt,
            max_words=max_words,
            progress_callback=progress_callback,
            num_scenes=num_scenes
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
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)
            elif os.path.exists(file_path):
                try:
                    os.unlink(file_path)
                except PermissionError:
                    print(f"No se pudo eliminar el archivo temporal {file_path} (puede estar en uso)")

# Botón para generar el video
if st.sidebar.button("Generar Video", type="primary", use_container_width=True):
    progress_placeholder = st.empty()
    custom_media_files = uploaded_files if use_custom_media else None
    try:
        # Asegurar que no haya conflictos con eventos asyncio
        generate_video(progress_placeholder, custom_media_files)
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
    2. **Creación de imágenes**: Genera imágenes que ilustran cada escena del guión (usando Stable Diffusion)
    3. **Síntesis de voz**: Convierte el texto a voz para narrar el video
    4. **Edición final**: Combina todo en un video con transiciones y música
    
    ### Opciones avanzadas
    
    - **Estilos de video**: Elige entre diferentes estilos predefinidos para el tono y aspecto del video
    - **Modelos locales**: Puedes usar modelos locales para evitar costes de API
    - **Referencia de voz**: Proporciona un archivo de audio para clonar esa voz
    - **Música de fondo**: Añade tu propia música como fondo del video
    
    ### Modelos utilizados
    
    - **Imágenes**: Stable Diffusion XL
    - **Texto**: OpenAI GPT-4.1, GPT-4o, GPT-3.5 (API) o Falcon-7B-Instruct (local)
    - **Voz**: ElevenLabs (API) o modelos locales de TTS
    
    ### Requisitos
    
    La herramienta requiere una GPU para usar modelos locales de manera eficiente.
    """) 