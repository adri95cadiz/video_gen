# Interfaz Gráfica para el Generador de Videos con IA

Esta interfaz gráfica proporciona una forma amigable de interactuar con el agente de generación de videos con IA.

## Características

- **Interfaz intuitiva**: Crea videos simplemente introduciendo un prompt de texto
- **Opciones configurables**: Personaliza la generación de videos con múltiples opciones
- **Seguimiento en tiempo real**: Visualiza el progreso de la generación paso a paso
- **Vista previa integrada**: Reproduce el video directamente en la interfaz
- **Descarga simplificada**: Descarga el video generado con un solo clic

## Instalación

1. Asegúrate de tener todas las dependencias instaladas:

```bash
pip install -r requirements.txt
```

2. Para sistemas Windows, ejecuta el instalador de FFMPEG (necesario para la generación de videos):

```bash
python install_ffmpeg.py
```

## Uso

1. Ejecuta la interfaz gráfica:

```bash
streamlit run app.py
```

2. Se abrirá una ventana del navegador con la interfaz. Si no se abre automáticamente, navega a la URL mostrada en la terminal (normalmente http://localhost:8501).
3. En la interfaz:

   - Introduce el prompt para el video en el panel lateral
   - Configura las opciones adicionales según tus necesidades
   - Haz clic en "Generar Video"
   - Espera a que se complete el proceso
   - Visualiza y descarga el video resultante

## Opciones Avanzadas

### Modelos Locales vs. APIs

Puedes elegir entre:

- **APIs externas**: OpenAI para el guión, Stability AI para imágenes y ElevenLabs para voz (requiere claves API configuradas en .env)
- **Modelos locales**: Utiliza modelos descargados para funcionar sin conexión y sin costes (requiere GPU)

### Configuración de Salida

- **Palabras máximas**: Controla la longitud del guión generado
- **Directorio de salida**: Especifica dónde guardar los videos generados
- **Archivos de referencia**: Sube archivos de audio para clonar la voz o música de fondo

## Requisitos de Sistema

- Python 3.8 o superior
- GPU recomendada para modelos locales
- FFmpeg instalado (el script install_ffmpeg.py puede instalarlo automáticamente)
- Conexión a Internet para APIs externas

## Solución de Problemas

- **Error de clave API**: Asegúrate de configurar las claves en el archivo .env (ver .env.example)
- **Modelos locales lentos**: Verifica que estás usando una GPU y que PyTorch está configurado correctamente
- **FFmpeg no encontrado**: Ejecuta python install_ffmpeg.py o instálalo manualmente
- **Error "no running event loop"**: Este error en Windows se ha solucionado en la última versión. Si persiste, ejecuta la aplicación con: `python -m streamlit run app.py`

## Licencia

Este proyecto utiliza las mismas licencias que el proyecto principal de generador de videos con IA.
