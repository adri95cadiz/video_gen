# VideoGen AI

VideoGen AI es un agente de inteligencia artificial que genera vídeos cortos para YouTube a partir de un prompt de texto.

## Funcionalidades

- Generación de guión utilizando la API de OpenAI (GPT-3.5-turbo)
- Narración de voz con ElevenLabs
- Generación de imágenes con Stability AI (API REST)
- Creación automática de vídeos con subtítulos y transiciones

## Requisitos

- Python 3.8+
- API keys de OpenAI, ElevenLabs y Stability AI
- FFmpeg (necesario para procesar audio y video)

## Instalación

1. Clona este repositorio
2. Instala las dependencias: `pip install -r requirements.txt`
3. Configura tus API keys en un archivo `.env`
4. Instala FFmpeg:
   - **Windows**: Descarga desde [ffmpeg.org](https://ffmpeg.org/download.html) y añade a PATH
   - **macOS**: `brew install ffmpeg`
   - **Linux**: `sudo apt-get install ffmpeg`
5. Ejecuta el script de verificación: `python test_environment.py`

## Uso

```bash
python main.py --prompt "Tu prompt de texto aquí"
```

## Opciones adicionales

```bash
python main.py --help
```

Muestra todas las opciones disponibles:

- `--prompt`: El prompt de texto para generar el video
- `--output`: Nombre del archivo de salida (por defecto genera un nombre con timestamp)
- `--max-words`: Número máximo de palabras para el guión (por defecto 200)
- `--output-dir`: Directorio donde guardar el video (por defecto 'videos') 