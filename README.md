# VideoGen AI

VideoGen AI es un agente de inteligencia artificial que genera vídeos cortos para YouTube a partir de un prompt de texto.

## Funcionalidades

- Generación de guión utilizando la API de OpenAI (GPT-3.5-turbo) o modelos locales
- Narración de voz con ElevenLabs
- Generación de imágenes con Stability AI (API REST) o modelos locales
- Creación automática de vídeos con subtítulos y transiciones

## Requisitos

- Python 3.8+
- API keys de OpenAI, ElevenLabs y Stability AI (opcional si se usan modelos locales)
- FFmpeg (necesario para procesar audio y video)
- GPU (opcional pero recomendada para modelos locales)

## Configuración

1. Clona este repositorio
2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

3. Crea un archivo `.env` basado en el archivo `.env-example` y configura tus API keys:
   ```
   OPENAI_API_KEY=tu_api_key_de_openai
   STABILITY_API_KEY=tu_api_key_de_stability_ai
   ELEVENLABS_API_KEY=tu_api_key_de_elevenlabs
   ```

4. Para usar ElevenLabs para voces de alta calidad:
   - Regístrate en [ElevenLabs](https://elevenlabs.io/) y obtén una API key
   - Establece la API key en tu archivo `.env`
   - Opcionalmente, personaliza la voz configurando `ELEVENLABS_VOICE_ID`

5. Instala FFmpeg si aún no lo tienes:
   - **Windows**: Descarga desde [ffmpeg.org](https://ffmpeg.org/download.html) y añade a PATH
   - **macOS**: `brew install ffmpeg`
   - **Linux**: `sudo apt-get install ffmpeg`

## Uso

```bash
# Usar APIs externas (OpenAI, Stability AI, ElevenLabs)
python main.py --prompt "Tu prompt de texto aquí"

# Usar modelos locales (sin costes de API)
python main.py --prompt "Tu prompt de texto aquí" --local
```

Para una generación rápida de prueba:
```bash
# Generar un video corto de ejemplo
python main.py --prompt "Inteligencia artificial en 30 segundos" --local
```

## Modelos locales

VideoGen AI puede utilizar modelos de IA locales para evitar los costes de API:

- **Generación de texto**: Utiliza modelos de Hugging Face como [Falcon-7B](https://huggingface.co/tiiuae/falcon-7b-instruct)
- **Generación de imágenes**: Utiliza Stable Diffusion localmente

Para configurar modelos personalizados, puedes definir estas variables en tu archivo `.env`:

```
# Rutas a modelos específicos (opcional)
LOCAL_TEXT_MODEL_PATH=tiiuae/falcon-7b-instruct  # o ruta a modelo descargado
LOCAL_IMAGE_MODEL_PATH=stabilityai/stable-diffusion-2-1  # o ruta a modelo descargado
```

### Requisitos de hardware

- Los modelos locales funcionan mejor con una GPU NVIDIA con al menos 8GB de VRAM
- Se recomienda 16GB de RAM del sistema
- Si no dispones de GPU, los modelos se ejecutarán en CPU pero serán significativamente más lentos

## Opciones adicionales

```bash
python main.py --help
```

Muestra todas las opciones disponibles:

- `--prompt`: El prompt de texto para generar el video
- `--output`: Nombre del archivo de salida (por defecto genera un nombre con timestamp)
- `--max-words`: Número máximo de palabras para el guión (por defecto 200)
- `--output-dir`: Directorio donde guardar el video (por defecto 'videos')
- `--local`: Usar modelos locales en lugar de APIs externas (evita costes de API) 