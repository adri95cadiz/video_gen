import os
import logging
import tempfile
from pathlib import Path
import torch
import pyttsx3
import requests
from dotenv import load_dotenv
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import numpy as np
import soundfile as sf
from langdetect import detect

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VoiceGenerator:
    def __init__(self, use_local_model=False):
        """
        Inicializa el generador de voz.

        Args:
            use_local_model: Si se debe usar un modelo local en lugar de una API externa
        """
        # Cargar variables de entorno
        load_dotenv()
        
        # Configuración general
        self.use_local_model = use_local_model
        self.is_gpu_available = torch.cuda.is_available()
        logger.info(f"GPU disponible: {self.is_gpu_available}")
        
        # Configuración para ElevenLabs
        self.api_key = os.environ.get("ELEVENLABS_API_KEY")
        self.voice_id = os.environ.get("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # Rachel es la voz predeterminada
        
        # Comprobar si tenemos configuración para ElevenLabs
        if not use_local_model and not self.api_key:
            logger.warning("No se encontró API key para ElevenLabs, se usará el modelo local como fallback")
            self.use_local_model = True

    def generate_voice(self, text, output_file, lang="es"):
        """
        Genera un archivo de audio a partir de texto.

        Args:
            text: Texto a convertir en audio
            output_file: Ruta del archivo de salida
            lang: Código de idioma (es, en, etc.)

        Returns:
            Ruta al archivo de audio generado
        """
        try:
            if not self.use_local_model:
                # Intentar usar ElevenLabs para la generación externa
                return self._generate_voice_elevenlabs(text, output_file, lang)
            else:
                try:
                    # Primero intentar con pyttsx3 para español
                    if lang == "es":
                        return self._generate_voice_pyttsx3(text, output_file, lang)
                    # Para otros idiomas usar SpeechT5
                    else:
                        return self._generate_voice_speech_t5(text, output_file)
                except Exception as e:
                    logger.warning(f"Error con pyttsx3: {str(e)}. Intentando con SpeechT5")
                    return self._generate_voice_speech_t5(text, output_file)
        except Exception as e:
            logger.error(f"Error al generar voz: {str(e)}")
            # Si falla con ElevenLabs, intentar el modelo local
            if not self.use_local_model:
                logger.warning("Error con ElevenLabs API, utilizando modelo local como fallback")
                self.use_local_model = True
                return self.generate_voice(text, output_file, lang)
            # Generar silencio como último recurso
            return self._generate_silence(output_file, duration=2.0)

    def _generate_voice_elevenlabs(self, text, output_file, lang="es"):
        """
        Genera voz usando la API de ElevenLabs
        
        Args:
            text: Texto a convertir en audio
            output_file: Ruta del archivo de salida
            lang: Código de idioma

        Returns:
            Ruta al archivo de audio generado
        """
        logger.info(f"Generando audio con ElevenLabs API para texto: '{text[:50]}...'")
        
        # URL de la API de ElevenLabs
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}"
        
        # Cabeceras con la API key
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.api_key
        }
        
        # Datos a enviar
        data = {
            "text": text,
            "model_id": "eleven_multilingual_v2",  # Modelo multilingüe
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75
            }
        }
        
        try:
            # Realizar la petición a la API
            response = requests.post(url, json=data, headers=headers)
            
            # Verificar si la petición fue exitosa
            if response.status_code == 200:
                # Guardar el audio recibido
                with open(output_file, 'wb') as f:
                    f.write(response.content)
                logger.info(f"Audio generado correctamente con ElevenLabs en {output_file}")
                return output_file
            else:
                # Si hay un error, lanzar excepción
                error_message = f"Error al llamar a ElevenLabs API: {response.status_code} - {response.text}"
                logger.error(error_message)
                raise Exception(error_message)
                
        except Exception as e:
            logger.error(f"Error al conectar con ElevenLabs: {str(e)}")
            raise

    def _generate_voice_pyttsx3(self, text, output_file, lang="es"):
        """
        Genera voz usando pyttsx3 (motor de síntesis de voz del sistema)
        
        Args:
            text: Texto a convertir en audio
            output_file: Ruta del archivo de salida
            lang: Código de idioma

        Returns:
            Ruta al archivo de audio generado
        """
        logger.info(f"Generando audio con pyttsx3 para texto: '{text[:50]}...'")
        
        try:
            # Inicializar el motor
            engine = pyttsx3.init()
            
            # Buscar una voz en español
            spanish_voice = None
            voices = engine.getProperty('voices')
            
            # Buscar voces en español
            for voice in voices:
                if 'spanish' in voice.name.lower() or 'español' in voice.name.lower() or 'es' in voice.id.lower():
                    spanish_voice = voice.id
                    logger.info(f"Voz en español encontrada: {voice.name}")
                    break
            
            # Configurar voz en español si se encontró
            if spanish_voice:
                engine.setProperty('voice', spanish_voice)
            else:
                logger.warning("No se encontró voz en español, usando voz predeterminada")
            
            # Configurar velocidad y volumen
            engine.setProperty('rate', 150)  # Velocidad normal
            engine.setProperty('volume', 1.0)  # Volumen máximo
            
            # Guardar a archivo
            engine.save_to_file(text, output_file)
            engine.runAndWait()
            
            logger.info(f"Audio generado correctamente en {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error generando audio con pyttsx3: {str(e)}")
            raise

    def _generate_voice_speech_t5(self, text, output_file):
        """
        Genera voz usando el modelo SpeechT5
        
        Args:
            text: Texto a convertir en audio
            output_file: Ruta del archivo de salida
            
        Returns:
            Ruta al archivo de audio generado
        """
        logger.info(f"Generando audio con SpeechT5 para texto: '{text[:50]}...'")
        
        try:
            # Detectar idioma del texto para dar contexto al modelo
            try:
                detected_lang = detect(text)
                logger.info(f"Idioma detectado: {detected_lang}")
            except:
                detected_lang = "es"  # Predeterminado a español
            
            # Cargar modelos
            processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
            model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
            vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
            
            # Mover a GPU si está disponible
            if self.is_gpu_available:
                model = model.to("cuda")
                vocoder = vocoder.to("cuda")
            
            # Procesar texto de entrada
            inputs = processor(text=text, return_tensors="pt")
            
            # Mover a GPU si está disponible
            if self.is_gpu_available:
                inputs = {name: tensor.to("cuda") for name, tensor in inputs.items()}
            
            # Generar embeddings de hablante (speaker embedding)
            # Usamos un embedding aleatorio pero fijo para consistencia
            speaker_embeddings = torch.zeros((1, 512))
            
            # Generar espectrograma de mel
            speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
            
            # Convertir a numpy y guardar como archivo
            speech = speech.cpu().numpy()
            sf.write(output_file, speech, samplerate=16000)
            
            logger.info(f"Audio generado correctamente en {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error generando audio con SpeechT5: {str(e)}")
            raise

    def _generate_silence(self, output_file, duration=1.0):
        """
        Genera un archivo de audio de silencio como último recurso
        
        Args:
            output_file: Ruta del archivo de salida
            duration: Duración del silencio en segundos
            
        Returns:
            Ruta al archivo de audio generado
        """
        logger.warning(f"Generando silencio de {duration} segundos como último recurso")
        
        # Crear un array de silencio (todos ceros)
        sample_rate = 16000
        samples = int(duration * sample_rate)
        silence = np.zeros(samples, dtype=np.float32)
        
        # Guardar a archivo
        sf.write(output_file, silence, samplerate=sample_rate)
        
        logger.info(f"Silencio generado en {output_file}")
        return output_file 