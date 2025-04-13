import os
import tempfile
from elevenlabs import save
from elevenlabs.client import ElevenLabs
import wave
import struct
import math
import numpy as np
from pydub import AudioSegment
import io

class VoiceGenerator:
    def __init__(self, api_key=None, voice_id=None, use_local_model=False):
        """
        Inicializa el generador de voz
        
        Args:
            api_key: API key de ElevenLabs (opcional)
            voice_id: ID de la voz a usar (opcional)
            use_local_model: Si se debe usar generación de voz local
        """
        self.use_local_model = use_local_model
        
        if not use_local_model:
            # Intentar usar ElevenLabs si hay API key
            self.api_key = api_key or os.environ.get("ELEVENLABS_API_KEY")
            self.voice_id = voice_id or os.environ.get("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # Rachel (voice de ElevenLabs)
            
            if not self.api_key:
                print("No se encontró API key de ElevenLabs, cambiando a generación local")
                self.use_local_model = True
            else:
                try:
                    self.client = ElevenLabs(api_key=self.api_key)
                except Exception as e:
                    print(f"Error al inicializar ElevenLabs: {e}")
                    print("Cambiando a generación local")
                    self.use_local_model = True
        
        if self.use_local_model:
            print("Usando generación de voz sintética local")
            
    def generate_voice(self, text, output_path=None):
        """
        Genera audio a partir de texto
        
        Args:
            text: El texto a convertir en voz
            output_path: Ruta donde guardar el archivo de audio (opcional)
            
        Returns:
            str: Ruta al archivo de audio generado
        """
        try:
            if self.use_local_model:
                return self._generate_voice_local(text, output_path)
            else:
                return self._generate_voice_elevenlabs(text, output_path)
        except Exception as e:
            # Si falla con ElevenLabs, intentar con la versión local
            if not self.use_local_model:
                print(f"Error al generar voz con ElevenLabs: {e}")
                print("Intentando con generación local")
                self.use_local_model = True
                return self._generate_voice_local(text, output_path)
            else:
                raise Exception(f"Error al generar la voz: {str(e)}")
    
    def _generate_voice_elevenlabs(self, text, output_path):
        """Genera audio usando ElevenLabs"""
        # Generar audio
        audio = self.client.text_to_speech.convert(
            text=text,
            voice_id=self.voice_id,
            model_id="eleven_multilingual_v2",  # Modelo más económico
            output_format="mp3_44100_128",
        )
        
        # Guardar el audio
        if not output_path:
            # Crear un archivo temporal si no se especifica una ruta
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            output_path = temp_file.name
            temp_file.close()
        
        save(audio, output_path)
        return output_path
    
    def _generate_voice_local(self, text, output_path):
        """
        Genera audio con un modelo TTS local
        """
        try:
            # Importar las librerías para TTS
            from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
            import torch
            import soundfile as sf
            import numpy as np
            
            print(f"Generando audio para: {text[:50]}...")
            
            # Verificar si hay GPU disponible
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Modelo TTS usando device: {device}")
            
            # Cargar el procesador y el modelo con la configuración adecuada para el dispositivo
            processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
            
            # IMPORTANTE: Usar float32 para evitar problemas con soundfile
            # soundfile no admite float16, así que siempre usamos float32
            dtype = torch.float32
            
            # Cargar modelo con la configuración adecuada
            model = SpeechT5ForTextToSpeech.from_pretrained(
                "microsoft/speecht5_tts",
                torch_dtype=dtype
            ).to(device)
            
            vocoder = SpeechT5HifiGan.from_pretrained(
                "microsoft/speecht5_hifigan",
                torch_dtype=dtype
            ).to(device)
            
            # Preprocess texto
            inputs = processor(text=text, return_tensors="pt").to(device)
            
            # Cargar embeddings de speaker (voz)
            speaker_embeddings = torch.randn(1, 512, dtype=dtype).to(device)  # Embedding aleatorio
            
            # Generar output de speech
            with torch.inference_mode():  # Usar inference_mode en lugar de no_grad (más eficiente)
                speech = model.generate_speech(
                    inputs["input_ids"], 
                    speaker_embeddings, 
                    vocoder=vocoder
                )
            
            # Convertir a numpy para guardar - asegurar que sea float32
            speech_np = speech.cpu().numpy().astype(np.float32)
            
            # Crear un archivo temporal si no se especifica una ruta
            if not output_path:
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
                output_path = temp_file.name
                temp_file.close()
            
            # Guardar en un archivo temporal wav primero
            temp_wav = output_path.replace('.mp3', '.wav')
            sf.write(temp_wav, speech_np, samplerate=16000)
            
            # Convertir de WAV a MP3 usando pydub
            audio = AudioSegment.from_wav(temp_wav)
            audio.export(output_path, format="mp3")
            
            # Eliminar el archivo temporal wav
            if os.path.exists(temp_wav):
                os.remove(temp_wav)
                
            return output_path
            
        except Exception as e:
            print(f"Error con modelo TTS local: {str(e)}")
            print("Usando generador de audio simple como fallback")
            return self._generate_voice_simplified(text, output_path)
            
    def _generate_voice_simplified(self, text, output_path):
        """
        Genera un audio sintético simple con un tono básico.
        Esta es una alternativa muy básica cuando no tenemos acceso a APIs de TTS.
        """
        # Crear un archivo temporal si no se especifica una ruta
        if not output_path:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            output_path = temp_file.name
            temp_file.close()
            
        # Configurar parámetros de audio
        duration = len(text) * 0.085  # Aproximadamente 85ms por carácter
        sample_rate = 44100
        frequency = 220.0  # Frecuencia base (A3)
        
        # Generar datos de audio
        samples = int(duration * sample_rate)
        audio_data = []
        
        # Generar una onda sinusoidal básica que varía en función del texto
        for i in range(samples):
            t = i / sample_rate
            # Variar la frecuencia basado en el texto (muy básico)
            char_position = min(int(t / duration * len(text)), len(text) - 1)
            freq_variation = (ord(text[char_position]) % 20) - 10
            
            # Generar onda sinusoidal
            value = math.sin(2 * math.pi * (frequency + freq_variation) * t)
            
            # Añadir envolvente para suavizar inicio y fin
            fade_time = 0.1
            if t < fade_time:
                value *= t / fade_time
            elif t > duration - fade_time:
                value *= (duration - t) / fade_time
                
            # Añadir variaciones de amplitud basadas en el texto
            volume = 0.8 + 0.2 * math.sin(2 * math.pi * 0.5 * t)
            value *= volume
            
            # Convertir a entero de 16 bits
            audio_data.append(int(value * 32767))
        
        # Crear un BytesIO para guardar el audio WAV
        with io.BytesIO() as wav_io:
            # Crear archivo WAV
            with wave.open(wav_io, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 2 bytes (16 bits)
                wav_file.setframerate(sample_rate)
                # Empaquetar los datos como valores de 16 bits
                wav_data = struct.pack('<%dh' % len(audio_data), *audio_data)
                wav_file.writeframes(wav_data)
            
            # Convertir de WAV a MP3 usando pydub
            wav_io.seek(0)
            audio = AudioSegment.from_wav(wav_io)
            audio.export(output_path, format="mp3")
        
        return output_path
    
    def generate_voice_for_sentences(self, sentences, output_dir="temp_audio"):
        """
        Genera archivos de audio para cada oración del guión
        
        Args:
            sentences: Lista de oraciones
            output_dir: Directorio donde guardar los archivos de audio
            
        Returns:
            list: Lista de rutas a los archivos de audio generados
        """
        os.makedirs(output_dir, exist_ok=True)
        audio_files = []
        
        for i, sentence in enumerate(sentences):
            output_path = os.path.join(output_dir, f"sentence_{i}.mp3")
            try:
                self.generate_voice(sentence, output_path)
                audio_files.append(output_path)
            except Exception as e:
                print(f"Error al generar audio para la oración {i}: {str(e)}")
        
        return audio_files 