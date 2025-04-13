import os
import tempfile
from elevenlabs import save
from elevenlabs.client import ElevenLabs

class VoiceGenerator:
    def __init__(self, api_key=None, voice_id=None):
        self.api_key = api_key or os.environ.get("ELEVENLABS_API_KEY")
        self.voice_id = voice_id or os.environ.get("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # Rachel (voice de ElevenLabs)
        
        if not self.api_key:
            raise ValueError("Se requiere una API key de ElevenLabs")
        
        self.client = ElevenLabs(api_key=self.api_key)
            
    def generate_voice(self, text, output_path=None):
        """
        Genera audio a partir de texto usando ElevenLabs
        
        Args:
            text: El texto a convertir en voz
            output_path: Ruta donde guardar el archivo de audio (opcional)
            
        Returns:
            str: Ruta al archivo de audio generado
        """
        try:
            # Generar audio
            audio = self.client.text_to_speech.convert(
                text=text,
                voice_id=self.voice_id,
                model="eleven_multilingual_v2",  # Modelo más económico
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
            
        except Exception as e:
            raise Exception(f"Error al generar la voz: {str(e)}")
    
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