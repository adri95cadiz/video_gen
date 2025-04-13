import os
import openai
import re

class ScriptGenerator:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Se requiere una API key de OpenAI")
        
        # Inicializar el cliente
        self.client = openai.OpenAI(api_key=self.api_key)
        
    def generate_script(self, prompt, max_words=200, model="gpt-3.5-turbo"):
        """
        Genera un guión para un vídeo corto basado en un prompt.
        
        Args:
            prompt: El prompt de entrada describiendo el contenido del vídeo
            max_words: Número máximo de palabras (por defecto 200 para vídeos cortos)
            model: Modelo de OpenAI a utilizar
            
        Returns:
            str: El guión generado
        """
        try:
            completion = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": f"Eres un experto en escribir guiones para vídeos cortos de YouTube. "
                                                 f"Crea un guión atractivo y conciso de máximo {max_words} palabras. "
                                                 f"El guión debe ser informativo, directo y mantener la atención del espectador. "
                                                 f"Evita introducciones largas y ve directo al tema principal."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            raise Exception(f"Error al generar el guión: {str(e)}")
            
    def split_script_into_sentences(self, script):
        """
        Divide el guión en oraciones para sincronizar con las imágenes
        
        Args:
            script: El guión completo
            
        Returns:
            list: Lista de oraciones
        """
        # Dividir por puntos, signos de exclamación y de interrogación
        sentences = re.split(r'(?<=[.!?])\s+', script)
        # Filtrar oraciones vacías
        return [s for s in sentences if s.strip()] 