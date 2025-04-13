import os
import io
import time
import requests
from PIL import Image
import tempfile
import base64
import json

class ImageGenerator:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("STABILITY_API_KEY")
        if not self.api_key:
            raise ValueError("Se requiere una API key de Stability AI")
        
        self.api_host = 'https://api.stability.ai'
        
    def generate_image(self, prompt, output_path=None, width=1024, height=1024, steps=30):
        """
        Genera una imagen basada en un prompt usando Stability AI
        
        Args:
            prompt: Descripción de la imagen a generar
            output_path: Ruta donde guardar la imagen (opcional)
            width: Ancho de la imagen (por defecto 1024)
            height: Alto de la imagen (por defecto 1024)
            steps: Pasos de generación (más pasos = más calidad pero más costoso)
            
        Returns:
            str: Ruta a la imagen generada
        """
        try:
            # Usar el modelo más económico de Stability AI
            engine_id = "stable-diffusion-xl-1024-v1-0"
            
            response = requests.post(
                f"{self.api_host}/v1/generation/{engine_id}/text-to-image",
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                },
                json={
                    "text_prompts": [
                        {
                            "text": prompt,
                            "weight": 1.0
                        }
                    ],
                    "cfg_scale": 7.0,  # Adherencia al prompt
                    "height": height,
                    "width": width,
                    "samples": 1,  # Solo una imagen para reducir costos
                    "steps": steps,  # Menos pasos para reducir costos
                },
            )
            
            if response.status_code != 200:
                raise Exception(f"Error en la API: {response.status_code} {response.text}")
                
            data = response.json()
            
            # Guardar la imagen
            if not output_path:
                # Crear un archivo temporal si no se especifica una ruta
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                output_path = temp_file.name
                temp_file.close()
            
            for i, image in enumerate(data["artifacts"]):
                # Solo procesamos la primera imagen (samples=1)
                img_data = base64.b64decode(image["base64"])
                img = Image.open(io.BytesIO(img_data))
                img.save(output_path)
                break
                
            return output_path
            
        except Exception as e:
            raise Exception(f"Error al generar la imagen: {str(e)}")
    
    def generate_images_for_sentences(self, sentences, output_dir="temp_images"):
        """
        Genera imágenes para cada oración del guión
        
        Args:
            sentences: Lista de oraciones
            output_dir: Directorio donde guardar las imágenes
            
        Returns:
            list: Lista de rutas a las imágenes generadas
        """
        os.makedirs(output_dir, exist_ok=True)
        image_files = []
        
        for i, sentence in enumerate(sentences):
            output_path = os.path.join(output_dir, f"image_{i}.png")
            
            # Preparar el prompt para generar una imagen relevante
            # Acortar la oración si es muy larga
            max_prompt_len = 200
            prompt = sentence[:max_prompt_len] if len(sentence) > max_prompt_len else sentence
            
            # Añadir un prefijo para mejorar la calidad
            enhanced_prompt = f"Imagen de alta calidad que muestra: {prompt}"
            
            try:
                self.generate_image(enhanced_prompt, output_path)
                image_files.append(output_path)
                # Pequeña pausa para no sobrecargar la API
                time.sleep(1)
            except Exception as e:
                print(f"Error al generar imagen para la oración {i}: {str(e)}")
        
        return image_files 