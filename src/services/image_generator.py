import os
import io
import time
import requests
from PIL import Image
import tempfile
import base64
import json

# Imports para modelo local básico
import torch
import numpy as np
from typing import Optional, List

class ImageGenerator:
    def __init__(self, api_key=None, use_local_model=False, local_model_path=None):
        """
        Inicializa el generador de imágenes.
        
        Args:
            api_key: API key de Stability AI (opcional)
            use_local_model: Si debe usar un modelo local en lugar de Stability AI
            local_model_path: Ruta al modelo local (no usado en la implementación básica)
        """
        self.use_local_model = use_local_model
        
        if use_local_model:
            # Usaremos una implementación alternativa simple
            self._setup_local_model()
        else:
            # Usar Stability AI API
            self.api_key = api_key or os.environ.get("STABILITY_API_KEY")
            if not self.api_key:
                raise ValueError("Se requiere una API key de Stability AI cuando no se usa modelo local")
            
            self.api_host = 'https://api.stability.ai'
    
    def _setup_local_model(self):
        """Configura una alternativa simple para generación local de imágenes"""
        try:
            print("Usando generación de imágenes local simplificada")
            # Verificar si hay GPU disponible
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Usando device: {self.device}")
        except Exception as e:
            raise Exception(f"Error al configurar generación de imágenes local: {str(e)}")
        
    def generate_image(self, prompt, output_path=None, width=1024, height=1024, steps=30):
        """
        Genera una imagen basada en un prompt
        
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
            if self.use_local_model:
                return self._generate_with_local_model(prompt, output_path, width, height, steps)
            else:
                return self._generate_with_stability(prompt, output_path, width, height, steps)
        except Exception as e:
            raise Exception(f"Error al generar la imagen: {str(e)}")
    
    def _generate_with_stability(self, prompt, output_path, width, height, steps):
        """Genera una imagen usando la API de Stability AI"""
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
    
    def _generate_with_local_model(self, prompt, output_path, width, height, steps):
        """
        Genera una imagen de color aleatorio con el texto como título
        (alternativa simple a Stable Diffusion)
        """
        # Crear un archivo temporal si no se especifica una ruta
        if not output_path:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            output_path = temp_file.name
            temp_file.close()
        
        # Generar una imagen sintética como alternativa
        # Creamos una imagen de color aleatorio con el texto como título
        
        # Generar un color aleatorio basado en el texto
        def text_to_color(text):
            import hashlib
            hash_obj = hashlib.md5(text.encode())
            hash_digest = hash_obj.digest()
            r = hash_digest[0] % 200 + 30  # Evitar colores muy oscuros
            g = hash_digest[1] % 200 + 30
            b = hash_digest[2] % 200 + 30
            return (r, g, b)
        
        # Crear una imagen del tamaño especificado
        color = text_to_color(prompt)
        img = Image.new('RGB', (width, height), color=color)
        
        # Añadir texto si PIL tiene soporte para dibujar
        try:
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(img)
            
            # Truncar el prompt si es muy largo
            display_text = prompt[:50] + "..." if len(prompt) > 50 else prompt
            
            # Usar una fuente por defecto
            try:
                # Intentar usar una fuente del sistema
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                # Usar fuente por defecto
                font = ImageFont.load_default()
            
            # Dibujar el texto centrado en la imagen
            text_width = font.getbbox(display_text)[2]
            x = (width - text_width) // 2
            y = 50  # Margen superior
            
            # Dibujar con sombra para mejor legibilidad
            shadow_color = (0, 0, 0)
            text_color = (255, 255, 255)
            
            # Sombra
            draw.text((x+2, y+2), display_text, font=font, fill=shadow_color)
            # Texto
            draw.text((x, y), display_text, font=font, fill=text_color)
            
            # Añadir un texto explicando que es una imagen generada localmente
            info_text = "Imagen generada localmente (sin usar API)"
            draw.text((10, height-30), info_text, font=font, fill=text_color)
            
        except Exception as e:
            print(f"No se pudo añadir texto a la imagen: {e}")
        
        # Guardar la imagen
        img.save(output_path)
        
        return output_path
    
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
                # Pequeña pausa para no sobrecargar la API (no necesario para modelo local)
                if not self.use_local_model:
                    time.sleep(1)
            except Exception as e:
                print(f"Error al generar imagen para la oración {i}: {str(e)}")
        
        return image_files 