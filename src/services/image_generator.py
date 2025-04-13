import os
import io
import time
import requests
from PIL import Image
import tempfile
import base64
import json

# Imports para modelo local de Stable Diffusion
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler


class ImageGenerator:
    def __init__(self, api_key=None, use_local_model=False, local_model_path=None, context=None):
        """
        Inicializa el generador de imágenes.

        Args:
            api_key: API key de Stability AI (opcional)
            use_local_model: Si debe usar un modelo local en lugar de Stability AI
            local_model_path: Ruta al modelo local (por defecto se usará stabilityai/stable-diffusion-2-1)
        """
        self.use_local_model = True
        self.context = context

        if use_local_model:
            # Usar modelo local - no necesita API key
            self.local_model_path = local_model_path or "stabilityai/stable-diffusion-2-1"
            self._setup_local_model()
        else:
            # Usar Stability AI API
            self.api_key = api_key or os.environ.get("STABILITY_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "Se requiere una API key de Stability AI cuando no se usa modelo local")

            self.api_host = 'https://api.stability.ai'

    def _setup_local_model(self):
        """Configura el modelo local para generación de imágenes"""
        try:
            print("Usando generación de imágenes local con Diffusers")

            # Verificar si hay GPU disponible
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Usando device: {self.device}")

            if self.device == "cuda":
                cuda_version = torch.version.cuda
                print(f"¡CUDA disponible! Versión: {cuda_version}")
                print(f"Dispositivo GPU: {torch.cuda.get_device_name(0)}")

            # Importar diffusers
            from diffusers import StableDiffusionPipeline

            # Modelo a utilizar - más ligero para CPU
            model_id = "stabilityai/stable-diffusion-2-1-base"  # Versión más ligera

            # Configurar dtype según el dispositivo
            if self.device == "cuda":
                dtype = torch.float16  # Usar precisión media en GPU para memoria
            else:
                dtype = torch.float32  # Precisión completa en CPU

            # Cargar el pipeline
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                safety_checker=None  # Desactivar para velocidad (opcional)
            )
            self.pipe = self.pipe.to(self.device)

            # Optimizaciones
            if self.device == "cuda":
                print("Aplicando optimizaciones para GPU...")
                # En GPU usar half precision para VRAM
                self.pipe.enable_attention_slicing()
            else:
                print("Aplicando optimizaciones para CPU...")
                self.pipe.enable_attention_slicing()
                # Usar sequential CPU offload para menos memoria
                self.pipe.enable_sequential_cpu_offload()

        except Exception as e:
            print(f"Error al configurar modelo de difusión: {str(e)}")
            print("Volviendo al modo simplificado")
            self.use_simplified = True

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
                return self._generate_with_stability(prompt, output_path)
        except Exception as e:
            raise Exception(f"Error al generar la imagen: {str(e)}")

    def _generate_with_stability(self, prompt, output_path):
        """Genera una imagen usando la API de Stability AI"""
        # Usar el modelo más económico de Stability AI
        engine_id = "stable-image"
        level = "core"

        response = requests.post(
            f"{self.api_host}/v2beta/{engine_id}/generate/{level}",
            headers={
                "Accept": "image/*",
                "Authorization": f"Bearer {self.api_key}"
            },
            files={"none": ''},
            data={
                "prompt": f"{prompt}",
                "negative_prompt": "ugly, blurry, poor quality, distorted, deformed",
                "output_format": "png",
            },
        )

        if response.status_code != 200:
            raise Exception(
                f"Error en la API: {response.status_code} {response.text}")

        data = response.json()

        # Guardar la imagen
        if not output_path:
            # Crear un archivo temporal si no se especifica una ruta
            temp_file = tempfile.NamedTemporaryFile(
                delete=False, suffix='.png')
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
        """Genera una imagen usando el modelo local de Stable Diffusion"""
        # Crear un archivo temporal si no se especifica una ruta
        if not output_path:
            temp_file = tempfile.NamedTemporaryFile(
                delete=False, suffix='.png')
            output_path = temp_file.name
            temp_file.close()

        # Ajustar dimensiones a múltiplos de 8 (requerimiento de Stable Diffusion)
        width = (width // 8) * 8
        height = (height // 8) * 8

        # Generar imagen
        negative_prompt = "ugly, blurry, poor quality, distorted, deformed"

        # Generar la imagen con el pipeline
        with torch.no_grad():
            image = self.pipe(
                prompt=f"{prompt}",
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=7.5,
                width=width,
                height=height
            ).images[0]

        # Guardar la imagen
        image.save(output_path)

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
            prompt = sentence[:max_prompt_len] if len(
                sentence) > max_prompt_len else sentence

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
