import os
import io
import time
import requests
from PIL import Image
import tempfile
import base64
import random
from glob import glob

# Imports para modelo local de Stable Diffusion
import torch


class ImageGenerator:
    def __init__(self, api_key=None, use_local_model=False, local_model_path=None, image_dir=None, device="cpu"):
        """
        Inicializa el generador de imágenes.

        Args:
            api_key: API key de Stability AI (opcional)
            use_local_model: Si debe usar un modelo local en lugar de Stability AI
            local_model_path: Ruta al modelo local (por defecto se usará stabilityai/stable-diffusion-2-1)
            image_dir: Directorio con imágenes existentes para usar en lugar de generar nuevas
            device: Dispositivo a utilizar (cpu o cuda)
        """
        self.use_local_model = use_local_model
        self.image_dir = image_dir
        self.use_existing_images = image_dir is not None and os.path.isdir(
            image_dir)
        self.device = device
        self.used_images = set()  # Para seguimiento de imágenes ya utilizadas
        self.use_simplified = False  # Inicializar el flag para modo simplificado
        self.pipe = None  # Inicializar el pipe como None

        if self.use_existing_images:
            print(
                f"Usando imágenes existentes del directorio: {self.image_dir}")
            # Verificar si hay imágenes en el directorio
            self.available_images = self._get_available_images()
            if not self.available_images:
                print(
                    "Advertencia: No se encontraron imágenes en el directorio especificado.")
                self.use_existing_images = False

        if use_local_model and not self.use_existing_images:
            # Usar modelo local - no necesita API key
            self.local_model_path = local_model_path or "stabilityai/stable-diffusion-xl-base-1.0"
            self._setup_local_model()
        elif not self.use_existing_images:
            # Usar Stability AI API
            self.api_key = api_key or os.environ.get("STABILITY_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "Se requiere una API key de Stability AI cuando no se usa modelo local ni imágenes existentes")

            self.api_host = 'https://api.stability.ai'
            self.context = None  # Para almacenar contexto para traducción

    def _get_available_images(self):
        """Obtiene la lista de imágenes disponibles en el directorio especificado"""
        image_extensions = ['*.jpg', '*.jpeg',
                            '*.png', '*.bmp', '*.gif', '*.webp']
        available_images = []

        for ext in image_extensions:
            available_images.extend(glob(os.path.join(self.image_dir, ext)))

        print(f"Se encontraron {len(available_images)} imágenes disponibles")
        return available_images

    def _setup_local_model(self):
        """Configura el modelo local para generación de imágenes"""
        try:
            print(
                f"Usando generación de imágenes local con modelo: {self.local_model_path}")

            # Verificar si hay GPU disponible
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Usando device: {self.device}")

            if self.device == "cuda":
                cuda_version = torch.version.cuda
                print(f"¡CUDA disponible! Versión: {cuda_version}")
                print(f"Dispositivo GPU: {torch.cuda.get_device_name(0)}")

            # Importar diffusers
            try:
                from diffusers import StableDiffusionPipeline, AutoPipelineForText2Image
            except ImportError as e:
                print(f"Error importando diffusers: {e}")
                print(
                    "Asegúrate de tener instalado diffusers: pip install diffusers>=0.26.0")
                raise

            # Configurar dtype según el dispositivo
            if self.device == "cuda":
                dtype = torch.float16  # Usar precisión media en GPU para memoria
            else:
                dtype = torch.float32  # Precisión completa en CPU

            # Usar la pipeline adecuada según el modelo
            print("Cargando modelo Stable Diffusion...")
            try:
                self.pipe = StableDiffusionPipeline.from_pretrained(
                    self.local_model_path,
                    torch_dtype=dtype
                )
                print("Modelo SD estándar cargado correctamente")
            except Exception as e:
                print(f"Error cargando modelo SD estándar: {e}")
                raise

            # Verificar que el pipe se ha cargado correctamente
            if self.pipe is None:
                raise Exception("El pipeline no se inicializó correctamente")

            # Mover el modelo al dispositivo correspondiente
            self.pipe = self.pipe.to(self.device)
            print(f"Modelo movido al dispositivo: {self.device}")

            # Optimizaciones
            if self.device == "cuda":
                print("Aplicando optimizaciones para GPU...")
                # En GPU usar half precision para VRAM
                if hasattr(self.pipe, 'enable_attention_slicing'):
                    self.pipe.enable_attention_slicing()
                    print("Attention slicing habilitado")
                if hasattr(self.pipe, 'enable_model_cpu_offload'):
                    self.pipe.enable_model_cpu_offload()
                    print("Model CPU offload habilitado")
            else:
                print("Aplicando optimizaciones para CPU...")
                if hasattr(self.pipe, 'enable_attention_slicing'):
                    self.pipe.enable_attention_slicing()
                    print("Attention slicing habilitado")
                # Usar sequential CPU offload para menos memoria
                if hasattr(self.pipe, 'enable_sequential_cpu_offload'):
                    self.pipe.enable_sequential_cpu_offload()
                    print("Sequential CPU offload habilitado")

            print("Configuración del modelo completada con éxito")
            return True

        except Exception as e:
            print(f"Error al configurar modelo de difusión: {str(e)}")
            print("Volviendo al modo simplificado con modelo de respaldo")
            self.use_simplified = True

            # Intentar cargar un modelo más sencillo y ligero como fallback
            try:
                print("Intentando cargar modelo de respaldo...")
                from diffusers import StableDiffusionPipeline

                # Usar un modelo más ligero y estable como respaldo
                fallback_model = "CompVis/stable-diffusion-v1-4"

                self.pipe = StableDiffusionPipeline.from_pretrained(
                    fallback_model,
                    torch_dtype=torch.float32  # Usar float32 para mayor compatibilidad
                )
                # Forzar CPU para evitar problemas de memoria
                self.pipe = self.pipe.to("cpu")
                print(
                    f"Modelo de respaldo {fallback_model} cargado correctamente en CPU")
                return True
            except Exception as fallback_e:
                print(f"Error al cargar modelo de respaldo: {fallback_e}")
                print("No se pudo configurar ningún modelo de difusión")
                self.pipe = None
                return False

    def generate_image(self, prompt, output_path=None, width=1024, height=1024, steps=30):
        """
        Genera o selecciona una imagen basada en un prompt

        Args:
            prompt: Descripción de la imagen a generar (ignorado si se usan imágenes existentes)
            output_path: Ruta donde guardar la imagen (opcional)
            width: Ancho de la imagen (por defecto 1024)
            height: Alto de la imagen (por defecto 1024)
            steps: Pasos de generación (más pasos = más calidad pero más costoso)

        Returns:
            str: Ruta a la imagen generada o seleccionada
        """
        try:
            if self.use_existing_images:
                return self._select_image_from_directory(output_path)
            elif self.use_local_model:
                return self._generate_with_local_model(prompt, output_path, width, height, steps)
            else:
                return self._generate_with_stability(prompt, output_path, width, height, steps)
        except Exception as e:
            raise Exception(f"Error al generar la imagen: {str(e)}")

    def _generate_with_stability(self, prompt, output_path, width, height, steps):
        """Genera una imagen usando la API de Stability AI"""
        # Usar el modelo más económico de Stability AI
        engine_id = "stable-image"
        level = "core"

        # Añadir contexto si existe
        if self.context:
            context_prompt = f"{self.context}: {prompt}"
        else:
            context_prompt = prompt

        response = requests.post(
            f"{self.api_host}/v2beta/{engine_id}/generate/{level}",
            headers={
                "Accept": "image/*",
                "Authorization": f"Bearer {self.api_key}"
            },
            files={"none": ''},
            data={
                "prompt": f"{context_prompt}",
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
        # Verificar si el pipe se inicializó correctamente
        if not hasattr(self, 'pipe') or self.pipe is None:
            raise Exception(
                "El pipeline de Stable Diffusion no está inicializado. Verifica la instalación de las dependencias.")

        # Crear un archivo temporal si no se especifica una ruta
        if not output_path:
            temp_file = tempfile.NamedTemporaryFile(
                delete=False, suffix='.png')
            output_path = temp_file.name
            temp_file.close()

        # Ajustar dimensiones a múltiplos de 8 (requerimiento de Stable Diffusion)
        width = (width // 8) * 8
        height = (height // 8) * 8

        # Prompt para generar mejor calidad
        positive_prompt = f"high quality, detailed, professional photography, {prompt}"
        negative_prompt = "ugly, blurry, low quality, distorted, deformed, watermark, text, signature"

        try:
            # Para modelos SD anteriores o el modelo de fallback
            print(f"Generando imagen con modelo estándar ({steps} pasos)...")

            with torch.no_grad():
                # Usar parámetros consistentes con el modelo
                try:
                    # Intentar con parámetros completos
                    image = self.pipe(
                        prompt=positive_prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=steps,
                        guidance_scale=7.5,
                        width=width,
                        height=height
                    ).images[0]
                except TypeError as te:
                    # Si hay error con los parámetros, intentar sin width/height (algunos modelos no los aceptan)
                    print(f"Ajustando parámetros: {te}")
                    image = self.pipe(
                        prompt=positive_prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=steps,
                        guidance_scale=7.5
                    ).images[0]

            # Guardar la imagen
            image.save(output_path)
            print(f"Imagen guardada en: {output_path}")

            return output_path

        except Exception as e:
            print(f"Error al generar imagen con modelo local: {str(e)}")
            # Si falla, intentar un modo simplificado con menos pasos y dimensiones más pequeñas
            try:
                print("Intentando con configuración simplificada...")
                # Intentar con dimensiones y pasos reducidos
                width, height = 512, 512
                steps = 20

                # Crear una imagen básica con PIL como último recurso
                from PIL import Image, ImageDraw, ImageFont
                import textwrap

                img = Image.new('RGB', (width, height), color=(0, 0, 0))
                d = ImageDraw.Draw(img)

                # Añadir el prompt como texto en la imagen
                font = ImageFont.load_default()
                wrapped_text = textwrap.fill(prompt, width=40)
                d.text((10, 10), wrapped_text, fill=(255, 255, 255), font=font)

                # Guardar la imagen
                img.save(output_path)
                print(
                    f"Generada imagen de texto como fallback en: {output_path}")
                return output_path

            except Exception as fallback_error:
                print(f"Error en modo simplificado: {str(fallback_error)}")
                raise Exception(f"No se pudo generar la imagen: {str(e)}")

    def _select_image_from_directory(self, output_path=None):
        """
        Selecciona una imagen del directorio especificado evitando repeticiones
        cuando sea posible.
        """
        if not self.available_images:
            raise Exception(
                "No hay imágenes disponibles en el directorio especificado")

        # Obtener imágenes que aún no se han usado
        unused_images = [
            img for img in self.available_images if img not in self.used_images]

        # Si todas las imágenes ya se han usado, reiniciar el seguimiento
        if not unused_images:
            print("Todas las imágenes ya han sido utilizadas. Reiniciando selección.")
            self.used_images.clear()
            unused_images = self.available_images

        # Seleccionar una imagen no utilizada
        selected_image = random.choice(unused_images)
        self.used_images.add(selected_image)  # Marcar como utilizada

        # Si no se especifica una ruta de salida, devolvemos la ruta de la imagen seleccionada
        if not output_path:
            return selected_image

        # Si se especifica una ruta, copiamos la imagen
        try:
            img = Image.open(selected_image)
            img.save(output_path)
            return output_path
        except Exception as e:
            raise Exception(
                f"Error al copiar la imagen seleccionada: {str(e)}")

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

        # Si estamos usando imágenes existentes, asegúrate de que hay suficientes
        if self.use_existing_images and len(sentences) > len(self.available_images):
            print(
                f"Advertencia: Hay más oraciones ({len(sentences)}) que imágenes disponibles ({len(self.available_images)}). Algunas imágenes se repetirán.")

        for i, sentence in enumerate(sentences):
            output_path = os.path.join(output_dir, f"image_{i}.png")

            if not self.use_existing_images:
                # Preparar el prompt para generar una imagen relevante
                # Acortar la oración si es muy larga
                max_prompt_len = 200
                prompt = sentence[:max_prompt_len] if len(
                    sentence) > max_prompt_len else sentence

                # Añadir un prefijo para mejorar la calidad
                enhanced_prompt = f"Imagen de alta calidad que muestra: {prompt}"
            else:
                # Si usamos imágenes existentes, el prompt se ignora
                enhanced_prompt = ""

            try:
                self.generate_image(enhanced_prompt, output_path)
                image_files.append(output_path)
                # Pequeña pausa para no sobrecargar la API (no necesario para modelo local)
                if not self.use_local_model and not self.use_existing_images:
                    time.sleep(1)
            except Exception as e:
                print(f"Error al generar imagen para la oración {i}: {str(e)}")

        return image_files
