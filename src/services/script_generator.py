import os
import re
import openai
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class ScriptGenerator:
    def __init__(self, api_key=None, use_local_model=False, local_model_path=None):
        """
        Inicializa el generador de guiones.

        Args:
            api_key: API key de OpenAI (opcional)
            use_local_model: Si debe usar un modelo local en lugar de OpenAI
            local_model_path: Ruta al modelo local o nombre del modelo en Hugging Face
        """
        self.use_local_model = use_local_model

        if use_local_model:
            # Usar generación de texto con modelo local de HuggingFace
            print("Usando generación de texto con modelo local de transformers")
            # Cambiar el modelo predeterminado a uno más ligero y rápido
            self.local_model_path = local_model_path or "distilgpt2"
            self._setup_local_model()
        else:
            # Usar OpenAI
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "Se requiere una API key de OpenAI cuando no se usa modelo local")

            # Inicializar el cliente de OpenAI
            self.client = openai.OpenAI(api_key=self.api_key)

    def _setup_local_model(self):
        """Configura el modelo local para la generación de texto"""
        try:
            # Verificar disponibilidad de GPU
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(
                f"Inicializando modelo para generación de texto en {self.device}")

            # Cargar modelo y tokenizador
            try:
                # Intentar primero con un modelo más pequeño y optimizado para español
                self.tokenizer = AutoTokenizer.from_pretrained(self.local_model_path,
                                                               trust_remote_code=True)
                self.model = AutoModelForCausalLM.from_pretrained(self.local_model_path,
                                                                  torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                                                                  device_map="auto" if self.device == "cuda" else None,
                                                                  trust_remote_code=True)
                self.model_loaded = True
                print(
                    f"Modelo de texto cargado correctamente: {self.local_model_path}")
            except Exception as e:
                print(
                    f"Error al cargar el modelo {self.local_model_path}: {e}")
                # Fallback a GPT-2 en español si está disponible
                try:
                    print("Intentando cargar modelo de respaldo...")
                    fallback_model = "DeepESP/gpt2-spanish"
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        fallback_model)
                    self.model = AutoModelForCausalLM.from_pretrained(fallback_model,
                                                                      torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                                                                      device_map="auto" if self.device == "cuda" else None)
                    self.model_loaded = True
                    print(f"Modelo de respaldo cargado: {fallback_model}")
                except Exception as e2:
                    print(f"No se pudo cargar el modelo de respaldo: {e2}")
                    self.model_loaded = False

        except Exception as e:
            print(f"Error al configurar el modelo: {str(e)}")
            self.model_loaded = False

    def _validate_openai_model(self, requested_model):
        """
        Valida que el modelo solicitado esté disponible o sugiere una alternativa.

        Args:
            requested_model: El modelo solicitado por el usuario

        Returns:
            str: El modelo a utilizar (el solicitado o una alternativa)
        """
        # Lista de modelos comunes disponibles en OpenAI
        available_models = {
            "gpt-3.5-turbo": "Modelo estándar para generación de texto",
            "gpt-3.5-turbo-0125": "Versión actualizada de GPT-3.5 Turbo con mejor formato",
            "gpt-4o": "Modelo multimodal de última generación para textos creativos",
            "gpt-4o-mini": "Versión más económica y rápida de GPT-4o",
            "gpt-4-turbo-2024-04-09": "Versión actualizada de GPT-4 Turbo",
            "gpt-4.1": "El modelo GPT más reciente de OpenAI (2025-04-14)",
            "gpt-4.5-preview": "Versión preliminar con capacidades avanzadas"
        }

        # Verificar si el modelo solicitado está en la lista de modelos disponibles
        if requested_model in available_models:
            # Modelos que pueden requerir un tratamiento especial o verificación adicional
            if requested_model in ["gpt-4.1", "gpt-4.5-preview"]:
                print(
                    f"Advertencia: El modelo '{requested_model}' es muy reciente y puede no estar disponible en todas las regiones o requerir acceso especial.")
                print(f"Si la generación falla, se intentará con un modelo alternativo.")
            return requested_model

        # Si el modelo no está en la lista, buscar una alternativa adecuada
        print(
            f"El modelo '{requested_model}' no está en la lista de modelos comunes disponibles.")

        # Alternativas para diferentes familias de modelos
        if requested_model.startswith("gpt-4."):
            alternative = "gpt-4o"
            print(f"Usando '{alternative}' como alternativa.")
            return alternative
        elif requested_model.startswith("gpt-4"):
            alternative = "gpt-4-turbo-2024-04-09"
            print(f"Usando '{alternative}' como alternativa.")
            return alternative
        elif requested_model.startswith("gpt-3.5"):
            alternative = "gpt-3.5-turbo"
            print(f"Usando '{alternative}' como alternativa.")
            return alternative
        else:
            # Si no hay una alternativa clara, usar el modelo predeterminado
            print(f"Usando el modelo predeterminado 'gpt-3.5-turbo'.")
            return "gpt-3.5-turbo"

    def generate_script(self, prompt, max_words=200, model="gpt-3.5-turbo", custom_prompt=None):
        """
        Genera un guión para un vídeo corto basado en un prompt.

        Args:
            prompt: El prompt de entrada describiendo el contenido del vídeo
            max_words: Número máximo de palabras (por defecto 200 para vídeos cortos)
            model: Modelo de OpenAI a utilizar (ignorado si use_local_model=True)
                   Opciones:
                   - "gpt-3.5-turbo": Modelo estándar, buen equilibrio entre calidad y costo
                   - "gpt-3.5-turbo-0125": Versión más reciente con mejor capacidad de formateado
                   - "gpt-4o": Modelo de última generación para textos más creativos y detallados
                   - "gpt-4o-mini": Versión más económica y rápida de gpt-4o
                   - "gpt-4-turbo-2024-04-09": Modelo de alta capacidad para textos complejos
                   - "gpt-4.1": El modelo GPT más reciente de OpenAI (2025-04-14)
                   - "gpt-4.5-preview": Versión preliminar con capacidades avanzadas (puede requerir acceso especial)
            custom_prompt: Texto personalizado para las instrucciones al modelo (opcional)

        Returns:
            str: El guión generado
        """
        try:
            if self.use_local_model:
                # Para ahorrar tiempo con el modelo local, usar directamente el generador de plantillas
                # si la longitud del guión solicitado es corta
                if max_words <= 150:
                    print(
                        "Usando generador rápido basado en plantillas para guiones cortos")
                    return self._generate_with_template_model(prompt, max_words)

                if hasattr(self, 'model_loaded') and self.model_loaded:
                    return self._generate_with_transformer_model(prompt, max_words)
                else:
                    print(
                        "Modelo no disponible, usando generador simplificado como respaldo")
                    return self._generate_with_template_model(prompt, max_words)
            else:
                # Validar y posiblemente ajustar el modelo solicitado
                validated_model = self._validate_openai_model(model)
                return self._generate_with_openai(prompt, max_words, validated_model, custom_prompt)
        except Exception as e:
            print(f"Error al generar el guión: {str(e)}")
            # Fallback a la generación con plantillas si falla todo lo demás
            return self._generate_with_template_model(prompt, max_words)

    def _generate_with_transformer_model(self, prompt, max_words):
        """Genera un guión usando un modelo de transformers"""
        try:
            # Construir un prompt más corto y directo para mejorar la velocidad
            full_prompt = f"Escribe un guión breve de {max_words} palabras en español sobre: {prompt}. "

            # Tokenizar el prompt
            inputs = self.tokenizer(
                full_prompt, return_tensors="pt").to(self.device)

            # Reducir tokens máximos para mayor velocidad
            # Aproximadamente 1.5 tokens por palabra
            max_tokens = min(512, max_words * 1.5)

            # Generar texto con configuración más eficiente
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=int(max_tokens),
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.92,
                    top_k=50,
                    num_beams=1,         # Desactivar beam search para mayor velocidad
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # Decodificar la salida
            generated_text = self.tokenizer.decode(
                outputs[0], skip_special_tokens=True)

            # Extraer solo el guión (después del prompt)
            script = generated_text.replace(full_prompt, "").strip()

            # Limitar el número de palabras
            words = script.split()
            if len(words) > max_words:
                script = " ".join(words[:max_words])

            return script

        except Exception as e:
            print(f"Error en la generación con transformers: {str(e)}")
            # Fallback al generador simplificado
            return self._generate_with_template_model(prompt, max_words)

    def _generate_with_openai(self, prompt, max_words, model, custom_prompt=None):
        """Genera un guión usando la API de OpenAI"""
        # Configurar parámetros óptimos según el modelo
        if model in ["gpt-4.1", "gpt-4.5-preview"]:
            # Configuración optimizada para los modelos más recientes
            temperature = 0.7
            # Mayor capacidad para texto detallado
            max_tokens = min(800, max_words * 2)
            model_instructions = (f"Eres un experto en escribir guiones para vídeos cortos de YouTube. "
                                  f"Crea un guión excepcional, creativo y narrativamente poderoso de máximo {max_words} palabras. "
                                  f"El guión debe ser informativo, captar la atención y mantener al espectador interesado "
                                  f"con un estilo dinámico y cautivador. Utiliza recursos narrativos efectivos y un lenguaje preciso y elegante.")
        elif model.startswith("gpt-4o"):
            # Configuración optimizada para modelos GPT-4o
            temperature = 0.75
            max_tokens = min(600, max_words * 1.8)  # GPT-4o es más conciso
            model_instructions = (f"Eres un experto en escribir guiones para vídeos cortos de YouTube. "
                                  f"Crea un guión atractivo, creativo y conciso de máximo {max_words} palabras. "
                                  f"El guión debe ser informativo, directo y mantener la atención del espectador "
                                  f"con un estilo dinámico y atractivo. Utiliza un lenguaje sencillo pero impactante.")
        elif model.startswith("gpt-4"):
            # Configuración optimizada para modelos GPT-4
            temperature = 0.7
            # GPT-4 puede requerir más tokens
            max_tokens = min(700, max_words * 2)
            model_instructions = (f"Eres un experto en escribir guiones para vídeos cortos de YouTube. "
                                  f"Crea un guión detallado y atractivo de máximo {max_words} palabras. "
                                  f"El guión debe ser informativo, bien estructurado y mantener la atención del espectador. "
                                  f"Incluye una introducción breve, puntos clave claros y una conclusión impactante.")
        else:
            # Configuración para modelos GPT-3.5
            temperature = 0.7
            max_tokens = min(500, max_words * 1.5)
            model_instructions = (f"Eres un experto en escribir guiones para vídeos cortos de YouTube. "
                                  f"Crea un guión atractivo y conciso de máximo {max_words} palabras. "
                                  f"El guión debe ser informativo, directo y mantener la atención del espectador. "
                                  f"Evita introducciones largas y ve directo al tema principal.")

        # Instrucciones comunes para todos los modelos
        model_instructions += (f"El guión debe ser en Español. "
                               f"El texto debe ser plano listo para ser leído por un lector de texto.")

        # Usar custom_prompt si está disponible
        if custom_prompt:
            model_instructions = custom_prompt
            if "palabras" not in custom_prompt:
                model_instructions += f" El guión debe tener un máximo de {max_words} palabras."
            if "español" not in custom_prompt.lower() and "idioma" not in custom_prompt.lower():
                model_instructions += f" El guión debe ser en Español."
            model_instructions += f"El texto debe ser plano listo para ser leído por un lector de texto."

        # Realizar la llamada a la API
        try:
            completion = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": model_instructions},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=int(max_tokens),
                temperature=temperature
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error al generar con el modelo {model}: {str(e)}")

            # Si el modelo falla, intentar con uno alternativo más estable
            if model in ["gpt-4.1", "gpt-4.5-preview"]:
                fallback_model = "gpt-4o"
                print(f"Intentando con modelo alternativo: {fallback_model}")
                return self._generate_with_openai(prompt, max_words, fallback_model, custom_prompt)
            elif model.startswith("gpt-4"):
                fallback_model = "gpt-3.5-turbo"
                print(f"Intentando con modelo alternativo: {fallback_model}")
                return self._generate_with_openai(prompt, max_words, fallback_model, custom_prompt)
            else:
                # Si incluso el modelo de respaldo falla, usar el enfoque de plantillas
                print("Usando generador basado en plantillas como respaldo final")
                return self._generate_with_template_model(prompt, max_words)

    def _generate_with_template_model(self, prompt, max_words):
        """
        Genera un guión usando una alternativa local simplificada
        Basada en plantillas y algo de aleatoriedad
        """
        # Plantillas básicas para diferentes tipos de videos
        templates = [
            "Bienvenidos a este video sobre {tema}. Hoy vamos a explorar {aspecto_clave} y descubriremos {beneficio}. {punto_uno}. {punto_dos}. {punto_tres}. Para concluir, recuerden que {conclusion}.",

            "{tema} es un tema fascinante que está revolucionando {area}. {punto_uno}. Un aspecto interesante es que {punto_dos}. ¿Sabías que {punto_tres}? La próxima vez que pienses en {tema}, recuerda {conclusion}.",

            "¿Alguna vez te has preguntado sobre {tema}? Hoy responderemos esta pregunta. {punto_uno}. {punto_dos}. Lo más sorprendente es que {punto_tres}. En resumen, {conclusion}."
        ]

        # Extracción de palabras clave del prompt
        keywords = self._extract_keywords(prompt)
        tema = keywords[0] if keywords else "este tema interesante"

        # Generación de puntos para el guión
        aspectos = [
            f"{tema} se está volviendo cada vez más importante en nuestra vida diaria",
            f"cada día más personas descubren la utilidad de {tema}",
            f"los expertos recomiendan aprender sobre {tema} cuanto antes",
            f"comprender {tema} puede cambiar nuestra perspectiva del mundo"
        ]

        beneficios = [
            "cómo sacarle el máximo provecho",
            "las mejores estrategias para dominarlo",
            "técnicas que pocos conocen",
            "aplicaciones prácticas para el día a día"
        ]

        puntos = [
            f"Primero debemos entender que {tema} se basa en principios simples pero poderosos",
            f"Es importante destacar que {tema} ha evolucionado mucho en los últimos años",
            f"Los estudios demuestran que {tema} puede mejorar significativamente nuestra productividad",
            f"Muchas personas subestiman el impacto de {tema} en sus vidas",
            f"Una estrategia efectiva es aplicar {tema} de forma gradual y consistente",
            f"Los expertos recomiendan practicar {tema} al menos tres veces por semana",
            f"Un error común es pensar que {tema} es complicado, cuando en realidad es accesible para todos"
        ]

        conclusiones = [
            f"{tema} puede transformar positivamente tu vida si lo aplicas correctamente",
            f"la constancia es clave cuando se trata de {tema}",
            f"nunca es tarde para comenzar a explorar {tema}",
            f"compartir tu experiencia con {tema} puede ayudar a otros"
        ]

        # Seleccionar aleatoriamente una plantilla y elementos para rellenarla
        template = random.choice(templates)
        punto_uno = random.choice(puntos)
        punto_dos = random.choice([p for p in puntos if p != punto_uno])
        punto_tres = random.choice(
            [p for p in puntos if p not in [punto_uno, punto_dos]])

        # Construir el guión
        script = template.format(
            tema=tema,
            aspecto_clave=random.choice(aspectos),
            beneficio=random.choice(beneficios),
            punto_uno=punto_uno,
            punto_dos=punto_dos,
            punto_tres=punto_tres,
            area="nuestra sociedad" if "area" in template else "",
            conclusion=random.choice(conclusiones)
        )

        # Asegurar que no excede el límite de palabras
        words = script.split()
        if len(words) > max_words:
            script = " ".join(words[:max_words]) + "."

        return script

    def _extract_keywords(self, text):
        """Extrae palabras clave de un texto"""
        # Lista de palabras de parada (stop words) comunes en español
        stop_words = ["el", "la", "los", "las", "un", "una", "unos", "unas", "y", "o", "pero",
                      "de", "en", "por", "para", "con", "sin", "a", "ante", "bajo", "cabe",
                      "como", "cuando", "desde", "durante", "entre", "hacia", "hasta", "mediante",
                      "que", "quien", "cuyo", "donde", "cual", "es", "son", "ser", "estar"]

        # Convertir a minúsculas y eliminar caracteres especiales
        text = re.sub(r'[^\w\s]', '', text.lower())

        # Dividir en palabras y filtrar stop words
        words = [word for word in text.split(
        ) if word not in stop_words and len(word) > 3]

        # Devolver las palabras más largas primero (tienden a ser más significativas)
        return sorted(words, key=len, reverse=True)

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

    def split_into_scenes(self, script, num_scenes=5):
        """
        Divide un guión en un número específico de escenas.

        Args:
            script: El guión completo a dividir
            num_scenes: Número de escenas a generar

        Returns:
            list: Lista de textos de escenas
        """
        # Primero dividimos en oraciones
        sentences = self.split_script_into_sentences(script)

        # Si hay menos oraciones que escenas requeridas
        if len(sentences) <= num_scenes:
            return sentences

        # Calculamos cuántas oraciones por escena
        sentences_per_scene = len(sentences) // num_scenes
        remainder = len(sentences) % num_scenes

        scenes = []
        start_idx = 0

        for i in range(num_scenes):
            # Añadir una oración extra a las primeras 'remainder' escenas
            scene_size = sentences_per_scene + (1 if i < remainder else 0)
            end_idx = start_idx + scene_size

            # Unir las oraciones para esta escena
            scene_text = " ".join(sentences[start_idx:end_idx])
            scenes.append(scene_text)

            # Actualizar el índice de inicio para la siguiente escena
            start_idx = end_idx

        return scenes
