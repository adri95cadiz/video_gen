import os
import re
import openai
import random

class ScriptGenerator:
    def __init__(self, api_key=None, use_local_model=False, local_model_path=None):
        """
        Inicializa el generador de guiones.
        
        Args:
            api_key: API key de OpenAI (opcional)
            use_local_model: Si debe usar un modelo local en lugar de OpenAI
            local_model_path: Ruta al modelo local (no usado en la implementación simplificada)
        """
        self.use_local_model = use_local_model
        
        if use_local_model:
            # Usar generación de texto local simplificada
            print("Usando generación de texto local simplificada")
        else:
            # Usar OpenAI
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("Se requiere una API key de OpenAI cuando no se usa modelo local")
            
            # Inicializar el cliente de OpenAI
            self.client = openai.OpenAI(api_key=self.api_key)
    
    def generate_script(self, prompt, max_words=200, model="gpt-3.5-turbo"):
        """
        Genera un guión para un vídeo corto basado en un prompt.
        
        Args:
            prompt: El prompt de entrada describiendo el contenido del vídeo
            max_words: Número máximo de palabras (por defecto 200 para vídeos cortos)
            model: Modelo de OpenAI a utilizar (ignorado si use_local_model=True)
            
        Returns:
            str: El guión generado
        """
        try:
            if self.use_local_model:
                return self._generate_with_local_model(prompt, max_words)
            else:
                return self._generate_with_openai(prompt, max_words, model)
        except Exception as e:
            raise Exception(f"Error al generar el guión: {str(e)}")
    
    def _generate_with_openai(self, prompt, max_words, model):
        """Genera un guión usando la API de OpenAI"""
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
    
    def _generate_with_local_model(self, prompt, max_words):
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
        punto_tres = random.choice([p for p in puntos if p not in [punto_uno, punto_dos]])
        
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
        words = [word for word in text.split() if word not in stop_words and len(word) > 3]
        
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