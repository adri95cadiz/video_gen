#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import sys
from src.ai_agent import AIVideoAgent

def main():
    """
    Función principal para ejecutar el agente de generación de videos
    """
    # Configurar parser de argumentos
    parser = argparse.ArgumentParser(
        description="Generador de vídeos cortos para YouTube a partir de un prompt de texto"
    )
    
    parser.add_argument(
        "--prompt", 
        type=str, 
        help="Prompt de texto para generar el vídeo"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        default=None,
        help="Nombre del archivo de salida (opcional)"
    )
    
    parser.add_argument(
        "--max-words", 
        type=int, 
        default=200,
        help="Número máximo de palabras para el guión (por defecto: 200)"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="videos",
        help="Directorio donde guardar los videos generados (por defecto: videos)"
    )
    
    parser.add_argument(
        "--local", 
        action="store_true",
        help="Usar modelos locales en lugar de APIs externas (evita costes de API)"
    )

    parser.add_argument(
        "--local-script",
        action="store_true",
        help="Usar modelo local para generar el guión"
    )
    
    parser.add_argument(
        "--local-image",
        action="store_true",
        help="Usar modelo local para generar las imágenes"
    )

    parser.add_argument(
        "--local-voice",
        action="store_true",
        help="Usar modelo local para generar la voz"
    )    
    
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Forzar uso de GPU si está disponible"
    )
    
    # Analizar argumentos
    args = parser.parse_args()
    
    # Si no hay prompt, solicitarlo por consola
    if not args.prompt:
        print("Por favor, introduce un prompt para generar el vídeo:")
        args.prompt = input("> ")
    
    # Validar prompt
    if not args.prompt:
        print("Error: Se requiere un prompt para generar el vídeo")
        sys.exit(1)
    
    # Inicializar el agente
    try:
        # Verificar si hay GPU disponible
        if args.gpu:
            import torch
            if not torch.cuda.is_available():
                print("Advertencia: Se solicitó usar GPU pero no se detectó ninguna disponible.")
                print("Verificar que PyTorch esté instalado con soporte CUDA y que los drivers estén actualizados.")
                raise Exception("No hay GPU disponible")
            else:
                print(f"GPU disponible: {torch.cuda.get_device_name(0)}")
        
        # Crear el agente usando modelos locales si se especifica
        agent = AIVideoAgent(
            local_script=args.local_script or args.local,
            local_image=args.local_image or args.local,
            local_voice=args.local_voice or args.local,
            transcribe_audio=False
        )
        
        # Generar el video
        video_path = agent.generate_video(
            prompt=args.prompt,
            max_words=args.max_words,
            output_filename=args.output
        )
        
        print(f"\n✨ Video generado con éxito: {video_path}")
        
    except Exception as e:
        print(f"\n❌ Error durante la generación del video: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 