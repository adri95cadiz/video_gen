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
        required=True,
        help="Prompt de texto para generar el vídeo"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Nombre del archivo de salida (opcional)"
    )

    parser.add_argument(
        "--num-scenes",
        type=int,
        default=5,
        help="Número de escenas para el guión (por defecto: 5)"
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
        default=False,
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
        "--script-model",
        type=str,
        default="gpt-3.5-turbo",
        help="Modelo de OpenAI para generar el guión"
    )

    parser.add_argument(
        "--gpu",
        action="store_true",
        default=False,
        help="Forzar uso de GPU si está disponible"
    )

    parser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="Directorio con imágenes existentes para usar en lugar de generar nuevas"
    )

    parser.add_argument(
        "--image-model",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Modelo de imagen para usar en lugar de generar nuevas"
    )

    parser.add_argument(
        "--background-music",
        type=str,
        default=None,
        help="Ruta a un archivo de música para usar como fondo en el video"
    )

    # Analizar argumentos
    args = parser.parse_args()

    # Si no hay prompt, solicitarlo por consola
    if not args.prompt:
        print("Por favor, introduce un prompt para generar el vídeo:")
        args.prompt = input("> ")

    # Validar prompt
    if not args.prompt or len(args.prompt.strip()) < 3:
        print("Por favor proporciona un prompt válido con al menos 3 caracteres")
        return

    # Validar que el archivo de música de fondo existe
    if args.background_music and not os.path.exists(args.background_music):
        print(
            f"El archivo de música de fondo no existe: {args.background_music}")
        return

    # Inicializar el agente
    try:
        # Verificar si hay GPU disponible
        if args.gpu:
            import torch
            if not torch.cuda.is_available():
                print(
                    "Advertencia: Se solicitó usar GPU pero no se detectó ninguna disponible.")
                print(
                    "Verificar que PyTorch esté instalado con soporte CUDA y que los drivers estén actualizados.")
                raise Exception("No hay GPU disponible")
            else:
                print(f"GPU disponible: {torch.cuda.get_device_name(0)}")

        # Verificar si se especificó un directorio de imágenes y si existe
        if args.image_dir and not os.path.isdir(args.image_dir):
            print(
                f"Advertencia: El directorio de imágenes especificado no existe: {args.image_dir}")
            print("Se generarán imágenes según el prompt.")
            args.image_dir = None

        # Crear el agente usando modelos locales si se especifica
        agent = AIVideoAgent(
            local_script=args.local_script or args.local,
            local_image=args.local_image or args.local,
            local_voice=args.local_voice or args.local,
            transcribe_audio=False,
            image_dir=args.image_dir,
            output_dir=args.output_dir,
            music_reference=args.background_music,
            script_model=args.script_model,
            image_model=args.image_model
        )

        # Generar el video
        video_path = agent.generate_video(
            video_topic=args.prompt,
            max_words=args.max_words,
            num_scenes=args.num_scenes
        )

        print(f"\n✨ Video generado con éxito: {video_path}")

    except Exception as e:
        print(f"\n❌ Error durante la generación del video: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
