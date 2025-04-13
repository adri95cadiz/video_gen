#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para probar que el entorno tiene las dependencias correctas
"""

import os
import sys
import subprocess
import shutil

def check_dependency(module_name, package_name=None):
    package_name = package_name or module_name
    try:
        __import__(module_name)
        print(f"✅ {package_name} instalado correctamente")
        return True
    except ImportError as e:
        print(f"❌ {package_name} NO está instalado: {str(e)}")
        return False

def check_ffmpeg():
    """Verifica si ffmpeg está instalado y disponible en el PATH"""
    try:
        # Buscar ffmpeg en el PATH
        ffmpeg_path = shutil.which('ffmpeg')
        if ffmpeg_path:
            print(f"✅ FFmpeg encontrado en: {ffmpeg_path}")
            return True
        else:
            print("❌ FFmpeg NO está en el PATH")
            return False
    except Exception as e:
        print(f"❌ Error al verificar FFmpeg: {str(e)}")
        return False

def main():
    print("Verificando dependencias para VideoGen AI...")
    print("\n1. Verificando paquetes Python:")
    
    dependencies = {
        "dotenv": "python-dotenv",
        "openai": "openai",
        "elevenlabs": "elevenlabs",
        "PIL": "pillow",
        "moviepy": "moviepy",
        "pydub": "pydub",
        "requests": "requests",
        "tqdm": "tqdm"
    }
    
    all_installed = True
    
    for module_name, package_name in dependencies.items():
        if not check_dependency(module_name, package_name):
            all_installed = False
    
    print("\n2. Verificando FFmpeg (necesario para audio/video):")
    ffmpeg_installed = check_ffmpeg()
    
    print("\n3. Verificando estructura del proyecto:")
    required_files = [
        "main.py",
        "src/ai_agent.py",
        "src/services/script_generator.py",
        "src/services/voice_generator.py",
        "src/services/image_generator.py",
        "src/services/video_generator.py"
    ]
    
    project_structure_ok = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} encontrado")
        else:
            print(f"❌ {file_path} NO encontrado")
            project_structure_ok = False
    
    print("\nRESUMEN:")
    if all_installed and ffmpeg_installed and project_structure_ok:
        print("✨ Todo configurado correctamente!")
        print("Puedes ejecutar el agente con: python main.py --prompt \"Tu prompt aquí\"")
    else:
        print("❌ Hay problemas con la configuración:")
        
        if not all_installed:
            print("   - Faltan algunas dependencias de Python. Ejecuta: pip install -r requirements.txt")
        
        if not ffmpeg_installed:
            print("   - FFmpeg no está instalado o no está en el PATH.")
            print("     * Windows: Descarga desde ffmpeg.org e incluye en PATH")
            print("     * macOS: brew install ffmpeg")
            print("     * Linux: sudo apt install ffmpeg")
        
        if not project_structure_ok:
            print("   - Hay archivos importantes del proyecto que faltan")
    
if __name__ == "__main__":
    main() 