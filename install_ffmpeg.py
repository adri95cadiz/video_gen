#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para ayudar a instalar FFmpeg en Windows
"""

import os
import sys
import zipfile
import tempfile
import shutil
import subprocess
import requests
from tqdm import tqdm

def download_file(url, output_path):
    """Descarga un archivo mostrando una barra de progreso"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f, tqdm(
        desc=f"Descargando FFmpeg",
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)

def extract_zip(zip_path, extract_to):
    """Extrae un archivo zip mostrando una barra de progreso"""
    with zipfile.ZipFile(zip_path) as z:
        for member in tqdm(z.infolist(), desc="Extrayendo archivos"):
            z.extract(member, extract_to)

def add_to_path(directory):
    """Agrega un directorio al PATH del usuario"""
    try:
        # Obtener el PATH actual
        cmd = f'[Environment]::GetEnvironmentVariable("PATH", "User")'
        result = subprocess.run(
            ["powershell", "-Command", cmd],
            capture_output=True,
            text=True
        )
        
        current_path = result.stdout.strip()
        
        # Verificar si el directorio ya está en el PATH
        if directory.lower() in current_path.lower():
            print(f"✅ {directory} ya está en el PATH")
            return True
        
        # Agregar directorio al PATH
        new_path = current_path + ";" + directory
        cmd = f'[Environment]::SetEnvironmentVariable("PATH", "{new_path}", "User")'
        
        subprocess.run(
            ["powershell", "-Command", cmd],
            check=True
        )
        
        print(f"✅ {directory} agregado al PATH")
        print("⚠️ Es posible que necesites reiniciar tu terminal o tu PC para que el cambio surta efecto")
        return True
    except Exception as e:
        print(f"❌ Error al agregar al PATH: {str(e)}")
        return False

def main():
    print("Instalador de FFmpeg para Windows")
    print("=================================")
    
    # Crear un directorio temporal para la descarga
    temp_dir = tempfile.mkdtemp()
    ffmpeg_zip = os.path.join(temp_dir, "ffmpeg.zip")
    
    # URL de descarga (versión estática para Windows)
    download_url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
    
    try:
        # Paso 1: Descargar FFmpeg
        print("\nPaso 1: Descargando FFmpeg...")
        download_file(download_url, ffmpeg_zip)
        
        # Paso 2: Extraer archivos
        print("\nPaso 2: Extrayendo archivos...")
        extract_dir = os.path.join(temp_dir, "ffmpeg_extract")
        extract_zip(ffmpeg_zip, extract_dir)
        
        # Encontrar el directorio extraído (usualmente contiene una carpeta raíz)
        extracted_dirs = [d for d in os.listdir(extract_dir) if os.path.isdir(os.path.join(extract_dir, d))]
        if not extracted_dirs:
            raise Exception("No se pudo encontrar el directorio extraído")
        
        extracted_root = os.path.join(extract_dir, extracted_dirs[0])
        
        # Paso 3: Mover a la ubicación final
        print("\nPaso 3: Instalando FFmpeg...")
        install_dir = os.path.expanduser("~\\ffmpeg")
        
        # Eliminar directorio existente si existe
        if os.path.exists(install_dir):
            shutil.rmtree(install_dir)
        
        # Mover archivos
        shutil.move(extracted_root, install_dir)
        
        # Paso 4: Agregar al PATH
        print("\nPaso 4: Configurando el PATH...")
        bin_dir = os.path.join(install_dir, "bin")
        add_to_path(bin_dir)
        
        print("\n✅ FFmpeg instalado correctamente!")
        print(f"📂 Ubicación: {install_dir}")
        print("\nPuedes verificar la instalación ejecutando: ffmpeg -version")
        print("Si no funciona, reinicia tu terminal o tu PC.")
        
    except Exception as e:
        print(f"\n❌ Error durante la instalación: {str(e)}")
    finally:
        # Limpiar archivos temporales
        try:
            shutil.rmtree(temp_dir)
        except:
            pass
    
if __name__ == "__main__":
    main() 