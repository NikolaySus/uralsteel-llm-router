from concurrent import futures
from io import BytesIO
import json
import os
import re
import subprocess
import uuid
import asyncio

import requests
from minio import Minio
from minio.error import S3Error

from util import process_engineer_url


import requests
import os
from pathlib import Path

def download_to_path(download_url: str, file_path: str, chunk_size: int = 8192) -> str:
    """
    Download a file from a URL to a specific file path.
    
    Args:
        download_url: The download URL (can be random/unstructured)
        file_path: Complete path including directories and filename (e.g., './data/images/photo.jpg')
        chunk_size: Size of chunks for streaming download (default: 8KB)
    
    Returns:
        str: The absolute path where the file was saved
    
    Raises:
        Exception: If download fails or path cannot be created
    """
    try:
        # Convert to Path object for robust path handling
        target_path = Path(file_path).resolve()
        
        # Extract directory from file path and create it if it doesn't exist
        target_dir = target_path.parent
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Send GET request with streaming
        response = requests.get(download_url, stream=True, timeout=30)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        # Get file size for progress info (optional)
        total_size = int(response.headers.get('content-length', 0))
        
        print(f"Downloading to: {target_path}")
        if total_size > 0:
            print(f"File size: {total_size / (1024*1024):.2f} MB")
        
        # Download the file in chunks
        downloaded_size = 0
        with open(target_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    file.write(chunk)
                    downloaded_size += len(chunk)
                    
                    # Optional: Print progress
                    if total_size > 0:
                        percent = (downloaded_size / total_size) * 100
                        print(f"\rProgress: {percent:.1f}%", end='', flush=True)
        
        print(f"\n✅ Successfully downloaded to: {target_path}")
        return str(target_path)
        
    except requests.exceptions.RequestException as e:
        raise Exception(f"Download failed: {str(e)}")
    except PermissionError as e:
        raise Exception(f"Permission denied: {str(e)}")
    except Exception as e:
        raise Exception(f"An error occurred: {str(e)}")


# Function with retry logic:
def download_to_path_with_retry(download_url: str, file_path: str, max_retries: int = 3) -> str:
    """
    Download with automatic retry on failure.
    """
    for attempt in range(max_retries):
        try:
            return download_to_path(download_url, file_path)
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Attempt {attempt + 1} failed. Retrying in {wait_time}s... Error: {str(e)}")
                import time
                time.sleep(wait_time)
            else:
                print(f"Failed after {max_retries} attempts: {str(e)}")
                raise


base_url="http://localhost:9621"
url = f"{base_url}/documents"

# Заголовки запроса
headers = {
    "accept": "application/json",
}

try:
    # Выполняем POST-запрос
    response = requests.post(url, headers=headers)
    response.raise_for_status()  # Вызовет исключение для HTTP ошибок

    # Парсим JSON-ответ
    data = response.json()
    file_paths = [chunk['file_path'] for chunk in data['statuses']['processed']]
except requests.exceptions.RequestException as e:
    print(f"Ошибка при выполнении запроса к RAG-сервису: {e}")
    # Возвращаем пустые значения в случае ошибки
    raise

save_dir = "./indexed_now/"

for url_now in file_paths:
    if "hierarchy_trailing_20260126_182731" in url_now:
        url_now = url_now.split("hierarchy_trailing_20260126_182731", 1)[1]
        url_now = "hierarchy_trailing_20260126_182731" + url_now
    url_now, path = process_engineer_url(url_now)
    # Usage with retry
    try:
        file_path = save_dir + path
        download_to_path_with_retry(url_now, file_path, max_retries=3)
    except Exception as e:
        print(f"Download failed: {e}")
