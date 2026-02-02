"""
Вспомогательные функции для обработки документов, изображений и веб-поиска.
"""

import re
import base64
import urllib
import imghdr
import asyncio
import json
import os
import tempfile
import subprocess

from markitdown import MarkItDown
import requests
import httpx
import pymupdf
from tavily import TavilyClient

try:
    from docx import Document
    HAS_PYTHON_DOCX = True
except ImportError:
    HAS_PYTHON_DOCX = False

from logger import logger

from minio_util import generate_presigned_download_url


def get_messages_wo_b64_images(messages):
    """Removes all base64-encoded images from messages.
    
    Modifies messages in-place, removing all image_url content blocks
    that contain data URLs (base64 images) while keeping text blocks
    and other content.
    
    Args:
        messages: List of messages in OpenAI API format. Each message
                 is a dict with "role" and "content" keys. Content can be
                 either a string or a list of content blocks.
    """
    for message in messages:
        if isinstance(message, dict) and "content" in message:
            content = message["content"]
            # If content is a list of content blocks
            if isinstance(content, list):
                # Filter out image_url blocks with base64 data
                filtered_content = []
                for block in content:
                    if isinstance(block, dict):
                        # Keep text blocks and other non-image content
                        if block.get("type") != "image_url":
                            filtered_content.append(block)
                        # Keep image_url only if it's not a base64 data URL
                        elif block.get("type") == "image_url":
                            image_url = block.get("image_url", {}).get("url", "")
                            # Only remove if it's a data URL (base64)
                            if not image_url.startswith("data:"):
                                filtered_content.append(block)
                message["content"] = filtered_content


def remote_pdf_to_b64_images(url: str):
    """
    Скачивает PDF по URL и преобразовывает MarkDown с base64 картинками
    """
    r = requests.get(url)
    data = r.content
    doc = pymupdf.Document(stream=data)
    ret = []
    for page in doc:
        pix = page.get_pixmap()  # рендер страницы
        ret.append(
            f"data:image/png;base64,{base64.b64encode(
                pix.tobytes(output="png")).decode("utf-8")}")
    return ret


def image_url_to_base64(url: str, imghdr_to_mime: dict) -> str:
    """
    Скачивает PNG по URL и преобразовывает в base64 картинку
    
    Args:
        url: URL изображения
        imghdr_to_mime: Словарь для преобразования типов изображений в MIME
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/130.0.0.0 Safari/537.36"
        )
    }

    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    data = response.content

    # Определяем тип изображения по байтам
    img_type = imghdr.what(None, data)
    if img_type is None:
        raise ValueError("Could not detect image type")

    # Получаем MIME
    mime = imghdr_to_mime.get(img_type)
    if not mime:
        raise ValueError(f"Unsupported image type: {img_type}")

    # Кодируем base64
    b64 = base64.b64encode(data).decode("utf-8")

    # Возвращает действительный URL-адрес по спецификациям OpenAI.
    return f"data:{mime};base64,{b64}"


def build_user_message(text_message: str, md_docs: dict,
                       images_urls, imghdr_to_mime: dict):
    """Собирает сообщение пользователя для мультимодели.

    Правила:
    - Если md_docs пуст и images_urls is None -> вернуть простой вариант
      {"role": "user", "content": text_message}
    - Если есть изображения, они добавляются в конец content как image_url
    - Если есть md_docs: каждую md строку помещать в text, но если
      встречается встроенная base64 картинка (data:image/...), то текущий
      text закрывается, добавляется image_url c image_url=data:..., и
      затем начинается новый text.
    
    Args:
        text_message: Текстовое сообщение пользователя
        md_docs: Словарь документов в markdown
        images_urls: Список URL изображений
        imghdr_to_mime: Словарь для преобразования типов изображений в MIME
    """
    is_there_images = False
    # Базовый случай без md и изображений
    if (not md_docs) and (images_urls is None):
        return {"role": "user", "content": text_message}, is_there_images

    content = []

    # Обработка markdown документов
    if md_docs:
        img_pattern = re.compile(r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+")
        for filename, md in md_docs.items():
            if not md:
                continue
            if not content:
                content.append({
                    "type": "text",
                    "text": "# FILES ADDED FOR CONTEXT"})
            # Гарантируем, что у нас есть текущий блок текста
            current_text = f'# FILE "{filename}" BEGIN\n'
            last_end = 0
            for m in img_pattern.finditer(md):
                is_there_images = True
                # Текст до изображения
                text_before = md[last_end:m.start()]
                if text_before:
                    current_text += text_before
                # Если накоплен текст — сохранить блок
                if current_text:
                    content.append({"type": "text", "text": current_text})
                    current_text = ""
                # Сохранить изображение отдельным блоком
                data_url = m.group(0)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": data_url}})
                last_end = m.end()
            # Хвостовой текст после последнего изображения
            tail = md[last_end:]
            if tail:
                current_text += tail
            current_text += f'\n# FILE "{filename}" END'
            content.append({"type": "text", "text": current_text})

    # Добавление внешних изображений в конце
    if images_urls:
        for url in images_urls:
            if url:
                is_there_images = True
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": image_url_to_base64(url, imghdr_to_mime)}})

    # Начальный текст пользователя как text, если он есть
    if text_message:
        content.append({"type": "text", "text": text_message})

    return {"role": "user", "content": content}, is_there_images


def sanitize_bucket_name(bucket_name):
    """Sanitize bucket name to comply with S3 naming rules."""
    # Replace underscores with hyphens
    sanitized = bucket_name.replace('_', '-')
    
    # Ensure it starts and ends with alphanumeric character
    sanitized = sanitized.strip('.-')
    
    # Remove any invalid characters (only allow lowercase alphanumeric and hyphens)
    sanitized = re.sub(r'[^a-z0-9\-]', '', sanitized.lower())
    
    # Ensure length is between 3 and 63 characters
    if len(sanitized) > 63:
        sanitized = sanitized[:63]
    elif len(sanitized) < 3:
        # If too short, pad with default name
        sanitized = sanitized.ljust(3, 'x')
    
    return sanitized


def proc_ref_list(text):
    result = {}
    for match in re.findall(r'- \[(\d+)\] (.+)', text):
        num, txt = match
        result[num] = txt.strip()
    return result


def engineer(query: str, base_url: str):
    """Выполняет инженерный запрос к RAG-системе и возвращает обработанный результат.

    Args:
        query: Текстовый запрос для поиска информации
        base_url: Базовый URL RAG-сервиса (например, 'http://localhost:9621')

    Returns:
        Кортеж (response_text, references), где:
        - response_text: Текст ответа из поля "response"
        - references: Список словарей с полями "title" и "url" для ссылок
    """
    # Постоянный user_prompt, не зависящий от аргументов
    USER_PROMPT = "Необходимо извлечь все данные по марке стали для выбора химического состава сплава и технологии обработки в зависимости от запроса пользователя"

    # Формируем URL для запроса
    url = f"{base_url}/query"

    # Формируем payload для POST-запроса
    payload = {
        "query": query,
        "mode": "mix",
        "top_k": 40,
        "chunk_top_k": 20,
        "max_entity_tokens": 6000,
        "max_relation_tokens": 8000,
        "max_total_tokens": 30000,
        "user_prompt": USER_PROMPT,
        "enable_rerank": False,
        "include_references": True,
        "include_chunk_content": False,
        "stream": False
    }

    # Заголовки запроса
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }
    logger.info("Called engineer: %s", query)

    try:
        # Выполняем POST-запрос
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # Вызовет исключение для HTTP ошибок

        # Парсим JSON-ответ
        data = response.json()

        # Извлекаем текст ответа
        response_text = data.get("response", "") or ""
        # В ответе может не быть разделителя "### References" — тогда работаем с
        # полной строкой без падения по IndexError
        if "### References" in response_text:
            _, references_block = response_text.split("### References", 1)
        else:
            references_block = response_text
        check = proc_ref_list(references_block)

        # Обрабатываем ссылки
        references = []
        for ref in data.get("references", []):
            # Извлекаем file_path и преобразуем в URL
            file_path = ref.get("file_path", "")
            # Удаляем все символы до "hierarchy_trailing_20260126_182731"
            if "hierarchy_trailing_20260126_182731" in file_path:
                url_path = file_path.split("hierarchy_trailing_20260126_182731", 1)[1]
                url_path = "hierarchy_trailing_20260126_182731" + url_path
            else:
                url_path = file_path  # Если паттерн не найден, используем как есть

            title = ref.get("reference_id", "")
            if title in check or not check:
                references.append({
                    "title": title,
                    "url": url_path
                })

        return references

    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка при выполнении запроса к RAG-сервису: {e}")
        # Возвращаем пустые значения в случае ошибки
        return []


def process_engineer_url(url):
    """
    Process engineer URL to extract bucket name and file name,
    then generate presigned download URL with .pdf extension.

    Args:
        url: Original URL in format "bucket_name/path/to/file.ext"

    Returns:
        Processed URL with presigned download link and .pdf extension
    """
    if not url or "/" not in url:
        return url

    # Split URL into bucket_name and file_path
    bucket_name, file_path = url.split("/", 1)

    bucket_name = sanitize_bucket_name(bucket_name)

    # Replace any file extension with .pdf
    if "." in file_path:
        file_path = file_path.rsplit(".", 1)[0] + ".pdf"

    # Generate presigned download URL
    presigned_url = generate_presigned_download_url(bucket_name, file_path)

    return presigned_url if presigned_url else url, file_path


def websearch(query: str, tavily_base_url: str):
    """Выполняет веб-поиск по запросу и возвращает список результатов.
    
    Args:
        query: Поисковый запрос
        tavily_base_url: Базовый URL сервиса Tavily
    
    Returns:
        Результаты поиска (список или строка с ошибкой)
    """
    results = []
    try:
        client = TavilyClient(
            api_key="meow",
            api_base_url=tavily_base_url
        )
        response = client.search(
            query=query,
            max_results=5,
            include_raw_content=True,
            include_images=False,
            include_favicon=False
        )
        results = response["results"]
    except Exception as e:
        results = f"An error occurred during search request: {e}"
    return results


def detect_file_format(file_data: bytes) -> str:
    """Определяет формат файла по его содержимому (магические числа).
    
    Args:
        file_data: Бинарные данные файла
    
    Returns:
        Строка с расширением файла или пустая строка если не определено
    """
    # Проверяем магические числа файлов
    if file_data.startswith(b'%PDF'):
        return 'pdf'
    elif file_data.startswith(b'PK\x03\x04'):
        # ZIP архив - может быть docx, pptx, xlsx
        if b'word/' in file_data[:8192]:
            return 'docx'
        elif b'ppt/' in file_data[:8192]:
            return 'pptx'
        elif b'xl/' in file_data[:8192]:
            return 'xlsx'
        else:
            # Неизвестный ZIP, вернем пустую строку
            return ''
    elif file_data.startswith(b'\xd0\xcf\x11\xe0'):
        # OLE2 формат - может быть DOC, XLS и т.д.
        # Проверяем является ли это Word документом
        if b'Word.Document' in file_data[:8192] or b'Microsoft Word' in file_data[:8192]:
            return 'doc'
        else:
            # Неизвестный OLE2 формат
            return ''
    elif file_data.startswith(b'<!DOCTYPE') or file_data.startswith(b'<html'):
        return 'html'
    
    # Проверяем изображения через imghdr
    img_type = imghdr.what(None, file_data)
    if img_type:
        return 'image'
    
    # Проверяем другие форматы
    if file_data.startswith(b'# ') or file_data.startswith(b'---'):
        # Возможно Markdown или YAML
        return 'md'
    
    return ''


def docx_to_markdown_via_markitdown(file_data: bytes, file_extension: str = 'docx') -> str:
    """Конвертирует документ (DOCX/DOC) в markdown с помощью markitdown.
    
    Args:
        file_data: Бинарные данные файла
        file_extension: Расширение файла ('docx' или 'doc')
    
    Returns:
        Markdown контент документа
    """
    try:
        # Если это DOC файл, сначала конвертируем в DOCX
        if file_extension == 'doc':
            file_data = convert_doc_to_docx(file_data)
            file_extension = 'docx'
        
        # Создаём временный файл с правильным расширением
        with tempfile.NamedTemporaryFile(suffix=f'.{file_extension}', delete=False) as tmp_file:
            tmp_file.write(file_data)
            tmp_path = tmp_file.name
        
        try:
            # Используем markitdown для конвертации
            md = MarkItDown()
            result = md.convert(tmp_path)
            return result.text_content
        finally:
            # Удаляем временный файл
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    except Exception as e:
        logger.error("Failed to convert document to markdown using markitdown: %s", e)
        raise


def convert_doc_to_docx(doc_data: bytes) -> bytes:
    """Конвертирует DOC файл в DOCX используя доступные инструменты.
    
    Пробует в порядке: unoconv, LibreOffice, python-docx.
    
    Args:
        doc_data: Бинарные данные DOC файла
    
    Returns:
        Бинарные данные DOCX файла
    """
    errors = []
    
    # Попытка 1: unoconv (самый легкий для сервера)
    try:
        return _convert_doc_to_docx_unoconv(doc_data)
    except Exception as e:
        logger.warning("unoconv conversion failed: %s", e)
        errors.append(f"unoconv: {e}")
    
    # Попытка 2: LibreOffice
    try:
        return _convert_doc_to_docx_libreoffice(doc_data)
    except Exception as e:
        logger.warning("LibreOffice conversion failed: %s", e)
        errors.append(f"LibreOffice: {e}")
    
    # Попытка 3: python-docx (если файл на самом деле DOCX с расширением .doc)
    if HAS_PYTHON_DOCX:
        try:
            return _convert_doc_to_docx_python_docx(doc_data)
        except Exception as e:
            logger.warning("python-docx conversion failed: %s", e)
            errors.append(f"python-docx: {e}")
    
    # Все методы failed
    error_msg = (
        "Could not convert DOC to DOCX. "
        "Please convert your DOC file to DOCX format using Microsoft Word or LibreOffice, "
        "then upload the DOCX file. "
        f"Errors: {'; '.join(errors)}"
    )
    logger.error(error_msg)
    raise RuntimeError(error_msg)


def _convert_doc_to_docx_unoconv(doc_data: bytes) -> bytes:
    """Конвертирует DOC в DOCX используя unoconv (универсальный конвертор для LibreOffice).
    
    Args:
        doc_data: Бинарные данные DOC файла
    
    Returns:
        Бинарные данные DOCX файла
    """
    try:
        # Создаём уникальную временную директорию для конвертации
        tmp_work_dir = tempfile.mkdtemp(prefix='doc2docx_')
        logger.debug("Created work directory for unoconv: %s", tmp_work_dir)
        
        try:
            # Создаём входной файл
            tmp_input_path = os.path.join(tmp_work_dir, 'input.doc')
            with open(tmp_input_path, 'wb') as f:
                f.write(doc_data)
            logger.debug("Wrote input file: %s (%d bytes)", tmp_input_path, len(doc_data))
            
            # Выходной файл
            tmp_output_path = os.path.join(tmp_work_dir, 'input.docx')
            
            # Запускаем unoconv: doc -> docx
            cmd = ['unoconv', '-f', 'docx', '-o', tmp_output_path, tmp_input_path]
            
            logger.info("Converting DOC to DOCX using unoconv...")
            logger.debug("Command: %s", ' '.join(cmd))
            result = subprocess.run(cmd, capture_output=True, timeout=30, text=True)
            
            logger.debug("unoconv return code: %d", result.returncode)
            if result.stdout:
                logger.debug("unoconv stdout: %s", result.stdout)
            if result.stderr:
                logger.debug("unoconv stderr: %s", result.stderr)
            
            if result.returncode != 0:
                raise RuntimeError(f"unoconv conversion failed with code {result.returncode}: {result.stderr}")
            
            if not os.path.exists(tmp_output_path):
                raise FileNotFoundError(f"unoconv did not create output file: {tmp_output_path}")
            
            # Читаем результат
            with open(tmp_output_path, 'rb') as f:
                docx_data = f.read()
            
            logger.debug("Successfully converted DOC to DOCX using unoconv (%d bytes)", len(docx_data))
            return docx_data
        finally:
            # Удаляем временную директорию
            import shutil
            if os.path.exists(tmp_work_dir):
                shutil.rmtree(tmp_work_dir, ignore_errors=True)
                logger.debug("Cleaned up work directory: %s", tmp_work_dir)
    except Exception as e:
        logger.error("Failed to convert DOC to DOCX using unoconv: %s", e)
        raise


def _convert_doc_to_docx_libreoffice(doc_data: bytes) -> bytes:
    """Конвертирует DOC в DOCX используя LibreOffice (headless).
    
    Args:
        doc_data: Бинарные данные DOC файла
    
    Returns:
        Бинарные данные DOCX файла
    """
    try:
        # Создаём уникальную временную директорию для этой конвертации
        tmp_work_dir = tempfile.mkdtemp(prefix='doc2docx_')
        logger.debug("Created work directory: %s", tmp_work_dir)
        
        try:
            # Создаём входной файл с фиксированным именем в работочной директории
            tmp_input_path = os.path.join(tmp_work_dir, 'input.doc')
            with open(tmp_input_path, 'wb') as tmp_input:
                tmp_input.write(doc_data)
            logger.debug("Wrote input file: %s (%d bytes)", tmp_input_path, len(doc_data))
            
            # Ожидаемый путь выходного файла
            tmp_output_path = os.path.join(tmp_work_dir, 'input.docx')
            
            # Пробуем разные команды запуска LibreOffice
            commands = [
                ['libreoffice', '--headless', '--norestore', '--convert-to', 'docx', '--outdir', tmp_work_dir, tmp_input_path],
                ['soffice', '--headless', '--norestore', '--convert-to', 'docx', '--outdir', tmp_work_dir, tmp_input_path],
            ]
            
            last_error = None
            for cmd in commands:
                try:
                    logger.debug("Trying command: %s", ' '.join(cmd))
                    result = subprocess.run(cmd, capture_output=True, timeout=30, text=True)
                    
                    logger.debug("Return code: %d", result.returncode)
                    if result.stdout:
                        logger.debug("stdout: %s", result.stdout)
                    if result.stderr:
                        logger.debug("stderr: %s", result.stderr)
                    
                    # Логируем содержимое рабочей директории
                    files_in_dir = os.listdir(tmp_work_dir)
                    logger.debug("Files in work directory: %s", files_in_dir)
                    
                    if result.returncode == 0 and os.path.exists(tmp_output_path):
                        # Успешно!
                        with open(tmp_output_path, 'rb') as f:
                            docx_data = f.read()
                        logger.debug("Successfully converted DOC to DOCX (%d bytes)", len(docx_data))
                        return docx_data
                    elif result.returncode == 0:
                        # Проверяем, создан ли DOCX файл с другим именем
                        docx_files = [f for f in files_in_dir if f.endswith('.docx')]
                        if docx_files:
                            tmp_output_path = os.path.join(tmp_work_dir, docx_files[0])
                            with open(tmp_output_path, 'rb') as f:
                                docx_data = f.read()
                            logger.debug("Successfully converted DOC to DOCX (%d bytes)", len(docx_data))
                            return docx_data
                    
                    last_error = f"Return code: {result.returncode}, Files: {files_in_dir}"
                except FileNotFoundError:
                    last_error = f"Command not found: {cmd[0]}"
                    continue
                except Exception as e:
                    last_error = str(e)
                    continue
            
            # Если мы здесь, то ни одна команда не сработала
            raise RuntimeError(f"LibreOffice conversion failed. Last error: {last_error}")
        finally:
            # Удаляем всю работочную директорию и её содержимое
            import shutil
            if os.path.exists(tmp_work_dir):
                shutil.rmtree(tmp_work_dir, ignore_errors=True)
                logger.debug("Cleaned up work directory: %s", tmp_work_dir)
    except Exception as e:
        logger.debug("LibreOffice conversion failed: %s", e)
        raise


def _convert_doc_to_docx_python_docx(doc_data: bytes) -> bytes:
    """Конвертирует DOC в DOCX используя python-docx (для простых DOC файлов).
    
    Примечание: Это работает только для простых DOC файлов. Для сложных форматирований
    используйте LibreOffice.
    
    Args:
        doc_data: Бинарные данные DOC файла
    
    Returns:
        Бинарные данные DOCX файла
    """
    if not HAS_PYTHON_DOCX:
        raise ImportError("python-docx is not installed. Install it with: pip install python-docx")
    
    try:
        import io
        logger.info("Converting DOC to DOCX using python-docx...")
        
        # python-docx может открывать DOCX, но не DOC напрямую
        # Однако, некоторые .doc файлы на самом деле DOCX в старом имени
        # Попробуем открыть как DOCX в памяти
        try:
            doc = Document(io.BytesIO(doc_data))
            
            # Сохраняем в DOCX формат
            output = io.BytesIO()
            doc.save(output)
            docx_data = output.getvalue()
            
            logger.debug("Successfully converted DOC to DOCX using python-docx (%d bytes)", len(docx_data))
            return docx_data
        except Exception as e:
            raise RuntimeError(f"python-docx failed to process file: {e}")
    except Exception as e:
        logger.error("Failed to convert DOC to DOCX using python-docx: %s", e)
        raise


async def convert_to_md_async(url: str, docling_address: str):
    """Асинхронно конвертирует документ в markdown через docling API.
    
    Поддерживаемые форматы: docx, pptx, html, image, pdf, asciidoc, md, xlsx
    
    Args:
        url: URL документа для конвертации
        docling_address: Адрес docling сервиса
    
    Returns:
        Кортеж (filename, md_content) с именем файла и markdown контентом
        или (None, None) в случае ошибки.
    
    Raises:
        ValueError: Если формат файла не поддерживается.
    """
    if not docling_address:
        logger.error("DOCLING_ADDRESS is not set")
        return None, None

    # Список поддерживаемых форматов
    from_formats = ["docx", "pptx", "html", "image",
                    "pdf", "asciidoc", "md", "csv",
                    "xlsx", "xml_uspto", "xml_jats",
                    "mets_gbs", "json_docling", "audio", "vtt"]

    filename = ""
    file_extension = ""

    try:
        # Сначала пытаемся получить расширение из URL
        parsed_url = urllib.parse.urlparse(url)
        url_path = parsed_url.path
        if "." in url_path:
            _, url_extension = urllib.parse.unquote(url_path).rsplit(".", 1)
            file_extension = url_extension.lower()

        logger.info("File extension from URL: %s",
                    file_extension if file_extension else "not found")

        # Скачиваем файл, чтобы узнать его размер и определить тип
        async_client = httpx.AsyncClient(timeout=60.0)

        try:
            get_response = await async_client.get(url)
            file_data = get_response.content
            original_size = len(file_data)
        except Exception as e:
            logger.error("Failed to download file: %s", e)
            raise

        logger.info("original_size=%s", original_size)

        # Если расширение не найдено в URL, определяем по содержимому файла
        if not file_extension:
            detected_extension = detect_file_format(file_data)
            if detected_extension:
                file_extension = detected_extension
                logger.info("Detected file extension from content: %s",
                             file_extension)

        # Проверяем, поддерживается ли формат
        if not file_extension or file_extension not in from_formats:
            error_msg = f"Unsupported file format: '{file_extension}'. Supported formats: {', '.join(from_formats)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Теперь кидаем запрос в API Docling.
        docling_url = f"http://{docling_address}/v1/convert/source"
        payload = {
            "options": {
                "from_formats": from_formats,
                "to_formats": ["md"],
                "image_export_mode": "embedded",
                "do_ocr": False,
                "abort_on_error": False,
            },
            "sources": [{
                "kind": "http",
                "url": url #.replace('localhost', 'minio-2')
            }]
        }
        response = await async_client.post(docling_url, json=payload)
        data = response.json()
        filename = urllib.parse.unquote(
            data.get("document", {"filename":""}).get("filename"))
        md_content = data.get("document", {}).get("md_content")
        # Считаем размер md_content в байтах
        assert(md_content is not None), "No md_content in docling response"
        md_size = len(md_content.encode('utf-8'))
        logger.info("md_size=%s", md_size)
        if original_size > 3.2 * md_size:
            raise AssertionError(
                f"Original file size ({original_size} bytes) is more than "
                f"3.2 times greater than markdown size ({md_size} bytes).")
        return filename, md_content
    except Exception as e:
        logger.error("Converting document to md failed: %s", e)
        logger.info("Falling back to dumb markdown... Format: %s", file_extension)
        
        try:
            if file_extension == 'pdf':
                # Для PDF используем конвертацию в изображения
                return filename, "\n".join(
                    [f"## PAGE {i}\n\n![page {i}]({u})\n\n"
                     for i, u
                     in enumerate(remote_pdf_to_b64_images(url))])
            
            elif file_extension in ('docx', 'doc'):
                # Для DOCX и DOC используем markitdown
                # Если у нас уже есть данные файла в переменной file_data (из текущего блока try)
                # То используем их, иначе скачиваем заново
                try:
                    md_content = docx_to_markdown_via_markitdown(file_data, file_extension)
                except NameError:
                    # file_data не определён, скачиваем файл заново
                    async_client = httpx.AsyncClient(timeout=60.0)
                    get_response = await async_client.get(url)
                    file_data = get_response.content
                    md_content = docx_to_markdown_via_markitdown(file_data, file_extension)
                return filename, md_content
            
            else:
                # Для других форматов выбрасываем исключение
                error_msg = f"Fallback conversion not supported for format: {file_extension}"
                logger.error(error_msg)
                raise ValueError(error_msg)
        
        except Exception as e2:
            logger.error("Converting document to dumb markdown failed: %s", e2)
            return None, None


async def check_docling_health(docling_address: str):
    """Проверяет доступность docling API.
    
    Args:
        docling_address: Адрес docling сервиса
    
    Returns:
        bool: True если API доступен и отвечает корректно, False иначе.
    """
    logger.info("Docling check on: %s", docling_address)
    if not docling_address:
        logger.error("DOCLING_ADDRESS is not set")
        return False

    try:
        # Используем стандартный health check endpoint
        health_url = f"http://{docling_address}/health"
        async_client = httpx.AsyncClient(timeout=10.0)

        # Отправляем GET запрос на health endpoint
        response = await async_client.get(health_url)

        # Проверяем, что сервер отвечает с успешным статусом
        if response.status_code == 200:
            # Проверка содержимого ответа, если API возвращает JSON с статусом
            try:
                data = response.json()
                if isinstance(data, dict) and data.get("status") == "ok":
                    logger.info("Docling health check: OK (status: %s)",
                                data.get('status'))
                else:
                    logger.info("Docling health check: OK (status code: %s)",
                                response.status_code)
            except (json.JSONDecodeError, ValueError):
                # Если ответ не JSON, просто проверяем статус код
                logger.info("Docling health check: OK (status code: %s)",
                            response.status_code)
            return True
        else:
            logger.error("Docling health check: FAILED (status code: %s)",
                         response.status_code)
            return False
    except httpx.ConnectError:
        logger.error("Cannot connect to docling API")
        return False
    except httpx.TimeoutException:
        logger.error("Docling API timeout")
        return False
    except Exception as e:
        logger.error("Checking docling health failed: %s", e)
        return False


def convert_to_md(url: str, docling_address: str):
    """Синхронная обёртка для конвертации документа в markdown.
    
    Args:
        url: URL документа для конвертации
        docling_address: Адрес docling сервиса
    
    Returns:
        Кортеж (filename, md_content) с именем файла и markdown контентом
        или (None, None) в случае ошибки.
    """
    try:
        eloop = asyncio.new_event_loop()
        asyncio.set_event_loop(eloop)
        result = eloop.run_until_complete(
            convert_to_md_async(url, docling_address))
        eloop.close()
        return result
    except Exception as e:
        logger.error("Convert to markdown wrapper failed: %s", e)
        return None, None
