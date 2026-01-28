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

from logger import logger


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
    """Конвертирует DOC файл в DOCX используя LibreOffice.
    
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
            
            # Используем LibreOffice для конвертации
            cmd = [
                'libreoffice',
                '--headless',
                '--norestore',
                '--convert-to', 'docx',
                '--outdir', tmp_work_dir,
                tmp_input_path
            ]
            
            logger.info("Converting DOC to DOCX using LibreOffice...")
            logger.debug("Command: %s", ' '.join(cmd))
            result = subprocess.run(cmd, capture_output=True, timeout=30, text=True)
            
            logger.debug("LibreOffice return code: %d", result.returncode)
            if result.stdout:
                logger.debug("LibreOffice stdout: %s", result.stdout)
            if result.stderr:
                logger.debug("LibreOffice stderr: %s", result.stderr)
            
            # Логируем содержимое рабочей директории
            try:
                files_in_dir = os.listdir(tmp_work_dir)
                logger.debug("Files in work directory: %s", files_in_dir)
            except Exception as e:
                logger.error("Could not list work directory: %s", e)
            
            if result.returncode != 0:
                raise RuntimeError(f"LibreOffice conversion failed with code {result.returncode}: {result.stderr}")
            
            # Проверяем, создан ли выходной файл
            if not os.path.exists(tmp_output_path):
                # Если нет, ищем любой DOCX файл в директории
                docx_files = [f for f in os.listdir(tmp_work_dir) if f.endswith('.docx')]
                if not docx_files:
                    raise FileNotFoundError(
                        f"No DOCX file created by LibreOffice in {tmp_work_dir}. "
                        f"Files found: {os.listdir(tmp_work_dir)}")
                tmp_output_path = os.path.join(tmp_work_dir, docx_files[0])
                logger.debug("Found converted file: %s", tmp_output_path)
            
            # Читаем конвертированный файл
            with open(tmp_output_path, 'rb') as f:
                docx_data = f.read()
            
            logger.debug("Successfully converted DOC to DOCX (%d bytes)", len(docx_data))
            return docx_data
        finally:
            # Удаляем всю работочную директорию и её содержимое
            import shutil
            if os.path.exists(tmp_work_dir):
                shutil.rmtree(tmp_work_dir, ignore_errors=True)
                logger.debug("Cleaned up work directory: %s", tmp_work_dir)
    except Exception as e:
        logger.error("Failed to convert DOC to DOCX: %s", e)
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
                    "pdf", "asciidoc", "md", "xlsx", "doc"]

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
