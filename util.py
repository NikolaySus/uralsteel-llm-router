"""
Вспомогательные функции для обработки документов, изображений и веб-поиска.
"""

import re
import base64
import urllib
import imghdr
import asyncio
import json

import requests
import httpx
import pymupdf
from tavily import TavilyClient

from logger import logger


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


async def convert_to_md_async(url: str, docling_address: str):
    """Асинхронно конвертирует документ в markdown через docling API.
    
    Поддерживаемые форматы: docx, pptx, html, image, pdf, asciidoc, md, xlsx
    
    Args:
        url: URL документа для конвертации
        docling_address: Адрес docling сервиса
    
    Returns:
        Кортеж (filename, md_content) с именем файла и markdown контентом
        или (None, None) в случае ошибки.
    """
    if not docling_address:
        logger.error("DOCLING_ADDRESS is not set")
        return None, None

    filename = ""

    try:
        # Сначала качаем файл, чтобы узнать его размер в байтах
        async_client = httpx.AsyncClient(timeout=60.0)

        # Чтобы узнать размер файла, сначала траим HEAD-запрос (эффективнее)
        original_size = 0
        try:
            head_response = await async_client.head(url)
            original_size = int(head_response.headers['content-length'])
        except:
            # Если HEAD не удался, то GET-запрос, но читаем только заголовки
            get_response = await async_client.get(url)
            original_size = len(get_response.content)
        logger.debug("original_size=%s", original_size)

        # Теперь кидаем запрос в API Docling.
        docling_url = f"http://{docling_address}/v1/convert/source"
        payload = {
            "options": {
                "from_formats": ["docx", "pptx", "html", "image", "pdf",
                                "asciidoc", "md", "xlsx"],
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
        if md_content:
            md_size = len(md_content.encode('utf-8'))
            logger.debug("md_size=%s", md_size)
            if original_size > 3 * md_size:
                raise AssertionError(
                    f"Original file size ({original_size} bytes) is more than "
                    f"3 times greater than markdown size ({md_size} bytes).")
        return filename, md_content
    except Exception as e:
        logger.error("Converting document to md failed: %s", e)
        logger.info("Falling back to dumb markdown...")
        try:
            return filename, "\n".join(
                [f"## PAGE {i}\n\n![page {i}]({u})\n\n"
                 for i, u
                 in enumerate(remote_pdf_to_b64_images(url))])
        except Exception as e2:
            logger.error("Converting document to dumb markdown failed: %s", e2)


async def check_docling_health(docling_address: str):
    """Проверяет доступность docling API.
    
    Args:
        docling_address: Адрес docling сервиса
    
    Returns:
        bool: True если API доступен и отвечает корректно, False иначе.
    """
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
