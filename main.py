r"""
gRPC сервер для взаимодействия с языковой моделью.
Команда для генерации llm_pb2.py и llm_pb2_grpc.py из llm.proto:
uv run -m grpc_tools.protoc -I.\uralsteel-grpc-api\llm\ --python_out=.
--grpc_python_out=. llm.proto
"""

from datetime import datetime
from concurrent import futures
from io import BytesIO
import json
import logging
import os
import re
import subprocess
import urllib
import base64
import uuid
import asyncio
import imghdr

import requests
from google.protobuf import empty_pb2
import grpc
import httpx
from openai import OpenAI
from tavily import TavilyClient
from minio import Minio
from minio.error import S3Error

import llm_pb2
import llm_pb2_grpc


# Словил кринж с systemd...
# Фикс для подстановки переменных окружения в другие переменные окружения
for key, value in os.environ.items():
    if "${" in value:
        os.environ[key] = os.path.expandvars(value)

# Переменные окружения вида CONST_api_case
# Пока CONST + YANDEXAI_/OPENAI_ + BASE_URL/FOLDER/KEY/MODEL/PRICES_URL
CONST = "INFERENCE_API_"
CONST_LEN = len(CONST)
ALL_API_VARS = dict()
for name, value in os.environ.items():
    if name.startswith(CONST):
        api_and_case = name[CONST_LEN:].lower()
        delim = api_and_case.find("_")
        api = api_and_case[:delim]
        case = api_and_case[delim + 1:]
        ALL_API_VARS.setdefault(api, dict())
        ALL_API_VARS[api][case] = value
# Секретный ключ для клиентского доступа к gRPC методам
SECRET_KEY = os.environ.get('SECRET_KEY', '')
# Формат даты и времени
DATETIME_FORMAT = os.environ.get('DATETIME_FORMAT', '%Y-%m-%dT%H:%M:%S')
# Путь к конфигурационному файлу
CONFIG_PATH = "config.json"
# Базовый URL Tavily сервиса
TAVILY_BASE_URL = os.environ.get('TAVILY_BASE_URL', 'http://localhost:8000')
# Максимальное количество результатов веб-поиска
MAX_RESULTS = int(os.environ.get('MAX_RESULTS', '5'))
# Когда генерировать конфигурационный файл (ресурсоёмкая операция)
GENERATE_CONFIG_WHEN = os.environ.get('GENERATE_CONFIG_WHEN', 'missing')
# docling-serve
DOCLING_ADDRESS = os.environ.get('DOCLING_ADDRESS', '')
# -
CHUNK_SIZE = int(os.environ.get('CHUNK_SIZE', 8192))
# Регулярные выражения для фильтрации моделей
WHITELIST_REGEX_TEXT2TEXT  = os.environ.get('WHITELIST_REGEX_TEXT2TEXT', '.*')
BLACKLIST_REGEX_TEXT2TEXT  = os.environ.get('BLACKLIST_REGEX_TEXT2TEXT', '$^')
WHITELIST_REGEX_SPEECH2TEXT= os.environ.get('WHITELIST_REGEX_SPEECH2TEXT', '.*')
BLACKLIST_REGEX_SPEECH2TEXT= os.environ.get('BLACKLIST_REGEX_SPEECH2TEXT', '$^')
# S3
MINIO_ADDRESS = os.environ.get('MINIO_ADDRESS', '')
BUCKET_NAME = os.environ.get('BUCKET_NAME', 'cache')
MINIO_ACCESS_KEY = os.environ.get('MINIO_ACCESS_KEY', None)
MINIO_SECRET_KEY = os.environ.get('MINIO_SECRET_KEY', None)
# Инструменты для моделей с поддержкой функций
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "websearch",
            "description": "Get websearch results by search query.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string"
                    }
                },
                "required": ["query"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "image_gen",
            "description": "Generate an image by query and get it's url.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string"
                    }
                },
                "required": ["query"],
                "additionalProperties": False
            }
        }
    }
]
# Имена инструментов
TOOLS_NAMES = [
    tool["function"]["name"] for tool in TOOLS
]
# url -> base64
IMGHDR_TO_MIME = {
    "jpeg": "image/jpeg",
    "png" : "image/png",
    "gif" : "image/gif",
    "bmp" : "image/bmp",
    "tiff": "image/tiff",
    "rgb" : "image/x-rgb",
    "webp": "image/webp",
    "pbm" : "image/x-portable-bitmap",
    "pgm" : "image/x-portable-graymap",
    "ppm" : "image/x-portable-pixmap",
    "rast":	"image/cmu-raster",
    "xbm" :	"image/x-xbitmap"
}


class AuthInterceptor(grpc.ServerInterceptor):
    """
    gRPC Interceptor для проверки авторизации на уровне протокола.
    
    Преимущества использования Interceptor вместо проверок в методах:
    1. Проверка выполняется ДО вызова бизнес-логики (минимум затрат CPU)
    2. Неавторизованные запросы отклоняются на уровне сети
    3. Нет дублирования кода в каждом методе
    4. Легко добавлять публичные методы без авторизации
    
    Методы без авторизации (public):
    - Ping: используется для health-check'ов, должен быть всегда доступен
    
    Методы требующие авторизации (protected):
    - NewMessage: взаимодействие с LLM, критичный ресурс
    - AvailableModelsText2Text: получение информации о моделях
    - AvailableModelsSpeech2Text: получение информации о моделях
    """

    # Методы, которые НЕ требуют авторизацию (public)
    PUBLIC_METHODS = {
        '/llm.Llm/Ping',
    }

    def intercept_service(self, continuation, handler_call_details):
        """
        Перехватывает каждый вызов RPC метода.
        
        handler_call_details.invocation_metadata содержит метаданные запроса,
        включая авторизационные заголовки.
        
        Метод должен вернуть либо обработанный ответ, либо вызвать
        continuation() для передачи запроса дальше.
        """
        method_name = handler_call_details.method

        # Если метод в списке публичных - пропускаем проверку авторизации
        if method_name in self.PUBLIC_METHODS:
            return continuation(handler_call_details)

        # Проверяем наличие авторизационных метаданных
        metadata = dict(handler_call_details.invocation_metadata or [])
        authorization = metadata.get('authorization', '')

        # Ожидаем формат "Bearer <SECRET_KEY>" или просто секретный ключ
        secret_from_header = authorization.replace('Bearer ', '').strip()

        # Проверка секретного ключа
        if not SECRET_KEY:
            # Если SECRET_KEY не задан в env - логируем предупреждение
            # но НЕ отклоняем запрос (для совместимости с dev средой)
            print(f"SECRET_KEY not set, auth skip for method {method_name}")
            return continuation(handler_call_details)

        if secret_from_header != SECRET_KEY:
            # Секретный ключ неверен - отклоняем запрос
            print(f"UNAUTHORIZED: bad SECRET_KEY for method {method_name}")
            # Возвращаем обработчик, который всегда абортит вызов
            def abort_unary_unary(request, context):
                context.abort(grpc.StatusCode.UNAUTHENTICATED,
                              "Invalid or missing authorization")

            def abort_unary_stream(request, context):
                context.abort(grpc.StatusCode.UNAUTHENTICATED,
                              "Invalid or missing authorization")
                if False:
                    yield  # pragma: no cover

            return grpc.RpcMethodHandler(
                request_streaming=False,
                response_streaming=False,
                unary_unary=abort_unary_unary,
                unary_stream=None,
                stream_unary=None,
                stream_stream=None,
            )

        # Секретный ключ верен - пропускаем запрос дальше
        return continuation(handler_call_details)


def image_url_to_base64(url: str) -> str:
    # Download image
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

    # Detect image type from bytes
    img_type = imghdr.what(None, data)
    if img_type is None:
        raise ValueError("Could not detect image type")

    # Get MIME
    mime = IMGHDR_TO_MIME.get(img_type)
    if not mime:
        raise ValueError(f"Unsupported image type: {img_type}")

    # Encode base64
    b64 = base64.b64encode(data).decode("utf-8")

    # Return valid OpenAI data URL
    return f"data:{mime};base64,{b64}"


def build_user_message(text_message: str, md_docs: dict, images_urls):
    """Собирает сообщение пользователя для мультимодели.

    Правила:
    - Если md_docs пуст и images_urls is None -> вернуть простой вариант
      {"role": "user", "content": text_message}
    - Если есть изображения, они добавляются в конец content как image_url
    - Если есть md_docs: каждую md строку помещать в text, но если
      встречается встроенная base64 картинка (data:image/...), то текущий
      text закрывается, добавляется image_url c image_url=data:..., и
      затем начинается новый text.
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
                content.append({"type": "text", "text": "# FILES ADDED FOR CONTEXT"})
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
                content.append({"type": "image_url", "image_url": {"url": data_url}})
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
                content.append({"type": "image_url", "image_url": {"url": image_url_to_base64(url)}})

    # Начальный текст пользователя как text, если он есть
    if text_message:
        content.append({"type": "text", "text": text_message})

    return {"role": "user", "content": content}, is_there_images


def available_models(base_url: str,
                     api_key: str,
                     project: str,
                     whitelist_regex: str = ".*",
                     blacklist_regex: str = "$^"):
    """Возвращает список доступных моделей в виде списка строк.

    Поддерживает фильтрацию по белому и чёрному списку в виде регулярных
    выражений. Модели, подходящие под whitelist, но при этом также попадающие
    под blacklist, будут исключены.

    Параметры по умолчанию сохраняют прежнее поведение:
    - `whitelist_regex='.*'` — всё разрешено
    - `blacklist_regex='$^'` — ничего не запрещено
    """
    try:
        wl = re.compile(whitelist_regex)
        bl = re.compile(blacklist_regex)
    except re.error as e:
        raise ValueError(f"Invalid regex provided: {e}") from e
    models_list = OpenAI(
        base_url=base_url,
        api_key=api_key,
        project=project,
    ).models.list()
    check_arr = []
    for model in models_list.data:
        model_id = getattr(model, "id", "")
        if not model_id:
            continue
        if wl.search(model_id) and not bl.search(model_id):
            check_arr.append(model_id)
    return check_arr


def transcribe_audio(audio_buffer: BytesIO,
                     speech2text_override: str = None):
    """Транскрибирует собранный audio_buffer и возвращает кортеж
    (transcription_text, TranscribeResponseType proto).

    Использует speech2text_override если задан, иначе
    ALL_API_VARS["openai"]["model"].
    Бросает ValueError, если конфигурация SPEECH2TEXT не задана.
    """
    if audio_buffer.tell() == 0:
        return None, None

    audio_buffer.seek(0)
    # Некоторым API требуется имя у файла
    audio_buffer.name = "audio.mp3"

    # Используем переданную модель или берём по умолчанию
    if speech2text_override:
        model_to_use = speech2text_override
    else:
        model_to_use = ALL_API_VARS["openai"]["model"]

    transcription = OpenAI(
        base_url=ALL_API_VARS["openai"]["base_url"],
        api_key=ALL_API_VARS["openai"]["key"],
    ).audio.transcriptions.create(
        model=model_to_use,
        file=audio_buffer
    )
    text = getattr(transcription, "text", "")
    usage = getattr(transcription, "usage", None)
    duration = None
    if usage:
        duration = getattr(usage, "duration", None)
    proto = llm_pb2.TranscribeResponseType(
        transcription=text,
        duration=duration,
        expected_cost_usd=(duration * ALL_API_VARS["openai"]["price_coef"]
                           if duration else 0.0),
        datetime=datetime.now().strftime(DATETIME_FORMAT)
    )
    return text, proto


def websearch(query: str):
    """Выполняет веб-поиск по запросу и возвращает список результатов."""
    results = []
    try:
        client = TavilyClient(
            api_key="meow",
            api_base_url=TAVILY_BASE_URL
        )
        response = client.search(
            query=query,
            max_results=5,
            include_raw_content=True,
            include_images=False,
            include_favicon=False
        )
        results = response["results"]
        print(results)
    except Exception as e:
        results = f"An error occurred during search request: {e}"
    url_list = [result["url"]
                for result
                in results] if isinstance(results, list) else []
    return results, llm_pb2.ToolMetadataResponse(
        websearch=llm_pb2.ToolWebSearchMetadata(
            url_list=url_list
        )
    )


def image_gen(query: str):
    """Генерирует изображение по запросу и возвращает его URL."""
    result = None
    image_base64 = None
    try:
        client = OpenAI(
            api_key=ALL_API_VARS["openai"]["key"],
            base_url=ALL_API_VARS["openai"]["base_url"],
        )

        response = client.images.generate(
            model=ALL_API_VARS["openaiimgen"]["model"],
            prompt=query,
            size="1024x1024"
        )

        image_base64 = response.data[0].b64_json
        result = "Done!"
    except Exception as e:
        result = f"An error occurred during image gen request: {e}"
    return result, llm_pb2.ToolMetadataResponse(
        image_gen=llm_pb2.ToolImageGenMetadata(
            image_base64=image_base64 or "",
            expected_cost=ALL_API_VARS["openaiimgen"]["price_coef"]
        )
    )


def call_function(name, args):
    """Вызов функции инструмента по имени с аргументами args."""
    if name == "websearch":
        result, meta = websearch(**args)
        return json.dumps(result,
                          ensure_ascii=False), meta
    elif name == "image_gen":
        result, meta = image_gen(**args)
        return result, meta
    return json.dumps({"error": f"Unknown tool {name}"}, ensure_ascii=False)


async def convert_to_md_async(url: str):
    """Асинхронно конвертирует документ в markdown через docling API.
    
    Поддерживаемые форматы: docx, pptx, html, image, pdf, asciidoc, md, xlsx
    
    Returns:
        Кортеж (filename, md_content) с именем файла и markdown контентом
        или (None, None) в случае ошибки.
    """
    if not DOCLING_ADDRESS:
        print("ERROR: DOCLING_ADDRESS is not set")
        return None, None

    try:
        docling_url = f"http://{DOCLING_ADDRESS}/v1/convert/source"
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
                "url": url
            }]
        }
        async_client = httpx.AsyncClient(timeout=90.0)
        response = await async_client.post(docling_url, json=payload)
        data = response.json()
        filename = urllib.parse.unquote(data.get("document", {}).get("filename"))
        md_content = data.get("document", {}).get("md_content")
        return filename, md_content
    except Exception as e:
        print(f"ERROR converting document to md: {e}")
        return None, None


def convert_to_md(url: str):
    """Синхронная обёртка для конвертации документа в markdown.
    
    Args:
        url: URL документа для конвертации
    
    Returns:
        Кортеж (filename, md_content) с именем файла и markdown контентом
        или (None, None) в случае ошибки.
    """
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(convert_to_md_async(url))
        loop.close()
        return result
    except Exception as e:
        print(f"ERROR in convert_to_md wrapper: {e}")
        return None, None


def generate_chat_name(user_message: str):
    """Генерирует название чата на основе сообщения пользователя.
    
    Использует llm для создания короткого названия (до 1024 токенов).
    Возвращает строку с названием чата или None в случае ошибки.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that generates short, concise "
                "chat names based on the user's initial message. The name "
                "should be brief (4 words max), descriptive, and in the "
                "same language as the user message. Return only the name, "
                "nothing else. You must NOT answer questions, just summarize."
            )
        },
        {
            "role": "user",
            "content": user_message
        }
    ]
    response = OpenAI(
        base_url=ALL_API_VARS["yandexai"]["base_url"],
        api_key=ALL_API_VARS["yandexai"]["key"],
        project=ALL_API_VARS["yandexai"]["folder"],
    ).chat.completions.create(
        model=ALL_API_VARS["yandexaisummary"]["model"],
        messages=messages,
        max_tokens=128,
        temperature=0.9
    )
    name_c = response.choices[0].message.content.strip()
    if not hasattr(response, "usage"):
        raise ValueError("usage required")
    prompt_tokens = getattr(response.usage, "prompt_tokens", 0)
    completion_tokens = getattr(response.usage, "completion_tokens", 0)
    total_tokens = getattr(response.usage, "total_tokens", 0)
    usd = ALL_API_VARS["yandexaisummary"]["price_coef"] * total_tokens
    return llm_pb2.ChatNameResponseType(name=name_c,
                                        prompt_tokens=prompt_tokens,
                                        completion_tokens=completion_tokens,
                                        total_tokens=total_tokens,
                                        expected_cost_usd=usd)


def build_messages_from_history(history, user_message: str,
                                text2text_override: str = None):
    """Собирает список сообщений для LLM на основании history и user_message.
    
    Использует text2text_override если задан, иначе
    ALL_API_VARS["yandexai"]["model"].
    """
    if text2text_override:
        model_to_use = text2text_override
    else:
        model_to_use = ALL_API_VARS["yandexai"]["model"]
    messages = [{
        "role": "system",
        "content": (
            "### You are *helpfull* and *highly skilled* LLM-powered "
            "assistant that always follows best practices.\n"
            f"The base LLM is **{model_to_use}**. Current date and time: "
            f"**{datetime.now().strftime(DATETIME_FORMAT)}**. "
            "*Note that current date and time are relevant only for last "
            "message, previous ones could be sent a long time ago*. "
            "You can use the **websearch** tool for relevant information "
            "retrieval and **image_gen** for image generation. "
            "**Do not insert any links or images in your answers. Respond in "
            "the same language as the user using MarkDown markup language**."
        )
    }]
    if history:
        # Minio get
        minio_client = Minio(
            MINIO_ADDRESS,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=True
        )
        for message_uuid in history:
            try:
                messages.append(json.loads(minio_client.get_object(BUCKET_NAME, message_uuid).read().decode('utf-8')))
            except S3Error as e:
                print(f"Error getting object from MinIO: {e}")
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
    messages.append(user_message)
    return messages


def change_model_msgs():
    """Сообщения о необходимости смены модели."""
    return [llm_pb2.NewMessageResponse(
                generate=llm_pb2.GenerateResponseType(
                    content="Ваш запрос слишком сложный для выбранной "
                            f"модели, пожалуйста, выберите {
                                ALL_API_VARS["openaivlm"]["model"]}",
                    reasoning_content="",
                    datetime=datetime.now().strftime(DATETIME_FORMAT)
                )
            ), llm_pb2.NewMessageResponse(
                complete=llm_pb2.CompleteResponseType(
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    expected_cost_usd=0,
                    datetime=datetime.now().strftime(DATETIME_FORMAT)
                )
            )]

def function_call_responses_from_llm_chunk(chunk):
    """Преобразует один элемент потока ответа LLM с функцией в один
    экземпляр llm_pb2.NewMessageResponse или возвращает None.
    
    Обрабатывает различные response. вызова функции:
    - .output_item.added: новый вызов функции (FunctionCallAdded)
    - .function_call_arguments.delta: промежуточные арг-ты (FunctionCallDelta)
    - .function_call_arguments.done: завершенные аргументы (FunctionCallDone)
    - .output_item.done: завершение вызова функции (FunctionCallComplete)
    """
    if hasattr(chunk, "type"):
        chunk_type = chunk.type
        # Обработка события начала вызова функции
        if chunk_type == "response.output_item.added":
            if hasattr(chunk, "item") and hasattr(chunk.item, "type"):
                if chunk.item.type == "function_call":
                    item = chunk.item
                    func_id = getattr(item, "id", "")
                    func_name = getattr(item, "name", "")
                    if func_id and func_name:
                        return llm_pb2.NewMessageResponse(
                            function_call_added=llm_pb2.FunctionCallAdded(
                                id=func_id,
                                name=func_name
                            )
                        ), None
        # Обработка события промежуточных аргументов функции
        elif chunk_type == "response.function_call_arguments.delta":
            if hasattr(chunk, "delta") and hasattr(chunk, "item_id"):
                delta = chunk.delta
                func_id = chunk.item_id
                return llm_pb2.NewMessageResponse(
                    function_call_delta=llm_pb2.FunctionCallDelta(
                        id=func_id,
                        content=delta
                    )
                ), None
        # Обработка события завершения аргументов функции
        elif chunk_type == "response.function_call_arguments.done":
            if hasattr(chunk, "arguments") and hasattr(chunk, "item_id"):
                arguments = chunk.arguments
                func_id = chunk.item_id
                return llm_pb2.NewMessageResponse(
                    function_call_done=llm_pb2.FunctionCallDone(
                        id=func_id,
                        arguments=arguments
                    )
                ), None
        # Обработка события завершения вызова функции
        elif chunk_type == "response.output_item.done":
            if hasattr(chunk, "item") and hasattr(chunk.item, "type"):
                if chunk.item.type == "function_call":
                    item = chunk.item
                    func_id = getattr(item, "id", "")
                    func_name = getattr(item, "name", "")
                    # Получаем аргументы
                    arguments = ""
                    if hasattr(item, "arguments"):
                        arguments = item.arguments
                    if func_id and func_name:
                        return llm_pb2.NewMessageResponse(
                            function_call_complete=llm_pb2.FunctionCallComplete(
                                id=func_id,
                                name=func_name,
                                arguments=arguments
                            )
                        ), item
    # Обработка события завершения вызова функции из choices[0]
    elif hasattr(chunk, "object") and (
        chunk.object == "chat.completion.chunk" and
        hasattr(chunk, "choices") and chunk.choices):
        choice0 = chunk.choices[0]
        delta = getattr(choice0, "delta", None)
        if delta and hasattr(delta, "tool_calls") and delta.tool_calls:
            for tool_call in delta.tool_calls:
                if hasattr(tool_call, "id") and hasattr(tool_call,
                                                        "function"):
                    func_id = tool_call.id
                    func = tool_call.function
                    func_name = getattr(func, "name", "")
                    arguments = getattr(func, "arguments", "")
                    if func_id and func_name:
                        return llm_pb2.NewMessageResponse(
                            function_call_complete=llm_pb2.FunctionCallComplete(
                                id=func_id,
                                name=func_name,
                                arguments=arguments
                            )
                        ), {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                "id": f"{chunk.id}-{func_id}",
                                "type": "function",
                                "function": {
                                    "name": func_name,
                                    "arguments": arguments
                                }
                                }
                            ]
                        }
    return None, None


def responses_from_llm_chunk(chunk):
    """Преобразует один элемент потока ответа LLM в один
    экземпляр llm_pb2.NewMessageResponse или возвращает None.
    """
    # delta content/reasoning_content, если есть
    delta_content = None
    delta_reasoning_content = None
    finish_reason = None
    if hasattr(chunk, "choices") and chunk.choices:
        choice0 = chunk.choices[0]
        delta = getattr(choice0, "delta", None)
        finish_reason = getattr(choice0, "finish_reason", None)
        if delta is not None:
            if getattr(delta, "content", None) is not None:
                delta_content = delta.content
            if getattr(delta, "reasoning_content", None) is not None:
                delta_reasoning_content = delta.reasoning_content

    # usage информация, если есть
    completion_tokens = None
    prompt_tokens = None
    total_tokens = None
    if hasattr(chunk, "usage") and chunk.usage:
        usage = chunk.usage
        completion_tokens = getattr(usage, "completion_tokens", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)

    # Если есть usage или конечный сигнал, создаём CompleteResponseType
    if (finish_reason == "stop" and delta_content is None) or (
        total_tokens is not None and (
            prompt_tokens is not None and completion_tokens is not None)):
        return llm_pb2.NewMessageResponse(
            complete=llm_pb2.CompleteResponseType(
                prompt_tokens=(prompt_tokens or 0),
                completion_tokens=(completion_tokens or 0),
                total_tokens=(total_tokens or 0),
                expected_cost_usd=ALL_API_VARS["yandexai"]["price_coef"] *
                                  (total_tokens or 0),
                datetime=datetime.now().strftime(DATETIME_FORMAT)
            )
        ), None
    # Если один из delta content есть, создаём GenerateResponseType
    elif delta_content is not None or delta_reasoning_content is not None:
        return llm_pb2.NewMessageResponse(
            generate=llm_pb2.GenerateResponseType(
                content=(delta_content or ""),
                reasoning_content=(delta_reasoning_content or ""),
                datetime=datetime.now().strftime(DATETIME_FORMAT)
            )
        ), delta_content
    else:
        # Не удалось разобрать часть ответа
        print(f"WARN: {chunk}")
        return None, None


def proc_llm_stream_responses(messages, tool_choice,
                              api_to_use, key_to_use, folder_to_use, model_to_use):
    """Генератор для обработки потока ответов от LLM.
    
    Args:
        messages: Сообщения для отправки в LLM
        tool_choice: Параметр tool_choice для LLM (например, "auto")
    
    Yields:
        Кортеж (response, item) где:
        - response: llm_pb2.NewMessageResponse или None
        - item: объект вызова функции или None
    """
    response = OpenAI(
        base_url=api_to_use,
        api_key=key_to_use,
        project=folder_to_use,
    ).chat.completions.create(
        model=model_to_use,
        messages=messages,
        stream=True,
        tool_choice=tool_choice,
        tools=TOOLS
    )
    try:
        for chunk in response:
            # Преобразуем chunk в один ответ protobuf (или None)
            resp, item = function_call_responses_from_llm_chunk(chunk)
            delta_content = None
            if resp is None:
                resp, delta_content = responses_from_llm_chunk(chunk)
            if resp is not None:
                yield (resp, item, delta_content)
    finally:
        response.response.close()


class LlmServicer(llm_pb2_grpc.LlmServicer):
    """Реализация LLM сервиса."""

    def Ping(self, request, context):
        """Простой метод для проверки работоспособности сервиса."""
        return empty_pb2.Empty()

    def AvailableModelsText2Text(self, request, context):
        """Получить список доступных Text2Text моделей."""
        try:
            t = available_models(ALL_API_VARS["yandexai"]["base_url"],
                                         ALL_API_VARS["yandexai"]["key"],
                                         ALL_API_VARS["yandexai"]["folder"],
                                         WHITELIST_REGEX_TEXT2TEXT,
                                         BLACKLIST_REGEX_TEXT2TEXT)
            t.append(ALL_API_VARS["openaivlm"]["model"])
            return llm_pb2.StringsListResponse(
                strings=t)
        except Exception as e:
            print(f"ERROR getting text2text models: {e}")
            context.set_details(f"ERROR getting text2text models: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return llm_pb2.StringsListResponse(strings=[])

    def AvailableModelsSpeech2Text(self, request, context):
        """Получить список доступных Speech2Text моделей."""
        try:
            return llm_pb2.StringsListResponse(
                strings=available_models(ALL_API_VARS["openai"]["base_url"],
                                         ALL_API_VARS["openai"]["key"], None,
                                         WHITELIST_REGEX_SPEECH2TEXT,
                                         BLACKLIST_REGEX_SPEECH2TEXT))
        except Exception as e:
            print(f"ERROR getting speech2text models: {e}")
            context.set_details(f"ERROR getting speech2text models: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return llm_pb2.StringsListResponse(strings=[])

    def AvailableTools(self, request, context):
        """Получить список доступных инструментов/функций."""
        return llm_pb2.StringsListResponse(strings=TOOLS_NAMES)

    def Transcribe(self, request_iter, context):
        """Транскрибация аудио без генерации текста.
        Принимает поток TranscribeRequest с mp3_chunk и опциональной speech2text_model.
        Возвращает одиночный TranscribeResponse.
        """
        try:
            audio_buffer = BytesIO()
            speech2text_override = None

            for request in request_iter:
                if speech2text_override is None and getattr(request, 'speech2text_model', None):
                    speech2text_override = request.speech2text_model
                if getattr(request, 'mp3_chunk', None):
                    audio_buffer.write(request.mp3_chunk)

            _, transcribe_proto = transcribe_audio(audio_buffer, speech2text_override)
            if transcribe_proto is None:
                raise ValueError("Ни одного аудио чанка не получено для транскрибации")
            return llm_pb2.TranscribeResponse(transcribe=transcribe_proto)
        except Exception as e:
            print(f"ERROR: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"ERROR: {e}")
            return llm_pb2.TranscribeResponse(
                transcribe=llm_pb2.TranscribeResponseType(
                    transcription="",
                    duration=0.0,
                    expected_cost_usd=0.0,
                    datetime=datetime.now().strftime(DATETIME_FORMAT)
                )
            )

    def NewMessage(self, request, context):
        """Метод для отправки сообщения языковой модели и получения ответа.
        
        request — одиночный запрос NewMessageRequest (только текст и опции).
        """
        try:
            # Извлекаем поля из одиночного запроса
            user_message = request.msg
            history = request.history if request.history else []
            text2text_override = request.text2text_model if request.text2text_model else None
            function_tool = request.function if request.function else None
            documents_urls = request.documents_urls if request.documents_urls else None
            images_urls = request.images_urls if request.images_urls else None
            markdown_urls = request.markdown_urls if request.markdown_urls else None

            # Если сообщение пусто, ошибка
            if not user_message:
                raise ValueError("Текст сообщения не получен")

            # История по умолчанию пуста
            if history is None:
                history = []

            # Если это новый чат (история пуста), генерируем название
            if not history:
                try:
                    chat_name_resp = generate_chat_name(user_message)
                    if chat_name_resp is not None:
                        yield llm_pb2.NewMessageResponse(chat_name=chat_name_resp)
                except Exception as e:
                    print(f"ERROR generating chat name: {e}")
                    # Не прерываем обработку запроса из-за ошибки имени чата

            # Обработка документов
            md_docs = dict()
            if documents_urls is not None:
                for url in documents_urls:
                    filename, md_content = convert_to_md(url)
                    if filename is None or md_content is None:
                        raise ValueError("Parse file error")
                    md_docs[filename] = md_content
                    # Разбиваем контент на чанки размером CHUNK_SIZE
                    for i in range(0, len(md_content), CHUNK_SIZE):
                        chunk = md_content[i:i + CHUNK_SIZE]
                        yield llm_pb2.NewMessageResponse(
                            markdown_chunk=llm_pb2.MarkdownChunkResponseType(
                                markdown_chunk=chunk,
                                original_url=url,
                                original_name=filename
                            )
                        )

            # Обработка markdown URLs
            if markdown_urls is not None:
                for md_url_obj in markdown_urls:
                    try:
                        with httpx.Client(timeout=60.0) as client:
                            response = client.get(md_url_obj.url)
                            md_content = response.text
                        md_docs[md_url_obj.original_name] = md_content
                    except Exception as e:
                        print(f"ERROR loading md from {md_url_obj.url}: {e}")

            # Сборка user_message
            user_message, vlm = build_user_message(user_message,
                                                   md_docs, images_urls)
            
            # Minio put
            minio_client = Minio(
                MINIO_ADDRESS,
                access_key=MINIO_ACCESS_KEY,
                secret_key=MINIO_SECRET_KEY,
                secure=True
            )
            object_name = str(uuid.uuid4())
            json_bytes = json.dumps(user_message).encode('utf-8')
            data_stream = BytesIO(json_bytes)
            try:
                minio_client.put_object(
                    BUCKET_NAME,
                    object_name,
                    data_stream,
                    length=len(json_bytes),
                    content_type='application/json'
                )
                yield llm_pb2.NewMessageResponse(
                    user_message_uid=object_name
                )
                print(f"Put user message: {object_name}")
            except Exception as e:
                print(f"Error uploading object: {e}")

            # Сборка контекста
            messages = build_messages_from_history(history, user_message,
                                                   text2text_override)

            # Определяем модель для запроса
            api_to_use = ALL_API_VARS["yandexai"]["base_url"]
            key_to_use = ALL_API_VARS["yandexai"]["key"]
            model_to_use = ALL_API_VARS["yandexai"]["model"]
            folder_to_use = ALL_API_VARS["yandexai"]["folder"]
            if text2text_override:
                model_to_use = text2text_override
                if vlm and model_to_use != ALL_API_VARS["openaivlm"]["model"]:
                    for msg in change_model_msgs():
                        yield msg
                    return
            elif vlm:
                model_to_use = ALL_API_VARS["openaivlm"]["model"]
            else:
                model_to_use = ALL_API_VARS["yandexai"]["model"]
            if model_to_use == ALL_API_VARS["openaivlm"]["model"]:
                api_to_use = ALL_API_VARS["openai"]["base_url"]
                key_to_use = ALL_API_VARS["openai"]["key"]
                folder_to_use = None

            # Определяем инструмент функции
            if function_tool is None:
                function_tool = "auto"
            else:
                function_tool = {
                    "type": "function",
                    "function": {"name" : function_tool}
                }

            # Отправка запроса в OpenAI API на генерацию ответа, если
            # пользователь не отменял запрос
            content = ""
            meta = None
            if context.is_active():
                try:
                    item = None
                    for r, i, d in proc_llm_stream_responses(
                        messages, function_tool, api_to_use, key_to_use, folder_to_use, model_to_use
                    ):
                        if d is not None:
                            content += d
                        if context.is_active() is False:
                            print("Client cancelled, stopping stream.")
                            break
                        if i is not None:
                            item = i
                        yield r
                    if item is not None:
                        messages.append(item)
                        result, meta = call_function(
                            item["tool_calls"][0]["function"]["name"],
                            json.loads(
                                item["tool_calls"][0]["function"]["arguments"]
                            )
                        )
                        if meta is not None:
                            yield llm_pb2.NewMessageResponse(
                                tool_metadata=meta
                            )
                        messages.append({
                            "role": "tool",
                            "tool_call_id": item["tool_calls"][0]["id"],
                            "content": result
                        })
                        for r, i, d in proc_llm_stream_responses(
                            messages, "none", api_to_use, key_to_use, folder_to_use, model_to_use
                        ):
                            if d is not None:
                                content += d
                            if context.is_active() is False:
                                print("Client cancelled, stopping stream.")
                                break
                            yield r
                except Exception as e:
                    print(f"STREAM ERROR: {e}")
                    context.set_code(grpc.StatusCode.INTERNAL)
                    context.set_details(f"STREAM ERROR: {e}")
            else:
                print("Client cancelled, stopping stream.")

            object_name_2 = str(uuid.uuid4())
            if meta is not None and hasattr(meta, "image_gen") and meta.image_gen.image_base64:
                content = [
                    {
                        "type": "text",
                        "text": content
                    },
                    {
                        "type": "image_url",
                        "image_url": { "url": meta.image_gen.image_base64 }
                    }
                ]
            json_bytes_2 = json.dumps({"role": "assistant", "content": content}).encode('utf-8')
            data_stream_2 = BytesIO(json_bytes_2)
            try:
                minio_client.put_object(
                    BUCKET_NAME,
                    object_name_2,
                    data_stream_2,
                    length=len(json_bytes_2),
                    content_type='application/json'
                )
                yield llm_pb2.NewMessageResponse(
                    llm_message_uid=object_name_2
                )
                print(f"Put llm message: {object_name_2}")
            except Exception as e:
                print(f"Error uploading object: {e}")
        except Exception as e:
            print(f"ERROR: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"ERROR: {e}")


def serve():
    """Запуск gRPC сервера с авторизацией."""
    with open("server.crt", "rb") as f:
        server_cert = f.read()
    with open("server.key", "rb") as f:
        server_key = f.read()
    server_creds = grpc.ssl_server_credentials(
        [(server_key, server_cert)]
    )

    # Создаём сервер с Interceptor для авторизации
    # Interceptor будет проверять авторизацию для каждого RPC вызова
    # ПЕРЕД тем как вызвать бизнес-логику метода
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        interceptors=[AuthInterceptor()]  # Добавляем Interceptor
    )

    llm_pb2_grpc.add_LlmServicer_to_server(
        LlmServicer(), server
    )
    server.add_secure_port("[::]:50051", server_creds)
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    # Вывод конфигурации API (ключи скрыты)
    for key, value in ALL_API_VARS.items():
        print(f"API config for {key}:")
        for case_key, case_value in value.items():
            if case_key == "key":
                print(f"  {case_key}=***")
            else:
                print(f"  {case_key}={case_value}")
    # Проверка конфигурации API
    assert ALL_API_VARS["yandexaisummary"]["model"], "summary model is n/a"
    assert ALL_API_VARS["yandexaisummary"]["prices_url"], "sum cost is n/a"
    assert ALL_API_VARS["yandexai"]["prices_url"], "yandexai prices url is n/a"
    assert ALL_API_VARS["yandexai"]["base_url"], "yandexai base url is n/a"
    assert ALL_API_VARS["yandexai"]["key"], "yandexai api key is n/a"
    assert ALL_API_VARS["yandexai"]["model"], "yandexai model is n/a"
    assert ALL_API_VARS["yandexai"]["folder"], "yandexai folder is n/a"
    assert ALL_API_VARS["openai"]["prices_url"], "openai prices url is n/a"
    assert ALL_API_VARS["openai"]["base_url"], "openai base url is n/a"
    assert ALL_API_VARS["openai"]["key"], "openai api key is n/a"
    assert ALL_API_VARS["openai"]["model"], "openai model is n/a"
    assert ALL_API_VARS["openaiimgen"]["prices_url"], "openai prices url is n/a"
    assert ALL_API_VARS["openaiimgen"]["model"], "openai model is n/a"
    assert ALL_API_VARS["openaivlm"]["prices_url"], "openai prices url is n/a"
    assert ALL_API_VARS["openaivlm"]["model"], "openai model is n/a"
    # Скрэппинг цен (prepare.py)
    if GENERATE_CONFIG_WHEN == "always" or (
        GENERATE_CONFIG_WHEN == "missing" and not os.path.exists(CONFIG_PATH)):
        print("Generating config...")
        try:
            subprocess.run(
                ["/root/.local/bin/uv", "run",
                 "--env-file", ".env", "prepare.py"],
                check=True
            )
        except Exception as e:
            print(f"ERROR: config gen fail: {e}")
            exit(1)
    try:
        with open(CONFIG_PATH) as f:
            config = json.load(f)
        if 'generated_at' not in config or not config['generated_at']:
            raise ValueError("Invalid config: missing 'generated_at'")
        for name, coef in config.get("prices_coefs", {}).items():
            ALL_API_VARS[name]["price_coef"] = coef
            print(f"Price coef for {name}: {coef}")
    except Exception as e:
        print(f"ERROR: invalid config: {e}")
        exit(1)
    # Информация о безопасности
    if SECRET_KEY:
        print("Authorization ENABLED")
    else:
        print("Authorization DISABLED - all requests will be accepted!")
    # Проверка доступности моделей
    print("Checking available models...")
    check_arr = available_models(ALL_API_VARS["yandexai"]["base_url"],
                                 ALL_API_VARS["yandexai"]["key"],
                                 ALL_API_VARS["yandexai"]["folder"],
                                 WHITELIST_REGEX_TEXT2TEXT,
                                 BLACKLIST_REGEX_TEXT2TEXT)
    check_arr.append(ALL_API_VARS["openaivlm"]["model"])
    check_arr_speech=available_models(ALL_API_VARS["openai"]["base_url"],
                                      ALL_API_VARS["openai"]["key"], None,
                                      WHITELIST_REGEX_SPEECH2TEXT,
                                      BLACKLIST_REGEX_SPEECH2TEXT)
    if ALL_API_VARS["yandexai"]["model"] not in check_arr:
        print(f"ERROR: Text2Text model {ALL_API_VARS["yandexai"]["model"]} "
              "not found in available models!")
    elif ALL_API_VARS["yandexaisummary"]["model"] not in check_arr:
        print(f"ERROR: summ model {ALL_API_VARS["yandexaisummary"]["model"]} "
              "not found in available models!")
    elif ALL_API_VARS["openai"]["model"] not in check_arr_speech:
        print(f"ERROR: Speech2Text model {ALL_API_VARS["openai"]["model"]} "
              "not found in available models!")
    else:
        print("Starting server...")
        logging.basicConfig()
        serve()
