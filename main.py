r"""
gRPC сервер для взаимодействия с языковой моделью.
Команда для генерации llm_pb2.py и llm_pb2_grpc.py из llm.proto:
uv run -m grpc_tools.protoc -I.\uralsteel-grpc-api\llm\ --python_out=.
--grpc_python_out=. llm.proto
"""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from concurrent import futures
from io import BytesIO
import json
import os
import re
import subprocess
import uuid
import asyncio

from google.protobuf import empty_pb2
import grpc
import httpx
from openai import OpenAI
from minio import Minio
from minio.error import S3Error

from grpc_health.v1 import health
from grpc_health.v1 import health_pb2
from grpc_health.v1 import health_pb2_grpc

import llm_pb2
import llm_pb2_grpc
from auth_interceptor import AuthInterceptor
from logger import logger
from util import (
    build_user_message, remote_pdf_to_b64_images, websearch, check_docling_health, convert_to_md,
    get_messages_wo_b64_images, engineer, process_engineer_url
)
from minio_util import MINIO_ADDRESS, BUCKET_NAME, MINIO_ACCESS_KEY, MINIO_SECRET_KEY

# Словил кринж с systemd...
# Фикс для подстановки переменных окружения в другие переменные окружения
for key, value in os.environ.items():
    if "${" in value:
        os.environ[key] = os.path.expandvars(value)

# Переменные окружения вида CONST_api_case
# Пока CONST + YANDEXAI_/OPENAI_ + BASE_URL/FOLDER/KEY/MODEL/PRICES_URL/TOOLS
CONST = "INFERENCE_API_"
CONST_LEN = len(CONST)
ALL_API_VARS = dict()
MODEL_TO_API = dict()
for name, value in os.environ.items():
    if name.startswith(CONST):
        api_and_case = name[CONST_LEN:].lower()
        delim = api_and_case.find("_")
        api = api_and_case[:delim]
        case = api_and_case[delim + 1:]
        ALL_API_VARS.setdefault(api, dict())
        ALL_API_VARS[api][case] = value
        if case == "model":
            MODEL_TO_API[value] = api
        elif case == "tools":
            ALL_API_VARS[api][case] = json.loads(value)
# Секретный ключ для клиентского доступа к gRPC методам
SECRET_KEY = os.environ.get('SECRET_KEY', '')
# Формат даты и времени
DATETIME_FORMAT = os.environ.get('DATETIME_FORMAT', '%Y-%m-%dT%H:%M:%S')
DATETIME_TZ = ZoneInfo('Europe/Moscow')
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
# Размер чанка MarkDown версий документов
CHUNK_SIZE = int(os.environ.get('CHUNK_SIZE', 8192))
# Регулярные выражения для фильтрации моделей
WHITELIST_REGEX_TEXT2TEXT  = os.environ.get('WHITELIST_REGEX_TEXT2TEXT', '.*')
BLACKLIST_REGEX_TEXT2TEXT  = os.environ.get('BLACKLIST_REGEX_TEXT2TEXT', '$^')
WHITELIST_REGEX_SPEECH2TEXT= os.environ.get('WHITELIST_REGEX_SPEECH2TEXT', '.*')
BLACKLIST_REGEX_SPEECH2TEXT= os.environ.get('BLACKLIST_REGEX_SPEECH2TEXT', '$^')
# # S3
# MINIO_ADDRESS = os.environ.get('MINIO_ADDRESS', '')
# BUCKET_NAME = os.environ.get('BUCKET_NAME', 'cache')
# MINIO_ACCESS_KEY = os.environ.get('MINIO_ACCESS_KEY', None)
# MINIO_SECRET_KEY = os.environ.get('MINIO_SECRET_KEY', None)
# Костыль для пустого openai usage
USAGE_FIX = 3.6
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
            "description": "Generate a 1024x1024 square image based on the query and get its url.",
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
            "name": "engineer",
            "description": "Retrieve relevant information and recommendations from an AI-agent on calculating the steel grade formula depending on the conditions.",
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
# Пояснения инструментов
TOOLS_SUMMARY = "\n".join([
    f"- {tool["function"]["description"]}"
    for tool in TOOLS
])
# Дополнительные системные подсказки для моделей
TYPICAL_SITUATIONS_SOLVING_PROMPT = (
"""If the user asks to create document, follow these rules:
- If the user asks to create, generate, draft, write, or output a document
  (e.g., report, contract, policy, specification, letter, instructions, manual,
  article, or similar), your response MUST contain
  ONLY the document content itself.
- DO NOT add explanations, comments, notes, suggestions, warnings, summaries,
  or introductory/outro text.
- DO NOT mention that the text was generated, exported, or formatted.
- Use Markdown syntax for structure (headings, lists, tables, emphasis)
  where appropriate.
- NEVER wrap the document content in Markdown code blocks (```),
  inline code, or quoted blocks.
- Output the document as direct, top-level Markdown content,
  suitable for immediate Markdown-to-PDF conversion.
If the user asks to calculate the steel grade formula depending on the conditions, follow these rules:
- Do not tell the user that the document was scanned or printed poorly somewhere; he won't be interested.
- You MUST ask user to tell the steel grade if not provided. You can abort tool call by passing ABORT as query.
- If you use a document in your response, refer to it in square brackets by its number.
- ALWAYS add references/sources list at the end of your final answer in the same language as user request.
- You MUST provide calculations of the values ​​of coefficients/equivalents that are used in the process of solving the problem.
"""
# If the user asks to do something that requires the following tools:
# """
# + TOOLS_SUMMARY + """
# then ask him to select the appropriate tool using the plus button
# next to the text input field.
# If there is no suitable tool, inform the user about it.
# If the user asks what you can do,
# pay attention to the presence of the above tools, but:
# IF TOOLS ARE NOT PROVIDED, DO NOT ATTEMPT TO USE THEM! THIS MEANS THE USER DID NOT SELECT ANY TOOL.
# IN THIS CASE YOU MUST RESPONSE WITH INFORMATION ABOUT PLUS BUTTON! EXAMPLE: "Please, use the plus button next to the text input field"
# """
)
RESTRICTIONS = (
"""Restrictions:
- STRICTLY FORBIDDEN: violence, murder, physical harm, threats, or instructions
  enabling harm, including fictional or historical descriptions
  focused on violent acts.
- STRICTLY FORBIDDEN: sensitive social or identity-related topics,
  including religion, religious beliefs or figures, gender identity,
  sexual orientation, sexuality, or any content likely to cause ideological,
  moral, or identity-based conflict.
- Geographic, scientific, environmental, cultural, and non-violent
  historical topics are allowed, even if they indirectly intersect with
  politics or treaties.
- If a request violates these rules, politely refuse ONLY the restricted part
  and, when possible, redirect to a safe, neutral alternative.
- You MUST NOT engage in roleplay, impersonation, or fictional personas.
  Always respond as a neutral, factual assistant.
"""
)
NEED_VLM_WARNING = {
    "role": "system",
    "content": """WARNING: USER REQUEST CONTAINS PICTURES,
BUT YOU ARE NOT ABLE TO SEE THEM CURRENTLY,
YOU MUST WARN THE USER ABOUT THIS,
TELL THAT YOU WILL IGNORE THEM,
MENTION THAT GPT-5.2 IS ABLE TO SEE PICTURES
AND USER CAN SWITCH TO IT IN THE LEFT UPPER CORNER.
"""
}
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


def tools_whitelist_by_model(model_name):
    """Возвращает список инструментов, разрешённых для модели model_name."""
    api_ = MODEL_TO_API.get(model_name, None)
    if api_ is None:
        return []
    return ALL_API_VARS.get(api_, {}).get("tools", [])


def update_model_to_api(models, new_api):
    """Обновляет глобальную переменную MODEL_TO_API новыми моделями."""
    for model in models:
        MODEL_TO_API[model] = new_api


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
    check_arr_ret = []
    for model in models_list.data:
        model_id = getattr(model, "id", "")
        if not model_id:
            continue
        if wl.search(model_id) and not bl.search(model_id):
            check_arr_ret.append(model_id)
    return check_arr_ret


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
        datetime=datetime.now(DATETIME_TZ).strftime(DATETIME_FORMAT)
    )
    return text, proto


def image_gen(query: str):
    """Генерирует изображение по запросу и возвращает его URL."""
    result = None
    image_base64 = None
    try:
        client = OpenAI(
            api_key=ALL_API_VARS["openaiimgen"]["key"],
            base_url=ALL_API_VARS["openaiimgen"]["base_url"],
        )

        response = client.images.generate(
            model=ALL_API_VARS["openaiimgen"]["model"],
            prompt=query,
            quality="medium",
            size="1024x1024"
        )

        image_base64 = response.data[0].b64_json
        result = ("The image has been generated, everything is fine. "
                  "Do not say that you can't generate the image, you "
                  "have already done this if you see this message.")
    except Exception as e:
        result = f"An error occurred during image gen request: {e}"
    return result, llm_pb2.ToolMetadataResponse(
        image_gen=llm_pb2.ToolImageGenMetadata(
            image_base64=image_base64 or "",
            expected_cost=ALL_API_VARS["openaiimgen"]["price_coef"]
        )
    )


def call_function(log_uid, name, args):
    """Вызов функции инструмента по имени с аргументами args."""
    logger.debug("(%s) calling %s with: %s", log_uid, name, args)
    if name == "websearch":
        result = websearch(**args, tavily_base_url=TAVILY_BASE_URL)
        meta = llm_pb2.ToolMetadataResponse(
            websearch=llm_pb2.ToolWebSearchMetadata(
                item=[llm_pb2.ToolWebSearchMetadataItem(
                        url=item["url"],
                        title=item["title"]
                    )
                    for item
                    in result] if isinstance(result, list) else []
            )
        )
        return json.dumps(result,
                          ensure_ascii=False), meta
    elif name == "image_gen":
        result, meta = image_gen(**args)
        return result, meta
    elif name == "engineer":
        meta = engineer(**args, base_url="http://localhost:9621")
        result = ""
        meta_proc = []
        if isinstance(meta, list):
            for item in meta:
                url, _ = process_engineer_url(item["url"])
                title=item["title"]
                result += f'\n# REFERENCE DOCUMENT [{title}] "{item["url"].split("/", 1)[1].rsplit(".", 1)[0]}"\n' + "\n".join(
                    [f"## PAGE {i+1}\n\n![page {i+1}]({u})\n\n"
                     for i, u
                     in enumerate(remote_pdf_to_b64_images(url))])
                meta_proc.append(llm_pb2.ToolWebSearchMetadataItem(
                        url=url,
                        title=title
                    ))
            result, _ = build_user_message("THIS IS TOOL CALL OUTPUT",
                                           {"CONCATENATION OF REFERENCE DOCUMENTS":result}, [],
                                           IMGHDR_TO_MIME)
            meta = llm_pb2.ToolMetadataResponse(
                websearch=llm_pb2.ToolWebSearchMetadata(
                    item=meta_proc
                )
            )
        else:
            result = "Tool call successfully aborted."
            meta = None
        return result, meta
    return json.dumps({"error": f"Unknown tool {name}"}, ensure_ascii=False)


def generate_chat_name(log_uid: str, user_message: str):
    """Генерирует название чата на основе сообщения пользователя.
    
    Использует llm для создания короткого названия (до 1024 токенов).
    Возвращает строку с названием чата или None в случае ошибки.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI agent that generates short, concise "
                "chat names based on the user's initial message. The name "
                "should be brief (4 words max), descriptive, and in the "
                "same language as the user message. Return only the name, "
                "nothing else. You must NOT answer questions. If the user "
                "message is empty, pointless or violates the restrictions, "
                "then name the chat with the current date and time (copy it). "
                "Current date and time: "
                f"{datetime.now(DATETIME_TZ).strftime(DATETIME_FORMAT)}. "
                "Do not insert any links in your answers. You MUST use "
                "the same language as the user. And again, you must NOT "
                "respond to message, just name the chat.\n" +
                RESTRICTIONS
            )
        },
        {
            "role": "user",
            "content": user_message
        }
    ]
    response = OpenAI(
        base_url=ALL_API_VARS["openaimini"]["base_url"],
        api_key=ALL_API_VARS["openaimini"]["key"],
    ).chat.completions.create(
        model=ALL_API_VARS["openaimini"]["model"],
        messages=messages,
    )
    name_c = response.choices[0].message.content.strip()
    if not hasattr(response, "usage"):
        raise ValueError("usage required")
    prompt_tokens = getattr(response.usage, "prompt_tokens", 0)
    completion_tokens = getattr(response.usage, "completion_tokens", 0)
    total_tokens = getattr(response.usage, "total_tokens", 0)

    # Рассчитываем стоимость с поддержкой разделения на входные и выходные токены
    if "price_coef_input" in ALL_API_VARS["openaimini"]:
        # Модель с разделением на входные и выходные токены
        input_coef = ALL_API_VARS["openaimini"]["price_coef_input"]
        output_coef = ALL_API_VARS["openaimini"]["price_coef_output"]
        usd = input_coef * prompt_tokens + output_coef * completion_tokens
    else:
        # Модель с единой ценой
        usd = ALL_API_VARS["openaimini"]["price_coef"] * total_tokens

    logger.info("(%s) generate_chat_name cost = %s\n\tprompt_tokens = %s\n\tcompletion_tokens = %s\n\ttotal_tokens = %s",
                log_uid, usd, prompt_tokens, completion_tokens, total_tokens)
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
            f"**{datetime.now(DATETIME_TZ).strftime(DATETIME_FORMAT)}**. "
            "*Note that current date and time are relevant only for last "
            "message, previous ones could be sent a long time ago*. "
            "You can use the **websearch** tool for relevant information "
            "retrieval and **image_gen** for image generation. "
            "**Do not insert any links or images in your answers. Respond in "
            "the same language as the user using MarkDown markup language. "
            "If tool output is provided, ALWAYS base your answer on it.**\n" +
            TYPICAL_SITUATIONS_SOLVING_PROMPT + RESTRICTIONS
        )
    }]
    vlm2 = False
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
                tmps = minio_client.get_object(
                    BUCKET_NAME, message_uuid).read().decode('utf-8')
                tmpd = json.loads(tmps)
                if isinstance(tmpd["content"], list) and any(
                    "image_url" in item for item in tmpd["content"]):
                    vlm2 = True
                messages.append(tmpd)
            except S3Error as e:
                logger.error("Getting object from MinIO: %s", e)
            except json.JSONDecodeError as e:
                logger.error("Decoding JSON: %s", e)
            logger.debug("Get one message: %s", tmps[:420])
    messages.append(user_message)
    return messages, vlm2


def change_model_msgs():
    """Сообщения о необходимости смены модели."""
    return [llm_pb2.NewMessageResponse(
                generate=llm_pb2.GenerateResponseType(
                    content="Ваш запрос слишком сложный для выбранной "
                            f"модели, пожалуйста, выберите {
                                ALL_API_VARS["openaivlm"]["model"]}",
                    reasoning_content="",
                    datetime=datetime.now(DATETIME_TZ).strftime(DATETIME_FORMAT)
                )
            ), llm_pb2.NewMessageResponse(
                complete=llm_pb2.CompleteResponseType(
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    expected_cost_usd=0,
                    datetime=datetime.now(DATETIME_TZ).strftime(DATETIME_FORMAT)
                )
            )]


def function_call_responses_from_llm_chunk(log_uid, chunk, id_="", nm_="", args=""):
    """Преобразует один элемент потока ответа LLM с функцией в один
    экземпляр llm_pb2.NewMessageResponse или возвращает None.
    
    Обрабатывает различные response. вызова функции:
    - .output_item.added: новый вызов функции (FunctionCallAdded)
    - .function_call_arguments.delta: промежуточные арг-ты (FunctionCallDelta)
    - .function_call_arguments.done: завершенные аргументы (FunctionCallDone)
    - .output_item.done: завершение вызова функции (FunctionCallComplete)
    """
    if hasattr(chunk, "object") and (
        chunk.object == "chat.completion.chunk" and
        hasattr(chunk, "choices") and chunk.choices):
        choice0 = chunk.choices[0]
        delta = getattr(choice0, "delta", None)
        if delta and hasattr(delta, "tool_calls"):
            finish_reason = getattr(choice0, "finish_reason", None)
            if delta.tool_calls is not None and delta.tool_calls:
                for tool_call in delta.tool_calls:
                    if hasattr(tool_call, "id") and hasattr(tool_call,
                                                            "function"):
                        func_id = tool_call.id
                        func = tool_call.function
                        func_name = getattr(func, "name", "")
                        arguments = getattr(func, "arguments", "")
                        if func_id and func_name:
                            if finish_reason is None:
                                return llm_pb2.NewMessageResponse(
                                    function_call_added=llm_pb2.FunctionCallAdded(
                                        id=func_id,
                                        name=func_name
                                    )
                                ), None, func_id, func_name, ""
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
                            }, None, None, None
                        else:
                            args += arguments
                            return llm_pb2.NewMessageResponse(
                                function_call_delta=llm_pb2.FunctionCallDelta(
                                    id=id_,
                                    content=arguments
                                )
                            ), None, id_, nm_, args
            elif delta.tool_calls is None and finish_reason == 'tool_calls':
                return llm_pb2.NewMessageResponse(
                    function_call_complete=llm_pb2.FunctionCallComplete(
                        id=id_,
                        name=nm_,
                        arguments=args
                    )
                ), {
                    "role": "assistant",
                    "tool_calls": [
                        {
                        "id": id_,
                        "type": "function",
                        "function": {
                            "name": nm_,
                            "arguments": args
                        }
                        }
                    ]
                }, None, None, None
    return None, None, None, None, None


def responses_from_llm_chunk(price_info, log_uid, chunk, summ, sumr):
    """Преобразует один элемент потока ответа LLM в один
    экземпляр llm_pb2.NewMessageResponse или возвращает None.
    
    price_info может быть либо float (для простых цен), либо dict с ключами
    'input' и 'output' для разных цен входных и выходных токенов.
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
    completion_tokens_fix = int(sumr / USAGE_FIX)
    prompt_tokens_fix = int(summ / USAGE_FIX)
    total_tokens_fix = completion_tokens_fix + prompt_tokens_fix
    if hasattr(chunk, "usage") and chunk.usage:
        usage = chunk.usage
        completion_tokens = getattr(usage, "completion_tokens", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)
        logger.info("(%s) completion_tokens %s, prompt_tokens %s, total_tokens %s",
            log_uid, completion_tokens or '-', prompt_tokens or '-', total_tokens or '-')

    # Если есть usage или конечный сигнал, создаём CompleteResponseType
    if (total_tokens is not None and (
            prompt_tokens is not None and completion_tokens is not None)):
        # Рассчитываем стоимость в зависимости от структуры price_info
        if isinstance(price_info, dict):
            # Разные цены для входных и выходных токенов
            input_coef = price_info.get("input", 0)
            output_coef = price_info.get("output", 0)
            expected_cost = (input_coef * (prompt_tokens or prompt_tokens_fix) +
                           output_coef * (completion_tokens or completion_tokens_fix))
        else:
            # Единая цена (для обратной совместимости)
            expected_cost = price_info * (total_tokens or total_tokens_fix)
        
        return llm_pb2.NewMessageResponse(
            complete=llm_pb2.CompleteResponseType(
                prompt_tokens=(prompt_tokens or prompt_tokens_fix),
                completion_tokens=(completion_tokens or completion_tokens_fix),
                total_tokens=(total_tokens or total_tokens_fix),
                expected_cost_usd=expected_cost,
                datetime=datetime.now(DATETIME_TZ).strftime(DATETIME_FORMAT)
            )
        ), None
    # Если один из delta content есть, создаём GenerateResponseType
    elif delta_content is not None or delta_reasoning_content is not None:
        return llm_pb2.NewMessageResponse(
            generate=llm_pb2.GenerateResponseType(
                content=(delta_content or ""),
                reasoning_content=(delta_reasoning_content or ""),
                datetime=datetime.now(DATETIME_TZ).strftime(DATETIME_FORMAT)
            )
        ), delta_content
    else:
        # Скип
        return None, None



def proc_llm_stream_responses(price_info, log_uid, messages, tool_choice,
                              api_to_use, key_to_use, dir_to_use,
                              model_to_use, summ, sumr):
    """Генератор для обработки потока ответов от LLM.
    
    Args:
        price_info: Информация о цене (может быть float или dict с ключами 'input'/'output')
        messages: Сообщения для отправки в LLM
        tool_choice: Параметр tool_choice для LLM (например, "auto")
    
    Yields:
        Кортеж (response, item) где:
        - response: llm_pb2.NewMessageResponse или None
        - item: объект вызова функции или None
    """
    logger.info("(%s) model_to_use is set to %s, tool_choice is set to %s",
          log_uid, model_to_use or '-', tool_choice or '-')
    if tool_choice != "none":
        response = OpenAI(
            base_url=api_to_use,
            api_key=key_to_use,
            project=dir_to_use,
        ).chat.completions.create(
            model=model_to_use,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
            tool_choice=tool_choice,
            tools=TOOLS
        )
    else:
        response = OpenAI(
            base_url=api_to_use,
            api_key=key_to_use,
            project=dir_to_use,
        ).chat.completions.create(
            model=model_to_use,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
        )
    try:
        id_ = ""
        nm_ = ""
        args = ""
        for chunk in response:
            # Преобразуем chunk в один ответ protobuf (или None)
            resp, item, id_, nm_, args = function_call_responses_from_llm_chunk(
                log_uid, chunk, id_, nm_, args)
            delta_content = None
            if resp is None:
                resp, delta_content = responses_from_llm_chunk(
                    price_info, log_uid, chunk, summ, sumr)
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
            t.append(ALL_API_VARS["deepseek"]["model"])
            t.append(ALL_API_VARS["openaimini"]["model"])
            return llm_pb2.StringsListResponse(strings=t)
        except Exception as e:
            logger.error("Getting text2text models: %s", e)
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
            logger.error("Getting speech2text models: %s", e)
            context.set_details(f"ERROR getting speech2text models: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return llm_pb2.StringsListResponse(strings=[])

    def AvailableTools(self, request, context):
        """Получить список доступных инструментов/функций."""
        model_name = getattr(request, 'model', None)
        if model_name:
            tools = tools_whitelist_by_model(model_name)
        else:
            tools = TOOLS_NAMES
        return llm_pb2.StringsListResponse(strings=tools)

    def Transcribe(self, request_iterator, context):
        """Транскрибация аудио без генерации текста.
        Принимает поток TranscribeRequest с mp3_chunk и опциональной
        speech2text_model.
        Возвращает одиночный TranscribeResponse.
        """
        try:
            audio_buffer = BytesIO()
            speech2text_override = None

            for request in request_iterator:
                if speech2text_override is None and getattr(
                    request, 'speech2text_model', None):
                    speech2text_override = request.speech2text_model
                if getattr(request, 'mp3_chunk', None):
                    audio_buffer.write(request.mp3_chunk)

            _, transcribe_proto = transcribe_audio(audio_buffer,
                                                   speech2text_override)
            if transcribe_proto is None:
                raise ValueError("Ни одного аудио чанка не получено!")
            return llm_pb2.TranscribeResponse(transcribe=transcribe_proto)
        except Exception as e:
            logger.error("Transcribe: %s", e)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"ERROR: {e}")
            return llm_pb2.TranscribeResponse(
                transcribe=llm_pb2.TranscribeResponseType(
                    transcription="",
                    duration=0.0,
                    expected_cost_usd=0.0,
                    datetime=datetime.now(DATETIME_TZ).strftime(DATETIME_FORMAT)
                )
            )

    def NewMessage(self, request, context):
        """Метод для отправки сообщения языковой модели и получения ответа.
        
        request — одиночный запрос NewMessageRequest (только текст и опции).
        """
        try:
            # Извлекаем поля из одиночного запроса
            user_message = request.msg
            history = getattr(request, 'history', [])
            text2text_override = getattr(request, 'text2text_model', None)
            log_uid = str(uuid.uuid4())
            logger.debug("(%s) text2text_model is set to %s",
                         log_uid, request.text2text_model or '-')
            function_tool = getattr(request, 'function', None)
            documents_urls = getattr(request, 'documents_urls', None)
            images_urls = getattr(request, 'images_urls', None)
            markdown_urls = getattr(request, 'markdown_urls', None)

            # Если сообщение пусто, ошибка
            if not user_message:
                raise ValueError("Текст сообщения не получен")

            # История по умолчанию пуста
            if history is None:
                history = []

            # Если это новый чат (история пуста), генерируем название
            if not history:
                try:
                    chat_name_r = generate_chat_name(log_uid, user_message)
                    if chat_name_r is not None:
                        yield llm_pb2.NewMessageResponse(chat_name=chat_name_r)
                except Exception as e:
                    logger.error("(%s) Error generating chat name: %s",
                                 log_uid, e)
                    # Не прерываем обработку запроса из-за ошибки имени чата

            # Обработка документов
            md_docs = dict()
            if documents_urls is not None:
                for url in documents_urls:
                    filename, md_content = convert_to_md(url, DOCLING_ADDRESS)
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
                        logger.error("(%s) Error loading md from %s: %s",
                                     log_uid, md_url_obj.url, e)

            # Сборка user_message
            user_message, vlm = build_user_message(user_message,
                                                   md_docs, images_urls,
                                                   IMGHDR_TO_MIME)

            # Minio put
            minio_client = Minio(
                MINIO_ADDRESS,
                access_key=MINIO_ACCESS_KEY,
                secret_key=MINIO_SECRET_KEY,
                secure=True
            )
            object_name = str(uuid.uuid4())
            tmps = json.dumps(user_message)
            json_bytes = tmps.encode('utf-8')
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
                logger.info("(%s) put user message: %s", log_uid, tmps[:420])
            except Exception as e:
                logger.error("(%s) error uploading user message object: %s",
                             log_uid, e)

            # Сборка контекста
            messages, vlm2 = build_messages_from_history(history, user_message,
                                                         text2text_override)

            # Определяем модель и нужно ли предупреждение об игноре картинок
            if text2text_override:
                model_to_use = text2text_override
                if ((vlm or vlm2) and
                    model_to_use != ALL_API_VARS["openaivlm"]["model"]):
                    get_messages_wo_b64_images(messages)
                    messages.append(NEED_VLM_WARNING)
                    # for msg in change_model_msgs():
                    #     yield msg
                    # return
            elif vlm or vlm2:
                model_to_use = ALL_API_VARS["openaivlm"]["model"]
            else:
                model_to_use = ALL_API_VARS["yandexai"]["model"]
            api_to_use = ALL_API_VARS[MODEL_TO_API[model_to_use]]["base_url"]
            key_to_use = ALL_API_VARS[MODEL_TO_API[model_to_use]]["key"]
            dir_to_use = ALL_API_VARS[MODEL_TO_API[model_to_use]].get("folder")
            # Получаем информацию о цене (может быть float или dict с 'input'/'output')
            api_name = MODEL_TO_API[model_to_use]
            if "price_coef_input" in ALL_API_VARS[api_name]:
                # Модель с разделением на входные и выходные токены
                price_info = {
                    "input": ALL_API_VARS[api_name]["price_coef_input"],
                    "output": ALL_API_VARS[api_name]["price_coef_output"]
                }
            else:
                # Модель с единой ценой
                price_info = ALL_API_VARS[api_name]["price_coef"]

            tools = tools_whitelist_by_model(model_to_use)
            # Определяем инструмент функции
            if function_tool is None or not function_tool:
                function_tool = "auto"  # "none"  # пока без "auto" живём, надо тестить
                if not tools:
                    function_tool = "none"
            elif function_tool in tools:
                function_tool = {
                    "type": "function",
                    "function": {"name" : function_tool}
                }
            else:
                raise ValueError(f"Tool {function_tool} not allowed for "
                                 f"model {model_to_use}")
            logger.info("(%s) TOOL: %s", log_uid, str(function_tool))

            # Отправка запроса в OpenAI API на генерацию ответа, если
            # пользователь не отменял запрос
            content = ""
            meta = None
            if context.is_active():
                try:
                    item = None
                    summ = 0
                    for message in messages:
                        summ += len(message.get("content", ""))
                    sumr = 0
                    for r, i, d in proc_llm_stream_responses(
                        price_info, log_uid, messages, function_tool, api_to_use,
                        key_to_use, dir_to_use, model_to_use, summ, sumr
                    ):
                        if d is not None:
                            content += d
                            sumr = len(content)
                        if context.is_active() is False:
                            logger.info(
                                "(%s) client cancelled, stopping stream",
                                log_uid)
                            break
                        if i is not None:
                            item = i
                        yield r
                    if item is not None:
                        messages.append(item)
                        result, meta = call_function(
                            log_uid,
                            item["tool_calls"][0]["function"]["name"],
                            json.loads(
                                item["tool_calls"][0]["function"]["arguments"]
                            )
                        )
                        #logger.debug("(%s) tool output: %s\n%s",
                        #             log_uid, result[:420], str(meta)[:420])
                        if meta is not None:
                            yield llm_pb2.NewMessageResponse(
                                tool_metadata=meta
                            )
                        if isinstance(result, dict) and "role" in result:
                            # Если результат уже в формате сообщения, добавляем его
                            messages.append({
                                "role": "tool",
                                "tool_call_id": item["tool_calls"][0]["id"],
                                "content": "The actual tool output will be provided as next assistant message"
                            })
                            result["role"] = "assistant"
                            messages.append(result)
                            logger.info(
                                "(%s) DONE SOME INSANE SHIT!1!!!!!111111",
                                log_uid)
                        else:
                            messages.append({
                                "role": "tool",
                                "tool_call_id": item["tool_calls"][0]["id"],
                                "content": result
                            })
                            logger.info(
                                "(%s) NOT DONE FUCKING ANYTHING!1!!!!!111111",
                                log_uid)
                        summ = 0
                        for message in messages:
                            summ += len(message.get("content", ""))
                        sumr = 0
                        for r, i, d in proc_llm_stream_responses(
                            price_info, log_uid, messages, "none", api_to_use, key_to_use,
                            dir_to_use, model_to_use, summ, sumr
                        ):
                            if d is not None:
                                content += d
                                sumr = len(content)
                            if context.is_active() is False:
                                logger.info(
                                    "(%s) client cancelled, stopping stream",
                                    log_uid)
                                break
                            yield r
                except Exception as e:
                    logger.error("Stream error: %s", e)
                    context.set_code(grpc.StatusCode.INTERNAL)
                    context.set_details(f"STREAM ERROR: {e}")
            else:
                logger.info("(%s) client cancelled, stopping stream", log_uid)

            object_name_2 = str(uuid.uuid4())
            if meta is not None and hasattr(
                meta, "image_gen") and meta.image_gen.image_base64:
                content = [
                    {
                        "type": "text",
                        "text": content
                    },
                    {
                        "type": "image_url",
                        "image_url":
                        {
                            "url": "data:image/png;base64," + meta.image_gen.image_base64
                        }
                    }
                ]
            tmps = json.dumps({"role": "assistant", "content": content})
            json_bytes_2 = tmps.encode('utf-8')
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
                logger.info("(%s) put llm message: %s", log_uid, tmps)
            except Exception as e:
                logger.error("(%s) error uploading llm message object: %s",
                             log_uid, e)
        except Exception as e:
            logger.error("NewMessage error: %s", e)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"ERROR: {e}")

        # for lh in logger.handlers:
        #     lh.flush()


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
        interceptors=[AuthInterceptor(SECRET_KEY)]  # Добавляем Interceptor
    )

    llm_pb2_grpc.add_LlmServicer_to_server(
        LlmServicer(), server
    )
    
    # Добавляем health check сервис
    health_servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    
    # Устанавливаем статус сервиса как HEALTHY (SERVING)
    health_servicer.set('', health_pb2.HealthCheckResponse.SERVING)  # Empty string for overall health
    health_servicer.set('llm.Llm', health_pb2.HealthCheckResponse.SERVING)
    
    server.add_secure_port("[::]:50051", server_creds)
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    # Проверка доступности docling API
    if DOCLING_ADDRESS:
        logger.info("Checking docling API health...")
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            is_healthy = loop.run_until_complete(
                check_docling_health(DOCLING_ADDRESS))
            loop.close()
            if not is_healthy:
                logger.error("Docling API health check failed")
                exit(1)
            else:
                logger.info("Docling API health check passed")
        except Exception as e:
            logger.error("Docling health check error: %s", e)
            exit(1)
    else:
        logger.error("Docling API address not set")
        exit(1)

    # Вывод конфигурации API (ключи скрыты)
    for key, value in ALL_API_VARS.items():
        logger.info("API config for %s:", key)
        for case_key, case_value in value.items():
            if case_key == "key":
                logger.info("  %s=***", case_key)
            else:
                logger.info("  %s=%s", case_key, case_value)
    # Проверка конфигурации API
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
    assert ALL_API_VARS["openaiimgen"]["base_url"], "openai base url is n/a"
    assert ALL_API_VARS["openaiimgen"]["key"], "openai api key is n/a"

    assert ALL_API_VARS["openaivlm"]["prices_url"], "openai prices url is n/a"
    assert ALL_API_VARS["openaivlm"]["model"], "openai model is n/a"
    assert ALL_API_VARS["openaivlm"]["base_url"], "openai base url is n/a"
    assert ALL_API_VARS["openaivlm"]["key"], "openai api key is n/a"

    # Скрэппинг цен (prepare.py)
    if GENERATE_CONFIG_WHEN == "always" or (
        GENERATE_CONFIG_WHEN == "missing" and not os.path.exists(CONFIG_PATH)):
        logger.info("Generating config...")
        try:
            subprocess.run(
                ["/root/.local/bin/uv", "run",
                 "--env-file", ".env", "prepare.py"],
                check=True
            )
        except Exception as e:
            logger.error("Config generation failed: %s", e)
            exit(1)
    try:
        with open(CONFIG_PATH) as f:
            config = json.load(f)
        if 'generated_at' not in config or not config['generated_at']:
            raise ValueError("Invalid config: missing 'generated_at'")
        for name, coef in config.get("prices_coefs", {}).items():
            if isinstance(coef, dict):
                # Модели с разделением на входные и выходные токены
                ALL_API_VARS[name]["price_coef_input"] = coef.get("input", 0)
                ALL_API_VARS[name]["price_coef_output"] = coef.get("output", 0)
                logger.info("Price coefficient for %s: input=%.10f, output=%.10f",
                           name, coef.get("input", 0), coef.get("output", 0))
            else:
                # Модели с единой ценой
                ALL_API_VARS[name]["price_coef"] = coef
                logger.info("Price coefficient for %s: %s", name, coef)
    except Exception as e:
        logger.error("Invalid config: %s", e)
        exit(1)
    # Информация о безопасности
    if SECRET_KEY:
        logger.info("Authorization ENABLED")
    else:
        logger.warning(
            "Authorization DISABLED - all requests will be accepted!")
    # Проверка доступности моделей
    logger.info("Checking available models...")
    check_arr = available_models(ALL_API_VARS["yandexai"]["base_url"],
                                 ALL_API_VARS["yandexai"]["key"],
                                 ALL_API_VARS["yandexai"]["folder"],
                                 WHITELIST_REGEX_TEXT2TEXT,
                                 BLACKLIST_REGEX_TEXT2TEXT)
    update_model_to_api(check_arr, "yandexai")
    check_arr.append(ALL_API_VARS["openaivlm"]["model"])
    update_model_to_api([ALL_API_VARS["openaivlm"]["model"]], "openaivlm")
    check_arr.append(ALL_API_VARS["deepseek"]["model"])
    update_model_to_api([ALL_API_VARS["deepseek"]["model"]], "deepseek")
    check_arr.append(ALL_API_VARS["openaimini"]["model"])
    update_model_to_api([ALL_API_VARS["openaimini"]["model"]], "openaimini")
    check_arr_speech=available_models(ALL_API_VARS["openai"]["base_url"],
                                      ALL_API_VARS["openai"]["key"], None,
                                      WHITELIST_REGEX_SPEECH2TEXT,
                                      BLACKLIST_REGEX_SPEECH2TEXT)
    if ALL_API_VARS["yandexai"]["model"] not in check_arr:
        logger.error("Text2Text model %s not found",
                     ALL_API_VARS['yandexai']['model'])
    elif ALL_API_VARS["openai"]["model"] not in check_arr_speech:
        logger.error("Speech2Text model %s not found",
                     ALL_API_VARS['openai']['model'])
    else:
        logger.info("Starting server...")
        serve()
