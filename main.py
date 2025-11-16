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
import subprocess

from google.protobuf import empty_pb2
import grpc
from openai import OpenAI

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


# ============================================================================
# АВТОРИЗАЦИЯ - Interceptor для проверки доступа
# ============================================================================

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
            # Создаём обработчик ошибки
            abort_handler = grpc.unary_unary_rpc_method_handler(
                lambda request, context: context.abort(
                    grpc.StatusCode.UNAUTHENTICATED,
                    "Invalid or missing authorization"
                )
            )
            return abort_handler(None, handler_call_details)

        # Секретный ключ верен - пропускаем запрос дальше
        return continuation(handler_call_details)


def available_models(base_url: str, api_key: str, project: str):
    """Возвращает список доступных моделей в виде списка строк."""
    models_list = OpenAI(
        base_url=base_url,
        api_key=api_key,
        project=project,
    ).models.list()
    check_arr = []
    for model in models_list.data:
        check_arr.append(model.id)
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
        return None, None, None

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
        datetime=datetime.now().strftime(DATETIME_FORMAT)
    )
    return text, proto, duration


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
    messages = [
        {
            "role": "system",
            "content": (
                "You are helpfull and highly skilled LLM-powered "
                "assistant that always follows best practices. "
                f"The base LLM is {model_to_use}. Current date and time: "
                f"{datetime.now().strftime(DATETIME_FORMAT)}. "
                "Note that current date and time are relevant only "
                "for last message, previous ones could be sent a long "
                "time ago. Respond in the same language as the user."
            )
        }
    ]
    if history:
        for message in history:
            messages.append({
                "role": llm_pb2.Role.Name(message.role),
                "content": message.body
            })
    messages.append({"role": "user", "content": user_message})
    return messages


def responses_from_llm_chunk(chunk, duration):
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
    if (finish_reason == "stop") or (
        total_tokens is not None and (
            prompt_tokens is not None and completion_tokens is not None)):
        return llm_pb2.NewMessageResponse(
            complete=llm_pb2.CompleteResponseType(
                prompt_tokens=(prompt_tokens or 0),
                completion_tokens=(completion_tokens or 0),
                total_tokens=(total_tokens or 0),
                expected_cost_usd=ALL_API_VARS["openai"]["price_coef"] *
                                   (duration or 0) +
                                   ALL_API_VARS["yandexai"]["price_coef"] *
                                   (total_tokens or 0),
                datetime=datetime.now().strftime(DATETIME_FORMAT)
            )
        )
    # Если один из delta content есть, создаём GenerateResponseType
    elif delta_content is not None or delta_reasoning_content is not None:
        return llm_pb2.NewMessageResponse(
            generate=llm_pb2.GenerateResponseType(
                content=(delta_content or ""),
                reasoning_content=(delta_reasoning_content or ""),
                datetime=datetime.now().strftime(DATETIME_FORMAT)
            )
        )
    else:
        # Не удалось разобрать часть ответа
        print(f"WARN: {chunk}")
        return None


class LlmServicer(llm_pb2_grpc.LlmServicer):
    """Реализация LLM сервиса."""

    def Ping(self, request, context):
        """Простой метод для проверки работоспособности сервиса."""
        return empty_pb2.Empty()

    def AvailableModelsText2Text(self, request, context):
        """Получить список доступных Text2Text моделей."""
        try:
            return llm_pb2.ModelsListResponse(
                models=available_models(ALL_API_VARS["yandexai"]["base_url"],
                                        ALL_API_VARS["yandexai"]["key"],
                                        ALL_API_VARS["yandexai"]["folder"]))
        except Exception as e:
            print(f"ERROR getting text2text models: {e}")
            context.set_details(f"ERROR getting text2text models: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return llm_pb2.ModelsListResponse(models=[])

    def AvailableModelsSpeech2Text(self, request, context):
        """Получить список доступных Speech2Text моделей."""
        try:
            return llm_pb2.ModelsListResponse(
                models=available_models(ALL_API_VARS["openai"]["base_url"],
                                        ALL_API_VARS["openai"]["key"], None))
        except Exception as e:
            print(f"ERROR getting speech2text models: {e}")
            context.set_details(f"ERROR getting speech2text models: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return llm_pb2.ModelsListResponse(models=[])

    def NewMessage(self, request_iter, context):
        """Метод для отправки сообщения языковой модели и получения ответа.
        
        request_iter — поток запросов (stream), может содержать:
        - Одно сообщение с msg (текстовое сообщение)
        - Несколько сообщений с mp3_chunk (потоковое аудио)
        
        История чата передаётся в одном из сообщений. Опциональные поля
        text2text_model и speech2text_model переопределяют модели по умолчанию.
        """
        try:
            user_message = None
            history = None
            audio_buffer = BytesIO()
            text2text_override = None
            speech2text_override = None
            duration = None

            # Собираем все данные из потока запросов
            for request in request_iter:
                # Извлекаем историю из первого непустого сообщения
                if history is None and request.history:
                    history = request.history

                # Извлекаем опциональные модели из первого сообщения
                if text2text_override is None and request.text2text_model:
                    text2text_override = request.text2text_model
                if speech2text_override is None and request.speech2text_model:
                    speech2text_override = request.speech2text_model

                # Проверяем какой тип payload пришел
                if request.HasField("msg"):
                    # Текстовое сообщение
                    user_message = request.msg
                elif request.HasField("mp3_chunk"):
                    # Собираем аудиочанк
                    audio_buffer.write(request.mp3_chunk)

            # Если пришло аудио, транскрибируем его
            if audio_buffer.tell() > 0:
                user_message, transcribe_proto, duration = transcribe_audio(
                    audio_buffer,
                    speech2text_override
                )
                if transcribe_proto is not None:
                    yield llm_pb2.NewMessageResponse(
                        transcribe=transcribe_proto)

            # Если сообщение пусто, ошибка
            if not user_message:
                raise ValueError("Ни текст сообщения, ни аудио не получены")

            # История по умолчанию пуста
            if history is None:
                history = []

            # Сборка контекста и отправка запроса на генерацию ответа
            messages = build_messages_from_history(history, user_message,
                                                   text2text_override)

            # Определяем модель для запроса
            if text2text_override:
                model_to_use = text2text_override
            else:
                model_to_use = ALL_API_VARS["yandexai"]["model"]

            # Отправка запроса в OpenAI API на генерацию ответа, если
            # пользователь не отменял запрос
            if context.is_active():
                try:
                    response = OpenAI(
                        base_url=ALL_API_VARS["yandexai"]["base_url"],
                        api_key=ALL_API_VARS["yandexai"]["key"],
                        project=ALL_API_VARS["yandexai"]["folder"],
                    ).chat.completions.create(
                        model=model_to_use,
                        messages=messages,
                        max_tokens=2000,
                        temperature=0.3,
                        stream=True
                    )
                    for chunk in response:
                        if context.is_active() is False:
                            print("Client cancelled, stopping stream.")
                            break
                        # Преобразуем chunk в один ответ protobuf (или None)
                        resp = responses_from_llm_chunk(chunk, duration)
                        if resp is not None:
                            yield resp
                except Exception as e:
                    print(f"STREAM ERROR: {e}")
                    context.set_code(grpc.StatusCode.INTERNAL)
                    context.set_details(f"STREAM ERROR: {e}")
                finally:
                    response.response.close()
            else:
                print("Client cancelled, stopping stream.")
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
    assert ALL_API_VARS["yandexai"]["prices_url"], "yandexai prices url is n/a"
    assert ALL_API_VARS["yandexai"]["base_url"], "yandexai base url is n/a"
    assert ALL_API_VARS["yandexai"]["key"], "yandexai api key is n/a"
    assert ALL_API_VARS["yandexai"]["model"], "yandexai model is n/a"
    assert ALL_API_VARS["yandexai"]["folder"], "yandexai folder is n/a"
    assert ALL_API_VARS["openai"]["prices_url"], "openai prices url is n/a"
    assert ALL_API_VARS["openai"]["base_url"], "openai base url is n/a"
    assert ALL_API_VARS["openai"]["key"], "openai api key is n/a"
    assert ALL_API_VARS["openai"]["model"], "openai model is n/a"
    # Скрэппинг цен (prepare.py)
    try:
        subprocess.run(
            ["/root/.local/bin/uv", "run", "--env-file", ".env", "prepare.py"],
            check=True
        )
        with open(CONFIG_PATH) as f:
            config = json.load(f)
            print(f"Config generated at: {config.get('generated_at', 'n/a')}")
            for name, coef in config.get("prices_coefs", {}).items():
                ALL_API_VARS[name]["price_coef"] = coef
                print(f"Price coef for {name}: {coef}")
    except Exception as e:
        print(f"ERROR: config gen fail: {e}")
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
                                 ALL_API_VARS["yandexai"]["folder"])
    check_arr_speech = available_models(ALL_API_VARS["openai"]["base_url"],
                                        ALL_API_VARS["openai"]["key"], None)
    if ALL_API_VARS["yandexai"]["model"] not in check_arr:
        print(f"ERROR: Text2Text model {ALL_API_VARS["yandexai"]["model"]} "
              "not found in available models!")
    elif ALL_API_VARS["openai"]["model"] not in check_arr_speech:
        print(f"ERROR: Speech2Text model {ALL_API_VARS["openai"]["model"]} "
              "not found in available models!")
    else:
        print("Starting server...")
        logging.basicConfig()
        serve()
