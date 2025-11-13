r"""
gRPC сервер для взаимодействия с языковой моделью.
Команда для генерации llm_pb2.py и llm_pb2_grpc.py из llm.proto:
uv run -m grpc_tools.protoc -I.\uralsteel-grpc-api\llm\ --python_out=.
--grpc_python_out=. llm.proto
"""

from datetime import datetime
from concurrent import futures
from io import BytesIO
import logging
import os

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


MODEL = os.getenv('MODEL', "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
BASE_URL = os.getenv('BASE_URL', "http://127.0.0.1:8008/v1")
API_KEY = os.getenv('API_KEY', "uralsteel")
CLOUD_FOLDER = os.getenv('CLOUD_FOLDER', "uralsteel")
SPEECH2TEXT_OPEN_AI = os.environ.get('SPEECH2TEXT_OPEN_AI', '')
SPEECH2TEXT_MODEL = os.environ.get('SPEECH2TEXT_MODEL', '')
BASE_URL_OPEN_AI = os.environ.get('BASE_URL_OPEN_AI', '')



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


def transcribe_audio_buffer(audio_buffer: BytesIO,
                            speech2text_override: str = None):
    """Транскрибирует собранный audio_buffer и возвращает кортеж
    (transcription_text, TranscribeResponseType proto).

    Использует speech2text_override если задан, иначе SPEECH2TEXT_MODEL.
    Бросает ValueError, если конфигурация SPEECH2TEXT не задана.
    """
    if audio_buffer.tell() == 0:
        return None, None

    if not SPEECH2TEXT_OPEN_AI or (
        not SPEECH2TEXT_MODEL or not BASE_URL_OPEN_AI):
        raise ValueError(
            "Speech-to-text сервис не настроен: "
            "SPEECH2TEXT_OPEN_AI, SPEECH2TEXT_MODEL, BASE_URL_OPEN_AI"
        )

    audio_buffer.seek(0)
    # Некоторым API требуется имя у файла
    audio_buffer.name = "audio.mp3"

    # Используем переданную модель или берём по умолчанию
    if speech2text_override:
        model_to_use = speech2text_override
    else:
        model_to_use = SPEECH2TEXT_MODEL

    transcription = OpenAI(
        base_url=BASE_URL_OPEN_AI,
        api_key=SPEECH2TEXT_OPEN_AI,
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
        datetime=datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    )
    return text, proto


def build_messages_from_history(history, user_message: str,
                                text2text_override: str = None):
    """Собирает список сообщений для LLM на основании history и user_message.
    
    Использует text2text_override если задан, иначе MODEL.
    """
    if text2text_override:
        model_to_use = text2text_override
    else:
        model_to_use = MODEL
    messages = [
        {
            "role": "system",
            "content": (
                "You are helpfull and highly skilled LLM-powered "
                "assistant that always follows best practices. "
                f"The base LLM is {model_to_use}. Current date and time: "
                f"{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}. "
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
    if (finish_reason == "stop") or (
        total_tokens is not None and (
            prompt_tokens is not None and completion_tokens is not None)):
        return llm_pb2.NewMessageResponse(
            complete=llm_pb2.CompleteResponseType(
                prompt_tokens=(prompt_tokens or 0),
                completion_tokens=(completion_tokens or 0),
                total_tokens=(total_tokens or 0),
                datetime=datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
            )
        )
    # Если один из delta content есть, создаём GenerateResponseType
    elif delta_content is not None or delta_reasoning_content is not None:
        return llm_pb2.NewMessageResponse(
            generate=llm_pb2.GenerateResponseType(
                content=(delta_content or ""),
                reasoning_content=(delta_reasoning_content or ""),
                datetime=datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
            )
        )
    else:
        # Не удалось разобрать часть ответа — предупреждаем и возвращаем None
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
                models=available_models(BASE_URL, API_KEY, CLOUD_FOLDER))
        except Exception as e:
            print(f"ERROR getting text2text models: {e}")
            context.set_details(f"ERROR getting text2text models: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return llm_pb2.ModelsListResponse(models=[])

    def AvailableModelsSpeech2Text(self, request, context):
        """Получить список доступных Speech2Text моделей."""
        try:
            return llm_pb2.ModelsListResponse(
                models=available_models(BASE_URL_OPEN_AI,
                                        SPEECH2TEXT_OPEN_AI, None))
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
                user_message, transcribe_proto = transcribe_audio_buffer(
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
                model_to_use = MODEL

            # Отправка запроса в OpenAI API на генерацию ответа, если
            # пользователь не отменял запрос
            if context.is_active():
                try:
                    response = OpenAI(
                        base_url=BASE_URL,
                        api_key=API_KEY,
                        project=CLOUD_FOLDER,
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
                        resp = responses_from_llm_chunk(chunk)
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
    """Запуск gRPC сервера."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    llm_pb2_grpc.add_LlmServicer_to_server(
        LlmServicer(), server
    )
    server.add_insecure_port("[::]:50051")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    print(f"Globals:\nMODEL={MODEL}\nBASE_URL={BASE_URL}\n"
          f"API_KEY={API_KEY}\nCLOUD_FOLDER={CLOUD_FOLDER}\n"
          f"SPEECH2TEXT_OPEN_AI={SPEECH2TEXT_OPEN_AI}\n"
          f"SPEECH2TEXT_MODEL={SPEECH2TEXT_MODEL}\n"
          f"BASE_URL_OPEN_AI={BASE_URL_OPEN_AI}")
    print("Checking available models...")
    check_arr = available_models(BASE_URL, API_KEY, CLOUD_FOLDER)
    check_arr_speech = available_models(BASE_URL_OPEN_AI,
                                        SPEECH2TEXT_OPEN_AI, None)
    if MODEL not in check_arr:
        print(f"ERROR: Text2Text model {MODEL} not found in available models!")
    elif SPEECH2TEXT_MODEL and (SPEECH2TEXT_MODEL not in check_arr_speech):
        print(f"ERROR: Speech2Text model {SPEECH2TEXT_MODEL} not found in "
              "available models!")
    else:
        print(f"Default Text2Text model set to {MODEL}\n"
              f"Default Speech2Text model set to {SPEECH2TEXT_MODEL}\n"
              "Starting server...")
        logging.basicConfig()
        serve()
