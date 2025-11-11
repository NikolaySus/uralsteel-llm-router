r"""
gRPC сервер для взаимодействия с языковой моделью.
Команда для генерации llm_pb2.py и llm_pb2_grpc.py из llm.proto:
uv run -m grpc_tools.protoc -I.\uralsteel-grpc-api\llm\ --python_out=.
--grpc_python_out=. llm.proto
"""

from datetime import datetime
from concurrent import futures
import logging
import os

from google.protobuf import empty_pb2
import grpc
from openai import OpenAI

import llm_pb2
import llm_pb2_grpc


MODEL = os.getenv('MODEL', "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
BASE_URL = os.getenv('BASE_URL', "http://127.0.0.1:8008/v1")
API_KEY = os.getenv('API_KEY', "uralsteel")
CLOUD_FOLDER = os.getenv('CLOUD_FOLDER', "uralsteel")


class LlmServicer(llm_pb2_grpc.LlmServicer):
    """Реализация LLM сервиса."""

    def Ping(self, request, context):
        """Простой метод для проверки работоспособности сервиса."""
        return empty_pb2.Empty()

    def NewMessage(self, request, context):
        """Метод для отправки сообщения языковой модели и получения ответа."""
        try:
            user_message = request.msg
            history = request.history
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are helpfull and highly skilled LLM-powered "
                        "assistant that always follows best practices. "
                        f"The base LLM is {MODEL}. Current date and time: "
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
            response = OpenAI(
                base_url=BASE_URL,
                api_key=API_KEY,
                project=CLOUD_FOLDER,
            ).chat.completions.create(
                model=MODEL,
                messages=messages,
                max_tokens=2000,
                temperature=0.3,
                stream=True
            )
            try:
                for chunk in response:
                    if context.is_active() is False:
                        print("Client cancelled, stopping stream.")
                        break
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
                    # Если есть usage или конечный сигнал, отправка CompleteResponseType
                    if (finish_reason == "stop") or (
                        total_tokens is not None and (prompt_tokens is not None and completion_tokens is not None)
                    ):
                        yield llm_pb2.NewMessageResponse(
                            complete=llm_pb2.CompleteResponseType(
                                prompt_tokens=(prompt_tokens or 0),
                                completion_tokens=(completion_tokens or 0),
                                total_tokens=(total_tokens or 0),
                                datetime=datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
                            )
                        )
                    # Если один из delta content есть, отправка GenerateResponseType
                    elif delta_content is not None or delta_reasoning_content is not None:
                        yield llm_pb2.NewMessageResponse(
                            generate=llm_pb2.GenerateResponseType(
                                content=(delta_content or ""),
                                reasoning_content=(delta_reasoning_content or ""),
                                datetime=datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
                            )
                        )
                    # Вывод предупреждения, если не удалось разобрать часть ответа
                    else:
                        print(f"WARN : {chunk}")
            except Exception as e:
                print(f"STREAM ERROR: {e}")
            finally:
                response.response.close()
        except Exception as e:
            print(f"ERROR: {e}")


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
          f"API_KEY={API_KEY}\nCLOUD_FOLDER={CLOUD_FOLDER}")
    client = OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
        project=CLOUD_FOLDER,
    )
    models_list = client.models.list()
    print("Available models:")
    check_arr = []
    for model in models_list.data:
        print(model.id)
        check_arr.append(model.id)
    if MODEL not in check_arr:
        print(f"ERROR: Model {MODEL} not found in available models!")
    else:
        print(f"Model {MODEL} found in available models. Starting server...")
        logging.basicConfig()
        serve()
