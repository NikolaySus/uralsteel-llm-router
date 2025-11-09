"""gRPC сервер для взаимодействия с языковой моделью."""

from datetime import datetime
from concurrent import futures
import logging
import os

from google.protobuf import empty_pb2
import grpc
from openai import OpenAI

import llm_pb2
import llm_pb2_grpc


# Команды для генерации llm_pb2.py и llm_pb2_grpc.py из llm.proto:
# uv run -m grpc_tools.protoc -I.\uralsteel-grpc-api\llm\ --python_out=.
# --grpc_python_out=. llm.proto
#
# Запуск vLLM сервера с моделью DeepSeek-R1-Distill-Qwen-1.5B для тестирования:
# docker run -it -v C:/Users/gorku/.cache/huggingface:/root/.cache/huggingface
# --runtime nvidia --gpus all
# --env "HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN" -p 8008:8000
# --ipc=host vllm/vllm-openai:latest
# --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
# --reasoning-parser deepseek_r1 --gpu-memory-utilization 0.8
# --max-model-len 65536 --api-key uralsteel
#
MODEL = os.getenv('MODEL', "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
BASE_URL = os.getenv('BASE_URL', "http://127.0.0.1:8008/v1")
API_KEY = os.getenv('API_KEY', "uralsteel")


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
                        f"{datetime.now().strftime("%Y-%m-%dT%H:%M:%S")}. "
                        "Note that current date and time are relevant only "
                        "for last message, previous ones could be sent a "
                        "long time ago."
                    )
                }
            ]
            if history:
                for message in history:
                    messages.append({
                        "role": llm_pb2.Role.Name(message.role), "content": message.body
                    })
            messages.append({"role": "user", "content": user_message})
            client = OpenAI(
                base_url=BASE_URL,
                api_key=API_KEY,
            )
            stream = client.responses.create(
                model=MODEL,
                input=messages,
                stream=True,
            )
            delta_id = 0
            type = None
            text = None
            number = None
            for event in stream:
                if event.type == "response.reasoning_text.delta":
                    number = event.sequence_number + delta_id
                    type = llm_pb2.NewMessageResponseType.middle
                    text = event.delta
                elif event.type == "response.output_text.delta":
                    number = event.sequence_number + delta_id
                    type = llm_pb2.NewMessageResponseType.middle
                    text = event.delta
                elif event.type == "response.reasoning_text.done":
                    number = event.sequence_number + delta_id
                    delta_id = event.sequence_number - 2
                    type = llm_pb2.NewMessageResponseType.last
                    text = event.text
                elif event.type == "response.output_text.done":
                    number = event.sequence_number + delta_id
                    delta_id = event.sequence_number - 2
                    type = llm_pb2.NewMessageResponseType.last
                    text = event.text
                elif event.type == "response.output_item.added":
                    delta_id -= event.sequence_number
                    number = event.sequence_number + delta_id
                    type = llm_pb2.NewMessageResponseType.first
                    text = event.item.type
                    delta_id -= 1
                else:
                    continue
                yield llm_pb2.NewMessageResponse(
                    type=type, body=text, msg_number=number)
        except Exception as e:
            print(e)


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
    print(f"Globals:\nMODEL={MODEL}\nBASE_URL={BASE_URL}\nAPI_KEY={API_KEY}")
    logging.basicConfig()
    serve()
