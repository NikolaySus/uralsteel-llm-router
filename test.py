"""Интеграционные тесты для gRPC LLM сервиса с авторизацией."""

import base64
import json
import os
import tempfile
import unittest

from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2
import grpc

import llm_pb2
import llm_pb2_grpc


# =============================================================================
# КОНСТАНТЫ ДЛЯ ТЕСТОВ
# =============================================================================

SERVER_ADDRESS = os.environ.get('SERVER_ADDRESS', 'localhost:50051')
SECRET_KEY = os.environ.get('SECRET_KEY', '')
TEST_MESSAGE = "Привет, как дела?"
TEST_MESSAGE_WITH_HISTORY = "Подскажи какой сериал посмотреть интересненький."
TEST_MESSAGE_IMAGE_GEN = "Создай изображение пейзажа с горами и озером."
TEST_MP3_FILE = "serial.mp3"  # Путь к mp3 файлу

# История для тестов с историей
TEST_HISTORY = [
    llm_pb2.Message(
        role=llm_pb2.Role.user,
        body="Привет, как дела?"
    ),
    llm_pb2.Message(
        role=llm_pb2.Role.assistant,
        body="Привет! Дела отлично, спасибо за вопрос!"
    ),
]

# Чтение доверенных сертификатов
with open("ca.crt", "rb") as f:
    TRUSTED_CERTS = f.read()
CREDS = grpc.ssl_channel_credentials(root_certificates=TRUSTED_CERTS)


# =============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =============================================================================


def open_image_from_base64(b64string, file_extension=".png"):
    """
    Decodes a Base64 string, saves it as an image file, and opens it on Windows.

    Args:
        b64string (str or bytes): The Base64 encoded image data.
        file_extension (str): The desired file extension for the image (e.g., ".png", ".jpg").
    """
    try:
        # 1. Decode the Base64 string
        # Ensure the input is bytes-like for b64decode
        if isinstance(b64string, str):
            decoded_bytes = base64.b64decode(b64string)
        else:
            decoded_bytes = base64.b64decode(b64string)

        # 2. Save the bytes to a temporary file
        # Create a temporary file to store the image
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            temp_file.write(decoded_bytes)
            temp_file_path = temp_file.name

        # 3. Open the file using the default system viewer
        os.startfile(temp_file_path)
        print(f"Image saved to {temp_file_path} and opened.")

    except Exception as e:
        print(f"Error opening image from Base64: {e}")
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.remove(temp_file_path) # Clean up temp file in case of error


def get_metadata():
    """Возвращает metadata с авторизацией для защищённых методов."""
    if SECRET_KEY:
        return [('authorization', f'Bearer {SECRET_KEY}')]
    return []


def process_llm_responses(responses):
    """Обрабатывает поток ответов от LLM и возвращает результаты.
    
    Args:
        responses: Поток ответов от LLM (iterator NewMessageResponse)
    
    Returns:
        Кортеж (has_trans, has_gen, has_complete, 
                transcription, content, reasoning,
                function_calls_info)
                где function_calls_info это словарь словарей с информацией 
                о вызовах функций
    """
    has_trans = False
    has_gen = False
    has_complete = False
    transcription = ""
    content_parts = []
    reasoning_parts = []
    function_calls = {}  # {id: {"name": str, "status": str, "arguments": str}}

    for response in responses:
        if response.HasField("transcribe"):
            has_trans = True
            trans = response.transcribe
            transcription = trans.transcription
            expected_cost_usd = trans.expected_cost_usd
            print(f"usd cost: {expected_cost_usd}")
            print(f"✓ Транскрипция: {transcription}")
            if trans.duration:
                print(f"  Длительность: {trans.duration}s")

        elif response.HasField("generate"):
            has_gen = True
            gen = response.generate
            if gen.content:
                content_parts.append(gen.content)
                print(f"Content: {gen.content}", flush=True)
            if gen.reasoning_content:
                reasoning_parts.append(gen.reasoning_content)
                print(f"Reasoning: {gen.reasoning_content}", flush=True)

        elif response.HasField("complete"):
            has_complete = True
            comp = response.complete
            print("\n✓ Завершено. Токены: "
                  f"prompt={comp.prompt_tokens}, "
                  f"completion={comp.completion_tokens}, "
                  f"total={comp.total_tokens}, "
                  f"expected_cost_usd={comp.expected_cost_usd}")

        elif response.HasField("function_call_added"):
            func_call = response.function_call_added
            func_id = func_call.id
            func_name = func_call.name
            function_calls[func_id] = {
                "name": func_name,
                "status": "added",
                "arguments": ""
            }

        elif response.HasField("function_call_delta"):
            func_call = response.function_call_delta
            func_id = func_call.id
            content = func_call.content
            if func_id in function_calls:
                function_calls[func_id]["arguments"] += content
                function_calls[func_id]["status"] = "delta"

        elif response.HasField("function_call_done"):
            func_call = response.function_call_done
            func_id = func_call.id
            arguments = func_call.arguments
            if func_id in function_calls:
                function_calls[func_id]["arguments"] = arguments
                function_calls[func_id]["status"] = "done"

        elif response.HasField("function_call_complete"):
            func_call = response.function_call_complete
            func_id = func_call.id
            func_name = func_call.name
            arguments = func_call.arguments
            function_calls[func_id] = {
                "name": func_name,
                "status": "complete",
                "arguments": arguments
            }

        elif response.HasField("tool_metadata"):
            tool_meta = response.tool_metadata
            content = json.loads(tool_meta.content)
            items = list(content.items())
            if items[0][0] == "url_list":
                print(f"ToolMetadata: {items[0][1]}", flush=True)
            elif items[0][0] == "image_base64":
                print(f"ToolMetadata: {items[1]}", flush=True)
                open_image_from_base64(items[0][1])

    return (has_trans, has_gen, has_complete, transcription,
            "".join(content_parts), "".join(reasoning_parts),
            function_calls)


# =============================================================================
# ТЕСТЫ
# =============================================================================

class TestLlmService(unittest.TestCase):
    """Интеграционные тесты для gRPC LLM сервиса с авторизацией."""

    def test_01_ping(self):
        """Тест 1: Ping - публичный метод, БЕЗ авторизации."""
        print("")
        try:
            stub = llm_pb2_grpc.LlmStub(
                grpc.secure_channel(SERVER_ADDRESS, CREDS))
            # Ping НЕ требует авторизацию
            response = stub.Ping(google_dot_protobuf_dot_empty__pb2.Empty())
            print("✓ Ping успешен! (метод доступен без SECRET_KEY)")
            self.assertIsNotNone(response)
        except Exception as e:
            print(f"✗ Ping не прошел: {e}")
            self.fail(f"Ping failed: {e}")

    def test_02_available_models_text2text(self):
        """Тест 2: AvailableModelsText2Text - требует авторизацию.
           Получить список доступных Text2Text моделей."""
        print("")
        try:
            stub = llm_pb2_grpc.LlmStub(
                grpc.secure_channel(SERVER_ADDRESS, CREDS))
            # Передаём авторизационный заголовок
            response = stub.AvailableModelsText2Text(
                google_dot_protobuf_dot_empty__pb2.Empty(),
                metadata=get_metadata()
            )

            print("✓ Успешно получен список моделей")
            print(f"Количество моделей: {len(response.strings)}")
            if response.strings:
                print("Список моделей:")
                for model in response.strings:
                    print(f"  - {model}")

            self.assertIsNotNone(response)
            self.assertGreater(len(response.strings), 0,
                               "Список моделей не должен быть пустым")

        except Exception as e:
            print(f"✗ Тест не прошел: {e}")
            self.fail(f"AvailableModelsText2Text failed: {e}")

    def test_03_available_models_speech2text(self):
        """Тест 3: AvailableModelsSpeech2Text - требует авторизацию.
           Получить список доступных Speech2Text моделей."""
        print("")
        try:
            stub = llm_pb2_grpc.LlmStub(
                grpc.secure_channel(SERVER_ADDRESS, CREDS))
            # Передаём авторизационный заголовок
            response = stub.AvailableModelsSpeech2Text(
                google_dot_protobuf_dot_empty__pb2.Empty(),
                metadata=get_metadata()
            )

            print("✓ Успешно получен ответ")
            print(f"Количество моделей: {len(response.strings)}")
            if response.strings:
                print("Список моделей:")
                for model in response.strings:
                    print(f"  - {model}")

            self.assertIsNotNone(response)
            self.assertGreater(len(response.strings), 0,
                               "Список Speech2Text моделей должен быть непуст")

        except Exception as e:
            print(f"✗ Тест не прошел: {e}")
            self.fail(f"AvailableModelsSpeech2Text failed: {e}")

    def test_04_new_message_text_no_history(self):
        """Тест 4: NewMessage с текстовым сообщением без истории -
           требует авторизацию."""
        print(f"\nСообщение: {TEST_MESSAGE}")

        stub = llm_pb2_grpc.LlmStub(grpc.secure_channel(SERVER_ADDRESS, CREDS))

        def request_generator():
            yield llm_pb2.NewMessageRequest(msg=TEST_MESSAGE)

        try:
            # Передаём авторизационный заголовок
            responses = stub.NewMessage(request_generator(),
                                        metadata=get_metadata())

            _, has_gen, has_complete, __, content, reasoning, fc = \
                process_llm_responses(responses)

            print("\nРезультаты:")
            print(f"  - Получены GenerateResponseType: {has_gen}")
            print(f"  - Получены CompleteResponseType: {has_complete}")
            if content:
                print(f"  - Content: {content}")
            if reasoning:
                print(f"  - Reasoning: {reasoning}")
            if fc:
                print(f"  - Вызовы функций: {len(fc)}")
                for func_id, func_info in fc.items():
                    print(f"    * {func_info['name']} (id={func_id}): "
                          f"status={func_info['status']}")

            self.assertTrue(has_gen or has_complete,
                            "Не получены ответы от сервера")
            self.assertTrue(has_complete,
                            "Не получен CompleteResponseType с статистикой")

        except Exception as e:
            print(f"✗ Тест не прошел: {e}")
            self.fail(f"NewMessage text no history failed: {e}")

    def test_05_new_message_text_with_history(self):
        """Тест 5: NewMessage с текстовым сообщением и историей -
           требует авторизацию."""
        print(f"\nИстория: {len(TEST_HISTORY)} сообщений")
        print(f"Новое сообщение: {TEST_MESSAGE_WITH_HISTORY}")

        stub = llm_pb2_grpc.LlmStub(grpc.secure_channel(SERVER_ADDRESS, CREDS))

        def request_generator():
            yield llm_pb2.NewMessageRequest(
                msg=TEST_MESSAGE_WITH_HISTORY,
                history=TEST_HISTORY
            )

        try:
            # Передаём авторизационный заголовок
            responses = stub.NewMessage(request_generator(),
                                        metadata=get_metadata())

            _, has_gen, has_complete, __, content, reasoning, fc = \
                process_llm_responses(responses)

            print("\nРезультаты:")
            print(f"  - Получены GenerateResponseType: {has_gen}")
            print(f"  - Получены CompleteResponseType: {has_complete}")
            if content:
                print(f"  - Content: {content}")
            if reasoning:
                print(f"  - Reasoning: {reasoning}")
            if fc:
                print(f"  - Вызовы функций: {len(fc)}")
                for func_id, func_info in fc.items():
                    print(f"    * {func_info['name']} (id={func_id}): "
                          f"status={func_info['status']}")

            self.assertTrue(has_gen or has_complete,
                            "Не получены ответы от сервера")
            self.assertTrue(has_complete,
                            "Не получен CompleteResponseType с статистикой")

        except Exception as e:
            print(f"✗ Тест не прошел: {e}")
            self.fail(f"NewMessage text with history failed: {e}")

    def test_06_new_message_audio_no_history(self):
        """Тест 6: NewMessage с потоком mp3 чанков без истории -
           требует авторизацию."""
        print(f"\nMP3 файл: {TEST_MP3_FILE}")

        if not os.path.exists(TEST_MP3_FILE):
            print(f"⚠ Файл {TEST_MP3_FILE} не найден, пропускаем тест")
            self.skipTest(f"MP3 file {TEST_MP3_FILE} not found")

        stub = llm_pb2_grpc.LlmStub(grpc.secure_channel(SERVER_ADDRESS, CREDS))

        def request_generator():
            chunk_size = 4096
            with open(TEST_MP3_FILE, 'rb') as mp3_file:
                while True:
                    chunk = mp3_file.read(chunk_size)
                    if not chunk:
                        break
                    yield llm_pb2.NewMessageRequest(mp3_chunk=chunk)

        try:
            # Передаём авторизационный заголовок
            responses = stub.NewMessage(request_generator(),
                                        metadata=get_metadata())

            has_trans, has_gen, has_complete, trans, content, reasoning, fc = \
                process_llm_responses(responses)

            print("\nРезультаты:")
            print(f"  - Получены TranscribeResponseType: {has_trans}")
            print(f"  - Получены GenerateResponseType: {has_gen}")
            print(f"  - Получены CompleteResponseType: {has_complete}")
            if trans:
                print(f"  - Транскрипция: {trans}")
            if content:
                print(f"  - Content: {content}")
            if reasoning:
                print(f"  - Reasoning: {reasoning}")
            if fc:
                print(f"  - Вызовы функций: {len(fc)}")
                for func_id, func_info in fc.items():
                    print(f"    * {func_info['name']} (id={func_id}): "
                          f"status={func_info['status']}")

            self.assertTrue(has_trans or has_gen or has_complete,
                            "Не получены ответы от сервера")

        except Exception as e:
            print(f"✗ Тест не прошел: {e}")
            self.fail(f"NewMessage audio no history failed: {e}")

    def test_07_new_message_audio_with_history(self):
        """Тест 7: NewMessage с потоком mp3 чанков и историей -
           требует авторизацию."""
        print(f"\nMP3 файл: {TEST_MP3_FILE}")
        print(f"История: {len(TEST_HISTORY)} сообщений")

        if not os.path.exists(TEST_MP3_FILE):
            print(f"⚠ Файл {TEST_MP3_FILE} не найден, пропускаем тест")
            self.skipTest(f"MP3 file {TEST_MP3_FILE} not found")

        stub = llm_pb2_grpc.LlmStub(grpc.secure_channel(SERVER_ADDRESS, CREDS))

        def request_generator():
            chunk_size = 4096
            first_chunk = True
            with open(TEST_MP3_FILE, 'rb') as mp3_file:
                while True:
                    chunk = mp3_file.read(chunk_size)
                    if not chunk:
                        break
                    if first_chunk:
                        yield llm_pb2.NewMessageRequest(
                            mp3_chunk=chunk,
                            history=TEST_HISTORY
                        )
                        first_chunk = False
                    else:
                        yield llm_pb2.NewMessageRequest(mp3_chunk=chunk)

        try:
            # Передаём авторизационный заголовок
            responses = stub.NewMessage(request_generator(),
                                        metadata=get_metadata())

            has_trans, has_gen, has_complete, trans, content, reasoning, fc = \
                process_llm_responses(responses)

            print("\nРезультаты:")
            print(f"  - Получены TranscribeResponseType: {has_trans}")
            print(f"  - Получены GenerateResponseType: {has_gen}")
            print(f"  - Получены CompleteResponseType: {has_complete}")
            if trans:
                print(f"  - Транскрипция: {trans}")
            if content:
                print(f"  - Content: {content}")
            if reasoning:
                print(f"  - Reasoning: {reasoning}")
            if fc:
                print(f"  - Вызовы функций: {len(fc)}")
                for func_id, func_info in fc.items():
                    print(f"    * {func_info['name']} (id={func_id}): "
                          f"status={func_info['status']}")

            self.assertTrue(has_trans or has_gen or has_complete,
                            "Не получены ответы от сервера")

        except Exception as e:
            print(f"✗ Тест не прошел: {e}")
            self.fail(f"NewMessage audio with history failed: {e}")

    def test_08_available_models_text2text_without_auth(self):
        """Тест 8: AvailableModelsText2Text ДОЛЖЕН ОТКЛОНИТЬ запрос БЕЗ
           авторизации (ошибка _InactiveRpcError)."""
        print("")
        try:
            stub = llm_pb2_grpc.LlmStub(
                grpc.secure_channel(SERVER_ADDRESS, CREDS))

            # Вызываем БЕЗ авторизационного заголовка (пустой metadata)
            stub.AvailableModelsText2Text(
                google_dot_protobuf_dot_empty__pb2.Empty(),
                metadata=[]  # Явно передаём пустую авторизацию
            )

            # Если мы сюда попали - ошибка, не должна была пройти авторизация
            print("✗ ОШИБКА: Запрос БЕЗ авторизации прошёл!")
            self.fail("AvailableModelsText2Text must return _InactiveRpcError")

        except grpc.RpcError as e:
            # Ожидаем ошибку _InactiveRpcError
            # На основе других тестов - используем str() для кода
            error_str = str(e)
            if "_InactiveRpcError" in error_str or (
                "Invalid or missing authorization" in error_str):
                print("✓ Запрос правильно отклонён с _InactiveRpcError")
                print(f"  Деталь ошибки: {error_str}")
                # Тест ПРОЙДЕН - правильно отклонил неавторизованный запрос
            else:
                print(f"✗ Получена неожиданная ошибка: {error_str}")
                self.fail("AvailableModelsText2Text must return "
                          f"_InactiveRpcError, получено: {error_str}")

    def test_09_available_tools(self):
        """Тест 9: AvailableTools - требует авторизацию.
           Получить список доступных инструментов/функций."""
        print("")
        try:
            stub = llm_pb2_grpc.LlmStub(
                grpc.secure_channel(SERVER_ADDRESS, CREDS))
            # Передаём авторизационный заголовок
            response = stub.AvailableTools(
                google_dot_protobuf_dot_empty__pb2.Empty(),
                metadata=get_metadata()
            )

            print("✓ Успешно получен список инструментов")
            print(f"Количество инструментов: {len(response.strings)}")
            if response.strings:
                print("Список инструментов:")
                for tool in response.strings:
                    print(f"  - {tool}")

            self.assertIsNotNone(response)
            self.assertGreater(len(response.strings), 0,
                               "Список инструментов не должен быть пустым")

        except Exception as e:
            print(f"✗ Тест не прошел: {e}")
            self.fail(f"AvailableTools failed: {e}")

    def test_10_new_message_text_with_websearch(self):
        """Тест 10: NewMessage с текстовым сообщением и function=websearch -
           требует авторизацию."""
        print(f"\nСообщение: {TEST_MESSAGE_WITH_HISTORY}")
        print("Функция: websearch")

        stub = llm_pb2_grpc.LlmStub(grpc.secure_channel(SERVER_ADDRESS, CREDS))

        def request_generator():
            yield llm_pb2.NewMessageRequest(
                msg=TEST_MESSAGE_WITH_HISTORY,
                function="websearch"
            )

        try:
            # Передаём авторизационный заголовок
            responses = stub.NewMessage(request_generator(),
                                        metadata=get_metadata())

            _, has_gen, has_complete, __, content, reasoning, fc = \
                process_llm_responses(responses)

            print("\nРезультаты:")
            print(f"  - Получены GenerateResponseType: {has_gen}")
            print(f"  - Получены CompleteResponseType: {has_complete}")
            if content:
                print(f"  - Content: {content}")
            if reasoning:
                print(f"  - Reasoning: {reasoning}")
            if fc:
                print(f"  - Вызовы функций: {len(fc)}")
                for func_id, func_info in fc.items():
                    print(f"    * {func_info['name']} (id={func_id}): "
                          f"status={func_info['status']}")

            self.assertTrue(has_gen or has_complete,
                            "Не получены ответы от сервера")
            self.assertTrue(has_complete,
                            "Не получен CompleteResponseType с статистикой")

        except Exception as e:
            print(f"✗ Тест не прошел: {e}")
            self.fail(f"NewMessage text with websearch failed: {e}")

    def test_11_new_message_text_with_image_gen(self):
        """Тест 11: NewMessage с текстовым сообщением и function=image_gen -
           требует авторизацию."""
        print(f"\nСообщение: {TEST_MESSAGE_IMAGE_GEN}")
        print("Функция: image_gen")

        stub = llm_pb2_grpc.LlmStub(grpc.secure_channel(SERVER_ADDRESS, CREDS))

        def request_generator():
            yield llm_pb2.NewMessageRequest(
                msg=TEST_MESSAGE_IMAGE_GEN,
                function="image_gen"
            )

        try:
            # Передаём авторизационный заголовок
            responses = stub.NewMessage(request_generator(),
                                        metadata=get_metadata())

            _, has_gen, has_complete, __, content, reasoning, fc = \
                process_llm_responses(responses)

            print("\nРезультаты:")
            print(f"  - Получены GenerateResponseType: {has_gen}")
            print(f"  - Получены CompleteResponseType: {has_complete}")
            if content:
                print(f"  - Content: {content}")
            if reasoning:
                print(f"  - Reasoning: {reasoning}")
            if fc:
                print(f"  - Вызовы функций: {len(fc)}")
                for func_id, func_info in fc.items():
                    print(f"    * {func_info['name']} (id={func_id}): "
                          f"status={func_info['status']}")

            self.assertTrue(has_gen or has_complete,
                            "Не получены ответы от сервера")
            self.assertTrue(has_complete,
                            "Не получен CompleteResponseType с статистикой")

        except Exception as e:
            print(f"✗ Тест не прошел: {e}")
            self.fail(f"NewMessage text with image_gen failed: {e}")


if __name__ == "__main__":
    # Информация о конфигурации
    print("\n" + "="*70)
    if SECRET_KEY:
        print(f"✓ SECRET_KEY установлен - авторизация ВКЛЮЧЕНА")
        print(f"  Защищённые методы требуют Bearer token")
    else:
        print(f"⚠ SECRET_KEY не установлен - авторизация отключена")
    print("="*70)
    unittest.main(verbosity=2)
