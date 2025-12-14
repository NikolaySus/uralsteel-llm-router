"""Интеграционные тесты для gRPC LLM сервиса с авторизацией."""

import base64
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
TEST_MESSAGE = "Придумай имя для персонажа D&D. Это хаотично-добрый молодой эльф маг (парень), который почти всю свою молодость провёл в лабораториях за магическими изысканиями. Конечный ответ должен содержать только один наилучший вариант имени."
TEST_MESSAGE_SERIAL = "Подскажи какой сериал посмотреть интересненький."
TEST_MESSAGE_WITH_HISTORY = "Хм, звучит знакомо, а не существовало ли главных героев с таким именем в фэнтэзийных книгах? Поищи в интернете"
TEST_MESSAGE_IMAGE_GEN = "Создай изображение пейзажа с горами и озером."
TEST_MP3_FILE = "serial.mp3"  # Путь к mp3 файлу

TEST_IMAGE_URL = os.environ.get('TEST_IMAGE_URL', '')
TEST_DOCX_URL = os.environ.get('TEST_DOCX_URL', '')
OPENAIVLM_MODEL = os.environ.get('INFERENCE_API_OPENAIVLM_MODEL', '')
TEST_MESSAGE_FILES_IMAGES = ("Напомни, пожалуйста, о чём файл и что в нём за "
                             "схемы? И почему мой преподаватель на лекции "
                             "показывал картинку, которую я прикрепил? "
                             "Ответь максимально кратко, я очень тороплюсь!")

# История для тестов с историей (храним uid сообщений)
TEST_HISTORY = []

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
    
    Копит content и reasoning_content по мере приходящих generate чанков
    и печатает их один раз в конце, после получения complete.
    
    Args:
        responses: Поток ответов от LLM (iterator NewMessageResponse)
    
    Returns:
        Кортеж (has_trans, has_gen, has_complete, 
                transcription, content, reasoning,
                function_calls_info, user_uid, llm_uid)
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
    user_uid = None
    llm_uid = None

    for response in responses:
        if response.HasField("generate"):
            has_gen = True
            gen = response.generate
            if gen.content:
                content_parts.append(gen.content)
            if gen.reasoning_content:
                reasoning_parts.append(gen.reasoning_content)

        elif response.HasField("complete"):
            has_complete = True
            comp = response.complete
            # Печатаем накопленные контент и reasoning перед статистикой
            final_content = "".join(content_parts)
            final_reasoning = "".join(reasoning_parts)
            if final_content:
                print(f"\nContent: {final_content}", flush=True)
            if final_reasoning:
                print(f"Reasoning: {final_reasoning}", flush=True)
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

        elif response.HasField("chat_name"):
            cn = response.chat_name
            print(
                f"Chat name: {cn.name} (tokens: prompt={cn.prompt_tokens}, "
                f"completion={cn.completion_tokens}, total={cn.total_tokens}, "
                f"expected_cost_usd={cn.expected_cost_usd})",
                flush=True
            )

        elif response.HasField("tool_metadata"):
            tool_meta = response.tool_metadata
            if tool_meta.HasField("websearch"):
                web_search = tool_meta.websearch
                for i in web_search.item:
                    print(f"ToolMetadata (WebSearch): '{i.title}' at {i.url}", flush=True)
            elif tool_meta.HasField("image_gen"):
                image_gen = tool_meta.image_gen
                print(f"ToolMetadata (ImageGen): cost={image_gen.expected_cost}", flush=True)
                if image_gen.image_base64:
                    open_image_from_base64(image_gen.image_base64)

        elif response.HasField("user_message_uid"):
            user_uid = response.user_message_uid
            print(f"User message uid: {user_uid}")
        elif response.HasField("llm_message_uid"):
            llm_uid = response.llm_message_uid
            print(f"LLM message uid: {llm_uid}")

    return (has_trans, has_gen, has_complete, transcription,
            "".join(content_parts), "".join(reasoning_parts),
            function_calls, user_uid, llm_uid)


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

        try:
            # Передаём авторизационный заголовок
            responses = stub.NewMessage(
                llm_pb2.NewMessageRequest(msg="Назови 5 марок стали", text2text_model="deepseek-chat"),
                metadata=get_metadata())

            _, has_gen, has_complete, __, content, reasoning, fc, user_uid, llm_uid = \
                process_llm_responses(responses)
            if user_uid:
                TEST_HISTORY.append(user_uid)
            if llm_uid:
                TEST_HISTORY.append(llm_uid)

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

        try:
            print(f"!!!! history = {TEST_HISTORY}")
            # Передаём авторизационный заголовок
            responses = stub.NewMessage(
                llm_pb2.NewMessageRequest(
                    msg=TEST_MESSAGE_WITH_HISTORY,
                    history=TEST_HISTORY
                ),
                metadata=get_metadata())

            _, has_gen, has_complete, __, content, reasoning, fc, user_uid, llm_uid = \
                process_llm_responses(responses)
            # if user_uid:
            #     TEST_HISTORY.append(user_uid)
            # if llm_uid:
            #     TEST_HISTORY.append(llm_uid)

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
    comment = '''
    def test_06_transcribe_audio_no_history(self):
        """Тест 6: Transcribe с потоком mp3 чанков без истории -
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
                    yield llm_pb2.TranscribeRequest(mp3_chunk=chunk)

        try:
            resp = stub.Transcribe(request_generator(), metadata=get_metadata())
            self.assertTrue(resp.HasField('transcribe'))
            print(f"✓ Транскрипция: {resp.transcribe.transcription}")

        except Exception as e:
            print(f"✗ Тест не прошел: {e}")
            self.fail(f"Transcribe audio no history failed: {e}")

    def test_07_transcribe_audio_with_model_override(self):
        """Тест 7: Transcribe с потоком mp3 чанков и указанием модели -
           требует авторизацию."""
        print(f"\nMP3 файл: {TEST_MP3_FILE}")

        if not os.path.exists(TEST_MP3_FILE):
            print(f"⚠ Файл {TEST_MP3_FILE} не найден, пропускаем тест")
            self.skipTest(f"MP3 file {TEST_MP3_FILE} not found")

        stub = llm_pb2_grpc.LlmStub(grpc.secure_channel(SERVER_ADDRESS, CREDS))

        def request_generator():
            chunk_size = 4096
            sent_model = False
            with open(TEST_MP3_FILE, 'rb') as mp3_file:
                while True:
                    chunk = mp3_file.read(chunk_size)
                    if not chunk:
                        break
                    if not sent_model and os.environ.get('INFERENCE_API_OPENAI_MODEL', ''):
                        yield llm_pb2.TranscribeRequest(
                            mp3_chunk=chunk,
                            speech2text_model=os.environ['INFERENCE_API_OPENAI_MODEL']
                        )
                        sent_model = True
                    else:
                        yield llm_pb2.TranscribeRequest(mp3_chunk=chunk)

        try:
            resp = stub.Transcribe(request_generator(), metadata=get_metadata())
            self.assertTrue(resp.HasField('transcribe'))
            print(f"✓ Транскрипция: {resp.transcribe.transcription}")

        except Exception as e:
            print(f"✗ Тест не прошел: {e}")
            self.fail(f"Transcribe audio with model override failed: {e}")

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
        print(f"\nСообщение: {TEST_MESSAGE_SERIAL}")
        print("Функция: websearch")

        stub = llm_pb2_grpc.LlmStub(grpc.secure_channel(SERVER_ADDRESS, CREDS))

        try:
            # Передаём авториза��ионный заголовок
            responses = stub.NewMessage(
                llm_pb2.NewMessageRequest(
                    msg=TEST_MESSAGE_SERIAL,
                    function="websearch"
                ),
                metadata=get_metadata())

            _, has_gen, has_complete, __, content, reasoning, fc, user_uid, llm_uid = \
                process_llm_responses(responses)
            # if user_uid:
            #     TEST_HISTORY.append(user_uid)
            # if llm_uid:
            #     TEST_HISTORY.append(llm_uid)

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

        try:
            # Передаём авторизационный заголовок
            responses = stub.NewMessage(
                llm_pb2.NewMessageRequest(
                    msg=TEST_MESSAGE_IMAGE_GEN,
                    function="image_gen"
                ),
                metadata=get_metadata())

            _, has_gen, has_complete, __, content, reasoning, fc, user_uid, llm_uid = \
                process_llm_responses(responses)
            # if user_uid:
            #     TEST_HISTORY.append(user_uid)
            # if llm_uid:
            #     TEST_HISTORY.append(llm_uid)

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

    def test_12_new_message_with_docs_and_images(self):
        """Тест 12: проверяет обработку documents_urls и images_urls.
        - Не передаёт history
        - Устанавливает text2text_model = ALL_API_VARS["openaivlm"]["model"]
        - Передаёт documents_urls = [TEST_DOCX_URL]
        - Передаёт images_urls = [TEST_IMAGE_URL]
        - Ожидает chat_name, хотя бы один markdown_chunk и хотя бы один generate
        - Склеивает все markdown_chunk для одного original_name и сохраняет в файл original_name + '.tmp'
        """
        if not TEST_DOCX_URL:
            self.skipTest("TEST_DOCX_URL is not set")
        if not TEST_IMAGE_URL:
            self.skipTest("TEST_IMAGE_URL is not set")
        if not OPENAIVLM_MODEL:
            self.skipTest("INFERENCE_API_OPENAIVLM_MODEL is not set")

        print(f"\nDoc URL: {TEST_DOCX_URL}")
        print(f"Image URL: {TEST_IMAGE_URL}")
        print(f"VLM model override: {OPENAIVLM_MODEL}")

        stub = llm_pb2_grpc.LlmStub(grpc.secure_channel(SERVER_ADDRESS, CREDS))

        try:
            responses = stub.NewMessage(
                llm_pb2.NewMessageRequest(msg="Проанализируй документ и учти картинку в ответе.",
                                          text2text_model=OPENAIVLM_MODEL,
                                          documents_urls=[TEST_DOCX_URL],
                                          images_urls=[TEST_IMAGE_URL]),
                metadata=get_metadata())

            chat_name_seen = False
            md_chunks_by_name = {}
            buffer = []  # buffer responses for reuse with helper

            for response in responses:
                buffer.append(response)
                if response.HasField("chat_name"):
                    chat_name_seen = True
                    cn = response.chat_name
                    print(
                        f"Chat name: {cn.name} (tokens: prompt={cn.prompt_tokens}, "
                        f"completion={cn.completion_tokens}, total={cn.total_tokens}, "
                        f"expected_cost_usd={cn.expected_cost_usd})",
                        flush=True
                    )
                elif response.HasField("markdown_chunk"):
                    mc = response.markdown_chunk
                    md_chunks_by_name.setdefault(mc.original_name, [])
                    md_chunks_by_name[mc.original_name].append(mc.markdown_chunk)
                    print(f"Markdown chunk size: {len(mc.markdown_chunk)} from {mc.original_name}")
                else:
                    # other responses are handled by helper
                    pass

            # Process accumulated responses via common helper to print and collect content/reasoning
            _, has_gen, has_complete, __, content, reasoning, fc, user_uid, llm_uid = \
                process_llm_responses(iter(buffer))
            # if user_uid:
            #     TEST_HISTORY.append(user_uid)
            # if llm_uid:
            #     TEST_HISTORY.append(llm_uid)

            self.assertTrue(chat_name_seen, "Chat name must be returned for new chat")
            self.assertTrue(len(md_chunks_by_name) > 0, "At least one markdown chunk must be returned")
            self.assertTrue(has_gen, "At least one generate response must be returned")

            # Concatenate and write out md chunks
            for original_name, chunks in md_chunks_by_name.items():
                out_path = f"{original_name}.tmp"
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write("".join(chunks))
                print(f"Saved markdown to {out_path} ({sum(len(c) for c in chunks)} bytes)")
                self.assertTrue(os.path.exists(out_path) and os.path.getsize(out_path) > 0,
                                "Concatenated markdown file must be created and non-empty")

        except Exception as e:
            print(f"✗ Тест не прошел: {e}")
            self.fail(f"NewMessage with docs and images failed: {e}")
'''

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
