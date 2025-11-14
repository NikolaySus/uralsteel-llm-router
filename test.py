"""Интеграционные тесты для gRPC LLM сервиса с авторизацией."""

import unittest
import os

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

def get_metadata():
    """Возвращает metadata с авторизацией для защищённых методов."""
    if SECRET_KEY:
        return [('authorization', f'Bearer {SECRET_KEY}')]
    return []


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
            print(f"Количество моделей: {len(response.models)}")
            if response.models:
                print("Список моделей:")
                for model in response.models:
                    print(f"  - {model}")

            self.assertIsNotNone(response)
            self.assertGreater(len(response.models), 0,
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
            print(f"Количество моделей: {len(response.models)}")
            if response.models:
                print("Список моделей:")
                for model in response.models:
                    print(f"  - {model}")

            self.assertIsNotNone(response)
            self.assertGreater(len(response.models), 0,
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

            has_generate = False
            has_complete = False
            content_parts = []
            reasoning_parts = []

            for response in responses:
                if response.HasField("generate"):
                    has_generate = True
                    gen = response.generate
                    if gen.content:
                        content_parts.append(gen.content)
                        print(f"Content: {gen.content}", flush=True)
                    if gen.reasoning_content:
                        reasoning_parts.append(gen.reasoning_content)
                        print(f"Reasoning: {gen.reasoning_content}",
                              flush=True)

                elif response.HasField("complete"):
                    has_complete = True
                    comp = response.complete
                    print("\n✓ Завершено. Токены: "
                          f"prompt={comp.prompt_tokens}, "
                          f"completion={comp.completion_tokens}, "
                          f"total={comp.total_tokens}")

            print("\nРезультаты:")
            print(f"  - Получены GenerateResponseType: {has_generate}")
            print(f"  - Получены CompleteResponseType: {has_complete}")
            if content_parts:
                print(f"  - Content: {''.join(content_parts)}")
            if reasoning_parts:
                print(f"  - Reasoning: {''.join(reasoning_parts)}")

            self.assertTrue(has_generate or has_complete,
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

            has_generate = False
            has_complete = False
            content_parts = []
            reasoning_parts = []

            for response in responses:
                if response.HasField("generate"):
                    has_generate = True
                    gen = response.generate
                    if gen.content:
                        content_parts.append(gen.content)
                        print(f"Content: {gen.content}", flush=True)
                    if gen.reasoning_content:
                        reasoning_parts.append(gen.reasoning_content)
                        print(f"Reasoning: {gen.reasoning_content}",
                              flush=True)

                elif response.HasField("complete"):
                    has_complete = True
                    comp = response.complete
                    print("\n✓ Завершено. Токены: "
                          f"prompt={comp.prompt_tokens}, "
                          f"completion={comp.completion_tokens}, "
                          f"total={comp.total_tokens}")

            print("\nРезультаты:")
            print(f"  - Получены GenerateResponseType: {has_generate}")
            print(f"  - Получены CompleteResponseType: {has_complete}")
            if content_parts:
                print(f"  - Content: {''.join(content_parts)}")
            if reasoning_parts:
                print(f"  - Reasoning: {''.join(reasoning_parts)}")

            self.assertTrue(has_generate or has_complete,
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

            has_transcribe = False
            has_generate = False
            has_complete = False
            transcription = ""
            content_parts = []
            reasoning_parts = []

            for response in responses:
                if response.HasField("transcribe"):
                    has_transcribe = True
                    trans = response.transcribe
                    transcription = trans.transcription
                    print(f"✓ Транскрипция: {transcription}")
                    if trans.duration:
                        print(f"  Длительность: {trans.duration}s")

                elif response.HasField("generate"):
                    has_generate = True
                    gen = response.generate
                    if gen.content:
                        content_parts.append(gen.content)
                        print(f"Content: {gen.content}", flush=True)
                    if gen.reasoning_content:
                        reasoning_parts.append(gen.reasoning_content)
                        print(f"Reasoning: {gen.reasoning_content}",
                              flush=True)

                elif response.HasField("complete"):
                    has_complete = True
                    comp = response.complete
                    print("\n✓ Завершено. Токены: "
                          f"prompt={comp.prompt_tokens}, "
                          f"completion={comp.completion_tokens}, "
                          f"total={comp.total_tokens}")

            print("\nРезультаты:")
            print(f"  - Получены TranscribeResponseType: {has_transcribe}")
            print(f"  - Получены GenerateResponseType: {has_generate}")
            print(f"  - Получены CompleteResponseType: {has_complete}")
            if transcription:
                print(f"  - Транскрипция: {transcription}")
            if content_parts:
                print(f"  - Content: {''.join(content_parts)}")
            if reasoning_parts:
                print(f"  - Reasoning: {''.join(reasoning_parts)}")

            self.assertTrue(has_transcribe or has_generate or has_complete,
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

            has_transcribe = False
            has_generate = False
            has_complete = False
            transcription = ""
            content_parts = []
            reasoning_parts = []

            for response in responses:
                if response.HasField("transcribe"):
                    has_transcribe = True
                    trans = response.transcribe
                    transcription = trans.transcription
                    print(f"✓ Транскрипция: {transcription}")
                    if trans.duration:
                        print(f"  Длительность: {trans.duration}s")

                elif response.HasField("generate"):
                    has_generate = True
                    gen = response.generate
                    if gen.content:
                        content_parts.append(gen.content)
                        print(f"Content: {gen.content}", flush=True)
                    if gen.reasoning_content:
                        reasoning_parts.append(gen.reasoning_content)
                        print(f"Reasoning: {gen.reasoning_content}",
                              flush=True)

                elif response.HasField("complete"):
                    has_complete = True
                    comp = response.complete
                    print("\n✓ Завершено. Токены: "
                          f"prompt={comp.prompt_tokens}, "
                          f"completion={comp.completion_tokens}, "
                          f"total={comp.total_tokens}")

            print("\nРезультаты:")
            print(f"  - Получены TranscribeResponseType: {has_transcribe}")
            print(f"  - Получены GenerateResponseType: {has_generate}")
            print(f"  - Получены CompleteResponseType: {has_complete}")
            if transcription:
                print(f"  - Транскрипция: {transcription}")
            if content_parts:
                print(f"  - Content: {''.join(content_parts)}")
            if reasoning_parts:
                print(f"  - Reasoning: {''.join(reasoning_parts)}")

            self.assertTrue(has_transcribe or has_generate or has_complete,
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
