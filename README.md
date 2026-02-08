# uralsteel-llm-router
## Сервис-роутер для взаимодействия с ML моделями.
### Клиентским сервисам
Сервис имеет публичные и защищённые методы (см. подробнее ниже):
- **Ping** — публичный (без авторизации) health-check
- **NewMessage, Transcribe, AvailableModelsText2Text, AvailableModelsSpeech2Text, AvailableTools** — защищённые (требуют SECRET_KEY)

Вам нужны:
1. `ca.crt` - SSL сертификат из этого репозитория
2. `SECRET_KEY` - Секретный ключ доступа (например, из `.env` файла)

**Документация по авторизации:**
[AUTHORIZATION.md](./AUTHORIZATION.md)

### RPC методы (I/O, внешние вызовы, ошибки)

> Формальные структуры сообщений см. в `uralsteel-grpc-api/llm/llm.proto`.

**Ping (public)**
- **Запрос:** `google.protobuf.Empty`
- **Ответ:** `google.protobuf.Empty`
- **Назначение:** health-check сервиса
- **Ошибки:** не генерирует ошибок авторизации, остальные — стандартные gRPC

**AvailableModelsText2Text (protected)**
- **Запрос:** `google.protobuf.Empty`
- **Ответ:** `StringsListResponse` — список моделей text2text
- **Обращения вовне:**
  - Yandex GPT (OpenAI‑совместимый endpoint) — получение списка моделей с фильтрацией по `WHITELIST_REGEX_TEXT2TEXT` / `BLACKLIST_REGEX_TEXT2TEXT`
  - Добавляются модели из env: `openaivlm.model`, `deepseek.model`, `openaimini.model`
- **Ошибки:** при ошибке запроса/фильтрации — `INTERNAL`, пустой список

**AvailableModelsSpeech2Text (protected)**
- **Запрос:** `google.protobuf.Empty`
- **Ответ:** `StringsListResponse` — список моделей speech2text
- **Обращения вовне:** OpenAI Speech-to-Text (Whisper или аналог) через `INFERENCE_API_openai_*`
- **Ошибки:** при ошибке запроса — `INTERNAL`, пустой список

**AvailableTools (protected)**
- **Запрос:** `google.protobuf.Empty`
- **Ответ:** `StringsListResponse` — названия доступных инструментов (сейчас `websearch`, `image_gen`)
- **Используются инструменты в `NewMessage` при function-calling**

**Transcribe (protected, streaming request → single response)**
- **Запрос (stream `TranscribeRequest`):**
  - `mp3_chunk` — байты mp3 чанка (обязательно хотя бы один)
  - `speech2text_model` — опционально, переопределяет модель; иначе берётся `INFERENCE_API_openai_model`
- **Ответ (одиночный `TranscribeResponse`):** `TranscribeResponseType` с полями `transcription`, `duration`, `expected_cost_usd`, `datetime`
- **Обращения вовне:** OpenAI Audio Transcriptions (`base_url`/`key` из `INFERENCE_API_openai_*`)
- **Ошибки:** отсутствие аудио чанков или ошибка инференса → `INTERNAL`, в ответе пустая транскрипция и нулевые поля

**NewMessage (protected, single request → streaming response)**
- **Запрос (`NewMessageRequest`):**
  - `msg` — текст сообщения **(обязательно)**
  - `history` — список `user_message_uid/llm_message_uid` из предыдущего диалога (хранятся в MinIO)
  - `text2text_model` — опциональный override для text2text модели
  - `function` — опциональный выбор инструмента (`websearch` или `image_gen`)
  - `documents_urls` — файлы для контекста; конвертируются через Docling, markdown чанки приходят отдельными ответами
  - `images_urls` — внешние изображения для контекста; подгружаются и инлайн-ятся в base64
  - `markdown_urls` — готовые markdown URL с указанием оригинального имени
- **Ответ (stream `NewMessageResponse`):** в потоке могут приходить
  - `chat_name` — автоимя чата для первого сообщения (модель `openaimini`)
  - `markdown_chunk` — чанки md из Docling конвертации (CHUNK_SIZE, по умолчанию 8192)
  - `user_message_uid` / `llm_message_uid` — идентификаторы сообщений, сохранённых в MinIO (`BUCKET_NAME`)
  - `generate` — части текста ответа
  - `complete` — финальная статистика токенов и `expected_cost_usd`
  - `function_call_*` — события function-calling
  - `tool_metadata` — метаданные инструмента (websearch urls или image_gen base64+cost)
- **Обращения вовне:**
  - LLM инференс по выбранной модели:
    - по умолчанию `INFERENCE_API_yandexai_model`
    - если есть картинки → `INFERENCE_API_openaivlm_model`
    - explicit override `text2text_model`
  - Docling (`DOCLING_ADDRESS`) для конвертации документов
  - Tavily (`TAVILY_BASE_URL`) для инструмента `websearch`
  - OpenAI Images (`INFERENCE_API_openaiimgen_*`) для `image_gen`
  - MinIO S3 (`MINIO_ADDRESS`, `BUCKET_NAME`) — хранение истории
- **Особенности:**
  - Пустой `msg` → ошибка
  - Если переданы изображения и выбранная модель без VLM — картинки вычищаются, добавляется системное предупреждение
  - Стоимость рассчитывается по `price_coef` (или `price_coef_input/output`) из `config.json`
- **Ошибки:** любые ошибки конвертации/инференса → `INTERNAL` с деталями в `context.set_details`, поток завершается

### Деплой на сервере
Перейти в:
```
cd /root/
```
Затем:
```bash
wget -O - https://raw.githubusercontent.com/NikolaySus/uralsteel-llm-router/main/deploy.sh | sudo bash
source $HOME/.local/bin/env
cd ~/uralsteel-llm-router/
uv run python -m playwright install
uv run python -m playwright install-deps
```
Затем скачать `.env` файл с переменными среды и комментариями с инструкцией по созданию сертификата безопасности, выполнить инструкцию и произвести запуск. Файл `.env` предоставляется посредством защищённого канала передачи информации (лс в тг). Также получить и положить рядом config.json (из лс в тг).
Затем:
```
mkdir docling-serve-cpu
cd docling-serve-cpu
touch docker-compose.yaml
```
Вставить туда:
```
services:
  docling-serve:
    image: ghcr.io/docling-project/docling-serve-cpu:v1.9.0
    container_name: docling-serve-cpu
    ports:
      - "5001:5001"
    restart: unless-stopped
```
И запустить:
```
docker compose up -d
```
Затем:
```
cd /root/
git clone https://github.com/vakovalskii/searxng-docker-tavily-adapter.git
cd searxng-docker-tavily-adapter

cp config.example.yaml config.yaml
```
Поменять там на:
```
user_agent: "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
```
Затем:
```
docker compose up -d
curl -X POST "http://localhost:8000/search" \
     -H "Content-Type: application/json" \
     -d '{"query": "цена bitcoin", "max_results": 3}'
```
### Тестовый запуск
```bash
cd ~/uralsteel-llm-router/
uv run --env-file .env main.py
```
### Интеграционный тест
```bash
cd ~/uralsteel-llm-router/
uv run --env-file .env test.py
```

#### Описание интеграционных тестов

Тестовый набор состоит из 7 интеграционных тестов:

1. **test_01_ping** - Проверка работоспособности сервиса
   - Вызывает RPC метод `Ping()`
   - Проверяет получение ответа

2. **test_02_available_models_text2text** - Список доступных Text2Text моделей
   - Вызывает RPC метод `AvailableModelsText2Text()`
   - Проверяет, что список моделей не пустой
   - Выводит все доступные модели

3. **test_03_available_models_speech2text** - Список доступных Speech2Text моделей
   - Вызывает RPC метод `AvailableModelsSpeech2Text()`
   - Проверяет, что список моделей не пустой
   - Выводит все доступные модели

4. **test_04_new_message_text_no_history** - NewMessage с текстом без истории
   - Отправляет текстовое сообщение
   - Проверяет получение GenerateResponseType и CompleteResponseType

5. **test_05_new_message_text_with_history** - NewMessage с текстом и историей
   - Отправляет текстовое сообщение с историей разговора
   - Проверяет получение GenerateResponseType и CompleteResponseType

6. **test_06_new_message_audio_no_history** - NewMessage с аудио (mp3) без истории
   - Отправляет mp3 файл чанками по 4KB
   - Проверяет получение TranscribeResponseType, GenerateResponseType и CompleteResponseType
   - Требует наличие файла `serial.mp3`

7. **test_07_new_message_audio_with_history** - NewMessage с аудио (mp3) и историей
   - Отправляет mp3 файл с историей разговора
   - Проверяет получение всех типов ответов
   - Требует наличие файла `serial.mp3`

8. **test_08_available_models_text2text_without_auth** - проверка авторизации
   - Вызывает RPC метод `AvailableModelsText2Text()` без секретного ключа
   - AvailableModelsText2Text ДОЛЖЕН ОТКЛОНИТЬ запрос БЕЗ авторизации

9. **test_09_available_tools** - Список доступных инструментов/функций
   - Вызывает RPC метод `AvailableTools()`
   - Проверяет, что список инструментов не пустой
   - Выводит все доступные инструменты

10. **test_10_new_message_text_with_websearch** - NewMessage с функцией websearch
    - Отправляет текстовое сообщение с параметром function="websearch"
    - Проверяет получение GenerateResponseType и CompleteResponseType

11. **test_11_new_message_text_with_image_gen** - NewMessage с функцией image_gen
    - Отправляет текстовое сообщение с параметром function="image_gen"
    - Проверяет получение GenerateResponseType и CompleteResponseType
    - Обрабатывает ToolMetadata ответы с генерируемыми изображениями

### Добавление в качестве сервиса и запуск

#### Создание systemd сервис-файла
Создайте файл сервиса:
```bash
sudo nano /etc/systemd/system/uralsteel-llm-router.service
```

Добавьте следующее содержимое:
```ini
[Unit]
Description=Uralsteel LLM Router
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/uralsteel-llm-router
EnvironmentFile=/root/uralsteel-llm-router/.env
ExecStart=/root/.local/bin/uv run --env-file .env main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

#### Запуск сервиса
```bash
# Перезагрузить конфигурацию systemd
sudo systemctl daemon-reload

# Включить сервис при загрузке системы
sudo systemctl enable uralsteel-llm-router

# Запустить сервис
sudo systemctl start uralsteel-llm-router

# Проверить статус
sudo systemctl status uralsteel-llm-router
```

#### Управление сервисом
```bash
# Остановить сервис
sudo systemctl stop uralsteel-llm-router

# Перезагрузить сервис
sudo systemctl restart uralsteel-llm-router

# Отключить автозапуск при загрузке
sudo systemctl disable uralsteel-llm-router
```
