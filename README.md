# uralsteel-llm-router
## Сервис-роутер для взаимодействия с ML моделями.
### Клиентским сервисам
Сервис имеет публичные и защищённые методы:
- **Ping** - публичный метод (без авторизации) для health-check
- **NewMessage, AvailableModelsText2Text, AvailableModelsSpeech2Text** - защищённые методы (требуют SECRET_KEY)

Вам нужны:
1. `ca.crt` - SSL сертификат из этого репозитория
2. `SECRET_KEY` - Секретный ключ доступа (например, из `.env` файла)

**Документация по авторизации:**
[AUTHORIZATION.md](./AUTHORIZATION.md)

### Деплой на сервере
```bash
wget -O - https://raw.githubusercontent.com/NikolaySus/uralsteel-llm-router/main/deploy.sh | sudo bash
source $HOME/.local/bin/env
cd ~/uralsteel-llm-router/
uv run python -m playwright install
```
Затем скачать `.env` файл с переменными среды и комментариями с инструкцией по созданию сертификата безопасности, выполнить инструкцию и произвести запуск. Файл `.env` предоставляется посредством защищённого канала передачи информации (лс в тг).
### Запуск
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
