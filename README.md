# uralsteel-llm-router
## Роутер.
### Деплой на сервере
```
wget -O - https://raw.githubusercontent.com/NikolaySus/uralsteel-llm-router/main/deploy.ssh | sudo bash
source $HOME/.local/bin/env
cd ~/uralsteel-llm-router/
```
Затем создание .env файла с переменными BASE_URL, API_KEY, CLOUD_FOLDER, MODEL и запуск.
### Запуск
```
cd ~/uralsteel-llm-router/
uv run --env-file .env main.py
```
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
