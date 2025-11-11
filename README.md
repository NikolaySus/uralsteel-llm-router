# uralsteel-llm-router
## Роутер.
### Деплой на сервере
```
wget -O - https://raw.githubusercontent.com/NikolaySus/uralsteel-llm-router/main/deploy.ssh | sudo bash
source $HOME/.local/bin/env
```
Затем создание .env файла с переменными BASE_URL, API_KEY, CLOUD_FOLDER, MODEL и запуск.
### Запуск
```
cd ~/uralsteel-llm-router/
uv run --env-file .env main.py
```
