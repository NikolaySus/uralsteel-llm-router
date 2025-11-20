"""
Скрипт для генерации конфигурационного файла config.json.
"""

import asyncio
from datetime import datetime
import json
import re
import os

from bs4 import BeautifulSoup
from markdownify import markdownify as md
from playwright.async_api import async_playwright
from openai import OpenAI


# Словил кринж с systemd...
# Фикс для подстановки переменных окружения в другие переменные окружения
for key, value in os.environ.items():
    if "${" in value:
        os.environ[key] = os.path.expandvars(value)

# Переменные окружения вида CONST_api_case
# Пока CONST + YANDEXAI_/OPENAI_ + BASE_URL/FOLDER/KEY/MODEL/PRICES_URL
CONST = "INFERENCE_API_"
CONST_LEN = len(CONST)
ALL_API_VARS = dict()
for name, value in os.environ.items():
    if name.startswith(CONST):
        api_and_case = name[CONST_LEN:].lower()
        delim = api_and_case.find("_")
        api = api_and_case[:delim]
        case = api_and_case[delim + 1:]
        ALL_API_VARS.setdefault(api, dict())
        ALL_API_VARS[api][case] = value
# Секретный ключ для клиентского доступа к gRPC методам
SECRET_KEY = os.environ.get('SECRET_KEY', '')
# Формат даты и времени
DATETIME_FORMAT = os.environ.get('DATETIME_FORMAT', '%Y-%m-%dT%H:%M:%S')
# Путь к конфигурационному файлу
CONFIG_PATH = "config.json"
# Системный промпт для извлечения цены
SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "You are LLM-powered agent for data extraction. "
        "User provides the model name and pricing page content. "
        "The model is selected by the user from the list of default models "
        "provided, but on the pricing page this name may differ "
        "slightly or be included in a more general category. "
        "The task is to find the tariff for using this model. "
        "The response must match the pattern: '$float / unit'. "
        "Examples of responses:\n"
        "$0.0006 / hour\n"
        "$0.000111222 / minute\n"
        "$0.000001248124 / second\n"
        "$0.0004 / 1M*unit\n"
        "$0.000123123 / 1k*unit\n"
        "$0.000001212121 / unit\n"
        "$0.042 / image\n"
        "You MUST find something."
    )
}
# Кэшированные страницы
CACHED_PAGES = dict()


async def fetch_openai_pricing(url: str) -> str:
    """Получить и обработать страницу с ценами."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=[
                '--disable-blink-features=AutomationControlled',
                '--disable-dev-shm-usage',
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-web-security',
                '--disable-features=VizDisplayCompositor',
                '--disable-background-timer-throttling',
                '--disable-backgrounding-occluded-windows',
                '--disable-renderer-backgrounding',
            ]
        )
        context = await browser.new_context(
            user_agent=("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/129.0.0.0 Safari/537.36"),
            viewport={"width": 1920, "height": 1080},
            extra_http_headers={
                'Accept': ('text/html,application/xhtml+xml,application/xml;'
                           'q=0.9,image/webp,*/*;q=0.8'),
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            },
            bypass_csp=True,
        )
        await context.add_init_script('''
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined,
            });
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3, 4, 5],
            });
            Object.defineProperty(navigator, 'languages', {
                get: () => ['en-US', 'en'],
            });
        ''')
        page = await context.new_page()
        try:
            await page.goto(url, wait_until="networkidle", timeout=60000)
            await page.wait_for_function(
                ("() => document.title && "
                 "!document.title.includes('Checking your browser')"),
                timeout=60000
            )
            await page.wait_for_selector('body', timeout=30000)
            await asyncio.sleep(3)
            page_title = await page.title()
            page_content = await page.content()
            if ("cloudflare" in page_content.lower()
                or "challenge" in page_content.lower()):
                print("Cloudflare challenge detected, waiting longer...")
                await asyncio.sleep(10)
                try:
                    await page.wait_for_function(
                        "() => document.body.innerText.length > 1000",
                        timeout=30000
                    )
                except Exception:
                    print("Content loading taking longer than expected...")
            raw_content = await page.content()
            soup = BeautifulSoup(raw_content, "html.parser")
            markdown_content = md(str(soup))
            return True, markdown_content
        except Exception as e:
            print(f"Error fetching OpenAI pricing: {e}")
            try:
                content = await page.content()
                return False, content[:1000]  # Первые 1000 символов для дебага
            except Exception as e2:
                return False, f"Also failed to get content: {e2}"
        finally:
            await browser.close()


def get_coef(api_vars) -> float:
    """Получить коэффициент цены для API."""
    query_string = api_vars["model"]
    if api_vars["prices_url"] in CACHED_PAGES:
        long_string = CACHED_PAGES[api_vars["prices_url"]]
        ok = True
    else:
        ok, long_string = asyncio.run(fetch_openai_pricing(api_vars["prices_url"]))
        CACHED_PAGES[api_vars["prices_url"]] = long_string
    if ok:
        messages = [
            SYSTEM_MESSAGE,
            {
                "role": "user",
                "content": (
                    f"The model name is '{query_string}'."
                    f"The page content:\n{long_string}"
                )
            }
        ]
        response = OpenAI(
            base_url=ALL_API_VARS["yandexai"]["base_url"],
            api_key=ALL_API_VARS["yandexai"]["key"],
            project=ALL_API_VARS["yandexai"]["folder"],
        ).chat.completions.create(
            model=ALL_API_VARS["yandexai"]["model"],
            messages=messages,
            temperature=0.3
        )
        # completion_tokens = response.usage.completion_tokens
        # prompt_tokens = response.usage.prompt_tokens
        # total_tokens = response.usage.total_tokens
        text = response.choices[0].message.content
        pattern = "/"
        match = re.search(pattern, text)
        if match:
            center_position = match.start()
            new_position = center_position + len(pattern)
            end_position = len(text)
            numerator = float(text[1:center_position].strip())
            denominator = text[new_position:end_position].strip()
            print(f"'{numerator}'\n'{denominator}'")
            if denominator == 'hour':
                denominator = 3600
            elif denominator == 'minute':
                denominator = 60
            elif denominator == 'second':
                denominator = 1
            elif denominator == '1M*unit':
                denominator = 1000000
            elif denominator == '1k*unit':
                denominator = 1000
            elif denominator == 'unit':
                denominator = 1
            elif denominator == 'image':
                denominator = 1
            result_coef = numerator / denominator
            return result_coef
        else:
            raise ValueError(f"Pattern not found in response: {text}")
    else:
        raise ValueError(f"Failed to fetch pricing page: {long_string}")


if __name__ == "__main__":
    try:
        config = {
            "prices_coefs": {
                name: get_coef(value) for name, value in ALL_API_VARS.items()
            },
            "generated_at": datetime.now().strftime(DATETIME_FORMAT),
        }
    except Exception as e:
        with open("error.txt", "w", encoding="utf-8") as f:
            f.write(str(e))
        raise e

    with open("config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("Config generated.")
