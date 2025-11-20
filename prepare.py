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
    """Получить и обработать страницу с ценами, имитируя поведение реального пользователя."""
    import random
    from urllib.parse import urlparse

    # Набор реалистичных user-agent строк разных платформ/версий
    UA_POOL = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.5993.70 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    ]
    ua = random.choice(UA_POOL)

    # Случайные размеры окна (как у реальных мониторов)
    viewports = [(1920, 1080), (1366, 768), (1536, 864), (1440, 900), (1680, 1050)]
    vw, vh = random.choice(viewports)

    # Локали/таймзоны для большей правдоподобности
    locales = ["en-US", "en-GB", "de-DE", "fr-FR"]
    locale = random.choice(locales)
    timezones = [
        "America/New_York", "America/Los_Angeles", "Europe/Berlin",
        "Europe/London", "Europe/Paris", "Asia/Tokyo"
    ]
    tz = random.choice(timezones)

    # Базовые задержки как у пользователя
    async def human_sleep(a=0.8, b=1.8):
        await asyncio.sleep(random.uniform(a, b))

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=[
                # Уменьшаем явные признаки автоматизации
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
            ],
        )
        context = await browser.new_context(
            user_agent=ua,
            viewport={"width": vw, "height": vh},
            locale=locale,
            timezone_id=tz,
            color_scheme=random.choice(["light", "dark"]),
            permissions=[],
        )

        # Маскируем webdriver и добавляем правдоподобные поля
        await context.add_init_script('''
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            // Имитируем наличие плагинов/медиа-устройств
            Object.defineProperty(navigator, 'plugins', { get: () => ({ length: 3 }) });
            Object.defineProperty(navigator, 'languages', { get: () => ['en-US','en'] });
            // webgl vendor/renderer
            const getParameter = WebGLRenderingContext.prototype.getParameter;
            WebGLRenderingContext.prototype.getParameter = function(parameter){
              if (parameter === 37445) return 'Intel Inc.'; // UNMASKED_VENDOR_WEBGL
              if (parameter === 37446) return 'Intel Iris OpenGL Engine'; // UNMASKED_RENDERER_WEBGL
              return getParameter.call(this, parameter);
            };
        ''')

        # Пытаемся повторно использовать куки для домена, чтобы выглядеть как возвращающийся пользователь
        storage_path = os.path.join(".playwright_storage", urlparse(url).netloc.replace(":", "_"))
        try:
            if os.path.exists(storage_path):
                await context.add_cookies(json.load(open(storage_path, "r", encoding="utf-8")))
        except Exception:
            pass

        page = await context.new_page()

        # Минимальная навигация как пользователь
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=60000)
            await human_sleep(1.0, 2.5)

            # Дожидаемся содержимого
            await page.wait_for_selector("body", timeout=30000)
            await human_sleep(0.8, 1.8)

            # Немного скроллим страницу
            try:
                for _ in range(random.randint(1, 3)):
                    await page.mouse.wheel(0, random.randint(400, 1200))
                    await human_sleep(0.5, 1.2)
            except Exception:
                pass

            # Ждём, пока уйдут возможные проверки/редиректы
            try:
                await page.wait_for_function(
                    "() => document.title && !/checking your browser/i.test(document.title)",
                    timeout=45000,
                )
            except Exception:
                pass

            # Если видим Cloudflare/челлендж, ждём дольше
            content_snapshot = (await page.content()).lower()
            if any(k in content_snapshot for k in ["cloudflare", "challenge", "attention required"]):
                await human_sleep(5.0, 8.0)
                try:
                    await page.wait_for_function(
                        "() => (document.body && document.body.innerText && document.body.innerText.length > 800)",
                        timeout=45000,
                    )
                except Exception:
                    pass

            raw_content = await page.content()
            soup = BeautifulSoup(raw_content, "html.parser")
            markdown_content = md(str(soup))

            # Сохраняем куки, чтобы в следующий раз выглядеть "постояннее"
            try:
                os.makedirs(os.path.dirname(storage_path), exist_ok=True)
                json.dump(await context.cookies(), open(storage_path, "w", encoding="utf-8"))
            except Exception:
                pass

            return True, markdown_content
        except Exception as e:
            print(f"Error fetching OpenAI pricing: {e}")
            try:
                content = await page.content()
                return False, content[:1000]
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
    config = {}
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

    with open("config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("Config generated.")
