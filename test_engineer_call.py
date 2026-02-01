#!/usr/bin/env python3
"""
Test script to test what call_function returns when calling the engineer function.
"""

import sys
import os
import json

# Add the current directory to Python path to import main module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import all constant variables from main
from main import (
    call_function, TOOLS, TOOLS_NAMES, TOOLS_SUMMARY,
    CONST, CONST_LEN, ALL_API_VARS, MODEL_TO_API, SECRET_KEY,
    DATETIME_FORMAT, DATETIME_TZ, CONFIG_PATH, TAVILY_BASE_URL,
    MAX_RESULTS, GENERATE_CONFIG_WHEN, DOCLING_ADDRESS, CHUNK_SIZE,
    WHITELIST_REGEX_TEXT2TEXT, BLACKLIST_REGEX_TEXT2TEXT,
    WHITELIST_REGEX_SPEECH2TEXT, BLACKLIST_REGEX_SPEECH2TEXT,
    MINIO_ADDRESS, BUCKET_NAME, MINIO_ACCESS_KEY, MINIO_SECRET_KEY,
    USAGE_FIX, TOOLS, IMGHDR_TO_MIME
)

def test_engineer_call():
    """Test what call_function returns when calling the engineer function."""
    # Generate a unique log UID
    log_uid = "test-engineer-call"

    # Define the engineer tool name and arguments
    tool_name = "engineer"
    args = {
        "query": "Нужна формула листовой стали для ключевых несущих элементов новой морской эксплуатационной платформы в Северном море. Марка стали: Предпочтительна марка Gr70 (например, по стандарту ASTM A572), но готовы рассмотреть аналоги. Габариты: Листы толщиной 25 мм, шириной 3000 мм, длина 12000 мм. Предел текучести (Yield Strength): Минимум 450 МПа. Для некоторых зон с повышенной нагрузкой может потребоваться 500 МПа — просим указать возможность и условия. Ударная вязкость (Charpy V-notch, kCV): Не менее 60 Дж при температуре -40°C. Это обязательное требование из-за суровых условий эксплуатации. Сталь должна обладать хорошей свариваемостью."
    }

    print(f"Testing call_function with tool: {tool_name}")
    print(f"Arguments: {json.dumps(args, indent=2, ensure_ascii=False)}")
    print("-" * 60)

    # Call the function
    result, meta = call_function(log_uid, tool_name, args)

    print("RESULT:")
    print(result)
    print("\nMETA:")
    print(meta)
    print("-" * 60)

    # Print meta details in a more readable format
    if hasattr(meta, 'websearch') and hasattr(meta.websearch, 'item'):
        print("\nReferences from metadata:")
        for i, item in enumerate(meta.websearch.item, 1):
            print(f"{i}. Title: {item.title}")
            print(f"   URL: {item.url}")

    return result, meta

if __name__ == "__main__":
    test_engineer_call()