"""
Конфигурация логирования для сервиса маршрутизации LLM через gRPC.

Работает в различных средах:
- systemd (использует стандартный вывод, совместим с journalctl)
- Docker (вывод в stdout/stderr)
- CLI (читаемый формат с метками времени)
"""

import logging
import sys
import os


class AutoFlushStreamHandler(logging.StreamHandler):
    """StreamHandler с автоматическим flush после каждого log message."""
    
    def emit(self, record):
        try:
            super().emit(record)
            self.flush()
        except Exception:
            self.handleError(record)


def setup_logger(name: str = "llm-router") -> logging.Logger:
    """
    Настройка и возврат экземпляра логгера.
    
    Логгер настроен для оптимальной работы в разных средах:
    - systemd: уровень INFO, вывод в stdout (перехватывается journalctl)
    - Docker: уровень INFO, вывод в stdout с метками времени
    - CLI: уровень INFO, читаемый формат с метками времени
    
    Аргументы:
        name: Имя логгера (по умолчанию: "llm-router")
    
    Возвращает:
        Настроенный экземпляр логгера
    """
    logger = logging.getLogger(name)

    # Предотвращаем добавление нескольких обработчиков при повторном вызове setup_logger
    if logger.handlers:
        return logger

    # Определяем уровень логирования из переменных окружения или используем INFO по умолчанию
    log_level_str = os.environ.get('LOG_LEVEL', 'INFO').upper()
    try:
        log_level = getattr(logging, log_level_str)
    except AttributeError:
        log_level = logging.INFO

    logger.setLevel(log_level)

    # Создаем форматтер
    # Формат: ВРЕМЯ [УРОВЕНЬ] СООБЩЕНИЕ
    # Systemd не требует меток времени (journalctl добавляет их),
    # но Docker/CLI выигрывают от их наличия
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Создаем и настраиваем обработчик для stdout
    # Используем stdout для всех уровней (systemd перехватывает stdout как обычные логи)
    stdout_handler = AutoFlushStreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    # Создаем и настраиваем обработчик для stderr только для ERROR и CRITICAL
    stderr_handler = AutoFlushStreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.ERROR)
    stderr_handler.setFormatter(formatter)
    logger.addHandler(stderr_handler)

    return logger


# Создаем глобальный экземпляр логгера для импорта
logger = setup_logger()
