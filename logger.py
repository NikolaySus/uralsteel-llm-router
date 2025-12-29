"""
Logging configuration for the gRPC LLM router service.

Works with:
- systemd (uses standard output, compatible with journalctl)
- Docker (outputs to stdout/stderr)
- CLI (human-readable format)
"""

import logging
import sys
import os


def setup_logger(name: str = "llm-router") -> logging.Logger:
    """
    Configure and return a logger instance.
    
    The logger is configured to work optimally in different environments:
    - systemd: Uses INFO level, outputs to stdout (captured by journalctl)
    - Docker: Uses INFO level, outputs to stdout with timestamps
    - CLI: Uses INFO level, readable format with timestamps
    
    Args:
        name: Logger name (default: "llm-router")
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid adding multiple handlers if setup_logger is called multiple times
    if logger.handlers:
        return logger
    
    # Determine log level from environment or default to INFO
    log_level_str = os.environ.get('LOG_LEVEL', 'INFO').upper()
    try:
        log_level = getattr(logging, log_level_str)
    except AttributeError:
        log_level = logging.INFO
    
    logger.setLevel(log_level)
    
    # Create formatter
    # Format: TIMESTAMP [LEVEL] MESSAGE
    # Systemd doesn't need timestamp (journalctl adds it), but Docker/CLI benefits from it
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create and configure stdout handler
    # Use stdout for all levels (systemd captures stdout as regular logs)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    
    # Create and configure stderr handler only for ERROR and CRITICAL
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.ERROR)
    stderr_handler.setFormatter(formatter)
    logger.addHandler(stderr_handler)
    
    return logger


# Create global logger instance that can be imported
logger = setup_logger()
