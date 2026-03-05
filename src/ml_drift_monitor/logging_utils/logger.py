import logging
from pathlib import Path
from typing import Optional

from ml_drift_monitor.config import get_default_config


_LOGGER_NAME = "ml_drift_monitor"
_configured = False


def configure_logger(logs_dir: Optional[Path] = None) -> logging.Logger:
    """
    Configure and return the project-wide logger.
    Safe to call multiple times; configuration is applied only once.
    """
    global _configured
    logger = logging.getLogger(_LOGGER_NAME)
    if _configured:
        return logger

    cfg = get_default_config()
    target_logs_dir = logs_dir or cfg.paths.logs_dir
    target_logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = target_logs_dir / "app.log"

    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    _configured = True
    logger.info("Logger configured. Logs dir: %s", target_logs_dir)
    return logger


def get_logger() -> logging.Logger:
    """
    Get the project-wide logger, configuring it if needed.
    """
    if not _configured:
        return configure_logger()
    return logging.getLogger(_LOGGER_NAME)

