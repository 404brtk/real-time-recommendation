import logging
import sys
from pathlib import Path


def setup_logging(
    log_file: str = "app.log", level: int = logging.INFO
) -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(logs_dir / log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger
