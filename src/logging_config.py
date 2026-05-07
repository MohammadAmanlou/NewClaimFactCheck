import logging
from pathlib import Path

def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger(log_file.name)
    logger.setLevel(logging.INFO)
    logger.handlers = []

    fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger