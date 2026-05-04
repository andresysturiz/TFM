# scripts/core/logger.py

import logging
import os

def get_logger(name):
    os.makedirs("logs", exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Formato
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Log a archivo
    fh = logging.FileHandler("logs/project.log")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Log a consola
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger
