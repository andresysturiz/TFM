# scripts/core/utils.py

import os
import pandas as pd
from src.repo_tfm.scripts.core.errors import DatasetError
from src.repo_tfm.scripts.core.logger import get_logger

logger = get_logger("utils")


def safe_read_csv(path):
    """
    Lectura segura de CSV con validación y logging.
    """
    if not os.path.exists(path):
        logger.error(f"Archivo no encontrado: {path}")
        raise DatasetError(f"Archivo no encontrado: {path}")

    try:
        df = pd.read_csv(path)
    except Exception as e:
        logger.error(f"Error leyendo {path}: {str(e)}")
        raise DatasetError(f"Error leyendo {path}: {str(e)}")

    if df.empty:
        logger.error(f"Dataset vacío: {path}")
        raise DatasetError(f"Dataset vacío: {path}")

    logger.info(f"Dataset cargado correctamente: {path}")
    return df


def safe_create_dir(path):
    """
    Crea un directorio de forma segura.
    """
    try:
        os.makedirs(path, exist_ok=True)
        logger.info(f"Directorio creado/verificado: {path}")
    except Exception as e:
        logger.error(f"No se pudo crear {path}: {str(e)}")
        raise
