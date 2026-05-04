# scripts/download_datasets.py

import ssl
import os
import pandas as pd
from ucimlrepo import fetch_ucirepo

from scripts.core.logger import get_logger
from scripts.core.errors import DownloadError
from scripts.core.utils import safe_create_dir

logger = get_logger("download")

# Desactivar SSL solo dentro de este script
ssl._create_default_https_context = ssl._create_unverified_context

def download_dataset(id, name, filename):
    try:
        logger.info(f"Descargando {name}...")
        data = fetch_ucirepo(id=id)
        df = pd.concat([data.data.features, data.data.targets], axis=1)
        df.to_csv(filename, index=False)
        logger.info(f"{name} guardado en {filename}")
    except Exception as e:
        logger.error(f"Error descargando {name}: {str(e)}")
        raise DownloadError(f"No se pudo descargar {name}")

def main():
    safe_create_dir("data/raw")

    download_dataset(
        id=291,
        name="Airfoil Self-Noise",
        filename="data/raw/airfoil_self_noise.csv"
    )

    download_dataset(
        id=186,
        name="Wine Quality",
        filename="data/raw/wine_quality.csv"
    )

    logger.info("Descarga completada correctamente.")

if __name__ == "__main__":
    main()
