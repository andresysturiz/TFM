# scripts/download_datasets.py

import ssl
import os
import pandas as pd
import requests
from ucimlrepo import fetch_ucirepo

from scripts.core.logger import get_logger
from scripts.core.errors import DownloadError
from scripts.core.utils import safe_create_dir

logger = get_logger("download")

# Desactivar SSL solo dentro de este script
ssl._create_default_https_context = ssl._create_unverified_context


# ============================================================
# DESCARGA DESDE UCI
# ============================================================

def download_dataset(id, name, filename):
    try:
        logger.info(f"Descargando {name} (UCI)...")
        data = fetch_ucirepo(id=id)

        df = pd.concat([data.data.features, data.data.targets], axis=1)

        if df.empty:
            raise DownloadError(f"{name} está vacío")

        df.to_csv(filename, index=False)
        logger.info(f"{name} guardado en {filename} ({df.shape[0]} filas, {df.shape[1]} columnas)")

    except Exception as e:
        logger.error(f"Error descargando {name}: {str(e)}")
        if os.path.exists(filename):
            os.remove(filename)
        raise DownloadError(f"No se pudo descargar {name}")


# ============================================================
# GASOLINE (dataset real NIR)
# ============================================================

def download_gasoline():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/gasoline.csv"
    filename = "data/raw/gasoline.csv"

    try:
        logger.info("Descargando Gasoline (NIR)...")
        r = requests.get(url, timeout=10)
        r.raise_for_status()

        with open(filename, "wb") as f:
            f.write(r.content)

        df = pd.read_csv(filename)
        logger.info(f"Gasoline guardado en {filename} ({df.shape[0]} filas, {df.shape[1]} columnas)")

    except Exception as e:
        logger.error(f"Error descargando Gasoline: {str(e)}")
        if os.path.exists(filename):
            os.remove(filename)
        raise DownloadError("No se pudo descargar Gasoline")


# ============================================================
# GOLUB (gene expression)
# ============================================================

def download_golub():
    url_expr = "https://raw.githubusercontent.com/ramhiser/datamicroarray/master/data/golub/golub_expression.csv"
    url_labels = "https://raw.githubusercontent.com/ramhiser/datamicroarray/master/data/golub/golub_class.csv"

    filename_expr = "data/raw/golub_expression.csv"
    filename_labels = "data/raw/golub_labels.csv"

    try:
        logger.info("Descargando Golub (gene expression)...")

        expr = pd.read_csv(url_expr, index_col=0)
        labels = pd.read_csv(url_labels)

        if expr.empty or labels.empty:
            raise DownloadError("Golub descargado vacío")

        expr.to_csv(filename_expr)
        labels.to_csv(filename_labels, index=False)

        logger.info(f"Golub guardado en data/raw/ (expr: {expr.shape}, labels: {labels.shape})")

    except Exception as e:
        logger.error(f"Error descargando Golub: {str(e)}")
        if os.path.exists(filename_expr):
            os.remove(filename_expr)
        if os.path.exists(filename_labels):
            os.remove(filename_labels)
        raise DownloadError("No se pudo descargar Golub")


# ============================================================
# MAIN
# ============================================================

def main():
    safe_create_dir("data/raw")

    # UCI datasets
    download_dataset(291, "Airfoil Self-Noise", "data/raw/airfoil_self_noise.csv")
    download_dataset(186, "Wine Quality", "data/raw/wine_quality.csv")

    # Nuevos datasets
    download_gasoline()
    download_golub()

    logger.info("Descarga completada correctamente.")


if __name__ == "__main__":
    main()
