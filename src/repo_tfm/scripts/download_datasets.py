# scripts/download_datasets.py

import ssl
import os
import pandas as pd
from sklearn.datasets import fetch_openml
import subprocess
import shutil
from pathlib import Path

from src.repo_tfm.scripts.core.logger import get_logger
from src.repo_tfm.scripts.core.errors import DownloadError
from src.repo_tfm.scripts.core.utils import safe_create_dir

logger = get_logger("download")

# Desactivar SSL solo dentro de este script (entorno corporativo)
ssl._create_default_https_context = ssl._create_unverified_context


# DESCARGA DESDE R (solo para Gasoline) 

def download_from_r(script_name: str, output_name: str, target_name: str):
    """
    Ejecuta un script R que genera un CSV y lo mueve a data/raw/.
    """
    logger.info(f"Descargando {target_name} usando R (CRAN)...")

    r_script = Path(f"R/{script_name}")
    output_csv = Path(output_name)
    target_csv = Path(f"data/raw/{target_name}")

    try:
        subprocess.run(["Rscript", str(r_script)], check=True)
        shutil.move(str(output_csv), str(target_csv))
        logger.info(f"{target_name} descargado correctamente desde R (CRAN).")

    except Exception as e:
        logger.error(f"Error ejecutando R para descargar {target_name}: {e}")
        raise DownloadError(f"No se pudo descargar {target_name}")


# DESCARGA DESDE OPENML 

def download_from_openml(dataset_name: str, output_filename: str):
    logger.info(f"Descargando {dataset_name} desde OpenML...")

    try:
        raw_dir = "data/raw"
        os.makedirs(raw_dir, exist_ok=True)

        csv_path = os.path.join(raw_dir, output_filename)

        data = fetch_openml(name=dataset_name, as_frame=True)
        df = pd.concat([data.data, data.target], axis=1)

        df.to_csv(csv_path, index=False)

        logger.info(
            f"{dataset_name} guardado en {csv_path} "
            f"({df.shape[0]} filas, {df.shape[1]} columnas)"
        )

    except Exception as e:
        logger.error(f"Error descargando {dataset_name}: {e}")
        raise DownloadError(f"No se pudo descargar {dataset_name}")


# MAIN 

def main():
    safe_create_dir("data/raw")

    # 1. Tecator 
    download_from_openml("tecator", "tecator.csv")

    # 2. Gasoline 
    download_from_r(
        script_name="export_gasoline.R",
        output_name="gasoline.csv",
        target_name="gasoline.csv"
    )

    # 3. Riboflavin 
    download_from_openml("riboflavin", "riboflavin.csv")

    logger.info("Descarga completada correctamente.")


if __name__ == "__main__":
    main()
