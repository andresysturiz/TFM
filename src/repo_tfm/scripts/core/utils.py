# scripts/core/utils.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from src.repo_tfm.scripts.core.errors import DatasetError
from src.repo_tfm.scripts.core.logger import get_logger

logger = get_logger("utils")

def plot_predicted_vs_observed(
    y_true,
    y_pred,
    model_name,
    dataset,
    out_path
):

    plt.figure(figsize=(6, 6))

    plt.scatter(y_true, y_pred, alpha=0.7)

    lims = [
        min(y_true.min(), y_pred.min()),
        max(y_true.max(), y_pred.max())
    ]

    plt.plot(lims, lims, "--", color="red")

    plt.xlabel("Observado")
    plt.ylabel("Predicho")

    plt.title(
        f"{model_name} - Predicho vs Observado ({dataset})"
    )

    plt.tight_layout()

    plt.savefig(out_path)

    plt.close()


def plot_residuals(
    y_true,
    y_pred,
    model_name,
    dataset,
    out_path
):

    residuals = y_true - y_pred

    plt.figure(figsize=(7, 5))

    plt.scatter(y_pred, residuals, alpha=0.7)

    plt.axhline(0, linestyle="--", color="red")

    plt.xlabel("Predicción")
    plt.ylabel("Residuo")

    plt.title(
        f"{model_name} - Residuos ({dataset})"
    )

    plt.tight_layout()

    plt.savefig(out_path)

    plt.close()


def plot_lasso_coefficients(
    coef,
    feature_names,
    dataset,
    out_path,
    top_n=20
):

    idx = np.argsort(np.abs(coef))[-top_n:]

    plt.figure(figsize=(10, 6))

    plt.barh(
        np.array(feature_names)[idx],
        coef[idx]
    )

    plt.xlabel("Coeficiente")

    plt.title(
        f"Lasso - Variables importantes ({dataset})"
    )

    plt.tight_layout()

    plt.savefig(out_path)

    plt.close()


def plot_pls_loadings(
    loadings,
    dataset,
    out_path
):

    plt.figure(figsize=(10, 5))

    plt.plot(loadings)

    plt.xlabel("Variable")
    plt.ylabel("Loading")

    plt.title(
        f"PLS - Loadings primera componente ({dataset})"
    )

    plt.tight_layout()

    plt.savefig(out_path)

    plt.close()



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
