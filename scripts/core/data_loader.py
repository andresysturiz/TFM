# scripts/core/data_loader.py

import pandas as pd
from scripts.core.utils import safe_read_csv
from scripts.core.logger import get_logger

logger = get_logger("DataLoader")


class DataLoader:
    """
    Clase para cargar datasets por nombre.
    Devuelve (df, X, y) para regresión.
    Devuelve (expr_df, labels_df) para Golub (clasificación).
    """

    def __init__(self, base_path="data/raw"):
        self.base_path = base_path

    # -------------------------
    # MÉTODO PRINCIPAL
    # -------------------------
    def load(self, name):
        name = name.lower()

        if name == "wine":
            return self._load_wine()

        if name == "airfoil":
            return self._load_airfoil()

        if name == "gasoline":
            return self._load_gasoline()

        if name == "golub":
            return self._load_golub()

        raise ValueError(f"Dataset '{name}' no está soportado.")

    # -------------------------
    # WINE QUALITY
    # -------------------------
    def _load_wine(self):
        path = f"{self.base_path}/wine_quality.csv"
        df = safe_read_csv(path)

        X = df.drop(columns=["quality"]).values
        y = df["quality"].values

        logger.info(f"Wine cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
        return df, X, y

    # -------------------------
    # AIRFOIL SELF-NOISE
    # -------------------------
    def _load_airfoil(self):
        path = f"{self.base_path}/airfoil_self_noise.csv"
        df = safe_read_csv(path)

        X = df.drop(columns=["sound_pressure_level"]).values
        y = df["sound_pressure_level"].values

        logger.info(f"Airfoil cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
        return df, X, y

    # -------------------------
    # GASOLINE (regresión)
    # -------------------------
    def _load_gasoline(self):
        path = f"{self.base_path}/gasoline.csv"
        df = safe_read_csv(path)

        if "octane" not in df.columns:
            raise ValueError("El dataset Gasoline no contiene la columna 'octane'.")

        X = df.drop(columns=["octane"]).values
        y = df["octane"].values

        logger.info(f"Gasoline cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
        return df, X, y

    # -------------------------
    # GOLUB (clasificación)
    # -------------------------
    def _load_golub(self):
        expr_path = f"{self.base_path}/golub_expression.csv"
        labels_path = f"{self.base_path}/golub_labels.csv"

        expr = safe_read_csv(expr_path)
        labels = safe_read_csv(labels_path)

        if "class" not in labels.columns:
            raise ValueError("El archivo de etiquetas Golub no contiene la columna 'class'.")

        logger.info(f"Golub cargado: expr={expr.shape}, labels={labels.shape}")
        return expr, labels
