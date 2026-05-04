# scripts/core/data_loader.py

import pandas as pd
from src.repo_tfm.scripts.core.utils import safe_read_csv
from src.repo_tfm.scripts.core.logger import get_logger

logger = get_logger("DataLoader")


class DataLoader:
    """
    Clase para cargar datasets de REGRESIÓN por nombre.
    Devuelve (df, X, y) para todos los datasets soportados.
    """

    def __init__(self, base_path="data/raw"):
        self.base_path = base_path

    # -------------------------
    # MÉTODO PRINCIPAL
    # -------------------------
    def load(self, name):
        name = name.lower()

        if name == "gasoline":
            return self._load_gasoline()

        if name == "tecator":
            return self._load_tecator()

        if name == "riboflavin":
            return self._load_riboflavin()

        raise ValueError(f"Dataset '{name}' no está soportado.")

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
    # TECATOR (regresión)
    # -------------------------
    def _load_tecator(self):
        path = f"{self.base_path}/tecator.csv"
        df = safe_read_csv(path)

        # Objetivo estándar
        if "fat" not in df.columns:
            raise ValueError("El dataset Tecator no contiene la columna 'fat'.")

        # Usamos solo absorbancias como predictores
        feature_cols = [c for c in df.columns if c.startswith("absorbance_")]

        X = df[feature_cols].values
        y = df["fat"].values

        logger.info(f"Tecator cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
        return df, X, y


    # -------------------------
    # RIBOFLAVIN (regresión)
    # -------------------------
    def _load_riboflavin(self):
        path = f"{self.base_path}/riboflavin.csv"
        df = safe_read_csv(path)

        if "x" not in df.columns:
            raise ValueError("El dataset Riboflavin no contiene la columna 'x'.")

        X = df.drop(columns=["x"]).values
        y = df["x"].values

        logger.info(f"Riboflavin cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
        return df, X, y
