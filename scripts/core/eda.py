# scripts/core/eda.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scripts.core.utils import safe_create_dir
from scripts.core.logger import get_logger

logger = get_logger("DatasetAnalyzer")


class DatasetAnalyzer:

    def __init__(self):
        safe_create_dir("results")

    def analyze(self, df, dataset_name, target=None):
        """
        Genera análisis exploratorio automático del dataset.
        Guarda gráficos y tablas en results/.
        """

        logger.info(f"=== Iniciando EDA para dataset: {dataset_name} ===")

        # -------------------------
        # 1. Información general
        # -------------------------
        summary_path = f"results/eda_{dataset_name}_summary.csv"
        df.describe().T.to_csv(summary_path)
        logger.info(f"Resumen estadístico guardado en {summary_path}")

        # -------------------------
        # 2. Valores faltantes
        # -------------------------
        missing = df.isna().sum()
        missing_path = f"results/eda_{dataset_name}_missing.csv"
        missing.to_csv(missing_path)
        logger.info(f"Valores faltantes guardados en {missing_path}")

        # -------------------------
        # 3. Histogramas
        # -------------------------
        for col in df.columns:
            plt.figure(figsize=(6, 4))
            sns.histplot(df[col], kde=True, color="steelblue")
            plt.title(f"Histograma - {col}")
            plt.tight_layout()
            out = f"results/eda_{dataset_name}_hist_{col}.png"
            plt.savefig(out)
            plt.close()
            logger.info(f"Histograma guardado: {out}")

        # -------------------------
        # 4. Boxplots
        # -------------------------
        for col in df.columns:
            plt.figure(figsize=(6, 4))
            sns.boxplot(x=df[col], color="orange")
            plt.title(f"Boxplot - {col}")
            plt.tight_layout()
            out = f"results/eda_{dataset_name}_box_{col}.png"
            plt.savefig(out)
            plt.close()
            logger.info(f"Boxplot guardado: {out}")

        # -------------------------
        # 5. Matriz de correlación
        # -------------------------
        if df.shape[1] <= 200:  # evitar heatmaps gigantes (Gasoline)
            corr = df.corr(numeric_only=True)
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr, cmap="coolwarm", annot=False)
            plt.title(f"Heatmap de correlación - {dataset_name}")
            plt.tight_layout()
            out = f"results/eda_{dataset_name}_corr_heatmap.png"
            plt.savefig(out)
            plt.close()
            logger.info(f"Heatmap de correlación guardado: {out}")

        # -------------------------
        # 6. Análisis específico por dataset
        # -------------------------
        if dataset_name == "gasoline":
            self._analyze_gasoline(df, dataset_name)

        if target is not None:
            self._analyze_target(df, dataset_name, target)

        logger.info(f"=== EDA finalizado para {dataset_name} ===")

    # ============================================================
    # ANÁLISIS ESPECÍFICO PARA GASOLINE (ESPECTROS)
    # ============================================================

    def _analyze_gasoline(self, df, dataset_name):
        logger.info("Análisis específico para Gasoline (espectros)")

        X = df.drop(columns=["octane"]).values

        # Espectro medio
        mean_spectrum = X.mean(axis=0)
        plt.figure(figsize=(10, 5))
        plt.plot(mean_spectrum)
        plt.title("Espectro medio - Gasoline")
        plt.xlabel("Longitud de onda (índice)")
        plt.ylabel("Absorbancia")
        out = f"results/eda_{dataset_name}_mean_spectrum.png"
        plt.savefig(out)
        plt.close()
        logger.info(f"Espectro medio guardado: {out}")

        # Varianza por longitud de onda
        var_spectrum = X.var(axis=0)
        plt.figure(figsize=(10, 5))
        plt.plot(var_spectrum, color="red")
        plt.title("Varianza por longitud de onda - Gasoline")
        plt.xlabel("Longitud de onda (índice)")
        plt.ylabel("Varianza")
        out = f"results/eda_{dataset_name}_variance_spectrum.png"
        plt.savefig(out)
        plt.close()
        logger.info(f"Varianza espectral guardada: {out}")

    # ============================================================
    # ANÁLISIS DE LA VARIABLE OBJETIVO
    # ============================================================

    def _analyze_target(self, df, dataset_name, target):
        logger.info(f"Análisis de la variable objetivo: {target}")

        y = df[target]

        # Histograma
        plt.figure(figsize=(6, 4))
        sns.histplot(y, kde=True, color="green")
        plt.title(f"Distribución de {target}")
        plt.tight_layout()
        out = f"results/eda_{dataset_name}_target_hist.png"
        plt.savefig(out)
        plt.close()

        # Boxplot
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=y, color="purple")
        plt.title(f"Boxplot de {target}")
        plt.tight_layout()
        out = f"results/eda_{dataset_name}_target_box.png"
        plt.savefig(out)
        plt.close()

        logger.info(f"Gráficos de la variable objetivo guardados para {dataset_name}")
