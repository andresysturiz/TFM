# scripts/core/eda.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.repo_tfm.scripts.core.utils import safe_create_dir
from src.repo_tfm.scripts.core.logger import get_logger

logger = get_logger("DatasetAnalyzer")


class DatasetAnalyzer:

    def __init__(self):
        pass

    def analyze(self, df, dataset_name, target=None):
        """
        EDA compacto y profesional:
        - Estadísticos descriptivos ampliados
        - Información estructural del dataset
        - Correlación media con el target
        - Gráficos mínimos y útiles
        """

        base_dir = f"results/{dataset_name}/eda"
        safe_create_dir(base_dir)

        logger.info(f"=== Iniciando EDA para dataset: {dataset_name} ===")

        # ============================================================
        # 1. Información estructural
        # ============================================================

        info = {
            "n_observaciones": df.shape[0],
            "n_variables": df.shape[1],
            "ratio_p_n": df.shape[1] / df.shape[0],
        }

        info_path = f"{base_dir}/eda_{dataset_name}_info.csv"
        pd.DataFrame([info]).to_csv(info_path, index=False)
        logger.info(f"Información estructural guardada en {info_path}")

        # ============================================================
        # 2. Estadísticos descriptivos ampliados
        # ============================================================

        desc = df.describe().T
        desc["coef_var"] = desc["std"] / desc["mean"]
        desc["skew"] = df.skew(numeric_only=True)
        desc["kurtosis"] = df.kurtosis(numeric_only=True)

        summary_path = f"{base_dir}/eda_{dataset_name}_summary.csv"
        desc.to_csv(summary_path)
        logger.info(f"Resumen estadístico guardado en {summary_path}")

        # ============================================================
        # 3. Valores faltantes
        # ============================================================

        missing = df.isna().sum()
        missing_path = f"{base_dir}/eda_{dataset_name}_missing.csv"
        missing.to_csv(missing_path)
        logger.info(f"Valores faltantes guardados en {missing_path}")

        # ============================================================
        # 4. Correlación media con el target
        # ============================================================

        if target is not None:
            corrs = df.corr(numeric_only=True)[target].drop(target)
            corr_info = {
                "correlacion_media": corrs.abs().mean(),
                "correlacion_max": corrs.abs().max(),
                "correlacion_min": corrs.abs().min(),
            }
            corr_path = f"{base_dir}/eda_{dataset_name}_target_corr.csv"
            pd.DataFrame([corr_info]).to_csv(corr_path, index=False)
            logger.info(f"Correlación con target guardada en {corr_path}")

        # ============================================================
        # 5. Gráficos mínimos
        # ============================================================

        if target is not None:
            self._plot_target(df, dataset_name, target, base_dir)

        # Heatmap solo si p ≤ 100
        if df.shape[1] <= 100:
            self._plot_heatmap(df, dataset_name, base_dir)

        # Análisis espectral para Gasoline y Tecator
        if dataset_name in ["gasoline", "tecator"]:
            self._plot_spectra(df, dataset_name, target, base_dir)

        logger.info(f"=== EDA finalizado para {dataset_name} ===")

    # ============================================================
    # FUNCIONES AUXILIARES
    # ============================================================

    def _plot_target(self, df, dataset_name, target, base_dir):
        y = df[target]

        plt.figure(figsize=(6, 4))
        sns.histplot(y, kde=True, color="green")
        plt.title(f"Distribución de {target}")
        plt.tight_layout()
        plt.savefig(f"{base_dir}/eda_{dataset_name}_target_hist.png")
        plt.close()

        plt.figure(figsize=(6, 4))
        sns.boxplot(x=y, color="purple")
        plt.title(f"Boxplot de {target}")
        plt.tight_layout()
        plt.savefig(f"{base_dir}/eda_{dataset_name}_target_box.png")
        plt.close()

    def _plot_heatmap(self, df, dataset_name, base_dir):
        corr = df.corr(numeric_only=True)
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, cmap="coolwarm", annot=False)
        plt.title(f"Heatmap de correlación - {dataset_name}")
        plt.tight_layout()
        plt.savefig(f"{base_dir}/eda_{dataset_name}_corr_heatmap.png")
        plt.close()

    def _plot_spectra(self, df, dataset_name, target, base_dir):
        X = df.drop(columns=[target]).values

        mean_spectrum = X.mean(axis=0)
        plt.figure(figsize=(10, 5))
        plt.plot(mean_spectrum)
        plt.title(f"Espectro medio - {dataset_name}")
        plt.tight_layout()
        plt.savefig(f"{base_dir}/eda_{dataset_name}_mean_spectrum.png")
        plt.close()

        var_spectrum = X.var(axis=0)
        plt.figure(figsize=(10, 5))
        plt.plot(var_spectrum, color="red")
        plt.title(f"Varianza espectral - {dataset_name}")
        plt.tight_layout()
        plt.savefig(f"{base_dir}/eda_{dataset_name}_variance_spectrum.png")
        plt.close()
