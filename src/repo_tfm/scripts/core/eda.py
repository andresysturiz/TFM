# scripts/core/eda.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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


        # 1. Información estructural


        info = {
            "n_observaciones": df.shape[0],
            "n_variables": df.shape[1],
            "ratio_p_n": df.shape[1] / df.shape[0],
        }

        info_path = f"{base_dir}/eda_{dataset_name}_info.csv"
        pd.DataFrame([info]).to_csv(info_path, index=False)
        logger.info(f"Información estructural guardada en {info_path}")


        # 2. Estadísticos descriptivos ampliados


        desc = df.describe().T
        desc["coef_var"] = desc["std"] / desc["mean"]
        desc["skew"] = df.skew(numeric_only=True)
        desc["kurtosis"] = df.kurtosis(numeric_only=True)

        summary_path = f"{base_dir}/eda_{dataset_name}_summary.csv"
        desc.to_csv(summary_path)
        logger.info(f"Resumen estadístico guardado en {summary_path}")

 
        # 3. Valores faltantes


        missing = df.isna().sum()
        missing_path = f"{base_dir}/eda_{dataset_name}_missing.csv"
        missing.to_csv(missing_path)
        logger.info(f"Valores faltantes guardados en {missing_path}")


        # 4. Correlación media con el target


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


        # 5. Gráficos mínimos


        if target is not None:
            self._plot_target(df, dataset_name, target, base_dir)

        # Heatmap solo si p ≤ 100
        if dataset_name == "tecator":
            self._plot_heatmap(df, dataset_name, base_dir)

        # PCA preliminar
        if target is not None:
            self._plot_pca(df, dataset_name, target, base_dir)

        # Análisis espectral para Gasoline y Tecator
        if dataset_name in ["gasoline", "tecator"]:
            self._plot_spectra(df, dataset_name, target, base_dir)

        # Análisis específico para Riboflavin
        if dataset_name == "riboflavin":
            self._plot_riboflavin_analysis(df, dataset_name, target, base_dir)

        logger.info(f"=== EDA finalizado para {dataset_name} ===")


    # FUNCIONES AUXILIARES


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
        std_spectrum = X.std(axis=0)

        wavelengths = np.arange(X.shape[1])

        plt.figure(figsize=(10, 5))

        # Algunas curvas individuales suaves
        for i in range(min(15, X.shape[0])):
            plt.plot(
                wavelengths,
                X[i],
                alpha=0.12,
                linewidth=1
            )

        # Banda de variabilidad
        plt.fill_between(
            wavelengths,
            mean_spectrum - std_spectrum,
            mean_spectrum + std_spectrum,
            color="red",
            alpha=0.25,
            label="±1 desviación típica"
        )

        # Espectro medio
        plt.plot(
            wavelengths,
            mean_spectrum,
            color="black",
            linewidth=1,
            label="Espectro medio"
        )

        plt.xlabel("Longitud de onda")
        plt.ylabel("Absorbancia")

        plt.title(f"Curvas espectrales - {dataset_name}")

        plt.legend()

        plt.tight_layout()

        plt.savefig(f"{base_dir}/eda_{dataset_name}_spectra.png")

        plt.close()



    '''def _plot_spectra(self, df, dataset_name, target, base_dir):
        X = df.drop(columns=[target]).values

        mean_spectrum = X.mean(axis=0)

        plt.figure(figsize=(10, 5))

        # Curvas espectrales individuales
        for i in range(min(20, X.shape[0])):
            plt.plot(
                X[i],
                alpha=0.3
            )

        # Espectro medio
        plt.plot(
            mean_spectrum,
            linewidth=1.5,
            color="black",
            label="Espectro medio"
        )

        plt.legend()
        plt.title(f"Curvas espectrales y espectro medio - {dataset_name}")
        plt.tight_layout()
        plt.savefig(f"{base_dir}/eda_{dataset_name}_mean_spectrum.png")
        plt.close()

        var_spectrum = X.var(axis=0)
        plt.figure(figsize=(10, 5))
        plt.plot(var_spectrum, color="red")
        plt.title(f"Varianza espectral - {dataset_name}")
        plt.tight_layout()
        plt.savefig(f"{base_dir}/eda_{dataset_name}_variance_spectrum.png")
        plt.close()'''

    def _plot_pca(self, df, dataset_name, target, base_dir):
        """
        PCA preliminar:
        - Scree plot
        - Scores plot PC1 vs PC2
        """

        X = df.drop(columns=[target]).values
        y = df[target].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        max_comp = min(20, X_scaled.shape[0], X_scaled.shape[1])

        pca = PCA(n_components=max_comp)
        X_pca = pca.fit_transform(X_scaled)

        # Scree plot

        explained = pca.explained_variance_ratio_
        cumulative = np.cumsum(explained)

        plt.figure(figsize=(8, 5))

        plt.plot(
            range(1, len(explained) + 1),
            explained,
            marker="o",
            label="Varianza individual"
        )

        plt.plot(
            range(1, len(cumulative) + 1),
            cumulative,
            marker="s",
            linestyle="--",
            label="Varianza acumulada"
        )

        plt.legend()

        plt.xlabel("Número de componentes")
        plt.ylabel("Varianza explicada")
        plt.title(f"Scree Plot PCA - {dataset_name}")
        plt.tight_layout()

        plt.savefig(f"{base_dir}/eda_{dataset_name}_pca_scree.png")
        plt.close()

        # =========================
        # Scores PCA
        # =========================

        plt.figure(figsize=(7, 6))

        scatter = plt.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=y,
            cmap="viridis"
        )

        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title(f"PCA Scores Plot - {dataset_name}")

        plt.colorbar(scatter, label=target)

        plt.tight_layout()

        plt.savefig(f"{base_dir}/eda_{dataset_name}_pca_scores.png")
        plt.close()


    def _plot_riboflavin_analysis(self, df, dataset_name, target, base_dir):
        """
        EDA específico para Riboflavin:
        - Histograma de varianzas
        - Heatmap top genes
        """

        X = df.drop(columns=[target])

        # Histograma de varianzas

        variances = X.var()

        plt.figure(figsize=(8, 5))

        plt.hist(
            variances,
            bins=50
        )

        plt.xlabel("Varianza")
        plt.ylabel("Frecuencia")
        plt.title("Distribución de varianzas génicas")

        plt.tight_layout()

        plt.savefig(f"{base_dir}/eda_{dataset_name}_variance_hist.png")
        plt.close()

        # Top genes por varianza

        top_genes = variances.sort_values(
            ascending=False
        ).head(50).index

        corr = X[top_genes].corr()

        plt.figure(figsize=(12, 10))

        sns.heatmap(
            corr,
            cmap="coolwarm",
            center=0
        )

        plt.title("Heatmap Top-50 genes más variables")

        plt.tight_layout()

        plt.savefig(f"{base_dir}/eda_{dataset_name}_top_genes_heatmap.png")
        plt.close()