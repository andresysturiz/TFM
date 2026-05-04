# scripts/compare_runner.py  (VERSIÓN PRO)

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scripts.core.logger import get_logger
from scripts.core.utils import safe_create_dir
from scripts.core.errors import DatasetError

logger = get_logger("CompareRunner")


# ============================================================
# CARGA SEGURA DE RESULTADOS
# ============================================================

def load_results(path):
    if not os.path.exists(path):
        logger.error(f"Archivo no encontrado: {path}")
        raise DatasetError(f"No existe el archivo: {path}")

    try:
        df = pd.read_csv(path)
        logger.info(f"Resultados cargados: {path}")
        return df
    except Exception as e:
        logger.error(f"Error leyendo {path}: {str(e)}")
        raise DatasetError(f"Error leyendo {path}")


# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

def compare_results(dataset):
    logger.info(f"Comparando resultados para dataset: {dataset}")

    # -------------------------
    # RUTAS AUTOMÁTICAS
    # -------------------------
    pls_cv_path = f"results/pls_cv_{dataset}.csv"
    classical_path = f"results/classical_results_{dataset}.csv"
    pls_final_path = f"results/pls_final_results_{dataset}.csv"

    # -------------------------
    # CARGAR RESULTADOS
    # -------------------------
    pls_df = load_results(pls_cv_path)
    classical_df = load_results(classical_path)

    # Mejor RMSE del PLS CV
    best_pls_rmse = pls_df["rmse_cv"].min()
    best_pls_components = pls_df.loc[pls_df["rmse_cv"].idxmin(), "n_components"]

    # Crear fila PLS CV
    pls_cv_row = {
        "model": f"PLS_CV_{int(best_pls_components)}_components",
        "mse": None,
        "rmse": best_pls_rmse,
        "mae": None,
        "mdae": None,
        "mape": None,
        "rmsle": None,
        "r2": None,
        "adj_r2": None,
        "nrmse": None,
        "train_time": None
    }

    # Tabla inicial
    comparison = pd.concat([
        classical_df,
        pd.DataFrame([pls_cv_row])
    ], ignore_index=True)

    # -------------------------
    # CARGAR PLS FINAL
    # -------------------------
    if os.path.exists(pls_final_path):
        pls_final_df = pd.read_csv(pls_final_path)
        comparison = pd.concat([comparison, pls_final_df], ignore_index=True)
        logger.info(f"PLS FINAL incluido en comparación: {pls_final_path}")
    else:
        logger.warning(f"No existe archivo PLS FINAL: {pls_final_path}")

    # -------------------------
    # ORDENAR POR RMSE
    # -------------------------
    if "rmse" in comparison.columns:
        comparison = comparison.sort_values(by="rmse", na_position="last").reset_index(drop=True)

    # -------------------------
    # GUARDAR TABLA
    # -------------------------
    out_csv = f"results/comparison_{dataset}.csv"
    comparison.to_csv(out_csv, index=False)
    logger.info(f"Tabla comparativa guardada en {out_csv}")

    # -------------------------
    # TABLA LATEX PARA EL TFM
    # -------------------------
    latex_path = f"results/comparison_{dataset}.tex"
    with open(latex_path, "w") as f:
        f.write(comparison.to_latex(index=False, float_format="%.4f"))
    logger.info(f"Tabla LaTeX guardada en {latex_path}")

    # -------------------------
    # GRÁFICOS POR MÉTRICA
    # -------------------------
    metrics = [
        "rmse", "mae", "mdae", "mape",
        "rmsle", "r2", "adj_r2", "nrmse"
    ]

    colors = plt.cm.tab20(np.linspace(0, 1, len(comparison)))

    for metric in metrics:
        if metric not in comparison.columns:
            continue

        df_plot = comparison.dropna(subset=[metric])
        if df_plot.empty:
            continue

        plt.figure(figsize=(10, 5))
        plt.bar(df_plot["model"], df_plot[metric], color=colors[:len(df_plot)])
        plt.xticks(rotation=45)
        plt.ylabel(metric.upper())
        plt.title(f"Comparación de {metric.upper()} - {dataset}")
        plt.tight_layout()

        out_png = f"results/{metric}_{dataset}.png"
        plt.savefig(out_png)
        plt.close()

        logger.info(f"Gráfico guardado: {out_png}")

    # -------------------------
    # GRÁFICO DE RANKING GLOBAL
    # -------------------------
    if "rmse" in comparison.columns:
        plt.figure(figsize=(10, 5))
        plt.plot(comparison["model"], comparison["rmse"], marker="o", color="darkred")
        plt.xticks(rotation=45)
        plt.ylabel("RMSE")
        plt.title(f"Ranking global por RMSE - {dataset}")
        plt.tight_layout()

        out_png = f"results/ranking_rmse_{dataset}.png"
        plt.savefig(out_png)
        plt.close()

        logger.info(f"Gráfico ranking guardado: {out_png}")

    # -------------------------
    # RADAR CHART (si hay métricas suficientes)
    # -------------------------
    radar_metrics = ["rmse", "mae", "mdae", "nrmse"]

    if all(m in comparison.columns for m in radar_metrics):
        df_radar = comparison.dropna(subset=radar_metrics)

        if not df_radar.empty:
            labels = radar_metrics
            angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
            angles += angles[:1]

            plt.figure(figsize=(8, 8))
            ax = plt.subplot(111, polar=True)

            for i, row in df_radar.iterrows():
                values = row[labels].tolist()
                values += values[:1]
                ax.plot(angles, values, linewidth=2, label=row["model"])
                ax.fill(angles, values, alpha=0.1)

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels)
            plt.title(f"Radar Chart - {dataset}")
            plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

            out_png = f"results/radar_{dataset}.png"
            plt.savefig(out_png)
            plt.close()

            logger.info(f"Radar chart guardado: {out_png}")


def main():
    safe_create_dir("results")

    datasets = ["airfoil", "wine", "gasoline"]

    for dataset in datasets:
        try:
            compare_results(dataset)
        except Exception as e:
            logger.error(f"Error comparando {dataset}: {str(e)}")


if __name__ == "__main__":
    main()
