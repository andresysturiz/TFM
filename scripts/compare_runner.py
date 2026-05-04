# scripts/compare_runner.py

import os
import pandas as pd
import matplotlib.pyplot as plt

from scripts.core.logger import get_logger
from scripts.core.utils import safe_create_dir
from scripts.core.errors import DatasetError

logger = get_logger("CompareRunner")


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


def compare_results(pls_cv_path, classical_path, output_prefix):
    logger.info(f"Comparando resultados para {output_prefix}")

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

    # Unificar tabla inicial
    comparison = pd.concat([
        classical_df,
        pd.DataFrame([pls_cv_row])
    ], ignore_index=True)

    # ============================
    # CARGAR PLS FINAL
    # ============================

    pls_final_path = f"results/pls_final_results_{output_prefix}.csv"

    if os.path.exists(pls_final_path):
        pls_final_df = pd.read_csv(pls_final_path)
        comparison = pd.concat([comparison, pls_final_df], ignore_index=True)
        logger.info(f"PLS FINAL incluido en comparación: {pls_final_path}")
    else:
        logger.warning(f"No existe archivo PLS FINAL: {pls_final_path}")

    # Guardar tabla comparativa
    out_csv = f"results/comparison_{output_prefix}.csv"
    comparison.to_csv(out_csv, index=False)
    logger.info(f"Tabla comparativa guardada en {out_csv}")

    # ============================
    # GRÁFICOS POR MÉTRICA
    # ============================

    metrics = [
        "rmse", "mae", "mdae", "mape",
        "rmsle", "r2", "adj_r2", "nrmse"
    ]

    for metric in metrics:
        if metric not in comparison.columns:
            continue

        # Filtrar métricas sin valores numéricos
        if comparison[metric].dropna().empty:
            logger.warning(f"No hay valores numéricos para {metric}, se omite gráfico.")
            continue

        # Filtrar filas válidas
        df_plot = comparison.dropna(subset=[metric])

        plt.figure(figsize=(10, 5))
        plt.bar(df_plot["model"], df_plot[metric], color="steelblue")
        plt.xticks(rotation=45)
        plt.ylabel(metric.upper())
        plt.title(f"Comparación de {metric.upper()} - {output_prefix}")
        plt.tight_layout()

        out_png = f"results/{metric}_{output_prefix}.png"
        plt.savefig(out_png)
        plt.close()

        logger.info(f"Gráfico guardado: {out_png}")


def main():
    safe_create_dir("results")

    datasets = [
        ("airfoil_self_noise",
         "results/pls_cv_airfoil_self_noise.csv",
         "results/classical_results_airfoil_self_noise.csv"),

        ("wine_quality",
         "results/pls_cv_wine_quality.csv",
         "results/classical_results_wine_quality.csv"),
    ]

    for name, pls_path, classical_path in datasets:
        try:
            compare_results(pls_path, classical_path, name)
        except Exception as e:
            logger.error(f"Error comparando {name}: {str(e)}")


if __name__ == "__main__":
    main()
