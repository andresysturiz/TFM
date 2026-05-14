# scripts/compare_runner.py 

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.repo_tfm.scripts.core.logger import get_logger
from src.repo_tfm.scripts.core.utils import safe_create_dir
from src.repo_tfm.scripts.core.errors import DatasetError

logger = get_logger("CompareRunner")



# CARGA SEGURA DE RESULTADOS 

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


# FUNCIÓN PRINCIPAL 

def compare_results(dataset):
    logger.info(f"Comparando resultados para dataset: {dataset}")

    # DIRECTORIOS POR DATASET
    
    dataset_dir = f"results/{dataset}"
    csv_dir = f"{dataset_dir}/csv"
    fig_dir = f"{dataset_dir}/figures"

    safe_create_dir(dataset_dir)
    safe_create_dir(csv_dir)
    safe_create_dir(fig_dir)


    # REGRESIÓN

    pls_cv_path = f"{csv_dir}/pls_cv_{dataset}.csv"
    classical_path = f"{csv_dir}/classical_results_{dataset}.csv"
    pls_final_path = f"{csv_dir}/pls_final_results_{dataset}.csv"

    pls_df = load_results(pls_cv_path)
    classical_df = load_results(classical_path)

    # Mejor RMSE del PLS CV
    best_pls_rmse = pls_df["rmse_cv"].min()
    best_pls_components = pls_df.loc[pls_df["rmse_cv"].idxmin(), "n_components"]

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
        "train_time": None,
        "best_components": best_pls_components
    }

    # Combinar resultados
    comparison = pd.concat([
        classical_df,
        pd.DataFrame([pls_cv_row])
    ], ignore_index=True)

    # Añadir PLS FINAL si existe
    if os.path.exists(pls_final_path):
        pls_final_df = pd.read_csv(pls_final_path)
        comparison = pd.concat([comparison, pls_final_df], ignore_index=True)
        logger.info(f"PLS FINAL incluido en comparación: {pls_final_path}")
    else:
        logger.warning(f"No existe archivo PLS FINAL: {pls_final_path}")

    # Ordenar por RMSE
    if "rmse" in comparison.columns:
        comparison = comparison.sort_values(by="rmse", na_position="last").reset_index(drop=True)

    # Mostrar best_components si existe
    if "best_components" in comparison.columns:
        for _, row in comparison.iterrows():
            if not pd.isna(row.get("best_components", None)):
                logger.info(
                    f"Modelo {row['model']} seleccionó {int(row['best_components'])} componentes"
                )

    # Guardar CSV
    out_csv = f"{csv_dir}/comparison_{dataset}.csv"
    comparison.to_csv(out_csv, index=False)
    logger.info(f"Tabla comparativa guardada en {out_csv}")

    # Guardar LaTeX
    latex_path = f"{csv_dir}/comparison_{dataset}.tex"
    with open(latex_path, "w") as f:
        f.write(comparison.to_latex(index=False, float_format="%.4f"))
    logger.info(f"Tabla LaTeX guardada en {latex_path}")

    # Gráficos regresión
    _plot_regression_metrics(comparison, dataset, fig_dir)



# GRÁFICOS REGRESIÓN 

def _plot_regression_metrics(comparison, dataset, fig_dir):
    metrics = ["rmse", "mae", "mdae", "mape", "rmsle", "r2", "adj_r2", "nrmse"]

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

        out_png = f"{fig_dir}/{metric}_{dataset}.png"
        plt.savefig(out_png)
        plt.close()

        logger.info(f"Gráfico guardado: {out_png}")
 
