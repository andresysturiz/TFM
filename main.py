# main.py

import pandas as pd

from scripts.core.logger import get_logger
from scripts.core.utils import safe_create_dir
from scripts.core.eda import DatasetAnalyzer
from scripts.core.data_loader import DataLoader
from scripts.experiments.pls_experiment import PLSExperiment
from scripts.experiments.pls_final_experiment import PLSFinalExperiment
from scripts.experiments.classical_experiment import ClassicalExperiment
from scripts.compare_runner import compare_results

logger = get_logger("MainPipeline")


def run_pipeline_for_dataset(dataset):
    logger.info(f"=== Procesando dataset: {dataset} ===")

    # ============================
    # 0. EDA AUTOMÁTICO
    # ============================
    df, X, y = DataLoader().load(dataset)

    # Detectar automáticamente la variable objetivo
    # (última columna del DataFrame)
    target_col = df.columns[-1]

    DatasetAnalyzer().analyze(df, dataset_name=dataset, target=target_col)

    # ============================
    # 1. PLS (VALIDACIÓN CRUZADA)
    # ============================
    logger.info(f"Ejecutando PLSExperiment (CV) para {dataset}")
    PLSExperiment(dataset=dataset).run()

    # Leer mejor número de componentes
    pls_cv_path = f"results/pls_cv_{dataset}.csv"
    pls_cv_df = pd.read_csv(pls_cv_path)
    best_components = int(pls_cv_df.loc[pls_cv_df["rmse_cv"].idxmin(), "n_components"])

    # ============================
    # 2. PLS FINAL (TRAIN/TEST)
    # ============================
    logger.info(f"Entrenando PLS FINAL para {dataset}")
    PLSFinalExperiment(dataset=dataset, best_components=best_components).run()

    # ============================
    # 3. MODELOS CLÁSICOS
    # ============================
    logger.info(f"Ejecutando ClassicalExperiment para {dataset}")
    ClassicalExperiment(dataset=dataset).run()

    # ============================
    # 4. COMPARACIÓN FINAL
    # ============================
    logger.info(f"Comparando resultados para {dataset}")
    compare_results(dataset)

    logger.info(f"=== Dataset {dataset} procesado correctamente ===")


def main():
    logger.info("=== INICIANDO PIPELINE COMPLETO ===")

    safe_create_dir("results")

    # Datasets soportados
    datasets = ["airfoil", "wine", "gasoline"]

    for dataset in datasets:
        run_pipeline_for_dataset(dataset)

    logger.info("=== PIPELINE COMPLETO FINALIZADO ===")


if __name__ == "__main__":
    main()
