# main.py

import pandas as pd

from src.repo_tfm.scripts.core.logger import get_logger
from src.repo_tfm.scripts.core.utils import safe_create_dir
from src.repo_tfm.scripts.core.eda import DatasetAnalyzer
from src.repo_tfm.scripts.core.data_loader import DataLoader
from src.repo_tfm.scripts.experiments.pls_experiment import PLSExperiment
from src.repo_tfm.scripts.experiments.pls_final_experiment import PLSFinalExperiment
from src.repo_tfm.scripts.experiments.classical_experiment import ClassicalExperiment
from src.repo_tfm.scripts.compare_runner import compare_results
from src.repo_tfm.scripts.download_datasets import download_datasets

logger = get_logger("MainPipeline")


def run_pipeline_for_dataset(dataset):
    logger.info(f"=== Descargando dataset: {dataset} ===")

    download_datasets.main()
    
    logger.info(f"=== Procesando dataset: {dataset} ===")

    # ============================
    # CARGA DEL DATASET
    # ============================
    df, X, y = DataLoader().load(dataset)
    target_col = df.columns[-1]

    # ============================
    # EDA (opcional)
    # ============================
    DatasetAnalyzer().analyze(df, dataset_name=dataset, target=target_col)

    # ============================
    # 1. PLS CROSS-VALIDATION
    # ============================
    logger.info(f"Ejecutando PLSExperiment (CV) para {dataset}")
    PLSExperiment(dataset=dataset).run()

    # ============================
    # 2. PLS FINAL (AUTOMÁTICO)
    # ============================
    logger.info(f"Entrenando PLS FINAL para {dataset}")
    PLSFinalExperiment(dataset=dataset).run()

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

    # SOLO REGRESIÓN
    datasets = ["gasoline", "tecator", "riboflavin"]

    for dataset in datasets:
        run_pipeline_for_dataset(dataset)

    logger.info("=== PIPELINE COMPLETO FINALIZADO ===")


if __name__ == "__main__":
    main()
