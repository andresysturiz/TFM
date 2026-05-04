# main.py

import pandas as pd
from scripts.core.logger import get_logger
from scripts.core.utils import safe_create_dir
from scripts.core.errors import DatasetError

from scripts.experiments.pls_experiment import PLSExperiment
from scripts.experiments.classical_experiment import ClassicalExperiment
from scripts.experiments.pls_final_experiment import PLSFinalExperiment
from scripts.compare_runner import main as compare_main

logger = get_logger("MainPipeline")


# ============================
# VALIDACIÓN DE COLUMNAS
# ============================

SCHEMAS = {
    "airfoil": [
        "frequency",
        "attack-angle",
        "chord-length",
        "free-stream-velocity",
        "suction-side-displacement-thickness",
        "scaled-sound-pressure"
    ],
    "wine": [
        "fixed_acidity",
        "volatile_acidity",
        "citric_acid",
        "residual_sugar",
        "chlorides",
        "free_sulfur_dioxide",
        "total_sulfur_dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol",
        "quality"
    ]
}


def validate_dataset(path, schema_name):
    logger.info(f"Validando dataset: {path}")

    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise DatasetError(f"No se pudo leer el dataset {path}: {str(e)}")

    expected_cols = SCHEMAS[schema_name]
    missing = [c for c in expected_cols if c not in df.columns]

    if missing:
        raise DatasetError(f"Columnas faltantes en {path}: {missing}")

    logger.info(f"Validación correcta para {path}")


# ============================
# PIPELINE PRINCIPAL
# ============================

def main():
    logger.info("=== INICIANDO PIPELINE COMPLETO ===")

    safe_create_dir("results")

    # ============================
    # 1. VALIDACIÓN DE DATASETS
    # ============================

    validate_dataset("data/raw/airfoil_self_noise.csv", "airfoil")
    validate_dataset("data/raw/wine_quality.csv", "wine")

    # ============================
    # 2. PLS (VALIDACIÓN CRUZADA)
    # ============================

    logger.info("Ejecutando PLSExperiment (CV)")

    PLSExperiment(
        "Airfoil",
        "data/raw/airfoil_self_noise.csv",
        "scaled-sound-pressure"
    ).run()

    PLSExperiment(
        "Wine",
        "data/raw/wine_quality.csv",
        "quality"
    ).run()

    # ============================
    # 3. PLS FINAL (TRAIN/TEST)
    # ============================

    logger.info("Entrenando PLS FINAL")

    # Airfoil
    pls_cv_airfoil = pd.read_csv("results/pls_cv_airfoil_self_noise.csv")
    best_airfoil_components = int(pls_cv_airfoil.loc[pls_cv_airfoil["rmse_cv"].idxmin(), "n_components"])

    PLSFinalExperiment(
        "Airfoil",
        "data/raw/airfoil_self_noise.csv",
        "scaled-sound-pressure",
        best_airfoil_components
    ).run()

    # Wine
    pls_cv_wine = pd.read_csv("results/pls_cv_wine_quality.csv")
    best_wine_components = int(pls_cv_wine.loc[pls_cv_wine["rmse_cv"].idxmin(), "n_components"])

    PLSFinalExperiment(
        "Wine",
        "data/raw/wine_quality.csv",
        "quality",
        best_wine_components
    ).run()

    # ============================
    # 4. MODELOS CLÁSICOS
    # ============================

    logger.info("Ejecutando ClassicalExperiment")

    ClassicalExperiment(
        "Airfoil",
        "data/raw/airfoil_self_noise.csv",
        "scaled-sound-pressure"
    ).run()

    ClassicalExperiment(
        "Wine",
        "data/raw/wine_quality.csv",
        "quality"
    ).run()

    # ============================
    # 5. COMPARACIÓN FINAL
    # ============================

    logger.info("Ejecutando CompareRunner")
    compare_main()

    logger.info("=== PIPELINE COMPLETO FINALIZADO ===")


if __name__ == "__main__":
    main()
