# scripts/core/preprocessing.py

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.repo_tfm.scripts.core.logger import get_logger

logger = get_logger("Preprocessing")


# ============================================================
# PREPROCESADO PARA REGRESIÓN (Gasoline, Tecator, Riboflavin)
# ============================================================

def preprocess_regression(X, y, test_size=0.2, random_state=42):
    """
    Preprocesado estándar para REGRESIÓN:
    - train/test split
    - escalado estándar de X
    - y se deja sin escalar (coherente con métricas)
    """
    logger.info("Preprocesando dataset de regresión")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
