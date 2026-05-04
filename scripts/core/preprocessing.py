# scripts/core/preprocessing.py

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scripts.core.logger import get_logger

logger = get_logger("Preprocessing")


# ============================================================
# PREPROCESADO PARA REGRESIÓN (Wine, Airfoil, Gasoline)
# ============================================================

def preprocess_regression(X, y, test_size=0.2, random_state=42):
    """
    Preprocesado estándar para regresión:
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


# ============================================================
# PREPROCESADO PARA GOLUB (Clasificación p >> n)
# ============================================================

def preprocess_golub(expr_df, labels_df, test_size=0.2, random_state=42):
    """
    Preprocesado para Golub (clasificación p >> n):
    - log-transform
    - transponer matriz (muestras en filas)
    - escalado
    - train/test split estratificado
    """
    logger.info("Preprocesando dataset Golub (p >> n)")

    # Log-transform para estabilizar varianzas
    X = np.log1p(expr_df.values.T)  # shape = (n_samples, n_genes)

    # Etiquetas
    y = labels_df["class"].values

    # Split estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Escalado
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
