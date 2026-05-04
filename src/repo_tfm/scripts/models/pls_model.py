from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, roc_auc_score
import numpy as np

from src.repo_tfm.scripts.core.logger import get_logger

logger = get_logger("PLSModel")


class PLSModel:
    """
    PLS para regresión y clasificación (PLS-DA) con selección automática de componentes.
    """

    def __init__(self, n_components=None, max_components=20):
        self.n_components = n_components
        self.max_components = max_components
        self.model = None
        self.best_components_ = None
        self.is_classification = False

    # ============================================================
    # Selección automática de componentes
    # ============================================================

    def _select_components_cv(self, X, y):
        logger.info("Seleccionando número óptimo de componentes para PLS...")

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        max_comp = min(self.max_components, X.shape[1])

        scores = []

        for n in range(1, max_comp + 1):
            fold_scores = []

            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                pls = PLSRegression(n_components=n)
                pls.fit(X_train, y_train)

                pred = pls.predict(X_val).ravel()

                if self.is_classification:
                    # Normalizar scores a [0,1]
                    probas = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
                    fold_scores.append(roc_auc_score(y_val, probas))
                else:
                    fold_scores.append(
                        np.sqrt(mean_squared_error(y_val, pred))
                    )

            # Para clasificación queremos maximizar AUC
            # Para regresión queremos minimizar RMSE
            if self.is_classification:
                scores.append(np.mean(fold_scores))
            else:
                scores.append(np.mean(fold_scores))

        if self.is_classification:
            best_n = np.argmax(scores) + 1
        else:
            best_n = np.argmin(scores) + 1

        logger.info(f"Mejor número de componentes: {best_n}")
        return best_n

    # ============================================================
    # Entrenamiento
    # ============================================================

    def fit(self, X, y):
        try:
            # Detectar si es clasificación
            self.is_classification = len(np.unique(y)) == 2

            # Selección automática
            if self.n_components is None:
                self.best_components_ = self._select_components_cv(X, y)
            else:
                self.best_components_ = self.n_components

            logger.info(f"Entrenando PLS con {self.best_components_} componentes")

            self.model = PLSRegression(n_components=self.best_components_)
            self.model.fit(X, y)

            logger.info("PLS ajustado correctamente")

        except Exception as e:
            logger.error(f"Error ajustando PLS: {str(e)}")
            raise

    # ============================================================
    # Predicción
    # ============================================================

    def predict(self, X):
        try:
            pred = self.model.predict(X).ravel()

            if self.is_classification:
                return (pred > 0).astype(int)

            return pred

        except Exception as e:
            logger.error(f"Error prediciendo con PLS: {str(e)}")
            raise

    def predict_proba(self, X):
        if not self.is_classification:
            raise ValueError("predict_proba solo está disponible para clasificación")

        scores = self.model.predict(X).ravel()
        probas = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

        return np.vstack([1 - probas, probas]).T
