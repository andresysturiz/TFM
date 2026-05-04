from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np

from src.repo_tfm.scripts.core.logger import get_logger

logger = get_logger("PCRModel")


class PCRModel:
    def __init__(self, n_components=None, max_components=20):
        """
        Si n_components es None → seleccionar automáticamente usando CV.
        max_components limita la búsqueda (por defecto 20).
        """
        self.n_components = n_components
        self.max_components = max_components
        self.pca = None
        self.reg = LinearRegression()
        self.best_components_ = None

    def _select_components_cv(self, X, y):
        """
        Selección automática del número óptimo de componentes mediante CV.
        """
        logger.info("Seleccionando número óptimo de componentes para PCR...")

        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        max_comp = min(self.max_components, X.shape[1])
        rmse_list = []

        for n in range(1, max_comp + 1):
            pca = PCA(n_components=n)
            Xp = pca.fit_transform(X)

            rmse_folds = []
            for train_idx, val_idx in kf.split(Xp):
                X_train, X_val = Xp[train_idx], Xp[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                reg = LinearRegression()
                reg.fit(X_train, y_train)
                pred = reg.predict(X_val)
                rmse_folds.append(np.sqrt(mean_squared_error(y_val, pred)))

            rmse_list.append(np.mean(rmse_folds))

        best_n = np.argmin(rmse_list) + 1
        logger.info(f"Mejor número de componentes para PCR: {best_n}")

        return best_n

    def fit(self, X, y):
        try:
            # Selección automática si no se especifica n_components
            if self.n_components is None:
                self.best_components_ = self._select_components_cv(X, y)
            else:
                self.best_components_ = self.n_components

            logger.info(f"Entrenando PCR con {self.best_components_} componentes")

            self.pca = PCA(n_components=self.best_components_)
            Xp = self.pca.fit_transform(X)
            self.reg.fit(Xp, y)

            logger.info("PCR ajustado correctamente")

        except Exception as e:
            logger.error(f"Error ajustando PCR: {str(e)}")
            raise

    def predict(self, X):
        try:
            Xp = self.pca.transform(X)
            return self.reg.predict(Xp)
        except Exception as e:
            logger.error(f"Error prediciendo con PCR: {str(e)}")
            raise
