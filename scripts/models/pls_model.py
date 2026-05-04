from sklearn.cross_decomposition import PLSRegression
from scripts.core.logger import get_logger

logger = get_logger("PLSModel")

class PLSModel:
    def __init__(self, n_components):
        self.n_components = n_components
        self.model = PLSRegression(n_components=n_components)
        logger.info(f"Inicializando PLS con {n_components} componentes")

    def fit(self, X, y):
        try:
            if self.n_components > X.shape[1]:
                raise ValueError(
                    f"n_components={self.n_components} es mayor que el número de variables ({X.shape[1]})"
                )

            self.model.fit(X, y)
            logger.info("PLS ajustado correctamente")

        except Exception as e:
            logger.error(f"Error ajustando PLS: {str(e)}")
            raise

    def predict(self, X):
        try:
            return self.model.predict(X).ravel()
        except Exception as e:
            logger.error(f"Error prediciendo con PLS: {str(e)}")
            raise
