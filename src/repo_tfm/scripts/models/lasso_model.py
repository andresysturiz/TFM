from sklearn.linear_model import Lasso
from src.repo_tfm.scripts.core.logger import get_logger

logger = get_logger("LassoModel")

class LassoModel:
    def __init__(self, alpha=0.01):
        self.name = "Lasso"
        self.alpha = alpha
        self.model = Lasso(alpha=alpha, max_iter=5000)
        logger.info(f"Inicializando Lasso con alpha={alpha}")

    def fit(self, X, y):
        try:
            self.model.fit(X, y)
            logger.info("Lasso ajustado correctamente")
        except Exception as e:
            logger.error(f"Error ajustando Lasso: {str(e)}")
            raise

    def predict(self, X):
        try:
            return self.model.predict(X).ravel()
        except Exception as e:
            logger.error(f"Error prediciendo con Lasso: {str(e)}")
            raise
