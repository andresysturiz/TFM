from sklearn.linear_model import Ridge
from src.repo_tfm.scripts.core.logger import get_logger

logger = get_logger("RidgeModel")

class RidgeModel:
    def __init__(self, alpha=1.0):
        self.name = "Ridge"
        self.alpha = alpha
        self.model = Ridge(alpha=alpha)
        logger.info(f"Inicializando Ridge con alpha={alpha}")

    def fit(self, X, y):
        try:
            self.model.fit(X, y)
            logger.info("Ridge ajustado correctamente")
        except Exception as e:
            logger.error(f"Error ajustando Ridge: {str(e)}")
            raise

    def predict(self, X):
        try:
            return self.model.predict(X).ravel()
        except Exception as e:
            logger.error(f"Error prediciendo con Ridge: {str(e)}")
            raise
