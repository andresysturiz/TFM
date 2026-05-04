# scripts/models/ols_model.py

from sklearn.linear_model import LinearRegression
from scripts.core.logger import get_logger

logger = get_logger("OLSModel")

class OLSModel:
    def __init__(self):
        self.model = LinearRegression()
        logger.info("Inicializando modelo OLS")

    def fit(self, X, y):
        try:
            self.model.fit(X, y)
            logger.info("OLS ajustado correctamente")
        except Exception as e:
            logger.error(f"Error ajustando OLS: {str(e)}")
            raise

    def predict(self, X):
        try:
            return self.model.predict(X)
        except Exception as e:
            logger.error(f"Error prediciendo con OLS: {str(e)}")
            raise
