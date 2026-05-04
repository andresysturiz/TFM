# scripts/models/rf_model.py

from sklearn.ensemble import RandomForestRegressor
from scripts.core.logger import get_logger

logger = get_logger("RFModel")

class RFModel:
    def __init__(self, n_estimators=300, random_state=42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state
        )
        logger.info(f"Inicializando RandomForest con {n_estimators} árboles")

    def fit(self, X, y):
        try:
            self.model.fit(X, y)
            logger.info("RandomForest ajustado correctamente")
        except Exception as e:
            logger.error(f"Error ajustando RandomForest: {str(e)}")
            raise

    def predict(self, X):
        try:
            return self.model.predict(X)
        except Exception as e:
            logger.error(f"Error prediciendo con RandomForest: {str(e)}")
            raise
