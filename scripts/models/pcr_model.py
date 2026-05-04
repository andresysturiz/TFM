# scripts/models/pcr_model.py

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from scripts.core.logger import get_logger

logger = get_logger("PCRModel")

class PCRModel:
    def __init__(self, n_components):
        self.pca = PCA(n_components=n_components)
        self.reg = LinearRegression()
        logger.info(f"Inicializando PCR con {n_components} componentes")

    def fit(self, X, y):
        try:
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
