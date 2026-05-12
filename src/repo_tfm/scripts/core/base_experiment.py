# scripts/core/base_experiment.py

from abc import ABC, abstractmethod
from src.repo_tfm.scripts.core.data_loader import DataLoader
from src.repo_tfm.scripts.core.preprocessing import preprocess_regression
from src.repo_tfm.scripts.core.logger import get_logger

logger = get_logger("experiment")


class BaseExperiment(ABC):
    """
    Clase Gestiona:
    - carga del dataset
    - preprocesado (solo regresión)
    - división train/test
    - logging
    """

    def __init__(self, name, dataset):
        """
        name: nombre del experimento 
        dataset: nombre del dataset 
        """
        self.name = name
        self.dataset = dataset
        self.loader = DataLoader()

        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None


    # PREPARACIÓN DE DATOS 

    def prepare_data(self):
        logger.info(f"Preparando datos para dataset: {self.dataset}")

        # Carga genérica (df, X, y)
        df, X, y = self.loader.load(self.dataset)
        self.df = df

        # Preprocesado estándar de regresión
        self.X_train, self.X_test, self.y_train, self.y_test = preprocess_regression(X, y)

        logger.info("Dataset de regresión preprocesado correctamente.")


    # MÉTODO PRINCIPAL

    @abstractmethod
    def run(self):
        """
        Cada experimento debe implementar:
        - entrenamiento del modelo
        - predicción
        - cálculo de métricas
        - guardado de resultados
        """
        pass
