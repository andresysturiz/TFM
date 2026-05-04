# scripts/core/base_experiment.py

from abc import ABC, abstractmethod
from scripts.core.data_loader import DataLoader
from scripts.core.preprocessing import preprocess_regression, preprocess_golub
from scripts.core.logger import get_logger

logger = get_logger("experiment")


class BaseExperiment(ABC):
    """
    Clase base para experimentos.
    Gestiona:
    - carga del dataset
    - preprocesado
    - división train/test
    - logging
    """

    def __init__(self, name, dataset):
        """
        name: nombre del experimento (PLS, Classical, etc.)
        dataset: nombre del dataset ("wine", "airfoil", "gasoline", "golub")
        """
        self.name = name
        self.dataset = dataset
        self.loader = DataLoader()

        # Estos se rellenan en prepare_data()
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    # -------------------------
    # PREPARACIÓN DE DATOS
    # -------------------------
    def prepare_data(self):
        logger.info(f"Preparando datos para dataset: {self.dataset}")

        data = self.loader.load(self.dataset)

        # REGRESIÓN
        if self.dataset in ["wine", "airfoil", "gasoline"]:
            df, X, y = data
            self.df = df
            self.X_train, self.X_test, self.y_train, self.y_test = preprocess_regression(X, y)
            logger.info("Dataset de regresión preprocesado correctamente.")

        # CLASIFICACIÓN (GOLUB)
        elif self.dataset == "golub":
            expr, labels = data
            self.X_train, self.X_test, self.y_train, self.y_test = preprocess_golub(expr, labels)
            logger.info("Dataset Golub preprocesado correctamente.")

        else:
            raise ValueError(f"Dataset '{self.dataset}' no soportado en BaseExperiment.")

    # -------------------------
    # MÉTODO PRINCIPAL
    # -------------------------
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
