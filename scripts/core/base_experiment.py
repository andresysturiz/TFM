# scripts/core/base_experiment.py

from abc import ABC, abstractmethod

class BaseExperiment(ABC):
    """
    Clase base para experimentos.
    Define la estructura mínima que deben seguir todos los experimentos.
    """

    def __init__(self, name, path, target_col):
        self.name = name
        self.path = path
        self.target_col = target_col

    @abstractmethod
    def run(self):
        """Método principal que debe implementar cada experimento."""
        pass
