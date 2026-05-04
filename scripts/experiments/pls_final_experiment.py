# scripts/experiments/pls_final_experiment.py

import os
import pandas as pd
from sklearn.cross_decomposition import PLSRegression

from scripts.core.base_experiment import BaseExperiment
from scripts.core.metrics import compute_metrics
from scripts.core.logger import get_logger
from scripts.core.utils import safe_create_dir

logger = get_logger("PLSFinalExperiment")


class PLSFinalExperiment(BaseExperiment):
    """
    Entrena un modelo PLS final usando el número óptimo de componentes
    y evalúa en el conjunto de test.
    """

    def __init__(self, dataset, best_components):
        super().__init__(name="PLS_FINAL", dataset=dataset)
        self.best_components = best_components

    def run(self):
        logger.info(f"Entrenando PLS FINAL para dataset: {self.dataset} "
                    f"con {self.best_components} componentes")

        # 1. Preparar datos
        self.prepare_data()

        X_train = self.X_train
        X_test = self.X_test
        y_train = self.y_train
        y_test = self.y_test

        safe_create_dir("results")

        # 2. Entrenar modelo final
        try:
            model = PLSRegression(n_components=self.best_components)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            # Asegurar vector 1D
            if y_pred.ndim > 1:
                y_pred = y_pred.ravel()

        except Exception as e:
            logger.error(f"Error entrenando PLS final: {str(e)}")
            return

        # 3. Calcular métricas
        try:
            metrics = compute_metrics(y_test, y_pred)
        except Exception as e:
            logger.error(f"Error calculando métricas PLS final: {str(e)}")
            return

        metrics["model"] = f"PLS_FINAL_{self.best_components}_components"

        # 4. Guardar resultados
        out_path = f"results/pls_final_results_{self.dataset}.csv"
        pd.DataFrame([metrics]).to_csv(out_path, index=False)

        logger.info(f"Resultados PLS FINAL guardados en {out_path}")
