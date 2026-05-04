# scripts/experiments/classical_experiment.py

import os
import pandas as pd

from scripts.core.base_experiment import BaseExperiment
from scripts.core.metrics import compute_metrics
from scripts.models.ols_model import OLSModel
from scripts.models.ridge_model import RidgeModel
from scripts.models.lasso_model import LassoModel
from scripts.models.pcr_model import PCRModel
from scripts.core.logger import get_logger
from scripts.core.utils import safe_create_dir

logger = get_logger("ClassicalExperiment")


class ClassicalExperiment(BaseExperiment):
    """
    Ejecuta modelos clásicos de regresión:
    - OLS
    - Ridge
    - Lasso
    - PCR
    sobre el dataset seleccionado.
    """

    def __init__(self, dataset):
        super().__init__(name="Classical", dataset=dataset)

    def run(self):
        logger.info(f"Iniciando métodos clásicos para dataset: {self.dataset}")

        # 1. Preparar datos
        self.prepare_data()

        X_train = self.X_train
        X_test = self.X_test
        y_train = self.y_train
        y_test = self.y_test

        safe_create_dir("results")

        # 2. Modelos clásicos
        models = {
            "OLS": OLSModel(),
            "Ridge": RidgeModel(),
            "Lasso": LassoModel(),
            "PCR": PCRModel(n_components=5)
        }

        results_list = []

        # 3. Entrenar y evaluar cada modelo
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                pred = model.predict(X_test)

                metrics = compute_metrics(y_test, pred)
                metrics["model"] = name

                results_list.append(metrics)

                logger.info(f"{self.dataset} | {name} | {metrics}")

            except Exception as e:
                logger.error(f"Error ejecutando {name}: {str(e)}")

        # 4. Guardar resultados
        df_out = pd.DataFrame(results_list)

        out_path = f"results/classical_results_{self.dataset}.csv"
        df_out.to_csv(out_path, index=False)

        logger.info(f"Resultados clásicos guardados en {out_path}")
