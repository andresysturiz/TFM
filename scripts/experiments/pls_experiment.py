# scripts/experiments/pls_experiment.py

import numpy as np
import pandas as pd
import os

from scripts.core.base_experiment import BaseExperiment
from scripts.core.metrics import compute_metrics
from scripts.models.pls_model import PLSModel
from scripts.core.logger import get_logger
from scripts.core.utils import safe_create_dir

logger = get_logger("PLSExperiment")


class PLSExperiment(BaseExperiment):
    """
    Experimento para seleccionar el número óptimo de componentes PLS
    mediante RMSE en el conjunto de entrenamiento.
    """

    def __init__(self, dataset, max_components=15):
        super().__init__(name="PLS_CV", dataset=dataset)
        self.max_components = max_components

    def run(self):
        logger.info(f"Iniciando experimento PLS para dataset: {self.dataset}")

        # 1. Preparar datos (X_train, y_train)
        self.prepare_data()

        X = self.X_train
        y = self.y_train

        safe_create_dir("results")

        rmse_list = []

        # 2. Probar distintos números de componentes
        max_comp = min(self.max_components, X.shape[1])

        for n in range(1, max_comp + 1):
            try:
                model = PLSModel(n_components=n)
                model.fit(X, y)
                pred = model.predict(X)

                rmse = compute_metrics(y, pred)["rmse"]
                rmse_list.append(rmse)

                logger.info(f"{self.dataset} | Componentes={n} | RMSE={rmse:.4f}")

            except Exception as e:
                logger.error(f"Error ejecutando PLS con {n} componentes: {str(e)}")

        # 3. Seleccionar el mejor número de componentes
        best = int(np.argmin(rmse_list) + 1)
        logger.info(f"Mejor número de componentes para {self.dataset}: {best}")

        # 4. Guardar resultados
        df_out = pd.DataFrame({
            "n_components": list(range(1, len(rmse_list) + 1)),
            "rmse_cv": rmse_list
        })

        out_path = f"results/pls_cv_{self.dataset}.csv"
        df_out.to_csv(out_path, index=False)

        logger.info(f"Resultados PLS guardados en {out_path}")
