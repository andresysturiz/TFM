# scripts/experiments/classical_experiment.py

import os
import pandas as pd

from src.repo_tfm.scripts.core.base_experiment import BaseExperiment
from src.repo_tfm.scripts.core.metrics import compute_metrics
from src.repo_tfm.scripts.models.ols_model import OLSModel
from src.repo_tfm.scripts.models.ridge_model import RidgeModel
from src.repo_tfm.scripts.models.lasso_model import LassoModel
from src.repo_tfm.scripts.models.pcr_model import PCRModel
from src.repo_tfm.scripts.core.logger import get_logger
from src.repo_tfm.scripts.core.utils import (
    safe_create_dir,
    plot_predicted_vs_observed,
    plot_residuals,
    plot_lasso_coefficients
)

logger = get_logger("ClassicalExperiment")


class ClassicalExperiment(BaseExperiment):
    """
    Ejecuta modelos clásicos de REGRESIÓN:
    - OLS
    - Ridge
    - Lasso
    - PCR
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

        # Crear carpeta de resultados
        base_dir = f"results/{self.dataset}/csv"
        fig_dir = f"results/{self.dataset}/figures"
        safe_create_dir(fig_dir)
        safe_create_dir(base_dir)

        # 2. Modelos clásicos de regresión
        models = {
            "OLS": OLSModel(),
            "Ridge": RidgeModel(),
            "Lasso": LassoModel(),
            "PCR": PCRModel()
        }

        results_list = []

        # 3. Entrenar y evaluar cada modelo
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                # Predicho vs observado

                plot_predicted_vs_observed(
                    y_test,
                    pred,
                    name,
                    self.dataset,
                    f"{fig_dir}/{name.lower()}_pred_vs_obs.png"
                )

                # Residuos

                plot_residuals(
                    y_test,
                    pred,
                    name,
                    self.dataset,
                    f"{fig_dir}/{name.lower()}_residuals.png"
                )

                # Importancia variables Lasso

                if name == "Lasso":

                    feature_names = [
                        f"X{i}"
                        for i in range(X_train.shape[1])
                    ]

                    plot_lasso_coefficients(
                        model.model.coef_,
                        feature_names,
                        self.dataset,
                        f"{fig_dir}/lasso_importance.png"
                    )

                metrics = compute_metrics(y_test, pred)
                metrics["model"] = name

                results_list.append(metrics)

                logger.info(f"{self.dataset} | {name} | {metrics}")

            except Exception as e:
                logger.error(f"Error ejecutando {name}: {str(e)}")

        # 4. Guardar resultados
        df_out = pd.DataFrame(results_list)
        out_path = f"{base_dir}/classical_results_{self.dataset}.csv"
        df_out.to_csv(out_path, index=False)

        logger.info(f"Resultados clásicos guardados en {out_path}")
