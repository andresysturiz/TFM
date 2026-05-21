# scripts/experiments/pls_final_experiment.py

import pandas as pd
from sklearn.cross_decomposition import PLSRegression

from src.repo_tfm.scripts.core.base_experiment import BaseExperiment
from src.repo_tfm.scripts.core.metrics import compute_metrics
from src.repo_tfm.scripts.core.logger import get_logger
from src.repo_tfm.scripts.core.utils import (
    safe_create_dir,
    plot_predicted_vs_observed,
    plot_residuals,
    plot_pls_loadings
)

logger = get_logger("PLSFinalExperiment")


class PLSFinalExperiment(BaseExperiment):
    """
    Entrena un modelo PLS FINAL usando el número óptimo de componentes
    obtenido en PLSExperiment (validación cruzada real).
    """

    def __init__(self, dataset):
        super().__init__(name="PLS_FINAL", dataset=dataset)

    def run(self):
        logger.info(f"Iniciando PLS FINAL para dataset: {self.dataset}")

        # Preparar datos
        self.prepare_data()
        X_train, X_test = self.X_train, self.X_test
        y_train, y_test = self.y_train, self.y_test

        # Leer mejor número de componentes
        cv_path = f"results/{self.dataset}/csv/pls_cv_{self.dataset}.csv"
        df_cv = pd.read_csv(cv_path)
        best_components = int(df_cv.loc[df_cv["rmse_cv"].idxmin(), "n_components"])

        logger.info(f"Mejor número de componentes detectado: {best_components}")

        # Entrenar modelo final
        model = PLSRegression(n_components=best_components)
        model.fit(X_train, y_train)
        pred = model.predict(X_test).ravel()

        # Guardar resultados
        base_dir = f"results/{self.dataset}/csv"
        fig_dir = f"results/{self.dataset}/figures"
        safe_create_dir(fig_dir)
        safe_create_dir(base_dir)

        # Predicho vs observado

        plot_predicted_vs_observed(
            y_test,
            pred,
            "PLS_FINAL",
            self.dataset,
            f"{fig_dir}/pls_pred_vs_obs.png"
        )

        # Residuos

        plot_residuals(
            y_test,
            pred,
            "PLS_FINAL",
            self.dataset,
            f"{fig_dir}/pls_residuals.png"
        )

        # Loadings PLS

        plot_pls_loadings(
            model.x_loadings_[:, 0],
            self.dataset,
            f"{fig_dir}/pls_loadings.png"
        )

        metrics = compute_metrics(y_test, pred)
        metrics["model"] = f"PLS_FINAL_{best_components}_components"
        metrics["best_components"] = best_components

        out_path = f"{base_dir}/pls_final_results_{self.dataset}.csv"
        pd.DataFrame([metrics]).to_csv(out_path, index=False)

        logger.info(f"Resultados PLS FINAL guardados en {out_path}")

