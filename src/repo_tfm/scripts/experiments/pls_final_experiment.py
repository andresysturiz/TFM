# scripts/experiments/pls_final_experiment.py

import pandas as pd
from sklearn.cross_decomposition import PLSRegression

from src.repo_tfm.scripts.core.base_experiment import BaseExperiment
from src.repo_tfm.scripts.core.metrics import compute_metrics
from src.repo_tfm.scripts.core.logger import get_logger
from src.repo_tfm.scripts.core.utils import safe_create_dir

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

        metrics = compute_metrics(y_test, pred)
        metrics["model"] = f"PLS_FINAL_{best_components}_components"
        metrics["best_components"] = best_components

        # Guardar resultados
        base_dir = f"results/{self.dataset}/csv"
        safe_create_dir(base_dir)

        out_path = f"{base_dir}/pls_final_results_{self.dataset}.csv"
        pd.DataFrame([metrics]).to_csv(out_path, index=False)

        logger.info(f"Resultados PLS FINAL guardados en {out_path}")

