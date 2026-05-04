# scripts/experiments/pls_experiment.py

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.cross_decomposition import PLSRegression

from src.repo_tfm.scripts.core.base_experiment import BaseExperiment
from src.repo_tfm.scripts.core.metrics import compute_metrics
from src.repo_tfm.scripts.core.logger import get_logger
from src.repo_tfm.scripts.core.utils import safe_create_dir

logger = get_logger("PLSExperiment")


class PLSExperiment(BaseExperiment):
    """
    Selección del número óptimo de componentes PLS mediante validación cruzada real.
    """

    def __init__(self, dataset, max_components=15, n_splits=5):
        super().__init__(name="PLS_CV", dataset=dataset)
        self.max_components = max_components
        self.n_splits = n_splits

    def run(self):
        logger.info(f"Iniciando PLS CV REAL para dataset: {self.dataset}")

        # Preparar datos
        self.prepare_data()
        X = self.X_train
        y = self.y_train

        base_dir = f"results/{self.dataset}/csv"
        safe_create_dir(base_dir)

        max_comp = min(self.max_components, X.shape[1])
        rmse_list = []

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        # Validación cruzada real
        for n in range(1, max_comp + 1):
            fold_errors = []

            for train_idx, val_idx in kf.split(X):
                X_tr, X_val = X[train_idx], X[val_idx]
                y_tr, y_val = y[train_idx], y[val_idx]

                model = PLSRegression(n_components=n)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_val).ravel()

                rmse = compute_metrics(y_val, pred)["rmse"]
                fold_errors.append(rmse)

            mean_rmse = np.mean(fold_errors)
            rmse_list.append(mean_rmse)

            logger.info(f"{self.dataset} | Componentes={n} | RMSE_CV={mean_rmse:.4f}")

        # Seleccionar mejor número de componentes
        best = int(np.argmin(rmse_list) + 1)
        logger.info(f"Mejor número de componentes (CV REAL) para {self.dataset}: {best}")

        # Guardar resultados
        df_out = pd.DataFrame({
            "n_components": list(range(1, len(rmse_list) + 1)),
            "rmse_cv": rmse_list
        })

        out_path = f"{base_dir}/pls_cv_{self.dataset}.csv"
        df_out.to_csv(out_path, index=False)

        logger.info(f"Resultados PLS CV guardados en {out_path}")

