# scripts/experiments/pls_experiment.py

import numpy as np
from scripts.core.base_experiment import BaseExperiment
from scripts.core.utils import safe_read_csv, safe_create_dir
from scripts.core.scaler import Scaler
from scripts.core.metrics import compute_metrics
from scripts.models.pls_model import PLSModel
from scripts.core.logger import get_logger
import pandas as pd
import os

logger = get_logger("PLSExperiment")

class PLSExperiment(BaseExperiment):

    def __init__(self, name, path, target_col, max_components=15):
        super().__init__(name, path, target_col)
        self.max_components = max_components

    def run(self):
        logger.info(f"Iniciando experimento PLS para {self.name}")

        try:
            df = safe_read_csv(self.path)
        except Exception as e:
            logger.error(f"Error cargando dataset: {str(e)}")
            return

        safe_create_dir("results")

        X = df.drop(columns=[self.target_col]).values
        y = df[self.target_col].values

        scaler = Scaler()
        Xs, ys = scaler.fit_transform(X, y)

        rmse_list = []

        for n in range(1, min(self.max_components, X.shape[1]) + 1):
            try:
                model = PLSModel(n)
                model.fit(Xs, ys)
                pred = model.predict(Xs)
                rmse = compute_metrics(ys, pred)["rmse"]
                rmse_list.append(rmse)
                logger.info(f"{self.name} | Componentes={n} | RMSE={rmse:.4f}")
            except Exception as e:
                logger.error(f"Error ejecutando PLS con {n} componentes: {str(e)}")

        best = int(np.argmin(rmse_list) + 1)
        logger.info(f"Mejor número de componentes para {self.name}: {best}")
        df_out = pd.DataFrame({
            "n_components": list(range(1, len(rmse_list) + 1)),
            "rmse_cv": rmse_list
        })

        out_path = f"results/pls_cv_{os.path.basename(self.path).replace('.csv','')}.csv"
        df_out.to_csv(out_path, index=False)

        logger.info(f"Resultados PLS guardados en {out_path}")
