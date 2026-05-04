# scripts/experiments/classical_experiment.py

from scripts.core.base_experiment import BaseExperiment
from scripts.core.utils import safe_read_csv, safe_create_dir
from scripts.core.scaler import Scaler
from scripts.core.metrics import compute_metrics
from scripts.models.ols_model import OLSModel
from scripts.models.ridge_model import RidgeModel
from scripts.models.lasso_model import LassoModel
from scripts.models.pcr_model import PCRModel
from scripts.core.logger import get_logger
import pandas as pd
import os

logger = get_logger("ClassicalExperiment")

class ClassicalExperiment(BaseExperiment):

    def run(self):
        logger.info(f"Iniciando métodos clásicos para {self.name}")

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

        models = {
            "OLS": OLSModel(),
            "Ridge": RidgeModel(),
            "Lasso": LassoModel(),
            "PCR": PCRModel(n_components=5)
        }
        results_list = []

        for name, model in models.items():
            try:
                model.fit(Xs, ys)
                pred = model.predict(Xs)
                metrics = compute_metrics(ys, pred)
                metrics["model"] = name
                results_list.append(metrics)
                logger.info(f"{self.name} | {name} | {metrics}")
            except Exception as e:
                logger.error(f"Error ejecutando {name}: {str(e)}")

        df_out = pd.DataFrame(results_list)

        out_path = f"results/classical_results_{os.path.basename(self.path).replace('.csv','')}.csv"
        df_out.to_csv(out_path, index=False)

        logger.info(f"Resultados clásicos guardados en {out_path}")
