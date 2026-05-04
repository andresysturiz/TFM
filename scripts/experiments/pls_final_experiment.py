# scripts/experiments/pls_final_experiment.py

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression

from scripts.core.logger import get_logger
from scripts.core.utils import safe_read_csv, safe_create_dir
from scripts.core.metrics import compute_metrics

logger = get_logger("PLSFinalExperiment")


class PLSFinalExperiment:

    def __init__(self, name, path, target_col, best_components, test_size=0.2, random_state=42):
        self.name = name
        self.path = path
        self.target_col = target_col
        self.best_components = best_components
        self.test_size = test_size
        self.random_state = random_state

    def run(self):
        logger.info(f"Entrenando PLS FINAL para {self.name} con {self.best_components} componentes")

        try:
            df = safe_read_csv(self.path)
        except Exception as e:
            logger.error(f"Error cargando dataset: {str(e)}")
            return

        safe_create_dir("results")

        X = df.drop(columns=[self.target_col]).values
        y = df[self.target_col].values

        # División train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state
        )

        # Entrenar modelo final
        model = PLSRegression(n_components=self.best_components)

        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Asegurar vector 1D
            if y_pred.ndim > 1:
                y_pred = y_pred.ravel()

        except Exception as e:
            logger.error(f"Error entrenando PLS final: {str(e)}")
            return

        # Calcular métricas completas
        try:
            metrics = compute_metrics(y_test, y_pred)
        except Exception as e:
            logger.error(f"Error calculando métricas PLS final: {str(e)}")
            return

        metrics["model"] = f"PLS_FINAL_{self.best_components}_components"

        # Guardar CSV
        out_path = f"results/pls_final_results_{os.path.basename(self.path).replace('.csv','')}.csv"
        pd.DataFrame([metrics]).to_csv(out_path, index=False)

        logger.info(f"Resultados PLS FINAL guardados en {out_path}")
