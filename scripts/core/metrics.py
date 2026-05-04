import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    median_absolute_error,
    r2_score
)
import time

def compute_metrics(y_true, y_pred):
    start = time.time()

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mdae = median_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # Evitar problemas con valores <= 0
    if np.all(y_true > 0) and np.all(y_pred > 0):
        rmsle = np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))
    else:
        rmsle = None

    r2 = r2_score(y_true, y_pred)

    # R2 ajustado
    n = len(y_true)
    p = 1  # si quieres, puedes pasar el número de features
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    # NRMSE
    nrmse = rmse / (y_true.max() - y_true.min())

    elapsed = time.time() - start

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mdae": mdae,
        "mape": mape,
        "rmsle": rmsle,
        "r2": r2,
        "adj_r2": adj_r2,
        "nrmse": nrmse,
        "train_time": elapsed
    }
