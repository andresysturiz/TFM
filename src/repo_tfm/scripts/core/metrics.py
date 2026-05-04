import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    median_absolute_error,
    r2_score
)
import time

def compute_metrics(y_true, y_pred, n_features=None):
    start = time.time()

    # MSE y RMSE
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    # MAE y MdAE
    mae = mean_absolute_error(y_true, y_pred)
    mdae = median_absolute_error(y_true, y_pred)

    # MAPE robusto (evita división por cero)
    with np.errstate(divide="ignore", invalid="ignore"):
        mape_vals = np.abs((y_true - y_pred) / y_true)
        mape_vals = mape_vals[~np.isinf(mape_vals)]
        mape_vals = mape_vals[~np.isnan(mape_vals)]
        mape = np.mean(mape_vals) * 100 if len(mape_vals) > 0 else None

    # RMSLE solo si todo es positivo
    if np.all(y_true > 0) and np.all(y_pred > 0):
        rmsle = np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))
    else:
        rmsle = None

    # R2 y R2 ajustado
    r2 = r2_score(y_true, y_pred)

    if n_features is None:
        n_features = 1  # fallback seguro

    n = len(y_true)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)

    # NRMSE robusto
    y_range = y_true.max() - y_true.min()
    nrmse = rmse / y_range if y_range != 0 else None

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
