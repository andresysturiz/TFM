compute_metrics <- function(y_true, y_pred) {
  mse  <- mean((y_true - y_pred)^2)
  rmse <- sqrt(mse)
  mae  <- mean(abs(y_true - y_pred))
  mdae <- median(abs(y_true - y_pred))
  mape <- mean(abs((y_true - y_pred) / y_true)) * 100
  r2   <- 1 - sum((y_true - y_pred)^2) / sum((y_true - mean(y_true))^2)

  data.frame(
    mse = mse,
    rmse = rmse,
    mae = mae,
    mdae = mdae,
    mape = mape,
    r2 = r2
  )
}
