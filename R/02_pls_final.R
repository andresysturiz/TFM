df <- read.csv("data/raw/wine_quality.csv")

set.seed(42)
train_idx <- createDataPartition(df$quality, p=0.8, list=FALSE)

train <- df[train_idx,]
test  <- df[-train_idx,]

best_comp <- read.csv("results/pls_cv_wine_quality_R.csv") %>%
  slice(which.min(rmse_cv)) %>%
  pull(n_components)

model <- plsr(quality ~ ., data=train, ncomp=best_comp)

pred <- predict(model, newdata=test, ncomp=best_comp)

metrics <- compute_metrics(test$quality, pred)

metrics$model <- paste0("PLS_FINAL_", best_comp, "_components")

write.csv(metrics, "results/pls_final_results_wine_quality_R.csv", row.names=FALSE)
