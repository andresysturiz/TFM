df <- read.csv("data/raw/wine_quality.csv")

set.seed(42)
train_idx <- createDataPartition(df$quality, p=0.8, list=FALSE)

train <- df[train_idx,]
test  <- df[-train_idx,]

X_train <- train %>% select(-quality)
y_train <- train$quality

X_test <- test %>% select(-quality)
y_test <- test$quality

results <- list()

# OLS
ols <- lm(quality ~ ., data=train)
pred <- predict(ols, test)
results[["OLS"]] <- compute_metrics(y_test, pred)

# Ridge
ridge <- cv.glmnet(as.matrix(X_train), y_train, alpha=0)
pred <- predict(ridge, as.matrix(X_test), s="lambda.min")
results[["Ridge"]] <- compute_metrics(y_test, pred)

# Lasso
lasso <- cv.glmnet(as.matrix(X_train), y_train, alpha=1)
pred <- predict(lasso, as.matrix(X_test), s="lambda.min")
results[["Lasso"]] <- compute_metrics(y_test, pred)

# PCR
pcr_model <- pcr(quality ~ ., data=train, validation="CV")
best_comp <- which.min(RMSEP(pcr_model)$val[1,1,-1])
pred <- predict(pcr_model, test, ncomp=best_comp)
results[["PCR"]] <- compute_metrics(y_test, pred)

df_out <- bind_rows(results, .id="model")
write.csv(df_out, "results/classical_results_wine_quality_R.csv", row.names=FALSE)
