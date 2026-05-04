df <- read.csv("data/raw/wine_quality.csv")

X <- df %>% select(-quality)
y <- df$quality

max_comp <- 15
rmse_list <- c()

for (n in 1:max_comp) {
  model <- plsr(y ~ ., data=df, ncomp=n, validation="CV")
  rmse_list[n] <- RMSEP(model)$val[1,1,n+1]
}

results <- data.frame(
  n_components = 1:max_comp,
  rmse_cv = rmse_list
)

write.csv(results, "results/pls_cv_wine_quality_R.csv", row.names=FALSE)
