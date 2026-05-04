pls_cv <- read.csv("results/pls_cv_wine_quality_R.csv")
pls_final <- read.csv("results/pls_final_results_wine_quality_R.csv")
classical <- read.csv("results/classical_results_wine_quality_R.csv")

comparison <- bind_rows(classical, pls_final)

write.csv(comparison, "results/comparison_wine_quality_R.csv", row.names=FALSE)
