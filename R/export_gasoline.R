# export_gasoline.R
install.packages("pls", repos="https://cloud.r-project.org")
library(pls)

data(gasoline)

write.csv(gasoline, "gasoline.csv", row.names = FALSE)
