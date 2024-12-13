##=============================================================================##
## Title: Install related packages
## Author: Tengku Muhd Hanis Bin Tengku Mokhtar, PhD
## Date: Dec 14-16, 2024
## https://jomresearch.netlify.app/
##=============================================================================##

# Install packages
install.packages("tidyverse")
install.packages("mlbench")
install.packages("skimr")
install.packages("rpart.plot")
install.packages("vip")
install.packages("janitor")
install.packages("naniar")

# Refresh session before installing the packages below
install.packages("tidymodels")
install.packages("discrim")
install.packages("mda")
install.packages("earth")
install.packages("randomForest")
install.packages("stacks")
install.packages("DALEX")
install.packages("DALEXtra")
install.packages("lime")
install.packages("finetune")

# Refresh session before installing the packages below
install.packages("reticulate")
reticulate::install_python()

# Refresh session before installing the packages below
install.packages("tensorflow")
install.packages("keras3")
keras3::install_keras(version = "default-cpu")

