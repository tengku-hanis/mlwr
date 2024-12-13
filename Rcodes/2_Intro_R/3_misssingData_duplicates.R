## ===========================================================================#
## Missing data and duplicates in R
## Author: Tengku Muhd Hanis Mokhtar
## Date: 21-10-2023
## https://jomresearch.netlify.app/
## ===========================================================================#

# Packages ----------------------------------------------------------------

library(tidyverse)
library(naniar)


# Read external data ------------------------------------------------------

diabetes <- read_csv("Data/diabetes.csv")

glimpse(diabetes)


# NA and NaN --------------------------------------------------------------

dat <- c(1, 2, 0, NA)
dat

dat / 0

# Missing data ------------------------------------------------------------

# Check missing data
anyNA(diabetes)
summary(diabetes)
miss_var_summary(diabetes)

# Plot missing data
gg_miss_var(diabetes)

# Drop missing data
diabetes_cleaned <- na.omit(diabetes)


# Duplicates --------------------------------------------------------------

anyDuplicated(diabetes$id)
ind <- duplicated(diabetes$id)
diabetes[ind, ] #duplicates

# Data w/o duplicates
diabetes_unique <- diabetes[!ind, ] 

# Another way to get data w/o duplicates
ind2 <- unique(diabetes$id)
diabetes_unique2 <- diabetes[ind2, ]




