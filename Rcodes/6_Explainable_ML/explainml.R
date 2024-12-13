##=============================================================================##
## Title: Explainable machine learning
## Author: Tengku Muhd Hanis Bin Tengku Mokhtar, PhD
## Date: Dec 14-16, 2024
## https://jomresearch.netlify.app/
##=============================================================================##

# Global interpretation and local interpretation


# Packages ----------------------------------------------------------------

library(mlbench)
library(tidymodels)
library(discrim) # for MARS model
library(DALEXtra)
library(lime)


# Data --------------------------------------------------------------------

# Data
data("PimaIndiansDiabetes2")

## Balanced data ----
set.seed(123)
pima <- 
  PimaIndiansDiabetes2 %>% 
  filter(diabetes == "neg") %>% 
  slice_sample(n = 268) %>% 
  bind_rows(
    PimaIndiansDiabetes2 %>% 
      filter(diabetes == "pos")
  ) %>% 
  mutate(diabetes = relevel(diabetes, ref = "pos"))

## Split data ----
set.seed(123)
split_ind <- initial_split(pima)
pima_train <- training(split_ind)
pima_test <- testing(split_ind)

## Preprocessing ----
pima_rc <- 
  recipe(diabetes~., data = pima_train) %>% 
  step_impute_knn(all_predictors())

pima_train_process <- 
  pima_rc %>% 
  prep() %>% 
  bake(new_data = NULL)

pima_test_process <- 
  pima_rc %>% 
  prep() %>% 
  bake(new_data = pima_test)


# Train the model ---------------------------------------------------------

## Specify model ----

# MARS
mars_spec <- 
  discrim_flexible(prod_degree = 1) %>% 
  set_engine("earth")

## Fit on training data ----
mars_fit <- 
  mars_spec %>% 
  fit(diabetes ~. , data = pima_train_process)


# Explainable ML ----------------------------------------------------------

## Global explanation ----

### 1) Variable importance from vip package ----
vip::vip(mars_fit$fit) #not available

### 2) Permutation-based variable importance (global explanation) -----

# Create explainer model
mars_exp <- explain_tidymodels(mars_fit, 
                               data = pima_test_process %>% select(-diabetes), 
                               y = pima_test_process$diabetes,
                               label = "MARS")

# Permute data
set.seed(123)
mars_vp <- model_parts(mars_exp, loss_function = loss_yardstick(yardstick::roc_auc)) 
mars_vp

plot(mars_vp) 

## Local explanation - LIME ----

# Create explainer model
set.seed(123)
explainer <- lime(x = pima_test_process %>% select(-diabetes), model = mars_fit)

# Data we want to explain
explanation <- explain(x = pima_test_process %>% select(-diabetes) %>% slice(24, 98), 
                       labels = "pos",
                       explainer = explainer, 
                       n_features = 8)

plot_features(explanation) 

# Prediction for data 24 and 98
predict(mars_fit, pima_test_process %>% select(-diabetes) %>% slice(24, 98))
