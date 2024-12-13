##=============================================================================##
## Title: Regression model - intermediate2
## Author: Tengku Muhammad Hanis Bin Tengku Mokhtar, PhD
## Date: Dec 14-16, 2024
##=============================================================================##

# Compare several models

# Packages ----------------------------------------------------------------

# install.packages("xgboost") # Make sure to install

library(tidyverse)
library(tidymodels)


# Data --------------------------------------------------------------------

## The data ----
data_bike <- read_csv("Data/daily-bike-share.csv")

## Explore data ----

# Edit variable type
data_bike2 <- 
  data_bike %>% 
  mutate(dteday = mdy(dteday), 
         across(c(season, yr, mnth, holiday, weekday, workingday, weathersit), as.factor)) %>% 
  select(-instant)

## Split data ----
set.seed(123)
split_ind <- initial_split(data_bike2)
bike_train <- training(split_ind)
bike_test <- testing(split_ind)

## Preprocessing ----
bike_rc <- 
  recipe(rentals ~., data = bike_train) %>% 
  step_dummy(all_factor()) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_nzv()

bike_train_process <- 
  bike_rc %>% 
  prep() %>% 
  bake(new_data = NULL)

bike_test_process <- 
  bike_rc %>% 
  prep() %>% 
  bake(new_data = bike_test)

## 10-fold CV ----
set.seed(123)
bike_cv <- vfold_cv(bike_train_process, v = 5) 


# Tuning ------------------------------------------------------------------

## Specify model ----

# 1) Decision tree
dt_mod <- 
  decision_tree(
    cost_complexity = tune(), #default range of parameters will be used
    tree_depth = tune(),
    min_n = tune()
  ) %>% 
  set_engine("rpart") %>% 
  set_mode("regression")

# 2) Linear regression
lr_mod <- 
  linear_reg(penalty = tune(), mixture = tune()) %>% 
  set_engine("glmnet")

# 3) Boosted tree
btree_mod <- 
  boost_tree(
    mtry = tune(), #number of predictors to randomly sample at each split
    trees = tune(), #total number of trees in the model
    min_n = tune(), #minimum number of data points required in a terminal node (leaf)
    tree_depth = tune()) %>% #maximum depth of each tree
  set_engine("xgboost") %>%
  set_mode("regression")

## Specify workflow ----
all_workflows <- 
  workflow_set(preproc = list("formula" = rentals ~.),
               models = list(dt_mod, lr_mod, btree_mod)) 
all_workflows

## tune_grid ----
set.seed(123)
all_workflows <- 
  all_workflows %>% 
  workflow_map(resamples = bike_cv, grid = 10, verbose = TRUE)

## Explore tuning result ----
all_workflows
rank_results(all_workflows, rank_metric = "rmse")

autoplot(all_workflows, metric = "rmse") + theme_bw()
autoplot(all_workflows, metric = "rsq") + theme_bw()

## Extract best model result ----
best_model <- 
  all_workflows %>% 
  extract_workflow_set_result("formula_boost_tree")
best_model

## Explore best models ----
autoplot(best_model) + theme_light()
best_model %>% collect_metrics()

best_model %>% show_best(metric = "rmse")
best_model %>% show_best(metric = "rsq")

best_tune <- 
  best_model %>% 
  select_best(metric = "rmse")

## Extract workflow ----
best_model_workflow <- 
  all_workflows %>% 
  extract_workflow("formula_boost_tree")

## Finalize workflow ----
wf_final <- 
  best_model_workflow %>% 
  finalize_workflow(best_tune)

# Re-fit on training data -------------------------------------------------

mod_train <- 
  wf_final %>% 
  fit(data = bike_train_process)

# Visualise (not good as the model becomes more complex)
vip::vip(mod_fit)


# Assess on testing data --------------------------------------------------

## Fit on test data ----
bike_pred <- 
  bike_test_process %>% 
  bind_cols(predict(mod_train, new_data = bike_test_process)) 

## Performance metrics ----

test_performance <- metric_set(rmse, mae, huber_loss, huber_loss_pseudo, ccc, rsq) 
test_performance(bike_pred, truth = rentals, estimate = .pred)

ggplot(bike_pred, aes(rentals, .pred)) +
  geom_point() +
  geom_abline(linetype = "dashed", color = "red") +
  labs(y = "Predicted rentals", 
       x = "Observed rentals") +
  theme_bw()



# Previous results:

# Regression1:
# 1 rmse              - 378.   
# 2 mae               - 236.   
# 3 huber_loss        - 236.   
# 4 huber_loss_pseudo - 235.   
# 5 ccc               -   0.841
# 6 rsq               -   0.712

# Regression2
# 1 rmse              - 365.   
# 2 mae               - 229.   
# 3 huber_loss        - 228.   
# 4 huber_loss_pseudo - 228.   
# 5 ccc               -   0.845
# 6 rsq               -   0.725

# Regression3:
# 1 rmse              - 297.   
# 2 mae               - 209.   
# 3 huber_loss        - 209.   
# 4 huber_loss_pseudo - 208.   
# 5 ccc               -   0.899
# 6 rsq               -   0.819