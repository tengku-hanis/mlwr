##=============================================================================##
## Title: Ensemble machine learning
## Author: Tengku Muhd Hanis Bin Tengku Mokhtar, PhD
## Date: Dec 14-16, 2024
## https://jomresearch.netlify.app/
##=============================================================================##

# Ensemble regression model

# Packages ----------------------------------------------------------------

# install.packages("xgboost") # Make sure to install

library(tidymodels)
library(stacks)
library(tidyverse)

# Set ggplot theme
theme_set(theme_bw())


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

# 3) General additive model
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
  option_add(
    control = control_stack_grid(),
    metrics = metric_set(rmse, rsq)
  ) %>% 
  workflow_map(
    resamples = bike_cv, 
    grid = 10, 
    verbose = TRUE
    )


# Stacking ----------------------------------------------------------------

## 1) Add candidate ----
bike_model_stack <- 
  stacks() %>%
  add_candidates(all_workflows) # add candidate members
bike_model_stack

## 2) Blend prediction (apply lasso) ----
bike_model_stack_blended <-
  bike_model_stack %>%
  blend_predictions() 

# mixture = 1 (default): lasso regularization, coeff can be 0
# mixture = 0, ridge regression coeff will not be zero
# penalty: the smaller more member can contribute and vice versa

# Explore blended prediction
autoplot(bike_model_stack_blended)

## 3) Fit ensemble candidates (finalise the stack) ----
bike_model_stack_fit <-
  bike_model_stack_blended %>%
  fit_members()
bike_model_stack_fit

# Explore fitted ensemble model
autoplot(bike_model_stack_fit)
autoplot(bike_model_stack_fit, type = "members")
autoplot(bike_model_stack_fit, type = "weights")


# Performance -------------------------------------------------------------

## Predict test data ----
bike_pred <- 
  bike_test_process %>%
  bind_cols(predict(bike_model_stack_fit, .))

## Evaluate performance ----
#1) Predict vs observed plot
ggplot(bike_pred, aes(x = rentals, y = .pred)) +
  geom_point() + 
  geom_abline(linetype = "dashed", color = "red") +
  labs(y = "Predicted rentals", 
       x = "Observed rentals") 

# 2) Metrics
test_performance <- metric_set(rmse, mae, huber_loss, huber_loss_pseudo, ccc, rsq) 
ensemble_performance <- 
  test_performance(bike_pred, truth = rentals, estimate = .pred)
ensemble_performance

# Previous results:

# Regression1 (decision tree):
# 1 rmse              - 378.   
# 2 mae               - 236.   
# 3 huber_loss        - 236.   
# 4 huber_loss_pseudo - 235.   
# 5 ccc               -   0.841
# 6 rsq               -   0.712

# Regression2 (decision tree with max entropy):
# 1 rmse              - 365.   
# 2 mae               - 229.   
# 3 huber_loss        - 228.   
# 4 huber_loss_pseudo - 228.   
# 5 ccc               -   0.845
# 6 rsq               -   0.725

# Regression3 (boosted tree):
# 1 rmse              - 297.   
# 2 mae               - 209.   
# 3 huber_loss        - 209.   
# 4 huber_loss_pseudo - 208.   
# 5 ccc               -   0.899
# 6 rsq               -   0.819


# Compare performance with candidates -------------------------------------

## Predict using ensemble candidates ----
member_preds <- 
  bike_test_process %>%
  select(rentals) %>%
  bind_cols(predict(bike_model_stack_fit, bike_test_process, members = TRUE))

## Data frame of performance metrics ----

# Data
member_data <- 
  map(member_preds, rmse_vec, truth = member_preds$rentals) %>%
  as_tibble() %>% 
  mutate(metric = "rmse") %>% 
  bind_rows(
    map(member_preds, rsq_vec, truth = member_preds$rentals) %>%
      as_tibble() %>% 
      mutate(metric = "rsquare")
  )

# Change data format to long 
colnames(member_data)

member_data_long <- 
  member_data %>% 
  select(-c(rentals, .pred)) %>% 
  pivot_longer(cols = 1:12, names_to = "members", values_to = "values") 

# Add ensemble model metrics
member_data_long <- 
  member_data_long %>% 
  mutate(color_gp = "gp1") %>% 
  add_row(metric = c("rsquare", "rmse"),    
          members = rep("ENSEMBLE_MODEL", 2), 
          values = c(ensemble_performance$.estimate[6], ensemble_performance$.estimate[1]),
          color_gp = rep("gp2", 2))

# Bar plot for rmse
member_data_long %>% 
  mutate(members = fct_reorder(members, desc(values))) %>% 
  filter(metric == "rmse") %>% 
  ggplot(aes(x = values, y = members, fill = color_gp)) +
  geom_col(alpha = 0.7) +
  theme(legend.position = "none") 

# Bar plot for rquare
member_data_long %>% 
  mutate(members = fct_reorder(members, desc(values))) %>% 
  filter(metric == "rsquare") %>% 
  ggplot(aes(x = values, y = members, fill = color_gp)) +
  geom_col(alpha = 0.7) +
  theme(legend.position = "none") 

