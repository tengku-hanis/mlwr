##=============================================================================##
## Title: Regression model - intermediate
## Author: Tengku Muhd Hanis Bin Tengku Mokhtar, PhD
## Date: Dec 14-16, 2024
## https://jomresearch.netlify.app/
##=============================================================================##

# Compare grid_regular(), grid_random(), grid_latin_hypercube(), and grid_max_entropy()
# Apply advanced search

# Packages ----------------------------------------------------------------

library(tidyverse)
library(tidymodels)


# Compare grid approach ---------------------------------------------------

## Regular grid search ----
param_reg_base <- expand.grid(cost_complexity = c(-13, -12, -11), tree_depth = c(10, 20), min_n = c(2, 3))
set.seed(123)
param_reg <- grid_regular(tree_depth(), min_n(), levels = 20)
#levels determine how many values to pick from the range of parameters

## Random grid search ----
set.seed(123)
param_rand <- grid_random(tree_depth(), min_n(), size = 20)
#size determine how many values to pick from the range of parameters

## Grid latin hypercube search ----
set.seed(123)
param_latin <- grid_latin_hypercube(tree_depth(), min_n(), size = 20)
#size determine how many values to pick from the range of parameters

## Grid maximum entropy search ----
set.seed(123)
param_max <- grid_max_entropy(tree_depth(), min_n(), size = 20)
#size determine how many values to pick from the range of parameters

## Compare
param_reg %>% 
  mutate(grid = "regular") %>% 
  bind_rows(
    param_rand %>% 
      mutate(grid = "random")
  ) %>% 
  bind_rows(
    param_latin %>% 
      mutate(grid = "latin")
  ) %>% 
  bind_rows(
    param_max %>% 
      mutate(grid = "max_entropy")
  ) %>% 
  ggplot(aes(tree_depth, min_n)) +
  geom_point() +
  facet_grid(cols = vars(grid))


# Data --------------------------------------------------------------------

## The data ----
data_bike <- read_csv("Data/daily-bike-share.csv")

## Explore data ----

# Check for missing data
anyNA(data_bike)

# Descriptive statistics
str(data_bike)

# Edit variable type
data_bike2 <- 
  data_bike %>% 
  mutate(dteday = mdy(dteday), 
         across(c(season, yr, mnth, holiday, weekday, workingday, weathersit), as.factor)) %>% 
  select(-instant)

skimr::skim(data_bike2)

# Correlation
data_bike2 %>% 
  select_if(is.numeric) %>% 
  DataExplorer::plot_correlation() 

## Split data ----
set.seed(123)
split_ind <- initial_split(data_bike2)
bike_train <- training(split_ind)
bike_test <- testing(split_ind)

## Preprocessing ----
bike_rc <- 
  recipe(rentals ~., data = bike_train) 

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
bike_cv <- vfold_cv(bike_train_process, v = 5) # normally we do 10 for large dataset


# Tuning ------------------------------------------------------------------

## Specify model ----
dt_mod <- 
  decision_tree(
    cost_complexity = tune(), #default range of parameters will be used
    tree_depth = tune(),
    min_n = tune()
  ) %>% 
  set_engine("rpart") %>% 
  set_mode("regression")

## Max entropy search ----
set.seed(123)
param_max <- grid_max_entropy(cost_complexity(), tree_depth(), min_n(), size = 100)

## Specify workflow ----
dt_wf <- workflow() %>% 
  add_model(dt_mod) %>% 
  add_recipe(bike_rc)

## tune_grid ----
set.seed(123)

ctrl <- control_resamples(save_pred = T, verbose = T)
dt_tune <- 
  dt_wf %>% 
  tune_grid(resamples = bike_cv,
            metrics = metric_set(rmse, rsq),
            grid = param_max,
            control = ctrl)

## Explore tuning result ----
autoplot(dt_tune) + theme_light()
dt_tune %>% collect_metrics() 

dt_tune %>% show_best(metric = "rmse")
dt_tune %>% show_best(metric = "rsq")

best_tune <- 
  dt_tune %>% 
  select_best(metric = "rmse")

## Finalize workflow ----
dt_wf_final <- 
  dt_wf %>% 
  finalize_workflow(best_tune)


# Re-fit on training data -------------------------------------------------

dt_train <- 
  dt_wf_final %>% 
  fit(data = bike_train_process)

# Visualise (not practical)
dt_fit <- 
  dt_train %>% 
  extract_fit_parsnip()
rpart.plot::rpart.plot(dt_fit$fit, roundint=FALSE)

# Another way to visualise
vip::vip(dt_fit)


# Assess on testing data --------------------------------------------------

## Fit on test data ----
bike_pred <- 
  bike_test_process %>% 
  bind_cols(predict(dt_train, new_data = bike_test_process)) 

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




