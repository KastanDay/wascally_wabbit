# NCSA Atmospheric Science Hackathon (Fall '22)

Here are two representitive hyperparameter searcher for optimal XGBoost parameters on a per-model basis. Unlike other teams, we got reasonable performance out of a purely XGBoost solution. 

Grid-search exploration (less productive, little variance between models): https://wandb.ai/kastan/HAL-Hack-XGBoost-Sweep

Bayesian0optimized hyperband search (more productive, much greater variance in fewer runs): https://wandb.ai/kastan/HAL-Hack-XGBoost-Sweep/sweeps/fbaf1tsh

Search params:

``` yml
method: bayes
metric:
  goal: minimize
  name: validation_0-rmse
name: lambda_XGB_sweep
parameters:
  learning_rate:
    distribution: uniform
    max: 0.5
    min: 0.0001
  max_depth:
    values:
    - 2
    - 4
    - 7
    - 10
    - 15
  min_child_weight:
    values:
    - 2
    - 4
    - 5
    - 6
  n_estimators:
    values:
    - 64
    - 500
    - 1000
  num_boost_round:
    values:
    - 10
    - 50
    - 500
```


## XGBoost Scoring

| Y_Var | RMSE     |
|-------|----------|
| Y_0   | 0.14216  |
| Y_1   | 0.172165 |
| Y_2   | 0.57913  |
| Y_3   | 0.581755 |
| Y_4   | 0.432873 |
| Y_5   | 0.160029 |
| Y_6   | 0.135456 |
| Y_7   | 0.3393   |
| Y_8   | 0.276963 |
| Y_9   | 0.15695  |
