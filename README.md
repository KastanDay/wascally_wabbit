# NCSA Atmospheric Science Hackathon (Fall '22)

## Data Pre-processing (filtering & dimensionality reduction)

![Data preprocessing](https://user-images.githubusercontent.com/13607221/165446984-1b6aa66f-4e32-422d-bab5-f141a941da2a.png)

## XGBoost: both feature selection & a full solution

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

My appologies that I was unable to report the L1 loss for XGBoost.

## CNN to LSTM model

![Hal Hackathon v5](https://user-images.githubusercontent.com/13607221/165447026-b137c627-67b7-4745-afb4-0ee538fc0e0e.png)


The CNN to LSTM model needs significant hyperparemeter tuning to be more accurate. Currently we compress the embedding layers far too much, our Space Embedding shape is `(time, "sentences", embedding-dim) == (133, 960, 32)`. So we compress each 3D shape from `(66 (our filtered X-variables) * 133 * 196 * 39) =~ 67 million` to `32`, which is a **huge** amount of compression. Also our LSTM is quite shallow, with only 2 LSTM layers in each direction. Therefore, if we had time to optimize (1) the Space Embedding size, (2) the number of LSTM layers and (3) the LSTM lauers' hidden size we would get much better results.

Nevertheless, the entire system is connected together and working! Here is our final result: 

These scores are calcuated by "flattening" the Y data, so it is averaged across all spatiotemporal positions and Y target variables. Therefore, we're pleasantly surprised by the good performance despite the need for more tuning, which suggests the model architecture is strong and worked pretty well on our first try.

![CleanShot 2022-04-26 at 18 22 37](https://user-images.githubusercontent.com/13607221/165410004-9319069c-c5a1-4a91-abd5-505f9007643f.png)

Thank you for hosting such an interesting competition and for the chance to try out such a fun new architecture! Many of us on this team have been wanting to try something hyper-custom like this for a while, and this was the perfect chance to stretch our abilities! It was a lot of fun, okay now back to my day job!

-Kastan, Seonghwan, Vardhan and Daniel.
