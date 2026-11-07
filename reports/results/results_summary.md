# Results summary (hour grain, FAST=False)

Best approved model: Random Forest (RMSE 294.185); best baseline: four-week seasonal mean (RMSE 335.014); improvement 12.2%.
Ranking: Random Forest, then SVR (RBF), then Ridge regression, then SVR (linear), then Bayesian ridge, then Lasso regression (lowest to highest RMSE).
Bootstrap: {"model": "rf", "baseline": "seasonal_mean_4w", "B": 2000, "n_dates": 61, "mean_rmse_reduction": 42.07049978313817, "ci95": [20.616925727770102, 68.62751963848997], "share_model_better": 1.0, "significant": true}

Current index coverage: 3,119 of 3,119 currently indexed chunk files processed; 2 older accumulator files retained. Rows read: 3,272,254,905 rows across 3,121 accumulator files.
## Table 2
                      label                   key    RMSE     MAE    R2   WAPE
Seasonal naive (t − 1 week)        seasonal_naive 418.725 134.606 0.688 22.649
        Same hour yesterday same_period_yesterday 493.112 186.791 0.568 31.429
    Four-week seasonal mean      seasonal_mean_4w 335.014 119.248 0.800 20.065
         Historical average        historical_avg 336.322 115.201 0.799 19.384
                      Ridge                 ridge 327.269 125.358 0.810 21.093
                      Lasso                 lasso 329.964 124.562 0.806 20.959
             Bayesian ridge                 bayes 327.305 125.372 0.810 21.095
              Random Forest                    rf 294.185  95.953 0.846 16.145
               SVR (linear)            svr_linear 327.297 125.371 0.810 21.095
                  SVR (RBF)               svr_rbf 314.937 102.488 0.824 17.245

## Worst stations (high volume)
                                          n  mean_actual  RMSE_model  WAPE_model  RMSE_baseline  WAPE_baseline
station                                                                                                       
Oud Metha Metro Station                1464      674.130     576.679       0.298        661.542          0.381
Emirates Towers Metro Station          1464      533.136    1020.270       0.245       1137.156          0.353
Burj Khalifa/Dubai Mall Metro Station  1464     1242.178     612.460       0.210        644.966          0.243
Business Bay Metro Station             1464      929.428     469.944       0.172        555.761          0.232
Deira City Centre Metro Station        1464     1035.290     425.962       0.170        462.615          0.205

Bayesian 90% interval coverage: 0.9065098598241863
Top RF features: ['lag_168', 'lag_mean_4w', 'lag_336', 'lag_504']