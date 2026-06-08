# GARCH vs EGARCH Study Report
**Date:** 2026-06-07 12:22
**Universe:** PETR4, VALE3, ITUB4, BBDC4, BBAS3, ABEV3, WEGE3, SUZB3, RENT3
**Data:** B3 M15 (15-min) candles, IS/OOS split: 80% / 20%
---
## 1. Convergence
| model_type   |   sum |   count |
|:-------------|------:|--------:|
| EGARCH       |     9 |       9 |
| GARCH        |     9 |       9 |

## 2. Model Comparison (AIC / BIC / Persistence / MSE OOS)
Lower AIC/BIC is better. Lower MSE OOS is better.
| symbol   |   aic_egarch |   aic_garch |   bic_egarch |   bic_garch |   converged_egarch |   converged_garch |   fit_time_seconds_egarch |   fit_time_seconds_garch |   mae_oos_egarch |   mae_oos_garch |   mse_oos_egarch |   mse_oos_garch |   persistence_egarch |   persistence_garch |
|:---------|-------------:|------------:|-------------:|------------:|-------------------:|------------------:|--------------------------:|-------------------------:|-----------------:|----------------:|-----------------:|----------------:|---------------------:|--------------------:|
| ABEV3    |       146780 |      146982 |       146813 |      147007 |                  1 |                 1 |                     8.313 |                    0.124 |          14.0532 |         13.9899 |         19880.6  |        19879.9  |             0.703526 |            0.71413  |
| BBAS3    |       148584 |      148469 |       148617 |      148494 |                  1 |                 1 |                     8.532 |                    0.141 |          16.6237 |         17.0286 |          8676.29 |         8675.11 |             0.601867 |            0.616413 |
| BBDC4    |       153064 |      153213 |       153097 |      153238 |                  1 |                 1 |                     8.582 |                    0.128 |          16.4285 |         16.5198 |          4584.44 |         4584.95 |             0.726889 |            0.735572 |
| ITUB4    |       144798 |      144997 |       144832 |      145022 |                  1 |                 1 |                     8.722 |                    0.135 |          11.9487 |         12.2199 |          1581.99 |         1584.55 |             0.714818 |            0.769949 |
| PETR4    |       158652 |      158862 |       158685 |      158887 |                  1 |                 1 |                     7.363 |                    0.11  |          20.5957 |         26.677  |          6626.88 |         6802.94 |             0.681294 |            0.820768 |
| RENT3    |       170366 |      170447 |       170399 |      170471 |                  1 |                 1 |                     7.378 |                    0.111 |          30.2269 |         30.3809 |         13576.1  |        13576.8  |             0.597816 |            0.640739 |
| SUZB3    |       155460 |      155604 |       155493 |      155629 |                  1 |                 1 |                     7.671 |                    0.105 |          14.3192 |         13.8644 |          2449.52 |         2441.79 |             0.659972 |            0.668354 |
| VALE3    |       153856 |      152448 |       153889 |      152473 |                  1 |                 1 |                     8.043 |                    0.159 |          15.4697 |         12.5828 |          1708.03 |         1672.76 |             0.52392  |            0.999419 |
| WEGE3    |       155170 |      155045 |       155204 |      155070 |                  1 |                 1 |                     7.663 |                    0.108 |          15.7534 |         15.5613 |          3246.56 |         3245.02 |             0.634573 |            0.690316 |

## 3. Per-Symbol Detail
### ABEV3
| Metric | GARCH | EGARCH |
|---|---|---|
| AIC | 146982.06 | 146780.07 |
| BIC | 147006.93 | 146813.22 |
| PERSISTENCE | 0.71 | 0.70 |
| MSE_OOS | 19879.89 | 19880.64 |
| MAE_OOS | 13.99 | 14.05 |

### BBAS3
| Metric | GARCH | EGARCH |
|---|---|---|
| AIC | 148468.95 | 148584.26 |
| BIC | 148493.81 | 148617.41 |
| PERSISTENCE | 0.62 | 0.60 |
| MSE_OOS | 8675.11 | 8676.29 |
| MAE_OOS | 17.03 | 16.62 |

### BBDC4
| Metric | GARCH | EGARCH |
|---|---|---|
| AIC | 153213.02 | 153063.73 |
| BIC | 153237.88 | 153096.88 |
| PERSISTENCE | 0.74 | 0.73 |
| MSE_OOS | 4584.95 | 4584.44 |
| MAE_OOS | 16.52 | 16.43 |

### ITUB4
| Metric | GARCH | EGARCH |
|---|---|---|
| AIC | 144996.82 | 144798.41 |
| BIC | 145021.69 | 144831.56 |
| PERSISTENCE | 0.77 | 0.71 |
| MSE_OOS | 1584.55 | 1581.99 |
| MAE_OOS | 12.22 | 11.95 |

### PETR4
| Metric | GARCH | EGARCH |
|---|---|---|
| AIC | 158861.91 | 158651.93 |
| BIC | 158886.77 | 158685.07 |
| PERSISTENCE | 0.82 | 0.68 |
| MSE_OOS | 6802.94 | 6626.88 |
| MAE_OOS | 26.68 | 20.60 |

### RENT3
| Metric | GARCH | EGARCH |
|---|---|---|
| AIC | 170446.51 | 170365.97 |
| BIC | 170471.37 | 170399.11 |
| PERSISTENCE | 0.64 | 0.60 |
| MSE_OOS | 13576.77 | 13576.06 |
| MAE_OOS | 30.38 | 30.23 |

### SUZB3
| Metric | GARCH | EGARCH |
|---|---|---|
| AIC | 155604.24 | 155459.52 |
| BIC | 155629.10 | 155492.66 |
| PERSISTENCE | 0.67 | 0.66 |
| MSE_OOS | 2441.79 | 2449.52 |
| MAE_OOS | 13.86 | 14.32 |

### VALE3
| Metric | GARCH | EGARCH |
|---|---|---|
| AIC | 152447.77 | 153855.53 |
| BIC | 152472.63 | 153888.68 |
| PERSISTENCE | 1.00 | 0.52 |
| MSE_OOS | 1672.76 | 1708.03 |
| MAE_OOS | 12.58 | 15.47 |

### WEGE3
| Metric | GARCH | EGARCH |
|---|---|---|
| AIC | 155044.73 | 155170.48 |
| BIC | 155069.59 | 155203.63 |
| PERSISTENCE | 0.69 | 0.63 |
| MSE_OOS | 3245.02 | 3246.56 |
| MAE_OOS | 15.56 | 15.75 |

## 4. EGARCH Improvement Over GARCH
**AIC Delta (GARCH - EGARCH):** positive means EGARCH is better.
| symbol   |   AIC_Delta |
|:---------|------------:|
| ABEV3    |    201.992  |
| BBAS3    |   -115.312  |
| BBDC4    |    149.287  |
| ITUB4    |    198.412  |
| PETR4    |    209.979  |
| RENT3    |     80.5401 |
| SUZB3    |    144.726  |
| VALE3    |  -1407.76   |
| WEGE3    |   -125.748  |

> **Best EGARCH improvement (AIC):** PETR4 (ΔAIC = 209.98)
> **Worst EGARCH improvement (AIC):** VALE3 (ΔAIC = -1407.76)

**MSE OOS Delta (GARCH - EGARCH):** positive = EGARCH forecasts better.
| symbol   |   MSE_OOS_Delta |
|:---------|----------------:|
| ABEV3    |       -0.75064  |
| BBAS3    |       -1.18767  |
| BBDC4    |        0.510476 |
| ITUB4    |        2.55822  |
| PETR4    |      176.067    |
| RENT3    |        0.713623 |
| SUZB3    |       -7.73647  |
| VALE3    |      -35.273    |
| WEGE3    |       -1.53365  |

> **Best MSE improvement:** PETR4 (ΔMSE = 176.067307)
> **Worst MSE improvement:** VALE3 (ΔMSE = -35.273043)

**Persistence comparison:**
| symbol   |   persistence_garch |   persistence_egarch |
|:---------|--------------------:|---------------------:|
| ABEV3    |            0.71413  |             0.703526 |
| BBAS3    |            0.616413 |             0.601867 |
| BBDC4    |            0.735572 |             0.726889 |
| ITUB4    |            0.769949 |             0.714818 |
| PETR4    |            0.820768 |             0.681294 |
| RENT3    |            0.640739 |             0.597816 |
| SUZB3    |            0.668354 |             0.659972 |
| VALE3    |            0.999419 |             0.52392  |
| WEGE3    |            0.690316 |             0.634573 |

## 5. Failures / Negative Results
All models converged successfully.

## 6. Computational Cost
| model_type   |   count |     mean |       std |   min |   25% |   50% |   75% |   max |
|:-------------|--------:|---------:|----------:|------:|------:|------:|------:|------:|
| EGARCH       |       9 | 8.02967  | 0.530258  | 7.363 | 7.663 | 8.043 | 8.532 | 8.722 |
| GARCH        |       9 | 0.124556 | 0.0181322 | 0.105 | 0.11  | 0.124 | 0.135 | 0.159 |

## 7. Parameter Estimates
### ABEV3
- **GARCH(1,1)**: omega=3.143797, alpha=0.256988, beta=0.457143, persistence=0.7141
- **EGARCH(1,1,1)**: omega=0.694974, alpha=0.388337, gamma=-0.059398, beta=0.703526, persistence=0.7035

### BBAS3
- **GARCH(1,1)**: omega=4.632653, alpha=0.315038, beta=0.301375, persistence=0.6164
- **EGARCH(1,1,1)**: omega=0.953830, alpha=0.379168, gamma=0.018959, beta=0.601867, persistence=0.6019

### BBDC4
- **GARCH(1,1)**: omega=3.678001, alpha=0.240347, beta=0.495226, persistence=0.7356
- **EGARCH(1,1,1)**: omega=0.703501, alpha=0.343313, gamma=-0.002459, beta=0.726889, persistence=0.7269

### ITUB4
- **GARCH(1,1)**: omega=2.549714, alpha=0.289289, beta=0.480660, persistence=0.7699
- **EGARCH(1,1,1)**: omega=0.653722, alpha=0.448826, gamma=-0.008816, beta=0.714818, persistence=0.7148

### PETR4
- **GARCH(1,1)**: omega=4.665021, alpha=0.422316, beta=0.398452, persistence=0.8208
- **EGARCH(1,1,1)**: omega=0.907602, alpha=0.481512, gamma=-0.079867, beta=0.681294, persistence=0.6813

### RENT3
- **GARCH(1,1)**: omega=8.636625, alpha=0.255934, beta=0.384806, persistence=0.6407
- **EGARCH(1,1,1)**: omega=1.255971, alpha=0.393434, gamma=-0.012848, beta=0.597816, persistence=0.5978

### SUZB3
- **GARCH(1,1)**: omega=4.557537, alpha=0.200900, beta=0.467454, persistence=0.6684
- **EGARCH(1,1,1)**: omega=0.889917, alpha=0.356525, gamma=0.019164, beta=0.659972, persistence=0.6600

### VALE3
- **GARCH(1,1)**: omega=0.006889, alpha=0.003451, beta=0.995968, persistence=0.9994
- **EGARCH(1,1,1)**: omega=1.240102, alpha=0.424800, gamma=-0.093544, beta=0.523920, persistence=0.5239

### WEGE3
- **GARCH(1,1)**: omega=4.150690, alpha=0.194957, beta=0.495359, persistence=0.6903
- **EGARCH(1,1,1)**: omega=0.943031, alpha=0.331647, gamma=0.012715, beta=0.634573, persistence=0.6346

## 8. Conclusions
- **Average AIC**: GARCH(1,1)=154007.3, EGARCH(1,1,1)=154081.1
- **Average Persistence**: GARCH(1,1)=0.7395, EGARCH(1,1,1)=0.6494
- **Average MSE OOS**: GARCH(1,1)=6940.420024, EGARCH(1,1,1)=6925.601339
- **EGARCH wins on MSE OOS**: True
- **EGARCH wins on AIC**: False
- **Stocks where EGARCH better (DeltaAIC > 0, 6):** ABEV3, BBDC4, ITUB4, PETR4, RENT3, SUZB3
- **Stocks where GARCH better (DeltaAIC < 0, 3):** BBAS3, VALE3, WEGE3
- **Highest GARCH persistence:** VALE3 (0.9994)
- **EGARCH gamma (leverage):** mean=-0.0229, min=-0.0935, max=0.0192
  - 6 stocks with negative gamma, 3 with positive.
  - Negative gamma means negative residuals increase volatility more than positive ones (leverage effect).

---
*Report generated automatically by experiment.py*
