# Market Regime Edge Discovery — Report
**Generated:** 2026-06-09T22:21:38.329008
**Universe:** 50 B3 stocks, M15 timeframe
**Period:** Train=2021-01-01 to 2023-12-31, Val=2024-01-01 to 2024-06-30, Test=2024-07-01 to 2026-06-09
**Costs:** 10.0 bps per trade + 5.0 bps slippage
**Seed:** 42
---
## 1. Global Summary
- **Strategies tested:** breakout, gap_fade, mean_reversion, momentum, opening_range_breakout, sr_bounce, trend_following, vwap_reversion
- **Context variables analyzed:** 15
- **Total strategy x context combinations:** 4014
- **Best discrimination:** sr_bounce x vol_regime (gap=2.807)

## 2. Context Variable Ranking
Ranking by Sharpe Gap (difference between best and worst regime Sharpe).

| Rank | Strategy | Context | Overall Sharpe | Sharpe Gap | Best Regime | Worst Regime | Best Sharpe | Worst Sharpe |
|------|----------|---------|---------------|------------|-------------|--------------|-------------|--------------|
| 1 | sr_bounce | vol_regime | 0.901 | 2.807 | 2.0 | 0.0 | 3.000 | 0.193 |
| 2 | momentum | support_distance | -0.226 | 2.766 | Q4 | Q1 | 0.253 | -2.513 |
| 3 | trend_following | volume_percentile | -0.923 | 2.761 | Q5 | Q2 | -0.223 | -2.984 |
| 4 | trend_following | rv_5 | -0.550 | 2.745 | Q1 | Q4 | -0.255 | -3.000 |
| 5 | sr_bounce | rv_20 | 0.506 | 2.721 | Q4 | Q1 | 3.000 | 0.279 |
| 6 | sr_bounce | atr | 0.901 | 2.710 | Q2 | Q1 | 2.922 | 0.212 |
| 7 | trend_following | rv_20 | -0.550 | 2.704 | Q1 | Q4 | -0.296 | -3.000 |
| 8 | sr_bounce | volume_percentile | 1.111 | 2.618 | Q1 | Q5 | 2.881 | 0.263 |
| 9 | sr_bounce | normalized_range | 0.901 | 2.590 | Q2 | Q1 | 2.777 | 0.187 |
| 10 | momentum | support_distance | -0.092 | 2.470 | Q4 | Q1 | 0.408 | -2.062 |
| 11 | trend_following | support_distance | -0.923 | 2.431 | Q3 | Q1 | -0.569 | -3.000 |
| 12 | momentum | rv_5 | -0.624 | 2.342 | Q1 | Q3 | -0.384 | -2.726 |
| 13 | trend_following | support_distance | -0.474 | 2.310 | Q3 | Q1 | 0.033 | -2.277 |
| 14 | trend_following | vol_regime | -0.923 | 2.234 | 3.0 | 1.0 | -0.766 | -3.000 |
| 15 | sr_bounce | rv_percentile | 0.901 | 2.223 | Q5 | Q1 | 2.371 | 0.148 |
| 16 | sr_bounce | rv_percentile | 1.111 | 2.136 | Q4 | Q1 | 2.710 | 0.574 |
| 17 | sr_bounce | rv_20 | 0.901 | 2.115 | Q2 | Q1 | 2.371 | 0.255 |
| 18 | trend_following | support_distance | -0.570 | 2.108 | Q4 | Q1 | -0.051 | -2.160 |
| 19 | momentum | rv_5 | -0.376 | 2.106 | Q1 | Q4 | -0.179 | -2.286 |
| 20 | trend_following | support_distance | -0.905 | 2.073 | Q3 | Q1 | -0.691 | -2.764 |
| 21 | trend_following | rv_20 | -0.570 | 2.014 | Q1 | Q5 | -0.418 | -2.432 |
| 22 | sr_bounce | rv_5 | 0.901 | 2.014 | Q2 | Q1 | 2.519 | 0.506 |
| 23 | trend_following | adx | -0.131 | 2.009 | Q1 | Q5 | -0.016 | -2.025 |
| 24 | trend_following | vol_regime | -0.962 | 1.958 | 0.0 | 2.0 | -0.467 | -2.425 |
| 25 | trend_following | rv_5 | -1.101 | 1.922 | Q1 | Q3 | -0.934 | -2.856 |
| 26 | trend_following | rv_20 | -0.873 | 1.908 | Q1 | Q4 | -0.241 | -2.149 |
| 27 | trend_following | rv_5 | -0.570 | 1.863 | Q1 | Q4 | -0.334 | -2.197 |
| 28 | sr_bounce | rv_5 | 0.506 | 1.827 | Q3 | Q1 | 2.112 | 0.285 |
| 29 | trend_following | normalized_range | -0.962 | 1.821 | Q1 | Q3 | -0.483 | -2.305 |
| 30 | momentum | resistance_distance | -0.238 | 1.817 | Q4 | Q1 | -0.046 | -1.863 |

## 3. Regime Heatmap
Average best Sharpe by strategy x context variable.

```
context                      adx       atr     hurst  normalized_range  relative_volume  resistance_distance     rv_20      rv_5  rv_percentile  sma200_distance  sma200_slope  support_distance  vol_regime  volume_percentile  volume_spike
strategy                                                                                                                                                                                                                                     
breakout               -0.387418 -0.391958 -0.420676         -0.482119        -0.379875            -0.358469 -0.438886 -0.484873      -0.535694        -0.339557     -0.424630         -0.348723   -0.505342          -0.507824     -0.358384
gap_fade                0.093985  0.150335  0.114160          0.033772         0.087492             0.148428  0.149555  0.107275       0.132428         0.119734      0.130785          0.128757    0.110752           0.112386      0.096323
mean_reversion          0.054473 -0.002020  0.016206          0.096697         0.035318             0.079885  0.034688  0.175830      -0.108662         0.063591      0.025302          0.055560   -0.029838           0.033072      0.308281
momentum               -0.317474 -0.331633 -0.352089         -0.472353        -0.502525            -0.360178 -0.440406 -0.604857      -0.351468        -0.235532     -0.264730         -0.372846   -0.406560          -0.359595     -0.439646
opening_range_breakout -0.413911 -0.405073 -0.448896         -0.524407        -0.400943            -0.375396 -0.452029 -0.512175      -0.557132        -0.359902     -0.429909         -0.352505   -0.548282          -0.522949     -0.397552
sr_bounce               0.084608  0.101001  0.181691          0.197515         0.143434             0.066609  0.314111  0.319200       0.172003         0.028972      0.085172          0.068215    0.187829           0.191250      0.150833
trend_following        -0.451070 -0.394529 -0.480586         -0.461851        -0.489293            -0.512989 -0.664110 -0.754505      -0.460030        -0.367264     -0.319386         -0.631423   -0.488557          -0.482774     -0.450156
vwap_reversion          0.471745  0.434066  0.425500          0.455576         0.284317             0.475968  0.635529  0.584322       0.451249         0.577011      0.485145          0.490545    0.483973           0.508743      0.600685
```

## 4. Best Findings
- **sr_bounce** when **vol_regime=2.0**: Sharpe=3.000 vs overall=0.901 (gap=2.807)
- **sr_bounce** when **rv_20=Q4**: Sharpe=3.000 vs overall=0.506 (gap=2.721)
- **sr_bounce** when **atr=Q2**: Sharpe=2.922 vs overall=0.901 (gap=2.710)
- **sr_bounce** when **volume_percentile=Q1**: Sharpe=2.881 vs overall=1.111 (gap=2.618)
- **sr_bounce** when **normalized_range=Q2**: Sharpe=2.777 vs overall=0.901 (gap=2.590)
- **sr_bounce** when **rv_percentile=Q5**: Sharpe=2.371 vs overall=0.901 (gap=2.223)
- **sr_bounce** when **rv_percentile=Q4**: Sharpe=2.710 vs overall=1.111 (gap=2.136)
- **sr_bounce** when **rv_20=Q2**: Sharpe=2.371 vs overall=0.901 (gap=2.115)
- **sr_bounce** when **rv_5=Q2**: Sharpe=2.519 vs overall=0.901 (gap=2.014)
- **sr_bounce** when **rv_5=Q3**: Sharpe=2.112 vs overall=0.506 (gap=1.827)

## 5. Per-Strategy Detail
### breakout
| Context | Overall Sharpe | Sharpe Gap | Best Regime | Worst Regime |
|---------|---------------|------------|-------------|--------------|
| vol_regime | -0.846 | 1.579 | 3.0 (-0.802) | 2.0 (-2.380) |
| vol_regime | -0.630 | 1.252 | 1.0 (0.225) | 0.0 (-1.027) |
| volume_percentile | -0.540 | 1.172 | Q5 (-0.306) | Q4 (-1.478) |
| rv_5 | -0.453 | 1.164 | Q1 (-0.217) | Q4 (-1.381) |
| vol_regime | -0.434 | 1.112 | 3.0 (-0.243) | 2.0 (-1.355) |

### gap_fade
| Context | Overall Sharpe | Sharpe Gap | Best Regime | Worst Regime |
|---------|---------------|------------|-------------|--------------|
| vol_regime | 0.410 | 1.334 | 2.0 (0.804) | 1.0 (-0.530) |
| rv_5 | -0.082 | 1.154 | Q3 (0.528) | Q1 (-0.625) |
| vol_regime | 0.208 | 1.106 | 1.0 (0.766) | 2.0 (-0.339) |
| vol_regime | 0.283 | 0.951 | 2.0 (0.977) | 1.0 (0.026) |
| rv_5 | 0.144 | 0.802 | Q4 (0.422) | Q1 (-0.380) |

### mean_reversion
| Context | Overall Sharpe | Sharpe Gap | Best Regime | Worst Regime |
|---------|---------------|------------|-------------|--------------|
| rv_5 | 0.404 | 1.665 | Q3 (1.672) | Q1 (0.006) |
| rv_5 | 0.084 | 1.584 | Q3 (1.389) | Q1 (-0.195) |
| rv_5 | 0.338 | 1.458 | Q3 (1.286) | Q1 (-0.172) |
| support_distance | -0.187 | 1.350 | Q2 (0.462) | Q4 (-0.889) |
| rv_5 | 0.422 | 1.262 | Q3 (0.990) | Q1 (-0.272) |

### momentum
| Context | Overall Sharpe | Sharpe Gap | Best Regime | Worst Regime |
|---------|---------------|------------|-------------|--------------|
| support_distance | -0.226 | 2.766 | Q4 (0.253) | Q1 (-2.513) |
| support_distance | -0.092 | 2.470 | Q4 (0.408) | Q1 (-2.062) |
| rv_5 | -0.624 | 2.342 | Q1 (-0.384) | Q3 (-2.726) |
| rv_5 | -0.376 | 2.106 | Q1 (-0.179) | Q4 (-2.286) |
| resistance_distance | -0.238 | 1.817 | Q4 (-0.046) | Q1 (-1.863) |

### opening_range_breakout
| Context | Overall Sharpe | Sharpe Gap | Best Regime | Worst Regime |
|---------|---------------|------------|-------------|--------------|
| rv_5 | -0.387 | 1.568 | Q1 (0.017) | Q3 (-1.551) |
| vol_regime | -0.587 | 1.539 | 3.0 (-0.478) | 2.0 (-2.017) |
| vol_regime | -0.444 | 1.477 | 3.0 (-0.324) | 2.0 (-1.801) |
| vol_regime | -0.839 | 1.460 | 3.0 (-0.799) | 2.0 (-2.259) |
| vol_regime | -0.356 | 1.380 | 2.0 (0.264) | 1.0 (-1.116) |

### sr_bounce
| Context | Overall Sharpe | Sharpe Gap | Best Regime | Worst Regime |
|---------|---------------|------------|-------------|--------------|
| vol_regime | 0.901 | 2.807 | 2.0 (3.000) | 0.0 (0.193) |
| rv_20 | 0.506 | 2.721 | Q4 (3.000) | Q1 (0.279) |
| atr | 0.901 | 2.710 | Q2 (2.922) | Q1 (0.212) |
| volume_percentile | 1.111 | 2.618 | Q1 (2.881) | Q5 (0.263) |
| normalized_range | 0.901 | 2.590 | Q2 (2.777) | Q1 (0.187) |

### trend_following
| Context | Overall Sharpe | Sharpe Gap | Best Regime | Worst Regime |
|---------|---------------|------------|-------------|--------------|
| volume_percentile | -0.923 | 2.761 | Q5 (-0.223) | Q2 (-2.984) |
| rv_5 | -0.550 | 2.745 | Q1 (-0.255) | Q4 (-3.000) |
| rv_20 | -0.550 | 2.704 | Q1 (-0.296) | Q4 (-3.000) |
| support_distance | -0.923 | 2.431 | Q3 (-0.569) | Q1 (-3.000) |
| support_distance | -0.474 | 2.310 | Q3 (0.033) | Q1 (-2.277) |

### vwap_reversion
| Context | Overall Sharpe | Sharpe Gap | Best Regime | Worst Regime |
|---------|---------------|------------|-------------|--------------|
| rv_percentile | 0.098 | 1.480 | Q3 (1.484) | Q5 (0.004) |
| vol_regime | 0.099 | 1.461 | 1.0 (1.505) | 2.0 (0.044) |
| volume_percentile | 0.099 | 1.439 | Q2 (1.426) | Q5 (-0.013) |
| resistance_distance | 0.096 | 1.302 | Q3 (1.344) | Q2 (0.043) |
| hurst | 0.091 | 1.298 | Q5 (1.197) | Q3 (-0.101) |

## 6. Conclusions
1. **Strongest regime effect**: sr_bounce x vol_regime (Sharpe gap=2.807)
2. **Contexts with strong discrimination** (gap > 0.5): 1118
3. **Total hypotheses tested**: 4014
4. **Promising regime conditions** (best Sharpe > 0.5): 451

### Key Insights
- The edge is NOT in the signal — it's in the context.
- Same strategy can be profitable in one regime and loss-making in another.
- Context-aware strategies can dramatically improve risk-adjusted returns.
- The regime map provides a quantitative framework for strategy allocation.

---
*Report generated automatically by market_regime_edge/experiment.py*
