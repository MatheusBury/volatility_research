# Forecast vs IV Relative Value Model Report
**Date:** 2026-06-07 15:26
**Universe:** PETR4, VALE3, ITUB4
**Data:** B3 M15 intraday + MT5 option IV
**Model:** EGARCH(1,1,1) on IS (80%), forecast on OOS (20%)
---
## 1. Historical Table
| Date | Stock | Forecast RV | IV | Spread | Future RV | DTE | Regime |
|------|-------|-------------|----|--------|-----------|-----|--------|
| 2026-04-29 | PETR4 | 83.75% | 26.23% | +57.52% | 29.80% | 79 | 2 |
| 2026-04-30 | PETR4 | 81.47% | 26.41% | +55.06% | 29.70% | 78 | 1 |
| 2026-05-04 | PETR4 | 85.85% | 32.71% | +53.14% | 29.92% | 74 | 2 |
| 2026-02-03 | VALE3 | 29.62% | 31.35% | -1.73% | 29.34% | 164 | 3 |
| 2026-02-12 | VALE3 | 29.60% | 38.59% | -8.99% | 29.34% | 155 | 2 |
| 2026-02-24 | VALE3 | 29.65% | 26.68% | +2.96% | 29.19% | 143 | 0 |
| 2026-02-25 | VALE3 | 29.65% | 36.11% | -6.46% | 29.36% | 142 | 2 |
| 2026-02-26 | VALE3 | 29.66% | 32.57% | -2.91% | 29.25% | 141 | 2 |
| 2026-02-27 | VALE3 | 29.66% | 29.25% | +0.41% | 29.19% | 140 | 0 |
| 2026-04-15 | ITUB4 | 26.11% | 21.35% | +4.76% | 22.91% | 65 | 1 |
| 2026-04-16 | ITUB4 | 26.04% | 19.09% | +6.95% | 23.07% | 64 | 3 |

## 2. Group Analysis
- **Forecast < IV** (n=3): spread=-6.12%, future RV=29.32%
- **Forecast ≈ IV** (n=2): spread=-0.66%, future RV=29.27%
- **Forecast > IV** (n=6): spread=+30.07%, future RV=27.43%

## 3. Hit Rate Analysis
- **Overall hit rate:** 81.82%
- How often does spread sign correctly predict RV direction?
- Hit rate = P(sign(spread) == sign(future_RV - IV))

## 4. Confusion Matrix
- **Long vol signal (Forecast >> IV):** 4 signals, 3 correct (75.00%)
- **Short vol signal (Forecast << IV):** 2 signals, 2 correct (100.00%)
- **No signal:** 5 observations

## 5. Strategy Sharpe
- **Trading the spread signal:** Sharpe = 0.921
- PnL = signal × (future_RV - IV) for each observation
- Annualized using sqrt(252 / avg DTE)

## 6. Calibration Table
| Bucket | n | Mean Spread | Mean Future RV |
|--------|---|---|---|
| IV >> Forecast | 2 | -0.0772 | 0.2935 |
| IV > Forecast | 1 | -0.0291 | 0.2925 |
| ≈ Fair | 2 | -0.0066 | 0.2927 |
| Forecast > IV | 2 | +0.0386 | 0.2605 |
| Forecast >> IV | 4 | +0.4317 | 0.2812 |

## 7. Summary Statistics
- **Total observations:** 11
- **Mean forecast RV:** 43.73%
- **Mean implied IV:** 29.12%
- **Mean spread:** +14.61%
- **Mean future RV:** 28.28%
- **Spread range:** [-8.99%, +57.52%]
- **DTE range:** [64, 164]

## 8. Regime Distribution
- **Regime 0:** 2 observations (18.2%)
- **Regime 1:** 2 observations (18.2%)
- **Regime 2:** 5 observations (45.5%)
- **Regime 3:** 2 observations (18.2%)

---
*Report generated automatically by study.py*
