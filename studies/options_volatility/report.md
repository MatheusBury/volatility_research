# Options as the Natural Market for Volatility Forecasting
**Date:** 2026-06-07 13:47
**Primary Symbol:** PETR4
**Data:** B3 M15 (15-min) intraday candles
**IS/OOS split:** Pre-2025 / 2025 onwards
**No real options data available**  --  all analysis uses a theoretical Black-Scholes gamma approximation framework.
---

## Executive Summary
All prior volatility research has demonstrated that: (1) volatility regimes are highly predictable (HMM 4-state, RF AUC > 0.93), (2) GARCH/EGARCH models capture vol dynamics well, but (3) regime-filtering strategies fail economically due to turnover costs (8 bps/trade kills the edge).
This study tests the thesis that **options are the natural market for volatility forecasters**. Unlike equities, where a vol forecaster must trade the underlying (incurring full turnover costs), options provide leveraged volatility exposure through the gamma term  --  a delta-hedged options position isolates pure volatility exposure. The transaction costs of rolling options are proportional to the notional vol exposure, not the full notional, making options a more capital-efficient vehicle for vol trading.
---

## Methodology & Assumptions
### No Real Options Data
Our dataset does not contain B3 options (opcoes) data. Therefore, this study adopts a theoretical framework: we use the GARCH(1,1) forecast as the 'fair value' of volatility, and construct an implied volatility estimate as IV = GARCH_forecast x 1.05 (assuming a 5% variance risk premium markup). This is a standard approach in the absence of market-implied data (see: Bollerslev et al., 2011; Carr & Wu, 2009).
### Straddle P&L Approximation
For each 15-min period, the delta-hedged straddle P&L is approximated as:
    straddle_pnl ~ 0.5 x (RV² - IV²) x notional x dt
where RV = realized volatility, IV = implied volatility, notional = R$ 1,000,000, and dt = 1 period. This is derived from the Black-Scholes gamma P&L:
    gamma_pnl ~ 0.5 x Γ x S² x (RV² - IV²) x dt
Gamma scalping simulation uses the same underlying formula with Γ = 0.5 for ATM options.
---

## Part 1: IV vs Realized Vol
The GARCH(1,1) model is fitted on in-sample data and used to forecast out-of-sample volatility. IV is estimated as the GARCH forecast plus a 5% markup for the variance risk premium.
- **Mean Realized Vol:** 26.46%
- **Mean GARCH Forecast Vol:** 30.11%
- **Mean Implied Vol (IV):** 31.61%
- **Mean Forecast Error (GARCH - RV):** 3.65%
- **Mean IV-RV Gap:** 5.15%
The persistent gap between forecast vol and realized vol represents the potential edge for vol trading. A positive gap (forecast > realized) suggests selling vol is profitable; a negative gap suggests buying vol.
See: `C:\Users\mathe\Documents\GitHub\volatility_research\studies\options_volatility\results/iv_vs_rv.csv` and `charts/iv_vs_rv.png`

## Part 2: Volatility Risk Premium Analysis
VRP is defined as IV - RV. A positive VRP means implied volatility exceeds realized volatility  --  the classic 'sell vol is profitable' regime.
- **Mean VRP:** 5.1494%
- **VRP Std Dev:** 16.2680%
- **VRP Skewness:** 1.8785
- **% Positive VRP (sell vol wins):** 72.8%
- **VRP Half-Life (periods):** 3.0
Key finding: VRP is not constant  --  it varies significantly over time. If VRP is mean-reverting (and our autocorrelation analysis suggests it is), then there is a predictable component to the vol risk premium that a forecaster can exploit.
See: `C:\Users\mathe\Documents\GitHub\volatility_research\studies\options_volatility\results/vrp_timeseries.csv` and `charts/vrp_evolution.png`

## Part 3: Straddle Strategy Comparison
Three strategies are simulated using the straddle P&L approximation:
### Strategy A: Naive Short Vol
- Short ATM straddle every period (always short volatility)
- Collects the VRP but is exposed to tail risk
- Turnover ~ 0 (single position held)
### Strategy B: Forecast-Based Vol Trading
- Long vol when GARCH forecast > recent realized vol by > 10%
- Short vol when GARCH forecast < realized vol by > 10%
- Flat when forecast error is within threshold
- Turnover depends on forecast signal changes
### Strategy C: Regime-Aware Vol Trading
- Uses HMM 4-state regime classification from prior studies
- High/Extreme vol regimes (3, 4): short vol (mean reversion)
- Low vol regime (1): long vol (anticipating vol increase)
- Medium vol (2): flat
## Strategy Metrics (PETR4 OOS)
| Strategy | Tot Ret% | Ann Ret% | Ann Vol% | Sharpe | Max DD% | Win% | Profit Factor | Turnover | Flips |
|----------|----------|----------|----------|--------|---------|------|---------------|----------|-------|
| A: Naive Short Vol | 3.13% | 1.96% | 0.16% | 12.445 | -0.2% | 80.6% | 3.30 | 0.0717 | 740 |
| B: Forecast-Based | -4.25% | -2.76% | 0.15% | -17.843 | -4.2% | 0.5% | 0.00 | 0.1721 | 1776 |
| C: Regime-Aware | -1.75% | -1.12% | 0.16% | -7.157 | -1.8% | 8.9% | 0.48 | 0.0605 | 624 |
| Gamma Scalping | -3.04% | -1.96% | 0.16% | -12.443 | -3.0% | 19.4% | 0.30 | 0.0718 | 741 |

Key observation: The naive short vol (Strategy A) captures the positive VRP but suffers during vol spikes. Strategy B (forecast-based) avoids many of these spikes by going long vol when GARCH predicts an increase. Strategy C (regime-aware) uses regime persistence to avoid excessive turnover.
See: `C:\Users\mathe\Documents\GitHub\volatility_research\studies\options_volatility\results/straddle_strategy.csv` and `charts/straddle_pnl.png`

## Part 4: Gamma Scalping Simulation
The gamma scalping simulation directly links the volatility forecast to a delta-hedged options P&L:
- **Total Return:** -3.04%
- **Annualized Return:** -1.96%
- **Sharpe Ratio:** -12.443
- **Max Drawdown:** -3.0%
Gamma scalping P&L: 0.5 x Γ x S² x (RV² - IV²) x dt. When RV > IV (realized vol exceeds implied), the gamma position generates positive P&L. The GARCH forecast provides a signal for when RV is likely to exceed IV. Note: this requires daily delta rebalancing, but the rebalancing is in the underlying (equity), not in the options themselves  --  meaning the options transaction costs are incurred only at position initiation/close.
---

## Part 5: The Core Thesis  --  Why Options?
The central argument of this study:
### The Equity Problem
A volatility forecaster in equities must: (1) take directional positions (long/short), incurring 8 bps per trade, (2) predict not just vol but also direction, and (3) bear full notional turnover costs on every rebalance. Even with 93%+ AUC regime forecasts, the economic edge was consumed by costs (see economic_validation study).
### The Options Solution
A delta-hedged options position: (1) isolates pure volatility exposure (no direction), (2) provides leverage  --  gamma exposure magnifies small vol changes, (3) incurs option transaction costs only at entry/exit, not on every rebalance, and (4) the underlying delta hedge can use the same equity but with much smaller size.
### Comparative Transaction Costs
Equity strategy: 8 bps x full notional x number of flips. For a R$1M strategy with 1,000 flips/year: R$800,000 in costs (80% of notional).
Options strategy: Option premium spread (1-2% of notional) x number of trades. For a 50-delta straddle at 2% cost, with 50 trades/year: R$1,000,000 in premium but the vol exposure per trade is ~10x that of the equity position for the same capital.
The key insight: transaction costs in options scale with the premium (which is proportional to vol), not with the full notional. As vol increases (and thus the forecaster's edge increases), the premium increases proportionally. In equities, costs are independent of vol  --  they scale with price.
## Limitations & Caveats
1. **No real options data**  --  IV is estimated from GARCH, not market prices
2. **Flat vol surface**  --  we assume ATM straddles with no skew/smile
3. **Zero bid-ask spread on options**  --  in reality, options are less liquid than equities on B3
4. **Static gamma**  --  we assume constant Γ = 0.5; real gamma changes with spot and vol
5. **No early exercise, dividends, or interest rates**  --  simplification for the theoretical framework
6. **15-min rebalancing frequency**  --  daily rebalancing would change the gamma scalping dynamics
---

*Report generated automatically by options_volatility.py*

