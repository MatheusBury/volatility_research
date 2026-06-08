# Economic Validation: Regime Filter with Realistic B3 Costs
**Generated:** 2026-06-07 13:35
**Universe:** PETR4, VALE3, ITUB4
**Data:** B3 M15 (15-min) intraday — HMM 4-State Regimes
**IS (HMM Training):** 2021-01-01 to 2024-12-31
**OOS (Strategy Evaluation):** 2025-01-01 to 2026-05-29
**Cost Model:** 5 bps commission + 3 bps slippage = **8 bps per trade**
---
## 1. Performance Summary (Net of Costs)

| Symbol | Strategy | Gross Ret% | Net Ret% | Ann. Ret% | Ann. Vol% | Net Sharpe | Max DD% | Trades | Costs% | Edge Erosion% |
|--------|----------|-----------|---------|-----------|----------|-----------|--------|--------|--------|---------------|
| ITUB4 | A: Always On | 41.47% | 41.25% | 21.98% | 20.47% | 1.074 | -18.9% | 2 | 0.16% | 0.5% |
| ITUB4 | B: Regime Filter | 19.48% | -8.61% | -5.73% | 15.78% | -0.363 | -19.4% | 335 | 26.80% | 150.5% |
| ITUB4 | C: Vol Scaled | 7.29% | -15.91% | -11.03% | 12.43% | -0.887 | -19.6% | 576 | 24.36% | 345.2% |
| PETR4 | A: Always On | 6.60% | 6.43% | 3.97% | 23.93% | 0.166 | -19.9% | 2 | 0.16% | 2.5% |
| PETR4 | B: Regime Filter | 11.40% | -13.00% | -8.87% | 17.74% | -0.500 | -17.7% | 309 | 24.72% | 228.6% |
| PETR4 | C: Vol Scaled | 2.61% | -18.86% | -13.31% | 14.93% | -0.891 | -20.5% | 504 | 23.47% | 909.0% |
| VALE3 | A: Always On | 53.19% | 52.95% | 27.06% | 21.33% | 1.269 | -16.3% | 2 | 0.16% | 0.4% |
| VALE3 | B: Regime Filter | 44.42% | 12.61% | 7.56% | 16.42% | 0.461 | -24.3% | 311 | 24.88% | 67.7% |
| VALE3 | C: Vol Scaled | 28.62% | 1.35% | 0.86% | 13.57% | 0.063 | -21.2% | 551 | 23.82% | 94.7% |

## 2. Gross vs Net Sharpe Comparison

| Symbol | Strategy | Gross Sharpe | Net Sharpe | Delta |
|--------|----------|-------------|-----------|-------|
| ITUB4 | A: Always On | 1.079 | 1.074 | -0.005 |
| ITUB4 | B: Regime Filter | 0.719 | -0.363 | -1.082 |
| ITUB4 | C: Vol Scaled | 0.362 | -0.887 | -1.249 |
| PETR4 | A: Always On | 0.170 | 0.166 | -0.004 |
| PETR4 | B: Regime Filter | 0.389 | -0.500 | -0.889 |
| PETR4 | C: Vol Scaled | 0.110 | -0.891 | -1.001 |
| VALE3 | A: Always On | 1.274 | 1.269 | -0.005 |
| VALE3 | B: Regime Filter | 1.426 | 0.461 | -0.966 |
| VALE3 | C: Vol Scaled | 1.182 | 0.063 | -1.118 |

## 3. Cost Sensitivity Analysis (Strategy B)

Net Sharpe at various cost levels:

| Cost (bps) | PETR4 | VALE3 | ITUB4 |
|------------|-------|-------|-------|
| 0 | 0.389 | 1.426 | 0.719 |
| 5 | -0.168 | 0.823 | 0.042 |
| 10 | -0.721 | 0.219 | -0.633 |
| 20 | -1.809 | -0.977 | -1.960 |
| 50 | -4.783 | -4.281 | -5.528 |

Break-even cost level (Sharpe = 0):

- **PETR4**: ~3.5 bps
- **VALE3**: ~11.8 bps
- **ITUB4**: ~5.3 bps

## 4. Critical Analysis: 6 Questions

### Q1: Does Strategy B survive costs?

- **PETR4**: Gross Sharpe 0.389 → Net Sharpe -0.500 (A net Sharpe: 0.166) — **NO**
- **VALE3**: Gross Sharpe 1.426 → Net Sharpe 0.461 (A net Sharpe: 1.269) — **NO**
- **ITUB4**: Gross Sharpe 0.719 → Net Sharpe -0.363 (A net Sharpe: 1.074) — **NO**

### Q2: Does Strategy C survive costs?

- **PETR4**: Gross Sharpe 0.110 → Net Sharpe -0.891 (A net Sharpe: 0.166) — **NO**
- **VALE3**: Gross Sharpe 1.182 → Net Sharpe 0.063 (A net Sharpe: 1.269) — **NO**
- **ITUB4**: Gross Sharpe 0.362 → Net Sharpe -0.887 (A net Sharpe: 1.074) — **NO**

### Q3: How much of the gross edge is consumed by costs? (Edge erosion %)

- **ITUB4 A: Always On**: 0.5% of gross edge consumed by costs
- **ITUB4 B: Regime Filter**: 150.5% of gross edge consumed by costs
- **ITUB4 C: Vol Scaled**: 345.2% of gross edge consumed by costs
- **PETR4 A: Always On**: 2.5% of gross edge consumed by costs
- **PETR4 B: Regime Filter**: 228.6% of gross edge consumed by costs
- **PETR4 C: Vol Scaled**: 909.0% of gross edge consumed by costs
- **VALE3 A: Always On**: 0.4% of gross edge consumed by costs
- **VALE3 B: Regime Filter**: 67.7% of gross edge consumed by costs
- **VALE3 C: Vol Scaled**: 94.7% of gross edge consumed by costs

### Q4: Is the net edge consistent across all 3 stocks?

- Strategy B (Regime Filter) beats Always-On net of costs in **0/3** stocks
- Strategy C (Vol Scaled) beats Always-On net of costs in **0/3** stocks
- **Conclusion**: Net edge is INCONSISTENT across stocks for B

### Q5: What is the break-even cost level?

The break-even cost level (where net Sharpe = 0) for each stock:

- **PETR4**: Break-even at **3.5 bps**
- **VALE3**: Break-even at **11.8 bps**
- **ITUB4**: Break-even at **5.3 bps**

### Q6: Final Verdict — Economic edge or statistical edge?

- **Verdict**: STATISTICAL EDGE
- **Evidence**: Net edge is consumed by costs; the gains are purely statistical
- **Avg Net Sharpe (A/B/C)**: 0.836 / -0.134 / -0.572
- **Avg Edge Erosion**: 199.9%
- **Avg Break-even Cost**: 6.9 bps
- **At 0 bps (no costs)**: Strategy B survives vs A (avg B Sharpe=0.845 vs avg A=0.836)

## 5. Conclusions & Recommendations

1. **Impact of costs is material**: Even 8 bps per trade significantly erodes the gross edge.
2. **Regime Filter (B)** does NOT have a net economic edge over Always-On.
3. **Vol Scaling (C)** does NOT have a net economic edge over Always-On.
4. **Edge buffer is narrow**: Break-even costs indicate how much room there is before the edge is fully consumed.
5. **Cost-aware strategy design**: To preserve the economic edge, minimize turnover and/or negotiate lower costs.

---
*Report generated automatically by economic_validation.py*
