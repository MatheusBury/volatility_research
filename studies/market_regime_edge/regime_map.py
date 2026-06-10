from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from studies.market_regime_edge.strategies import TradeRecord


@dataclass
class ContextAnalysisResult:
    context_column: str
    strategy: str
    overall_sharpe: float
    overall_n_trades: int
    regime_sharpes: dict = field(default_factory=dict)
    sharpe_gap: float = 0.0  # best regime sharpe - worst regime sharpe
    best_regime: str = ""
    worst_regime: str = ""
    improvement_ratio: float = 0.0


TRADE_ANNUALIZATION = 252 * 26  # M15 bars per year


def _mean_return_sharpe(rets: np.ndarray) -> float:
    if len(rets) < 3:
        return 0.0
    m = float(np.mean(rets))
    s = float(np.std(rets, ddof=1))
    if s < 1e-10:
        return 0.0
    trade_sharpe = m / s
    return float(np.clip(trade_sharpe, -3.0, 3.0))


def analyze_context_variable(
    trades: list[TradeRecord],
    context_column: str,
    n_buckets: int = 5,
    min_trades_per_bucket: int = 30,
) -> ContextAnalysisResult | None:
    if not trades:
        return None

    strategy = trades[0].strategy
    pairs = [
        (t.simple_return, t.context.get(context_column))
        for t in trades
        if context_column in t.context
        and t.context[context_column] is not None
        and not (isinstance(t.context[context_column], float) and np.isnan(t.context[context_column]))
    ]
    if len(pairs) < min_trades_per_bucket:
        return None

    values = np.array([p[1] for p in pairs], dtype=float)
    rets = np.array([p[0] for p in pairs], dtype=float)

    overall_sharpe = _mean_return_sharpe(rets)

    if len(np.unique(values)) < 2:
        return None

    regime_sharpes: dict[str, float] = {}

    if len(np.unique(values)) <= n_buckets:
        unique_vals = sorted(np.unique(values))
        for val in unique_vals:
            mask = values == val
            bucket_rets = rets[mask]
            if len(bucket_rets) < min_trades_per_bucket:
                continue
            regime_sharpes[str(val)] = _mean_return_sharpe(bucket_rets)
    else:
        buckets = np.linspace(values.min(), values.max(), n_buckets + 1)
        for i in range(n_buckets):
            if i == n_buckets - 1:
                mask = values >= buckets[i]
            else:
                mask = (values >= buckets[i]) & (values < buckets[i + 1])
            bucket_rets = rets[mask]
            if len(bucket_rets) < min_trades_per_bucket:
                continue
            regime_sharpes[f"Q{i+1}"] = _mean_return_sharpe(bucket_rets)

    sharpes_list = [v for v in regime_sharpes.values() if isinstance(v, (int, float))]
    if len(sharpes_list) < 2:
        return None

    best_key = max(regime_sharpes, key=lambda k: regime_sharpes[k])
    worst_key = min(regime_sharpes, key=lambda k: regime_sharpes[k])

    result = ContextAnalysisResult(
        context_column=context_column,
        strategy=strategy,
        overall_sharpe=float(overall_sharpe),
        overall_n_trades=len(pairs),
        regime_sharpes=regime_sharpes,
        sharpe_gap=float(regime_sharpes[best_key] - regime_sharpes[worst_key]),
        best_regime=str(best_key),
        worst_regime=str(worst_key),
        improvement_ratio=float(
            regime_sharpes[best_key] / max(abs(regime_sharpes[worst_key]), 0.01)
            if abs(regime_sharpes[worst_key]) > 1e-10
            else regime_sharpes[best_key] * 10
        ),
    )
    return result


def rank_context_variables(
    all_results: list[ContextAnalysisResult],
) -> pd.DataFrame:
    rows = []
    for r in all_results:
        if r is None:
            continue
        rows.append(
            {
                "strategy": r.strategy,
                "context": r.context_column,
                "overall_sharpe": r.overall_sharpe,
                "n_trades": r.overall_n_trades,
                "sharpe_gap": r.sharpe_gap,
                "best_regime": r.best_regime,
                "worst_regime": r.worst_regime,
                "improvement_ratio": r.improvement_ratio,
                "best_sharpe": max(r.regime_sharpes.values()),
                "worst_sharpe": min(r.regime_sharpes.values()),
            }
        )
    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values("sharpe_gap", ascending=False).reset_index(drop=True)
    return df


def build_regime_heatmap(
    all_results: list[ContextAnalysisResult],
) -> pd.DataFrame:
    rows = []
    for r in all_results:
        if r is None:
            continue
        for regime, sharpe in r.regime_sharpes.items():
            rows.append(
                {
                    "strategy": r.strategy,
                    "context": r.context_column,
                    "regime": regime,
                    "sharpe": sharpe,
                }
            )
    df = pd.DataFrame(rows)
    if len(df) > 0:
        pivot = df.pivot_table(
            values="sharpe",
            index="strategy",
            columns="context",
            aggfunc="mean",
        )
        return pivot
    return pd.DataFrame()
