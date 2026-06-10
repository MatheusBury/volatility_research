from __future__ import annotations

import numpy as np
from scipy import stats

N_BARS_PER_YEAR = 252 * 26


def bootstrap_sharpe(
    returns: np.ndarray,
    n_iterations: int = 10000,
    seed: int = 42,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n_bars = N_BARS_PER_YEAR
    sharpe_values = np.zeros(n_iterations)
    n = len(returns)
    for i in range(n_iterations):
        sample = rng.choice(returns, size=n, replace=True)
        ann_ret = np.mean(sample) * n_bars
        ann_vol = np.std(sample, ddof=1) * np.sqrt(n_bars)
        sharpe_values[i] = ann_ret / ann_vol if ann_vol > 1e-10 else 0.0

    mean_sharpe = float(np.mean(sharpe_values))
    ci_lower = float(np.percentile(sharpe_values, 2.5))
    ci_upper = float(np.percentile(sharpe_values, 97.5))
    return mean_sharpe, ci_lower, ci_upper


def bootstrap_test(
    returns: np.ndarray,
    n_iterations: int = 10000,
    seed: int = 42,
) -> float:
    rng = np.random.default_rng(seed)
    n = len(returns)
    count_positive = 0
    n_bars = N_BARS_PER_YEAR
    for _ in range(n_iterations):
        sample = rng.choice(returns, size=n, replace=True)
        ann_ret = np.mean(sample) * n_bars
        ann_vol = np.std(sample, ddof=1) * np.sqrt(n_bars)
        sharpe = ann_ret / ann_vol if ann_vol > 1e-10 else 0.0
        if sharpe > 0:
            count_positive += 1
    return count_positive / n_iterations


def reality_check(
    returns_matrix: np.ndarray,
    n_permutations: int = 1000,
    seed: int = 42,
) -> float:
    rng = np.random.default_rng(seed)
    n_bars = N_BARS_PER_YEAR

    n_strategies, _ = returns_matrix.shape
    observed_sharpes = np.zeros(n_strategies)
    for s in range(n_strategies):
        ann_ret = np.mean(returns_matrix[s]) * n_bars
        ann_vol = np.std(returns_matrix[s], ddof=1) * np.sqrt(n_bars)
        observed_sharpes[s] = ann_ret / ann_vol if ann_vol > 1e-10 else 0.0
    max_observed = float(np.max(observed_sharpes))

    perm_maxes = np.zeros(n_permutations)
    for p in range(n_permutations):
        perm_sharpes = np.zeros(n_strategies)
        for s in range(n_strategies):
            shuffled = rng.permutation(returns_matrix[s])
            ann_ret = np.mean(shuffled) * n_bars
            ann_vol = np.std(shuffled, ddof=1) * np.sqrt(n_bars)
            perm_sharpes[s] = ann_ret / ann_vol if ann_vol > 1e-10 else 0.0
        perm_maxes[p] = np.max(perm_sharpes)

    p_value = float(np.mean(perm_maxes >= max_observed))
    return p_value


def deflated_sharpe_ratio(
    sharpe_observed: float,
    n_observations: int,
    n_trials: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    if sharpe_observed <= 0:
        return 0.0
    gamma1 = skewness
    gamma2 = kurtosis
    e_max = (
        (1 - np.euler_gamma) * stats.norm.ppf(1 - 1 / n_trials)
        + np.euler_gamma * stats.norm.ppf(1 - 1 / (n_trials * np.e))
    )
    v_max = 1 / (n_trials - 1) * (
        (n_trials - 1)
        * (
            stats.norm.ppf(1 - 1 / n_trials)
            - (
                (1 - np.euler_gamma)
                * stats.norm.ppf(1 - 1 / n_trials)
                + np.euler_gamma * stats.norm.ppf(1 - 1 / (n_trials * np.e))
            )
        )
        ** 2
    )
    numerator = sharpe_observed - e_max
    denominator = np.sqrt(
        (1 - gamma1 * sharpe_observed + (gamma2 - 1) / 4 * sharpe_observed**2)
        / (n_observations - 1)
    )
    if denominator < 1e-10:
        return 0.0
    dsr = numerator / denominator
    dsr = dsr * v_max + e_max
    p_value = float(1 - stats.norm.cdf(dsr))
    return p_value


def regime_conditional_sharpe(
    trade_returns: list[float],
    trade_contexts: list[dict],
    context_column: str,
    n_buckets: int = 5,
) -> dict:
    pairs = [
        (r, ctx.get(context_column))
        for r, ctx in zip(trade_returns, trade_contexts, strict=False)
        if ctx.get(context_column) is not None and not np.isnan(ctx.get(context_column))
    ]
    if len(pairs) < 10:
        return {}

    values = np.array([p[1] for p in pairs])
    rets = np.array([p[0] for p in pairs])
    n_bars = N_BARS_PER_YEAR
    buckets = np.linspace(values.min(), values.max(), n_buckets + 1)
    bucket_labels = [f"Q{i+1}" for i in range(n_buckets)]
    result = {}

    for i in range(n_buckets):
        if i == n_buckets - 1:
            mask = values >= buckets[i]
        else:
            mask = (values >= buckets[i]) & (values < buckets[i + 1])

        bucket_rets = rets[mask]
        if len(bucket_rets) < 3:
            continue

        ann_ret = np.mean(bucket_rets) * n_bars
        ann_vol = np.std(bucket_rets, ddof=1) * np.sqrt(n_bars)
        sharpe = ann_ret / ann_vol if ann_vol > 1e-10 else 0.0

        result[bucket_labels[i]] = {
            "n_trades": int(np.sum(mask)),
            "avg_return": float(np.mean(bucket_rets)),
            "sharpe": float(sharpe),
            "hit_ratio": float(np.mean(bucket_rets > 0)),
        }

    return result
