"""
VALE3 Volatility Forecast Calibration Study
=============================================
Tests EGARCH forecast power against realized RV across 2021-2026.
Builds decile table, regime-conditional forecasts, and monotonicity tests.

Usage:
    python studies/calibration_vale3/study.py
"""
from __future__ import annotations

import sys
import logging
import warnings
from pathlib import Path
from typing import Any

import matplotlib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

from models.egarch import EGARCHModel
from models.hmm_regime import HMMRegimeModel

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("calibration")

STUDY_DIR = Path(__file__).parent.resolve()
RESULTS_DIR = STUDY_DIR / "results"
CHARTS_DIR = STUDY_DIR / "charts"
PARQUET_DIR = r"C:\Users\mathe\Documents\GitHub\mt5\dataset\export_mt5\intraday\avista\M15"

STOCK = "VALE3"
PREFIX = "VALE"
HORIZON_DAYS = 5
MIN_TRAIN_YEARS = 1
ROLL_FREQ_DAYS = 1
RETURN_SCALE = 1000.0
N_BARS_PER_YEAR = 252 * 26


def load_m15(symbol: str) -> pd.DataFrame:
    path = Path(PARQUET_DIR) / f"{symbol}.parquet"
    df = pd.read_parquet(path)
    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df = df.dropna(subset=["log_return"]).reset_index(drop=True)
    return df


def compute_future_rv(returns: np.ndarray, n_bars: int) -> float:
    window = returns[:n_bars]
    if len(window) < 5:
        return np.nan
    return float(window.std() * np.sqrt(N_BARS_PER_YEAR))


def run_study() -> None:
    logger.info("=" * 70)
    logger.info("VALE3 VOLATILITY FORECAST CALIBRATION")
    logger.info("=" * 70)

    # ── Load data ──────────────────────────────────────────────────────
    logger.info("\nLoading M15 data for VALE3...")
    df = load_m15(STOCK)
    n_total = len(df)
    start_date = df["time"].iloc[0]
    end_date = df["time"].iloc[-1]
    logger.info(f"  {n_total:,} bars from {start_date.date()} to {end_date.date()}")

    # ── Train HMM once on full dataset ─────────────────────────────────
    logger.info("\nFitting HMM(4) on full dataset...")
    hmm = HMMRegimeModel(n_regimes=4)
    log_ret_series = df["log_return"].copy()
    hmm.fit(log_ret_series)
    predicted_regimes = hmm.predict(log_ret_series)
    # Pad front to align with df index (HMM drops first 29 bars for vol_window=30)
    pad_len = len(df) - len(predicted_regimes)
    all_regimes = np.concatenate([np.full(pad_len, 0, dtype=int), predicted_regimes])
    regime_counts = pd.Series(all_regimes).value_counts().sort_index()
    logger.info(f"  Regime distribution: {dict(regime_counts)}")

    # ── Rolling forecast ───────────────────────────────────────────────
    n_train_bars = int(MIN_TRAIN_YEARS * 252 * 26)
    n_forecast_bars = int(HORIZON_DAYS * 26)
    roll_step = int(ROLL_FREQ_DAYS * 26)

    records: list[dict[str, Any]] = []
    last_log_idx = 0

    logger.info(f"\nRolling EGARCH forecast every {ROLL_FREQ_DAYS}d...")
    logger.info(f"  Train window: {MIN_TRAIN_YEARS}y ({n_train_bars:,} bars)")
    logger.info(f"  Forecast horizon: {HORIZON_DAYS}d ({n_forecast_bars} bars)")
    logger.info(f"  Step size: {ROLL_FREQ_DAYS}d ({roll_step} bars)")

    n_forecasts = 0
    n_fail = 0

    for start_idx in range(0, n_total - n_train_bars - n_forecast_bars, roll_step):
        train_slice = df.iloc[start_idx:start_idx + n_train_bars]
        train_ret = train_slice["log_return"]

        try:
            model = EGARCHModel(p=1, o=1, q=1, scale=RETURN_SCALE)
            model.fit(train_ret)
            fcast_vols = model.forecast(n_forecast_bars)
            forecast_rv = float(np.mean(fcast_vols))
        except Exception:
            n_fail += 1
            continue

        future_ret = df["log_return"].iloc[start_idx + n_train_bars: start_idx + n_train_bars + n_forecast_bars].values
        future_rv = compute_future_rv(future_ret, n_forecast_bars)
        if np.isnan(future_rv):
            continue

        forecast_date = df["time"].iloc[start_idx + n_train_bars]
        regime_idx = start_idx + n_train_bars
        regime = int(all_regimes[min(regime_idx, len(all_regimes) - 1)])

        records.append({
            "date": forecast_date,
            "forecast_rv": forecast_rv,
            "future_rv": future_rv,
            "error": future_rv - forecast_rv,
            "abs_error": abs(future_rv - forecast_rv),
            "regime": regime,
            "n_train": n_train_bars,
        })
        n_forecasts += 1

        if n_forecasts % 20 == 0:
            logger.info(f"  {n_forecasts} forecasts generated...")

    logger.info(f"\n  Total: {n_forecasts} forecasts, {n_fail} failures")

    if n_forecasts == 0:
        logger.error("No forecasts generated. Check data availability.")
        return

    result_df = pd.DataFrame(records)

    # ── Decile Analysis ────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("DECILE ANALYSIS")
    logger.info("=" * 70)

    result_df["decile"] = pd.qcut(result_df["forecast_rv"], 10, labels=range(1, 11))
    decile_table = result_df.groupby("decile", observed=True).agg(
        n=("future_rv", "count"),
        forecast_rv=("forecast_rv", "mean"),
        future_rv=("future_rv", "mean"),
        error=("error", "mean"),
        abs_error=("abs_error", "mean"),
    ).round(4)
    decile_table["future_rv_pct"] = decile_table["future_rv"] * 100
    decile_table["forecast_rv_pct"] = decile_table["forecast_rv"] * 100

    logger.info("\nDecil | N | Forecast RV | Future RV | Erro | |Erro|")
    logger.info("-" * 65)
    for dec in range(1, 11):
        row = decile_table.loc[dec]
        logger.info(
            f"  {dec:2d}    | {int(row['n']):3d} | "
            f"{row['forecast_rv_pct']:6.2f}% | "
            f"{row['future_rv_pct']:6.2f}% | "
            f"{row['error']*100:+6.2f}% | "
            f"{row['abs_error']*100:6.2f}%"
        )

    # Monotonicity test
    decile_means = result_df.groupby("decile", observed=True)["future_rv"].mean().values
    forecast_means = result_df.groupby("decile", observed=True)["forecast_rv"].mean().values
    corr, pval = spearmanr(forecast_means, decile_means)
    logger.info(f"\n  Spearman correlation (decile means): {corr:.4f} (p={pval:.4f})")

    rank_corr, rank_pval = spearmanr(result_df["forecast_rv"], result_df["future_rv"])
    logger.info(f"  Spearman rank correlation (all obs): {rank_corr:.4f} (p={rank_pval:.4f})")

    # ── Regime-Conditional Forecast ────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("REGIME-CONDITIONAL FORECAST")
    logger.info("=" * 70)

    regime_table = result_df.groupby("regime").agg(
        n=("future_rv", "count"),
        forecast_rv=("forecast_rv", "mean"),
        future_rv=("future_rv", "mean"),
        error=("error", "mean"),
        abs_error=("abs_error", "mean"),
    ).round(4)

    logger.info(f"\n{'Regime':10s} | {'N':5s} | {'Forecast':8s} | {'Future':8s} | {'Error':8s} | {'|Err|':8s}")
    logger.info("-" * 55)
    regime_names = {0: "Low", 1: "Med", 2: "High", 3: "Extreme"}
    for r in sorted(regime_table.index):
        row = regime_table.loc[r]
        name = regime_names.get(r, str(r))
        logger.info(
            f"  {name:8s} | {int(row['n']):4d} | "
            f"{row['forecast_rv']*100:6.2f}% | "
            f"{row['future_rv']*100:6.2f}% | "
            f"{row['error']*100:+6.2f}% | "
            f"{row['abs_error']*100:6.2f}%"
        )

    # Regime-conditional decile
    logger.info("\nDecile by Regime:")
    cross = result_df.groupby(["decile", "regime"], observed=True).size().unstack(fill_value=0)
    logger.info(f"\n{cross.to_string()}")

    # ── Regime-Aware EGARCH Forecast ───────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("REGIME-AWARE EGARCH (conditional fit)")
    logger.info("=" * 70)

    regime_aware_records: list[dict[str, Any]] = []

    for start_idx in range(0, n_total - n_train_bars - n_forecast_bars, roll_step):
        train_slice = df.iloc[start_idx:start_idx + n_train_bars]
        train_ret = train_slice["log_return"].values
        train_regimes = all_regimes[start_idx:start_idx + n_train_bars]

        forecast_date = df["time"].iloc[start_idx + n_train_bars]
        regime_at_entry = int(all_regimes[min(start_idx + n_train_bars, len(all_regimes) - 1)])

        train_ret_series = train_slice["log_return"]
        low_regime_mask = train_regimes <= 1
        if low_regime_mask.sum() > 100:
            train_ret_filtered = train_ret_series.iloc[:len(train_regimes)][low_regime_mask]
        else:
            train_ret_filtered = train_ret_series

        try:
            model = EGARCHModel(p=1, o=1, q=1, scale=RETURN_SCALE)
            model.fit(train_ret_filtered)
            fcast_vols = model.forecast(n_forecast_bars)
            forecast_rv_reg = float(np.mean(fcast_vols))
        except Exception:
            continue

        future_ret = df["log_return"].iloc[start_idx + n_train_bars: start_idx + n_train_bars + n_forecast_bars].values
        future_rv = compute_future_rv(future_ret, n_forecast_bars)
        if np.isnan(future_rv):
            continue

        regime_aware_records.append({
            "date": forecast_date,
            "forecast_rv_regime": forecast_rv_reg,
            "future_rv": future_rv,
            "regime": regime_at_entry,
        })

    if regime_aware_records:
        ra_df = pd.DataFrame(regime_aware_records)

        # Merge with unconditional forecast
        merged = result_df.merge(
            ra_df[["date", "forecast_rv_regime"]], on="date", how="inner"
        )

        if len(merged) > 0:
            mse_uncond = (merged["future_rv"] - merged["forecast_rv"]) ** 2
            mse_cond = (merged["future_rv"] - merged["forecast_rv_regime"]) ** 2

            logger.info(f"\n  Unconditional MSE: {mse_uncond.mean():.6f}")
            logger.info(f"  Regime-aware MSE:  {mse_cond.mean():.6f}")
            logger.info(f"  Improvement:       {(1 - mse_cond.mean() / mse_uncond.mean()) * 100:.2f}%")

    # ── Save ───────────────────────────────────────────────────────────
    result_path = RESULTS_DIR / "calibration_vale3.csv"
    result_df.to_csv(result_path, index=False)
    logger.info(f"\nResults saved: {result_path}")

    decile_path = RESULTS_DIR / "decile_table.csv"
    decile_table.to_csv(decile_path)
    logger.info(f"Decile table: {decile_path}")

    # ── Charts ─────────────────────────────────────────────────────────
    logger.info("\nGenerating charts...")

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Forecast vs Future scatter
    ax = axes[0, 0]
    ax.scatter(result_df["forecast_rv"], result_df["future_rv"], alpha=0.4, s=10)
    lims = [0, max(result_df["forecast_rv"].max(), result_df["future_rv"].max()) * 1.1]
    ax.plot(lims, lims, "r--", alpha=0.5)
    ax.set_xlabel("Forecast RV")
    ax.set_ylabel("Future RV")
    ax.set_title(f"Forecast vs Future RV (VALE3)\nSpearman r={rank_corr:.3f}")
    ax.text(0.05, 0.95, f"n={len(result_df)}", transform=ax.transAxes, va="top")

    # 2. Decile bar chart
    ax = axes[0, 1]
    x = np.arange(1, 11)
    width = 0.35
    ax.bar(x - width / 2, decile_table["forecast_rv_pct"], width, label="Forecast", alpha=0.7)
    ax.bar(x + width / 2, decile_table["future_rv_pct"], width, label="Future", alpha=0.7)
    ax.set_xlabel("Forecast Decile")
    ax.set_ylabel("Annualized Vol (%)")
    ax.set_title("Decile Calibration")
    ax.legend()
    ax.set_xticks(x)

    # 3. Time series
    ax = axes[0, 2]
    ax.plot(result_df["date"], result_df["forecast_rv"] * 100, label="Forecast RV", alpha=0.7, linewidth=0.8)
    ax.plot(result_df["date"], result_df["future_rv"] * 100, label="Future RV", alpha=0.7, linewidth=0.8)
    ax.set_xlabel("Date")
    ax.set_ylabel("Annualized Vol (%)")
    ax.set_title("Forecast vs Future RV Time Series")
    ax.legend()

    # 4. Regime distribution
    ax = axes[1, 0]
    regime_counts_pct = result_df["regime"].value_counts(normalize=True).sort_index() * 100
    colors = ["green", "yellowgreen", "orange", "red"]
    ax.bar(regime_counts_pct.index, regime_counts_pct.values, color=colors[:len(regime_counts_pct)])
    ax.set_xlabel("Regime")
    ax.set_ylabel("% of Forecasts")
    ax.set_title("Regime Distribution at Forecast Date")
    labels = [regime_names.get(i, str(i)) for i in regime_counts_pct.index]
    ax.set_xticks(regime_counts_pct.index)
    ax.set_xticklabels(labels)

    # 5. Error distribution
    ax = axes[1, 1]
    ax.hist(result_df["error"] * 100, bins=30, alpha=0.6, edgecolor="black")
    ax.axvline(0, color="r", linestyle="--")
    ax.set_xlabel("Forecast Error (%)")
    ax.set_ylabel("Frequency")
    ax.set_title("Forecast Error Distribution")

    # 6. Regime-aware vs unconditional
    ax = axes[1, 2]
    if len(merged) > 0:
        ax.scatter(merged["forecast_rv"], merged["future_rv"], alpha=0.3, s=10, label="Unconditional", c="blue")
        ax.scatter(merged["forecast_rv_regime"], merged["future_rv"], alpha=0.3, s=10, label="Regime-aware", c="red")
        lims = [0, max(merged["forecast_rv"].max(), merged["forecast_rv_regime"].max(), merged["future_rv"].max()) * 1.1]
        ax.plot(lims, lims, "k--", alpha=0.3)
        ax.legend()
        ax.set_xlabel("Forecast RV")
        ax.set_ylabel("Future RV")
        ax.set_title("Regime-Aware vs Unconditional")
    else:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax.transAxes)

    plt.tight_layout()
    chart_path = CHARTS_DIR / "calibration_charts.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Chart saved: {chart_path}")

    # ── Report ─────────────────────────────────────────────────────────
    report_path = RESULTS_DIR / "report.md"
    lines = [
        "# VALE3 Volatility Forecast Calibration",
        "",
        f"**Period:** {start_date.date()} to {end_date.date()}",
        f"**Data:** {n_total:,} M15 bars",
        f"**Forecasts:** {n_forecasts} ({ROLL_FREQ_DAYS}d cadence, {HORIZON_DAYS}d horizon)",
        f"**Model:** EGARCH(1,1,1) on {MIN_TRAIN_YEARS}y rolling window",
        "",
        "## 1. Decile Calibration",
        "",
        "| Decil | N | Forecast RV | Future RV | Erro | |Erro| |",
        "|-------|---|-------------|-----------|------|--------|",
    ]
    for dec in range(1, 11):
        row = decile_table.loc[dec]
        lines.append(
            f"| {dec} | {int(row['n'])} | "
            f"{row['forecast_rv_pct']:.2f}% | "
            f"{row['future_rv_pct']:.2f}% | "
            f"{row['error']*100:+.2f}% | "
            f"{row['abs_error']*100:.2f}% |"
        )
    lines.extend([
        "",
        f"**Spearman correlation (decile means):** {corr:.4f} (p={pval:.4f})",
        f"**Spearman rank correlation (all obs):** {rank_corr:.4f} (p={rank_pval:.4f})",
        "",
        "## 2. Regime-Conditional Analysis",
        "",
        "| Regime | N | Forecast | Future | Error | |Err| |",
        "|--------|---|----------|--------|-------|-------|",
    ])
    for r in sorted(regime_table.index):
        row = regime_table.loc[r]
        name = regime_names.get(r, str(r))
        lines.append(
            f"| {name} | {int(row['n'])} | "
            f"{row['forecast_rv']*100:.2f}% | "
            f"{row['future_rv']*100:.2f}% | "
            f"{row['error']*100:+.2f}% | "
            f"{row['abs_error']*100:.2f}% |"
        )

    if len(merged) > 0:
        lines.extend([
            "",
            "## 3. Regime-Aware EGARCH",
            "",
            f"| Model | MSE |",
            f"|-------|-----|",
            f"| Unconditional EGARCH | {mse_uncond.mean():.6f} |",
            f"| Regime-filtered EGARCH | {mse_cond.mean():.6f} |",
            f"| Improvement | {(1 - mse_cond.mean() / mse_uncond.mean()) * 100:.2f}% |",
        ])

    lines.extend([
        "",
        "## 4. Conclusions",
        "",
        f"- **{int(n_forecasts)}** rolling forecasts generated",
    ])

    if corr > 0.5 and pval < 0.05:
        lines.append("- Monotonicity: **CONFIRMED** (higher forecast → higher future RV)")
    else:
        lines.append("- Monotonicity: **NOT CONFIRMED** (forecast vs future correlation too weak)")

    if rank_corr > 0.3 and rank_pval < 0.05:
        lines.append("- Rank correlation: **positive** (model has information content)")
    else:
        lines.append("- Rank correlation: **weak** (model may need improvement)")

    if len(merged) > 0 and mse_cond.mean() < mse_uncond.mean():
        lines.append("- Regime-aware EGARCH: **improves** over unconditional")
    elif len(merged) > 0:
        lines.append("- Regime-aware EGARCH: **does NOT improve** over unconditional")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"Report: {report_path}")

    logger.info("\n" + "=" * 70)
    logger.info("STUDY COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    run_study()
