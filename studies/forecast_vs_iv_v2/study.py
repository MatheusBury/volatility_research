"""
Forecast RV vs ATM IV — VALE3 Daily
=====================================
Central question: Is my RV forecast better than the market's IV?

Builds a daily dataset (2021-2026) with:
  date, close, return,
  rv_5d, rv_10d, rv_20d, rv_30d (realized),
  forecast_rv_20d (EGARCH rolling),
  regime (HMM),
  iv_atm (from MT5 options),
  rv_futura_20d (future realized)

Usage:
    python studies/forecast_vs_iv_v2/study.py
"""
from __future__ import annotations

import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

from models.egarch import EGARCHModel
from models.hmm_regime import HMMRegimeModel
from models.iv_collector import IVCollector

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("forecast_vs_iv_v2")

STUDY_DIR = Path(__file__).parent.resolve()
RESULTS_DIR = STUDY_DIR / "results"
CHARTS_DIR = STUDY_DIR / "charts"
D1_PATH = r"C:\Users\mathe\Documents\GitHub\mt5\dataset\export_mt5\avista\D1\VALE3_D1_20210329_20260329.parquet"

STOCK = "VALE3"
N_BARS_PER_YEAR = 252
RETURN_SCALE = 100.0

TRAIN_WINDOW_YEARS = 2
HORIZON_DAYS = 20


def load_daily(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df.reset_index(drop=True)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    df["return"] = np.log(df["close"] / df["close"].shift(1))
    df = df.dropna(subset=["return"]).reset_index(drop=True)
    df["date"] = df["time"].dt.date
    return df


def compute_rolling_rvs(df: pd.DataFrame) -> pd.DataFrame:
    windows = {"rv_5d": 5, "rv_10d": 10, "rv_20d": 20, "rv_30d": 30}
    ann = np.sqrt(N_BARS_PER_YEAR)
    for col, w in windows.items():
        df[col] = df["return"].rolling(window=w).std() * ann
    return df


def compute_future_rv(df: pd.DataFrame, horizon: int = 20) -> pd.DataFrame:
    ann = np.sqrt(N_BARS_PER_YEAR)
    col = f"rv_futura_{horizon}d"
    df[col] = (
        df["return"]
        .rolling(window=horizon)
        .std()
        .shift(-horizon + 1) * ann
    )
    return df


def build_dataset() -> pd.DataFrame:
    logger.info("=" * 60)
    logger.info("BUILDING DAILY DATASET — VALE3")
    logger.info("=" * 60)

    logger.info("Loading D1 data...")
    df = load_daily(D1_PATH)
    logger.info(f"  {len(df)} bars ({df['date'].iloc[0]} to {df['date'].iloc[-1]})")

    logger.info("Computing rolling realized volatilities...")
    df = compute_rolling_rvs(df)
    logger.info("  rv_5d, rv_10d, rv_20d, rv_30d")

    logger.info("Computing future realized volatilities...")
    df = compute_future_rv(df, 20)
    logger.info("  rv_futura_20d (20d forward)")

    logger.info("Running HMM(4) on daily returns...")
    hmm = HMMRegimeModel(n_regimes=4)
    hmm.fit(df["return"])
    predicted = hmm.predict(df["return"])
    pad_len = len(df) - len(predicted)
    df["regime"] = np.concatenate([np.full(pad_len, 0, dtype=int), predicted])
    logger.info(f"  Regimes: {dict(df['regime'].value_counts().sort_index())}")

    logger.info("Running rolling EGARCH(1,1,1) forecast (20d horizon)...")
    n_train = int(TRAIN_WINDOW_YEARS * N_BARS_PER_YEAR)
    forecasts: list[float | None] = [None] * len(df)

    n_ok = 0
    n_reject = 0
    for i in range(n_train, len(df)):
        train_ret = df["return"].iloc[i - n_train: i]
        try:
            model = EGARCHModel(p=1, o=1, q=1, scale=RETURN_SCALE, n_bars_per_year=252)
            model.fit(train_ret)
            vols = model.forecast(HORIZON_DAYS)
            fcast = float(np.mean(vols))
            # Reject unreasonable forecasts (convergence failure)
            if fcast < 0.01 or fcast > 2.0:
                n_reject += 1
                forecasts[i] = None
            else:
                forecasts[i] = fcast
        except Exception:
            forecasts[i] = None
        n_ok += 1
        if n_ok % 100 == 0:
            logger.info(f"  {n_ok} forecasts generated... (rejected: {n_reject})")

    df["forecast_rv_20d"] = forecasts
    n_valid = df["forecast_rv_20d"].notna().sum()
    logger.info(f"  {n_valid} valid forecasts")

    df = df.dropna(subset=["forecast_rv_20d", "rv_futura_20d"]).reset_index(drop=True)
    logger.info(f"  Final rows (after dropna): {len(df)}")

    return df


def collect_iv(daily_df: pd.DataFrame) -> pd.DataFrame:
    logger.info("\nCollecting ATM IV from MT5 options...")
    collector = IVCollector()
    if not collector.connect():
        logger.warning("  MT5 connection failed — no IV data")
        return daily_df

    try:
        start = pd.Timestamp(daily_df["date"].iloc[0])
        end = pd.Timestamp(daily_df["date"].iloc[-1])
        logger.info(f"  Requesting IV from {start.date()} to {end.date()}...")
        iv_df = collector.get_iv_timeseries(STOCK, start, end)
        if iv_df.empty:
            logger.warning("  No IV data returned from MT5")
            return daily_df

        logger.info(f"  Got {len(iv_df)} IV observations")
        iv_df["date"] = pd.to_datetime(iv_df["date"]).dt.date
        iv_map = iv_df.set_index("date")["iv"].to_dict()

        daily_df["iv_atm"] = daily_df["date"].map(iv_map)
        n_iv = daily_df["iv_atm"].notna().sum()
        logger.info(f"  Merged IV for {n_iv} dates")
    finally:
        collector.disconnect()

    return daily_df


def run_analysis(df: pd.DataFrame) -> None:
    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS")
    logger.info("=" * 60)

    if df.empty:
        logger.error("Empty dataset — nothing to analyze")
        return

    has_iv = "iv_atm" in df.columns and df["iv_atm"].notna().sum() > 0
    n_with_iv = df["iv_atm"].notna().sum() if has_iv else 0
    logger.info(f"  Total rows: {len(df)}")
    if has_iv:
        logger.info(f"  Rows with IV: {n_with_iv}")

    # ── 1. Forecast vs Future RV (decile) ──────────────────────────────
    logger.info("\n" + "-" * 50)
    logger.info("Decile: Forecast RV vs Future RV")
    logger.info("-" * 50)

    df["decile"] = pd.qcut(df["forecast_rv_20d"], 10, labels=range(1, 11))
    decile = df.groupby("decile", observed=True).agg(
        n=("rv_futura_20d", "count"),
        forecast=("forecast_rv_20d", "mean"),
        future=("rv_futura_20d", "mean"),
    ).round(4)
    decile["forecast_pct"] = decile["forecast"] * 100
    decile["future_pct"] = decile["future"] * 100

    logger.info(f"\n{'Decil':6s} | {'N':4s} | {'Forecast':9s} | {'Future':9s} | {'Dif':8s}")
    logger.info("-" * 45)
    for d in range(1, 11):
        r = decile.loc[d]
        logger.info(f"  {d:3d}   | {int(r['n']):3d} | {r['forecast_pct']:6.2f}% | {r['future_pct']:6.2f}% | {r['future']-r['forecast']:+7.2%}")

    dcorr, dpval = spearmanr(
        decile["forecast"].values, decile["future"].values
    )
    rcorr, rpval = spearmanr(df["forecast_rv_20d"], df["rv_futura_20d"])
    logger.info(f"\n  Spearman decile means: {dcorr:.4f} (p={dpval:.4f})")
    logger.info(f"  Spearman all obs:      {rcorr:.4f} (p={rpval:.4f})")

    # ── 2. Regime analysis ────────────────────────────────────────────
    logger.info("\n" + "-" * 50)
    logger.info("Regime: Forecast vs Future RV")
    logger.info("-" * 50)

    regime_names = {0: "Low", 1: "Med", 2: "High", 3: "Extreme"}
    regime_table = df.groupby("regime").agg(
        n=("rv_futura_20d", "count"),
        forecast=("forecast_rv_20d", "mean"),
        future=("rv_futura_20d", "mean"),
    ).round(4)

    logger.info(f"\n{'Regime':10s} | {'N':4s} | {'Forecast':9s} | {'Future':9s} | {'Dif':8s}")
    logger.info("-" * 50)
    for r in sorted(regime_table.index):
        row = regime_table.loc[r]
        name = regime_names.get(r, str(r))
        logger.info(f"  {name:8s} | {int(row['n']):3d} | {row['forecast']*100:6.2f}% | {row['future']*100:6.2f}% | {row['future']-row['forecast']:+7.2%}")

    # ── 3. Forecast vs IV (when available) ────────────────────────────
    if has_iv:
        logger.info("\n" + "-" * 50)
        logger.info("Forecast vs ATM IV")
        logger.info("-" * 50)

        iv_df = df.dropna(subset=["iv_atm"]).copy()
        logger.info(f"  Observations with IV: {len(iv_df)}")

        if len(iv_df) >= 3:
            iv_df["iv_error"] = iv_df["rv_futura_20d"] - iv_df["iv_atm"]
            iv_df["fcast_error"] = iv_df["rv_futura_20d"] - iv_df["forecast_rv_20d"]

            fcast_mae = iv_df["fcast_error"].abs().mean()
            iv_mae = iv_df["iv_error"].abs().mean()
            logger.info(f"  Forecast MAE: {fcast_mae:.4f}")
            logger.info(f"  IV MAE:       {iv_mae:.4f}")
            logger.info(f"  Forecast is {'BETTER' if fcast_mae < iv_mae else 'WORSE'} than IV")

            fcast_bias = iv_df["fcast_error"].mean()
            iv_bias = iv_df["iv_error"].mean()
            logger.info(f"  Forecast bias: {fcast_bias:+.4f}")
            logger.info(f"  IV bias:       {iv_bias:+.4f}")

            # Spread = Forecast - IV
            iv_df["spread"] = iv_df["forecast_rv_20d"] - iv_df["iv_atm"]

            # Spread quintile
            iv_df["spread_decile"] = pd.qcut(iv_df["spread"], 5, labels=range(1, 6), duplicates="drop")
            spread_table = iv_df.groupby("spread_decile", observed=True).agg(
                n=("rv_futura_20d", "count"),
                spread=("spread", "mean"),
                future_rv=("rv_futura_20d", "mean"),
                iv=("iv_atm", "mean"),
                forecast=("forecast_rv_20d", "mean"),
            ).round(4)

            logger.info(f"\n{'Spread':8s} | {'N':3s} | {'Forecast':9s} | {'IV':9s} | {'Future':9s}")
            logger.info("-" * 45)
            for d in range(1, 6):
                if d in spread_table.index:
                    r = spread_table.loc[d]
                    logger.info(f"  {d:5d}   | {int(r['n']):2d} | {r['forecast']*100:6.2f}% | {r['iv']*100:6.2f}% | {r['future_rv']*100:6.2f}%")
        else:
            logger.info(f"  Too few IV observations ({len(iv_df)}) for meaningful comparison")

        # Save IV subset
        iv_path = RESULTS_DIR / "vale3_with_iv.csv"
        iv_df.to_csv(iv_path, index=False)
        logger.info(f"  IV subset saved: {iv_path}")

    # ── 4. Summary stats ─────────────────────────────────────────────
    logger.info("\n" + "-" * 50)
    logger.info("Summary Statistics")
    logger.info("-" * 50)

    logger.info(f"  Mean forecast RV: {df['forecast_rv_20d'].mean():.2%}")
    logger.info(f"  Mean future RV:   {df['rv_futura_20d'].mean():.2%}")
    logger.info(f"  Mean error:       {(df['rv_futura_20d'] - df['forecast_rv_20d']).mean():+.2%}")
    logger.info(f"  MAE:              {(df['rv_futura_20d'] - df['forecast_rv_20d']).abs().mean():.2%}")
    logger.info(f"  RMSE:             {((df['rv_futura_20d'] - df['forecast_rv_20d'])**2).mean()**0.5:.2%}")

    # ── Charts ───────────────────────────────────────────────────────
    logger.info("\nGenerating charts...")
    sns.set_style("whitegrid")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Forecast vs Future scatter
    ax = axes[0, 0]
    ax.scatter(df["forecast_rv_20d"], df["rv_futura_20d"], alpha=0.3, s=5)
    lims = [
        min(df["forecast_rv_20d"].min(), df["rv_futura_20d"].min()) * 0.9,
        max(df["forecast_rv_20d"].max(), df["rv_futura_20d"].max()) * 1.1,
    ]
    ax.plot(lims, lims, "r--", alpha=0.4)
    ax.set_xlabel("Forecast RV (20d)")
    ax.set_ylabel("Future RV (20d)")
    ax.set_title(f"Forecast vs Future (n={len(df)})\nSpearman r={rcorr:.3f}")

    # 2. Decile bar chart
    ax = axes[0, 1]
    x = np.arange(1, 11)
    w = 0.35
    ax.bar(x - w / 2, decile["forecast_pct"], w, label="Forecast", alpha=0.7)
    ax.bar(x + w / 2, decile["future_pct"], w, label="Future", alpha=0.7)
    ax.set_xlabel("Forecast Decile")
    ax.set_ylabel("Annualized Vol (%)")
    ax.set_title("Decile Calibration")
    ax.legend()
    ax.set_xticks(x)

    # 3. Time series
    ax = axes[0, 2]
    ax.plot(df["date"], df["forecast_rv_20d"] * 100, label="Forecast", alpha=0.7, linewidth=0.6)
    ax.plot(df["date"], df["rv_futura_20d"] * 100, label="Future", alpha=0.7, linewidth=0.6)
    if has_iv:
        iv_plot = df.dropna(subset=["iv_atm"])
        if len(iv_plot) > 0:
            ax.scatter(iv_plot["date"], iv_plot["iv_atm"] * 100, label="ATM IV", s=8, c="red", alpha=0.6)
    ax.set_xlabel("Date")
    ax.set_ylabel("Annualized Vol (%)")
    ax.set_title("Forecast vs Future vs IV — VALE3")
    ax.legend(fontsize=8)

    # 4. Regime distribution
    ax = axes[1, 0]
    regime_pct = df["regime"].value_counts(normalize=True).sort_index() * 100
    colors = ["green", "yellowgreen", "orange", "red"]
    ax.bar(regime_pct.index, regime_pct.values, color=colors[:len(regime_pct)])
    ax.set_xlabel("Regime")
    ax.set_ylabel("% of Days")
    ax.set_title("Regime Distribution")
    labels = [regime_names.get(i, str(i)) for i in regime_pct.index]
    ax.set_xticks(regime_pct.index)
    ax.set_xticklabels(labels)

    # 5. Error distribution
    ax = axes[1, 1]
    errors = (df["rv_futura_20d"] - df["forecast_rv_20d"]) * 100
    ax.hist(errors, bins=40, alpha=0.6, edgecolor="black")
    ax.axvline(0, color="r", linestyle="--")
    ax.axvline(errors.mean(), color="g", linestyle=":", label=f"Mean={errors.mean():.2f}%")
    ax.set_xlabel("Forecast Error (%)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Forecast Error\nMAE={errors.abs().mean():.2f}%")
    ax.legend(fontsize=8)

    # 6. Regime-conditional forecast error
    ax = axes[1, 2]
    regime_errors = df.groupby("regime")["rv_futura_20d"].apply(
        lambda x: (x - df.loc[x.index, "forecast_rv_20d"]).mean() * 100
    )
    names = [regime_names.get(i, str(i)) for i in regime_errors.index]
    ax.bar(names, regime_errors.values, color=colors[:len(regime_errors)])
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xlabel("Regime")
    ax.set_ylabel("Mean Error (%)")
    ax.set_title("Forecast Bias by Regime")

    plt.tight_layout()
    chart_path = CHARTS_DIR / "forecast_vs_iv_v2.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Chart saved: {chart_path}")


def generate_report(df: pd.DataFrame) -> str:
    has_iv = "iv_atm" in df.columns and df["iv_atm"].notna().sum() > 0
    n_with_iv = df["iv_atm"].notna().sum() if has_iv else 0

    dcorr, dpval = spearmanr(
        df.groupby("decile", observed=True)["forecast_rv_20d"].mean().values,
        df.groupby("decile", observed=True)["rv_futura_20d"].mean().values,
    )
    rcorr, rpval = spearmanr(df["forecast_rv_20d"], df["rv_futura_20d"])

    lines: list[str] = [
        "# VALE3: Forecast RV vs ATM IV",
        "",
        f"**Period:** {df['date'].iloc[0]} to {df['date'].iloc[-1]}",
        f"**Data:** Daily, VALE3",
        f"**Model:** EGARCH(1,1,1), {TRAIN_WINDOW_YEARS}y rolling, {HORIZON_DAYS}d horizon",
        f"**Total rows:** {len(df)}",
        f"**Rows with IV:** {n_with_iv}",
        "",
        "---",
        "",
        "## 1. Decile Calibration: Forecast RV → Future RV",
        "",
        "| Decil | N | Forecast | Futuro | Diferença |",
        "|-------|---|----------|--------|-----------|",
    ]
    for d in range(1, 11):
        row = df[df["decile"] == d]
        lines.append(
            f"| {d} | {len(row)} | "
            f"{row['forecast_rv_20d'].mean()*100:.2f}% | "
            f"{row['rv_futura_20d'].mean()*100:.2f}% | "
            f"{(row['rv_futura_20d'] - row['forecast_rv_20d']).mean()*100:+.2f}% |"
        )

    lines.extend([
        "",
        f"**Spearman (decile means):** r={dcorr:.4f} (p={dpval:.4f})",
        f"**Spearman (all obs):** r={rcorr:.4f} (p={rpval:.4f})",
        "",
        "---",
        "",
        "## 2. Regime Analysis",
        "",
        "| Regime | N | Forecast | Futuro | Diferença |",
        "|--------|---|----------|--------|-----------|",
    ])

    regime_names = {0: "Low", 1: "Med", 2: "High", 3: "Extreme"}
    for r in sorted(df["regime"].unique()):
        sub = df[df["regime"] == r]
        name = regime_names.get(r, str(r))
        lines.append(
            f"| {name} | {len(sub)} | "
            f"{sub['forecast_rv_20d'].mean()*100:.2f}% | "
            f"{sub['rv_futura_20d'].mean()*100:.2f}% | "
            f"{(sub['rv_futura_20d'] - sub['forecast_rv_20d']).mean()*100:+.2f}% |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Summary Statistics",
        "",
        f"- **Mean forecast:** {df['forecast_rv_20d'].mean()*100:.2f}%",
        f"- **Mean future RV:** {df['rv_futura_20d'].mean()*100:.2f}%",
        f"- **Mean error:** {(df['rv_futura_20d'] - df['forecast_rv_20d']).mean()*100:+.2f}%",
        f"- **MAE:** {(df['rv_futura_20d'] - df['forecast_rv_20d']).abs().mean()*100:.2f}%",
        f"- **RMSE:** {((df['rv_futura_20d'] - df['forecast_rv_20d'])**2).mean()**0.5*100:.2f}%",
        "",
    ])

    if has_iv:
        iv_df = df.dropna(subset=["iv_atm"])
        fcast_mae = (iv_df["rv_futura_20d"] - iv_df["forecast_rv_20d"]).abs().mean()
        iv_mae = (iv_df["rv_futura_20d"] - iv_df["iv_atm"]).abs().mean()
        better = "BETTER" if fcast_mae < iv_mae else "WORSE"

        lines.extend([
            "---",
            "",
            "## 4. Forecast vs ATM IV",
            "",
            f"**Observations with IV:** {len(iv_df)}",
            "",
            f"| Metric | Forecast | IV |",
            f"|--------|----------|----|",
            f"| MAE | {fcast_mae*100:.2f}% | {iv_mae*100:.2f}% |",
            f"| Bias | {(iv_df['rv_futura_20d'] - iv_df['forecast_rv_20d']).mean()*100:+.2f}% | "
            f"{(iv_df['rv_futura_20d'] - iv_df['iv_atm']).mean()*100:+.2f}% |",
            "",
            f"**Conclusion:** EGARCH forecast is **{better}** than ATM IV",
        ])

        if len(iv_df) >= 5:
            lines.extend([
                "",
                "### Spread (Forecast − IV) Quintiles",
                "",
                "| Quintil | N | Forecast | IV | Spread | Future RV |",
                "|---------|---|----------|----|--------|-----------|",
            ])
            iv_df["spread"] = iv_df["forecast_rv_20d"] - iv_df["iv_atm"]
            iv_df["spread_q"] = pd.qcut(iv_df["spread"], 5, labels=range(1, 6), duplicates="drop")
            for q in range(1, 6):
                if q in iv_df["spread_q"].values:
                    sub = iv_df[iv_df["spread_q"] == q]
                    lines.append(
                        f"| {q} | {len(sub)} | "
                        f"{sub['forecast_rv_20d'].mean()*100:.2f}% | "
                        f"{sub['iv_atm'].mean()*100:.2f}% | "
                        f"{sub['spread'].mean()*100:+.2f}% | "
                        f"{sub['rv_futura_20d'].mean()*100:.2f}% |"
                    )

    lines.extend([
        "",
        "---",
        "",
        "*Generated by forecast_vs_iv_v2/study.py*",
    ])

    return "\n".join(lines)


def run_study() -> None:
    df = build_dataset()
    df = collect_iv(df)

    # Save full dataset
    csv_path = RESULTS_DIR / "vale3_daily_dataset.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"\nDataset saved: {csv_path}")

    df["date"] = pd.to_datetime(df["date"])
    run_analysis(df)

    report = generate_report(df)
    report_path = RESULTS_DIR / "report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"Report saved: {report_path}")

    logger.info("\n" + "=" * 60)
    logger.info("STUDY COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_study()
