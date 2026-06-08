"""
Forecast vs IV Relative Value Model
=====================================
Compares EGARCH volatility forecasts against real B3 implied volatilities
from options to identify relative value opportunities.

Usage:
    python studies/forecast_vs_iv/study.py
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

from models.egarch import EGARCHModel
from models.hmm_regime import HMMRegimeModel
from models.iv_collector import IVCollector
from models.vrp_model import VRPRelativeValueModel, VRPRecord

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("forecast_vs_iv")

STUDY_DIR = Path(__file__).parent.resolve()
RESULTS_DIR = STUDY_DIR / "results"
CHARTS_DIR = STUDY_DIR / "charts"
DATA_DIR = Path(r"C:\Users\mathe\Documents\GitHub\mt5\dataset\export_mt5\intraday\avista\M15")

SYMBOLS = ["PETR4", "VALE3", "ITUB4"]
OOS_FRAC = 0.2
N_BARS_PER_YEAR = 252 * 26


def set_style() -> None:
    sns.set_theme(style="darkgrid", palette="viridis")
    plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 150, "font.size": 10})


def load_m15(symbol: str) -> pd.DataFrame:
    path = DATA_DIR / f"{symbol}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"M15 data not found: {path}")
    df = pd.read_parquet(path)
    df = df.reset_index()
    df = df.rename(columns={
        "time": "timestamp",
        "Open": "open_price",
        "High": "high_price",
        "Low": "low_price",
        "Close": "close_price",
        "Tick_volume": "volume",
    })
    df["symbol"] = symbol
    df["volume"] = df["volume"].astype("int64")
    df["timestamp"] = df["timestamp"].dt.tz_localize("America/Sao_Paulo")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df[df["volume"] > 0].reset_index(drop=True)
    return df


def run_study() -> None:
    logger.info("=" * 70)
    logger.info("FORECAST vs IV RELATIVE VALUE MODEL")
    logger.info("=" * 70)

    all_records: list[VRPRecord] = []
    evaluation_results: dict[str, Any] = {}

    for stock in SYMBOLS:
        logger.info(f"\n{'=' * 70}")
        logger.info(f"  {stock}")
        logger.info(f"{'=' * 70}")

        logger.info("Loading M15 data...")
        df = load_m15(stock)
        logger.info(f"  {len(df):,} bars ({df['timestamp'].min().date()} to {df['timestamp'].max().date()})")

        n = len(df)
        split_idx = int(n * (1 - OOS_FRAC))
        df_is = df.iloc[:split_idx].copy()
        df_oos = df.iloc[split_idx:].copy()
        logger.info(f"  IS: {len(df_is):,} bars | OOS: {len(df_oos):,} bars")

        df_is["log_return"] = np.log(df_is["close_price"] / df_is["close_price"].shift(1))
        returns_is = df_is["log_return"].dropna()
        logger.info(f"  Fitting EGARCH(1,1,1) on IS data...")
        egarch = EGARCHModel(p=1, o=1, q=1)
        egarch.fit(returns_is)
        params = egarch.get_params()
        logger.info(f"    omega={params.get('omega', 0):.6f} alpha={params.get('alpha', 0):.6f} "
                     f"gamma={params.get('gamma', 0):.6f} beta={params.get('beta', 0):.6f}")

        logger.info("Fitting HMM regime model on IS data...")
        hmm_model = HMMRegimeModel(n_regimes=4)
        hmm_model.fit(returns_is)
        logger.info(f"    Regime labels: {hmm_model.labels}")

        df_all = df.copy()
        df_all["log_return"] = np.log(df_all["close_price"] / df_all["close_price"].shift(1))
        hmm_regimes = hmm_model.predict(df_all["log_return"].dropna())
        full_regimes = np.full(len(df_all), -1)
        aligned = df_all["log_return"].dropna()
        valid_idx = aligned.index[aligned.notna()].values
        min_len = min(len(valid_idx), len(hmm_regimes))
        for i in range(min_len):
            full_regimes[valid_idx[i]] = hmm_regimes[i]

        logger.info("Connecting to MT5 for option IV data...")
        iv_collector = IVCollector()
        if not iv_collector.connect():
            logger.warning("  MT5 connection failed, skipping stock")
            continue

        try:
            oos_start = df_oos["timestamp"].min()
            oos_end = df_oos["timestamp"].max()
            logger.info(f"  Extracting IV timeseries for OOS period ({oos_start.date()} to {oos_end.date()})...")
            iv_df = iv_collector.get_iv_timeseries(stock, oos_start, oos_end)
            if iv_df.empty:
                logger.warning("  No IV data available, skipping stock")
                continue
            logger.info(f"  Got {len(iv_df)} IV observations")

            iv_path = RESULTS_DIR / f"{stock}_iv_timeseries.csv"
            iv_df.to_csv(iv_path, index=False)
            logger.info(f"  Saved IV data to {iv_path}")

            logger.info("Running VRP backtest...")
            vrp_model = VRPRelativeValueModel(egarch, iv_collector)
            records = vrp_model.backtest(stock, oos_start, oos_end, df_all, iv_df, full_regimes)
            logger.info(f"  Generated {len(records)} VRP records")

            if records:
                all_records.extend(records)
                records_df = pd.DataFrame([
                    {
                        "date": r.date, "stock": r.stock,
                        "forecast_rv": r.forecast_rv, "implied_iv": r.implied_iv,
                        "spread": r.spread, "future_rv": r.future_rv,
                        "dte": r.dte, "regime": r.regime,
                    }
                    for r in records
                ])
                csv_path = RESULTS_DIR / f"{stock}_vrp_records.csv"
                records_df.to_csv(csv_path, index=False)
                logger.info(f"  Saved records to {csv_path}")

        finally:
            iv_collector.disconnect()

    if not all_records:
        logger.error("No records generated across any stock. Exiting.")
        return

    logger.info(f"\n{'=' * 70}")
    logger.info("  EVALUATION")
    logger.info(f"{'=' * 70}")

    eval_result = VRPRelativeValueModel.evaluate(all_records)
    evaluation_results = eval_result

    logger.info(f"  Total observations: {eval_result['n_observations']}")
    logger.info(f"  Hit rate: {eval_result['hit_rate']:.2%}")
    logger.info(f"  Sharpe ratio: {eval_result['sharpe_ratio']:.3f}")
    logger.info(f"  Mean spread: {eval_result['mean_spread']:.4f}")

    cm = eval_result["confusion_matrix"]
    logger.info(f"  Long vol signals: {cm['long_vol_signals']} (hit rate: {cm['long_vol_hit_rate']:.2%})")
    logger.info(f"  Short vol signals: {cm['short_vol_signals']} (hit rate: {cm['short_vol_hit_rate']:.2%})")
    logger.info(f"  No signal: {cm['no_signal']}")

    cal = eval_result["calibration"]
    if not cal.empty:
        logger.info(f"\n  Calibration by spread bucket:")
        for _, row in cal.iterrows():
            logger.info(f"    {row['spread_bucket']:20s}  n={int(row['n']):4d}  "
                         f"mean_spread={row['mean_spread']:+.4f}  "
                         f"mean_future_rv={row['mean_future_rv']:.4f}")

    logger.info("\nGenerating charts...")
    set_style()
    _generate_charts(all_records, eval_result)

    logger.info("\nGenerating report...")
    report = _generate_report(all_records, evaluation_results)
    report_path = STUDY_DIR / "report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"Report saved to {report_path}")
    logger.info(f"\n{'=' * 70}")
    logger.info("  STUDY COMPLETE")
    logger.info(f"{'=' * 70}")


def _generate_charts(records: list[VRPRecord], eval_result: dict[str, Any]) -> None:
    df = pd.DataFrame([
        {"date": r.date, "stock": r.stock, "forecast_rv": r.forecast_rv,
         "implied_iv": r.implied_iv, "spread": r.spread, "future_rv": r.future_rv}
        for r in records
    ])
    if df.empty:
        return

    min_date = df["date"].min()
    max_date = df["date"].max()

    fig, ax = plt.subplots(figsize=(16, 6))
    for stock in df["stock"].unique():
        sub = df[df["stock"] == stock].sort_values("date")
        if len(sub) < 3:
            continue
        ax.plot(sub["date"], sub["forecast_rv"], label=f"{stock} Forecast RV", linewidth=0.7, alpha=0.8)
        ax.plot(sub["date"], sub["implied_iv"], label=f"{stock} IV", linewidth=0.7, alpha=0.8, linestyle="--")
    ax.set_ylabel("Annualized Volatility")
    ax.set_title("Forecast RV vs Implied Volatility")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "rv_vs_iv_timeseries.png", bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Chart saved: {CHARTS_DIR / 'rv_vs_iv_timeseries.png'}")

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = {"PETR4": "#e74c3c", "VALE3": "#3498db", "ITUB4": "#2ecc71"}
    for stock in df["stock"].unique():
        sub = df[df["stock"] == stock]
        ax.scatter(sub["spread"], sub["future_rv"], c=colors.get(stock, "#333"),
                   label=stock, s=20, alpha=0.6)
    ax.axhline(0, color="gray", linestyle=":", alpha=0.4)
    ax.axvline(0, color="gray", linestyle=":", alpha=0.4)
    ax.set_xlabel("Spread (Forecast RV - IV)")
    ax.set_ylabel("Future Realized Vol")
    ax.set_title("Spread vs Future Realized Volatility")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "spread_vs_future_rv.png", bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Chart saved: {CHARTS_DIR / 'spread_vs_future_rv.png'}")

    cal = eval_result.get("calibration")
    if cal is not None and not cal.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        buckets = cal["spread_bucket"].astype(str)
        x = np.arange(len(buckets))
        w = 0.35
        ax.bar(x - w / 2, cal["mean_spread"].values, w, label="Mean Spread", color="#3498db", alpha=0.7)
        ax.bar(x + w / 2, cal["mean_future_rv"].values, w, label="Mean Future RV", color="#e74c3c", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(buckets, rotation=30, ha="right")
        ax.set_ylabel("Annualized Volatility")
        ax.set_title("Calibration: Forecast Buckets vs Actual RV")
        ax.legend(fontsize=9)
        for i, (_, row) in enumerate(cal.iterrows()):
            ax.annotate(f"n={int(row['n'])}", (x[i], max(row["mean_spread"], row["mean_future_rv"])),
                        ha="center", va="bottom", fontsize=7)
        fig.tight_layout()
        fig.savefig(CHARTS_DIR / "calibration_plot.png", bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  Chart saved: {CHARTS_DIR / 'calibration_plot.png'}")

    df["signal"] = df["spread"].apply(
        lambda s: 1 if s > 0.05 else (-1 if s < -0.05 else 0)
    )
    df["pnl"] = np.where(
        df["signal"] == 1,
        df["future_rv"] - df["implied_iv"],
        np.where(df["signal"] == -1, df["implied_iv"] - df["future_rv"], 0.0),
    )
    df_sorted = df.sort_values("date").reset_index(drop=True)
    df_sorted["cum_pnl"] = df_sorted["pnl"].cumsum()

    fig, ax = plt.subplots(figsize=(16, 6))
    colors_sig = {1: "#2ecc71", -1: "#e74c3c", 0: "#95a5a6"}
    for sig in [1, -1, 0]:
        mask = df_sorted["signal"] == sig
        label = {1: "Long Vol", -1: "Short Vol", 0: "No Trade"}[sig]
        ax.scatter(df_sorted["date"][mask], df_sorted["cum_pnl"][mask],
                   c=colors_sig[sig], label=label, s=8, alpha=0.5)
    ax.plot(df_sorted["date"], df_sorted["cum_pnl"], color="#333", linewidth=0.8, alpha=0.6)
    ax.axhline(0, color="gray", linestyle=":", alpha=0.4)
    ax.set_ylabel("Cumulative PnL (vol points)")
    ax.set_title("Signal Performance: Long/Short Vol Based on Spread")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "signal_performance.png", bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Chart saved: {CHARTS_DIR / 'signal_performance.png'}")


def _generate_report(records: list[VRPRecord], eval_result: dict[str, Any]) -> str:
    df = pd.DataFrame([
        {"date": r.date, "stock": r.stock, "forecast_rv": r.forecast_rv,
         "implied_iv": r.implied_iv, "spread": r.spread, "future_rv": r.future_rv,
         "dte": r.dte, "regime": r.regime}
        for r in records
    ])

    lines: list[str] = []

    lines.append("# Forecast vs IV Relative Value Model Report\n")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    lines.append(f"**Universe:** {', '.join(SYMBOLS)}\n")
    lines.append(f"**Data:** B3 M15 intraday + MT5 option IV\n")
    lines.append(f"**Model:** EGARCH(1,1,1) on IS (80%), forecast on OOS (20%)\n")
    lines.append("---\n")

    lines.append("## 1. Historical Table\n")
    lines.append("| Date | Stock | Forecast RV | IV | Spread | Future RV | DTE | Regime |\n")
    lines.append("|------|-------|-------------|----|--------|-----------|-----|--------|\n")
    for r in records[:50]:
        d = r.date.strftime("%Y-%m-%d") if hasattr(r.date, "strftime") else str(r.date)
        lines.append(
            f"| {d} | {r.stock} | {r.forecast_rv:.2%} | {r.implied_iv:.2%} | "
            f"{r.spread:+.2%} | {r.future_rv:.2%} | {r.dte} | {r.regime} |\n"
        )
    if len(records) > 50:
        lines.append(f"| ... | ... | ... | ... | ... | ... | ... | ... |\n")
        lines.append(f"*Showing first 50 of {len(records)} records*\n")
    lines.append("\n")

    lines.append("## 2. Group Analysis\n")
    if not df.empty:
        df["group"] = pd.cut(
            df["spread"],
            bins=[-np.inf, -0.02, 0.02, np.inf],
            labels=["Forecast < IV", "Forecast ≈ IV", "Forecast > IV"],
        )
        group_stats = df.groupby("group", observed=True).agg(
            n=("future_rv", "count"),
            mean_future_rv=("future_rv", "mean"),
            mean_spread=("spread", "mean"),
        ).reset_index()
        for _, row in group_stats.iterrows():
            lines.append(f"- **{row['group']}** (n={int(row['n'])}): "
                         f"spread={row['mean_spread']:+.2%}, "
                         f"future RV={row['mean_future_rv']:.2%}\n")
    lines.append("\n")

    lines.append("## 3. Hit Rate Analysis\n")
    hit_rate = eval_result.get("hit_rate", 0)
    lines.append(f"- **Overall hit rate:** {hit_rate:.2%}\n")
    lines.append(f"- How often does spread sign correctly predict RV direction?\n")
    lines.append(f"- Hit rate = P(sign(spread) == sign(future_RV - IV))\n\n")

    lines.append("## 4. Confusion Matrix\n")
    cm = eval_result.get("confusion_matrix", {})
    lines.append(f"- **Long vol signal (Forecast >> IV):** {cm.get('long_vol_signals', 0)} signals, "
                 f"{cm.get('long_vol_correct', 0)} correct "
                 f"({cm.get('long_vol_hit_rate', 0):.2%})\n")
    lines.append(f"- **Short vol signal (Forecast << IV):** {cm.get('short_vol_signals', 0)} signals, "
                 f"{cm.get('short_vol_correct', 0)} correct "
                 f"({cm.get('short_vol_hit_rate', 0):.2%})\n")
    lines.append(f"- **No signal:** {cm.get('no_signal', 0)} observations\n\n")

    lines.append("## 5. Strategy Sharpe\n")
    sharpe = eval_result.get("sharpe_ratio", 0)
    lines.append(f"- **Trading the spread signal:** Sharpe = {sharpe:.3f}\n")
    lines.append(f"- PnL = signal × (future_RV - IV) for each observation\n")
    lines.append(f"- Annualized using sqrt(252 / avg DTE)\n\n")

    lines.append("## 6. Calibration Table\n")
    cal = eval_result.get("calibration")
    if cal is not None and not cal.empty:
        lines.append("| Bucket | n | Mean Spread | Mean Future RV |\n")
        lines.append("|--------|---|---|---|\n")
        for _, row in cal.iterrows():
            lines.append(f"| {row['spread_bucket']} | {int(row['n'])} | "
                         f"{row['mean_spread']:+.4f} | {row['mean_future_rv']:.4f} |\n")
    lines.append("\n")

    lines.append("## 7. Summary Statistics\n")
    if not df.empty:
        lines.append(f"- **Total observations:** {len(df)}\n")
        lines.append(f"- **Mean forecast RV:** {df['forecast_rv'].mean():.2%}\n")
        lines.append(f"- **Mean implied IV:** {df['implied_iv'].mean():.2%}\n")
        lines.append(f"- **Mean spread:** {df['spread'].mean():+.2%}\n")
        lines.append(f"- **Mean future RV:** {df['future_rv'].mean():.2%}\n")
        lines.append(f"- **Spread range:** [{df['spread'].min():+.2%}, {df['spread'].max():+.2%}]\n")
        lines.append(f"- **DTE range:** [{df['dte'].min()}, {df['dte'].max()}]\n\n")

    lines.append("## 8. Regime Distribution\n")
    if not df.empty and "regime" in df.columns:
        regime_counts = df["regime"].value_counts().sort_index()
        for reg, cnt in regime_counts.items():
            lines.append(f"- **Regime {reg}:** {cnt} observations ({cnt/len(df)*100:.1f}%)\n")
    lines.append("\n---\n")
    lines.append("*Report generated automatically by study.py*\n")

    return "".join(lines)


if __name__ == "__main__":
    run_study()
