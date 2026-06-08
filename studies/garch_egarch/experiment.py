"""GARCH vs EGARCH Study for B3 Stocks (15-min data).

Compares GARCH(1,1) and EGARCH(1,1) across top-10 liquid B3 stocks.
Follows RESEARCH_PLAYBOOK.md: IS/OOS split, OOS forecast on last 20%.

Usage:
    python studies/garch_egarch/experiment.py
"""

from __future__ import annotations

import logging
import os
import time
import warnings
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from arch import arch_model
from arch.univariate.base import ARCHModelResult

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("garch_egarch")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STUDY_DIR = Path(__file__).parent.resolve()
RESULTS_DIR = STUDY_DIR / "results"
CHARTS_DIR = STUDY_DIR / "charts"
CONFIG_PATH = STUDY_DIR / "config.yaml"

DATA_DIR = r"C:\Users\mathe\Documents\GitHub\mt5\dataset\export_mt5\intraday\avista\M15"

RETURN_SCALE = 1000.0

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def load_config() -> dict:
    """Load YAML config."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_b3_data(symbols: List[str], symbol_map: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """Load and normalize M15 parquet data for given symbols.

    Returns a single DataFrame with columns:
        symbol, timestamp, open_price, high_price, low_price, close_price, volume
    """
    smap = symbol_map or {}
    pieces: List[pd.DataFrame] = []
    for sym in symbols:
        fname = smap.get(sym, sym)
        fpath = os.path.join(DATA_DIR, f"{fname}.parquet")
        if not os.path.isfile(fpath):
            logger.warning("File not found for symbol %s (looked for %s)", sym, fpath)
            continue
        df = pd.read_parquet(fpath)
        df = df.reset_index()
        df = df.rename(columns={
            "time": "timestamp",
            "Open": "open_price",
            "High": "high_price",
            "Low": "low_price",
            "Close": "close_price",
            "Tick_volume": "volume",
        })
        df["symbol"] = sym
        df["volume"] = df["volume"].astype("int64")
        df["timestamp"] = df["timestamp"].dt.tz_localize("America/Sao_Paulo")
        pieces.append(df)

    if not pieces:
        raise ValueError("No data loaded for any symbol.")

    result = pd.concat(pieces, ignore_index=True)
    result = result.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    return result


def compute_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add log_return column (log close_price / close_price.shift(1)) per symbol."""
    df = df.copy()
    df["log_return"] = df.groupby("symbol")["close_price"].transform(
        lambda s: np.log(s / s.shift(1))
    )
    return df


# ---------------------------------------------------------------------------
# Dataclass for fit results
# ---------------------------------------------------------------------------
@dataclass
class ModelFitResult:
    """Holds the output of a single GARCH/EGARCH fit and forecast."""

    symbol: str
    model_type: str
    converged: bool
    n_obs_is: int
    n_obs_oos: int

    omega: Optional[float] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None
    gamma: Optional[float] = None

    aic: Optional[float] = None
    bic: Optional[float] = None
    log_likelihood: Optional[float] = None

    persistence: Optional[float] = None

    forecast_variance_oos_mean: Optional[float] = None
    realized_variance_oos: Optional[float] = None
    mse_oos: Optional[float] = None
    mae_oos: Optional[float] = None

    fit_time_seconds: Optional[float] = None
    error_message: str = ""


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------
class VolatilityModel:
    """Wraps arch_model for GARCH/EGARCH with fit + OOS forecast."""

    def __init__(
        self,
        symbol: str,
        model_type: str,
        p: int = 1,
        o: int = 0,
        q: int = 1,
    ):
        self.symbol = symbol
        self.model_type = model_type
        self.p = p
        self.o = o
        self.q = q
        self._result: Optional[ARCHModelResult] = None
        self._returns_is_scaled: Optional[pd.Series] = None
        self._returns_oos_scaled: Optional[pd.Series] = None

    def fit(
        self,
        returns_is: pd.Series,
        returns_oos: pd.Series,
        disp: str = "off",
    ) -> ModelFitResult:
        """Fit the model on IS data, forecast OOS, return structured result."""
        self._returns_is_scaled = returns_is.dropna() * RETURN_SCALE
        self._returns_oos_scaled = returns_oos.dropna() * RETURN_SCALE

        result = ModelFitResult(
            symbol=self.symbol,
            model_type=self.model_type,
            converged=False,
            n_obs_is=int(self._returns_is_scaled.count()),
            n_obs_oos=int(self._returns_oos_scaled.count()),
        )

        t0 = time.perf_counter()
        try:
            vol_type = "GARCH" if self.model_type == "GARCH" else "EGARCH"
            am = arch_model(
                self._returns_is_scaled,
                mean="zero",
                vol=vol_type,
                p=self.p,
                o=self.o if self.model_type == "EGARCH" else 0,
                q=self.q,
                dist="normal",
            )
            res = am.fit(disp=disp, update_freq=0)
            self._result = res
            result.converged = True

            params = res.params
            result.omega = float(params.get("omega", np.nan))
            result.alpha = float(params.get("alpha[1]", np.nan))
            result.beta = float(params.get("beta[1]", np.nan))
            if self.model_type == "EGARCH":
                result.gamma = float(params.get("gamma[1]", np.nan))

            result.aic = float(res.aic)
            result.bic = float(res.bic)
            result.log_likelihood = float(res.loglikelihood)

            # Persistence
            if self.model_type == "GARCH":
                a = result.alpha if result.alpha is not None else 0.0
                b = result.beta if result.beta is not None else 0.0
                result.persistence = a + b
            else:
                result.persistence = result.beta if result.beta is not None else 0.0

            # ---- OOS forecast ----
            n_oos = len(self._returns_oos_scaled)
            if n_oos > 0:
                n_is = len(self._returns_is_scaled)
                fcast_method = "simulation" if self.model_type == "EGARCH" else "analytic"

                fcast = res.forecast(
                    horizon=n_oos,
                    start=n_is - 1,
                    method=fcast_method,
                    reindex=False,
                )
                fcast_vars = fcast.variance.iloc[0, :].values.astype(float)
                n_fcast = len(fcast_vars)
                realized_vars = self._returns_oos_scaled.iloc[:n_fcast].values ** 2
                min_len = min(n_fcast, len(realized_vars))

                if min_len > 0:
                    fv = fcast_vars[:min_len]
                    rv = realized_vars[:min_len]
                    result.forecast_variance_oos_mean = float(fv.mean())
                    result.realized_variance_oos = float(rv.mean())
                    result.mse_oos = float(np.mean((fv - rv) ** 2))
                    result.mae_oos = float(np.mean(np.abs(fv - rv)))

            logger.info(
                "%s %s: AIC=%.1f BIC=%.1f Pers=%.4f MSE_OOS=%.6f",
                self.symbol,
                self.model_type,
                result.aic if result.aic is not None else 0.0,
                result.bic if result.bic is not None else 0.0,
                result.persistence if result.persistence is not None else 0.0,
                result.mse_oos if result.mse_oos is not None else 0.0,
            )

        except Exception as exc:
            result.error_message = str(exc)
            logger.warning("Fit failed for %s %s: %s", self.symbol, self.model_type, exc)

        t1 = time.perf_counter()
        result.fit_time_seconds = round(t1 - t0, 3)
        return result

    @property
    def conditional_volatility(self) -> Optional[np.ndarray]:
        if self._result is None:
            return None
        return self._result.conditional_volatility / RETURN_SCALE


# ---------------------------------------------------------------------------
# Study orchestrator
# ---------------------------------------------------------------------------
@dataclass
class StudyResults:
    """Aggregated results across all symbols and models."""

    fits: List[ModelFitResult] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(f) for f in self.fits])

    def save_csv(self, path: Path) -> None:
        df = self.to_dataframe()
        df.to_csv(path, index=False)
        logger.info("Saved results to %s", path)

    def summary_table(self) -> pd.DataFrame:
        """Return a pivot table comparing GARCH vs EGARCH per symbol."""
        df = self.to_dataframe()
        df = df[df["converged"]].copy()
        if df.empty:
            return pd.DataFrame()

        piv = df.pivot_table(
            index="symbol",
            columns="model_type",
            values=[
                "aic", "bic", "persistence", "mse_oos", "mae_oos",
                "converged", "fit_time_seconds",
            ],
        )
        piv.columns = [f"{c[0]}_{c[1].lower()}" for c in piv.columns]
        piv = piv.reset_index()
        return piv


def run_study(cfg: dict) -> StudyResults:
    """Run the full GARCH vs EGARCH study."""
    symbols = cfg["universe"]["symbols"]
    symbol_map = cfg["universe"].get("symbol_map", {})
    oos_frac = cfg["split"]["oos_fraction"]
    garch_p = cfg["model"]["garch"]["p"]
    garch_q = cfg["model"]["garch"]["q"]
    egarch_p = cfg["model"]["egarch"]["p"]
    egarch_o = cfg["model"]["egarch"].get("o", 1)
    egarch_q = cfg["model"]["egarch"]["q"]
    seed = cfg.get("seed", 42)

    np.random.seed(seed)

    logger.info("Loading data for %d symbols ...", len(symbols))
    df = load_b3_data(symbols, symbol_map)
    logger.info("Data loaded: %d rows, %d symbols", len(df), df["symbol"].nunique())

    df = compute_log_returns(df)
    df = df.dropna(subset=["log_return"])
    logger.info("Log returns computed. %d valid rows.", len(df))

    results = StudyResults()
    available_symbols = df["symbol"].unique().tolist()

    fig, axes = plt.subplots(
        nrows=len(available_symbols), ncols=2,
        figsize=(14, 3 * len(available_symbols)),
        squeeze=False,
    )

    for i, sym in enumerate(available_symbols):
        sym_df = df[df["symbol"] == sym].sort_values("timestamp").reset_index(drop=True)
        if len(sym_df) < 500:
            logger.warning("Symbol %s has insufficient data (%d rows), skipping.", sym, len(sym_df))
            continue

        returns = sym_df["log_return"]
        n = len(returns)
        split_idx = int(n * (1 - oos_frac))

        returns_is = returns.iloc[:split_idx]
        returns_oos = returns.iloc[split_idx:]

        logger.info(
            "%s: %d IS / %d OOS observations", sym, len(returns_is), len(returns_oos)
        )

        models_for_sym: Dict[str, VolatilityModel] = {}

        model = VolatilityModel(sym, "GARCH", p=garch_p, o=0, q=garch_q)
        fit_res = model.fit(returns_is, returns_oos)
        results.fits.append(fit_res)
        models_for_sym["GARCH"] = model

        model = VolatilityModel(sym, "EGARCH", p=egarch_p, o=egarch_o, q=egarch_q)
        fit_res = model.fit(returns_is, returns_oos)
        results.fits.append(fit_res)
        models_for_sym["EGARCH"] = model

        for col_idx, (model_type, color) in enumerate([("GARCH", "C0"), ("EGARCH", "C1")]):
            model_obj = models_for_sym.get(model_type)
            if model_obj is None:
                continue
            cv = model_obj.conditional_volatility
            if cv is None or len(cv) == 0:
                continue

            ax = axes[i, col_idx]
            ts_is = sym_df["timestamp"].iloc[:split_idx].iloc[:len(cv)]
            ax.plot(ts_is, cv, color=color, linewidth=0.6, label=f"{model_type} cond. vol.")
            ax.set_title(f"{sym} - {model_type}")
            ax.set_ylabel("Volatility")
            ax.legend(fontsize=7)

    plt.tight_layout()
    chart_path = CHARTS_DIR / "conditional_volatility.png"
    fig.savefig(chart_path, dpi=120)
    plt.close(fig)
    logger.info("Chart saved to %s", chart_path)

    return results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def generate_report(results: StudyResults, cfg: dict) -> str:
    """Write a markdown report summarizing findings."""
    df = results.to_dataframe()
    summary = results.summary_table()

    lines: List[str] = []
    lines.append("# GARCH vs EGARCH Study Report\n")
    lines.append(f"**Date:** {pd.Timestamp.now('America/Sao_Paulo').strftime('%Y-%m-%d %H:%M')}\n")
    lines.append(f"**Universe:** {', '.join(cfg['universe']['symbols'])}\n")
    lines.append(f"**Data:** B3 M15 (15-min) candles, "
                 f"IS/OOS split: {1-cfg['split']['oos_fraction']:.0%} / "
                 f"{cfg['split']['oos_fraction']:.0%}\n")
    lines.append("---\n")

    # 1. Convergence
    lines.append("## 1. Convergence\n")
    conv = df.groupby("model_type")["converged"].agg(["sum", "count"])
    lines.append(conv.to_markdown())
    lines.append("\n\n")

    # 2. Summary table
    lines.append("## 2. Model Comparison (AIC / BIC / Persistence / MSE OOS)\n")
    lines.append("Lower AIC/BIC is better. Lower MSE OOS is better.\n")
    if not summary.empty:
        lines.append(summary.to_markdown(index=False))
    lines.append("\n\n")

    # 3. Per-symbol breakdown
    lines.append("## 3. Per-Symbol Detail\n")
    for _, row in summary.iterrows():
        sym = row["symbol"]
        lines.append(f"### {sym}\n")
        lines.append("| Metric | GARCH | EGARCH |\n")
        lines.append("|---|---|---|\n")
        for col in ["aic", "bic", "persistence", "mse_oos", "mae_oos"]:
            g_val = row.get(f"{col}_garch", "N/A")
            e_val = row.get(f"{col}_egarch", "N/A")
            g_str = f"{g_val:.2f}" if isinstance(g_val, (int, float)) else str(g_val)
            e_str = f"{e_val:.2f}" if isinstance(e_val, (int, float)) else str(e_val)
            lines.append(f"| {col.upper()} | {g_str} | {e_str} |\n")
        lines.append("\n")

    # 4. EGARCH improvement - build pivot directly (column names use original case)
    lines.append("## 4. EGARCH Improvement Over GARCH\n")
    df_piv = df.pivot_table(
        index="symbol", columns="model_type",
        values=["aic", "bic", "mse_oos", "persistence", "mae_oos"],
    )
    # Flatten with lowercased model type
    df_piv.columns = [f"{c[0]}_{c[1].lower()}" for c in df_piv.columns]

    for col in ["aic", "bic", "mse_oos", "persistence", "mae_oos"]:
        g_col = f"{col}_garch"
        e_col = f"{col}_egarch"
        if g_col not in df_piv.columns or e_col not in df_piv.columns:
            continue

        df_piv[f"{col}_delta"] = df_piv[g_col] - df_piv[e_col]

    if "aic_delta" in df_piv.columns:
        lines.append("**AIC Delta (GARCH - EGARCH):** positive means EGARCH is better.\n")
        lines.append(df_piv["aic_delta"].to_frame("AIC_Delta").to_markdown())
        lines.append("\n\n")
        best_sym = df_piv["aic_delta"].idxmax()
        lines.append(
            f"> **Best EGARCH improvement (AIC):** {best_sym} "
            f"(ΔAIC = {df_piv.loc[best_sym, 'aic_delta']:.2f})\n"
        )
        worst_sym = df_piv["aic_delta"].idxmin()
        lines.append(
            f"> **Worst EGARCH improvement (AIC):** {worst_sym} "
            f"(ΔAIC = {df_piv.loc[worst_sym, 'aic_delta']:.2f})\n"
        )

    if "mse_oos_delta" in df_piv.columns:
        lines.append("\n**MSE OOS Delta (GARCH - EGARCH):** positive = EGARCH forecasts better.\n")
        lines.append(df_piv["mse_oos_delta"].to_frame("MSE_OOS_Delta").to_markdown())
        lines.append("\n\n")
        best_mse_sym = df_piv["mse_oos_delta"].idxmax()
        lines.append(
            f"> **Best MSE improvement:** {best_mse_sym} "
            f"(ΔMSE = {df_piv.loc[best_mse_sym, 'mse_oos_delta']:.6f})\n"
        )
        worst_mse_sym = df_piv["mse_oos_delta"].idxmin()
        lines.append(
            f"> **Worst MSE improvement:** {worst_mse_sym} "
            f"(ΔMSE = {df_piv.loc[worst_mse_sym, 'mse_oos_delta']:.6f})\n"
        )

    if "persistence_garch" in df_piv.columns and "persistence_egarch" in df_piv.columns:
        lines.append("\n**Persistence comparison:**\n")
        lines.append(
            df_piv[["persistence_garch", "persistence_egarch"]].to_markdown()
        )
        lines.append("\n\n")

    # 5. Failures
    lines.append("## 5. Failures / Negative Results\n")
    failed = df[~df["converged"]]
    if len(failed) > 0:
        lines.append(failed[["symbol", "model_type", "error_message"]].to_markdown(index=False))
    else:
        lines.append("All models converged successfully.\n")
    lines.append("\n")

    # 6. Timing
    lines.append("## 6. Computational Cost\n")
    timing = df.groupby("model_type")["fit_time_seconds"].describe()
    lines.append(timing.to_markdown())
    lines.append("\n\n")

    # 7. Parameter estimates
    lines.append("## 7. Parameter Estimates\n")
    for sym in sorted(df["symbol"].unique()):
        lines.append(f"### {sym}\n")
        sym_df = df[df["symbol"] == sym]
        for mt in ["GARCH", "EGARCH"]:
            row = sym_df[sym_df["model_type"] == mt]
            if row.empty:
                continue
            r = row.iloc[0]
            if not r["converged"]:
                lines.append(f"- **{mt}**: FAILED - {r['error_message']}\n")
                continue
            if mt == "GARCH":
                lines.append(
                    f"- **{mt}(1,1)**: omega={r['omega']:.6f}, alpha={r['alpha']:.6f}, "
                    f"beta={r['beta']:.6f}, persistence={r['persistence']:.4f}\n"
                )
            else:
                g_str = f"gamma={r['gamma']:.6f}" if r["gamma"] is not None else "gamma=0"
                lines.append(
                    f"- **{mt}(1,1,1)**: omega={r['omega']:.6f}, alpha={r['alpha']:.6f}, "
                    f"{g_str}, beta={r['beta']:.6f}, persistence={r['persistence']:.4f}\n"
                )
        lines.append("\n")

    # 8. Conclusions
    lines.append("## 8. Conclusions\n")

    garch_df = df[df["model_type"] == "GARCH"]
    egarch_df = df[df["model_type"] == "EGARCH"]

    if not garch_df.empty and not egarch_df.empty:
        mean_aic_g = garch_df["aic"].mean()
        mean_aic_e = egarch_df["aic"].mean()
        mean_pers_g = garch_df["persistence"].mean()
        mean_pers_e = egarch_df["persistence"].mean()
        mean_mse_g = garch_df["mse_oos"].dropna().mean()
        mean_mse_e = egarch_df["mse_oos"].dropna().mean()

        lines.append(f"- **Average AIC**: GARCH(1,1)={mean_aic_g:.1f}, EGARCH(1,1,1)={mean_aic_e:.1f}\n")
        lines.append(
            f"- **Average Persistence**: GARCH(1,1)={mean_pers_g:.4f}, "
            f"EGARCH(1,1,1)={mean_pers_e:.4f}\n"
        )
        lines.append(
            f"- **Average MSE OOS**: GARCH(1,1)={mean_mse_g:.6f}, "
            f"EGARCH(1,1,1)={mean_mse_e:.6f}\n"
        )
        lines.append(f"- **EGARCH wins on MSE OOS**: {mean_mse_e < mean_mse_g}\n")
        lines.append(f"- **EGARCH wins on AIC**: {mean_aic_e < mean_aic_g}\n")

        if "aic_delta" in df_piv.columns:
            best_positive = df_piv[df_piv["aic_delta"] > 0]
            best_negative = df_piv[df_piv["aic_delta"] < 0]
            lines.append(
                f"- **Stocks where EGARCH better (DeltaAIC > 0, {len(best_positive)}):** "
                f"{', '.join(best_positive.index.tolist()) if len(best_positive) > 0 else 'none'}\n"
            )
            lines.append(
                f"- **Stocks where GARCH better (DeltaAIC < 0, {len(best_negative)}):** "
                f"{', '.join(best_negative.index.tolist()) if len(best_negative) > 0 else 'none'}\n"
            )

        if "persistence_garch" in df_piv.columns:
            max_pers_sym = df_piv["persistence_garch"].idxmax()
            max_pers_val = df_piv.loc[max_pers_sym, "persistence_garch"]
            lines.append(
                f"- **Highest GARCH persistence:** {max_pers_sym} "
                f"({max_pers_val:.4f})\n"
            )

        egarch_conv = egarch_df[egarch_df["converged"]]
        if not egarch_conv.empty and egarch_conv["gamma"].notna().any():
            gamma_vals = egarch_conv["gamma"].dropna()
            n_negative = (gamma_vals < 0).sum()
            n_positive = (gamma_vals > 0).sum()
            lines.append(
                f"- **EGARCH gamma (leverage):** mean={gamma_vals.mean():.4f}, "
                f"min={gamma_vals.min():.4f}, max={gamma_vals.max():.4f}\n"
            )
            lines.append(f"  - {n_negative} stocks with negative gamma, {n_positive} with positive.\n")
            lines.append(
                "  - Negative gamma means negative residuals increase volatility "
                "more than positive ones (leverage effect).\n"
            )

    lines.append("\n---\n*Report generated automatically by experiment.py*\n")

    return "".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    cfg = load_config()
    logger.info("Config loaded: %s", CONFIG_PATH)

    results = run_study(cfg)

    csv_path = RESULTS_DIR / "garch_egarch_results.csv"
    results.save_csv(csv_path)

    summary = results.summary_table()
    if not summary.empty:
        summary_path = RESULTS_DIR / "summary.csv"
        summary.to_csv(summary_path, index=False)
        logger.info("Summary saved to %s", summary_path)

    report_md = generate_report(results, cfg)
    report_path = STUDY_DIR / "report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    logger.info("Report saved to %s", report_path)

    # Print key findings
    print("\n" + "=" * 75)
    print("KEY FINDINGS")
    print("=" * 75)
    df = results.to_dataframe()
    for sym in sorted(df["symbol"].unique()):
        for mt in ["GARCH", "EGARCH"]:
            row = df[(df["symbol"] == sym) & (df["model_type"] == mt)]
            if row.empty:
                continue
            r = row.iloc[0]
            if r["converged"]:
                extra = ""
                if mt == "EGARCH" and r["gamma"] is not None:
                    extra = f" gamma={r['gamma']:.4f}"
                aic_s = f"AIC={r['aic']:8.1f}"
                bic_s = f"BIC={r['bic']:8.1f}"
                pers_s = f"Pers={r['persistence']:.4f}"
                mse_s = f"MSE_OOS={r['mse_oos']:.3f}" if r["mse_oos"] is not None else "MSE_OOS=N/A"
                print(f"{sym:6s} {mt:8s} | {aic_s} {bic_s} {pers_s} {mse_s}{extra}")
            else:
                print(f"{sym:6s} {mt:8s} | FAILED: {r['error_message']}")
    print("=" * 75)

    # Delta summary
    print("\nEGARCH Improvement (DeltaAIC > 0 means EGARCH better):")
    for sym in sorted(df["symbol"].unique()):
        g = df[(df["symbol"] == sym) & (df["model_type"] == "GARCH")]
        e = df[(df["symbol"] == sym) & (df["model_type"] == "EGARCH")]
        if len(g) > 0 and len(e) > 0:
            gr = g.iloc[0]
            er = e.iloc[0]
            if gr["converged"] and er["converged"]:
                daic = gr["aic"] - er["aic"]
                dmse = (gr["mse_oos"] or 0) - (er["mse_oos"] or 0)
                marker = "*** EGARCH ***" if daic > 0 else "GARCH better"
                print(f"  {sym:6s}: DeltaAIC={daic:+8.1f}  DeltaMSE={dmse:+12.3f}  {marker}")

    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"Report saved to: {STUDY_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
