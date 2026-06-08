"""
Regime Forecast: Forward probability and supervised regime prediction for B3 stocks
====================================================================================
Refits 4-state HMMs for PETR4/VALE3/ITUB4, computes forward probability forecasts,
and trains classifiers to predict High/Extreme vol regimes 1-step ahead.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from arch import arch_model
from hmmlearn import hmm
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = Path(r"C:\Users\mathe\Documents\GitHub\mt5\dataset\export_mt5\intraday\avista\M15")
STUDY_DIR = Path(r"C:\Users\mathe\Documents\GitHub\volatility_research\studies\regime_forecast")
CHARTS_DIR = STUDY_DIR / "charts"
RESULTS_DIR = STUDY_DIR / "results"
for d in [CHARTS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SYMBOLS: List[str] = ["PETR4", "VALE3", "ITUB4"]
N_STATES: int = 4
VOL_WINDOW: int = 30
IS_END: str = "2024-12-31"
RANDOM_STATE: int = 42
N_BARS_PER_YEAR: int = 252 * 26
RETURN_SCALE: float = 100.0

sns.set_theme(style="darkgrid", palette="viridis")
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 150, "font.size": 10})


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_b3_data(symbol: str) -> pd.DataFrame:
    path = DATA_DIR / f"{symbol}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Data not found: {path}")
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


def compute_hmm_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_return"] = np.log(df["close_price"] / df["close_price"].shift(1))
    df["realized_vol"] = (
        df["log_return"].rolling(window=VOL_WINDOW).std() * np.sqrt(N_BARS_PER_YEAR)
    )
    df = df.dropna(subset=["log_return", "realized_vol"]).reset_index(drop=True)
    return df


def split_is_oos(df: pd.DataFrame, cutoff: str = IS_END) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ts = pd.Timestamp(cutoff, tz="America/Sao_Paulo") if "TZ" not in cutoff else pd.Timestamp(cutoff)
    cutoff_dt = pd.Timestamp(cutoff).tz_localize("America/Sao_Paulo")
    df_is = df[df["timestamp"] <= cutoff_dt].copy()
    df_oos = df[df["timestamp"] > cutoff_dt].copy()
    return df_is, df_oos


# ---------------------------------------------------------------------------
# HMM utilities
# ---------------------------------------------------------------------------
def fit_hmm(X: np.ndarray, n_states: int, n_iter: int = 1000) -> hmm.GaussianHMM:
    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=n_iter,
        tol=1e-4,
        random_state=RANDOM_STATE,
        init_params="stmc",
    )
    model.fit(X)
    return model


def label_regimes(
    model: hmm.GaussianHMM, X: np.ndarray, vol_col_idx: int = 1
) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    states = model.predict(X)
    state_means = {s: float(np.mean(X[states == s, vol_col_idx])) for s in range(model.n_components)}
    sorted_states = sorted(state_means, key=state_means.get)
    n = model.n_components
    if n == 4:
        labels = {
            sorted_states[0]: "Low Vol",
            sorted_states[1]: "Medium Vol",
            sorted_states[2]: "High Vol",
            sorted_states[3]: "Extreme Vol",
        }
    elif n == 3:
        labels = {sorted_states[0]: "Low Vol", sorted_states[1]: "Medium Vol", sorted_states[2]: "High Vol"}
    elif n == 2:
        labels = {sorted_states[0]: "Low Vol", sorted_states[1]: "High Vol"}
    else:
        labels = {s: f"State {s}" for s in range(n)}
    regime_map = {s: i for i, s in enumerate(sorted_states)}
    regimes = np.array([regime_map[s] for s in states])
    return states, regimes, labels


# ---------------------------------------------------------------------------
# Forward probability forecast
# ---------------------------------------------------------------------------
def compute_forward_forecasts(
    model: hmm.GaussianHMM,
    X: np.ndarray,
    timestamps: pd.DatetimeIndex,
    high_vol_state: int,
    extreme_vol_state: int,
) -> pd.DataFrame:
    T = model.transmat_
    T_10 = np.linalg.matrix_power(T, 10)

    filtered_probs = model.predict_proba(X)

    n = len(filtered_probs)
    records: List[Dict[str, Any]] = []
    for t in range(n):
        p_now = filtered_probs[t]
        p_fwd_1 = p_now @ T
        p_fwd_10 = p_now @ T_10

        records.append({
            "timestamp": timestamps[t],
            "p_highvol_fwd1": float(p_fwd_1[high_vol_state] + p_fwd_1[extreme_vol_state]),
            "p_extreme_fwd1": float(p_fwd_1[extreme_vol_state]),
            "p_highvol_fwd10": float(p_fwd_10[high_vol_state] + p_fwd_10[extreme_vol_state]),
            "p_extreme_fwd10": float(p_fwd_10[extreme_vol_state]),
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df[["timestamp", "symbol"]].copy()
    result["log_return"] = df["log_return"]

    result["rv_5"] = df["log_return"].rolling(5).std() * np.sqrt(N_BARS_PER_YEAR)
    result["rv_10"] = df["log_return"].rolling(10).std() * np.sqrt(N_BARS_PER_YEAR)
    result["rv_20"] = df["log_return"].rolling(20).std() * np.sqrt(N_BARS_PER_YEAR)

    result["skew_20"] = df["log_return"].rolling(20).skew()
    result["kurt_20"] = df["log_return"].rolling(20).kurt()

    result["ret_1"] = df["log_return"]
    result["ret_5"] = df["log_return"].rolling(5).sum()
    result["ret_10"] = df["log_return"].rolling(10).sum()

    result["dow"] = df["timestamp"].dt.dayofweek.astype(float)
    result["hour"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0

    vol_ma20 = df["volume"].rolling(20).mean().replace(0, np.nan)
    result["volume_ratio"] = (df["volume"] / vol_ma20).clip(0, 10)

    return result


def fit_garch_conditional_vol(returns: pd.Series) -> np.ndarray:
    try:
        am = arch_model(returns.dropna() * RETURN_SCALE, mean="zero", vol="GARCH", p=1, q=1, dist="normal")
        res = am.fit(disp="off", update_freq=0)
        cv = res.conditional_volatility.values / RETURN_SCALE
        full = np.full(len(returns), np.nan)
        idx = returns.dropna().index
        full[np.where(~returns.isna())[0][-len(cv):]] = cv
        return full
    except Exception:
        return np.full(len(returns), np.nan)


# ---------------------------------------------------------------------------
# Classifier training and evaluation
# ---------------------------------------------------------------------------
@dataclass
class ForecastMetrics:
    symbol: str
    model_name: str
    auc: float
    accuracy: float
    precision: float
    recall: float
    y_test: np.ndarray
    y_pred_proba: np.ndarray


def train_evaluate_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    symbol: str,
) -> List[ForecastMetrics]:
    results: List[ForecastMetrics] = []

    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced")
    lr.fit(X_train, y_train)
    lr_proba = lr.predict_proba(X_test)[:, 1]
    lr_pred = lr.predict(X_test)
    results.append(ForecastMetrics(
        symbol=symbol,
        model_name="LogisticRegression",
        auc=roc_auc_score(y_test, lr_proba),
        accuracy=accuracy_score(y_test, lr_pred),
        precision=precision_score(y_test, lr_pred, zero_division=0),
        recall=recall_score(y_test, lr_pred, zero_division=0),
        y_test=y_test,
        y_pred_proba=lr_proba,
    ))

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=RANDOM_STATE, class_weight="balanced",
        min_samples_leaf=5, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_proba = rf.predict_proba(X_test)[:, 1]
    rf_pred = rf.predict(X_test)
    results.append(ForecastMetrics(
        symbol=symbol,
        model_name="RandomForest",
        auc=roc_auc_score(y_test, rf_proba),
        accuracy=accuracy_score(y_test, rf_pred),
        precision=precision_score(y_test, rf_pred, zero_division=0),
        recall=recall_score(y_test, rf_pred, zero_division=0),
        y_test=y_test,
        y_pred_proba=rf_proba,
    ))

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_roc_curves(all_metrics: Dict[str, List[ForecastMetrics]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), squeeze=False)
    colors = {"LogisticRegression": "#3498db", "RandomForest": "#e74c3c"}
    for ax_idx, symbol in enumerate(SYMBOLS):
        ax = axes[0, ax_idx]
        metrics_list = all_metrics.get(symbol, [])
        for m in metrics_list:
            fpr, tpr, _ = roc_curve(m.y_test, m.y_pred_proba)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colors.get(m.model_name, "#333"),
                    label=f"{m.model_name} (AUC={roc_auc:.3f})", linewidth=1.5)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=0.8)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{symbol}  ROC Curve")
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Regime Forecast: ROC Curves (OOS 20252026)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "roc_curves.png", bbox_inches="tight")
    plt.close(fig)


def plot_forecast_probabilities(
    timestamps: pd.DatetimeIndex,
    actual_regimes: np.ndarray,
    fwd_probs: pd.DataFrame,
    symbol: str,
    split_idx: int,
) -> None:
    fig, ax = plt.subplots(figsize=(18, 5))
    ax.plot(timestamps, fwd_probs["p_highvol_fwd1"].values, color="#3498db",
            label="P(High/Extreme | t+1)", linewidth=0.6, alpha=0.8)
    ax.fill_between(timestamps, 0, fwd_probs["p_extreme_fwd1"].values, color="#e74c3c",
                     alpha=0.15, label="P(Extreme | t+1)")
    if split_idx > 0 and split_idx < len(timestamps):
        ax.axvline(timestamps[split_idx], color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("Probability")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"{symbol}  Forward Probability Forecasts (1-step ahead)")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / f"{symbol}_forecast_probabilities.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main study
# ---------------------------------------------------------------------------
def run_study() -> None:
    print("=" * 80)
    print("  REGIME FORECAST STUDY  Forward Probability + Supervised Forecast")
    print("=" * 80)

    all_metrics: Dict[str, List[ForecastMetrics]] = {s: [] for s in SYMBOLS}
    transition_forecasts: List[Dict[str, Any]] = []
    metrics_rows: List[Dict[str, Any]] = []

    for symbol in SYMBOLS:
        print(f"\n{'-'*80}")
        print(f"  {symbol}")
        print(f"{'-'*80}")

        df_raw = load_b3_data(symbol)
        df_feat = compute_hmm_features(df_raw)
        print(f"  Data: {len(df_feat):,} bars ({df_feat['timestamp'].min().date()}  {df_feat['timestamp'].max().date()})")

        # --- GARCH conditional vol ---
        print("  Fitting GARCH(1,1)...")
        garch_cv = fit_garch_conditional_vol(df_feat["log_return"])
        df_feat["garch_cond_vol"] = garch_cv * np.sqrt(N_BARS_PER_YEAR)

        # --- Split IS/OOS ---
        cutoff = pd.Timestamp(IS_END).tz_localize("America/Sao_Paulo")
        df_is = df_feat[df_feat["timestamp"] <= cutoff].copy()
        df_oos = df_feat[df_feat["timestamp"] > cutoff].copy()
        split_idx = len(df_is)
        print(f"  IS: {len(df_is):,} bars  |  OOS: {len(df_oos):,} bars")

        # --- HMM features and scaling ---
        hmm_features = ["log_return", "realized_vol"]
        X_is = df_is[hmm_features].values.astype(np.float64)
        X_oos = df_oos[hmm_features].values.astype(np.float64)
        X_all = df_feat[hmm_features].values.astype(np.float64)

        scaler = StandardScaler()
        X_is_s = scaler.fit_transform(X_is)
        X_oos_s = scaler.transform(X_oos)
        X_all_s = scaler.transform(X_all)

        # --- Fit HMM ---
        print("  Fitting 4-state HMM...")
        model = fit_hmm(X_is_s, N_STATES)
        print(f"    Converged: {model.monitor_.converged} (iter={model.monitor_.iter})")
        _, regimes_all, labels = label_regimes(model, X_all_s)
        label_names = [labels[k] for k in sorted(labels.keys())]
        print(f"    Labels (by vol): {label_names}")

        state_counts = pd.Series(regimes_all).value_counts(normalize=True).sort_index()
        for sid, pct in state_counts.items():
            print(f"      {labels[sid]}: {pct*100:.1f}%")

        # --- Forward probability forecasts ---
        print("  Computing forward probability forecasts...")
        regime_key = {v: k for k, v in labels.items()}
        high_orig = regime_key["High Vol"]
        extreme_orig = regime_key["Extreme Vol"]
        fwd_df = compute_forward_forecasts(
            model, X_all_s, df_feat["timestamp"],
            high_vol_state=high_orig, extreme_vol_state=extreme_orig,
        )
        transition_forecasts.append(fwd_df.assign(symbol=symbol))

        # --- Plot forward probabilities ---
        plot_forecast_probabilities(
            df_feat["timestamp"], regimes_all, fwd_df, symbol, split_idx
        )

        # --- Build supervised feature set ---
        print("  Building feature set for supervised forecast...")
        feat_df = build_features(df_feat)
        feat_df["garch_cond_vol"] = df_feat["garch_cond_vol"]

        # Target: regime at t+1 is High/Extreme (states 2 or 3)
        y_all = (regimes_all >= 2).astype(int)

        # Features and target, aligned (X at t, y at t+1)
        X_all_feat = feat_df.iloc[:-1].copy()
        y_aligned = y_all[1:]

        # Keep timestamps aligned to features
        ts_aligned = feat_df["timestamp"].iloc[:-1].values

        # Filter out rows with NaN features
        feature_cols = [
            "rv_5", "rv_10", "rv_20", "garch_cond_vol",
            "skew_20", "kurt_20", "ret_1", "ret_5", "ret_10",
            "dow", "hour", "volume_ratio",
        ]
        X_mat = X_all_feat[feature_cols].values.astype(np.float64)
        valid = ~np.isnan(X_mat).any(axis=1)
        X_clean = X_mat[valid]
        y_clean = y_aligned[valid]
        ts_clean = ts_aligned[valid]

        print(f"    Feature matrix: {X_clean.shape}")

        # --- Split supervised train/test ---
        cutoff_ts = pd.Timestamp(IS_END).tz_localize("America/Sao_Paulo")
        cutoff_naive = cutoff_ts.tz_localize(None)
        ts_clean_naive = pd.DatetimeIndex(ts_clean).tz_localize(None)
        train_idx = ts_clean_naive <= cutoff_naive
        test_idx = ~train_idx

        X_train = X_clean[train_idx]
        y_train = y_clean[train_idx]
        X_test = X_clean[test_idx]
        y_test = y_clean[test_idx]

        print(f"    Train: {len(X_train)}  |  Test: {len(X_test)}")
        print(f"    Target prevalence (train): {y_train.mean():.3f}")

        # --- Standardize features ---
        feat_scaler = StandardScaler()
        X_train_s = feat_scaler.fit_transform(X_train)
        X_test_s = feat_scaler.transform(X_test)

        # --- Train and evaluate ---
        print("  Training classifiers...")
        metrics_list = train_evaluate_models(X_train_s, y_train, X_test_s, y_test, symbol)
        all_metrics[symbol] = metrics_list

        for m in metrics_list:
            row = {
                "symbol": symbol,
                "model": m.model_name,
                "auc": round(m.auc, 4),
                "accuracy": round(m.accuracy, 4),
                "precision": round(m.precision, 4),
                "recall": round(m.recall, 4),
            }
            metrics_rows.append(row)
            print(f"    {m.model_name:22s} | AUC={m.auc:.4f} | Acc={m.accuracy:.4f} | "
                  f"Prec={m.precision:.4f} | Rec={m.recall:.4f}")

        # --- Print transition matrix info ---
        print(f"\n  Transition matrix diagonal persistence:")
        tm = model.transmat_
        for sid, lbl in labels.items():
            print(f"    {lbl}: diag={tm[sid, sid]:.4f}")

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    # Metrics
    df_metrics = pd.DataFrame(metrics_rows)
    df_metrics.to_csv(RESULTS_DIR / "forecast_metrics.csv", index=False)
    print(f"\n  Metrics saved to {RESULTS_DIR / 'forecast_metrics.csv'}")

    # Transition forecasts
    df_tf = pd.concat(transition_forecasts, ignore_index=True)
    df_tf.to_csv(RESULTS_DIR / "transition_forecasts.csv", index=False)
    print(f"  Transition forecasts saved to {RESULTS_DIR / 'transition_forecasts.csv'}")

    # ROC curves
    plot_roc_curves(all_metrics)
    print(f"  ROC curves saved to {CHARTS_DIR / 'roc_curves.png'}")

    print(f"\n{'='*80}")
    print("  STUDY COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    run_study()
