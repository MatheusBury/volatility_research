"""
Hidden Markov Models for Volatility Regime Detection in B3 Stocks
===================================================================
Study: 15-minute intraday data for PETR4, VALE3, ITUB4
Period: IS 2021-2024, OOS 2025-2026
"""

import os, sys, warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DATA_DIR = Path(r"C:\Users\mathe\Documents\GitHub\mt5\dataset\export_mt5\intraday\avista\M15")
STUDY_DIR = Path(r"C:\Users\mathe\Documents\GitHub\volatility_research\studies\hmm_regimes")
CHARTS_DIR = STUDY_DIR / "charts"
RESULTS_DIR = STUDY_DIR / "data"
for d in [CHARTS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SYMBOLS = ["PETR4", "VALE3", "ITUB4"]
N_STATES_CANDIDATES = [2, 3, 4]
VOL_WINDOW = 30
IS_END = "2024-12-31"
RANDOM_STATE = 42

def load_data(symbol: str) -> pd.DataFrame:
    path = DATA_DIR / f"{symbol}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Data not found: {path}")
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index, utc=False)
    df = df.sort_index()
    df = df[df["Tick_volume"] > 0].copy()
    return df

def compute_features(df: pd.DataFrame, vol_window: int = VOL_WINDOW) -> pd.DataFrame:
    df = df.copy()
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    N_BARS_PER_YEAR = 252 * 26
    df["realized_vol"] = (df["log_return"].rolling(window=vol_window).std() * np.sqrt(N_BARS_PER_YEAR))
    df = df.dropna(subset=["log_return", "realized_vol"]).copy()
    return df

def split_is_oos(df: pd.DataFrame, is_end: str = IS_END) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = pd.Timestamp(is_end)
    df_is = df[df.index <= cutoff].copy()
    df_oos = df[df.index > cutoff].copy()
    return df_is, df_oos

def fit_hmm(features: np.ndarray, n_states: int, n_iter: int = 1000, random_state: int = RANDOM_STATE) -> hmm.GaussianHMM:
    model = hmm.GaussianHMM(n_components=n_states, covariance_type="full", n_iter=n_iter, tol=1e-4, random_state=random_state, init_params="stmc")
    model.fit(features)
    return model

def compute_bic(model: hmm.GaussianHMM, X: np.ndarray) -> float:
    n_params = model.n_components * (model.n_components - 1) + model.n_components * X.shape[1]
    if model.covariance_type == "full":
        n_params += model.n_components * X.shape[1] * (X.shape[1] + 1) // 2
    elif model.covariance_type == "diag":
        n_params += model.n_components * X.shape[1]
    elif model.covariance_type == "spherical":
        n_params += model.n_components
    elif model.covariance_type == "tied":
        n_params += X.shape[1] * (X.shape[1] + 1) // 2
    log_lik = model.score(X)
    return -2 * log_lik + n_params * np.log(len(X))

def compute_aic(model: hmm.GaussianHMM, X: np.ndarray) -> float:
    n_params = model.n_components * (model.n_components - 1) + model.n_components * X.shape[1]
    if model.covariance_type == "full":
        n_params += model.n_components * X.shape[1] * (X.shape[1] + 1) // 2
    log_lik = model.score(X)
    return -2 * log_lik + 2 * n_params

def label_regimes(model: hmm.GaussianHMM, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
    states = model.predict(X)
    state_means = {s: np.mean(X[states == s, 1]) for s in range(model.n_components)}
    sorted_states = sorted(state_means, key=state_means.get)
    n = model.n_components
    if n == 4:
        labels = {sorted_states[0]: "Low Vol", sorted_states[1]: "Medium Vol", sorted_states[2]: "High Vol", sorted_states[3]: "Extreme Vol"}
    elif n == 3:
        labels = {sorted_states[0]: "Low Vol", sorted_states[1]: "Medium Vol", sorted_states[2]: "High Vol"}
    elif n == 2:
        labels = {sorted_states[0]: "Low Vol", sorted_states[1]: "High Vol"}
    else:
        labels = {s: f"State {s}" for s in range(n)}
    regime_map = {s: i for i, s in enumerate(sorted_states)}
    regimes = np.array([regime_map[s] for s in states])
    return states, regimes, labels

def transition_matrix(model: hmm.GaussianHMM) -> pd.DataFrame:
    tm = pd.DataFrame(model.transmat_)
    tm.index = [f"From {i}" for i in range(len(tm))]
    tm.columns = [f"To {i}" for i in range(len(tm))]
    return tm

def state_durations(states: np.ndarray) -> Dict:
    durations = {}
    if len(states) == 0:
        return durations
    cs, cl = states[0], 1
    for s in states[1:]:
        if s == cs:
            cl += 1
        else:
            durations.setdefault(cs, []).append(cl)
            cs, cl = s, 1
    durations.setdefault(cs, []).append(cl)
    return durations

def set_style():
    sns.set_theme(style="darkgrid", palette="viridis")
    plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 150, "font.size": 10})

def plot_regime_probabilities(model, X, dates, symbol, n_states, split_idx=None):
    set_style()
    probs = model.predict_proba(X)
    fig, axes = plt.subplots(n_states, 1, figsize=(16, 2.5*n_states), sharex=True, squeeze=False)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, n_states))
    for i in range(n_states):
        ax = axes[i][0]
        ax.fill_between(dates, 0, probs[:, i], color=colors[i], alpha=0.7)
        if split_idx is not None:
            ax.axvline(dates[split_idx], color="red", linestyle="--", alpha=0.5)
        ax.set_ylabel(f"State {i}", fontsize=9)
        ax.set_ylim(-0.02, 1.02)
    axes[-1][0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[-1][0].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate()
    fig.suptitle(f"{symbol} - Regime Probabilities ({n_states}-State HMM)", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = CHARTS_DIR / f"{symbol}_probs_{n_states}states.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

def plot_regime_timeline(df, regimes, symbol, n_states, labels, split_idx=None):
    set_style()
    fig, ax = plt.subplots(figsize=(18, 6))
    colors = ["#2ecc71", "#f39c12", "#e74c3c", "#8e44ad"][:n_states]
    for state_id in range(n_states):
        mask = regimes == state_id
        keys = list(labels.keys())
        label_name = labels.get(keys[state_id], f"State {state_id}")
        ax.scatter(df.index[mask], df["Close"][mask], c=colors[state_id], label=label_name, s=2, alpha=0.6, rasterized=True)
    if split_idx is not None:
        ax.axvline(df.index[split_idx], color="gray", linestyle="--", alpha=0.7)
    ax.set_xlabel("Date"); ax.set_ylabel("Price (R$)")
    ax.set_title(f"{symbol} - Price by Regime ({n_states}-State HMM)", fontsize=14, fontweight="bold")
    ax.legend(markerscale=5, fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate()
    fig.tight_layout()
    path = CHARTS_DIR / f"{symbol}_timeline_{n_states}states.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

def plot_transition_heatmap(model, symbol, n_states, labels):
    set_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    tm = model.transmat_
    label_names = [labels.get(i, f"S{i}") for i in range(n_states)]
    sns.heatmap(tm, annot=True, fmt=".3f", cmap="Blues", xticklabels=label_names, yticklabels=label_names, ax=ax, vmin=0, vmax=1, square=True)
    ax.set_title(f"{symbol} - Transition Matrix ({n_states}-State HMM)", fontsize=14, fontweight="bold")
    ax.set_xlabel("To"); ax.set_ylabel("From")
    fig.tight_layout()
    path = CHARTS_DIR / f"{symbol}_transmat_{n_states}states.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

def plot_vol_by_regime(df, regimes, symbol, n_states, labels):
    set_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    keys = sorted(labels.keys())
    data_by_reg = [df[regimes == list(labels.keys()).index(k)]["realized_vol"].dropna().values for k in keys]
    reg_names = [labels[k] for k in keys]
    colors = ["#2ecc71", "#f39c12", "#e74c3c", "#8e44ad"][:n_states]
    bp = ax.boxplot(data_by_reg, labels=reg_names, patch_artist=True, showmeans=True, meanprops=dict(marker="D", markerfacecolor="red"))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color); patch.set_alpha(0.6)
    ax.set_title(f"{symbol} - Vol by Regime ({n_states}-State HMM)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Annualized Realized Volatility")
    fig.tight_layout()
    path = CHARTS_DIR / f"{symbol}_volbox_{n_states}states.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

def run_study():
    print("=" * 80)
    print("HMM VOLATILITY REGIME DETECTION - B3 STOCKS (15min data)")
    print("=" * 80)
    all_results = []

    for symbol in SYMBOLS:
        print(f"\n{'---'*20}")
        print(f"  SYMBOL: {symbol}")
        print(f"{'---'*20}")
        df_raw = load_data(symbol)
        df_feat = compute_features(df_raw)
        print(f"  Rows: {len(df_feat):,}")
        df_is, df_oos = split_is_oos(df_feat)
        print(f"  IS: {df_is.index.min().date()} to {df_is.index.max().date()} ({len(df_is):,})")
        print(f"  OOS: {df_oos.index.min().date()} to {df_oos.index.max().date()} ({len(df_oos):,})")

        feature_cols = ["log_return", "realized_vol"]
        X_is = df_is[feature_cols].values.astype(np.float64)
        X_oos = df_oos[feature_cols].values.astype(np.float64)
        X_all = df_feat[feature_cols].values.astype(np.float64)

        scaler = StandardScaler()
        X_is_s = scaler.fit_transform(X_is)
        X_oos_s = scaler.transform(X_oos)
        X_all_s = scaler.transform(X_all)
        split_idx = len(X_is_s)

        for n_states in N_STATES_CANDIDATES:
            print(f"\n  >> {n_states}-State HMM")
            try:
                model = fit_hmm(X_is_s, n_states)
                print(f"     Converged: {model.monitor_.converged} (iter={model.monitor_.iter})")
            except Exception as e:
                print(f"     FAIL: {e}")
                all_results.append({"symbol": symbol, "n_states": n_states, "status": "FAIL", "error": str(e)})
                continue

            bic = compute_bic(model, X_is_s)
            aic = compute_aic(model, X_is_s)
            log_lik = model.score(X_is_s)
            print(f"     LogL={log_lik:.1f}  BIC={bic:.1f}  AIC={aic:.1f}")

            states_is, regimes_is, labels = label_regimes(model, X_is_s)
            states_all, regimes_all, _ = label_regimes(model, X_all_s)

            for k, v in labels.items():
                cnt = int(np.sum(regimes_is == list(labels.keys()).index(k)))
                print(f"     {v}: {cnt} bars ({cnt/len(regimes_is)*100:.1f}%)")

            tm = pd.DataFrame(model.transmat_)
            print(f"     Transition Matrix:\n{tm.to_string()}")

            durs = state_durations(states_is)
            print(f"     State durations (15-min bars):")
            for s, dlist in durs.items():
                mean_d = np.mean(dlist)
                print(f"       {labels.get(s, s)}: mean={mean_d:.1f} bars ({mean_d*0.25:.1f}h)")

            if len(X_oos_s) > 0:
                oos_ll = model.score(X_oos_s)
                print(f"     OOS LogL={oos_ll:.1f}")
            else:
                oos_ll = None

            # Plots
            try:
                plot_regime_probabilities(model, X_all_s, df_feat.index, symbol, n_states, split_idx)
                plot_regime_timeline(df_feat, regimes_all, symbol, n_states, labels, split_idx)
                plot_transition_heatmap(model, symbol, n_states, labels)
                plot_vol_by_regime(df_feat, regimes_all, symbol, n_states, labels)
                print(f"     Charts saved to {CHARTS_DIR}")
            except Exception as e:
                print(f"     Plot error: {e}")

            all_results.append({
                "symbol": symbol, "n_states": n_states, "status": "OK",
                "model": model, "bic": bic, "aic": aic, "log_lik": log_lik,
                "oos_log_lik": oos_ll, "labels": labels,
                "transmat": model.transmat_.copy(),
            })

        # Summary per symbol
        print(f"\n  Summary for {symbol}:")
        for r in [x for x in all_results if x.get("symbol")==symbol and x.get("status")=="OK"]:
            oos = f"{r['oos_log_lik']:.1f}" if r['oos_log_lik'] else "N/A"
            print(f"    {r['n_states']}-state: BIC={r['bic']:.1f}  AIC={r['aic']:.1f}  OOS-L={oos}")

    # Best model selection
    print(f"\n{'='*70}")
    print("  BEST MODEL BY BIC")
    print(f"{'='*70}")
    best_by_symbol = {}
    for symbol in SYMBOLS:
        valid = [r for r in all_results if r.get("symbol")==symbol and r.get("status")=="OK"]
        if not valid: continue
        best = min(valid, key=lambda r: r["bic"])
        best_by_symbol[symbol] = best
        print(f"\n  {symbol}: Best = {best['n_states']}-state (BIC={best['bic']:.1f})")
        print(f"  Labels: {best['labels']}")
        print(f"  Transition Matrix:\n{pd.DataFrame(best['transmat']).to_string()}")
        tm_best = best['transmat']
        for sid, lbl in best['labels'].items():
            p_stay = tm_best[sid, sid]
            exp_bars = 1.0/(1-p_stay) if p_stay < 1 else float('inf')
            print(f"    {lbl}: E[duration]={exp_bars:.1f} bars ({exp_bars*0.25:.1f}h) [p_stay={p_stay:.4f}]")

    # Save CSV summaries
    rows_s = []
    for r in all_results:
        if r.get("status")=="OK":
            rows_s.append({"symbol":r["symbol"],"n_states":r["n_states"],"bic":r["bic"],"aic":r["aic"],"log_lik":r["log_lik"],"oos_log_lik":r["oos_log_lik"]})
    pd.DataFrame(rows_s).to_csv(RESULTS_DIR/"model_summary.csv", index=False)
    print(f"\n  Summary saved to {RESULTS_DIR/'model_summary.csv'}")
    print(f"\n  All charts saved to {CHARTS_DIR}")
    print(f"\n{'='*70}")
    print("  STUDY COMPLETE")
    print(f"{'='*70}")
    return all_results, best_by_symbol

if __name__ == "__main__":
    run_study()
