"""Options Volatility Engine — Daily Study for B3 Options.

Studies:
  1. Forecast RV vs IV (VRP analysis)
  2. VRP Deciles and future returns
  3. VRP by Volatility Regime
  4. Volatility Surface (ATM/OTM/ITM vs RV Forecast)
"""

import json
import os
import warnings
from datetime import datetime

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from arch import arch_model
from hmmlearn import hmm
from vollib.black_scholes.implied_volatility import implied_volatility as iv_calc

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Paths ──────────────────────────────────────────────────────────────
STUDY_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(STUDY_DIR, "results")
CHARTS_DIR = os.path.join(STUDY_DIR, "charts")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)

# Data paths
D1_DATA_DIR = r"C:\Users\mathe\Documents\GitHub\mt5\dataset\export_mt5\intraday\avista\D1"
OPTIONS_DATA_DIR = r"C:\Users\mathe\Documents\GitHub\volatility_research\studies\real_vrp\results\options_data"
OPTION_CHAIN_PATH = r"C:\Users\mathe\Documents\GitHub\volatility_research\studies\real_vrp\results\option_chain.csv"
OPLAB_RAW_DIR = os.path.join(RESULTS_DIR, "oplab_raw")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(STUDY_DIR)), "data")

SYMBOLS = [
    "ABEV3", "ASAI3", "AZUL4", "B3SA3", "BBAS3", "BBDC4", "BPAC11", "BRFS3",
    "CCRO3", "CMIG4", "COGN3", "CPLE6", "CRFB3", "CSNA3", "CVCB3", "ELET3",
    "EMBR3", "ENEV3", "ENGI11", "EQTL3", "FLRY3", "GGBR4", "GOLL4", "HAPV3",
    "ITUB4", "JBSS3", "KLBN11", "LREN3", "MGLU3", "MRFG3", "NEOE3", "ODPV3",
    "PCAR3", "PETR4", "PRIO3", "RADL3", "RAIL3", "RDOR3", "RENT3", "SANB11",
    "SUZB3", "TIMS3", "TOTS3", "UGPA3", "VALE3", "VBBR3", "VIIA3", "VIVT3",
    "WEGE3", "YDUQS3",
]
SYMBOL_PREFIXES = {"PETR4": "PETR", "VALE3": "VALE", "ITUB4": "ITUB",
                   "BBDC4": "BBDC", "BBAS3": "BBAS", "WEGE3": "WEGE",
                   "ABEV3": "ABEV", "RENT3": "RENT", "SUZB3": "SUZB", "GGBR4": "GGBR"}
SEED = 42
N_BARS_PER_YEAR = 252
RISK_FREE_RATE = 0.1475  # SELIC
GARCH_WINDOW = 504  # 2 years of daily data for refit
GARCH_FORECAST_HORIZON = 20  # 20-day ahead vol forecast
RV_WINDOW = 20  # 20-day rolling realized vol

np.random.seed(SEED)


# ═══════════════════════════════════════════════════════════════════════
#  Data Loading
# ═══════════════════════════════════════════════════════════════════════

def load_daily_underlying(symbol: str) -> pd.DataFrame:
    path = os.path.join(D1_DATA_DIR, f"{symbol}.parquet")
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
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize("America/Sao_Paulo")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def load_option_chain() -> pd.DataFrame:
    df = pd.read_csv(OPTION_CHAIN_PATH)
    df["strike"] = df["strike"].astype(float)
    df["expiration_date"] = pd.to_datetime(df["expiration_date"])
    return df


def load_option_ts(symbol: str) -> pd.DataFrame:
    chain = load_option_chain()
    options = chain[chain["stock"] == symbol].copy()

    rows = []
    for _, opt in options.iterrows():
        fname = f"{opt['symbol']}.csv"
        path = os.path.join(OPTIONS_DATA_DIR, fname)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["strike"] = opt["strike"]
        df["option_type"] = opt["option_type"]
        df["expiration_date"] = opt["expiration_date"]
        df["option_symbol"] = opt["symbol"]
        df = df.sort_values("timestamp").reset_index(drop=True)
        rows.append(df)

    if not rows:
        return pd.DataFrame()

    full = pd.concat(rows, ignore_index=True)
    full = full.drop_duplicates(subset=["timestamp", "option_symbol"]).sort_values("timestamp")
    return full


# ═══════════════════════════════════════════════════════════════════════
#  Volatility Calculations
# ═══════════════════════════════════════════════════════════════════════

def compute_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_return"] = np.log(df["close_price"] / df["close_price"].shift(1))
    return df


def compute_realized_vol(df: pd.DataFrame, window: int = RV_WINDOW) -> pd.DataFrame:
    df = df.copy()
    df["rv_20d"] = df["log_return"].rolling(window).std() * np.sqrt(N_BARS_PER_YEAR)
    return df


def fit_garch_forecast(returns: pd.Series) -> tuple:
    model = arch_model(returns.dropna() * 100, vol="GARCH", p=1, q=1, dist="normal")
    res = model.fit(disp="off", show_warning=False)
    omega = res.params["omega"]
    alpha = res.params["alpha[1]"]
    beta = res.params["beta[1]"]
    persistence = alpha + beta
    forecast = res.forecast(horizon=GARCH_FORECAST_HORIZON)
    fv = np.sqrt(forecast.variance.values[-1, :].mean()) / 100 * np.sqrt(N_BARS_PER_YEAR)
    return fv, persistence, omega, alpha, beta


def compute_garch_forecast_series(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["garch_forecast_20d"] = np.nan
    df["garch_persistence"] = np.nan

    returns = df["log_return"].copy()
    n = len(df)

    for i in range(GARCH_WINDOW, n):
        try:
            r = returns.iloc[i - GARCH_WINDOW : i].dropna()
            if len(r) < 252:
                continue
            fv, pers, _, _, _ = fit_garch_forecast(r)
            df.loc[df.index[i], "garch_forecast_20d"] = fv
            df.loc[df.index[i], "garch_persistence"] = pers
        except Exception:
            continue

    return df


def compute_option_iv(price: float, spot: float, strike: float, tte: float,
                      flag: str, r: float = RISK_FREE_RATE) -> float | None:
    if price <= 0 or spot <= 0 or strike <= 0 or tte <= 0:
        return None
    try:
        iv = iv_calc(price, spot, strike, tte, r, flag)
        if 0 < iv < 5.0:
            return iv
    except Exception:
        return None
    return None


def compute_daily_iv(options_df: pd.DataFrame,
                     underlying_df: pd.DataFrame) -> pd.DataFrame:
    if options_df.empty:
        return pd.DataFrame()

    options_df = options_df.copy()
    options_df["timestamp"] = options_df["timestamp"].dt.tz_localize("America/Sao_Paulo", ambiguous="NaT")
    options_df = options_df.dropna(subset=["timestamp"])
    underlying = underlying_df[["timestamp", "close_price"]].copy()
    underlying = underlying.rename(columns={"close_price": "underlying_price"})

    merged = options_df.merge(underlying, on="timestamp", how="inner")
    merged["dte"] = (merged["expiration_date"] - merged["timestamp"].dt.tz_localize(None)).dt.days
    merged = merged[merged["dte"].between(20, 45)].copy()
    merged["tte"] = merged["dte"] / 365.0
    merged["moneyness"] = merged["strike"] / merged["underlying_price"]
    merged = merged[merged["moneyness"].between(0.95, 1.05)].copy()
    merged = merged[merged["volume"] > 0].copy()
    merged["iv"] = np.nan

    for idx, row in merged.iterrows():
        flag = "p" if row["option_type"] == "P" else "c"
        iv = compute_option_iv(
            price=row["close_price"],
            spot=row["underlying_price"],
            strike=row["strike"],
            tte=row["tte"],
            flag=flag
        )
        merged.loc[idx, "iv"] = iv

    merged = merged.dropna(subset=["iv"])
    merged = merged[merged["iv"].between(0.05, 1.50)].copy()
    return merged


def load_oplab_options(symbol: str) -> pd.DataFrame:
    """Load saved OpLab API option snapshot data."""
    path = os.path.join(OPLAB_RAW_DIR, f"{symbol}_all_options.parquet")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["underlying"] = symbol
    if "type" in df.columns and "option_type" not in df.columns:
        df = df.rename(columns={"type": "option_type"})
    if "option_type" not in df.columns:
        df["option_type"] = ""
    df["option_type"] = df["option_type"].astype(str).str.upper().str[0]
    # Compute moneyness if not present (needed for surface study)
    if "moneyness" not in df.columns:
        spot_col = "spot_price" if "spot_price" in df.columns else "underlying_price"
        if spot_col in df.columns:
            df["moneyness"] = df["strike"] / df[spot_col]
    # Quality filters: IV range, volume > 0, DTE 20-45 (keep all moneyness for surface study)
    df = df[df["iv"].notna()].copy()
    df = df[df["iv"].between(0.05, 1.50)].copy()
    df = df[df["volume"] > 0].copy()
    df = df[df["dte"].between(20, 45)].copy()
    return df


def load_ivx_data(symbol: str) -> pd.DataFrame:
    """Load IVX (implied volatility index) daily time series."""
    path = os.path.join(DATA_DIR, f"{symbol}IVX.json")
    if not os.path.exists(path):
        return pd.DataFrame()
    with open(path) as f:
        d = json.load(f)
    df = pd.DataFrame(d["data"])
    df["timestamp"] = pd.to_datetime(df["time"], unit="ms")
    df["timestamp"] = df["timestamp"].dt.tz_localize("America/Sao_Paulo", ambiguous="NaT")
    # Normalize to midnight for alignment with underlying data
    df["timestamp"] = df["timestamp"].dt.normalize()
    df["iv"] = df["close"] / 100.0
    df["source"] = "ivx"
    df["moneyness"] = 1.0
    return df[["timestamp", "iv", "source", "moneyness"]].dropna().sort_values("timestamp").reset_index(drop=True)


def load_ewma_data(symbol: str) -> pd.DataFrame:
    """Load EWMAB3 (B3 realized vol estimate) daily time series."""
    path = os.path.join(DATA_DIR, f"{symbol}EWMAB3.json")
    if not os.path.exists(path):
        return pd.DataFrame()
    with open(path) as f:
        d = json.load(f)
    df = pd.DataFrame(d["data"])
    df["timestamp"] = pd.to_datetime(df["time"], unit="ms")
    df["timestamp"] = df["timestamp"].dt.tz_localize("America/Sao_Paulo", ambiguous="NaT")
    df["rv_ewma"] = df["close"] / 100.0
    return df[["timestamp", "rv_ewma"]].dropna().sort_values("timestamp").reset_index(drop=True)


def get_atm_iv(merged: pd.DataFrame) -> pd.DataFrame:
    """Get ATM IV for each day (closest strike among 0.95-1.05 filtered options)."""
    atm_days = []
    for _, group in merged.groupby("timestamp"):
        group = group.copy()
        group["dist"] = np.abs(group["moneyness"] - 1.0)
        best = group.loc[group["dist"].idxmin()]
        atm_days.append(best.to_dict())

    atm = pd.DataFrame(atm_days)
    if not atm.empty:
        atm = atm.sort_values("timestamp").reset_index(drop=True)
    return atm


def classify_vol_regime(rv_series: pd.Series, n_states: int = 4) -> np.ndarray:
    """Classify days into Low/Medium/High/Extreme vol using HMM."""
    valid = rv_series.dropna().values.reshape(-1, 1)
    model = hmm.GaussianHMM(n_components=n_states, covariance_type="full",
                            random_state=SEED, n_iter=1000)
    model.fit(valid)
    states = model.predict(valid)
    # Sort states by mean vol (0=lowest, n-1=highest)
    means = model.means_.flatten()
    order = np.argsort(means)
    mapper = {old: new for new, old in enumerate(order)}
    return np.array([mapper[s] for s in states]), model


# ═══════════════════════════════════════════════════════════════════════
#  Studies
# ═══════════════════════════════════════════════════════════════════════

def study1_forecast_vs_iv(atm_iv: pd.DataFrame,
                           underlying: pd.DataFrame,
                           symbol: str) -> dict:
    """Compare GARCH forecast RV vs ATM IV, measure VRP and future returns."""
    merged = atm_iv.merge(
        underlying[["timestamp", "rv_20d", "garch_forecast_20d"]],
        on="timestamp", how="inner"
    )
    if merged.empty:
        return {"error": "No data"}

    merged["vrp"] = merged["iv"] - merged["garch_forecast_20d"]
    merged["vrp_rv"] = merged["iv"] - merged["rv_20d"]

    # Future RV: compute forward-looking 20d realized vol
    rv_series = underlying.set_index("timestamp")["rv_20d"]
    merged["future_rv_20d"] = merged["timestamp"].map(
        lambda t: rv_series.shift(-RV_WINDOW).reindex(rv_series.index).loc[t]
        if t in rv_series.index else np.nan
    )

    # Future option return (simplified: change in IV)
    iv_series = merged.set_index("timestamp")["iv"]
    merged["future_iv_change"] = (
        iv_series.shift(-1).reindex(iv_series.index).values
        - merged["iv"].values
    )

    # Signal: long vol when VRP < -5%, short when VRP > +5%
    merged["signal"] = 0
    merged.loc[merged["vrp"] < -0.05, "signal"] = 1  # long vol
    merged.loc[merged["vrp"] > 0.05, "signal"] = -1  # short vol

    merged["signal_return"] = merged["signal"] * merged["future_iv_change"]
    valid = merged.dropna(subset=["signal_return", "signal"])
    long = valid[valid["signal"] == 1]
    short = valid[valid["signal"] == -1]

    results = {
        "symbol": symbol,
        "n_obs": len(merged),
        "mean_iv": float(merged["iv"].mean()),
        "mean_garch_forecast": float(merged["garch_forecast_20d"].mean()),
        "mean_rv": float(merged["rv_20d"].mean()),
        "mean_vrp_iv_garch": float(merged["vrp"].mean()),
        "mean_vrp_iv_rv": float(merged["vrp_rv"].mean()),
        "vrp_positive_pct": float((merged["vrp"] > 0).mean() * 100),
        "long_vol_n": len(long),
        "long_vol_mean_return": float(long["signal_return"].mean()) if len(long) > 0 else 0,
        "short_vol_n": len(short),
        "short_vol_mean_return": float(short["signal_return"].mean()) if len(short) > 0 else 0,
        "signal_hit_rate": float(
            (valid["signal_return"] > 0).mean() if len(valid) > 0 else 0
        ),
    }

    # Save
    merged.to_csv(os.path.join(RESULTS_DIR, f"{symbol}_vrp_analysis.csv"), index=False)

    # Chart: VRP timeseries
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(pd.to_datetime(merged["timestamp"]), merged["vrp"],
            label="VRP (IV - GARCH Forecast)", color="navy", lw=0.8)
    ax.axhline(0, color="gray", ls="--", lw=0.5)
    ax.axhline(0.05, color="red", ls=":", lw=0.5, alpha=0.5)
    ax.axhline(-0.05, color="green", ls=":", lw=0.5, alpha=0.5)
    ax.fill_between(pd.to_datetime(merged["timestamp"]), 0, merged["vrp"],
                    where=merged["vrp"] > 0, color="red", alpha=0.1, label="IV cara")
    ax.fill_between(pd.to_datetime(merged["timestamp"]), 0, merged["vrp"],
                    where=merged["vrp"] < 0, color="green", alpha=0.1, label="IV barata")
    ax.set_title(f"VRP Timeseries — {symbol}")
    ax.set_ylabel("VRP (% anualizada)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, f"{symbol}_vrp_timeseries.png"), dpi=150)
    plt.close(fig)

    return results


def study2_vrp_deciles(atm_iv: pd.DataFrame,
                       underlying: pd.DataFrame,
                       symbol: str) -> dict:
    """Separate VRP into deciles, measure future RV and option returns."""
    merged = atm_iv.merge(
        underlying[["timestamp", "rv_20d", "garch_forecast_20d"]],
        on="timestamp", how="inner"
    )
    if merged.empty or len(merged) < 10:
        return {"error": "Insufficient data"}

    merged["vrp"] = merged["iv"] - merged["garch_forecast_20d"]

    # Future RV
    rv_series = underlying.set_index("timestamp")["rv_20d"]
    merged["future_rv_20d"] = merged["timestamp"].map(
        lambda t: rv_series.shift(-RV_WINDOW).reindex(rv_series.index).loc[t]
        if t in rv_series.index else np.nan
    )

    # Deciles
    merged["decile"] = pd.qcut(merged["vrp"], 10, labels=False, duplicates="drop")

    decile_stats = merged.groupby("decile").agg(
        n=("vrp", "count"),
        mean_vrp=("vrp", "mean"),
        mean_iv=("iv", "mean"),
        mean_future_rv=("future_rv_20d", "mean"),
        mean_rv=("rv_20d", "mean"),
    ).reset_index()
    decile_stats["symbol"] = symbol

    decile_stats.to_csv(os.path.join(RESULTS_DIR, f"{symbol}_vrp_deciles.csv"), index=False)

    # Chart: VRP Deciles
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = ["#2ecc71" if i < 3 else "#f39c12" if i < 7 else "#e74c3c"
              for i in range(10)]

    ax1.bar(range(len(decile_stats)), decile_stats["mean_vrp"],
            color=colors, edgecolor="black", lw=0.5)
    ax1.set_title(f"VRP por Decil — {symbol}")
    ax1.set_xlabel("Decil (0 = IV mais barata)")
    ax1.set_ylabel("VRP Médio (% anualizada)")
    ax1.axhline(0, color="gray", ls="--", lw=0.5)
    ax1.grid(alpha=0.3)

    ax2.bar(range(len(decile_stats)), decile_stats["mean_future_rv"],
            color=colors, edgecolor="black", lw=0.5, alpha=0.7)
    ax2.axhline(merged["rv_20d"].mean(), color="blue", ls="--",
                lw=0.5, label="RV Média Geral")
    ax2.set_title(f"RV Futura por Decil de VRP — {symbol}")
    ax2.set_xlabel("Decil (0 = IV mais barata)")
    ax2.set_ylabel("RV Futura 20d (% anualizada)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, f"{symbol}_vrp_deciles.png"), dpi=150)
    plt.close(fig)

    results = {
        "symbol": symbol,
        "decile_0_vrp": float(decile_stats.iloc[0]["mean_vrp"]),
        "decile_9_vrp": float(decile_stats.iloc[-1]["mean_vrp"]),
        "decile_0_future_rv": float(decile_stats.iloc[0]["mean_future_rv"]),
        "decile_9_future_rv": float(decile_stats.iloc[-1]["mean_future_rv"]),
        "monotonic": float(decile_stats["mean_future_rv"].is_monotonic_increasing),
    }

    return results


def study3_vrp_by_regime(atm_iv: pd.DataFrame,
                          underlying: pd.DataFrame,
                          symbol: str) -> dict:
    """Analyze VRP conditional on volatility regime."""
    merged = atm_iv.merge(
        underlying[["timestamp", "rv_20d", "garch_forecast_20d"]],
        on="timestamp", how="inner"
    )
    if merged.empty:
        return {"error": "No data"}

    merged["vrp"] = merged["iv"] - merged["garch_forecast_20d"]

    # Use underlying regime classification
    underlying_reg = underlying.dropna(subset=["rv_20d"]).copy()
    if len(underlying_reg) < 100:
        return {"error": "Insufficient data for regime classification"}

    states, _ = classify_vol_regime(underlying_reg["rv_20d"], n_states=4)
    underlying_reg = underlying_reg.iloc[:len(states)].copy()
    underlying_reg["regime"] = states

    regime_map = {0: "Low", 1: "Medium", 2: "High", 3: "Extreme"}
    underlying_reg["regime_label"] = underlying_reg["regime"].map(regime_map)

    # Merge regimes with VRP data
    reg_subset = underlying_reg[["timestamp", "regime_label"]]
    merged = merged.merge(reg_subset, on="timestamp", how="inner")

    regime_stats = merged.groupby("regime_label").agg(
        n=("vrp", "count"),
        mean_vrp=("vrp", "mean"),
        std_vrp=("vrp", "std"),
        mean_iv=("iv", "mean"),
        mean_rv=("rv_20d", "mean"),
    ).reset_index()

    regime_order = ["Low", "Medium", "High", "Extreme"]
    regime_stats["regime_label"] = pd.Categorical(
        regime_stats["regime_label"], categories=regime_order, ordered=True
    )
    regime_stats = regime_stats.sort_values("regime_label").reset_index(drop=True)

    regime_stats.to_csv(os.path.join(RESULTS_DIR, f"{symbol}_vrp_by_regime.csv"), index=False)

    # Chart: VRP by Regime
    fig, ax = plt.subplots(figsize=(10, 6))
    colors_bar = {"Low": "#2ecc71", "Medium": "#f1c40f",
                  "High": "#e67e22", "Extreme": "#e74c3c"}
    bar_colors = [colors_bar.get(r, "#95a5a6") for r in regime_stats["regime_label"]]

    x = np.arange(len(regime_stats))
    ax.bar(x, regime_stats["mean_vrp"], color=bar_colors, edgecolor="black", lw=0.5,
           yerr=regime_stats["std_vrp"], capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(regime_stats["regime_label"])
    ax.axhline(0, color="gray", ls="--", lw=0.5)
    ax.set_title(f"VRP Médio por Regime de Volatilidade — {symbol}")
    ax.set_ylabel("VRP Médio (% anualizada)")
    ax.grid(alpha=0.3, axis="y")

    # Add n labels
    for i, row in regime_stats.iterrows():
        ax.text(i, row["mean_vrp"] + 0.01, f"n={row['n']}",
                ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, f"{symbol}_vrp_by_regime.png"), dpi=150)
    plt.close(fig)

    results = {
        "symbol": symbol,
        "regime_low_vrp": float(regime_stats.loc[regime_stats["regime_label"] == "Low", "mean_vrp"].values[0])
        if "Low" in regime_stats["regime_label"].values else None,
        "regime_medium_vrp": float(regime_stats.loc[regime_stats["regime_label"] == "Medium", "mean_vrp"].values[0])
        if "Medium" in regime_stats["regime_label"].values else None,
        "regime_high_vrp": float(regime_stats.loc[regime_stats["regime_label"] == "High", "mean_vrp"].values[0])
        if "High" in regime_stats["regime_label"].values else None,
        "regime_extreme_vrp": float(regime_stats.loc[regime_stats["regime_label"] == "Extreme", "mean_vrp"].values[0])
        if "Extreme" in regime_stats["regime_label"].values else None,
    }

    return results


def study4_surface(options_iv: pd.DataFrame,
                   _underlying: pd.DataFrame,
                   symbol: str) -> dict:
    """Analyze IV across strikes (ATM/OTM/ITM) and compare with RV forecast."""
    if options_iv.empty or len(options_iv) < 10:
        return {"error": "Insufficient options data"}

    df = options_iv.copy()
    df["moneyness_bucket"] = pd.cut(
        df["moneyness"],
        bins=[0, 0.9, 0.95, 0.975, 1.0, 1.025, 1.05, 1.10, np.inf],
        labels=["Deep OTM P", "OTM P", "OTM P Near", "ATM",
                "OTM C Near", "OTM C", "Deep OTM C", "Far OTM C"],
        include_lowest=False,
    )

    surface = df.groupby("moneyness_bucket", observed=True).agg(
        n=("iv", "count"),
        mean_iv=("iv", "mean"),
        std_iv=("iv", "std"),
        mean_moneyness=("moneyness", "mean"),
        mean_dte=("dte", "mean"),
    ).reset_index()

    surface.to_csv(os.path.join(RESULTS_DIR, f"{symbol}_surface.csv"), index=False)

    # Chart: Volatility Smile
    fig, ax = plt.subplots(figsize=(10, 6))

    if "C" in df["option_type"].unique():
        calls = df[df["option_type"] == "C"].copy()
        calls["moneyness_bin"] = pd.cut(calls["moneyness"], bins=30)
        calls_grp = calls.groupby("moneyness_bin", observed=True)["iv"].agg(["mean", "std"]).reset_index()
        calls_grp["moneyness_mid"] = calls_grp["moneyness_bin"].apply(lambda x: x.mid)
        ax.errorbar(calls_grp["moneyness_mid"], calls_grp["mean"], yerr=calls_grp["std"],
                    fmt="o", color="green", label="Calls", alpha=0.7, capsize=3)

    if "P" in df["option_type"].unique():
        puts = df[df["option_type"] == "P"].copy()
        puts["moneyness_bin"] = pd.cut(puts["moneyness"], bins=30)
        puts_grp = puts.groupby("moneyness_bin", observed=True)["iv"].agg(["mean", "std"]).reset_index()
        puts_grp["moneyness_mid"] = puts_grp["moneyness_bin"].apply(lambda x: x.mid)
        ax.errorbar(puts_grp["moneyness_mid"], puts_grp["mean"], yerr=puts_grp["std"],
                    fmt="s", color="red", label="Puts", alpha=0.7, capsize=3)

    ax.axvline(1.0, color="gray", ls="--", lw=0.5, alpha=0.5)
    ax.set_xlabel("Moneyness (K/S)")
    ax.set_ylabel("IV Médio (% anualizada)")
    ax.set_title(f"Volatilidade Smile — {symbol}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, f"{symbol}_surface.png"), dpi=150)
    plt.close(fig)

    atm_mean = surface.loc[surface["moneyness_bucket"] == "ATM", "mean_iv"].values
    otm_put_mean = surface.loc[
        surface["moneyness_bucket"].isin(["OTM P", "OTM P Near"]), "mean_iv"
    ].mean() if "OTM P" in surface["moneyness_bucket"].values else np.nan
    otm_call_mean = surface.loc[
        surface["moneyness_bucket"].isin(["OTM C", "OTM C Near"]), "mean_iv"
    ].mean() if "OTM C" in surface["moneyness_bucket"].values else np.nan

    results = {
        "symbol": symbol,
        "atm_iv": float(atm_mean[0]) if len(atm_mean) > 0 else np.nan,
        "otm_put_iv": float(otm_put_mean) if not np.isnan(otm_put_mean) else np.nan,
        "otm_call_iv": float(otm_call_mean) if not np.isnan(otm_call_mean) else np.nan,
        "put_skew": float(otm_put_mean - atm_mean[0])
        if len(atm_mean) > 0 and not np.isnan(otm_put_mean) else np.nan,
        "call_skew": float(otm_call_mean - atm_mean[0])
        if len(atm_mean) > 0 and not np.isnan(otm_call_mean) else np.nan,
        "n_options": len(df),
    }

    return results


def study5_straddle_backtest(atm_iv: pd.DataFrame,
                              options_iv: pd.DataFrame,
                              underlying: pd.DataFrame,
                              symbol: str,
                              vrp_threshold: float = 0.20,
                              cost_per_leg: float = 0.005) -> dict:
    """Backtest short ATM straddle when VRP > threshold.

    Entrada: vender 1 call + 1 put ATM (cada lado no seu strike mais próximo)
    no fechamento. Pode ter strikes distintos (realidade B3: nem sempre
    call+put no mesmo strike são líquidos simultaneamente).
    Saída: no próximo fechamento disponível.
    P&L = (preço_call_entrada + preço_put_entrada)
        - (preço_call_saída + preço_put_saída)
    """
    if atm_iv.empty or options_iv.empty:
        return {"error": "No data for straddle backtest"}

    merged = atm_iv.merge(
        underlying[["timestamp", "rv_20d", "garch_forecast_20d"]],
        on="timestamp", how="inner"
    )
    if merged.empty:
        return {"error": "No merged data"}
    merged["vrp"] = merged["iv"] - merged["garch_forecast_20d"]

    trades = []
    timestamps = sorted(merged["timestamp"].unique())
    straddle_open = None

    for ts in timestamps:
        vrp = merged[merged["timestamp"] == ts].iloc[0]["vrp"]

        day_opts = options_iv[options_iv["timestamp"] == ts].copy()
        if day_opts.empty:
            continue

        # Find closest ATM call and closest ATM put (independent strikes)
        calls = day_opts[day_opts["option_type"] == "C"].copy()
        puts = day_opts[day_opts["option_type"] == "P"].copy()

        # B3 reality: often only puts trade ATM. Use whatever is available.
        has_both = not calls.empty and not puts.empty
        has_put = not puts.empty
        has_call = not calls.empty

        if not has_put and not has_call:
            continue

        call_price = 0.0
        put_price = 0.0
        call_symbol = ""
        put_symbol = ""
        call_strike = 0.0
        put_strike = 0.0

        if has_call:
            calls["dist"] = np.abs(calls["moneyness"] - 1.0)
            best_call = calls.loc[calls["dist"].idxmin()]
            call_price = best_call["close_price"]
            call_symbol = best_call.get("option_symbol", "")
            call_strike = best_call["strike"]

        if has_put:
            puts["dist"] = np.abs(puts["moneyness"] - 1.0)
            best_put = puts.loc[puts["dist"].idxmin()]
            put_price = best_put["close_price"]
            put_symbol = best_put.get("option_symbol", "")
            put_strike = best_put["strike"]

        # Close existing position using today's best available prices
        if straddle_open is not None:
            entry_val = straddle_open["entry_call"] + straddle_open["entry_put"]
            exit_val = call_price + put_price
            gross_pnl = entry_val - exit_val
            cost = (entry_val + exit_val) * cost_per_leg
            net_pnl = gross_pnl - cost
            hold_days = (ts - straddle_open["entry_ts"]).days

            trades.append({
                "entry_ts": straddle_open["entry_ts"],
                "exit_ts": ts,
                "entry_call": straddle_open["entry_call"],
                "entry_put": straddle_open["entry_put"],
                "exit_call": call_price,
                "exit_put": put_price,
                "gross_pnl": gross_pnl,
                "cost": cost,
                "net_pnl": net_pnl,
                "hold_days": hold_days,
                "entry_vrp": straddle_open["entry_vrp"],
                "entry_legs": straddle_open["legs"],
                "call_strike": straddle_open["call_strike"],
                "put_strike": straddle_open["put_strike"],
                "call_symbol": straddle_open["call_symbol"],
                "put_symbol": straddle_open["put_symbol"],
            })

        # Open new position if VRP > threshold
        if vrp > vrp_threshold:
            legs = "straddle" if has_both else ("call" if has_call else "put")
            straddle_open = {
                "entry_ts": ts,
                "entry_call": call_price,
                "entry_put": put_price,
                "entry_vrp": vrp,
                "legs": legs,
                "call_strike": call_strike,
                "put_strike": put_strike,
                "call_symbol": call_symbol,
                "put_symbol": put_symbol,
            }
        else:
            straddle_open = None

    # Close any remaining position at last observation
    if straddle_open is not None and len(trades) > 0:
        last = trades[-1]
        entry_val = straddle_open["entry_call"] + straddle_open["entry_put"]
        exit_val = last["exit_call"] + last["exit_put"]
        gross_pnl = entry_val - exit_val
        cost = (entry_val + exit_val) * cost_per_leg
        net_pnl = gross_pnl - cost
        hold_days = (last["exit_ts"] - straddle_open["entry_ts"]).days
        trades.append({
            "entry_ts": straddle_open["entry_ts"],
            "exit_ts": last["exit_ts"],
            "entry_call": straddle_open["entry_call"],
            "entry_put": straddle_open["entry_put"],
            "exit_call": last["exit_call"],
            "exit_put": last["exit_put"],
            "gross_pnl": gross_pnl,
            "cost": cost,
            "net_pnl": net_pnl,
            "hold_days": hold_days,
            "entry_vrp": straddle_open["entry_vrp"],
            "entry_legs": straddle_open["legs"],
            "call_strike": straddle_open["call_strike"],
            "put_strike": straddle_open["put_strike"],
            "call_symbol": straddle_open["call_symbol"],
            "put_symbol": straddle_open["put_symbol"],
        })

    if not trades:
        return {"error": "No straddle trades executed"}

    trade_df = pd.DataFrame(trades)
    n_trades = len(trade_df)
    total_gross = trade_df["gross_pnl"].sum()
    total_cost = trade_df["cost"].sum()
    total_net = trade_df["net_pnl"].sum()
    avg_hold = trade_df["hold_days"].mean()
    win_rate = (trade_df["net_pnl"] > 0).mean()

    # Sharpe ratio from daily P&L
    trade_df["daily_return"] = trade_df["net_pnl"] / (
        trade_df["entry_call"] + trade_df["entry_put"]
    )
    daily_returns = []
    for _, t in trade_df.iterrows():
        daily_returns.extend([t["daily_return"]] * max(1, t["hold_days"]))
    daily_returns = np.array(daily_returns)
    sharpe = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else 0

    # Max drawdown on cumulative P&L
    cum_pnl = trade_df["net_pnl"].cumsum()
    running_max = cum_pnl.cummax()
    drawdown = cum_pnl - running_max
    max_dd = drawdown.min()

    trade_df.to_csv(os.path.join(RESULTS_DIR, f"{symbol}_straddle_trades.csv"), index=False)

    # Chart: cumulative P&L
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(range(len(trade_df)), cum_pnl, color="navy", lw=1.5)
    ax.fill_between(range(len(trade_df)), cum_pnl, 0,
                    where=cum_pnl >= 0, color="green", alpha=0.1)
    ax.fill_between(range(len(trade_df)), cum_pnl, 0,
                    where=cum_pnl < 0, color="red", alpha=0.1)
    ax.set_title(f"P&L Acumulado Straddle Curto — {symbol} (VRP > {vrp_threshold*100:.0f}%)")
    ax.set_xlabel("Trade #")
    ax.set_ylabel("P&L (R$)")
    ax.axhline(0, color="gray", ls="--", lw=0.5)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, f"{symbol}_straddle_pnl.png"), dpi=150)
    plt.close(fig)

    results = {
        "n_trades": n_trades,
        "total_gross_pnl": float(total_gross),
        "total_cost": float(total_cost),
        "total_net_pnl": float(total_net),
        "avg_hold_days": float(avg_hold) if not np.isnan(avg_hold) else 0.0,
        "win_rate": float(win_rate),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "avg_return_per_trade": float(total_net / n_trades) if n_trades > 0 else 0.0,
        "vrp_threshold": vrp_threshold,
        "cost_per_leg": cost_per_leg,
    }
    return results


def study_forecast_comparison(ivx_df: pd.DataFrame,
                               underlying: pd.DataFrame,
                               symbol: str) -> dict:
    """Compare IVX vs GARCH forecast vs future realized RV 20d.

    This is the decisive test: does the GARCH model add value beyond
    the market's implied vol (IVX)?
    """
    if ivx_df.empty:
        return {"error": "No IVX data"}

    merged = ivx_df.merge(
        underlying[["timestamp", "rv_20d", "garch_forecast_20d"]],
        on="timestamp", how="inner"
    )
    if merged.empty:
        return {"error": "No merged data"}

    # Future RV: realized vol over the NEXT 20 days
    rv_series = underlying.set_index("timestamp")["rv_20d"]
    merged["future_rv_20d"] = merged["timestamp"].map(
        lambda t: rv_series.shift(-RV_WINDOW).reindex(rv_series.index).loc[t]
        if t in rv_series.index else np.nan
    )
    merged = merged.dropna(subset=["future_rv_20d", "garch_forecast_20d"])
    merged.to_csv(os.path.join(RESULTS_DIR, f"{symbol}_forecast_comparison.csv"), index=False)

    if len(merged) < 10:
        return {"error": f"Insufficient data ({len(merged)} obs)"}

    y_true = merged["future_rv_20d"].values
    y_ivx = merged["iv"].values
    y_garch = merged["garch_forecast_20d"].values
    y_comb = merged[["iv", "garch_forecast_20d"]].mean(axis=1).values

    from scipy.stats import spearmanr

    def mae(a, b):
        return np.mean(np.abs(a - b))

    def rmse(a, b):
        return np.sqrt(np.mean((a - b) ** 2))

    def spearman_corr(a, b):
        r, _ = spearmanr(a, b)
        return r

    def pearson_corr(a, b):
        return np.corrcoef(a, b)[0, 1]

    results = {
        "symbol": symbol,
        "n_obs": len(merged),
        "ivx_mae": float(mae(y_ivx, y_true)),
        "garch_mae": float(mae(y_garch, y_true)),
        "combined_mae": float(mae(y_comb, y_true)),
        "ivx_rmse": float(rmse(y_ivx, y_true)),
        "garch_rmse": float(rmse(y_garch, y_true)),
        "combined_rmse": float(rmse(y_comb, y_true)),
        "ivx_spearman": float(spearman_corr(y_ivx, y_true)),
        "garch_spearman": float(spearman_corr(y_garch, y_true)),
        "combined_spearman": float(spearman_corr(y_comb, y_true)),
        "ivx_pearson": float(pearson_corr(y_ivx, y_true)),
        "garch_pearson": float(pearson_corr(y_garch, y_true)),
        "combined_pearson": float(pearson_corr(y_comb, y_true)),
    }

    # Chart: scatter of forecasted vs actual
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, y_pred, label, color in zip(
        axes,
        [y_ivx, y_garch, y_comb],
        ["IVX", "GARCH(1,1)", "Média IVX+GARCH"],
        ["red", "blue", "green"], strict=False
    ):
        ax.scatter(y_pred * 100, y_true * 100, alpha=0.4, s=10, c=color)
        lims = [min(y_pred.min(), y_true.min()) * 100,
                max(y_pred.max(), y_true.max()) * 100]
        ax.plot(lims, lims, "k--", lw=0.5, alpha=0.5)
        ax.set_xlabel(f"{label} (%)")
        ax.set_ylabel("RV Futura 20d (%)")
        ax.set_title(f"{label}")
        ax.grid(alpha=0.3)
    fig.suptitle(f"Previsão vs Realizado — {symbol}", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, f"{symbol}_forecast_comparison.png"), dpi=150)
    plt.close(fig)

    return results


# ═══════════════════════════════════════════════════════════════════════
#  Report Generation
# ═══════════════════════════════════════════════════════════════════════

def generate_report(all_results: dict[str, dict]) -> str:
    lines = [
        "# Motor de Volatilidade para Opções B3 — Relatório",
        f"**Gerado em:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Ativos:** {', '.join(SYMBOLS)}",
        "**Frequência:** Diária (D1)",
        f"**Taxa Livre de Risco:** {RISK_FREE_RATE:.2%} (SELIC)",
        f"**Horizonte RV/GARCH:** {RV_WINDOW} dias",
        "",
        "---",
        "",
        "## Dados",
        "",
        "| Fonte | Descrição |",
        "|---|---|",
        "| **MT5 D1 Parquet** | OHLCV diário dos subjacentes (2021-2026, 1252 dias) |",
        "| **IVX (B3)** | Índice de volatilidade implícita B3 (252 dias, 2025-2026) |",
        "| **EWMAB3 (B3)** | Estimativa de volatilidade realizada B3 via EWMA |",
        "| **OpLab API (snapshot)** | Chain completa de opções com IV calculado via Black-Scholes |",
        "| **CSVs históricos** | Preços diários de opções extraídos via MT5 (2025-2026) |",
        "",
        "### Filtros Aplicados",
        "",
        "- **VRP:** IVX (índice B3) como IV ATM primário, GARCH(1,1) como forecast RV",
        "- **Opções (superfície/straddle):** IV ∈ [0.05, 1.50], volume > 0, moneyness [0.95, 1.05], DTE [20, 45]",
        "",
        "---",
        "",
        "## Estudo 1 — Forecast RV vs IV (VRP Analysis)",
        "",
        "Comparação entre o índice de volatilidade implícita IVX (B3) e o forecast "
        "GARCH(1,1). VRP = IVX - GARCH_Forecast.",
        "",
    ]

    # Study 1 table
    s1 = {s: r.get("study1", {}) for s, r in all_results.items()
          if "error" not in r.get("study1", {})}
    if s1:
        lines.append(f"| {'Métrica':30s} | " + " | ".join(f"{s:>14s}" for s in s1) + " |")
        lines.append("|" + "|".join("-" * 32 for _ in range(len(s1) + 1)) + "|")

        metrics = ["mean_iv", "mean_garch_forecast", "mean_rv",
                    "mean_vrp_iv_garch", "vrp_positive_pct",
                    "long_vol_mean_return", "short_vol_mean_return",
                    "signal_hit_rate", "n_obs"]
        labels = ["IV ATM Média (%)", "GARCH Forecast Média (%)",
                   "RV 20d Média (%)", "VRP (IV - GARCH) Médio (%)",
                   "VRP Positivo (% obs)", "Retorno Long Vol Médio (%)",
                   "Retorno Short Vol Médio (%)", "Hit Rate do Sinal (%)",
                   "N Obs"]
        for met, lbl in zip(metrics, labels, strict=False):
            vals = []
            for s_name in s1:
                v = s1[s_name].get(met, 0)
                if met in ("vrp_positive_pct",):
                    vals.append(f"{v:>13.1f}")
                elif met in ("n_obs",):
                    vals.append(f"{v:>14.0f}")
                else:
                    vals.append(f"{v*100:>13.2f}" if isinstance(v, float) and abs(v) < 10 else f"{v:>14.4f}")
            lines.append(f"| {lbl:30s} | " + " | ".join(vals) + " |")

        lines.append("")
    lines.append("")

    # OpLab Snapshot Summary
    lines.extend([
        "",
        "## OpLab Snapshot — Chain Atual",
        "",
        "Dados obtidos via API OpLab (plano PRO, REAL_TIME). "
        "IV calculado via Black-Scholes com taxa SELIC 14.75%.",
        "",
    ])
    lines.append(f"| {'Métrica':25s} | " + " | ".join(f"{s:>14s}" for s in s1) + " |")
    lines.append("|" + "|".join("---" for _ in range(len(s1) + 1)) + "|")
    for met, lbl in [("n_options", "Total Opções na Chain"),
                      ("n_valid_iv", "Com IV Válido"),
                      ]:
        vals = [f"{lbl:25s}"]
        for s_name in s1:
            opl = all_results[s_name].get("oplab", {})
            v = opl.get(met, 0)
            vals.append(f"{v:>14.0f}")
        lines.append("| " + " | ".join(vals) + " |")
    lines.append("")

    # Study 2: Deciles table
    lines.extend([
        "---",
        "",
        "## Estudo 2 — Decis de VRP",
        "",
        "Separa o VRP em 10 decis e mede a RV futura em cada decil.",
        "",
    ])

    decile_dfs = {}
    for s_name in s1:
        csv_path = os.path.join(RESULTS_DIR, f"{s_name}_vrp_deciles.csv")
        if os.path.exists(csv_path):
            decile_dfs[s_name] = pd.read_csv(csv_path)

    lines.append(f"| {'Decil':8s} | " + " | ".join(f"{s:>22s}" for s in s1) + " |")
    lines.append("|" + "|".join("---" for _ in range(len(s1) + 1)) + "|")
    for d in range(10):
        vals = [f"{d:>8d}"]
        for s_name in s1:
            dd = decile_dfs.get(s_name)
            if dd is not None and d in dd["decile"].values:
                row = dd[dd["decile"] == d].iloc[0]
                vals.append(f"VRP {row['mean_vrp']*100:>5.1f}% / RV {row['mean_future_rv']*100:>5.1f}%")
            else:
                vals.append(f"{'N/A':>22s}")
        lines.append("| " + " | ".join(vals) + " |")
    lines.append("")

    for s_name in s1:
        dd = decile_dfs.get(s_name)
        if dd is not None:
            d0_vrp = dd.iloc[0]["mean_vrp"] * 100
            d9_vrp = dd.iloc[-1]["mean_vrp"] * 100
            d0_rv = dd.iloc[0]["mean_future_rv"] * 100
            d9_rv = dd.iloc[-1]["mean_future_rv"] * 100
            lines.append(
                f"- **{s_name}**: Decil 0 VRP = {d0_vrp:.2f}% → "
                f"RV futura = {d0_rv:.2f}% | "
                f"Decil 9 VRP = {d9_vrp:.2f}% → "
                f"RV futura = {d9_rv:.2f}%"
            )
    lines.append("")

    # Study 3: VRP by Regime
    lines.extend([
        "---",
        "",
        "## Estudo 3 — VRP por Regime de Volatilidade",
        "",
        "VRP médio condicional ao regime de volatilidade (HMM 4 estados).",
        "",
    ])

    lines.append(f"| {'Regime':12s} | " + " | ".join(f"{s:>14s}" for s in s1) + " |")
    lines.append("|" + "|".join("-" * 14 for _ in range(len(s1) + 1)) + "|")
    for regime in ["Low", "Medium", "High", "Extreme"]:
        vals = [f"{regime:12s}"]
        for s_name in s1:
            s3 = all_results[s_name].get("study3", {})
            key = f"regime_{regime.lower()}_vrp"
            v = s3.get(key)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                vals.append(f"{'N/A':>14s}")
            else:
                vals.append(f"{v*100:>13.2f}")
        lines.append("| " + " | ".join(vals) + " |")
    lines.append("")

    # Study 4: Surface
    lines.extend([
        "---",
        "",
        "## Estudo 4 — Superfície de Volatilidade",
        "",
        "Comparação da IV por moneyness (ATM, OTM Put, OTM Call).",
        "",
    ])

    lines.append(f"| {'Métrica':20s} | " + " | ".join(f"{s:>14s}" for s in s1) + " |")
    lines.append("|" + "|".join("-" * 22 for _ in range(len(s1) + 1)) + "|")
    for met, lbl in [("atm_iv", "ATM IV Média (%)"),
                      ("otm_put_iv", "OTM Put IV Média (%)"),
                      ("otm_call_iv", "OTM Call IV Média (%)"),
                      ("put_skew", "Put Skew (pct points)"),
                      ("call_skew", "Call Skew (pct points)")]:
        vals = [f"{lbl:20s}"]
        for s_name in s1:
            s4 = all_results[s_name].get("study4", {})
            v = s4.get(met, np.nan)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                vals.append(f"{'N/A':>14s}")
            else:
                vals.append(f"{v*100:>13.2f}")
        lines.append("| " + " | ".join(vals) + " |")
    lines.append("")

    # Conclusions
    # Study 5: Straddle Backtest table
    s5 = {s: r.get("study5", {}) for s, r in all_results.items()
          if "error" not in r.get("study5", {})}
    if s5:
        lines.extend([
            "",
            "---",
            "",
            "## Estudo 5 — Straddle Backtest (VRP > 20%)",
            "",
            "Estratégia: vender opções ATM (idealmente straddle 1 call + 1 put, "
            "mas na B3 apenas puts têm liquidez, então o backtest usa puts como proxy) "
            "quando VRP > 20%. Rebalanceamento diário. "
            f"Custo por perna: {next(iter(s5.values())).get('cost_per_leg', 0.005)*100:.1f}%.",
            "",
        ])

        lines.append(f"| {'Métrica':22s} | " + " | ".join(f"{s:>14s}" for s in s5) + " |")
        lines.append("|" + "|".join("-" * 24 for _ in range(len(s5) + 1)) + "|")
        s5_metrics = [
            ("n_trades", "N Trades", "{:>14.0f}"),
            ("total_net_pnl", "P&L Líquido (R$)", "{:>14.2f}"),
            ("total_cost", "Custo Total (R$)", "{:>14.2f}"),
            ("sharpe", "Sharpe", "{:>14.2f}"),
            ("max_drawdown", "Max Drawdown (R$)", "{:>14.2f}"),
            ("win_rate", "Win Rate (%)", "{:>14.1f}"),
            ("avg_hold_days", "Hold Médio (dias)", "{:>14.0f}"),
            ("avg_return_per_trade", "Retorno Médio/Trade (R$)", "{:>14.2f}"),
        ]
        for met, lbl, fmt in s5_metrics:
            vals = [f"{lbl:22s}"]
            for s_name in s5:
                v = s5[s_name].get(met, 0)
                if met == "win_rate":
                    vals.append(f"{v*100:>14.1f}")
                else:
                    vals.append(fmt.format(v))
            lines.append("| " + " | ".join(vals) + " |")
        lines.append("")

    # Forecast Comparison table
    fc = {s: r["forecast_comp"] for s, r in all_results.items()
          if "forecast_comp" in r and "error" not in r["forecast_comp"]}
    if fc:
        lines.extend([
            "",
            "---",
            "",
            "## Comparação de Forecasts — IVX vs GARCH vs RV Futura",
            "",
            "Teste decisivo: o GARCH(1,1) agrega valor além da volatilidade implícita do mercado (IVX)? "
            "A tabela abaixo compara cada modelo contra a volatilidade realizada nos 20 dias seguintes.",
            "",
        ])

        lines.append(f"| {'Métrica':20s} | " + " | ".join(f"{s:>14s}" for s in fc) + " |")
        lines.append("|" + "|".join("-" * 22 for _ in range(len(fc) + 1)) + "|")

        fc_metrics = [
            ("n_obs", "N Obs", "{:>14.0f}"),
            ("ivx_mae", "IVX - MAE (%)", "{:>14.2f}"),
            ("garch_mae", "GARCH - MAE (%)", "{:>14.2f}"),
            ("combined_mae", "Média - MAE (%)", "{:>14.2f}"),
            ("ivx_rmse", "IVX - RMSE (%)", "{:>14.2f}"),
            ("garch_rmse", "GARCH - RMSE (%)", "{:>14.2f}"),
            ("combined_rmse", "Média - RMSE (%)", "{:>14.2f}"),
            ("ivx_spearman", "IVX - Spearman", "{:>14.3f}"),
            ("garch_spearman", "GARCH - Spearman", "{:>14.3f}"),
            ("ivx_pearson", "IVX - Pearson", "{:>14.3f}"),
            ("garch_pearson", "GARCH - Pearson", "{:>14.3f}"),
        ]
        for met, lbl, fmt in fc_metrics:
            vals = [f"{lbl:20s}"]
            for s_name in fc:
                v = fc[s_name].get(met, 0)
                if "mae" in met or "rmse" in met:
                    vals.append(f"{v*100:>13.2f}")
                else:
                    vals.append(fmt.format(v))
            lines.append("| " + " | ".join(vals) + " |")
        lines.append("")

        # Interpretation
        lines.append("### Interpretação")
        lines.append("")
        for s_name in fc:
            f = fc[s_name]
            ivx_mae = f["ivx_mae"] * 100
            garch_mae = f["garch_mae"] * 100
            diff = garch_mae - ivx_mae
            better = "GARCH" if diff < 0 else "IVX"
            lines.append(
                f"- **{s_name}**: IVX MAE={ivx_mae:.2f}%, GARCH MAE={garch_mae:.2f}% → "
                f"{better} é mais preciso ({abs(diff):.2f}pp de diferença)"
            )
        lines.append("")

    # Conclusions
    lines.extend([
        "---",
        "",
        "## Conclusões",
        "",
    ])

    # Group symbols that have VRP studies vs surface-only
    vrp_symbols = {s for s in s1}
    {s for s in all_results if s not in vrp_symbols
                    and "error" not in all_results[s].get("study4", {})}

    for s_name in sorted(all_results.keys(), key=lambda x: (x not in vrp_symbols, x)):
        sym_r = all_results[s_name]
        if "error" in sym_r:
            lines.append(f"### {s_name} — {sym_r['error']}")
            lines.append("")
            continue

        s1res = sym_r.get("study1", {})
        s3res = sym_r.get("study3", {})
        s4res = sym_r.get("study4", {})
        s5res = sym_r.get("study5", {})

        has_vrp = "error" not in s1res and s1res
        has_surface = "error" not in s4res

        lines.append(f"### {s_name}")
        lines.append("")

        if has_vrp:
            lines.append(f"- **VRP (IVX - GARCH):** {s1res.get('mean_vrp_iv_garch', 0)*100:.2f}% "
                         f"(positivo em {s1res.get('vrp_positive_pct', 0):.0f}% das {s1res.get('n_obs', 0)} observações IVX)")
            lines.append(f"- **IVX Médio:** {s1res.get('mean_iv', 0)*100:.2f}% | "
                         f"**GARCH Forecast:** {s1res.get('mean_garch_forecast', 0)*100:.2f}% | "
                         f"**RV 20d:** {s1res.get('mean_rv', 0)*100:.2f}%")
            lines.append(f"- **Short Vol (VRP > 5%):** "
                         f"retorno médio = {s1res.get('short_vol_mean_return', 0)*100:.2f}%")
            lines.append(f"- **Long Vol (VRP < -5%):** "
                         f"retorno médio = {s1res.get('long_vol_mean_return', 0)*100:.2f}%")
            lines.append(f"- **Hit Rate do Sinal:** {s1res.get('signal_hit_rate', 0)*100:.1f}%")

            if "error" not in s3res:
                lines.append("- **VRP por Regime:**")
                for r in ["Low", "Medium", "High", "Extreme"]:
                    v = s3res.get(f"regime_{r.lower()}_vrp")
                    if v is not None and not (isinstance(v, float) and np.isnan(v)):
                        lines.append(f"  - {r}: {v*100:.2f}%")

            if "study5" in sym_r and "error" not in s5res and "n_trades" in s5res:
                lines.append("- **Straddle Backtest (VRP > 20%):**")
                lines.append(f"  - Trades: {s5res['n_trades']}")
                lines.append(f"  - P&L Líquido: R$ {s5res['total_net_pnl']:.2f}")
                lines.append(f"  - Sharpe: {s5res['sharpe']:.2f}")
                lines.append(f"  - Drawdown Máx: R$ {s5res['max_drawdown']:.2f}")
                lines.append(f"  - Win Rate: {s5res['win_rate']*100:.1f}%")
                lines.append(f"  - Hold Médio: {s5res['avg_hold_days']:.0f} dias")
        elif has_surface:
            lines.append("- **Dados disponíveis apenas para superfície de volatilidade (OpLab snapshot).**")
            lines.append(f"  - ATM IV: {s4res.get('atm_iv', 0)*100:.1f}%")
            lines.append(f"  - Put Skew: {s4res.get('put_skew', 0)*100:.1f}pp")
            lines.append(f"  - Call Skew: {s4res.get('call_skew', 0)*100:.1f}pp")

        lines.append("")

    lines.extend([
        "",
        "### Nota Metodológica",
        "",
        "Estudos 1-3 usam o índice IVX (B3) como proxy de volatilidade implícita ATM "
        "e GARCH(1,1) para forecast de volatilidade realizada, com 252 observações diárias. "
        "Estudos 4-5 usam dados de opções individuais com filtros: IV entre 5% e 150%, "
        "volume > 0, moneyness entre 0,95 e 1,05 (ATM), DTE entre 20 e 45 dias. "
        "O Estudo 4 (superfície) mantém todos os níveis de moneyness.",
        "",
        "### Disponibilidade de Dados",
        "",
        "| Ativo | Histórico (CSV) | OpLab Snapshot | Estudos |",
        "|---|---|---|---|",
    ])
    for s_name in sorted(all_results.keys()):
        has_ivx = "study1" in all_results[s_name] and "error" not in all_results[s_name].get("study1", {})
        has_csv_opts = ("study5" in all_results[s_name] and "error" not in all_results[s_name].get("study5", {})
                        and all_results[s_name]["study5"].get("n_trades", 0) > 0)
        csv_cell = "IVX + CSVs" if has_ivx and has_csv_opts else "IVX (252 dias)" if has_ivx else "—"
        oplab_cell = f"{all_results[s_name].get('oplab', {}).get('n_options', 0)} opções" if all_results[s_name].get("oplab", {}).get("n_options", 0) > 0 else "—"
        surface_only = "study1" not in all_results[s_name] or "error" in all_results[s_name].get("study1", {})
        studies_cell = "1-5 (VRP + Straddle)" if not surface_only else "4 (Superfície)"
        lines.append(f"| {s_name:8s} | {csv_cell:20s} | {oplab_cell:18s} | {studies_cell:20s} |")
    lines.append("")

    lines.extend([
        "",
        "---",
        "*Relatório gerado automaticamente por experiment.py*",
    ])

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def run_study():
    results = {}

    for symbol in SYMBOLS:
        print(f"\n{'='*60}")
        print(f"Processing {symbol}...")
        print(f"{'='*60}")

        # 1. Load underlying daily data
        print("  Loading underlying daily data...")
        try:
            underlying = load_daily_underlying(symbol)
        except (FileNotFoundError, Exception) as e:
            print(f"  SKIP: No underlying data for {symbol}: {e}")
            results[symbol] = {"error": f"No underlying data: {e}"}
            continue
        underlying = compute_log_returns(underlying)
        underlying = compute_realized_vol(underlying)
        print(f"  {len(underlying)} days loaded ({underlying['timestamp'].min().date()} to "
              f"{underlying['timestamp'].max().date()})")

        # 2. GARCH forecast
        print("  Computing GARCH(1,1) forecast...")
        underlying = compute_garch_forecast_series(underlying)
        n_garch = underlying["garch_forecast_20d"].notna().sum()
        print(f"  {n_garch} GARCH forecasts computed")

        # 3a. Load IVX (implied volatility index) data
        print("  Loading IVX implied volatility index...")
        ivx_df = load_ivx_data(symbol)
        print(f"  {len(ivx_df)} IVX observations")

        # 3b. Load EWMAB3 (B3 realized vol estimate)
        ewma_df = load_ewma_data(symbol)
        print(f"  {len(ewma_df)} EWMA realized vol observations")

        # 3c. Load historical options data (saved CSVs)
        print("  Loading historical options data (CSVs)...")
        options_ts = load_option_ts(symbol)
        print(f"  {len(options_ts)} historical option-day observations")

        # 3d. Load OpLab snapshot
        print("  Loading OpLab snapshot data...")
        oplab_df = load_oplab_options(symbol)
        print(f"  {len(oplab_df)} OpLab options loaded (snapshot)")

        sym_results = {}
        sym_results["oplab"] = {
            "n_options": len(oplab_df),
            "n_valid_iv": int(oplab_df["iv"].notna().sum()) if not oplab_df.empty else 0,
        }

        # VRP Studies: use IVX as primary ATM IV (252 obs vs 17 from options)
        if not ivx_df.empty:
            print(f"  Using IVX as primary ATM IV source ({len(ivx_df)} obs)")

            # Study 1
            print("  Study 1: Forecast RV vs IV (IVX-based)...")
            sym_results["study1"] = study1_forecast_vs_iv(ivx_df, underlying, symbol)
            if "error" in sym_results["study1"]:
                print(f"    {sym_results['study1']['error']}")
            else:
                print(f"    IVX VRP = {sym_results['study1']['mean_vrp_iv_garch']*100:.2f}%")

            # Study 2
            print("  Study 2: VRP Deciles...")
            sym_results["study2"] = study2_vrp_deciles(ivx_df, underlying, symbol)
            if "error" in sym_results["study2"]:
                print(f"    {sym_results['study2']['error']}")

            # Study 3
            print("  Study 3: VRP by Regime...")
            sym_results["study3"] = study3_vrp_by_regime(ivx_df, underlying, symbol)
            if "error" in sym_results["study3"]:
                print(f"    {sym_results['study3']['error']}")

            # Forecast comparison: IVX vs GARCH vs RV Futura
            print("  Forecast Comparison: IVX vs GARCH vs RV Futura...")
            sym_results["forecast_comp"] = study_forecast_comparison(ivx_df, underlying, symbol)
            if "error" in sym_results["forecast_comp"]:
                print(f"    {sym_results['forecast_comp']['error']}")
            else:
                fc = sym_results["forecast_comp"]
                print(f"    IVX MAE={fc['ivx_mae']*100:.2f}%  GARCH MAE={fc['garch_mae']*100:.2f}%")
        elif not options_ts.empty:
            # Fallback: use options-based ATM IV
            print("  Computing IVs from historical options (fallback)...")
            options_iv = compute_daily_iv(options_ts, underlying)
            print(f"  {len(options_iv)} valid IVs from historical data")
            if not options_iv.empty:
                atm_iv = get_atm_iv(options_iv)
                print(f"  {len(atm_iv)} ATM IV observations")

                print("  Study 1: Forecast RV vs IV (options-based)...")
                sym_results["study1"] = study1_forecast_vs_iv(atm_iv, underlying, symbol)
                if "error" in sym_results["study1"]:
                    print(f"    {sym_results['study1']['error']}")
                else:
                    print(f"    VRP = {sym_results['study1']['mean_vrp_iv_garch']*100:.2f}%")

                print("  Study 2: VRP Deciles...")
                sym_results["study2"] = study2_vrp_deciles(atm_iv, underlying, symbol)
                if "error" in sym_results["study2"]:
                    print(f"    {sym_results['study2']['error']}")

                print("  Study 3: VRP by Regime...")
                sym_results["study3"] = study3_vrp_by_regime(atm_iv, underlying, symbol)
                if "error" in sym_results["study3"]:
                    print(f"    {sym_results['study3']['error']}")

        # Study 5: Straddle backtest (uses options data, not IVX)
        if not options_ts.empty:
            print("  Computing IVs from historical options for straddle backtest...")
            options_iv = compute_daily_iv(options_ts, underlying)
            if not options_iv.empty:
                atm_iv = get_atm_iv(options_iv)
                if len(atm_iv) > 0:
                    print("  Study 5: Straddle Backtest (VRP > 20%)...")
                    sym_results["study5"] = study5_straddle_backtest(atm_iv, options_iv, underlying, symbol)
                    if "error" in sym_results["study5"]:
                        print(f"    {sym_results['study5']['error']}")
                    else:
                        print(f"    {sym_results['study5']['n_trades']} trades, "
                              f"Sharpe={sym_results['study5']['sharpe']:.2f}, "
                              f"P&L={sym_results['study5']['total_net_pnl']:.2f}")

        # Study 4: Surface — uses OpLab snapshot for all symbols
        print("  Study 4: Vol Surface (OpLab snapshot)...")
        combined_iv = oplab_df[oplab_df["iv"].notna()].copy() if not oplab_df.empty else pd.DataFrame()
        if not combined_iv.empty:
            sym_results["study4"] = study4_surface(combined_iv, underlying, symbol)
        else:
            sym_results["study4"] = {"error": "No OpLab IV data"}
        if "error" in sym_results.get("study4", {}):
            print(f"    {sym_results['study4']['error']}")

        results[symbol] = sym_results

    # Generate report
    print("\nGenerating report...")
    report = generate_report(results)
    report_path = os.path.join(STUDY_DIR, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report saved to {report_path}")

    # Save metadata
    metadata = {
        "study": "Options Volatility Engine",
        "generated_at": datetime.now().isoformat(),
        "symbols": SYMBOLS,
        "seed": SEED,
        "risk_free_rate": RISK_FREE_RATE,
        "garch_window": GARCH_WINDOW,
        "rv_window": RV_WINDOW,
        "results": {s: {k: {mk: mv for mk, mv in v.items()
                            if not isinstance(mv, dict)}
                        for k, v in r.items() if isinstance(v, dict)}
                    for s, r in results.items() if isinstance(r, dict)},
    }
    meta_path = os.path.join(STUDY_DIR, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"Metadata saved to {meta_path}")

    print("\nDone!")
    return results


if __name__ == "__main__":
    run_study()
