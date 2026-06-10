from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(
    r"C:\Users\mathe\Documents\GitHub\mt5\dataset\export_mt5\intraday\avista\M15"
)
N_BARS_PER_YEAR = 252 * 26


def list_available_symbols() -> list[str]:
    return sorted(p.stem for p in DATA_DIR.glob("*.parquet"))


def load_symbol(symbol: str) -> pd.DataFrame:
    path = DATA_DIR / f"{symbol}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Data not found: {path}")
    df = pd.read_parquet(path)
    df = df.reset_index()
    df = df.rename(
        columns={
            "time": "timestamp",
            "Open": "open_price",
            "High": "high_price",
            "Low": "low_price",
            "Close": "close_price",
            "Tick_volume": "volume",
        }
    )
    df["symbol"] = symbol
    df["volume"] = df["volume"].astype("int64")
    df["timestamp"] = df["timestamp"].dt.tz_localize("America/Sao_Paulo")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df[df["volume"] > 0].reset_index(drop=True)
    return df


def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_return"] = np.log(df["close_price"] / df["close_price"].shift(1))
    df["simple_return"] = df["close_price"].pct_change()
    df = df.dropna(subset=["log_return", "simple_return"]).reset_index(drop=True)
    return df


def compute_realized_vol(
    df: pd.DataFrame, windows: list[int] | None = None
) -> pd.DataFrame:
    if windows is None:
        windows = [5, 10, 20, 30, 60]
    df = df.copy()
    for w in windows:
        df[f"realized_vol_{w}"] = (
            df["log_return"].rolling(w).std() * np.sqrt(N_BARS_PER_YEAR)
        )
    df["realized_vol"] = df["realized_vol_30"]
    return df


def load_universe(symbols: list[str] | None = None, max_assets: int = 50) -> dict[str, pd.DataFrame]:
    available = list_available_symbols()
    if symbols:
        available = [s for s in available if s in symbols]
    # Filter to likely stocks (code ends with digit, not 11/34/35/39)
    stocks = [
        s
        for s in available
        if s[-1].isdigit() and not any(s.endswith(x) for x in ("11", "34", "35", "39"))
    ]
    selected = stocks[:max_assets]
    result = {}
    for sym in selected:
        try:
            df = load_symbol(sym)
            df = compute_returns(df)
            df = compute_realized_vol(df)
            result[sym] = df
        except Exception:
            continue
    return result
