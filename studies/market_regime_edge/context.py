from __future__ import annotations

import numpy as np
import pandas as pd

from models.hmm_regime import HMMRegimeModel

N_BARS_PER_YEAR = 252 * 26


class ContextComputer:
    def __init__(self, df: pd.DataFrame, config: dict | None = None):
        self.df = df.copy()
        self.config = config or {}
        self._hmm_model: HMMRegimeModel | None = None

    def compute_all(self) -> pd.DataFrame:
        df = self.df.copy()
        # Volatilidade
        df = self._compute_vol_context(df)
        # Volume
        df = self._compute_volume_context(df)
        # Tendência
        df = self._compute_trend_context(df)
        # Estrutura
        df = self._compute_structure_context(df)
        return df

    def _compute_vol_context(self, df: pd.DataFrame) -> pd.DataFrame:
        df["rv_5"] = (
            df["log_return"].rolling(5).std() * np.sqrt(N_BARS_PER_YEAR)
        )
        df["rv_20"] = (
            df["log_return"].rolling(20).std() * np.sqrt(N_BARS_PER_YEAR)
        )
        df["rv_percentile"] = df["rv_20"].rank(pct=True)
        # Vol regime via HMM
        try:
            hmm = HMMRegimeModel(n_regimes=4)
            hmm.fit(df["log_return"])
            regimes = hmm.predict(df["log_return"])
            pad = len(df) - len(regimes)
            df["vol_regime"] = np.concatenate([np.full(pad, -1), regimes])
            self._hmm_model = hmm
        except Exception:
            df["vol_regime"] = -1
        return df

    def _compute_volume_context(self, df: pd.DataFrame) -> pd.DataFrame:
        df["volume_sma_20"] = df["volume"].rolling(20).mean()
        df["relative_volume"] = df["volume"] / df["volume_sma_20"]
        df["volume_percentile"] = df["volume"].rank(pct=True)
        df["volume_spike"] = (df["relative_volume"] > 2.0).astype(float)
        return df

    def _compute_trend_context(self, df: pd.DataFrame) -> pd.DataFrame:
        df["sma_200"] = df["close_price"].rolling(200).mean()
        df["sma200_distance"] = df["close_price"] / df["sma_200"] - 1
        df["sma200_slope"] = df["sma_200"].diff(20) / df["sma_200"].shift(20)
        df["sma200_slope"] = df["sma200_slope"].fillna(0)
        # ADX simplificado
        df = self._compute_adx(df, period=14)
        # Hurst exponent aproximado
        df["hurst"] = df["log_return"].rolling(100).apply(
            self._hurst_exponent, raw=False
        )
        return df

    @staticmethod
    def _compute_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        df["tr"] = np.maximum(
            df["high_price"] - df["low_price"],
            np.maximum(
                np.abs(df["high_price"] - df["close_price"].shift(1)),
                np.abs(df["low_price"] - df["close_price"].shift(1)),
            ),
        )
        df["up_move"] = df["high_price"] - df["high_price"].shift(1)
        df["down_move"] = df["low_price"].shift(1) - df["low_price"]
        df["plus_dm"] = np.where(
            (df["up_move"] > df["down_move"]) & (df["up_move"] > 0),
            df["up_move"],
            0,
        )
        df["minus_dm"] = np.where(
            (df["down_move"] > df["up_move"]) & (df["down_move"] > 0),
            df["down_move"],
            0,
        )
        df["atr"] = df["tr"].rolling(period).mean()
        df["plus_di"] = 100 * (df["plus_dm"].rolling(period).mean() / df["atr"])
        df["minus_di"] = 100 * (df["minus_dm"].rolling(period).mean() / df["atr"])
        df["dx"] = (
            100
            * np.abs(df["plus_di"] - df["minus_di"])
            / (df["plus_di"] + df["minus_di"]).replace(0, np.nan)
        )
        df["adx"] = df["dx"].rolling(period).mean()
        return df

    @staticmethod
    def _hurst_exponent(ts: pd.Series) -> float:
        try:
            ts = ts.dropna().values
            if len(ts) < 50:
                return 0.5
            lags = np.arange(2, min(21, len(ts) // 2))
            tau = [np.std(np.diff(ts, lag)) for lag in lags]
            if any(t == 0 for t in tau):
                return 0.5
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] / 2
        except Exception:
            return 0.5

    def _compute_structure_context(self, df: pd.DataFrame) -> pd.DataFrame:
        lookback = 50
        df["support_distance"] = np.nan
        df["resistance_distance"] = np.nan
        df["normalized_range"] = (df["high_price"] - df["low_price"]) / df["close_price"]
        for i in range(lookback, len(df)):
            window = df.iloc[i - lookback : i]
            support = window["low_price"].min()
            resistance = window["high_price"].max()
            price = df["close_price"].iloc[i]
            df.loc[df.index[i], "support_distance"] = (price - support) / price
            df.loc[df.index[i], "resistance_distance"] = (resistance - price) / price
        return df

    def get_context_column_names(self) -> list[str]:
        return [
            "rv_5",
            "rv_20",
            "rv_percentile",
            "vol_regime",
            "relative_volume",
            "volume_percentile",
            "volume_spike",
            "sma200_distance",
            "sma200_slope",
            "adx",
            "hurst",
            "support_distance",
            "resistance_distance",
            "atr",
            "normalized_range",
        ]

    def get_regime_labels(self) -> dict[int, str]:
        if self._hmm_model is not None:
            return self._hmm_model.labels
        return {0: "Low Vol", 1: "Medium Vol", 2: "High Vol", 3: "Extreme Vol"}
