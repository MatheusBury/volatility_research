from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class TradeRecord:
    timestamp: pd.Timestamp
    symbol: str
    strategy: str
    direction: int  # 1 = long, -1 = short
    entry_price: float
    exit_price: float
    log_return: float
    simple_return: float
    context: dict = field(default_factory=dict)


class BaseStrategy(ABC):
    def __init__(self, name: str, params: dict | None = None):
        self.name = name
        self.params = params or {}

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> np.ndarray:
        ...

    def __repr__(self) -> str:
        return f"{self.name}({self.params})"


class MomentumStrategy(BaseStrategy):
    def __init__(self, params: dict | None = None):
        super().__init__("momentum", params)
        self.lookback = self.params.get("lookback", 20)

    def generate_signals(self, df: pd.DataFrame) -> np.ndarray:
        ret = df["log_return"].rolling(self.lookback).sum()
        signals = np.zeros(len(df), dtype=float)
        signals[ret > 0] = 1.0
        signals[ret < 0] = -1.0
        return signals


class MeanReversionStrategy(BaseStrategy):
    def __init__(self, params: dict | None = None):
        super().__init__("mean_reversion", params)
        self.lookback = self.params.get("lookback", 20)
        self.threshold = self.params.get("threshold", 2.0)

    def generate_signals(self, df: pd.DataFrame) -> np.ndarray:
        ret = df["log_return"].rolling(self.lookback).sum()
        zscore = (ret - ret.expanding().mean()) / ret.expanding().std()
        signals = np.zeros(len(df), dtype=float)
        signals[zscore > self.threshold] = -1.0
        signals[zscore < -self.threshold] = 1.0
        return signals


class GapFadeStrategy(BaseStrategy):
    def __init__(self, params: dict | None = None):
        super().__init__("gap_fade", params)
        self.threshold = self.params.get("threshold", 0.02)

    def generate_signals(self, df: pd.DataFrame) -> np.ndarray:
        gap = df["open_price"] / df["close_price"].shift(1) - 1
        signals = np.zeros(len(df), dtype=float)
        signals[gap > self.threshold] = -1.0
        signals[gap < -self.threshold] = 1.0
        return signals


class BreakoutStrategy(BaseStrategy):
    def __init__(self, params: dict | None = None):
        super().__init__("breakout", params)
        self.lookback = self.params.get("lookback", 20)

    def generate_signals(self, df: pd.DataFrame) -> np.ndarray:
        high_roll = df["high_price"].rolling(self.lookback).max().shift(1)
        low_roll = df["low_price"].rolling(self.lookback).min().shift(1)
        signals = np.zeros(len(df), dtype=float)
        signals[df["close_price"] > high_roll] = 1.0
        signals[df["close_price"] < low_roll] = -1.0
        return signals


class VWAPReversionStrategy(BaseStrategy):
    def __init__(self, params: dict | None = None):
        super().__init__("vwap_reversion", params)
        self.lookback = self.params.get("lookback", 20)

    def generate_signals(self, df: pd.DataFrame) -> np.ndarray:
        df = df.copy()
        df["cum_vol"] = df["volume"].cumsum()
        df["cum_pv"] = (df["close_price"] * df["volume"]).cumsum()
        df["vwap"] = df["cum_pv"] / df["cum_vol"]
        deviation = df["close_price"] / df["vwap"] - 1
        signals = np.zeros(len(df), dtype=float)
        signals[deviation > 0.01] = -1.0
        signals[deviation < -0.01] = 1.0
        return signals


class OpeningRangeBreakoutStrategy(BaseStrategy):
    def __init__(self, params: dict | None = None):
        super().__init__("opening_range_breakout", params)
        self.window = self.params.get("window", 4)

    def generate_signals(self, df: pd.DataFrame) -> np.ndarray:
        signals = np.zeros(len(df), dtype=float)
        for i in range(self.window, len(df)):
            or_high = df["high_price"].iloc[i - self.window : i].max()
            or_low = df["low_price"].iloc[i - self.window : i].min()
            if df["close_price"].iloc[i] > or_high:
                signals[i] = 1.0
            elif df["close_price"].iloc[i] < or_low:
                signals[i] = -1.0
        return signals


class SupportResistanceBounceStrategy(BaseStrategy):
    def __init__(self, params: dict | None = None):
        super().__init__("sr_bounce", params)
        self.lookback = self.params.get("lookback", 50)
        self.tolerance = self.params.get("tolerance", 0.01)

    def generate_signals(self, df: pd.DataFrame) -> np.ndarray:
        signals = np.zeros(len(df), dtype=float)
        for i in range(self.lookback, len(df)):
            window = df.iloc[i - self.lookback : i]
            support = window["low_price"].min()
            resistance = window["high_price"].max()
            price = df["close_price"].iloc[i]
            dist_to_support = (price - support) / price
            dist_to_resistance = (resistance - price) / price
            if dist_to_support < self.tolerance:
                signals[i] = 1.0
            elif dist_to_resistance < self.tolerance:
                signals[i] = -1.0
        return signals


class TrendFollowingStrategy(BaseStrategy):
    def __init__(self, params: dict | None = None):
        super().__init__("trend_following", params)
        self.lookback = self.params.get("lookback", 50)

    def generate_signals(self, df: pd.DataFrame) -> np.ndarray:
        sma = df["close_price"].rolling(self.lookback).mean()
        signals = np.zeros(len(df), dtype=float)
        signals[df["close_price"] > sma] = 1.0
        signals[df["close_price"] < sma] = -1.0
        return signals


STRATEGY_MAP = {
    "momentum": MomentumStrategy,
    "mean_reversion": MeanReversionStrategy,
    "gap_fade": GapFadeStrategy,
    "breakout": BreakoutStrategy,
    "vwap_reversion": VWAPReversionStrategy,
    "opening_range_breakout": OpeningRangeBreakoutStrategy,
    "sr_bounce": SupportResistanceBounceStrategy,
    "trend_following": TrendFollowingStrategy,
}


def create_strategies(config: dict) -> list[BaseStrategy]:
    names = config.get("strategies", list(STRATEGY_MAP.keys()))
    default_params = config.get("strategy_params", {})
    strategies = []
    for name in names:
        cls = STRATEGY_MAP.get(name)
        if cls is None:
            continue
        param_mapping = {
            "momentum": {"lookback": "momentum_lookback"},
            "mean_reversion": {"lookback": "mean_reversion_lookback", "threshold": "mean_reversion_threshold"},
            "gap_fade": {"threshold": "gap_fade_threshold"},
            "breakout": {"lookback": "breakout_lookback"},
            "vwap_reversion": {"lookback": "vwap_lookback"},
            "opening_range_breakout": {"window": "or_window"},
            "sr_bounce": {"lookback": "sr_lookback", "tolerance": "sr_tolerance"},
            "trend_following": {"lookback": "tf_lookback"},
        }
        params = {}
        mapping = param_mapping.get(name, {})
        for pname, cfg_key in mapping.items():
            if cfg_key in default_params:
                params[pname] = default_params[cfg_key]
        strategies.append(cls(params))
    return strategies
