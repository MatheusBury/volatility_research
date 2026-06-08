from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from models.egarch import EGARCHModel
from models.iv_collector import IVCollector

warnings.filterwarnings("ignore")

N_BARS_PER_YEAR = 252 * 26


@dataclass
class VRPRecord:
    date: pd.Timestamp
    stock: str
    forecast_rv: float
    implied_iv: float
    spread: float
    future_rv: float
    dte: int
    regime: int


class VRPRelativeValueModel:
    def __init__(self, egarch: EGARCHModel, iv_collector: IVCollector):
        self.egarch = egarch
        self.iv_collector = iv_collector

    def compute_spread(self, stock: str, date: pd.Timestamp) -> Optional[float]:
        iv_data = self.iv_collector.get_atm_iv(stock, date)
        if iv_data is None:
            return None
        return iv_data["iv"]

    @staticmethod
    def generate_signal(spread: float, threshold: float = 0.05) -> int:
        if spread > threshold:
            return 1
        if spread < -threshold:
            return -1
        return 0

    def backtest(
        self,
        stock: str,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        m15_data: pd.DataFrame,
        iv_timeseries: pd.DataFrame,
        hmm_regimes: np.ndarray | None = None,
    ) -> list[VRPRecord]:
        m15 = m15_data.copy()
        m15 = m15.sort_values("timestamp").reset_index(drop=True)
        m15["log_return"] = np.log(m15["close_price"] / m15["close_price"].shift(1))

        if hmm_regimes is None:
            hmm_regimes = np.full(len(m15), -1)

        records: list[VRPRecord] = []

        for _, row in iv_timeseries.iterrows():
            date = pd.Timestamp(row["date"])
            if m15["timestamp"].dt.tz is not None:
                date = date.tz_localize("America/Sao_Paulo")
            iv = float(row["iv"])
            dte = int(row["dte"])

            n_bars_forecast = dte * 26
            last_is_idx = len(m15[m15["timestamp"] <= date])
            if last_is_idx <= 0 or last_is_idx >= len(m15):
                continue

            returns_until = m15["log_return"].iloc[:last_is_idx].dropna()
            if len(returns_until) < 100:
                continue

            try:
                model = EGARCHModel(p=1, o=1, q=1)
                model.fit(returns_until)
                fcast_vols = model.forecast(min(n_bars_forecast, 500))
                forecast_rv = float(np.mean(fcast_vols))
            except Exception:
                continue

            future_end_idx = min(last_is_idx + n_bars_forecast, len(m15))
            future_returns = m15["log_return"].iloc[last_is_idx:future_end_idx].dropna()
            if len(future_returns) < 5:
                continue
            future_rv = float(future_returns.std() * np.sqrt(N_BARS_PER_YEAR))

            spread = forecast_rv - iv

            regime_idx = min(last_is_idx - 1, len(hmm_regimes) - 1)
            regime = int(hmm_regimes[regime_idx]) if regime_idx >= 0 else -1

            records.append(VRPRecord(
                date=date,
                stock=stock,
                forecast_rv=forecast_rv,
                implied_iv=iv,
                spread=spread,
                future_rv=future_rv,
                dte=dte,
                regime=regime,
            ))

        return records

    @staticmethod
    def evaluate(records: list[VRPRecord]) -> dict[str, Any]:
        if not records:
            return {"error": "no records"}

        df = pd.DataFrame([
            {"forecast_rv": r.forecast_rv, "implied_iv": r.implied_iv,
             "spread": r.spread, "future_rv": r.future_rv,
             "dte": r.dte, "regime": r.regime}
            for r in records
        ])

        df["spread_bucket"] = pd.cut(
            df["spread"],
            bins=[-np.inf, -0.05, -0.02, 0.02, 0.05, np.inf],
            labels=["IV >> Forecast", "IV > Forecast", "≈ Fair", "Forecast > IV", "Forecast >> IV"],
        )

        calibration = df.groupby("spread_bucket", observed=True).agg(
            n=("future_rv", "count"),
            mean_spread=("spread", "mean"),
            mean_future_rv=("future_rv", "mean"),
        ).reset_index()

        correct_dir = ((df["spread"] > 0) & (df["future_rv"] > df["implied_iv"])) | \
                       ((df["spread"] < 0) & (df["future_rv"] < df["implied_iv"]))
        hit_rate = float(correct_dir.mean())

        df["signal"] = df["spread"].apply(
            lambda s: 1 if s > 0.05 else (-1 if s < -0.05 else 0)
        )

        df["long_vol_ok"] = (df["signal"] == 1) & (df["future_rv"] > df["implied_iv"])
        df["short_vol_ok"] = (df["signal"] == -1) & (df["future_rv"] < df["implied_iv"])

        n_long = int((df["signal"] == 1).sum())
        n_short = int((df["signal"] == -1).sum())
        n_none = int((df["signal"] == 0).sum())

        confusion_matrix = {
            "long_vol_signals": n_long,
            "long_vol_correct": int(df["long_vol_ok"].sum()),
            "long_vol_hit_rate": float(
                df.loc[df["signal"] == 1, "long_vol_ok"].mean()
            ) if n_long > 0 else 0.0,
            "short_vol_signals": n_short,
            "short_vol_correct": int(df["short_vol_ok"].sum()),
            "short_vol_hit_rate": float(
                df.loc[df["signal"] == -1, "short_vol_ok"].mean()
            ) if n_short > 0 else 0.0,
            "no_signal": n_none,
        }

        df["pnl"] = np.where(
            df["signal"] == 1,
            df["future_rv"] - df["implied_iv"],
            np.where(
                df["signal"] == -1,
                df["implied_iv"] - df["future_rv"],
                0.0,
            ),
        )
        sharpe = float(df["pnl"].mean() / df["pnl"].std() * np.sqrt(252 / df["dte"].mean())) \
            if df["pnl"].std() > 1e-10 else 0.0

        return {
            "n_observations": len(df),
            "hit_rate": hit_rate,
            "calibration": calibration,
            "confusion_matrix": confusion_matrix,
            "sharpe_ratio": sharpe,
            "mean_spread": float(df["spread"].mean()),
            "mean_future_rv": float(df["future_rv"].mean()),
        }
