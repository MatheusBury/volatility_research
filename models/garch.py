from __future__ import annotations

import pickle
import warnings
from typing_extensions import Self

import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate.base import ARCHModelResult

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

N_BARS_PER_YEAR = 252 * 26
RETURN_SCALE = 1000.0


class GARCHModel:
    def __init__(self, p: int = 1, q: int = 1, scale: float = RETURN_SCALE):
        self.p = p
        self.q = q
        self.scale = scale
        self._result: ARCHModelResult | None = None
        self._returns_scaled: pd.Series | None = None

    def fit(self, returns: pd.Series) -> Self:
        self._returns_scaled = returns.dropna() * self.scale
        am = arch_model(
            self._returns_scaled,
            mean="zero",
            vol="GARCH",
            p=self.p,
            q=self.q,
            dist="normal",
        )
        res = am.fit(disp="off", update_freq=0)
        self._result = res
        return self

    def forecast(self, horizon: int) -> np.ndarray:
        if self._result is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        if self._returns_scaled is None:
            raise RuntimeError("No returns data available.")
        n_is = len(self._returns_scaled)
        fcast = self._result.forecast(
            horizon=horizon,
            start=n_is - 1,
            method="analytic",
            reindex=False,
        )
        fcast_vars = fcast.variance.iloc[0, :].values.astype(float)
        fcast_vols = np.sqrt(fcast_vars) / self.scale
        annualized = fcast_vols * np.sqrt(N_BARS_PER_YEAR)
        return annualized

    def get_conditional_vol(self) -> np.ndarray:
        if self._result is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self._result.conditional_volatility / self.scale

    def get_params(self) -> dict[str, float]:
        if self._result is None:
            return {}
        p = self._result.params
        return {
            "omega": float(p.get("omega", 0.0)),
            "alpha": float(p.get("alpha[1]", 0.0)),
            "beta": float(p.get("beta[1]", 0.0)),
            "aic": float(self._result.aic),
            "bic": float(self._result.bic),
        }

    def save(self, path: str) -> None:
        data = {
            "p": self.p,
            "q": self.q,
            "scale": self.scale,
            "result": self._result,
            "returns_scaled": self._returns_scaled,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str) -> GARCHModel:
        with open(path, "rb") as f:
            data = pickle.load(f)
        model = cls(p=data["p"], q=data["q"], scale=data["scale"])
        model._result = data["result"]
        model._returns_scaled = data["returns_scaled"]
        return model
