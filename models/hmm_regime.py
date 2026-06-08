from __future__ import annotations

from typing_extensions import Self

import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler


class HMMRegimeModel:
    def __init__(self, n_regimes: int = 4):
        self.n_regimes = n_regimes
        self._model: hmm.GaussianHMM | None = None
        self._scaler: StandardScaler | None = None
        self._labels: dict[int, str] = {}
        self._regime_map: dict[int, int] = {}

    def fit(self, returns: pd.Series) -> Self:
        df = returns.dropna().to_frame("log_return")
        vol_window = 30
        N_BARS_PER_YEAR = 252 * 26
        df["realized_vol"] = (
            df["log_return"].rolling(window=vol_window).std() * np.sqrt(N_BARS_PER_YEAR)
        )
        df = df.dropna()
        features = df[["log_return", "realized_vol"]].values.astype(np.float64)
        self._scaler = StandardScaler()
        X = self._scaler.fit_transform(features)
        self._model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=1000,
            tol=1e-4,
            random_state=42,
            init_params="stmc",
        )
        self._model.fit(X)
        self._set_labels(X)
        return self

    def _set_labels(self, X: np.ndarray) -> None:
        if self._model is None:
            return
        states = self._model.predict(X)
        vol_col = 1
        state_means = {
            s: float(np.mean(X[states == s, vol_col])) for s in range(self.n_regimes)
        }
        sorted_states = sorted(state_means, key=state_means.get)
        names = ["Low Vol", "Medium Vol", "High Vol", "Extreme Vol"]
        self._labels = {s: names[i] for i, s in enumerate(sorted_states)}
        self._regime_map = {s: i for i, s in enumerate(sorted_states)}

    def predict(self, returns: pd.Series) -> np.ndarray:
        if self._model is None or self._scaler is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        df = returns.dropna().to_frame("log_return")
        vol_window = 30
        N_BARS_PER_YEAR = 252 * 26
        df["realized_vol"] = (
            df["log_return"].rolling(window=vol_window).std() * np.sqrt(N_BARS_PER_YEAR)
        )
        df = df.dropna()
        features = df[["log_return", "realized_vol"]].values.astype(np.float64)
        X = self._scaler.transform(features)
        states = self._model.predict(X)
        return np.array([self._regime_map.get(s, s) for s in states])

    def get_regime_probs(self, returns: pd.Series) -> np.ndarray:
        if self._model is None or self._scaler is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        df = returns.dropna().to_frame("log_return")
        vol_window = 30
        N_BARS_PER_YEAR = 252 * 26
        df["realized_vol"] = (
            df["log_return"].rolling(window=vol_window).std() * np.sqrt(N_BARS_PER_YEAR)
        )
        df = df.dropna()
        features = df[["log_return", "realized_vol"]].values.astype(np.float64)
        X = self._scaler.transform(features)
        return self._model.predict_proba(X)

    @property
    def labels(self) -> dict[int, str]:
        return self._labels

    @property
    def transition_matrix(self) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted.")
        return self._model.transmat_
