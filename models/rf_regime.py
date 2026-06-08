from __future__ import annotations

from typing_extensions import Self

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

N_BARS_PER_YEAR = 252 * 26


class RFRegimeModel:
    def __init__(self, n_estimators: int = 200, max_depth: int = 6):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self._model: RandomForestClassifier | None = None
        self._feature_cols: list[str] = []

    def build_features(self, prices: pd.Series) -> pd.DataFrame:
        df = prices.to_frame("close_price")
        df["log_return"] = np.log(df["close_price"] / df["close_price"].shift(1))

        df["rv_5"] = df["log_return"].rolling(5).std() * np.sqrt(N_BARS_PER_YEAR)
        df["rv_10"] = df["log_return"].rolling(10).std() * np.sqrt(N_BARS_PER_YEAR)
        df["rv_20"] = df["log_return"].rolling(20).std() * np.sqrt(N_BARS_PER_YEAR)

        df["skew_20"] = df["log_return"].rolling(20).skew()
        df["kurt_20"] = df["log_return"].rolling(20).kurt()

        df["ret_1"] = df["log_return"]
        df["ret_5"] = df["log_return"].rolling(5).sum()
        df["ret_10"] = df["log_return"].rolling(10).sum()

        df["volume_ratio"] = 1.0

        return df.dropna()

    def fit(self, features: pd.DataFrame, target: pd.Series) -> Self:
        self._feature_cols = [
            c for c in features.columns if c not in ("close_price", "log_return")
        ]
        X = features[self._feature_cols].values
        y = target.values
        self._model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=42,
            class_weight="balanced",
            min_samples_leaf=5,
            n_jobs=-1,
        )
        self._model.fit(X, y)
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X = features[self._feature_cols].values
        return self._model.predict(X)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X = features[self._feature_cols].values
        return self._model.predict_proba(X)

    def feature_importance(self) -> pd.DataFrame:
        if self._model is None:
            raise RuntimeError("Model not fitted.")
        return pd.DataFrame({
            "feature": self._feature_cols,
            "importance": self._model.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
