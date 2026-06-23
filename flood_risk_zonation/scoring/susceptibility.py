"""Transparent, deterministic flood-susceptibility model."""
from __future__ import annotations

import numpy as np
import pandas as pd


# Positive direction means larger values increase susceptibility; negative
# direction means smaller values increase susceptibility. Aspect is excluded
# because it has no globally consistent relationship with flooding.
FACTOR_WEIGHTS: dict[str, tuple[float, int]] = {
    "elevation_m": (0.15, -1),
    "slope_deg": (0.05, -1),
    "twi": (0.15, 1),
    "rainfall_mean_mm": (0.10, 1),
    "rainfall_max_24h_mm": (0.15, 1),
    "dist_water_m": (0.15, -1),
    "drainage_capacity": (0.15, -1),
    "population_density": (0.05, 1),
    "curvature": (0.05, -1),
}


class WeightedSusceptibilityModel:
    """Robustly normalise factors and combine them with declared weights."""

    def __init__(self, factor_weights: dict[str, tuple[float, int]] | None = None) -> None:
        self.factor_weights = factor_weights or FACTOR_WEIGHTS
        self.feature_names = list(self.factor_weights)
        self.lower_: dict[str, float] = {}
        self.upper_: dict[str, float] = {}
        self.feature_importances_ = np.array(
            [self.factor_weights[name][0] for name in self.feature_names],
            dtype=np.float64,
        )
        total = float(self.feature_importances_.sum())
        if total <= 0:
            raise ValueError("Susceptibility factor weights must sum to a positive value.")
        self.feature_importances_ /= total

    def fit(self, X: pd.DataFrame) -> "WeightedSusceptibilityModel":
        """Fit robust 5th/95th-percentile normalisation bounds."""
        for name in self.feature_names:
            values = np.asarray(X[name], dtype=np.float64)
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                self.lower_[name] = 0.0
                self.upper_[name] = 1.0
            else:
                self.lower_[name] = float(np.percentile(finite, 5))
                self.upper_[name] = float(np.percentile(finite, 95))
        return self

    def _positive_probability(self, X: pd.DataFrame) -> np.ndarray:
        if not self.lower_:
            raise ValueError("WeightedSusceptibilityModel must be fitted before scoring.")
        score = np.zeros(len(X), dtype=np.float64)
        for idx, name in enumerate(self.feature_names):
            values = np.asarray(X[name], dtype=np.float64)
            lo, hi = self.lower_[name], self.upper_[name]
            if hi <= lo:
                normalised = np.full(len(X), 0.5, dtype=np.float64)
            else:
                normalised = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
            if self.factor_weights[name][1] < 0:
                normalised = 1.0 - normalised
            score += self.feature_importances_[idx] * normalised
        return np.clip(score, 0.0, 1.0)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return two columns compatible with sklearn-style scorers."""
        positive = self._positive_probability(X)
        return np.column_stack((1.0 - positive, positive))

    @property
    def feature_importances(self) -> dict[str, float]:
        return dict(zip(self.feature_names, self.feature_importances_.tolist()))
