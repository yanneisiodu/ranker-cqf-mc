"""Plausibility scoring utilities for candidate stress scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from scipy.stats import multivariate_normal


@dataclass
class RegimeContext:
    atm_iv: float
    label: str
    extras: Dict[str, float] | None = None


class PlausibilityScorer:
    """Scores scenarios using a regime-conditioned Gaussian copula."""

    def __init__(self, mean: np.ndarray, cov: np.ndarray, iv_bounds: tuple[float, float] = (0.05, 1.2), corr_cap: float = 0.8):
        self._mean = np.asarray(mean, dtype=float)
        self._cov = np.asarray(cov, dtype=float)
        self._iv_bounds = iv_bounds
        self._corr_cap = corr_cap

    def _base_score(self, vec: np.ndarray) -> float:
        ll = multivariate_normal.logpdf(vec, mean=self._mean, cov=self._cov)
        return float(1.0 / (1.0 + np.exp(-0.05 * ll)))  # sigmoid to (0,1)

    def __call__(self, scenario: Dict[str, float], regime: RegimeContext) -> float:
        vec = np.array([
            scenario['spot_mult'] - 1.0,
            scenario['iv_jump'],
            scenario['smile_tilt'],
            scenario['corr_break'],
            scenario['liq_haircut'],
        ])
        base = self._base_score(vec)
        iv = regime.atm_iv + scenario['iv_jump']
        lo, hi = self._iv_bounds
        iv_penalty = 1.0 if lo <= iv <= hi else 0.0
        corr_penalty = 1.0 if scenario['corr_break'] <= self._corr_cap else 0.0
        extras = 0.0
        if regime.extras:
            extras = float(sum(regime.extras.get(k, 0.0) for k in ('liquidity', 'macro_pressure')))
        score = base * 0.6 + iv_penalty * 0.2 + corr_penalty * 0.2 + extras
        return float(np.clip(score, 0.0, 1.0))


__all__ = ["PlausibilityScorer", "RegimeContext"]
