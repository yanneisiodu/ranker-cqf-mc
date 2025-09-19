"""Adapters that let the ultra-agent reuse production stress simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Protocol
import numpy as np


@dataclass
class ImpactStats:
    """Summary of scenario impact on the portfolio."""

    var95: float
    cvar95: float
    mean: float
    skew: float
    affected_books: Dict[str, float]


class StressOracle(Protocol):
    """Protocol implemented by prod_stress_mc wrappers."""

    def run_scenario(self, scenario: Dict[str, float]) -> Dict[str, np.ndarray]:
        """Run the stress scenario and return portfolio/book PnL arrays."""


class OracleAdapter:
    """Thin adapter around the legacy stress engine."""

    def __init__(self, oracle: StressOracle):
        self._oracle = oracle

    def evaluate(self, scenario: Dict[str, float]) -> ImpactStats:
        result = self._oracle.run_scenario(scenario)
        pnl = np.asarray(result["portfolio_pnl"], dtype=float)
        if pnl.size == 0:
            raise ValueError("Oracle returned empty PnL vector")
        mean = float(np.mean(pnl))
        std = float(np.std(pnl))
        var95 = float(np.percentile(pnl, 5))
        tail = pnl[pnl <= np.percentile(pnl, 5)]
        cvar95 = float(np.mean(tail)) if tail.size else var95
        if std > 1e-9:
            skew = float(np.mean(((pnl - mean) / std) ** 3))
        else:
            skew = 0.0
        affected = {book: float(np.mean(values)) for book, values in result.get("book_pnl", {}).items()}
        return ImpactStats(var95=var95, cvar95=cvar95, mean=mean, skew=skew, affected_books=affected)


__all__ = ["ImpactStats", "OracleAdapter"]
