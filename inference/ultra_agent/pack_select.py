"""Scenario pack selection utilities (diversity + impact)."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np

from .scenario_schema import ScenarioSpec
from .oracle_adapter import ImpactStats


def scenario_vector(spec: ScenarioSpec) -> np.ndarray:
    return np.array([
        spec.spot_mult - 1.0,
        spec.iv_jump,
        spec.smile_tilt,
        spec.corr_break,
        spec.liq_haircut,
    ], dtype=float)


def kernel_matrix(specs: Sequence[ScenarioSpec], sigma: float = 0.25) -> np.ndarray:
    X = np.stack([scenario_vector(s) for s in specs])
    d2 = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1)
    return np.exp(-d2 / (2 * sigma ** 2))


def greedy_select(candidates: Sequence[ScenarioSpec], impacts: Sequence[ImpactStats], plaus_scores: Sequence[float], pack_size: int,
                  w_impact: float = 1.0, w_plaus: float = 0.2, w_div: float = 0.3) -> List[ScenarioSpec]:
    if len(candidates) <= pack_size:
        return list(candidates)
    K = kernel_matrix(candidates)
    chosen_idx: List[int] = []
    for _ in range(pack_size):
        best_idx, best_gain = None, -1e18
        for idx in range(len(candidates)):
            if idx in chosen_idx:
                continue
            impact = abs(impacts[idx].cvar95)
            plaus = plaus_scores[idx]
            diversity = 0.0 if not chosen_idx else -np.max(K[idx, chosen_idx])
            gain = w_impact * impact + w_plaus * plaus + w_div * diversity
            if gain > best_gain:
                best_idx, best_gain = idx, gain
        if best_idx is None:
            break
        chosen_idx.append(best_idx)
    return [candidates[i] for i in chosen_idx]


__all__ = ["greedy_select"]
