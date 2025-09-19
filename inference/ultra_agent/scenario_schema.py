"""Scenario schema and guardrail utilities for the ultra-agent stress engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping

DEFAULT_BOUNDS: Mapping[str, tuple[float, float]] = {
    "spot_mult": (0.85, 1.10),
    "iv_jump": (-0.10, 0.60),
    "smile_tilt": (-0.30, 0.30),
    "corr_break": (0.0, 0.7),
    "liq_haircut": (0.0, 0.7),
}


@dataclass
class ScenarioSpec:
    """Canonical scenario description used by the agent."""

    name: str
    prob: float
    spot_mult: float
    iv_jump: float
    smile_tilt: float
    corr_break: float
    liq_haircut: float
    narrative: str
    meta: Dict[str, float] | None = None


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp *value* into the inclusive range [lo, hi]."""
    return max(lo, min(hi, value))


def apply_bounds(spec: Mapping[str, float], bounds: Mapping[str, tuple[float, float]] | None = None) -> Dict[str, float]:
    """Return a bounded copy of *spec* using the provided bounds."""
    bounds = bounds or DEFAULT_BOUNDS
    bounded = dict(spec)
    for key, (lo, hi) in bounds.items():
        if key in bounded:
            bounded[key] = clamp(float(bounded[key]), lo, hi)
    return bounded


def normalise_probabilities(specs: Iterable[ScenarioSpec], min_prob: float = 1e-4) -> list[ScenarioSpec]:
    """Normalise scenario probabilities and ensure they respect *min_prob*."""
    specs = list(specs)
    total = sum(max(min_prob, s.prob) for s in specs)
    if total <= 0:
        equal = 1.0 / len(specs) if specs else 0.0
        return [ScenarioSpec(**{**s.__dict__, "prob": equal}) for s in specs]
    normalised: list[ScenarioSpec] = []
    for s in specs:
        prob = max(min_prob, s.prob) / total
        normalised.append(ScenarioSpec(**{**s.__dict__, "prob": prob}))
    return normalised


__all__ = [
    "ScenarioSpec",
    "DEFAULT_BOUNDS",
    "clamp",
    "apply_bounds",
    "normalise_probabilities",
]
