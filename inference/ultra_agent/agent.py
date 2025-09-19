"""Ultra stress agent orchestrator."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .scenario_schema import ScenarioSpec, apply_bounds, normalise_probabilities, DEFAULT_BOUNDS
from .oracle_adapter import OracleAdapter, ImpactStats
from .plausibility import PlausibilityScorer, RegimeContext
from .pack_select import greedy_select


@dataclass
class BookSnapshot:
    greek_buckets: Dict[str, float]
    moneyness_mix: Dict[str, float]
    top_exposures: List[str]


@dataclass
class AgentConfig:
    bounds: Dict[str, tuple[float, float]]
    n_candidates: int
    pack_size: int
    min_plausibility: float
    weights: Dict[str, float]

    @classmethod
    def default(cls) -> "AgentConfig":
        return cls(
            bounds=dict(DEFAULT_BOUNDS),
            n_candidates=64,
            pack_size=8,
            min_plausibility=0.15,
            weights={"impact": 1.0, "plaus": 0.2, "div": 0.3},
        )


class ScenarioCache:
    def __init__(self):
        self._store: Dict[str, List[Dict[str, Any]]] = {}

    def get(self, key: str) -> Optional[List[Dict[str, Any]]]:
        return self._store.get(key)

    def set(self, key: str, scenarios: List[ScenarioSpec]) -> None:
        self._store[key] = [s.__dict__ for s in scenarios]


class UltraStressAgent:
    """High-level orchestrator joining LLM proposals with oracle scoring."""

    def __init__(self, llm_client, oracle: OracleAdapter, plausibility: PlausibilityScorer,
                 cache: Optional[ScenarioCache] = None, config: Optional[AgentConfig] = None):
        self._llm = llm_client
        self._oracle = oracle
        self._plausibility = plausibility
        self._cache = cache or ScenarioCache()
        self._config = config or AgentConfig.default()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_pack(self, book: BookSnapshot, regime: RegimeContext) -> List[ScenarioSpec]:
        key = self._cache_key(book, regime)
        cached = self._cache.get(key)
        if cached is not None:
            return [ScenarioSpec(**s) for s in cached]

        raw_candidates = self._propose_candidates(book, regime)
        bounded = [ScenarioSpec(**apply_bounds(spec.__dict__, self._config.bounds)) for spec in raw_candidates]
        impacts = [self._oracle.evaluate(spec.__dict__) for spec in bounded]
        plaus_scores = [self._plausibility(spec.__dict__, regime) for spec in bounded]
        filtered = [spec for spec, score in zip(bounded, plaus_scores) if score >= self._config.min_plausibility]
        filtered_impacts = [impact for impact, score in zip(impacts, plaus_scores) if score >= self._config.min_plausibility]
        filtered_scores = [score for score in plaus_scores if score >= self._config.min_plausibility]

        selected = greedy_select(
            filtered,
            filtered_impacts,
            filtered_scores,
            pack_size=self._config.pack_size,
            w_impact=self._config.weights['impact'],
            w_plaus=self._config.weights['plaus'],
            w_div=self._config.weights['div'],
        )

        normalised = normalise_probabilities(selected)
        self._cache.set(key, normalised)
        return normalised

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _cache_key(self, book: BookSnapshot, regime: RegimeContext) -> str:
        payload = json.dumps({
            "book": book.__dict__,
            "regime": regime.__dict__,
            "config": {
                "bounds": self._config.bounds,
                "pack_size": self._config.pack_size,
                "weights": self._config.weights,
            },
        }, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()

    def _propose_candidates(self, book: BookSnapshot, regime: RegimeContext) -> List[ScenarioSpec]:
        prompt = self._build_prompt(book, regime)
        response = self._llm.generate(prompt)
        parsed = self._parse_response(response)
        return parsed

    def _build_prompt(self, book: BookSnapshot, regime: RegimeContext) -> str:
        return (
            "You are a senior risk manager. Generate stress scenarios for the current book.\n"
            f"Regime: {regime.label}, ATM IV {regime.atm_iv:.2f}.\n"
            f"Greek buckets: {book.greek_buckets}.\n"
            f"Moneyness mix: {book.moneyness_mix}.\n"
            f"Top exposures: {book.top_exposures}.\n"
            "Return a JSON list with fields: name, prob, spot_mult, iv_jump, smile_tilt, corr_break, liq_haircut, narrative."
        )

    def _parse_response(self, raw: str) -> List[ScenarioSpec]:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM returned invalid JSON: {exc}") from exc
        if isinstance(data, dict):
            data = data.get('scenarios', [])
        specs = []
        for item in data:
            try:
                specs.append(ScenarioSpec(
                    name=item.get('name', 'unnamed'),
                    prob=float(item.get('prob', item.get('weight', 0.0))),
                    spot_mult=float(item['spot_mult']),
                    iv_jump=float(item.get('iv_jump', 0.0)),
                    smile_tilt=float(item.get('smile_tilt', 0.0)),
                    corr_break=float(item.get('corr_break', 0.0)),
                    liq_haircut=float(item.get('liq_haircut', 0.0)),
                    narrative=str(item.get('narrative', '')),
                    meta=item.get('meta'),
                ))
            except KeyError:
                continue
        return specs


__all__ = [
    "UltraStressAgent",
    "BookSnapshot",
    "AgentConfig",
    "ScenarioCache",
]
