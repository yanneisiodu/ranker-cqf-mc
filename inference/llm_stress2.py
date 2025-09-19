"""LLM-driven stress testing utilities.

This module wires together a scenario generator (LLM-backed or fallback),
a validator that enforces bounds and probability normalization, and a
numerical engine that converts scenarios into option PnL risk metrics via
Greeks approximation. It is deliberately self-contained so the higher-level
pipeline can run the engine in shadow mode alongside the legacy Monte Carlo
implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence

import logging

import numpy as np
import pandas as pd

from inference.ultra_agent.agent import (
    AgentConfig,
    BookSnapshot,
    RegimeContext,
    ScenarioCache,
    UltraStressAgent,
)
from inference.ultra_agent.oracle_adapter import OracleAdapter
from inference.ultra_agent.plausibility import PlausibilityScorer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class StressScenario:
    """Single stress scenario expressed as underlying returns and IV change."""

    underlying_return_pct: float  # expressed in percentage points (e.g. -6.5)
    vol_change: float             # absolute change in vol points (e.g. +4.0)
    scenario_prob: float          # probability weight in [0, 1]
    narrative: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScenarioConfig:
    """Configuration for scenario generation and validation."""

    max_scenarios: int = 6
    spot_return_bounds: tuple[float, float] = (-0.5, 0.5)  # +/- 50%
    vol_change_bounds: tuple[float, float] = (-0.4, 0.6)
    min_prob: float = 1e-4
    group_key: str = "underlying"  # fallbacks if column missing handled later
    horizon_days: int = 5
    fallback_returns: Sequence[float] = field(
        default_factory=lambda: [0.0, -5.0, -12.0, 5.0]
    )
    fallback_vol_points: Sequence[float] = field(
        default_factory=lambda: [0.0, 3.0, 5.5, -2.5]
    )


@dataclass
class LLMStressConfig:
    """Tuning parameters for the deterministic PnL computation."""

    risk_aversion: float = 0.5
    min_prob_profit: float = 0.45
    max_downside_var: Optional[float] = 0.15
    skew_bonus: float = 0.0
    horizon_days: int = 5
    prob_calibrator: Optional[Any] = None
    var_adjust: float = 0.0
    cvar_adjust: float = 0.0


# ---------------------------------------------------------------------------
# Scenario generator protocol and helpers
# ---------------------------------------------------------------------------


class LLMClient(Protocol):
    """Protocol for the underlying LLM client."""

    def generate(self, prompt: str, **kwargs: Any) -> str:
        ...


class _GroupOracle:
    """Lightweight oracle that aggregates group Greeks for agent scoring."""

    def __init__(self, group: pd.DataFrame, context: Dict[str, Any], stress_config: LLMStressConfig):
        self.delta = float(group.get('delta', pd.Series([0.0])).astype(float).sum())
        self.gamma = float(group.get('gamma', pd.Series([0.0])).astype(float).sum())
        self.vega = float(group.get('vega', pd.Series([0.0])).astype(float).sum())
        self.theta = float(group.get('theta', pd.Series([0.0])).astype(float).sum())
        first = group.iloc[0]
        fallback_price = float(first.get('future_option_price', 0.0) or 0.0)
        self.underlying_price = _extract_underlying_price(first, context, fallback_price)
        self.horizon_days = stress_config.horizon_days

    def run_scenario(self, scenario: Dict[str, float]) -> Dict[str, np.ndarray]:
        spot_mult = float(scenario.get('spot_mult', 1.0))
        iv_jump = float(scenario.get('iv_jump', 0.0))
        dS = self.underlying_price * (spot_mult - 1.0)
        pnl = (
            self.delta * dS
            + 0.5 * self.gamma * (dS ** 2)
            + self.vega * iv_jump
            + self.theta * self.horizon_days
        )
        return {'portfolio_pnl': np.array([pnl]), 'book_pnl': {}}


def weighted_percentile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Compute weighted percentile with q in [0, 1]."""
    if values.size == 0:
        return float("nan")
    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]
    cum_weights = np.cumsum(weights)
    target = q * weights.sum()
    idx = np.searchsorted(cum_weights, target, side="right")
    idx = min(idx, len(values) - 1)
    return float(values[idx])


def _normalise_probs(scenarios: List[StressScenario], min_prob: float) -> List[StressScenario]:
    weights = [max(s.scenario_prob, 0.0) for s in scenarios]
    total = sum(weights)
    if total <= 0.0:
        # equal weights
        n = len(scenarios)
        for s in scenarios:
            s.scenario_prob = 1.0 / n
        return scenarios
    for s, w in zip(scenarios, weights):
        s.scenario_prob = max(min_prob, w / total)
    # renormalise after min_prob clamp
    total = sum(s.scenario_prob for s in scenarios)
    for s in scenarios:
        s.scenario_prob /= total if total > 0 else 1.0
    return scenarios


class ScenarioValidator:
    """Apply clipping, probability normalisation, and fallback injection."""

    def __init__(self, config: ScenarioConfig):
        self.config = config

    def validate(self,
                 scenarios: List[StressScenario],
                 fallback: Iterable[StressScenario]) -> List[StressScenario]:
        """Validate the provided scenarios; use fallback if needed."""
        valid: List[StressScenario] = []
        for sc in scenarios:
            spot = max(self.config.spot_return_bounds[0] * 100,
                       min(self.config.spot_return_bounds[1] * 100,
                           sc.underlying_return_pct))
            vol = max(self.config.vol_change_bounds[0],
                      min(self.config.vol_change_bounds[1], sc.vol_change))
            prob = max(self.config.min_prob, sc.scenario_prob)
            valid.append(StressScenario(spot, vol, prob, sc.narrative, sc.meta))
        if not valid:
            valid = list(fallback)
        else:
            valid = valid[: self.config.max_scenarios]
        return _normalise_probs(valid, self.config.min_prob)


class ScenarioCache:
    """Simple per-key scenario cache to avoid redundant LLM calls."""

    def __init__(self) -> None:
        self._cache: Dict[str, List[StressScenario]] = {}

    def get(self, key: str) -> Optional[List[StressScenario]]:
        return self._cache.get(key)

    def set(self, key: str, scenarios: List[StressScenario]) -> None:
        self._cache[key] = scenarios


class ScenarioEngine:
    """High-level scenario generator using an optional LLM."""

    def __init__(
        self,
        config: ScenarioConfig,
        llm_client: Optional[LLMClient] = None,
    ) -> None:
        self.config = config
        self.llm_client = llm_client
        self.cache = ScenarioCache()

    def build_fallback(self) -> List[StressScenario]:
        fallback: List[StressScenario] = []
        vols = list(self.config.fallback_vol_points)
        # pad vol list if needed
        if len(vols) < len(self.config.fallback_returns):
            vols = vols + [vols[-1]] * (len(self.config.fallback_returns) - len(vols))
        for ret, vol in zip(self.config.fallback_returns, vols):
            fallback.append(StressScenario(ret, vol, 1.0 / len(self.config.fallback_returns)))
        return fallback

    def _build_prompt(self,
                      key: str,
                      group: pd.DataFrame,
                      context: Dict[str, Any]) -> str:
        summary = {
            'count': len(group),
            'mean_q05': float(group['q0.05'].astype(float).mean()),
            'mean_q95': float(group['q0.95'].astype(float).mean()),
            'mean_iv': float(group.get('implied_volatility', pd.Series([np.nan])).astype(float).mean()),
            'regime': context.get('regime', 'unknown'),
        }
        analogs = context.get('analogs', 'none')
        prompt = (
            f"Generate stress scenarios for underlying {key}. Provide JSON list with fields "
            "underlying_return_pct, vol_change, scenario_prob, narrative.\n"
            f"Stats: {summary}. Historical analogs: {analogs}\n"
            "Ensure probabilities sum to 1 and include downside crash narrative."
        )
        return prompt

    def _call_llm(self, prompt: str) -> Optional[str]:
        if self.llm_client is None:
            return None
        try:
            # GPT-5 doesn't support temperature parameter, pass it through generate kwargs only for supported models
            kwargs = {} if hasattr(self.llm_client, '_model') and self.llm_client._model.startswith('gpt-5') else {'temperature': 0}
            return self.llm_client.generate(prompt, **kwargs)
        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            return None

    @staticmethod
    def _parse_response(response: str) -> List[StressScenario]:
        try:
            import json
            data = json.loads(response)
        except Exception:
            return []
        scenarios: List[StressScenario] = []
        if isinstance(data, dict):
            data = data.get('scenarios', [])
        if not isinstance(data, list):
            return []
        for item in data:
            try:
                scenarios.append(
                    StressScenario(
                        underlying_return_pct=float(item['underlying_return_pct']),
                        vol_change=float(item.get('vol_change', 0.0)),
                        scenario_prob=float(item.get('scenario_prob', item.get('weight', 0.0))),
                        narrative=str(item.get('narrative', '')),
                        meta={k: v for k, v in item.items() if k not in {
                            'underlying_return_pct', 'vol_change', 'scenario_prob', 'weight', 'narrative'
                        }}
                    )
                )
            except Exception:
                continue
        return scenarios

    def scenarios_for_group(self,
                            key: str,
                            group: pd.DataFrame,
                            context: Dict[str, Any]) -> List[StressScenario]:
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        prompt = self._build_prompt(key, group, context)
        raw = self._call_llm(prompt)
        if raw is None:
            scenarios: List[StressScenario] = []
        else:
            scenarios = self._parse_response(raw)
        if not scenarios:
            scenarios = self.build_fallback()
        self.cache.set(key, scenarios)
        return scenarios


# ---------------------------------------------------------------------------
# Stress computation
# ---------------------------------------------------------------------------


def _resolve_group_key(df: pd.DataFrame, candidate: str) -> str:
    if candidate in df.columns:
        return candidate
    for fallback in ("underlying", "underlying_symbol", "ticker", "root", "contractID"):
        if fallback in df.columns:
            return fallback
    return "contractID"


def _extract_underlying_price(row: pd.Series, context: Dict[str, Any], fallback_price: float) -> float:
    for col in ("underlying_price", "spot", "spot_price", "underlying_last"):
        if col in row and pd.notna(row[col]):
            return float(row[col])
    prices = context.get('underlying_prices')
    if isinstance(prices, dict):
        underlying_key = row.get(context.get('group_key', 'underlying'))
        if underlying_key in prices:
            return float(prices[underlying_key])
    return fallback_price


def _compute_pnl(row: pd.Series,
                 scenarios: Sequence[StressScenario],
                 config: LLMStressConfig,
                 context: Dict[str, Any]) -> Dict[str, float]:
    delta = float(row.get('delta', 0.0) or 0.0)
    gamma = float(row.get('gamma', 0.0) or 0.0)
    vega = float(row.get('vega', 0.0) or 0.0)
    theta = float(row.get('theta', 0.0) or 0.0)
    option_price = float(row.get('last', row.get('last_raw', 0.0)) or 0.0)
    fallback_price = option_price if option_price > 0 else float(row.get('future_option_price', 0.0) or 0.0)
    underlying_price = _extract_underlying_price(row, context, fallback_price)

    pnl_vals: List[float] = []
    weights: List[float] = []

    for sc in scenarios:
        ret = sc.underlying_return_pct / 100.0
        spot_new = underlying_price * (1.0 + ret)
        dS = spot_new - underlying_price
        dv = sc.vol_change
        pnl = delta * dS + 0.5 * gamma * (dS ** 2) + vega * dv + theta * config.horizon_days
        pnl_vals.append(pnl)
        weights.append(sc.scenario_prob)

    pnl_arr = np.asarray(pnl_vals, dtype=float)
    weight_arr = np.asarray(weights, dtype=float)
    if pnl_arr.size == 0:
        pnl_arr = np.zeros(1)
        weight_arr = np.ones(1)

    weight_arr = weight_arr / weight_arr.sum() if weight_arr.sum() > 0 else np.ones_like(weight_arr) / len(weight_arr)

    exp_pnl = float(np.dot(weight_arr, pnl_arr))
    std = float(np.sqrt(np.dot(weight_arr, (pnl_arr - exp_pnl) ** 2)))

    var_95 = weighted_percentile(pnl_arr, weight_arr, 0.05)
    tail_mask = pnl_arr <= var_95 + 1e-12
    if tail_mask.any():
        tail_weights = weight_arr[tail_mask]
        tail_sum = tail_weights.sum()
        cvar_95 = float(np.dot(tail_weights, pnl_arr[tail_mask]) / tail_sum) if tail_sum > 0 else var_95
    else:
        cvar_95 = var_95

    prob_profit = float(np.dot(weight_arr, (pnl_arr > 0).astype(float)))
    if config.prob_calibrator is not None:
        try:
            prob_profit = float(config.prob_calibrator.predict([prob_profit])[0])
        except Exception as exc:
            logger.warning("Prob calibrator failed: %s", exc)

    var_95 += config.var_adjust
    cvar_95 += config.cvar_adjust

    if std > 1e-12:
        m3 = np.dot(weight_arr, (pnl_arr - exp_pnl) ** 3)
        m4 = np.dot(weight_arr, (pnl_arr - exp_pnl) ** 4)
        skew = float(m3 / (std ** 3))
        kurt = float(m4 / (std ** 4))
    else:
        skew, kurt = 0.0, 0.0

    crash_loss = float(pnl_arr.min())

    utility = exp_pnl - config.risk_aversion * cvar_95 + config.skew_bonus * skew

    violated = False
    if prob_profit < config.min_prob_profit:
        violated = True
    if config.max_downside_var is not None and var_95 < -abs(config.max_downside_var):
        violated = True
    if violated:
        utility = -1e9

    return {
        'expected_pnl': exp_pnl,
        'std': std,
        'var_95': var_95,
        'cvar_95': cvar_95,
        'prob_profit': max(0.0, min(1.0, prob_profit)),
        'skew': skew,
        'kurt': kurt,
        'crash_loss': crash_loss,
        'return_to_var': float(exp_pnl / (abs(var_95) + 1e-6)) if var_95 != 0 else float('nan'),
        'crash_var95': float(var_95),
        'utility_score': utility,
    }


class LLMStressEngine:
    """High-level orchestrator that glues scenario generation + risk metrics."""

    def __init__(
        self,
        scenario_engine: ScenarioEngine,
        validator: ScenarioValidator,
        stress_config: LLMStressConfig,
    ) -> None:
        self.scenario_engine = scenario_engine
        self.validator = validator
        self.stress_config = stress_config
        self._agent_cache = ScenarioCache()
        self._agent_config = AgentConfig.default()

    def evaluate(self,
                 contracts_df: pd.DataFrame,
                 context: Optional[Dict[str, Any]] = None,
                 store_scenarios: bool = True) -> pd.DataFrame:
        context = context or {}
        group_key = _resolve_group_key(contracts_df, self.scenario_engine.config.group_key)
        context['group_key'] = group_key
        records: List[Dict[str, Any]] = []
        scenario_records: List[Dict[str, Any]] = []

        use_agent = bool(context.get('use_agent', False))

        for key, group in contracts_df.groupby(group_key, dropna=False):
            group_context = self._build_group_context(key, group, context)
            if use_agent and self.scenario_engine.llm_client is not None:
                scenarios = self._generate_agent_scenarios(group, group_context)
            else:
                scenarios = self.scenario_engine.scenarios_for_group(key, group, group_context)
                scenarios = self.validator.validate(scenarios, self.scenario_engine.build_fallback())
            for _, row in group.iterrows():
                metrics = _compute_pnl(row, scenarios, self.stress_config, group_context)
                metrics['contractID'] = row['contractID']
                metrics['underlying_key'] = key
                records.append(metrics)
            if store_scenarios:
                scenario_records.append({
                    'underlying_key': key,
                    'scenarios': [s.__dict__ for s in scenarios],
                    'context': group_context,
                })

        results = pd.DataFrame(records)
        if store_scenarios:
            results['_scenario_log'] = results['underlying_key'].map(
                {rec['underlying_key']: rec['scenarios'] for rec in scenario_records}
            )
        return results

    @staticmethod
    def _build_group_context(key: Any,
                             group: pd.DataFrame,
                             context: Dict[str, Any]) -> Dict[str, Any]:
        group_ctx = dict(context)
        group_ctx['group_key_value'] = key
        group_ctx['sample_size'] = len(group)
        group_ctx.setdefault('regime', context.get('regime'))
        group_ctx.setdefault('analogs', context.get('analogs', ''))
        if 'date' in group:
            group_ctx['latest_date'] = str(group['date'].max())
        return group_ctx

    def _generate_agent_scenarios(self, group: pd.DataFrame, context: Dict[str, Any]) -> List[StressScenario]:
        oracle = OracleAdapter(_GroupOracle(group, context, self.stress_config))
        regime_label = context.get('regime', 'unknown') or 'unknown'
        atm_iv = float(group.get('implied_volatility', pd.Series([0.2])).astype(float).mean())
        book = BookSnapshot(
            greek_buckets={
                'delta': float(group.get('delta', pd.Series([0.0])).astype(float).sum()),
                'gamma': float(group.get('gamma', pd.Series([0.0])).astype(float).sum()),
                'vega': float(group.get('vega', pd.Series([0.0])).astype(float).sum()),
                'theta': float(group.get('theta', pd.Series([0.0])).astype(float).sum()),
            },
            moneyness_mix=self._moneyness_mix(group),
            top_exposures=[regime_label],
        )
        regime_obj = RegimeContext(atm_iv=atm_iv, label=regime_label)
        plausibility = PlausibilityScorer(mean=np.zeros(5), cov=np.eye(5) * 0.1)
        agent = UltraStressAgent(
            self.scenario_engine.llm_client,
            oracle,
            plausibility,
            self._agent_cache,
            self._agent_config,
        )
        try:
            pack = agent.build_pack(book, regime_obj)
        except Exception as exc:
            logger.warning("Agent scenario generation failed (%s); reverting to baseline scenario engine.", exc)
            fallback = self.scenario_engine.build_fallback()
            return self.validator.validate(fallback, fallback)
        scenarios = [
            StressScenario(
                underlying_return_pct=(spec.spot_mult - 1.0) * 100.0,
                vol_change=spec.iv_jump,
                scenario_prob=spec.prob,
                narrative=spec.narrative,
                meta={'name': spec.name},
            )
            for spec in pack
        ]
        return self.validator.validate(scenarios, self.scenario_engine.build_fallback())

    @staticmethod
    def _moneyness_mix(group: pd.DataFrame) -> Dict[str, float]:
        m = group.get('moneyness')
        if m is None:
            return {}
        m = m.astype(float)
        total = len(m)
        if total == 0:
            return {}
        buckets = {
            'deep_otm': float((m < 0.85).sum()) / total,
            'otm': float(((m >= 0.85) & (m < 0.97)).sum()) / total,
            'atm': float(((m >= 0.97) & (m <= 1.03)).sum()) / total,
            'itm': float((m > 1.03).sum()) / total,
        }
        return buckets


__all__ = [
    'StressScenario',
    'ScenarioConfig',
    'LLMStressConfig',
    'ScenarioValidator',
    'ScenarioEngine',
    'LLMStressEngine',
]
