#!/usr/bin/env python3
"""
Production-Ready Enhanced Stress MC for CQF (no leaks, unit-safe, reproducible)

What this does
--------------
- Takes CQF quantile forecasts per contract: q0.05, q0.50, q0.95  (returns)
- Samples base outcome distribution that matches those quantiles (two-piece normal)
- Applies scenario shocks derived from SPY history (or defaults)
- Converts shocks to *return* space correctly (via option price)
- Aggregates risk on the *mixture* (not average of VaRs)
- Ranks contracts by a utility that penalizes downside (CVaR) and rewards skew

Key correctness details
-----------------------
- No global RNG resets; per-contract, per-scenario rng seeded deterministically
- No unit mix: shock_dollars -> shock_return using option price
- VaR/CVaR computed on scenario-mixture samples (coherent for a mixture)
- Optional parallelism via joblib (safe pickling, top-level functions)

Inputs
------
- contracts_df with columns (required):
    'contractID', 'q0.05', 'q0.50', 'q0.95'
  and (strongly recommended):
    'last_raw' or 'last' (option price),
    'delta', 'gamma', 'vega', 'theta', 'moneyness', 'date'
- spy_history (optional) with columns:
    'date' (datetime or parseable), 'close' (float)

CLI (example)
-------------
python enhanced_stress_mc.py \
  --contracts preds.csv \
  --spy-history spy.csv \
  --n-paths 8000 \
  --top-k 25 \
  --out ranked.csv
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except Exception:
    JOBLIB_AVAILABLE = False


# ----------------------------- Logging ---------------------------------

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("enhanced_stress_mc")


# ----------------------------- Config ----------------------------------

@dataclass
class StressConfig:
    """Configuration for stress testing and ranking."""
    n_paths: int = 5000
    risk_aversion: float = 0.5
    min_prob_profit: float = 0.45
    max_downside_var: float = 0.15     # allow at most -15% 5%-VaR in return space
    lookback_days: int = 252          # for SPY shock calibration
    random_seed: int = 42

    # Parallelism (joblib). If joblib not available, runs single-thread.
    n_jobs: int = 1                    # <= 1 means sequential
    batch_size: int = 128

    # Scenario probabilities (rough defaults). Should sum to ~1.0
    scenario_weights: Dict[str, float] = field(default_factory=lambda: {
        "base_case": 0.70,
        "mild_drop": 0.15,
        "sharp_drop": 0.10,
        "crash": 0.05,
    })

    # Pricing / Greek conventions
    vega_is_per_vol_point: bool = True     # True: per 1.00 vol point; False: per 1%
    theta_is_per_day: bool = True          # True if theta already per day
    horizon_days_for_theta: int = 1        # stress horizon for theta impact

    # Long-only: cap return at -100% to reflect limited loss on long options
    long_only: bool = True

    # Use dynamic SPY price from contracts_df when available
    use_dynamic_spy: bool = True

    # Utility tuning
    rtv_weight: float = 0.05           # reward better return-to-VaR ratios
    crash_penalty: float = 0.05        # penalise severe crash VaR


# -------------------------- Helper functions ---------------------------

def _stable_hash32(s: str) -> int:
    """Stable 32-bit hash for seeding RNG."""
    h = hashlib.sha256(str(s).encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _rng_for(contract_id: str, scenario: str, base_seed: int) -> np.random.Generator:
    """Deterministic, per-contract, per-scenario RNG."""
    seed_seq = np.random.SeedSequence(
        [base_seed, _stable_hash32(contract_id), _stable_hash32(scenario)]
    )
    return np.random.default_rng(seed_seq)


def _two_piece_normal_from_quantiles(q05: float, q50: float, q95: float,
                                     n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sample from a two-piece normal that matches tail widths implied by (q05, q50, q95).
    This preserves skew and spreads better than triangular + clipping.
    """
    if not np.isfinite(q05) or not np.isfinite(q50) or not np.isfinite(q95):
        return np.full(n, np.nan)

    if not (q05 <= q50 <= q95):
        q05, q50, q95 = sorted([q05, q50, q95])

    if (q95 - q05) < 1e-12:
        return np.full(n, q50)

    sL = max((q50 - q05) / 1.645, 1e-10)  # 5th percentile is ~ -1.645 sigma
    sR = max((q95 - q50) / 1.645, 1e-10)  # 95th percentile is ~ +1.645 sigma

    u = rng.random(n)
    z = np.abs(rng.standard_normal(n))
    left = q50 - z * sL
    right = q50 + z * sR
    return np.where(u < 0.5, left, right)


def _kurtosis_excess(x: np.ndarray) -> float:
    m = np.mean(x)
    s = np.std(x)
    if s == 0:
        return 0.0
    return float(np.mean(((x - m) / s) ** 4) - 3.0)


def _skewness(x: np.ndarray) -> float:
    m = np.mean(x)
    s = np.std(x)
    if s == 0:
        return 0.0
    return float(np.mean(((x - m) / s) ** 3))


# ------------------------------ Core -----------------------------------

class EnhancedStressMC:
    """Production-ready stress testing for ranking options using CQF outputs."""

    def __init__(self, config: Optional[StressConfig] = None):
        self.config = config or StressConfig()
        self._spy_cache: Dict[str, float] = {}
        self._shock_cache: Dict[str, Dict[str, Dict[str, float]]] = {}

        # Validate scenario weights
        s = sum(self.config.scenario_weights.values())
        if not (0.99 <= s <= 1.01):
            logger.warning(f"Scenario weights sum to {s:.3f}; normalizing.")
            total = max(s, 1e-9)
            self.config.scenario_weights = {
                k: v / total for k, v in self.config.scenario_weights.items()
            }

    # -------------------- Data access & validation ---------------------

    @staticmethod
    def _require_cols(df: pd.DataFrame, cols: List[str]):
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _validate_contracts(self, df: pd.DataFrame):
        self._require_cols(df, ["contractID", "q0.05", "q0.50", "q0.95"])
        # Strongly recommended for unit conversion
        if ("last_raw" not in df.columns) and ("last" not in df.columns):
            logger.warning("No option price found (last_raw/last). "
                           "Shocks will be skipped (unit conversion unavailable).")

    # ------------------------- Market inputs ---------------------------

    def _get_spy_price(self, contracts_df: pd.DataFrame, date: datetime) -> float:
        """
        Return SPY (or underlying) level for the given date.
        Tries several columns; robust fallback to medians.
        """
        if not self.config.use_dynamic_spy:
            return 400.0

        key = date.strftime("%Y-%m-%d")
        if key in self._spy_cache:
            return self._spy_cache[key]

        candidates = ["spy_d_close_raw", "spy_d_close", "spy_close", "underlying_price"]
        date_mask = contracts_df["date"] == date if "date" in contracts_df.columns else slice(None)

        for col in candidates:
            if col in contracts_df.columns:
                d = contracts_df.loc[date_mask, col]
                if len(d):
                    val = float(np.nanmedian(d.values))
                    if np.isfinite(val) and val > 0:
                        self._spy_cache[key] = val
                        return val

        # Cross-date median as final fallback
        for col in candidates:
            if col in contracts_df.columns:
                val = float(np.nanmedian(contracts_df[col].values))
                if np.isfinite(val) and val > 0:
                    logger.warning(f"Using cross-date median SPY from {col}: {val:.2f}")
                    self._spy_cache[key] = val
                    return val

        logger.warning("No SPY price available; using default 400.0")
        return 400.0

    def _default_shocks(self) -> Dict[str, Dict[str, float]]:
        """Default spot & IV shocks per scenario."""
        w = self.config.scenario_weights
        return {
            "base_case": {"prob": w["base_case"], "spot_mult": 1.00, "iv_shock": 0.00},
            "mild_drop": {"prob": w["mild_drop"], "spot_mult": 0.97, "iv_shock": 0.03},
            "sharp_drop": {"prob": w["sharp_drop"], "spot_mult": 0.93, "iv_shock": 0.06},
            "crash":     {"prob": w["crash"],     "spot_mult": 0.88, "iv_shock": 0.10},
        }

    def calibrate_shocks(self, spy_history: Optional[pd.DataFrame],
                         target_date: datetime) -> Dict[str, Dict[str, float]]:
        """
        Calibrate spot & IV shocks using recent SPY returns and vol regime.
        Relies only on data up to (target_date - 1) -> no leakage.
        """
        key = target_date.strftime("%Y-%m-%d")
        if key in self._shock_cache:
            return self._shock_cache[key]

        if spy_history is None or spy_history.empty:
            shocks = self._default_shocks()
            logger.warning("No SPY history provided - using default shocks")
            self._shock_cache[key] = shocks
            return shocks

        # Required cols
        if "date" not in spy_history.columns or "close" not in spy_history.columns:
            logger.warning("Spy history must have 'date' and 'close'; using defaults")
            shocks = self._default_shocks()
            self._shock_cache[key] = shocks
            return shocks

        # Filter lookback window BEFORE target date
        end_date = target_date - timedelta(days=1)
        start_date = end_date - timedelta(days=self.config.lookback_days)
        hist = spy_history.copy()
        hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
        mask = (hist["date"] >= start_date) & (hist["date"] <= end_date)
        hist = hist.loc[mask].sort_values("date")
        if len(hist) < 25:
            logger.warning(f"Insufficient SPY history ({len(hist)} days) - using defaults")
            shocks = self._default_shocks()
            self._shock_cache[key] = shocks
            return shocks

        hist["ret"] = hist["close"].pct_change()
        hist["vol20"] = hist["ret"].rolling(20).std() * np.sqrt(252)
        current_vol = float(hist["vol20"].dropna().iloc[-1]) if hist["vol20"].notna().any() else 0.20
        vol_mult = current_vol / 0.15  # relative to a "normal" 15% vol

        w = self.config.scenario_weights
        rets = hist["ret"].dropna()
        shocks = {
            "base_case": {
                "prob": w["base_case"],
                "spot_mult": 1.00,
                "iv_shock": 0.00,
            },
            "mild_drop": {
                "prob": w["mild_drop"],
                "spot_mult": 1.0 + float(np.percentile(rets, 20)) * vol_mult,
                "iv_shock": 0.02 * vol_mult,
            },
            "sharp_drop": {
                "prob": w["sharp_drop"],
                "spot_mult": 1.0 + float(np.percentile(rets, 5)) * vol_mult,
                "iv_shock": 0.05 * vol_mult,
            },
            "crash": {
                "prob": w["crash"],
                "spot_mult": 1.0 + float(np.percentile(rets, 1)) * vol_mult,
                "iv_shock": 0.08 * vol_mult,
            },
        }

        # Guardrails on spot multipliers to avoid pathological scenarios
        for params in shocks.values():
            params["spot_mult"] = float(np.clip(params["spot_mult"], 0.6, 1.1))

        logger.info(f"Shocks @ {key} (vol ~ {current_vol:.1%}) "
                    f"mild={shocks['mild_drop']['spot_mult']:.3f}, "
                    f"crash={shocks['crash']['spot_mult']:.3f}")
        self._shock_cache[key] = shocks
        return shocks

    # ------------------------- Greeks & shocks -------------------------

    def _option_shock_dollars(self, row: pd.Series, dS: float, dIV: float) -> float:
        """
        Dollar PnL impact from scenario shock using Taylor expansion:
            Δ dS + 1/2 Γ dS^2 + Vega * dIV + Theta * horizon_days
        dIV is absolute vol change (e.g., +0.05 for +5 vol points)
        """
        delta = float(row.get("delta", 0.0) or 0.0)
        gamma = float(row.get("gamma", 0.0) or 0.0)
        vega = float(row.get("vega", 0.0) or 0.0)
        theta = float(row.get("theta", 0.0) or 0.0)

        # If vega is quoted per 1% vol, multiply dIV by 100; else leave as vol point
        if not self.config.vega_is_per_vol_point:
            vega_multiplier = 100.0
        else:
            vega_multiplier = 1.0

        # Theta horizon (if theta already per day, multiply by horizon_days)
        theta_effect = theta * (self.config.horizon_days_for_theta if self.config.theta_is_per_day else 1.0)

        pnl = delta * dS + 0.5 * gamma * (dS ** 2) + vega * (dIV * vega_multiplier) + theta_effect

        # Optional sensitivity boost for far OTM/ITM (vol-of-vol proxy)
        m = float(row.get("moneyness", 1.0) or 1.0)
        if abs(m - 1.0) > 0.2:
            pnl *= (1.0 + 0.1 * abs(m - 1.0))

        return float(pnl)

    # -------------------- Per-contract simulation ----------------------

    def _simulate_one_contract(self,
                               row: pd.Series,
                               scenarios: Dict[str, Dict[str, float]],
                               spy_price: float) -> Dict[str, object]:
        """
        Simulate mixture returns for one contract and compute metrics.
        Returns a dict suitable for aggregation into a DataFrame.
        """
        cid = str(row["contractID"])

        # Option price (for unit conversion)
        price = float(row.get("last_raw", row.get("last", np.nan)))
        have_price = np.isfinite(price) and price > 0

        # Base distribution in *return* space from CQF quantiles
        rng_base = _rng_for(cid, "base", self.config.random_seed)
        base_returns = _two_piece_normal_from_quantiles(
            float(row["q0.05"]), float(row["q0.50"]), float(row["q0.95"]),
            self.config.n_paths, rng_base
        )

        # If price is missing, skip shocks and rank on base only
        if not have_price:
            mix = base_returns.copy()
            # For long-only, floor at -100%
            if self.config.long_only:
                mix = np.maximum(mix, -1.0)
            return self._metrics_from_samples(mix, {}, cid, include_crash=False)

        # Build mixture samples according to scenario probabilities
        # We create scenario-specific samples and slice counts by probability
        scenario_names = list(scenarios.keys())
        if "base_case" in scenario_names:
            scenario_names.remove("base_case")
            scenario_names.insert(0, "base_case")

        counts: Dict[str, int] = {}
        remaining = self.config.n_paths
        for idx, name in enumerate(scenario_names):
            if idx == len(scenario_names) - 1:
                count = max(1, remaining)
            else:
                prob = float(scenarios[name]["prob"])
                count = max(1, int(round(self.config.n_paths * prob)))
                min_remaining = len(scenario_names) - idx - 1
                count = min(count, max(1, remaining - min_remaining))
                remaining -= count
            counts[name] = count

        total_count = sum(counts.values())
        if total_count != self.config.n_paths:
            counts[scenario_names[-1]] += (self.config.n_paths - total_count)
            total_count = sum(counts.values())

        rng_base = _rng_for(cid, "base", self.config.random_seed)
        base_returns = _two_piece_normal_from_quantiles(
            float(row["q0.05"]), float(row["q0.50"]), float(row["q0.95"]),
            total_count, rng_base
        )

        samples_per_scenario: Dict[str, np.ndarray] = {}
        offset = 0
        for name in scenario_names:
            count = counts[name]
            slice_returns = base_returns[offset:offset + count].copy()
            offset += count

            if name != "base_case":
                rng = _rng_for(cid, name, self.config.random_seed)
                dS = spy_price * (scenarios[name]["spot_mult"] - 1.0)
                dIV = float(scenarios[name]["iv_shock"])
                shock_dollars = self._option_shock_dollars(row, dS, dIV)
                shock_ret = shock_dollars / price
                z = rng.standard_normal(count)
                slice_returns += shock_ret * (1.0 + 0.2 * z)

            if self.config.long_only:
                slice_returns = np.maximum(slice_returns, -1.0)

            samples_per_scenario[name] = slice_returns

        mix = np.concatenate([samples_per_scenario[name] for name in scenario_names], axis=0)

        # Compute overall metrics and also keep crash-only stats for transparency
        crash_stats = None
        if "crash" in samples_per_scenario:
            crash = samples_per_scenario["crash"]
            crash_var = np.percentile(crash, 5)
            crash_stats = {
                "crash_mean": float(np.mean(crash)),
                "crash_var95": float(crash_var),
            }

        metrics = self._metrics_from_samples(mix, crash_stats or {}, cid, include_crash=True)
        return metrics

    @staticmethod
    def _metrics_from_samples(mix: np.ndarray,
                              extra: Dict[str, float],
                              contract_id: str,
                              include_crash: bool) -> Dict[str, object]:
        """Compute risk/return metrics from mixture samples."""
        mean = float(np.mean(mix))
        std = float(np.std(mix))
        var95 = float(np.percentile(mix, 5))
        tail = mix[mix <= var95]
        cvar95 = float(np.mean(tail)) if len(tail) else var95
        prob_profit = float(np.mean(mix > 0))
        skew = _skewness(mix)
        kurt = _kurtosis_excess(mix)

        # Utility: expected - λ * downside (CVaR) + bonus for positive skew
        # λ will be set by config at ranking time; keep just the components here.
        # We'll finalize utility in rank_contracts.
        result = {
            "contractID": contract_id,
            "expected_pnl": mean,
            "std": std,
            "var_95": var95,
            "cvar_95": cvar95,
            "prob_profit": prob_profit,
            "skew": skew,
            "kurtosis": kurt,
            "return_to_var": mean / (abs(var95) + 1e-9),
        }
        if include_crash:
            result.update(extra)
        return result

    # ----------------------------- Public API ---------------------------

    def rank_contracts(self,
                       contracts_df: pd.DataFrame,
                       spy_history: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Rank contracts with stress MC.
        Returns a DataFrame of metrics sorted by 'utility_score' (desc).
        """
        self._validate_contracts(contracts_df)

        # Target date for shocks/SPY reference
        if "date" in contracts_df.columns and len(contracts_df["date"].dropna()):
            target_date = pd.to_datetime(contracts_df["date"].iloc[0])
        else:
            target_date = datetime.now()

        spy_price = self._get_spy_price(contracts_df, target_date)
        shocks = self.calibrate_shocks(spy_history, target_date)

        rows = list(contracts_df.itertuples(index=False, name=None))
        cols = list(contracts_df.columns)

        def _row_to_series(tpl) -> pd.Series:
            return pd.Series(dict(zip(cols, tpl)))

        # Process contracts (parallel if requested and joblib available)
        if self.config.n_jobs and self.config.n_jobs > 1 and JOBLIB_AVAILABLE:
            logger.info(f"Ranking {len(rows)} contracts with n_jobs={self.config.n_jobs}")
            chunks = [rows[i:i + self.config.batch_size] for i in range(0, len(rows), self.config.batch_size)]

            def _process_chunk(chunk):
                out = []
                for tpl in chunk:
                    try:
                        s = _row_to_series(tpl)
                        out.append(self._simulate_one_contract(s, shocks, spy_price))
                    except Exception as e:
                        logger.error(f"Contract failed: {e}")
                return out

            all_results: List[Dict[str, object]] = []
            for part in Parallel(n_jobs=self.config.n_jobs, prefer="processes")(
                    delayed(_process_chunk)(c) for c in chunks):
                all_results.extend(part)
        else:
            logger.info(f"Ranking {len(rows)} contracts sequentially")
            all_results = []
            for tpl in rows:
                try:
                    s = _row_to_series(tpl)
                    all_results.append(self._simulate_one_contract(s, shocks, spy_price))
                except Exception as e:
                    logger.error(f"Contract failed: {e}")

        if not all_results:
            raise ValueError("No contracts successfully processed")

        res = pd.DataFrame(all_results)

        # Filters
        pre = len(res)
        res = res[
            (res["prob_profit"] >= self.config.min_prob_profit) &
            (res["var_95"] >= -self.config.max_downside_var)
        ].copy()
        logger.info(f"Filtered {pre} → {len(res)} contracts")

        # Utility score (apply λ and skew bonus)
        skew_bonus = 0.1 * np.maximum(res["skew"].values, 0.0)
        rtv = np.clip(res["return_to_var"].values, -5.0, 5.0)
        crash_var = np.abs(res.get("crash_var95", pd.Series(0.0, index=res.index)).values)
        downside = np.abs(np.minimum(res["cvar_95"].values, 0.0))
        res["utility_score"] = (
            res["expected_pnl"]
            - self.config.risk_aversion * downside
            + skew_bonus
            + self.config.rtv_weight * rtv
            - self.config.crash_penalty * crash_var
        )

        res = res.sort_values("utility_score", ascending=False).reset_index(drop=True)
        res["rank"] = np.arange(1, len(res) + 1)
        logger.info(
            "Utility summary – mean: %.4f, min: %.4f, max: %.4f",
            float(res["utility_score"].mean()),
            float(res["utility_score"].min()),
            float(res["utility_score"].max()),
        )
        return res


# ------------------------- Data Processing Layer ---------------------

class SourceDataProcessor:
    """Processes raw source data to extract SPY prices and history for Enhanced Stress MC."""
    
    def __init__(self, source_data_dir: str = "data_combined"):
        self.source_data_dir = source_data_dir
        self._spy_history_cache = None
        self._raw_data_cache = {}
        
    def get_spy_history(self) -> pd.DataFrame:
        """Extract SPY daily history from source data for shock calibration."""
        if self._spy_history_cache is not None:
            return self._spy_history_cache
            
        logger.info("Extracting SPY history from source data...")
        
        # Load both train and eval data to get full SPY history
        train_path = f"{self.source_data_dir}/train_2019_2023.csv"
        eval_path = f"{self.source_data_dir}/eval_2024_2025.csv"
        
        spy_data = []
        
        for path in [train_path, eval_path]:
            try:
                # Read only date and SPY columns for efficiency
                cols_to_read = ['date', 'spy_d_close', 'spy_d_open', 'spy_d_high', 'spy_d_low']
                df = pd.read_csv(path, usecols=cols_to_read, low_memory=False)
                
                # Get unique date-level SPY data
                daily_spy = df.groupby('date').agg({
                    'spy_d_close': 'first',  # All contracts same date have same SPY prices
                    'spy_d_open': 'first',
                    'spy_d_high': 'first', 
                    'spy_d_low': 'first'
                }).reset_index()
                
                spy_data.append(daily_spy)
                logger.info(f"Loaded SPY data from {path}: {len(daily_spy)} days")
                
            except Exception as e:
                logger.warning(f"Could not load SPY data from {path}: {e}")
        
        if not spy_data:
            logger.error("No SPY history available from source data")
            return pd.DataFrame()
            
        # Combine and clean
        full_spy = pd.concat(spy_data, ignore_index=True)
        full_spy['date'] = pd.to_datetime(full_spy['date'])
        full_spy = full_spy.drop_duplicates('date').sort_values('date')
        
        # Rename for Enhanced Stress MC compatibility
        full_spy = full_spy.rename(columns={'spy_d_close': 'close'})
        
        logger.info(f"SPY history: {len(full_spy)} days, range {full_spy['date'].min()} to {full_spy['date'].max()}")
        logger.info(f"SPY price range: {full_spy['close'].min():.1f} to {full_spy['close'].max():.1f}")
        
        self._spy_history_cache = full_spy
        return full_spy
    
    def enrich_contracts_with_raw_data(self, contracts_df: pd.DataFrame, 
                                     target_period: str = "eval") -> pd.DataFrame:
        """Merge CQF predictions with raw SPY prices from source data."""
        
        cache_key = target_period
        if cache_key not in self._raw_data_cache:
            logger.info(f"Loading raw data for {target_period} period...")
            
            if target_period == "eval":
                raw_path = f"{self.source_data_dir}/eval_2024_2025.csv"
            else:
                raw_path = f"{self.source_data_dir}/train_2019_2023.csv"
                
            # Load raw data with key columns
            key_cols = ['date', 'contractID', 'spy_d_close', 'spy_d_open', 'spy_d_high', 'spy_d_low']
            try:
                raw_df = pd.read_csv(raw_path, usecols=key_cols, low_memory=False)
                raw_df['date'] = pd.to_datetime(raw_df['date'])
                self._raw_data_cache[cache_key] = raw_df
                logger.info(f"Loaded {len(raw_df)} raw records from {raw_path}")
            except Exception as e:
                logger.error(f"Failed to load raw data from {raw_path}: {e}")
                return contracts_df
        
        raw_df = self._raw_data_cache[cache_key]
        
        # Merge CQF predictions with raw SPY data
        contracts_df = contracts_df.copy()
        if 'date' in contracts_df.columns:
            contracts_df['date'] = pd.to_datetime(contracts_df['date'])
        
        # Merge on contractID + date
        merge_cols = ['contractID', 'date']
        if all(col in contracts_df.columns for col in merge_cols) and all(col in raw_df.columns for col in merge_cols):
            
            # Add raw SPY columns with _raw suffix to avoid conflicts
            spy_raw_cols = ['spy_d_close_raw', 'spy_d_open_raw', 'spy_d_high_raw', 'spy_d_low_raw']
            raw_df_rename = raw_df.rename(columns={
                'spy_d_close': 'spy_d_close_raw',
                'spy_d_open': 'spy_d_open_raw',
                'spy_d_high': 'spy_d_high_raw', 
                'spy_d_low': 'spy_d_low_raw'
            })
            
            enhanced_df = contracts_df.merge(
                raw_df_rename[merge_cols + spy_raw_cols], 
                on=merge_cols, 
                how='left'
            )
            
            # Check merge success
            spy_available = enhanced_df['spy_d_close_raw'].notna().sum()
            logger.info(f"Enhanced {len(enhanced_df)} contracts with raw SPY data ({spy_available} with SPY prices)")
            
            return enhanced_df
        else:
            logger.warning("Cannot merge - missing contractID or date columns")
            return contracts_df


# -------------------------------- CLI ----------------------------------

def _read_contracts(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    # Ensure required types
    for c in ["date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def _read_spy_history(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    df = pd.read_csv(path, low_memory=False)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def main():
    p = argparse.ArgumentParser(description="Enhanced Stress MC ranking for CQF outputs")
    p.add_argument("--contracts", required=True, help="CSV with CQF quantiles & Greeks per contract")
    p.add_argument("--spy-history", default=None, help="CSV with columns date,close for SPY (optional - will use source data)")
    p.add_argument("--source-data-dir", default="data_combined", help="Directory with raw source data for SPY history")
    p.add_argument("--n-paths", type=int, default=5000)
    p.add_argument("--risk-aversion", type=float, default=0.5)
    p.add_argument("--min-prob-profit", type=float, default=0.45)
    p.add_argument("--max-downside-var", type=float, default=0.15)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--out", default="ranked_contracts.csv")
    p.add_argument("--top-k", type=int, default=50)
    args = p.parse_args()

    cfg = StressConfig(
        n_paths=args.n_paths,
        risk_aversion=args.risk_aversion,
        min_prob_profit=args.min_prob_profit,
        max_downside_var=args.max_downside_var,
        n_jobs=args.n_jobs
    )

    # Initialize data processor and Enhanced Stress MC
    processor = SourceDataProcessor(args.source_data_dir)
    mc = EnhancedStressMC(cfg)
    
    # Load and enrich contracts with raw SPY data
    contracts = _read_contracts(args.contracts)
    logger.info(f"Loaded {len(contracts)} contracts from {args.contracts}")
    
    # Enrich with raw SPY prices from source data
    contracts_enhanced = processor.enrich_contracts_with_raw_data(contracts, target_period="eval")
    
    # Get SPY history from source data (prioritize over manual spy-history file)
    if args.spy_history:
        logger.info("Using provided SPY history file")
        spy_hist = _read_spy_history(args.spy_history)
    else:
        logger.info("Extracting SPY history from source data")
        spy_hist = processor.get_spy_history()

    # Rank contracts with enhanced data
    ranked = mc.rank_contracts(contracts_enhanced, spy_hist)
    if args.top_k and args.top_k > 0:
        ranked = ranked.head(args.top_k).copy()

    ranked.to_csv(args.out, index=False)
    logger.info(f"Saved ranked contracts → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
