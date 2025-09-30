#!/usr/bin/env python3
"""End-to-end offline IQL training on CQF + Ranker outputs.

Usage (example):

Precomputed inference artifacts:
    python3 iql_pipeline.py \
        --cqf-preds inference_output_june2023_30pct_risk/cqf_predictions.csv \
        --ranker-candidates inference_output_june2023_30pct_risk/ranker_candidates.csv \
        --stress-metrics inference_output_june2023_30pct_risk/stress_metrics_llm.csv \
        --outdir iql_artifacts

End-to-end (run inference first):
    python3 iql_pipeline.py \
        --raw-data year_2023_data.csv \
        --ranker-model model_output/xgboost_ranker_2022_2022_fixed_params_20250918_185021.joblib \
        --ranker-features model_output/xgb_feature_names_2022_2022_20250918_185021.pkl \
        --sharpe-edges model_output/sharpe_qcut_edges_2022_2022_20250918_185021.pkl \
        --cqf-model model_output/optimal_cqf_step8.joblib \
        --outdir iql_artifacts \
        --group-top-n 50 \
        --stress-mode shadow \
        --stress-source none

The script performs four stages:
1. Load + merge inference artifacts (ranker candidates, CQF quantiles, optional stress metrics).
2. Build a decision-level dataset (one row per date/underlying with TOP_K candidates slotted).
3. Create a d3rlpy ``MDPDataset`` suitable for Implicit Q-Learning.
4. Optionally train IQL and persist policy + metadata for live serving.

See README-style guidance near the bottom for command-line options.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:  # Optional dependency: only required when --no-train is False
    import d3rlpy  # type: ignore
    from d3rlpy.algos import DiscreteCQL, DiscreteCQLConfig  # type: ignore
    from d3rlpy.datasets import MDPDataset  # type: ignore
except Exception:  # pragma: no cover - handled at runtime when training requested
    d3rlpy = None
    DiscreteCQL = None
    DiscreteCQLConfig = None
    MDPDataset = None

from sklearn.preprocessing import StandardScaler

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import load_config as training_load_config, preprocess_data  # type: ignore
from prod_cqf import OptimalCQF  # type: ignore

DEFAULT_HORIZON = 5

# ----------------------------- Config dataclasses -----------------------------


@dataclass
class BuildConfig:
    top_k: int
    size_bins: Sequence[float]
    min_prob: float
    min_q05: float
    reward_col: str
    risk_lambda: float
    group_keys: Sequence[str]
    group_top_n: Optional[int]


@dataclass
class TrainConfig:
    steps: int
    batch_size: int
    expectile: float
    temperature: float
    gamma: float
    seed: int


# ----------------------------- Utility helpers -----------------------------


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def _rename_with_prefix(df: pd.DataFrame, prefix: str, skip: Iterable[str]) -> pd.DataFrame:
    rename = {col: f"{prefix}{col}" for col in df.columns if col not in skip}
    return df.rename(columns=rename)


def _available(columns: Iterable[str], df: pd.DataFrame) -> List[str]:
    return [col for col in columns if col in df.columns]


def _pad_candidates(df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    if len(df) >= top_k:
        return df.head(top_k).copy().reset_index(drop=True)
    pad_rows = top_k - len(df)
    if pad_rows <= 0:
        return df.copy().reset_index(drop=True)
    pad = pd.DataFrame([{}] * pad_rows)
    return pd.concat([df, pad], ignore_index=True)


def _encode_action(slot_idx: Optional[int], size_idx: int, size_bins: Sequence[float]) -> int:
    if slot_idx is None:
        return 0
    return 1 + slot_idx * len(size_bins) + size_idx


def _action_mapping(top_k: int, size_bins: Sequence[float]) -> Dict[int, Dict[str, float]]:
    mapping: Dict[int, Dict[str, float]] = {0: {"slot": 0, "size_idx": -1, "size_value": 0.0}}
    action_id = 1
    for slot in range(top_k):
        for size_idx, size_value in enumerate(size_bins):
            mapping[action_id] = {
                "slot": slot + 1,
                "size_idx": size_idx,
                "size_value": float(size_value),
            }
            action_id += 1
    return mapping


def _choose_behavior_slot(candidates: pd.DataFrame, cfg: BuildConfig) -> Optional[int]:
    """Choose behavior slot using optimized strategy mix from config.yaml"""
    if candidates.empty:
        return None
    
    # Load optimized behavior policy from config
    config = training_load_config("config.yaml")
    behavior_config = config.get('behavior_policy', {})
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    # Get optimized parameters
    prob_weight = behavior_config.get('prob_weight', 0.4434)
    exp_weight = behavior_config.get('exp_weight', 0.2433)
    expl_weight = behavior_config.get('expl_weight', 0.3134)
    prob_threshold = behavior_config.get('prob_threshold', 0.5435)
    exp_threshold = behavior_config.get('exp_threshold', 0.0863)
    
    # Exploration slot preferences
    slot_probs_config = behavior_config.get('exploration_slot_probs', {})
    exploration_probs = [
        slot_probs_config.get('slot1', 0.2781),
        slot_probs_config.get('slot2', 0.2425),
        slot_probs_config.get('slot3', 0.2575),
        slot_probs_config.get('slot4', 0.1721),
        slot_probs_config.get('slot5', 0.0498)
    ]
    
    # Randomly choose strategy based on optimized mix
    rand = np.random.random()
    
    if rand < prob_weight:
        # Best probability of profit strategy
        best_slot = None
        best_prob = 0.0
        for idx, row in candidates.iterrows():
            prob_profit = row.get("prob_profit", 0.0)
            if pd.notna(prob_profit) and prob_profit > best_prob:
                best_prob = prob_profit
                best_slot = idx
        return best_slot
        
    elif rand < prob_weight + exp_weight:
        # Best expected return strategy
        best_slot = None
        best_return = float('-inf')
        for idx, row in candidates.iterrows():
            expected_return = row.get("expected_return", float('-inf'))
            if pd.notna(expected_return) and expected_return > best_return:
                best_return = expected_return
                best_slot = idx
        return best_slot
        
    else:
        # Random exploration with slot preferences
        available_slots = list(range(len(candidates)))
        if len(available_slots) == 0:
            return None
        
        # Adjust exploration probabilities for available slots
        max_slots = min(len(exploration_probs), len(available_slots))
        adjusted_probs = exploration_probs[:max_slots]
        
        # Normalize probabilities
        total_prob = sum(adjusted_probs)
        if total_prob > 0:
            adjusted_probs = [p / total_prob for p in adjusted_probs]
            chosen_idx = np.random.choice(available_slots[:max_slots], p=adjusted_probs)
            return chosen_idx
        else:
            return np.random.choice(available_slots)


def _choose_position_size(candidates: pd.DataFrame, behavior_slot: Optional[int], cfg: BuildConfig) -> int:
    """Choose position size using optimized thresholds from config.yaml"""
    if behavior_slot is None or behavior_slot >= len(candidates):
        return 0  # Default to conservative sizing
    
    # Load optimized behavior policy from config
    config = training_load_config("config.yaml")
    behavior_config = config.get('behavior_policy', {})
    
    # Get optimized thresholds and sizing ratios
    prob_threshold = behavior_config.get('prob_threshold', 0.5435)
    exp_threshold = behavior_config.get('exp_threshold', 0.0863)
    size_conservative = behavior_config.get('size_ratio_conservative', 0.3001)
    size_aggressive = behavior_config.get('size_ratio_aggressive', 0.9841)
    
    # Get metrics for chosen slot
    row = candidates.iloc[behavior_slot]
    prob_profit = row.get("prob_profit", 0.0)
    expected_return = row.get("expected_return", 0.0)
    
    # Determine if we should use aggressive sizing
    use_aggressive = False
    if pd.notna(prob_profit) and prob_profit > prob_threshold:
        use_aggressive = True
    elif pd.notna(expected_return) and expected_return > exp_threshold:
        use_aggressive = True
    
    # Map continuous size ratios to discrete size bins
    # Assuming size_bins are [0.5, 1.0] (conservative, aggressive)
    if len(cfg.size_bins) >= 2:
        if use_aggressive:
            return 1  # Use second bin (aggressive)
        else:
            return 0  # Use first bin (conservative)
    else:
        return 0  # Default to first available bin


def _risk_adjusted_reward(raw_pnl: float, risk_lambda: float) -> float:
    downside = max(0.0, -raw_pnl)
    return raw_pnl - risk_lambda * downside


def _make_episode_ids(df: pd.DataFrame, keys: Sequence[str]) -> pd.Series:
    if not keys:
        return pd.Series(["episode_0"] * len(df))
    pieces = []
    for key in keys:
        if key not in df.columns:
            raise KeyError(f"Cannot form episodes: column '{key}' missing from decision table")
        values = df[key].astype(str).fillna("NA")
        pieces.append(values)
    return pd.Series(["|".join(parts) for parts in zip(*pieces)], index=df.index)


# ----------------------------- Dataset builder -----------------------------


def load_and_merge(
    cqf_path: Path,
    ranker_path: Path,
    stress_path: Optional[Path],
) -> pd.DataFrame:
    cqf = _read_csv(cqf_path)
    ranker = _read_csv(ranker_path)

    merge_keys = ["contractID", "date"] if "date" in cqf.columns else ["contractID"]
    merged = ranker.merge(
        cqf,
        on=merge_keys,
        how="left",
        suffixes=("", "_cqf"),
    )

    if stress_path is not None:
        stress = _read_csv(stress_path)
        if "contractID" not in stress.columns:
            raise KeyError("Stress metrics file must contain 'contractID'")
        stress = stress.drop_duplicates("contractID")
        stress_pref = _rename_with_prefix(stress, "stress_", skip=["contractID"])
        merged = merged.merge(stress_pref, on="contractID", how="left")

    return merged


def build_decision_table(df: pd.DataFrame, cfg: BuildConfig) -> pd.DataFrame:
    if df.empty:
        raise ValueError("Merged dataframe is empty; nothing to build")

    if cfg.group_top_n is not None:
        missing_keys = [key for key in cfg.group_keys if key not in df.columns]
        if missing_keys:
            raise KeyError(f"Cannot apply group_top_n; missing columns: {missing_keys}")
        if "ranker_score" not in df.columns:
            raise KeyError("Merged dataframe missing 'ranker_score' required for group_top_n")
        sort_cols = list(cfg.group_keys) + ["ranker_score"]
        ascending = [True] * len(cfg.group_keys) + [False]
        df = (
            df.sort_values(sort_cols, ascending=ascending)
            .groupby(list(cfg.group_keys), as_index=False, group_keys=False)
            .head(cfg.group_top_n)
            .reset_index(drop=True)
        )

    context_cols = _available(
        ["vix_d_close_raw", "spy_momentum_raw", "vol_severity", "vol_emergency", "stress_score"],
        df,
    )
    summary_cols = _available(
        ["q0.05", "q0.50", "q0.95", "expected_return", "prob_profit", "utility"],
        df,
    )
    candidate_cols = _available(
        [
            "ranker_score",
            "expected_return",
            "prob_profit",
            "q0.05",
            "q0.50",
            "q0.95",
            "moneyness",
            "days_to_exp",
            "implied_volatility_raw",
            "relative_spread",
            "option_volume_oi_ratio",
            "delta",
            "gamma",
            "theta",
            "vega",
            "stress_expected_pnl",
            "stress_cvar_95",
            "stress_prob_profit",
            "stress_utility_score",
        ],
        df,
    )

    if not candidate_cols:
        raise ValueError("No candidate-level features available after merge")

    required_cols = set(cfg.group_keys) | {"contractID", cfg.reward_col, "ranker_score"}
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Merged dataframe missing required columns: {missing}")

    rows: List[Dict[str, object]] = []
    grouped = df.sort_values(list(cfg.group_keys) + ["ranker_score"], ascending=False).groupby(list(cfg.group_keys), sort=True)

    for group_key, group in grouped:
        group_sorted = group.sort_values("ranker_score", ascending=False).reset_index(drop=True)
        candidates = _pad_candidates(group_sorted, cfg.top_k)

        base: Dict[str, object] = {}
        if isinstance(group_key, tuple):
            for key_name, key_value in zip(cfg.group_keys, group_key):
                base[key_name] = key_value
        else:
            base[cfg.group_keys[0]] = group_key

        for col in context_cols:
            base[f"s_{col}"] = group_sorted[col].iloc[0] if col in group_sorted.columns else np.nan
        for col in summary_cols:
            base[f"s_{col}"] = group_sorted[col].iloc[0] if col in group_sorted.columns else np.nan
        base["group_count"] = float(len(group_sorted))

        for slot in range(cfg.top_k):
            prefix = f"c{slot + 1}_"
            if slot < len(candidates):
                cand_row = candidates.iloc[slot]
                for col in candidate_cols:
                    base[f"{prefix}{col}"] = cand_row.get(col, np.nan)
                base[f"{prefix}contractID"] = cand_row.get("contractID")
            else:
                for col in candidate_cols:
                    base[f"{prefix}{col}"] = np.nan
                base[f"{prefix}contractID"] = None

        behavior_slot = _choose_behavior_slot(candidates, cfg)
        if behavior_slot is None and not group_sorted.empty:
            behavior_slot = 0
        chosen_reward = 0.0
        if behavior_slot is not None and behavior_slot < len(group_sorted):
            raw_value = group_sorted.iloc[behavior_slot].get(cfg.reward_col, 0.0)
            raw_pnl = 0.0 if pd.isna(raw_value) else float(raw_value)
            chosen_reward = _risk_adjusted_reward(raw_pnl, cfg.risk_lambda)
            base["behavior_contractID"] = group_sorted.iloc[behavior_slot].get("contractID")
        else:
            base["behavior_contractID"] = None

        # Optimized position sizing based on config
        size_idx = _choose_position_size(group_sorted, behavior_slot, cfg)
        action_id = _encode_action(behavior_slot, size_idx=size_idx, size_bins=cfg.size_bins)
        base["action_id"] = int(action_id)
        base["raw_reward"] = chosen_reward if behavior_slot is not None else 0.0
        base["reward"] = chosen_reward if behavior_slot is not None else 0.0
        base["behavior_slot"] = 0 if behavior_slot is None else int(behavior_slot + 1)
        base["behavior_size_idx"] = size_idx if behavior_slot is not None else -1

        rows.append(base)

    decision_df = pd.DataFrame(rows)
    decision_df.sort_values(list(cfg.group_keys), inplace=True)
    decision_df.reset_index(drop=True, inplace=True)

    # Episode boundary: by month within group keys (assumes first key is date)
    if "date" in cfg.group_keys:
        decision_df["episode_month"] = pd.to_datetime(decision_df["date"]).dt.to_period("M").astype(str)
        episode_keys = ["episode_month"] + [key for key in cfg.group_keys if key != "date"]
    else:
        episode_keys = list(cfg.group_keys)
    decision_df["episode_id"] = _make_episode_ids(decision_df, episode_keys)

    # Terminal flag when episode id changes
    episode_shift = decision_df["episode_id"].shift(-1, fill_value=decision_df["episode_id"].iloc[-1])
    decision_df["terminal"] = (decision_df["episode_id"] != episode_shift).astype(int)

    return decision_df


def to_mdp_dataset(decision_df: pd.DataFrame, scaler: Optional[StandardScaler] = None) -> Tuple[MDPDataset, StandardScaler, List[str]]:
    if decision_df.empty:
        raise ValueError("Decision table is empty; cannot create dataset")

    forbidden_terms = ("_target_pnl", "future_option_price", "contractID")
    state_cols = [
        c
        for c in decision_df.columns
        if (c.startswith("s_") or c.startswith("c"))
        and not any(term in c for term in forbidden_terms)
        and c != "s_target_pnl"
    ]
    if not state_cols:
        raise ValueError("No state columns detected (expected prefixes 's_' or 'c')")

    bad = [c for c in decision_df.columns if any(term in c for term in forbidden_terms)]
    if "s_target_pnl" in decision_df.columns:
        bad.append("s_target_pnl")
    assert not any(c in state_cols for c in bad), f"Leakage in state: {sorted(set(state_cols) & set(bad))}"

    states = decision_df[state_cols].copy()
    states = states.apply(pd.to_numeric, errors="coerce")
    states = states.fillna(0.0)
    scaler = scaler or StandardScaler()
    scaled = scaler.fit_transform(states.to_numpy(dtype=np.float32))
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    actions = decision_df["action_id"].to_numpy(dtype=np.int64)
    rewards = decision_df["reward"].to_numpy(dtype=np.float32)
    terminals = decision_df["terminal"].to_numpy(dtype=np.float32)

    if d3rlpy is None or MDPDataset is None:
        raise RuntimeError("d3rlpy is not installed; install it or rerun with --no-train")

    dataset = MDPDataset(observations=scaled, actions=actions, rewards=rewards, terminals=terminals)
    return dataset, scaler, state_cols


def train_iql(dataset: MDPDataset, action_size: int, cfg: TrainConfig, outdir: Path) -> DiscreteCQL:
    if d3rlpy is None or DiscreteCQL is None or DiscreteCQLConfig is None:
        raise RuntimeError("d3rlpy is not installed. Run `pip install d3rlpy` to enable training.")

    algo_cfg = DiscreteCQLConfig(
        learning_rate=3e-4,
        gamma=cfg.gamma,
        batch_size=cfg.batch_size,
        n_critics=2,
        alpha=5.0,  # CQL regularization parameter
    )

    algo = DiscreteCQL(config=algo_cfg, device="mps", enable_ddp=False)
    algo.build_with_dataset(dataset)
    algo.fit(
        dataset=dataset,
        n_steps=cfg.steps,
        n_steps_per_epoch=min(10_000, cfg.steps),
        experiment_name="discrete_cql_cqf_ranker",
        save_interval=max(cfg.steps // 5, 10_000),
    )
    return algo


def export_policy(
    algo: DiscreteCQL,
    scaler: StandardScaler,
    state_cols: Sequence[str],
    action_map: Dict[int, Dict[str, float]],
    outdir: Path,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    algo.save(str(outdir / "discrete_cql_policy.d3"))
    meta = {
        "state_columns": list(state_cols),
        "scaler_mean": scaler.mean_.astype(float).tolist(),
        "scaler_scale": scaler.scale_.astype(float).tolist(),
        "action_map": action_map,
    }
    with open(outdir / "policy_meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)


def save_decision_table(df: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(outdir / "decision_table.parquet", index=False)
    except ImportError:
        print("pyarrow/fastparquet missing; skipping Parquet export", file=sys.stderr)
    df.to_csv(outdir / "decision_table.csv", index=False)


def compute_realized_targets(raw_path: Path, config_path: Path, horizon: int) -> pd.DataFrame:
    logger = logging.getLogger("iql_pipeline")
    logger.info("Computing realized rewards from %s", raw_path)
    raw_df = pd.read_csv(raw_path, low_memory=False)
    raw_df['date'] = pd.to_datetime(raw_df.get('date'), errors='coerce')
    config = training_load_config(str(config_path))
    processed_df, _ = preprocess_data(raw_df, config, scaler=None)
    cqf = OptimalCQF(horizon=horizon)
    pnl_df = cqf.calculate_delta_hedged_pnl(processed_df.copy(), horizon)
    realized = pnl_df[['contractID', 'date', 'target_pnl', 'future_option_price']].copy()
    realized['date'] = pd.to_datetime(realized['date'], errors='coerce')
    return realized


def run_inference_stage(args: argparse.Namespace) -> Tuple[Path, Path, Optional[Path]]:
    required = [
        ("ranker_model", args.ranker_model),
        ("ranker_features", args.ranker_features),
        ("sharpe_edges", args.sharpe_edges),
        ("cqf_model", args.cqf_model),
    ]
    missing = [name for name, value in required if value is None]
    if missing:
        raise ValueError(
            "Running inference requires model artifacts. Missing: " + ", ".join(missing)
        )

    outdir = args.inference_outdir or (args.outdir / "inference_outputs")
    outdir = outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    project_root = Path(__file__).resolve().parents[1]
    inference_pkg = project_root / "inference"
    if not inference_pkg.exists():
        raise FileNotFoundError(f"Inference package not found at {inference_pkg}")

    cmd = [
        sys.executable,
        "-m",
        "inference.run_inference",
        "--raw-data",
        str(args.raw_data.resolve()),
        "--config",
        str(args.config.resolve()),
        "--ranker-model",
        str(args.ranker_model.resolve()),
        "--ranker-features",
        str(args.ranker_features.resolve()),
        "--sharpe-edges",
        str(args.sharpe_edges.resolve()),
        "--cqf-model",
        str(args.cqf_model.resolve()),
        "--output-dir",
        str(outdir),
    ]
    if not args.include_future_targets:
        cmd.append("--skip-future-targets")
    if args.top_n is not None:
        cmd.extend(["--top-n", str(args.top_n)])
    if args.stress_mode is not None:
        cmd.extend(["--stress-mode", args.stress_mode])
    if args.llm_engine is not None:
        cmd.extend(["--llm-engine", args.llm_engine])
    if args.llm_max_contracts is not None:
        cmd.extend(["--llm-max-contracts", str(args.llm_max_contracts)])

    print("Running inference pipeline:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(project_root))

    cqf_path = outdir / "cqf_predictions.csv"
    ranker_path = outdir / "ranker_candidates.csv"
    if not cqf_path.exists() or not ranker_path.exists():
        raise FileNotFoundError(
            f"Inference outputs missing expected files in {outdir}"
        )

    stress_path: Optional[Path] = None
    if args.stress_source in ("auto", "mc"):
        mc_path = outdir / "stress_metrics.csv"
        if mc_path.exists():
            stress_path = mc_path
            if args.stress_source == "mc":
                return cqf_path, ranker_path, stress_path
    if args.stress_source in ("auto", "llm"):
        llm_path = outdir / "stress_metrics_llm.csv"
        if llm_path.exists():
            stress_path = llm_path
    if args.stress_source == "none":
        stress_path = None

    return cqf_path, ranker_path, stress_path


# ----------------------------- CLI interface -----------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build offline IQL dataset and policy from CQF + Ranker outputs")
    parser.add_argument("--cqf-preds", type=Path, default=None, help="Precomputed cqf_predictions.csv")
    parser.add_argument("--ranker-candidates", type=Path, default=None, help="Precomputed ranker_candidates.csv")
    parser.add_argument("--stress-metrics", type=Path, default=None, help="Optional stress metrics CSV (MC or LLM)")
    parser.add_argument("--outdir", type=Path, default=Path("iql_out"), help="Output directory for artifacts")
    parser.add_argument("--top-k", type=int, default=5, help="Number of candidates per decision state")
    parser.add_argument(
        "--size-bins",
        type=str,
        default="0.5,1.0",
        help="Comma-separated trade size bins (relative sizing)",
    )
    parser.add_argument("--min-prob", type=float, default=0.45, help="Behaviour policy threshold on prob_profit")
    parser.add_argument("--min-q05", type=float, default=-0.20, help="Behaviour policy threshold on q0.05")
    parser.add_argument("--reward-col", type=str, default="target_pnl", help="Column used as realized reward")
    parser.add_argument("--risk-lambda", type=float, default=0.5, help="Downside penalty in reward shaping")
    parser.add_argument("--group-keys", type=str, default="date,underlying", help="Comma-separated grouping keys for decisions")
    parser.add_argument("--no-train", action="store_true", help="Build dataset only (skip IQL training)")
    parser.add_argument("--train-steps", type=int, default=200_000, help="IQL training steps")
    parser.add_argument("--train-batch-size", type=int, default=1024, help="IQL batch size")
    parser.add_argument("--expectile", type=float, default=0.7, help="IQL expectile (0.5..0.9 typical)")
    parser.add_argument("--temperature", type=float, default=3.0, help="IQL temperature for advantage weights")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    # Inference stage options
    parser.add_argument("--raw-data", type=Path, default=None, help="Raw CSV to run inference on (optional)")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Config YAML for inference preprocessing")
    parser.add_argument("--ranker-model", type=Path, default=None, help="Ranker model artifact (.joblib)")
    parser.add_argument("--ranker-features", type=Path, default=None, help="Ranker feature list pickle")
    parser.add_argument("--sharpe-edges", type=Path, default=None, help="Sharpe edge boundaries pickle")
    parser.add_argument("--cqf-model", type=Path, default=None, help="CQF artifact (.joblib)")
    parser.add_argument("--inference-outdir", type=Path, default=None, help="Where to store inference outputs (default: <outdir>/inference_outputs)")
    parser.add_argument("--stress-source", choices=["auto", "mc", "llm", "none"], default="auto", help="Which stress metrics file to merge when inference is run")
    parser.add_argument("--stress-mode", choices=["mc", "llm", "shadow"], default=None, help="Pass-through stress mode for run_inference.py")
    parser.add_argument("--llm-engine", choices=["basic", "agent"], default=None, help="Override LLM engine when stress-mode uses LLM")
    parser.add_argument("--llm-max-contracts", type=int, default=None, help="Override llm-max-contracts when running inference")
    parser.add_argument("--top-n", type=int, default=None, help="Override --top-n for run_inference.py")
    parser.add_argument("--group-top-n", type=int, default=None, help="Keep at most N candidates per group when building the decision table (e.g., per date)")
    parser.add_argument("--include-future-targets", action="store_true", help="Allow inference to compute future-dependent targets (useful for backtests)")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    args.outdir = args.outdir.resolve()
    args.outdir.mkdir(parents=True, exist_ok=True)

    size_bins = [float(x.strip()) for x in args.size_bins.split(",") if x.strip()]
    if not size_bins:
        raise ValueError("--size-bins must contain at least one numeric value")

    group_keys = [key.strip() for key in args.group_keys.split(",") if key.strip()]
    if not group_keys:
        raise ValueError("--group-keys must contain at least one column name")

    cfg_build = BuildConfig(
        top_k=args.top_k,
        size_bins=size_bins,
        min_prob=args.min_prob,
        min_q05=args.min_q05,
        reward_col=args.reward_col,
        risk_lambda=args.risk_lambda,
        group_keys=group_keys,
        group_top_n=args.group_top_n,
    )

    cfg_train = TrainConfig(
        steps=args.train_steps,
        batch_size=args.train_batch_size,
        expectile=args.expectile,
        temperature=args.temperature,
        gamma=args.gamma,
        seed=args.seed,
    )

    cqf_path: Optional[Path] = args.cqf_preds.resolve() if args.cqf_preds else None
    ranker_path: Optional[Path] = args.ranker_candidates.resolve() if args.ranker_candidates else None
    stress_path: Optional[Path] = args.stress_metrics.resolve() if args.stress_metrics else None

    if args.raw_data is not None:
        if not args.raw_data.exists():
            raise FileNotFoundError(args.raw_data)
        cqf_path, ranker_path, inferred_stress = run_inference_stage(args)
        if stress_path is None:
            stress_path = inferred_stress

    if cqf_path is None or ranker_path is None:
        raise ValueError("Either provide --raw-data (with models) or precomputed --cqf-preds and --ranker-candidates")

    merged = load_and_merge(cqf_path, ranker_path, stress_path)
    print(f"Loaded merged dataset with {len(merged):,} rows and {len(merged.columns)} columns")

    if args.raw_data is not None and not args.include_future_targets:
        realized = compute_realized_targets(args.raw_data.resolve(), args.config.resolve(), DEFAULT_HORIZON)
        for col in ["target_pnl", "future_option_price"]:
            if col in merged.columns:
                merged.drop(columns=col, inplace=True)
        merged = merged.merge(realized, on=["contractID", "date"], how="left")

    decision_df = build_decision_table(merged, cfg_build)
    print(f"Built decision table with {len(decision_df):,} decisions")
    action_counts = decision_df["action_id"].value_counts().sort_index()
    print("Action distribution:\n" + action_counts.to_string())

    save_decision_table(decision_df, args.outdir)
    action_map = _action_mapping(cfg_build.top_k, cfg_build.size_bins)

    if args.no_train:
        print("--no-train flag set; skipping IQL training. Decision table saved in", args.outdir)
        with open(args.outdir / "action_map.json", "w", encoding="utf-8") as fh:
            json.dump(action_map, fh, indent=2)
        return

    dataset, scaler, state_cols = to_mdp_dataset(decision_df, scaler=None)
    action_size = 1 + cfg_build.top_k * len(cfg_build.size_bins)
    algo = train_iql(dataset, action_size, cfg_train, args.outdir)
    export_policy(algo, scaler, state_cols, action_map, args.outdir)

    print("Training complete. Artifacts saved to", args.outdir)


if __name__ == "__main__":
    main()
