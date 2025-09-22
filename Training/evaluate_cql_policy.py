#!/usr/bin/env python3
"""Evaluate a trained DiscreteCQL policy on a decision table.

This replays the saved policy across the offline decision table produced by
``Training/iql_pipeline.py`` and compares its predicted rewards to the logged
behaviour policy.

The script supports both d3rlpy saving formats:
- Policy files saved with ``algo.save()`` (recommended)
- Policy files saved with ``algo.save_model()`` (legacy format with manual reconstruction)

The script expects:
- ``decision_table.csv`` with columns ``s_*`` and ``c{slot}_*`` features.
- ``discrete_cql_policy.d3`` saved by d3rlpy.
- ``policy_meta.json`` containing the scaler parameters and action map.

Example
-------
    python3 Training/evaluate_cql_policy.py \
        --decision-table iql_cql_artifacts/decision_table.csv \
        --policy iql_cql_artifacts/discrete_cql_policy.d3 \
        --meta iql_cql_artifacts/policy_meta.json \
        --outdir iql_cql_artifacts/eval_cql
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from d3rlpy import load_learnable
from d3rlpy.algos import DiscreteCQL, DiscreteCQLConfig
import torch
import yaml


RISK_LAMBDA_DEFAULT = 0.5


def _load_meta(meta_path: Path) -> Dict:
    with meta_path.open("r", encoding="utf-8") as fh:
        meta = json.load(fh)
    required = {"state_columns", "scaler_mean", "scaler_scale", "action_map"}
    missing = required - set(meta)
    if missing:
        raise KeyError(f"policy_meta.json missing keys: {missing}")
    return meta


def _load_policy_robust(policy_path: Path, meta: Dict) -> DiscreteCQL:
    """Robustly load CQL policy, handling both save() and save_model() formats."""
    logger = logging.getLogger("evaluate_cql")
    
    try:
        # Method 1: Try standard load_learnable (works with algo.save() format)
        logger.info("Attempting to load policy with load_learnable...")
        algo = load_learnable(str(policy_path))
        logger.info("✅ Successfully loaded policy with load_learnable")
        return algo
        
    except Exception as e1:
        logger.warning("load_learnable failed: %s", str(e1))
        logger.info("Attempting manual reconstruction for save_model() format...")
        
        try:
            # Method 2: Manual reconstruction for save_model() format
            # Detect model architecture from saved weights
            model_data = torch.load(str(policy_path), map_location='cpu')
            
            # Extract dimensions from model structure
            if 'q_funcs' in model_data:
                # Get action size from Q-function output layer
                q_func_keys = [k for k in model_data['q_funcs'].keys() if k.endswith('._fc.weight')]
                if q_func_keys:
                    action_size = model_data['q_funcs'][q_func_keys[0]].shape[0]
                else:
                    action_size = len(meta["action_map"])
                
                # Get observation size from input layer  
                input_keys = [k for k in model_data['q_funcs'].keys() if k.endswith('._encoder._layers.0.weight')]
                if input_keys:
                    observation_size = model_data['q_funcs'][input_keys[0]].shape[1]
                else:
                    observation_size = len(meta["state_columns"])
                
                # Count number of Q-networks
                n_critics = len(set(k.split('.')[0] for k in model_data['q_funcs'].keys()))
                
                logger.info("Detected model architecture: obs=%d, actions=%d, critics=%d", 
                           observation_size, action_size, n_critics)
                
                # Load training config if available
                config_path = policy_path.parent.parent / "config.yaml"
                if config_path.exists():
                    with open(config_path, 'r') as f:
                        training_config = yaml.safe_load(f)
                    
                    # Create CQL config with training parameters
                    config = DiscreteCQLConfig(
                        learning_rate=training_config.get('learning_rate', 3e-4),
                        gamma=training_config.get('gamma', 0.99),
                        batch_size=training_config.get('batch_size', 256),
                        n_critics=n_critics
                    )
                else:
                    # Use default config with detected n_critics
                    config = DiscreteCQLConfig(n_critics=n_critics)
                
                # Create and load algorithm
                algo = config.create()
                algo.create_impl((observation_size,), action_size)
                algo.load_model(str(policy_path))
                
                logger.info("✅ Successfully loaded policy with manual reconstruction")
                return algo
                
            else:
                raise ValueError("Unrecognized model format")
                
        except Exception as e2:
            logger.error("Manual reconstruction also failed: %s", str(e2))
            raise RuntimeError(f"Failed to load policy with both methods. "
                             f"load_learnable error: {e1}. "
                             f"Manual reconstruction error: {e2}")


def _standardise_states(df: pd.DataFrame, state_cols, mean, scale) -> np.ndarray:
    numeric_df = df[state_cols].apply(pd.to_numeric, errors='coerce')
    states = numeric_df.to_numpy(dtype=np.float32)
    states = np.nan_to_num(states, nan=0.0, posinf=0.0, neginf=0.0)
    mean = np.asarray(mean, dtype=np.float32)
    scale = np.asarray(scale, dtype=np.float32)
    if states.shape[1] != mean.shape[0]:
        raise ValueError("Scaler mean length does not match state dimension")
    denom = np.where(scale == 0.0, 1.0, scale)
    standardised = (states - mean) / denom
    return standardised


def _decode_action(action_id: int, action_map: Dict[str, Dict[str, float]]) -> Tuple[int, float]:
    info = action_map.get(str(int(action_id)))
    if info is None:
        raise KeyError(f"Action id {action_id} not found in action map")
    return int(info["slot"]), float(info.get("size_value", 0.0))


def _compute_reward(row: pd.Series, slot: int, risk_lambda: float) -> Tuple[float, float]:
    if slot <= 0:
        return 0.0, 0.0
    pnl = row.get(f"c{slot}_target_pnl", np.nan)
    if pd.isna(pnl):
        return 0.0, np.nan
    pnl = float(pnl)
    downside = max(0.0, -pnl)
    reward = pnl - risk_lambda * downside
    return reward, pnl


def evaluate(decision_table: Path, policy_path: Path, meta_path: Path, outdir: Path, risk_lambda: float) -> Dict:
    logger = logging.getLogger("evaluate_cql")
    logger.info("Loading decision table: %s", decision_table)
    df = pd.read_csv(decision_table)
    df['date'] = pd.to_datetime(df.get('date'), errors='coerce')

    meta = _load_meta(meta_path)
    state_cols = meta["state_columns"]
    states = _standardise_states(df, state_cols, meta["scaler_mean"], meta["scaler_scale"])

    logger.info("Loading policy: %s", policy_path)
    algo = _load_policy_robust(policy_path, meta)
    predicted_actions = algo.predict(states)

    action_map = meta["action_map"]
    slots = []
    sizes = []
    rewards = []
    pnls = []
    for action_id, (_, row) in zip(predicted_actions, df.iterrows()):
        slot, size = _decode_action(int(action_id), action_map)
        reward, pnl = _compute_reward(row, slot, risk_lambda)
        slots.append(slot)
        sizes.append(size)
        rewards.append(reward)
        pnls.append(pnl)

    df['cql_action_id'] = predicted_actions.astype(int)
    df['cql_slot'] = slots
    df['cql_size'] = sizes
    df['cql_raw_pnl'] = pnls
    df['cql_reward'] = rewards

    behaviour_reward = df.get('reward', pd.Series(np.zeros(len(df))))
    summary = {
        "decisions": int(len(df)),
        "cql_total_reward": float(np.nansum(df['cql_reward'])),
        "cql_mean_reward": float(np.nanmean(df['cql_reward'])),
        "behaviour_total_reward": float(np.nansum(behaviour_reward)),
        "behaviour_mean_reward": float(np.nanmean(behaviour_reward)),
        "reward_lift": float(np.nansum(df['cql_reward']) - np.nansum(behaviour_reward)),
        "action_counts": df['cql_action_id'].value_counts().to_dict(),
    }

    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / 'evaluation_results.csv', index=False)
    with (outdir / 'evaluation_summary.json').open('w', encoding='utf-8') as fh:
        json.dump(summary, fh, indent=2)

    logger.info("Evaluation complete: %s", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DiscreteCQL policy against decision table")
    parser.add_argument('--decision-table', type=Path, required=True, help="Path to decision_table.csv")
    parser.add_argument('--policy', type=Path, required=True, help="Path to discrete_cql_policy.d3")
    parser.add_argument('--meta', type=Path, required=True, help="Path to policy_meta.json")
    parser.add_argument('--outdir', type=Path, default=Path('policy_evaluation'), help="Output directory for evaluation artifacts")
    parser.add_argument('--risk-lambda', type=float, default=RISK_LAMBDA_DEFAULT, help="Downside penalty coefficient used during reward shaping")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    args = parse_args()
    evaluate(args.decision_table, args.policy, args.meta, args.outdir, args.risk_lambda)


if __name__ == '__main__':
    main()
