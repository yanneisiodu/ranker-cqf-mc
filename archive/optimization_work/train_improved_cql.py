#!/usr/bin/env python3
"""
Train DiscreteCQL directly on the improved decision table.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
import sys

# Add Training to path
sys.path.insert(0, str(Path(__file__).parent / "Training"))

from iql_pipeline import (
    to_mdp_dataset, train_iql, export_policy, 
    BuildConfig, TrainConfig, _action_mapping
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    print("🚀 Training DiscreteCQL with Improved Behavior Policy")
    print("=" * 60)
    
    # Load improved decision table
    decision_table_path = "improved_behavior_policy/improved_decision_table.csv"
    df = pd.read_csv(decision_table_path)
    logger.info(f"Loaded improved decision table: {len(df)} decisions")
    
    # Create output directory
    outdir = Path("improved_cql_artifacts")
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Configuration
    cfg_build = BuildConfig(
        top_k=5,
        size_bins=[0.5, 1.0], 
        min_prob=0.0,  # No filtering, we already have good data
        min_q05=-10.0,  # No filtering
        reward_col="reward",
        risk_lambda=0.5,
        group_keys=["date", "underlying"],
        group_top_n=None
    )
    
    cfg_train = TrainConfig(
        steps=5000,
        batch_size=64,  # Smaller batch for limited data
        expectile=0.7,
        temperature=3.0,
        gamma=0.99,
        seed=42
    )
    
    print(f"📊 Training configuration:")
    print(f"  Training steps: {cfg_train.steps}")
    print(f"  Batch size: {cfg_train.batch_size}")
    print(f"  Expectile: {cfg_train.expectile}")
    
    # Create MDP dataset
    logger.info("Creating MDP dataset...")
    dataset, scaler, state_cols = to_mdp_dataset(df, scaler=None)
    
    # Get action size from dataset
    action_size = 11  # 0 (no action) + 10 (slot × size combinations)
    
    print(f"📈 Dataset created:")
    print(f"  Decisions: {len(df)}")
    print(f"  State features: {len(state_cols)}")
    print(f"  Action space size: {action_size}")
    
    # Action distribution
    action_dist = df['action_id'].value_counts().sort_index()
    print(f"  Action distribution:")
    for action_id, count in action_dist.items():
        print(f"    Action {action_id}: {count}")
    
    # Train DiscreteCQL
    logger.info("Training DiscreteCQL...")
    algo = train_iql(dataset, action_size, cfg_train, outdir)
    
    # Create action mapping
    action_map = _action_mapping(cfg_build.top_k, cfg_build.size_bins)
    
    # Export policy
    logger.info("Exporting trained policy...")
    export_policy(algo, scaler, state_cols, action_map, outdir)
    
    # Save decision table for reference
    df.to_csv(outdir / "improved_decision_table.csv", index=False)
    df.to_parquet(outdir / "improved_decision_table.parquet", index=False)
    
    # Save training configuration
    training_config = {
        'build_config': {
            'top_k': cfg_build.top_k,
            'size_bins': cfg_build.size_bins,
            'min_prob': cfg_build.min_prob,
            'min_q05': cfg_build.min_q05,
            'reward_col': cfg_build.reward_col,
            'risk_lambda': cfg_build.risk_lambda,
        },
        'train_config': {
            'steps': cfg_train.steps,
            'batch_size': cfg_train.batch_size,
            'expectile': cfg_train.expectile,
            'temperature': cfg_train.temperature,
            'gamma': cfg_train.gamma,
            'seed': cfg_train.seed,
        },
        'dataset_stats': {
            'total_decisions': len(df),
            'action_distribution': action_dist.to_dict(),
            'total_reward': df['reward'].sum(),
            'mean_reward': df['reward'].mean(),
            'hit_rate': (df['reward'] > 0).mean(),
        }
    }
    
    with open(outdir / "training_config.json", 'w') as f:
        json.dump(training_config, f, indent=2, default=str)
    
    print(f"\n✅ Training complete! Artifacts saved to: {outdir}")
    print(f"📁 Key files:")
    print(f"  - discrete_cql_policy.d3 (trained model)")
    print(f"  - policy_meta.json (scaler + action mapping)")
    print(f"  - training_config.json (hyperparameters)")
    print(f"  - improved_decision_table.csv (training data)")

if __name__ == "__main__":
    main()