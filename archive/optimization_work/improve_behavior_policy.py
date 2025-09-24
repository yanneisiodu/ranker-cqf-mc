#!/usr/bin/env python3
"""
Improve the behavior policy by creating a mixed strategy that combines:
1. Best probability of profit (60%)
2. Best expected return (25%)  
3. Random exploration (15%)

This creates a more diverse and higher-performing behavior policy for RL training.
"""

import pandas as pd
import numpy as np
import json
import yaml
from pathlib import Path
from typing import Dict, Tuple
import argparse

def load_decision_table(path: Path) -> pd.DataFrame:
    """Load the original decision table."""
    df = pd.read_csv(path)
    print(f"📊 Loaded decision table: {len(df)} decisions")
    return df

def create_action_mapping() -> Dict[int, Dict]:
    """Create the action mapping for 11 discrete actions."""
    action_map = {}
    action_id = 0
    
    # Action 0: No trade
    action_map[action_id] = {"slot": 0, "size_idx": -1, "size_value": 0.0}
    action_id += 1
    
    # Actions 1-10: slot (1-5) × size (0.5, 1.0)
    for slot in range(1, 6):
        for size_idx, size_value in enumerate([0.5, 1.0]):
            action_map[action_id] = {
                "slot": slot, 
                "size_idx": size_idx, 
                "size_value": size_value
            }
            action_id += 1
            
    return action_map

def find_best_slot_strategy(row: pd.Series, strategy: str) -> Tuple[int, float]:
    """Find the best slot according to a given strategy."""
    best_slot = 1
    best_value = float('-inf') if strategy != 'prob_profit' else 0.0
    
    for slot in range(1, 6):
        if strategy == 'expected_return':
            value = row.get(f'c{slot}_expected_return', float('-inf'))
        elif strategy == 'prob_profit':
            value = row.get(f'c{slot}_prob_profit', 0.0)
        elif strategy == 'ranker_score':
            value = row.get(f'c{slot}_ranker_score', float('-inf'))
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
            
        if pd.notna(value) and value > best_value:
            best_value = value
            best_slot = slot
            
    return best_slot, best_value

def load_config() -> Dict:
    """Load configuration from config.yaml."""
    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}

def create_improved_behavior_policy(df: pd.DataFrame, strategy_mix: Dict[str, float] = None) -> pd.DataFrame:
    """Create an improved behavior policy with mixed strategies."""
    if strategy_mix is None:
        # Load optimized parameters from config
        config = load_config()
        behavior_config = config.get('behavior_policy', {})
        
        strategy_mix = {
            'best_prob_profit': behavior_config.get('prob_weight', 0.4943),
            'best_expected_return': behavior_config.get('exp_weight', 0.1814),
            'random_exploration': behavior_config.get('expl_weight', 0.3243)
        }
    
    print(f"🎯 Creating improved behavior policy with mix:")
    for strategy, weight in strategy_mix.items():
        print(f"  {strategy}: {weight:.0%}")
    
    df = df.copy()
    action_map = create_action_mapping()
    
    # Load optimized parameters from config  
    config = load_config()
    behavior_config = config.get('behavior_policy', {})
    
    # Optimized thresholds and sizing
    prob_threshold = behavior_config.get('prob_threshold', 0.7315)
    exp_threshold = behavior_config.get('exp_threshold', 0.1035)
    size_conservative = behavior_config.get('size_ratio_conservative', 0.3298)
    size_aggressive = behavior_config.get('size_ratio_aggressive', 0.9974)
    
    # Optimized exploration slot preferences
    slot_probs_config = behavior_config.get('exploration_slot_probs', {})
    exploration_probs = [
        slot_probs_config.get('slot1', 0.3124),
        slot_probs_config.get('slot2', 0.2085),
        slot_probs_config.get('slot3', 0.1609), 
        slot_probs_config.get('slot4', 0.1703),
        slot_probs_config.get('slot5', 0.1479)
    ]
    
    # Track strategy selections
    selected_strategies = []
    selected_slots = []
    selected_sizes = []
    selected_action_ids = []
    target_pnls = []
    
    np.random.seed(42)  # For reproducible results
    
    for idx, row in df.iterrows():
        # Randomly choose strategy based on mix
        rand = np.random.random()
        
        if rand < strategy_mix['best_prob_profit']:
            strategy = 'best_prob_profit'
            slot, _ = find_best_slot_strategy(row, 'prob_profit')
        elif rand < strategy_mix['best_prob_profit'] + strategy_mix['best_expected_return']:
            strategy = 'best_expected_return'
            slot, _ = find_best_slot_strategy(row, 'expected_return')
        else:
            strategy = 'random_exploration'
            # Use optimized exploration slot preferences
            slot = np.random.choice([1, 2, 3, 4, 5], p=exploration_probs)
        
        # Choose position size using optimized thresholds
        if strategy == 'best_prob_profit':
            prob_profit = row.get(f'c{slot}_prob_profit', 0.0)
            size_value = size_aggressive if prob_profit > prob_threshold else size_conservative
        elif strategy == 'best_expected_return':
            expected_return = row.get(f'c{slot}_expected_return', 0.0)
            size_value = size_aggressive if expected_return > exp_threshold else size_conservative
        else:  # random exploration
            # Conservative exploration sizing
            size_value = np.random.choice([size_conservative, size_aggressive], p=[0.7, 0.3])
        
        # Map continuous size values to discrete action space (0.5 or 1.0)
        discrete_size = 1.0 if size_value > 0.65 else 0.5
        
        # Find corresponding action_id
        action_id = None
        for aid, action_info in action_map.items():
            if action_info['slot'] == slot and action_info['size_value'] == discrete_size:
                action_id = aid
                break
        
        if action_id is None:
            print(f"Warning: Could not find action_id for slot={slot}, size={discrete_size}")
            action_id = 1  # Default to action 1
        
        # Get target PnL for this choice (use original continuous size for reward calculation)
        target_pnl = row.get(f'c{slot}_target_pnl', 0.0)
        reward = target_pnl * size_value if pd.notna(target_pnl) else 0.0
        
        selected_strategies.append(strategy)
        selected_slots.append(slot)
        selected_sizes.append(discrete_size)  # Store discrete size for action mapping
        selected_action_ids.append(action_id)
        target_pnls.append(reward)  # Store the continuous-sized reward
    
    # Update the dataframe
    df['improved_strategy'] = selected_strategies
    df['improved_slot'] = selected_slots
    df['improved_size'] = selected_sizes
    df['improved_action_id'] = selected_action_ids
    df['improved_target_pnl'] = [row[1].get(f'c{slot}_target_pnl', 0.0) for slot, row in zip(selected_slots, df.iterrows())]
    df['improved_reward'] = target_pnls  # Already calculated with optimal continuous sizing
    
    # Update the behavior columns for RL training
    df['behavior_slot'] = df['improved_slot']
    df['behavior_size_idx'] = df['improved_size'].map({0.5: 0, 1.0: 1})
    df['action_id'] = df['improved_action_id']
    df['reward'] = df['improved_reward']
    df['raw_reward'] = df['improved_target_pnl']
    
    return df

def analyze_improved_policy(df: pd.DataFrame):
    """Analyze the improved behavior policy."""
    print("\n📊 IMPROVED BEHAVIOR POLICY ANALYSIS")
    print("=" * 50)
    
    # Strategy distribution
    strategy_counts = df['improved_strategy'].value_counts()
    print("\nStrategy distribution:")
    for strategy, count in strategy_counts.items():
        pct = count / len(df) * 100
        print(f"  {strategy}: {count} ({pct:.1f}%)")
    
    # Action distribution
    action_counts = df['improved_action_id'].value_counts().sort_index()
    print("\nAction distribution:")
    for action_id, count in action_counts.items():
        pct = count / len(df) * 100
        print(f"  Action {action_id}: {count} ({pct:.1f}%)")
    
    # Slot distribution
    slot_counts = df['improved_slot'].value_counts().sort_index()
    print("\nSlot distribution:")
    for slot, count in slot_counts.items():
        pct = count / len(df) * 100
        print(f"  Slot {slot}: {count} ({pct:.1f}%)")
    
    # Size distribution
    size_counts = df['improved_size'].value_counts().sort_index()
    print("\nSize distribution:")
    for size, count in size_counts.items():
        pct = count / len(df) * 100
        print(f"  Size {size}: {count} ({pct:.1f}%)")
    
    # Performance comparison
    old_total = df['reward'].sum() if 'reward' in df.columns else 0
    new_total = df['improved_reward'].sum()
    
    print(f"\nPerformance comparison:")
    print(f"  Original total PnL: {old_total:.4f}")
    print(f"  Improved total PnL: {new_total:.4f}")
    print(f"  Improvement: {((new_total / old_total - 1) * 100) if old_total != 0 else 'N/A':.1f}%")
    
    # Hit rate comparison
    old_hit_rate = (df['reward'] > 0).mean() if 'reward' in df.columns else 0
    new_hit_rate = (df['improved_reward'] > 0).mean()
    
    print(f"  Original hit rate: {old_hit_rate:.1%}")
    print(f"  Improved hit rate: {new_hit_rate:.1%}")

def save_improved_decision_table(df: pd.DataFrame, output_path: Path):
    """Save the improved decision table."""
    df.to_csv(output_path, index=False)
    df.to_parquet(output_path.with_suffix('.parquet'), index=False)
    print(f"💾 Saved improved decision table to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Improve behavior policy for RL training")
    parser.add_argument("--input", type=Path, default=Path("iql_cql_artifacts/decision_table.csv"),
                       help="Input decision table")
    parser.add_argument("--output", type=Path, default=Path("improved_behavior_policy"),
                       help="Output directory")
    parser.add_argument("--prob-weight", type=float, default=0.60,
                       help="Weight for best prob profit strategy")
    parser.add_argument("--expected-weight", type=float, default=0.25,
                       help="Weight for best expected return strategy")
    parser.add_argument("--random-weight", type=float, default=0.15,
                       help="Weight for random exploration")
    
    args = parser.parse_args()
    
    # Validate weights
    total_weight = args.prob_weight + args.expected_weight + args.random_weight
    if abs(total_weight - 1.0) > 1e-6:
        raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
    
    strategy_mix = {
        'best_prob_profit': args.prob_weight,
        'best_expected_return': args.expected_weight,
        'random_exploration': args.random_weight
    }
    
    print("🚀 IMPROVING BEHAVIOR POLICY")
    print("=" * 50)
    
    # Load original decision table
    df = load_decision_table(args.input)
    
    # Create improved behavior policy
    df_improved = create_improved_behavior_policy(df, strategy_mix)
    
    # Analyze the improvements
    analyze_improved_policy(df_improved)
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "improved_decision_table.csv"
    save_improved_decision_table(df_improved, output_path)
    
    # Save strategy mix configuration
    config_path = output_dir / "strategy_mix.json"
    with open(config_path, 'w') as f:
        json.dump({
            'strategy_mix': strategy_mix,
            'improvements': {
                'action_diversity': len(df_improved['improved_action_id'].unique()),
                'slot_diversity': len(df_improved['improved_slot'].unique()),
                'total_pnl': df_improved['improved_reward'].sum(),
                'hit_rate': (df_improved['improved_reward'] > 0).mean(),
            }
        }, f, indent=2)
    
    print(f"💾 Configuration saved to: {config_path}")
    print("✅ Behavior policy improvement complete!")
    
    return output_path

if __name__ == "__main__":
    main()