#!/usr/bin/env python3
"""
Optuna-based optimization of behavior policy parameters for DiscreteCQL training.
Systematically searches for optimal strategy mix weights, thresholds, and other hyperparameters.
"""

import pandas as pd
import numpy as np
import json
import logging
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Tuple, Optional
import argparse

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# Add Training to path
sys.path.insert(0, str(Path(__file__).parent / "Training"))

from iql_pipeline import (
    to_mdp_dataset, train_iql, export_policy, 
    BuildConfig, TrainConfig, _action_mapping
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BehaviorPolicyOptimizer:
    """Optuna-based optimizer for behavior policy parameters."""
    
    def __init__(self, decision_table_path: str, n_trials: int = 50, fast_training: bool = True):
        self.decision_table_path = decision_table_path
        self.n_trials = n_trials
        self.fast_training = fast_training
        
        # Load base decision table
        self.base_df = pd.read_csv(decision_table_path)
        logger.info(f"Loaded base decision table: {len(self.base_df)} decisions")
        
        # Create temp directory for optimization runs
        self.temp_dir = Path(tempfile.mkdtemp(prefix="optuna_behavior_"))
        logger.info(f"Created temp directory: {self.temp_dir}")
        
    def create_action_mapping(self) -> Dict[int, Dict]:
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

    def find_best_slot_strategy(self, row: pd.Series, strategy: str) -> Tuple[int, float]:
        """Find the best slot according to a given strategy."""
        best_slot = 1
        best_value = float('-inf') if strategy != 'prob_profit' else 0.0
        
        for slot in range(1, 6):
            if strategy == 'expected_return':
                value = row.get(f'c{slot}_expected_return', float('-inf'))
            elif strategy == 'prob_profit':
                value = row.get(f'c{slot}_prob_profit', 0.0)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
                
            if pd.notna(value) and value > best_value:
                best_value = value
                best_slot = slot
                
        return best_slot, best_value

    def create_optimized_behavior_policy(self, trial_params: Dict) -> pd.DataFrame:
        """Create behavior policy with Optuna-suggested parameters."""
        df = self.base_df.copy()
        action_map = self.create_action_mapping()
        
        # Extract trial parameters
        strategy_mix = {
            'best_prob_profit': trial_params['prob_weight'],
            'best_expected_return': trial_params['exp_weight'],
            'random_exploration': trial_params['expl_weight']
        }
        
        prob_threshold = trial_params['prob_threshold']
        exp_threshold = trial_params['exp_threshold']
        exploration_probs = trial_params['exploration_probs']
        size_ratio_conservative = trial_params['size_ratio_conservative']
        size_ratio_aggressive = trial_params['size_ratio_aggressive']
        
        # Track decisions
        selected_strategies = []
        selected_action_ids = []
        rewards = []
        
        np.random.seed(42)  # Reproducible results
        
        for idx, (_, row) in enumerate(df.iterrows()):
            # Choose strategy based on optimized mix
            rand = np.random.random()
            
            if rand < strategy_mix['best_prob_profit']:
                strategy = 'best_prob_profit'
                slot, _ = self.find_best_slot_strategy(row, 'prob_profit')
            elif rand < strategy_mix['best_prob_profit'] + strategy_mix['best_expected_return']:
                strategy = 'best_expected_return'
                slot, _ = self.find_best_slot_strategy(row, 'expected_return')
            else:
                strategy = 'random_exploration'
                # Use optimized exploration probabilities
                slot = np.random.choice([1, 2, 3, 4, 5], p=exploration_probs)
            
            # Choose position size with optimized thresholds
            if strategy == 'best_prob_profit':
                prob_profit = row.get(f'c{slot}_prob_profit', 0.0)
                if prob_profit > prob_threshold:
                    size_value = size_ratio_aggressive
                else:
                    size_value = size_ratio_conservative
            elif strategy == 'best_expected_return':
                expected_return = row.get(f'c{slot}_expected_return', 0.0)
                if expected_return > exp_threshold:
                    size_value = size_ratio_aggressive
                else:
                    size_value = size_ratio_conservative
            else:  # random exploration
                # Conservative sizing for exploration
                size_value = np.random.choice([size_ratio_conservative, size_ratio_aggressive], 
                                            p=[0.7, 0.3])
            
            # Map continuous size values to discrete action space (0.5 or 1.0)
            discrete_size = 1.0 if size_value > 0.65 else 0.5
            
            # Find corresponding action_id
            action_id = None
            for aid, action_info in action_map.items():
                if action_info['slot'] == slot and action_info['size_value'] == discrete_size:
                    action_id = aid
                    break
            
            if action_id is None:
                # Fallback to closest match
                logger.warning(f"Could not find action_id for slot={slot}, discrete_size={discrete_size}")
                action_id = 1  # Default to action 1
            
            # Get reward (target PnL adjusted by original continuous size for accurate reward calculation)
            target_pnl_col = f'c{slot}_target_pnl'
            target_pnl = row.get(target_pnl_col, 0.0) if target_pnl_col in row else 0.0
            reward = target_pnl * size_value if pd.notna(target_pnl) else 0.0
            
            selected_strategies.append(strategy)
            selected_action_ids.append(action_id)
            rewards.append(reward)
        
        # Add behavior policy columns
        df['behavior_strategy'] = selected_strategies
        df['action_id'] = selected_action_ids
        df['reward'] = rewards
        
        # Add slot and size info for analysis
        df['behavior_slot'] = [action_map[aid]['slot'] for aid in selected_action_ids]
        df['behavior_size_idx'] = [action_map[aid]['size_idx'] for aid in selected_action_ids]
        
        return df

    def train_fast_cql(self, df: pd.DataFrame, trial_num: int) -> Tuple[float, Dict]:
        """Train DiscreteCQL quickly for optimization (reduced steps)."""
        
        # Create trial-specific output directory
        trial_dir = self.temp_dir / f"trial_{trial_num}"
        trial_dir.mkdir(exist_ok=True)
        
        algo = None
        dataset = None
        
        try:
            # Configuration for fast training
            cfg_build = BuildConfig(
                top_k=5,
                size_bins=[0.5, 1.0],
                min_prob=0.0,
                min_q05=-10.0,
                reward_col="reward",
                risk_lambda=0.5,
                group_keys=["date", "underlying"],
                group_top_n=None
            )
            
            # Reduced training steps for optimization speed
            training_steps = 500 if self.fast_training else 1500  # Even more reduced for stability
            
            cfg_train = TrainConfig(
                steps=training_steps,
                batch_size=16,  # Smaller batch for memory efficiency  
                expectile=0.7,
                temperature=3.0,
                gamma=0.99,
                seed=42
            )
            
            # Create MDP dataset
            dataset, scaler, state_cols = to_mdp_dataset(df, scaler=None)
            action_size = 11
            
            # Train DiscreteCQL
            algo = train_iql(dataset, action_size, cfg_train, trial_dir)
            
            # Evaluate performance on training data
            performance_metrics = self.evaluate_policy_performance(df)
            
            return performance_metrics['objective_score'], performance_metrics
            
        except Exception as e:
            logger.warning(f"Trial {trial_num} failed: {e}")
            return -1000.0, {'error': str(e)}
        
        finally:
            # Aggressive cleanup
            del algo
            del dataset
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Cleanup trial directory
            if trial_dir.exists():
                shutil.rmtree(trial_dir)

    def evaluate_policy_performance(self, df: pd.DataFrame) -> Dict:
        """Evaluate the performance of the behavior policy."""
        rewards = df['reward'].dropna()
        
        if len(rewards) == 0:
            return {'objective_score': -1000.0, 'error': 'no_rewards'}
        
        # Calculate multiple metrics
        total_pnl = rewards.sum()
        mean_pnl = rewards.mean()
        hit_rate = (rewards > 0).mean()
        sharpe = mean_pnl / (rewards.std() + 1e-8)
        max_drawdown = (rewards.cumsum() - rewards.cumsum().cummax()).min()
        action_diversity = len(df['action_id'].unique())
        
        # Composite objective score (weighted combination)
        objective_score = (
            0.4 * total_pnl +           # Total return
            0.3 * sharpe +              # Risk-adjusted return  
            0.2 * hit_rate * 100 +      # Success rate
            0.1 * action_diversity      # Diversity bonus
        )
        
        return {
            'objective_score': objective_score,
            'total_pnl': total_pnl,
            'mean_pnl': mean_pnl,
            'hit_rate': hit_rate,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'action_diversity': action_diversity,
            'num_decisions': len(df)
        }

    def objective(self, trial) -> float:
        """Optuna objective function."""
        
        # Suggest strategy mix weights (must sum to 1.0)
        prob_weight = trial.suggest_float('prob_weight', 0.3, 0.8)
        exp_weight_raw = trial.suggest_float('exp_weight', 0.1, 0.4)

        # Ensure weights are valid (allocate at least 5% to exploration)
        max_exp_weight = min(exp_weight_raw, 1.0 - prob_weight - 0.05)
        exp_weight = max(0.05, max_exp_weight)
        expl_weight = max(0.05, 1.0 - prob_weight - exp_weight)
        total_weight = prob_weight + exp_weight + expl_weight
        if total_weight != 1.0:
            prob_weight /= total_weight
            exp_weight /= total_weight
            expl_weight /= total_weight
        
        # Suggest thresholds
        prob_threshold = trial.suggest_float('prob_threshold', 0.4, 0.8)
        exp_threshold = trial.suggest_float('exp_threshold', 0.05, 0.2)
        
        # Suggest exploration slot preferences (must sum to 1.0)
        slot1_prob = trial.suggest_float('slot1_prob', 0.2, 0.6)
        slot2_prob = trial.suggest_float('slot2_prob', 0.1, 0.3)
        slot3_prob = trial.suggest_float('slot3_prob', 0.1, 0.3)
        slot4_prob = trial.suggest_float('slot4_prob', 0.05, 0.2)
        
        # Normalize exploration probabilities
        total_prob = slot1_prob + slot2_prob + slot3_prob + slot4_prob
        slot5_prob = max(0.05, 1.0 - total_prob)  # Ensure slot5 gets at least 5%
        total_with_slot5 = total_prob + slot5_prob
        
        exploration_probs = [
            slot1_prob / total_with_slot5,
            slot2_prob / total_with_slot5,
            slot3_prob / total_with_slot5,
            slot4_prob / total_with_slot5,
            slot5_prob / total_with_slot5
        ]
        
        # Suggest position sizing ratios
        size_ratio_conservative = trial.suggest_float('size_ratio_conservative', 0.3, 0.7)
        size_ratio_aggressive = trial.suggest_float('size_ratio_aggressive', 0.8, 1.0)
        
        trial.set_user_attr('prob_weight_adj', prob_weight)
        trial.set_user_attr('exp_weight_adj', exp_weight)
        trial.set_user_attr('expl_weight_adj', expl_weight)
        trial.set_user_attr('exploration_probs', exploration_probs)

        # Package trial parameters for behaviour policy creation
        trial_params = {
            'prob_weight': prob_weight,
            'exp_weight': exp_weight,
            'expl_weight': expl_weight,
            'prob_threshold': prob_threshold,
            'exp_threshold': exp_threshold,
            'exploration_probs': exploration_probs,
            'size_ratio_conservative': size_ratio_conservative,
            'size_ratio_aggressive': size_ratio_aggressive
        }
        
        # Create behavior policy with these parameters
        df = self.create_optimized_behavior_policy(trial_params)
        
        # Train and evaluate
        objective_score, metrics = self.train_fast_cql(df, trial.number)
        
        # Log trial results
        logger.info(f"Trial {trial.number}: Objective={objective_score:.4f}, "
                   f"TotalPnL={metrics.get('total_pnl', 0):.4f}, "
                   f"Sharpe={metrics.get('sharpe', 0):.3f}, "
                   f"HitRate={metrics.get('hit_rate', 0):.1%}")
        
        return objective_score

    def optimize(self, output_dir: str = "optuna_behavior_results") -> Dict:
        """Run Optuna optimization."""
        
        print("🚀 Starting Optuna Behavior Policy Optimization")
        print("=" * 60)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Create database for persistent storage
        db_path = output_path / "optuna_study.db"
        study_name = "behavior_policy_optimization"
        storage = f"sqlite:///{db_path}"
        
        # Create Optuna study with persistent storage
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction='maximize',
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5),
            load_if_exists=True  # Resume if study exists
        )
        
        # Track completed trials
        completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        remaining_trials = max(0, self.n_trials - completed_trials)
        
        print(f"📊 Study status: {completed_trials}/{self.n_trials} trials completed")
        if remaining_trials > 0:
            print(f"⏳ Running {remaining_trials} remaining trials...")
            
            # Run optimization with checkpointing
            try:
                study.optimize(self.objective, n_trials=remaining_trials)
            except KeyboardInterrupt:
                print("\n⚠️  Optimization interrupted by user")
                print(f"Completed {len(study.trials)} trials before interruption")
            except Exception as e:
                print(f"\n❌ Optimization failed: {e}")
                print(f"Completed {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])} trials before failure")
                
        else:
            print("✅ All trials already completed!")
        
        # Get best parameters
        best_trial = study.best_trial
        best_params = dict(best_trial.params)
        best_params.update({
            'prob_weight': best_trial.user_attrs.get('prob_weight_adj', best_params.get('prob_weight')),
            'exp_weight': best_trial.user_attrs.get('exp_weight_adj', best_params.get('exp_weight')),
            'expl_weight': best_trial.user_attrs.get('expl_weight_adj', 1.0 - best_params.get('prob_weight', 0) - best_params.get('exp_weight', 0)),
            'exploration_probs': best_trial.user_attrs.get('exploration_probs')
        })
        best_value = best_trial.value
        
        print(f"\n✅ Optimization Complete!")
        print(f"Best objective score: {best_value:.4f}")
        print(f"Best parameters:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        
        # Create optimal behavior policy
        optimal_df = self.create_optimized_behavior_policy(best_params)
        optimal_metrics = self.evaluate_policy_performance(optimal_df)
        
        # Save results
        results = {
            'optimization_summary': {
                'n_trials': self.n_trials,
                'best_objective_score': best_value,
                'best_parameters': best_params,
                'optimal_metrics': optimal_metrics
            },
            'study_trials': []
        }
        
        # Save trial history
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                results['study_trials'].append({
                    'number': trial.number,
                    'value': trial.value,
                    'params': trial.params
                })
        
        # Save optimal decision table
        optimal_df.to_csv(output_path / "optimal_decision_table.csv", index=False)
        try:
            optimal_df.to_parquet(output_path / "optimal_decision_table.parquet", index=False)
        except ImportError:
            logger.warning("pyarrow/fastparquet missing; skipping Parquet export for optimal decision table")
        
        # Save optimization results
        with open(output_path / "optimization_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save Optuna study
        study_path = output_path / "optuna_study.pkl"
        with open(study_path, 'wb') as f:
            import pickle
            pickle.dump(study, f)
        
        print(f"\n📁 Results saved to: {output_path}")
        print(f"  - optimal_decision_table.csv (ready for DiscreteCQL training)")
        print(f"  - optimization_results.json (detailed results)")
        print(f"  - optuna_study.pkl (for further analysis)")
        
        # Print some immediate results for verification
        print(f"\n📊 Optimal Policy Performance:")
        print(f"  Total PnL: {optimal_metrics['total_pnl']:.4f}")
        print(f"  Sharpe: {optimal_metrics['sharpe']:.3f}") 
        print(f"  Hit Rate: {optimal_metrics['hit_rate']:.1%}")
        print(f"  Action Diversity: {optimal_metrics['action_diversity']}")
        
        # Cleanup temp directory
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Optimize behavior policy with Optuna")
    parser.add_argument("--input", type=str, 
                       default="iql_cql_artifacts/decision_table.csv",
                       help="Input decision table path")
    parser.add_argument("--output", type=str,
                       default="optuna_behavior_results",
                       help="Output directory")
    parser.add_argument("--n-trials", type=int, default=50,
                       help="Number of Optuna trials")
    parser.add_argument("--fast-training", action="store_true", default=True,
                       help="Use fast training (1000 steps) for optimization")
    
    args = parser.parse_args()
    
    # Initialize optimizer
    optimizer = BehaviorPolicyOptimizer(
        decision_table_path=args.input,
        n_trials=args.n_trials,
        fast_training=args.fast_training
    )
    
    # Run optimization
    results = optimizer.optimize(args.output)
    
    print(f"\n🎯 Next Steps:")
    print(f"  1. Review optimization results in {args.output}/")
    print(f"  2. Train full DiscreteCQL with optimal parameters:")
    print(f"     python train_improved_cql.py --input {args.output}/optimal_decision_table.csv")

if __name__ == "__main__":
    main()
