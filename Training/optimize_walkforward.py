#!/usr/bin/env python3
"""
Modern Walkforward Optimizer - Clean Architecture

Optimizes trading parameters using the unified walkforward_simulator.py.
Supports multiple optimization objectives with hard constraints.

Key Features:
- Direct import (no subprocess overhead)
- Leak-free by design (uses walkforward_simulator's leak protection)
- Multiple objectives: Calmar ratio, Sharpe ratio, or custom
- Hard constraints on win rate and trade count
- Proper cleanup and result tracking

Usage:
    python optimize_walkforward.py \
        --decision-table data.csv \
        --policy policy.d3 \
        --meta meta.json \
        --objective calmar \
        --trials 100 \
        --outdir results/optimization

Objectives:
    - calmar: Maximize return/drawdown ratio (default)
    - sharpe: Maximize return/volatility ratio
    - return: Maximize returns with drawdown constraint
    - drawdown: Minimize drawdown with return floor
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# Import unified simulator
from walkforward_simulator import (
    simulate_walkforward,
    load_decision_table,
    _load_meta,
    _load_policy_robust,
    _standardise_states,
)

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class WalkforwardOptimizer:
    """
    Modern optimizer for walkforward trading parameters.
    
    Directly uses walkforward_simulator.py for clean, debuggable optimization.
    """
    
    def __init__(
        self,
        decision_table_path: Path,
        policy_path: Path,
        meta_path: Path,
        objective: str = "calmar",
        initial_capital: float = 10_000.0,
        min_win_rate: float = 0.80,
        min_trades: int = 70,
        max_drawdown_tolerance: float = 0.60,
        min_return_threshold: float = 1000.0,  # Minimum 1000% returns
        mode: str = "backtest",
    ):
        self.decision_table_path = decision_table_path
        self.policy_path = policy_path
        self.meta_path = meta_path
        self.objective = objective
        self.initial_capital = initial_capital
        self.min_win_rate = min_win_rate
        self.min_trades = min_trades
        self.max_drawdown_tolerance = max_drawdown_tolerance
        self.min_return_threshold = min_return_threshold
        self.mode = mode
        
        self.logger = logging.getLogger("walkforward_optimizer")
        self.logger.info(f"🎯 WALKFORWARD OPTIMIZATION")
        self.logger.info(f"  Objective: {objective.upper()}")
        self.logger.info(f"  Mode: {mode.upper()}")
        self.logger.info(f"  Constraints: WR≥{min_win_rate:.1%}, Trades≥{min_trades}, MDD≤{max_drawdown_tolerance:.1%}")
        
        # Cache loaded data (reuse across trials)
        self.meta = _load_meta(self.meta_path)
        self.decision_df = load_decision_table(
            self.decision_table_path,
            self.meta['action_map'],
            mode=self.mode,
            preprocess=True,
            logger=self.logger,
        )
        self.algo = _load_policy_robust(self.policy_path, self.meta)
        
        # Pre-compute states (constant across trials)
        raw_state_cols = self.meta['state_columns']
        self.filtered_state_cols = [
            col for col in raw_state_cols
            if not (
                col.endswith('target_pnl') or col.endswith('future_option_price')
                or col == 's_target_pnl' or col.endswith('contractID')
            )
        ]
        
        if len(self.filtered_state_cols) != len(raw_state_cols):
            keep_indices = [raw_state_cols.index(col) for col in self.filtered_state_cols]
            self.scaler_mean = [self.meta['scaler_mean'][idx] for idx in keep_indices]
            self.scaler_scale = [self.meta['scaler_scale'][idx] for idx in keep_indices]
        else:
            self.scaler_mean = self.meta['scaler_mean']
            self.scaler_scale = self.meta['scaler_scale']
        
        self.states = _standardise_states(
            self.decision_df, self.filtered_state_cols,
            self.scaler_mean, self.scaler_scale
        )
        self.predicted_actions = self.algo.predict(self.states)
        
        self.logger.info(f"✅ Cached data: {len(self.decision_df)} rows, {len(self.filtered_state_cols)} state features")
        
        # Track results
        self.trial_results: list[Dict] = []
    
    def objective_function(self, trial: optuna.Trial) -> float:
        """Optuna objective function with constraint handling."""
        try:
            # Sample parameters
            params = self._sample_parameters(trial)
            
            # Run simulation (fast - no reloading)
            results_df, summary = simulate_walkforward(
                self.decision_df,
                self.predicted_actions,
                self.meta['action_map'],
                params,
                self.initial_capital,
                self.mode,
            )
            
            # Extract metrics
            win_rate = float(summary.get('win_rate', 0.0))
            total_trades = int(summary.get('total_trades', 0))
            max_drawdown = float(summary.get('max_drawdown', 1.0))
            return_pct = float(summary.get('return_pct', 0.0))
            
            # Hard constraints check
            constraints_satisfied = True
            penalty = 0.0
            
            if win_rate < self.min_win_rate:
                penalty += (self.min_win_rate - win_rate) * 10_000
                constraints_satisfied = False
            
            if total_trades < self.min_trades:
                penalty += (self.min_trades - total_trades) * 100
                constraints_satisfied = False
            
            if max_drawdown > self.max_drawdown_tolerance:
                penalty += (max_drawdown - self.max_drawdown_tolerance) * 10_000
                constraints_satisfied = False
            
            if return_pct < self.min_return_threshold:
                penalty += (self.min_return_threshold - return_pct) * 10
                constraints_satisfied = False
            
            # Calculate objective value
            if self.objective == "calmar":
                if max_drawdown <= 0.001:
                    objective_value = return_pct * 1000
                else:
                    objective_value = return_pct / (max_drawdown * 100)
                
                # Bonuses for exceptional performance
                if win_rate >= 0.85:
                    objective_value *= 1.1
                if total_trades >= 85:
                    objective_value *= 1.05
                if max_drawdown <= 0.15:
                    objective_value *= 1.3
                elif max_drawdown <= 0.20:
                    objective_value *= 1.15
                    
            elif self.objective == "sharpe":
                # Sharpe-like: return/volatility (using equity curve)
                equity_series = results_df['equity_curve_value'] if 'equity_curve_value' in results_df.columns else results_df['equity_after']
                equity_returns = equity_series.pct_change().dropna()
                volatility = float(equity_returns.std()) if len(equity_returns) > 0 else 1.0
                objective_value = return_pct / (volatility * 100 * np.sqrt(252)) if volatility > 0 else 0.0
                
            elif self.objective == "return":
                # Pure return with drawdown penalty
                objective_value = return_pct - max_drawdown * 1000
                
            elif self.objective == "drawdown":
                # Minimize drawdown with return shortfall penalty
                shortfall = max(0, self.min_return_threshold - return_pct)
                objective_value = -(max_drawdown + 0.001 * shortfall)  # Negative for minimization
            
            else:
                raise ValueError(f"Unknown objective: {self.objective}")
            
            # Apply penalty if constraints violated
            if self.objective in ["calmar", "sharpe", "return"]:
                # Maximization objectives
                objective_value -= penalty
            else:
                # Minimization objectives
                objective_value += penalty
            
            # Store results
            self.trial_results.append({
                'trial_number': trial.number,
                'params': params.copy(),
                'summary': summary,
                'objective_value': objective_value,
                'constraints_satisfied': constraints_satisfied,
                'metrics': {
                    'win_rate': win_rate,
                    'return_pct': return_pct,
                    'max_drawdown': max_drawdown,
                    'total_trades': total_trades,
                    'calmar_ratio': summary.get('calmar_ratio', 0),
                }
            })
            
            # Log progress
            status = "✅" if constraints_satisfied else "❌"
            self.logger.info(
                f"Trial {trial.number:3d}: {status} obj={objective_value:8.2f} | "
                f"WR={win_rate:.1%} | Ret={return_pct:6.1f}% | MDD={max_drawdown:.1%} | "
                f"Trades={total_trades:3d}"
            )
            
            return objective_value
            
        except Exception as e:
            self.logger.error(f"Trial {trial.number} failed: {e}")
            return -1e9 if self.objective in ["calmar", "sharpe", "return"] else 1e9
    
    def _sample_parameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Sample parameter space for optimization."""
        
        params: Dict[str, Any] = {
            'approach': f'optimized_{self.objective}',
            'base_contracts': 10,
            'base_notional': 1000.0,
            'bypass_all_normal_controls': trial.suggest_categorical('bypass_all_normal_controls', [True, False]),
            'holding_period_days': 5,
            'commission_per_side': 0.65,
            'exchange_fee_per_side': 0.05,
            'slippage_min': 0.02,
            'slippage_pct': 0.20,
        }
        
        # Equity cap
        params['max_notional_pct'] = trial.suggest_float('max_notional_pct', 0.08, 0.20, step=0.01)
        
        # Portfolio stop loss
        params['enable_portfolio_stop_loss'] = trial.suggest_categorical('enable_portfolio_stop_loss', [True, False])
        if params['enable_portfolio_stop_loss']:
            params['portfolio_stop_loss_pct'] = trial.suggest_float('portfolio_stop_loss_pct', 0.30, 0.80, step=0.05)
        
        # Single trade cap
        params['enable_single_trade_cap'] = trial.suggest_categorical('enable_single_trade_cap', [True, False])
        if params['enable_single_trade_cap']:
            params['max_single_trade_notional'] = trial.suggest_int('max_single_trade_notional', 50_000, 200_000, step=25_000)
        
        # Market halt protection
        params['enable_market_halt_protection'] = trial.suggest_categorical('enable_market_halt_protection', [True, False])
        if params['enable_market_halt_protection']:
            params['halt_vol_emergency_only'] = trial.suggest_categorical('halt_vol_emergency_only', [True, False])
            if not params['halt_vol_emergency_only']:
                params['halt_vol_severity_threshold'] = trial.suggest_float('halt_vol_severity_threshold', 1.5, 3.0, step=0.25)
        
        # Consecutive loss breaker
        params['enable_consecutive_loss_breaker'] = trial.suggest_categorical('enable_consecutive_loss_breaker', [True, False])
        if params['enable_consecutive_loss_breaker']:
            params['max_consecutive_losses'] = trial.suggest_int('max_consecutive_losses', 5, 50, step=5)
        
        # Position multiplier (key return lever)
        params['enable_position_multiplier'] = trial.suggest_categorical('enable_position_multiplier', [True, False])
        if params['enable_position_multiplier']:
            params['position_multiplier'] = trial.suggest_float('position_multiplier', 1.0, 3.0, step=0.1)
        
        # Dynamic sizing
        params['enable_dynamic_sizing'] = trial.suggest_categorical('enable_dynamic_sizing', [True, False])
        if params['enable_dynamic_sizing']:
            params['lookback_window'] = trial.suggest_int('lookback_window', 5, 25, step=5)
        
        # Volatility adjustment
        params['enable_vol_adjustment'] = trial.suggest_categorical('enable_vol_adjustment', [True, False])
        if params['enable_vol_adjustment']:
            params['vol_lookback'] = trial.suggest_int('vol_lookback', 10, 40, step=5)
        
        # Return filter (FIXED: uses expected_return from CQF, not target_pnl)
        params['enable_return_filter'] = trial.suggest_categorical('enable_return_filter', [True, False])
        if params['enable_return_filter']:
            params['min_expected_return'] = trial.suggest_float('min_expected_return', 0.0, 0.05, step=0.005)
        
        return params
    
    def optimize(
        self,
        n_trials: int = 100,
        study_name: Optional[str] = None,
    ) -> Tuple[optuna.Study, Dict[str, Any]]:
        """Run optimization study."""
        
        if study_name is None:
            study_name = f"walkforward_{self.objective}_{self.mode}"
        
        # Determine direction
        direction = "maximize" if self.objective in ["calmar", "sharpe", "return"] else "minimize"
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"🚀 STARTING OPTIMIZATION")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"  Study: {study_name}")
        self.logger.info(f"  Objective: {self.objective.upper()} ({direction})")
        self.logger.info(f"  Trials: {n_trials}")
        self.logger.info(f"  Data: {len(self.decision_df)} decision points")
        
        # Create study
        study = optuna.create_study(
            direction=direction,
            sampler=TPESampler(seed=42, multivariate=True, group=True),
            pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=20),
            study_name=study_name,
        )
        
        # Run optimization
        study.optimize(
            self.objective_function,
            n_trials=n_trials,
            show_progress_bar=True,
            gc_after_trial=True,
        )
        
        # Analyze results
        best_trial = study.best_trial
        
        # Find best constraint-satisfying trial
        valid_trials = [r for r in self.trial_results if r['constraints_satisfied']]
        
        if valid_trials:
            if direction == "maximize":
                best_valid = max(valid_trials, key=lambda x: x['objective_value'])
            else:
                best_valid = min(valid_trials, key=lambda x: x['objective_value'])
        else:
            best_valid = None
            self.logger.warning("⚠️  NO trials satisfied all constraints!")
        
        # Log summary
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"✅ OPTIMIZATION COMPLETE")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"  Best trial: #{best_trial.number}")
        self.logger.info(f"  Best {self.objective}: {best_trial.value:.2f}")
        self.logger.info(f"  Valid trials: {len(valid_trials)}/{len(self.trial_results)}")
        
        if best_valid:
            metrics = best_valid['metrics']
            summary = best_valid['summary']
            
            self.logger.info(f"\n📊 BEST VALID CONFIGURATION:")
            self.logger.info(f"  Trial: #{best_valid['trial_number']}")
            self.logger.info(f"  Win Rate: {metrics['win_rate']:.2%}")
            self.logger.info(f"  Return: {metrics['return_pct']:.1f}%")
            self.logger.info(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
            self.logger.info(f"  Calmar Ratio: {metrics['calmar_ratio']:.1f}")
            self.logger.info(f"  Total Trades: {metrics['total_trades']}")
            self.logger.info(f"  Halted Trades: {summary.get('halted_trades', 0)}")
            
            # Compare to baseline
            baseline = {'win_rate': 0.839, 'return_pct': 1119.7, 'max_drawdown': 0.649}
            self.logger.info(f"\n🚀 vs BASELINE:")
            self.logger.info(f"  Win Rate: {(metrics['win_rate'] - baseline['win_rate'])*100:+.1f}pp")
            self.logger.info(f"  Return: {(metrics['return_pct'] / baseline['return_pct'] - 1)*100:+.1f}%")
            self.logger.info(f"  Drawdown: {(baseline['max_drawdown'] - metrics['max_drawdown'])/baseline['max_drawdown']*100:+.1f}%")
        
        # Compile optimization summary
        optimization_summary = {
            'study_name': study_name,
            'objective': self.objective,
            'mode': self.mode,
            'n_trials': n_trials,
            'constraints': {
                'min_win_rate': self.min_win_rate,
                'min_trades': self.min_trades,
                'max_drawdown_tolerance': self.max_drawdown_tolerance,
                'min_return_threshold': self.min_return_threshold,
            },
            'baseline': {
                'win_rate': 0.839,
                'return_pct': 1119.7,
                'max_drawdown': 0.649,
                'trades': 87,
                'calmar_ratio': 17.3,
            },
            'best_trial_number': best_trial.number,
            'best_objective_value': best_trial.value,
            'best_params': best_trial.params,
            'best_valid_trial': best_valid['trial_number'] if best_valid else None,
            'best_valid_params': best_valid['params'] if best_valid else None,
            'best_valid_summary': best_valid['summary'] if best_valid else None,
            'best_valid_metrics': best_valid['metrics'] if best_valid else None,
            'valid_trials_count': len(valid_trials),
            'total_trials': len(self.trial_results),
        }
        
        return study, optimization_summary


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Modern walkforward parameter optimizer")
    
    # Required inputs
    parser.add_argument('--decision-table', type=Path, required=True,
                       help="Decision table CSV with candidate features")
    parser.add_argument('--policy', type=Path, required=True,
                       help="Trained IQL policy (.d3 file)")
    parser.add_argument('--meta', type=Path, required=True,
                       help="Policy metadata JSON")
    
    # Optimization settings
    parser.add_argument('--objective', choices=['calmar', 'sharpe', 'return', 'drawdown'],
                       default='calmar', help="Optimization objective")
    parser.add_argument('--mode', choices=['backtest', 'leakfree'], default='backtest',
                       help="Simulation mode")
    parser.add_argument('--trials', type=int, default=100,
                       help="Number of Optuna trials")
    parser.add_argument('--study-name', type=str, default=None,
                       help="Optuna study name (default: auto-generated)")
    
    # Constraints
    parser.add_argument('--min-win-rate', type=float, default=0.80,
                       help="Minimum win rate constraint")
    parser.add_argument('--min-trades', type=int, default=70,
                       help="Minimum number of trades")
    parser.add_argument('--max-drawdown', type=float, default=0.60,
                       help="Maximum drawdown tolerance")
    parser.add_argument('--min-return', type=float, default=1000.0,
                       help="Minimum return threshold (percent)")
    
    # Output
    parser.add_argument('--outdir', type=Path, default=Path('results/walkforward_optimization'),
                       help="Output directory for results")
    parser.add_argument('--initial-capital', type=float, default=10_000.0,
                       help="Initial capital for simulation")
    
    args = parser.parse_args()
    
    # Validate inputs
    for file_path in [args.decision_table, args.policy, args.meta]:
        if not file_path.exists():
            raise FileNotFoundError(f"Required file not found: {file_path}")
    
    # Create output directory
    args.outdir.mkdir(parents=True, exist_ok=True)
    
    # Create optimizer
    optimizer = WalkforwardOptimizer(
        decision_table_path=args.decision_table,
        policy_path=args.policy,
        meta_path=args.meta,
        objective=args.objective,
        initial_capital=args.initial_capital,
        min_win_rate=args.min_win_rate,
        min_trades=args.min_trades,
        max_drawdown_tolerance=args.max_drawdown,
        min_return_threshold=args.min_return,
        mode=args.mode,
    )
    
    # Run optimization
    study, summary = optimizer.optimize(
        n_trials=args.trials,
        study_name=args.study_name,
    )
    
    # Save results
    results_file = args.outdir / f'optimization_{args.objective}_{args.mode}.json'
    with open(results_file, 'w') as f:
        # Convert for JSON serialization
        clean_summary = json.loads(json.dumps(summary, default=str))
        json.dump(clean_summary, f, indent=2, default=str)
    
    optimizer.logger.info(f"\n💾 Results saved to: {results_file}")
    
    # Save best parameters
    if summary['best_valid_params']:
        best_params_file = args.outdir / f'best_params_{args.objective}.json'
        with open(best_params_file, 'w') as f:
            json.dump(summary['best_valid_params'], f, indent=2)
        optimizer.logger.info(f"💾 Best params saved to: {best_params_file}")
        
        # Save best summary
        best_summary_file = args.outdir / f'best_summary_{args.objective}.json'
        with open(best_summary_file, 'w') as f:
            json.dump(summary['best_valid_summary'], f, indent=2, default=str)
        optimizer.logger.info(f"💾 Best summary saved to: {best_summary_file}")
    
    # Print final results
    print(f"\n{'='*80}")
    print(f"🏆 OPTIMIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"Objective: {args.objective.upper()}")
    print(f"Mode: {args.mode.upper()}")
    print(f"Trials: {summary['total_trials']} ({summary['valid_trials_count']} valid)")
    
    if summary['best_valid_metrics']:
        m = summary['best_valid_metrics']
        print(f"\n📊 Best Configuration:")
        print(f"  Win Rate: {m['win_rate']:.2%}")
        print(f"  Return: {m['return_pct']:.1f}%")
        print(f"  Max Drawdown: {m['max_drawdown']:.2%}")
        print(f"  Calmar Ratio: {m['calmar_ratio']:.1f}")
        print(f"  Trades: {m['total_trades']}")
        
        print(f"\n🎯 Key Parameters:")
        if summary['best_valid_params']:
            for key, value in summary['best_valid_params'].items():
                if key.startswith('enable_') and value:
                    print(f"  ✅ {key}: {value}")
                elif not key.startswith('enable_') and key not in ['approach', 'base_contracts', 'base_notional', 'commission_per_side', 'exchange_fee_per_side', 'slippage_min', 'slippage_pct']:
                    print(f"  {key}: {value}")
    
    print(f"\n💾 Results: {args.outdir}")
    
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
