#!/usr/bin/env python3
"""
Optuna Risk Parameter Optimization for Trading System

This script uses Optuna to systematically optimize risk management parameters
to maximize risk-adjusted returns while maintaining drawdown control.

Key optimization targets:
- Maximize Sharpe ratio (return/volatility)
- Minimize maximum drawdown
- Maintain high win rate
- Optimize profit factor
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner

# Import our optimized walkforward simulation
import sys
sys.path.append(str(Path(__file__).parent))
from optimized_walkforward_simulation import (
    simulate_optimized_walkforward, _load_meta, _load_policy_robust, _standardise_states
)

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO)


class RiskOptimizer:
    """Optuna-based risk parameter optimizer for trading systems."""
    
    def __init__(
        self,
        decision_table_path: Path,
        policy_path: Path,
        meta_path: Path,
        initial_capital: float = 10_000.0,
        optimization_metric: str = "sharpe_ratio"  # sharpe_ratio, return_drawdown_ratio, profit_factor
    ):
        self.decision_table_path = decision_table_path
        self.policy_path = policy_path
        self.meta_path = meta_path
        self.initial_capital = initial_capital
        self.optimization_metric = optimization_metric
        
        # Load data once for all trials
        self.logger = logging.getLogger("risk_optimizer")
        self.logger.info("Loading decision table and policy...")
        
        self.df = pd.read_csv(decision_table_path)
        self.meta = _load_meta(meta_path)
        self.states = _standardise_states(
            self.df, self.meta['state_columns'], self.meta['scaler_mean'], self.meta['scaler_scale']
        )
        self.algo = _load_policy_robust(policy_path, self.meta)
        self.predicted_actions = self.algo.predict(self.states)
        
        self.logger.info(f"Loaded {len(self.df)} decision points for optimization")
        self.logger.info(f"Optimization metric: {optimization_metric}")
        
        # Track best results
        self.best_results = []
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Optuna objective function to optimize risk parameters.
        
        Returns the metric to be maximized (higher = better).
        """
        try:
            # Sample risk parameters from defined ranges
            params = self._sample_parameters(trial)
            
            # Run simulation with sampled parameters
            results, summary = simulate_optimized_walkforward(
                self.df.copy(),
                self.predicted_actions,
                self.meta['action_map'],
                initial_capital=self.initial_capital,
                **params
            )
            
            # Calculate optimization metric
            metric_value = self._calculate_metric(results, summary, params)
            
            # Store results for analysis
            trial_result = {
                'trial_number': trial.number,
                'params': params,
                'summary': summary,
                'metric_value': metric_value
            }
            self.best_results.append(trial_result)
            
            # Log trial results
            self.logger.info(
                f"Trial {trial.number}: {self.optimization_metric}={metric_value:.4f}, "
                f"Return={summary['return_pct']:.1f}%, Win Rate={summary['win_rate']:.1%}, "
                f"Max DD={summary.get('max_drawdown', 0):.1%}"
            )
            
            return metric_value
            
        except Exception as e:
            self.logger.error(f"Trial {trial.number} failed: {e}")
            return -np.inf  # Return very bad score for failed trials
    
    def _sample_parameters(self, trial: optuna.Trial) -> Dict:
        """Sample risk parameters for optimization."""
        
        # Core position sizing parameters
        base_contracts = trial.suggest_int('base_contracts', 3, 12)  # 3-12 contracts
        base_notional = trial.suggest_float('base_notional', 500.0, 2000.0, step=100.0)  # $500-$2000
        
        # Risk control parameters
        enable_risk_controls = trial.suggest_categorical('enable_risk_controls', [True, False])
        
        if enable_risk_controls:
            # Portfolio allocation limits
            max_notional_pct = trial.suggest_float('max_notional_pct', 0.10, 0.40, step=0.05)  # 10-40%
            
            # Volatility regime scaling (how much to reduce during vol spikes)
            vol_emergency_mult = trial.suggest_float('vol_emergency_mult', 0.2, 0.8, step=0.1)  # 20-80%
            vol_high_mult = trial.suggest_float('vol_high_mult', 0.4, 0.9, step=0.1)  # 40-90%
            vol_mod_mult = trial.suggest_float('vol_mod_mult', 0.6, 1.0, step=0.1)  # 60-100%
            
            # Drawdown scaling parameters
            drawdown_scale_factor = trial.suggest_float('drawdown_scale_factor', 0.5, 3.0, step=0.25)  # 0.5x-3x
            min_drawdown_scale = trial.suggest_float('min_drawdown_scale', 0.1, 0.5, step=0.1)  # 10-50%
            
            # Equity scaling (let winners run)
            enable_equity_scaling = trial.suggest_categorical('enable_equity_scaling', [True, False])
            max_equity_mult = trial.suggest_float('max_equity_mult', 1.0, 2.0, step=0.1) if enable_equity_scaling else 1.0
            equity_threshold = trial.suggest_float('equity_threshold', 1.2, 2.0, step=0.1) if enable_equity_scaling else 1.5
        else:
            # No risk controls - use defaults
            max_notional_pct = 0.5  # Allow higher allocation
            vol_emergency_mult = vol_high_mult = vol_mod_mult = 1.0  # No vol scaling
            drawdown_scale_factor = 0.0  # No drawdown scaling
            min_drawdown_scale = 1.0
            enable_equity_scaling = False
            max_equity_mult = 1.0
            equity_threshold = 1.5
        
        # Transaction costs (allow optimization of cost assumptions)
        commission_per_side = trial.suggest_float('commission_per_side', 0.50, 1.00, step=0.05)  # $0.50-$1.00
        exchange_fee_per_side = trial.suggest_float('exchange_fee_per_side', 0.02, 0.10, step=0.01)  # $0.02-$0.10
        slippage_min = trial.suggest_float('slippage_min', 0.01, 0.05, step=0.01)  # $0.01-$0.05
        slippage_pct = trial.suggest_float('slippage_pct', 0.10, 0.30, step=0.05)  # 10-30%
        
        return {
            # Core parameters
            'base_contracts': base_contracts,
            'base_notional': base_notional,
            'enable_risk_controls': enable_risk_controls,
            'max_notional_pct': max_notional_pct,
            
            # Cost parameters  
            'commission_per_side': commission_per_side,
            'exchange_fee_per_side': exchange_fee_per_side,
            'slippage_min': slippage_min,
            'slippage_pct': slippage_pct,
            
            # Store risk control parameters for custom simulation logic
            '_vol_emergency_mult': vol_emergency_mult,
            '_vol_high_mult': vol_high_mult,
            '_vol_mod_mult': vol_mod_mult,
            '_drawdown_scale_factor': drawdown_scale_factor,
            '_min_drawdown_scale': min_drawdown_scale,
            '_enable_equity_scaling': enable_equity_scaling,
            '_max_equity_mult': max_equity_mult,
            '_equity_threshold': equity_threshold,
        }
    
    def _calculate_metric(self, results: pd.DataFrame, summary: Dict, params: Dict) -> float:
        """Calculate the optimization metric."""
        
        if summary['total_trades'] == 0:
            return -1000.0  # Heavily penalize no trading
        
        return_pct = summary['return_pct'] / 100.0  # Convert to decimal
        
        if self.optimization_metric == "sharpe_ratio":
            # Calculate Sharpe ratio using daily returns
            if len(results) < 10:
                return -100.0
            
            daily_returns = results['equity_after'].pct_change().dropna()
            if len(daily_returns) == 0 or daily_returns.std() == 0:
                return -100.0
            
            sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)  # Annualized
            return float(sharpe)
            
        elif self.optimization_metric == "return_drawdown_ratio":
            # Return / Max Drawdown ratio (higher is better)
            max_dd = max(summary.get('max_drawdown', 0.01), 0.01)  # Minimum 1% to avoid division by zero
            return return_pct / max_dd
            
        elif self.optimization_metric == "profit_factor":
            # Profit factor (higher is better)
            return summary.get('profit_factor', 0.0)
            
        elif self.optimization_metric == "win_rate_adjusted_return":
            # Return * Win Rate (favors consistent strategies)
            win_rate = summary.get('win_rate', 0.0)
            return return_pct * win_rate
            
        elif self.optimization_metric == "composite_score":
            # Composite score balancing multiple objectives
            win_rate = summary.get('win_rate', 0.0)
            profit_factor = summary.get('profit_factor', 1.0)
            max_dd = max(summary.get('max_drawdown', 0.01), 0.01)
            
            # Weighted composite score
            score = (
                return_pct * 0.3 +           # 30% weight on returns
                win_rate * 0.25 +            # 25% weight on win rate
                (profit_factor - 1) * 0.25 + # 25% weight on profit factor above 1
                (1 - max_dd) * 0.20          # 20% weight on drawdown control
            )
            return float(score)
            
        else:
            # Default to simple return
            return return_pct
    
    def optimize(
        self,
        n_trials: int = 200,
        study_name: Optional[str] = None,
        storage: Optional[str] = None
    ) -> optuna.Study:
        """Run Optuna optimization study."""
        
        if study_name is None:
            study_name = f"risk_optimization_{self.optimization_metric}"
        
        # Create study with pruning for efficiency
        sampler = TPESampler(seed=42, n_startup_trials=20)
        pruner = HyperbandPruner(min_resource=10, reduction_factor=3)
        
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            pruner=pruner,
            study_name=study_name,
            storage=storage
        )
        
        self.logger.info(f"Starting optimization with {n_trials} trials...")
        self.logger.info(f"Optimizing for: {self.optimization_metric}")
        
        # Run optimization
        study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)
        
        return study
    
    def analyze_results(self, study: optuna.Study, output_dir: Path) -> Dict:
        """Analyze optimization results and save insights."""
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get best trial
        best_trial = study.best_trial
        
        # Sort all results by metric value
        self.best_results.sort(key=lambda x: x['metric_value'], reverse=True)
        
        # Analyze top trials
        top_n = min(10, len(self.best_results))
        top_trials = self.best_results[:top_n]
        
        analysis = {
            'optimization_summary': {
                'total_trials': len(study.trials),
                'best_metric_value': best_trial.value,
                'best_params': best_trial.params,
                'optimization_metric': self.optimization_metric,
            },
            'best_trial_performance': self.best_results[0]['summary'],
            'top_trials': top_trials,
            'parameter_importance': {},
        }
        
        # Parameter importance analysis
        try:
            importance = optuna.importance.get_param_importances(study)
            analysis['parameter_importance'] = importance
        except Exception as e:
            self.logger.warning(f"Could not calculate parameter importance: {e}")
        
        # Save detailed results
        with open(output_dir / 'optimization_results.json', 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        # Save study object
        with open(output_dir / 'optuna_study.pkl', 'wb') as f:
            import pickle
            pickle.dump(study, f)
        
        # Create summary report
        self._create_summary_report(analysis, output_dir)
        
        return analysis
    
    def _create_summary_report(self, analysis: Dict, output_dir: Path) -> None:
        """Create a human-readable summary report."""
        
        report_lines = [
            "# Risk Parameter Optimization Results",
            f"**Optimization Metric:** {self.optimization_metric}",
            f"**Total Trials:** {analysis['optimization_summary']['total_trials']}",
            f"**Best Metric Value:** {analysis['optimization_summary']['best_metric_value']:.4f}",
            "",
            "## Best Parameters:",
        ]
        
        best_params = analysis['optimization_summary']['best_params']
        for param, value in best_params.items():
            report_lines.append(f"- **{param}:** {value}")
        
        report_lines.extend([
            "",
            "## Best Performance:",
        ])
        
        best_perf = analysis['best_trial_performance']
        key_metrics = [
            ('return_pct', 'Total Return', ':.1f%'),
            ('win_rate', 'Win Rate', ':.1%'),
            ('total_trades', 'Total Trades', ':d'),
            ('profit_factor', 'Profit Factor', ':.2f'),
            ('max_drawdown', 'Max Drawdown', ':.1%'),
        ]
        
        for key, label, fmt in key_metrics:
            if key in best_perf:
                if fmt:
                    value_str = f"{best_perf[key]:{fmt}}"
                else:
                    value_str = str(best_perf[key])
                report_lines.append(f"- **{label}:** {value_str}")
        
        if analysis['parameter_importance']:
            report_lines.extend([
                "",
                "## Parameter Importance:",
            ])
            for param, importance in sorted(
                analysis['parameter_importance'].items(), 
                key=lambda x: x[1], 
                reverse=True
            ):
                report_lines.append(f"- **{param}:** {importance:.3f}")
        
        # Save report
        with open(output_dir / 'optimization_summary.md', 'w') as f:
            f.write('\n'.join(report_lines))
        
        self.logger.info(f"Summary report saved to {output_dir / 'optimization_summary.md'}")


def main():
    """Main optimization runner."""
    parser = argparse.ArgumentParser(description="Optimize risk parameters using Optuna")
    parser.add_argument('--decision-table', type=Path, required=True, help="Path to decision table CSV")
    parser.add_argument('--policy', type=Path, required=True, help="Path to trained policy")
    parser.add_argument('--meta', type=Path, required=True, help="Path to policy metadata JSON")
    parser.add_argument('--output-dir', type=Path, default=Path('optimization_results'), help="Output directory")
    parser.add_argument('--n-trials', type=int, default=200, help="Number of optimization trials")
    parser.add_argument('--initial-capital', type=float, default=10_000.0, help="Initial capital")
    parser.add_argument('--metric', type=str, default='composite_score', 
                       choices=['sharpe_ratio', 'return_drawdown_ratio', 'profit_factor', 
                               'win_rate_adjusted_return', 'composite_score'],
                       help="Optimization metric")
    parser.add_argument('--study-name', type=str, help="Optuna study name")
    parser.add_argument('--storage', type=str, help="Optuna storage URL (for persistence)")
    
    args = parser.parse_args()
    
    # Create optimizer
    optimizer = RiskOptimizer(
        decision_table_path=args.decision_table,
        policy_path=args.policy,
        meta_path=args.meta,
        initial_capital=args.initial_capital,
        optimization_metric=args.metric
    )
    
    # Run optimization
    study = optimizer.optimize(
        n_trials=args.n_trials,
        study_name=args.study_name,
        storage=args.storage
    )
    
    # Analyze and save results
    analysis = optimizer.analyze_results(study, args.output_dir)
    
    # Print summary
    print("\n" + "="*50)
    print("🎯 OPTIMIZATION COMPLETE!")
    print("="*50)
    print(f"Best {args.metric}: {study.best_trial.value:.4f}")
    print(f"Best parameters: {study.best_trial.params}")
    print(f"Results saved to: {args.output_dir}")
    print("\n📊 Best Performance:")
    best_perf = analysis['best_trial_performance']
    print(f"  Return: {best_perf['return_pct']:.1f}%")
    print(f"  Win Rate: {best_perf.get('win_rate', 0):.1%}")
    print(f"  Total Trades: {best_perf.get('total_trades', 0)}")
    print(f"  Profit Factor: {best_perf.get('profit_factor', 0):.2f}")
    print(f"  Max Drawdown: {best_perf.get('max_drawdown', 0):.1%}")


if __name__ == '__main__':
    main()