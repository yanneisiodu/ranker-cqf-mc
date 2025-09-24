#!/usr/bin/env python3
"""
Win-Rate Aware Optuna Risk Parameter Optimization

This script creates a multi-objective optimization that balances:
1. High win rates (60%+) for psychological comfort
2. Strong absolute returns for profitability  
3. Controlled drawdowns for risk management
4. High profit factor for edge confirmation

The composite score heavily weights win rate to ensure real-world trading viability.
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


class WinRateOptimizer:
    """Win-rate aware Optuna optimizer for trading systems."""
    
    def __init__(
        self,
        decision_table_path: Path,
        policy_path: Path,
        meta_path: Path,
        initial_capital: float = 10_000.0,
        min_win_rate: float = 0.60,  # Minimum acceptable win rate
        win_rate_weight: float = 0.4,  # Weight for win rate in composite score
        return_weight: float = 0.3,    # Weight for returns
        drawdown_weight: float = 0.2,  # Weight for drawdown control
        profit_factor_weight: float = 0.1  # Weight for profit factor
    ):
        self.decision_table_path = decision_table_path
        self.policy_path = policy_path
        self.meta_path = meta_path
        self.initial_capital = initial_capital
        self.min_win_rate = min_win_rate
        
        # Composite score weights
        self.win_rate_weight = win_rate_weight
        self.return_weight = return_weight
        self.drawdown_weight = drawdown_weight
        self.profit_factor_weight = profit_factor_weight
        
        # Load data once for all trials
        self.logger = logging.getLogger("winrate_optimizer")
        self.logger.info("Loading decision table and policy...")
        
        self.df = pd.read_csv(decision_table_path)
        self.meta = _load_meta(meta_path)
        self.states = _standardise_states(
            self.df, self.meta['state_columns'], self.meta['scaler_mean'], self.meta['scaler_scale']
        )
        self.algo = _load_policy_robust(policy_path, self.meta)
        self.predicted_actions = self.algo.predict(self.states)
        
        self.logger.info(f"Loaded {len(self.df)} decision points for optimization")
        self.logger.info(f"Target win rate: {min_win_rate:.1%}")
        self.logger.info(f"Score weights - Win Rate: {win_rate_weight}, Returns: {return_weight}, Drawdown: {drawdown_weight}, PF: {profit_factor_weight}")
        
        # Track best results
        self.best_results = []
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Win-rate aware objective function.
        
        Returns composite score balancing win rate, returns, drawdown, and profit factor.
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
            
            # Calculate win-rate aware composite score
            composite_score = self._calculate_winrate_composite_score(results, summary, params)
            
            # Store results for analysis
            trial_result = {
                'trial_number': trial.number,
                'params': params.copy(),
                'summary': summary.copy(),
                'composite_score': composite_score,
                'win_rate': summary.get('win_rate', 0.0),
                'return_pct': summary.get('return_pct', 0.0),
                'max_drawdown': summary.get('max_drawdown', 1.0),
                'profit_factor': summary.get('profit_factor', 1.0)
            }
            self.best_results.append(trial_result)
            
            # Log trial results with focus on win rate
            self.logger.info(
                f"Trial {trial.number}: Win Rate: {summary.get('win_rate', 0):.1%}, "
                f"Return: {summary.get('return_pct', 0):.1f}%, "
                f"Drawdown: {summary.get('max_drawdown', 0):.1%}, "
                f"PF: {summary.get('profit_factor', 1):.2f}x, "
                f"Score: {composite_score:.4f}"
            )
            
            return composite_score
            
        except Exception as e:
            self.logger.error(f"Trial {trial.number} failed: {e}")
            return -1000.0  # Very low score for failed trials
    
    def _sample_parameters(self, trial: optuna.Trial) -> Dict[str, float]:
        """Sample risk parameters with ranges focused on win rate optimization."""
        
        return {
            # Core trading parameters
            'base_contracts': trial.suggest_int('base_contracts', 3, 8),  # Smaller positions for higher win rates
            'base_notional': trial.suggest_float('base_notional', 800.0, 1500.0, step=100.0),
            
            # Risk controls (enabled by default for win rate focus)
            'enable_risk_controls': True,
            'max_notional_pct': trial.suggest_float('max_notional_pct', 0.15, 0.30, step=0.05),
            
            # Transaction costs
            'commission_per_side': trial.suggest_float('commission_per_side', 0.50, 1.00, step=0.05),
            'exchange_fee_per_side': trial.suggest_float('exchange_fee_per_side', 0.03, 0.10, step=0.01),
            'slippage_min': trial.suggest_float('slippage_min', 0.01, 0.05, step=0.005),
            'slippage_pct': trial.suggest_float('slippage_pct', 0.10, 0.35, step=0.05),
            
            # Volatility regime scaling (conservative for higher win rates)
            '_vol_emergency_mult': trial.suggest_float('vol_emergency_mult', 0.2, 0.6, step=0.05),
            '_vol_high_mult': trial.suggest_float('vol_high_mult', 0.4, 0.8, step=0.05),
            '_vol_mod_mult': trial.suggest_float('vol_mod_mult', 0.6, 1.0, step=0.05),
            
            # Drawdown protection (aggressive for win rate preservation)
            '_drawdown_scale_factor': trial.suggest_float('drawdown_scale_factor', 1.0, 3.0, step=0.1),
            '_min_drawdown_scale': trial.suggest_float('min_drawdown_scale', 0.1, 0.5, step=0.05),
            
            # Equity scaling (conservative)
            '_enable_equity_scaling': trial.suggest_categorical('enable_equity_scaling', [True, False]),
            '_max_equity_mult': trial.suggest_float('max_equity_mult', 1.2, 2.0, step=0.1),
            '_equity_threshold': trial.suggest_float('equity_threshold', 1.3, 2.0, step=0.1),
        }
    
    def _calculate_winrate_composite_score(
        self, 
        results: pd.DataFrame, 
        summary: Dict, 
        params: Dict
    ) -> float:
        """
        Calculate win-rate focused composite score.
        
        Heavily weights win rate while considering returns, drawdown, and profit factor.
        """
        # Extract key metrics
        win_rate = summary.get('win_rate', 0.0)
        return_pct = summary.get('return_pct', 0.0)
        max_drawdown = summary.get('max_drawdown', 1.0)  # Default to 100% if missing
        profit_factor = summary.get('profit_factor', 1.0)
        total_trades = summary.get('total_trades', 0)
        
        # Minimum trade requirement
        if total_trades < 10:
            return -100.0
        
        # Hard constraint: win rate must be above minimum threshold
        if win_rate < self.min_win_rate:
            penalty = (self.min_win_rate - win_rate) * 1000  # Heavy penalty
            return -penalty
        
        # Normalize components for scoring
        
        # 1. Win Rate Component (0-100, target 60%+)
        win_rate_score = min(100.0, (win_rate / 0.80) * 100)  # Scale so 80% win rate = 100 points
        
        # 2. Return Component (0-100, target 1000%+ return)
        return_score = min(100.0, (return_pct / 2000.0) * 100)  # Scale so 2000% return = 100 points
        
        # 3. Drawdown Component (0-100, target <10% drawdown)
        drawdown_score = max(0.0, 100.0 - (max_drawdown * 1000))  # Penalize drawdown heavily
        
        # 4. Profit Factor Component (0-100, target 2.0+)
        profit_factor_score = min(100.0, ((profit_factor - 1.0) / 2.0) * 100)  # Scale so PF=3.0 gives 100 points
        
        # Calculate weighted composite score
        composite_score = (
            self.win_rate_weight * win_rate_score +
            self.return_weight * return_score +
            self.drawdown_weight * drawdown_score +
            self.profit_factor_weight * profit_factor_score
        )
        
        # Bonus for exceptional win rates (70%+)
        if win_rate >= 0.70:
            composite_score *= 1.1  # 10% bonus
        
        # Bonus for very high returns with good win rate
        if return_pct > 1500.0 and win_rate >= 0.65:
            composite_score *= 1.05  # 5% bonus
        
        return composite_score
    
    def optimize(
        self, 
        n_trials: int = 100, 
        timeout: Optional[float] = None,
        study_name: str = "winrate_risk_optimization"
    ) -> Tuple[optuna.Study, Dict]:
        """Run optimization with win-rate focus."""
        
        self.logger.info(f"Starting win-rate aware optimization with {n_trials} trials")
        
        # Create study with TPE sampler and Hyperband pruner
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42),
            pruner=HyperbandPruner(),
            study_name=study_name
        )
        
        # Run optimization
        study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )
        
        # Analyze results
        best_trial = study.best_trial
        best_params = best_trial.params
        
        # Find the corresponding detailed results
        best_detailed = None
        for result in self.best_results:
            if result['trial_number'] == best_trial.number:
                best_detailed = result
                break
        
        self.logger.info("🎯 Win-Rate Optimization Complete!")
        self.logger.info(f"Best trial: #{best_trial.number}")
        self.logger.info(f"Best composite score: {best_trial.value:.4f}")
        
        if best_detailed:
            self.logger.info(f"📊 Best Performance:")
            self.logger.info(f"  Win Rate: {best_detailed['win_rate']:.1%} (target: {self.min_win_rate:.1%})")
            self.logger.info(f"  Total Return: {best_detailed['return_pct']:.1f}%")
            self.logger.info(f"  Max Drawdown: {best_detailed['max_drawdown']:.1%}")
            self.logger.info(f"  Profit Factor: {best_detailed['profit_factor']:.2f}x")
        
        # Compile comprehensive results
        optimization_summary = {
            'study_name': study_name,
            'n_trials': n_trials,
            'optimization_focus': 'win_rate_aware',
            'min_win_rate_target': self.min_win_rate,
            'score_weights': {
                'win_rate': self.win_rate_weight,
                'returns': self.return_weight,
                'drawdown': self.drawdown_weight,
                'profit_factor': self.profit_factor_weight
            },
            'best_trial_number': best_trial.number,
            'best_composite_score': best_trial.value,
            'best_params': best_params,
            'best_performance': best_detailed
        }
        
        return study, optimization_summary


def main():
    """Run win-rate aware risk optimization."""
    parser = argparse.ArgumentParser(description="Win-rate aware risk parameter optimization")
    parser.add_argument('--decision-table', type=Path, required=True)
    parser.add_argument('--policy', type=Path, required=True)  
    parser.add_argument('--meta', type=Path, required=True)
    parser.add_argument('--trials', type=int, default=50, help="Number of optimization trials")
    parser.add_argument('--min-win-rate', type=float, default=0.60, help="Minimum acceptable win rate")
    parser.add_argument('--outdir', type=Path, default=Path('results/winrate_optimization'))
    parser.add_argument('--study-name', default='winrate_risk_opt', help="Optuna study name")
    
    args = parser.parse_args()
    
    # Create output directory
    args.outdir.mkdir(parents=True, exist_ok=True)
    
    # Create optimizer
    optimizer = WinRateOptimizer(
        decision_table_path=args.decision_table,
        policy_path=args.policy,
        meta_path=args.meta,
        min_win_rate=args.min_win_rate
    )
    
    # Run optimization
    study, summary = optimizer.optimize(
        n_trials=args.trials,
        study_name=args.study_name
    )
    
    # Save results
    results_file = args.outdir / 'winrate_optimization_results.json'
    with open(results_file, 'w') as f:
        # Convert any non-serializable objects
        clean_summary = json.loads(json.dumps(summary, default=str))
        json.dump({
            'optimization_summary': clean_summary,
            'all_trials': optimizer.best_results
        }, f, indent=2, default=str)
    
    # Save Optuna study
    study_file = args.outdir / 'winrate_optuna_study.pkl'
    optuna.save_study(study, str(study_file))
    
    # Run final simulation with best parameters
    best_params = summary['best_params']
    
    from optimized_walkforward_simulation import simulate_optimized_walkforward
    
    final_results, final_summary = simulate_optimized_walkforward(
        optimizer.df.copy(),
        optimizer.predicted_actions,
        optimizer.meta['action_map'],
        initial_capital=optimizer.initial_capital,
        **best_params
    )
    
    # Save final backtest results
    final_dir = args.outdir / 'final_winrate_backtest'
    final_dir.mkdir(exist_ok=True)
    
    final_results.to_csv(final_dir / 'winrate_optimized_trades.csv', index=False)
    with open(final_dir / 'winrate_optimized_summary.json', 'w') as f:
        json.dump(final_summary, f, indent=2, default=str)
    
    print(f"\n✅ Win-rate optimization complete!")
    print(f"📊 Results saved to: {args.outdir}")
    print(f"🎯 Final backtest in: {final_dir}")
    print(f"📈 Best Win Rate: {summary['best_performance']['win_rate']:.1%}")
    print(f"💰 Best Return: {summary['best_performance']['return_pct']:.1f}%")


if __name__ == '__main__':
    main()