#!/usr/bin/env python3
"""
Simple Win-Rate Optimization using Existing Working Components

This uses the working hybrid simulation as a base and focuses on win-rate improvement
through parameter optimization without policy loading complications.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
import subprocess
import sys

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO)


class SimpleWinRateOptimizer:
    """Simple win-rate optimizer using subprocess calls to avoid loading issues."""
    
    def __init__(
        self,
        decision_table_path: Path,
        policy_path: Path,
        meta_path: Path,
        initial_capital: float = 10_000.0,
        target_win_rate: float = 0.45,
        win_rate_weight: float = 0.4,
        return_weight: float = 0.3,
        drawdown_weight: float = 0.2,
        profit_factor_weight: float = 0.1
    ):
        self.decision_table_path = decision_table_path
        self.policy_path = policy_path
        self.meta_path = meta_path
        self.initial_capital = initial_capital
        self.target_win_rate = target_win_rate
        
        # Score weights
        self.win_rate_weight = win_rate_weight
        self.return_weight = return_weight
        self.drawdown_weight = drawdown_weight
        self.profit_factor_weight = profit_factor_weight
        
        self.logger = logging.getLogger("simple_winrate")
        self.logger.info(f"Target win rate: {target_win_rate:.1%}")
        
        # Track best results
        self.best_results = []
        
        # Create working directory for temp results
        self.temp_dir = Path("temp_optimization")
        self.temp_dir.mkdir(exist_ok=True)
    
    def objective(self, trial: optuna.Trial) -> float:
        """Run trial using subprocess to avoid loading issues."""
        try:
            # Sample parameters
            params = self._sample_parameters(trial)
            
            # Create a temporary config file for this trial
            temp_config = self.temp_dir / f"trial_{trial.number}_config.json"
            with open(temp_config, 'w') as f:
                json.dump(params, f, indent=2)
            
            # Run simulation using optimized_walkforward_simulation.py
            temp_results = self.temp_dir / f"trial_{trial.number}_results.csv"
            temp_summary = self.temp_dir / f"trial_{trial.number}_summary.json"
            
            cmd = [
                sys.executable, "Training/optimized_walkforward_simulation.py",
                "--decision-table", str(self.decision_table_path),
                "--policy", str(self.policy_path),
                "--meta", str(self.meta_path),
                "--params-file", str(temp_config),
                "--outdir", str(self.temp_dir / f"trial_{trial.number}")
            ]
            
            # Run with timeout
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                if result.returncode != 0:
                    self.logger.error(f"Trial {trial.number} subprocess failed: {result.stderr}")
                    return 0.0
            except subprocess.TimeoutExpired:
                self.logger.error(f"Trial {trial.number} timed out")
                return 0.0
            
            # Load results
            summary_file = self.temp_dir / f"trial_{trial.number}" / "optimized_walkforward_summary.json"
            if not summary_file.exists():
                self.logger.error(f"Trial {trial.number}: Summary file not found")
                return 0.0
            
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            
            # Calculate composite score
            composite_score = self._calculate_composite_score(summary)
            
            # Store results
            trial_result = {
                'trial_number': trial.number,
                'params': params.copy(),
                'summary': summary.copy(),
                'composite_score': composite_score
            }
            self.best_results.append(trial_result)
            
            # Log results
            win_rate = summary.get('win_rate', 0.0)
            return_pct = summary.get('return_pct', 0.0)
            max_drawdown = summary.get('max_drawdown', 0.0)
            profit_factor = summary.get('profit_factor', 1.0)
            
            self.logger.info(
                f"Trial {trial.number}: Win Rate: {win_rate:.1%}, "
                f"Return: {return_pct:.1f}%, "
                f"Drawdown: {max_drawdown:.1%}, "
                f"PF: {profit_factor:.2f}x, "
                f"Score: {composite_score:.2f}"
            )
            
            # Cleanup temp files
            temp_config.unlink(missing_ok=True)
            
            return composite_score
            
        except Exception as e:
            self.logger.error(f"Trial {trial.number} failed: {e}")
            return 0.0
    
    def _sample_parameters(self, trial: optuna.Trial) -> Dict:
        """Sample parameters focused on win rate improvement."""
        return {
            # Core parameters - smaller positions for higher win rates
            'base_contracts': trial.suggest_int('base_contracts', 2, 5),
            'base_notional': trial.suggest_float('base_notional', 600.0, 1000.0, step=50.0),
            
            # Always enable risk controls for consistency
            'enable_risk_controls': True,
            'max_notional_pct': trial.suggest_float('max_notional_pct', 0.15, 0.25, step=0.02),
            
            # Transaction costs - realistic ranges
            'commission_per_side': trial.suggest_float('commission_per_side', 0.55, 0.75, step=0.05),
            'exchange_fee_per_side': trial.suggest_float('exchange_fee_per_side', 0.03, 0.08, step=0.01),
            'slippage_min': trial.suggest_float('slippage_min', 0.015, 0.035, step=0.005),
            'slippage_pct': trial.suggest_float('slippage_pct', 0.15, 0.25, step=0.025),
            
            # Conservative volatility scaling for higher win rates
            '_vol_emergency_mult': trial.suggest_float('_vol_emergency_mult', 0.2, 0.4, step=0.05),
            '_vol_high_mult': trial.suggest_float('_vol_high_mult', 0.4, 0.6, step=0.05),
            '_vol_mod_mult': trial.suggest_float('_vol_mod_mult', 0.7, 0.9, step=0.05),
            
            # Moderate drawdown protection
            '_drawdown_scale_factor': trial.suggest_float('_drawdown_scale_factor', 1.2, 2.0, step=0.1),
            '_min_drawdown_scale': trial.suggest_float('_min_drawdown_scale', 0.25, 0.4, step=0.05),
            
            # Balanced equity scaling
            '_enable_equity_scaling': trial.suggest_categorical('_enable_equity_scaling', [True, False]),
            '_max_equity_mult': trial.suggest_float('_max_equity_mult', 1.2, 1.6, step=0.1),
            '_equity_threshold': trial.suggest_float('_equity_threshold', 1.3, 1.7, step=0.1),
        }
    
    def _calculate_composite_score(self, summary: Dict) -> float:
        """Calculate win-rate focused composite score."""
        win_rate = summary.get('win_rate', 0.0)
        return_pct = summary.get('return_pct', 0.0)
        max_drawdown = summary.get('max_drawdown', 0.0)
        profit_factor = summary.get('profit_factor', 1.0)
        total_trades = summary.get('total_trades', 0)
        
        # Minimum trades filter
        if total_trades < 10:
            return 0.0
        
        # Normalize components (0-100 scale)
        
        # 1. Win Rate - improvement over baseline ~38%
        baseline_win_rate = 0.38
        win_rate_improvement = win_rate - baseline_win_rate
        win_rate_score = max(0.0, 50.0 + (win_rate_improvement / 0.15) * 50.0)  # +15% improvement = 100 points
        
        # 2. Returns - target reasonable returns
        return_score = min(100.0, (return_pct / 1000.0) * 100)  # 1000% = 100 points
        
        # 3. Drawdown - heavily penalize high drawdowns
        drawdown_score = max(0.0, 100.0 - (max_drawdown * 1000))  # Each 1% drawdown = -10 points
        
        # 4. Profit Factor - target healthy edge
        profit_factor_score = min(100.0, ((profit_factor - 1.0) / 1.0) * 100)  # PF=2.0 gives 100 points
        
        # Weighted composite
        composite_score = (
            self.win_rate_weight * win_rate_score +
            self.return_weight * return_score +
            self.drawdown_weight * drawdown_score +
            self.profit_factor_weight * profit_factor_score
        )
        
        # Bonus for exceeding target win rate
        if win_rate >= self.target_win_rate:
            bonus = 1.0 + min(0.3, (win_rate - self.target_win_rate) * 3.0)  # Up to 30% bonus
            composite_score *= bonus
        
        # Bonus for balanced performance
        if win_rate >= 0.42 and return_pct >= 600.0:
            composite_score *= 1.1  # 10% balance bonus
        
        return composite_score
    
    def optimize(self, n_trials: int = 40, study_name: str = "simple_winrate_opt") -> Tuple[optuna.Study, Dict]:
        """Run optimization."""
        self.logger.info(f"Starting simple win-rate optimization with {n_trials} trials")
        
        # Create study
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42),
            pruner=HyperbandPruner(),
            study_name=study_name
        )
        
        # Run optimization
        study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)
        
        # Get best results
        best_trial = study.best_trial
        best_detailed = None
        
        for result in self.best_results:
            if result['trial_number'] == best_trial.number:
                best_detailed = result
                break
        
        self.logger.info("🎯 Simple Win-Rate Optimization Complete!")
        self.logger.info(f"Best trial: #{best_trial.number}")
        self.logger.info(f"Best composite score: {best_trial.value:.2f}")
        
        if best_detailed:
            summary = best_detailed['summary']
            self.logger.info(f"📊 Best Performance:")
            self.logger.info(f"  Win Rate: {summary.get('win_rate', 0):.1%} (target: {self.target_win_rate:.1%})")
            self.logger.info(f"  Total Return: {summary.get('return_pct', 0):.1f}%")
            self.logger.info(f"  Max Drawdown: {summary.get('max_drawdown', 0):.1%}")
            self.logger.info(f"  Profit Factor: {summary.get('profit_factor', 1):.2f}x")
        
        # Compile results summary
        optimization_summary = {
            'study_name': study_name,
            'n_trials': n_trials,
            'optimization_focus': 'simple_winrate_improvement',
            'target_win_rate': self.target_win_rate,
            'baseline_win_rate': 0.38,
            'score_weights': {
                'win_rate': self.win_rate_weight,
                'returns': self.return_weight,
                'drawdown': self.drawdown_weight,
                'profit_factor': self.profit_factor_weight
            },
            'best_trial_number': best_trial.number,
            'best_composite_score': best_trial.value,
            'best_params': best_trial.params,
            'best_performance': best_detailed['summary'] if best_detailed else None
        }
        
        return study, optimization_summary
    
    def cleanup(self):
        """Clean up temporary files."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)


def main():
    """Run simple win-rate optimization."""
    parser = argparse.ArgumentParser(description="Simple win-rate optimization")
    parser.add_argument('--decision-table', type=Path, required=True)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--meta', type=Path, required=True)
    parser.add_argument('--trials', type=int, default=40, help="Number of trials")
    parser.add_argument('--target-win-rate', type=float, default=0.45, help="Target win rate")
    parser.add_argument('--outdir', type=Path, default=Path('results/simple_winrate_optimization'))
    parser.add_argument('--study-name', default='simple_winrate_opt')
    
    args = parser.parse_args()
    
    # Create output directory
    args.outdir.mkdir(parents=True, exist_ok=True)
    
    # Create optimizer
    optimizer = SimpleWinRateOptimizer(
        decision_table_path=args.decision_table,
        policy_path=args.policy,
        meta_path=args.meta,
        target_win_rate=args.target_win_rate
    )
    
    try:
        # Run optimization
        study, summary = optimizer.optimize(n_trials=args.trials, study_name=args.study_name)
        
        # Save results
        results_file = args.outdir / 'simple_winrate_results.json'
        with open(results_file, 'w') as f:
            clean_summary = json.loads(json.dumps(summary, default=str))
            json.dump({
                'optimization_summary': clean_summary,
                'all_trials': optimizer.best_results
            }, f, indent=2, default=str)
        
        # Save Optuna study
        study_file = args.outdir / 'simple_winrate_study.pkl'
        optuna.save_study(study, str(study_file))
        
        # Run final simulation with best parameters
        if summary['best_performance']:
            best_params = summary['best_params']
            
            # Create final results directory  
            final_dir = args.outdir / 'final_simple_winrate_backtest'
            final_dir.mkdir(exist_ok=True)
            
            # Create config for final run
            final_config = final_dir / 'final_params.json'
            with open(final_config, 'w') as f:
                json.dump(best_params, f, indent=2)
            
            # Run final simulation
            cmd = [
                sys.executable, "Training/optimized_walkforward_simulation.py",
                "--decision-table", str(args.decision_table),
                "--policy", str(args.policy),
                "--meta", str(args.meta),
                "--params-file", str(final_config),
                "--outdir", str(final_dir)
            ]
            
            subprocess.run(cmd, check=True)
            
            print(f"\n✅ Simple win-rate optimization complete!")
            print(f"📊 Results saved to: {args.outdir}")
            print(f"🎯 Final backtest in: {final_dir}")
            print(f"📈 Best Win Rate: {summary['best_performance']['win_rate']:.1%}")
            print(f"💰 Best Return: {summary['best_performance']['return_pct']:.1f}%")
            print(f"📉 Max Drawdown: {summary['best_performance']['max_drawdown']:.1%}")
            print(f"⚡ Profit Factor: {summary['best_performance']['profit_factor']:.2f}x")
        
    finally:
        # Clean up
        optimizer.cleanup()


if __name__ == '__main__':
    main()