#!/usr/bin/env python3
"""
Hybrid Win-Rate Optimization

Uses the working hybrid simulation as a base and optimizes parameters to improve
win rates from the baseline 37.5% to a target of 42-45% while maintaining
strong returns and controlling drawdowns.

This approach uses subprocess calls to the working hybrid_walkforward_simulation.py
to avoid the policy loading complications.
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
import tempfile
import shutil

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO)


class HybridWinRateOptimizer:
    """Win-rate optimizer using the working hybrid simulation."""
    
    def __init__(
        self,
        decision_table_path: Path,
        policy_path: Path,
        meta_path: Path,
        initial_capital: float = 10_000.0,
        target_win_rate: float = 0.42,      # Realistic target from 37.5% baseline
        win_rate_weight: float = 0.5,       # Heavy emphasis on win rate
        return_weight: float = 0.3,         # Still care about returns
        drawdown_weight: float = 0.15,      # Risk management
        profit_factor_weight: float = 0.05  # Edge confirmation
    ):
        self.decision_table_path = decision_table_path
        self.policy_path = policy_path
        self.meta_path = meta_path
        self.initial_capital = initial_capital
        self.target_win_rate = target_win_rate
        
        # Composite score weights
        self.win_rate_weight = win_rate_weight
        self.return_weight = return_weight
        self.drawdown_weight = drawdown_weight
        self.profit_factor_weight = profit_factor_weight
        
        self.logger = logging.getLogger("hybrid_winrate")
        self.logger.info(f"Baseline win rate: 37.5% -> Target: {target_win_rate:.1%}")
        self.logger.info(f"Score weights - Win Rate: {win_rate_weight}, Returns: {return_weight}")
        
        # Track results
        self.best_results = []
        
        # Create temp directory for results
        self.temp_dir = Path(tempfile.mkdtemp(prefix="winrate_opt_"))
        self.logger.info(f"Using temp directory: {self.temp_dir}")
    
    def objective(self, trial: optuna.Trial) -> float:
        """Optimize using subprocess calls to hybrid simulation."""
        try:
            # Sample parameters focused on win rate improvement
            params = self._sample_winrate_parameters(trial)
            
            # Prepare directories
            trial_dir = self.temp_dir / f"trial_{trial.number}"
            trial_dir.mkdir(exist_ok=True)
            
            # Build command for hybrid simulation
            cmd = [
                sys.executable, "Training/hybrid_walkforward_simulation.py",
                "--decision-table", str(self.decision_table_path),
                "--policy", str(self.policy_path), 
                "--meta", str(self.meta_path),
                "--outdir", str(trial_dir)
            ]
            
            # Add parameter overrides
            for param_name, param_value in params.items():
                if param_name.startswith('base_'):
                    cmd.extend([f"--{param_name.replace('_', '-')}", str(param_value)])
            
            # Run simulation with timeout
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=45, cwd=Path.cwd())
                if result.returncode != 0:
                    self.logger.warning(f"Trial {trial.number} failed: {result.stderr[:200]}")
                    return 0.0
            except subprocess.TimeoutExpired:
                self.logger.warning(f"Trial {trial.number} timed out")
                return 0.0
            
            # Load results
            summary_file = trial_dir / "hybrid_walkforward_summary.json"
            if not summary_file.exists():
                self.logger.warning(f"Trial {trial.number}: No summary file found")
                return 0.0
            
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            
            # Calculate win-rate focused composite score
            composite_score = self._calculate_winrate_score(summary, params)
            
            # Store results
            trial_result = {
                'trial_number': trial.number,
                'params': params.copy(),
                'summary': summary.copy(),
                'composite_score': composite_score
            }
            self.best_results.append(trial_result)
            
            # Log detailed results
            win_rate = summary.get('win_rate', 0.0)
            return_pct = summary.get('return_pct', 0.0)
            max_drawdown = summary.get('max_drawdown', 0.0)
            profit_factor = summary.get('profit_factor', 1.0)
            total_trades = summary.get('total_trades', 0)
            
            improvement = (win_rate - 0.375) * 100  # Improvement over 37.5% baseline
            
            self.logger.info(
                f"Trial {trial.number}: "
                f"Win Rate: {win_rate:.1%} (+{improvement:+.1f}pp), "
                f"Return: {return_pct:.1f}%, "
                f"Drawdown: {max_drawdown:.1%}, "
                f"PF: {profit_factor:.2f}x, "
                f"Trades: {total_trades}, "
                f"Score: {composite_score:.2f}"
            )
            
            return composite_score
            
        except Exception as e:
            self.logger.error(f"Trial {trial.number} error: {e}")
            return 0.0
    
    def _sample_winrate_parameters(self, trial: optuna.Trial) -> Dict:
        """Sample parameters specifically designed to improve win rates."""
        
        # These parameter ranges are designed based on trading psychology:
        # - Smaller position sizes tend to improve win rates
        # - More conservative risk controls help preserve wins
        # - Higher transaction costs force better trade selection
        
        return {
            # Position sizing - smaller sizes typically improve win rates
            'base_contracts': trial.suggest_int('base_contracts', 2, 4),  # Smaller than default 5
            
            # Notional sizing - conservative  
            'base_notional': trial.suggest_float('base_notional', 600.0, 900.0, step=50.0),
            
            # Risk management - always enabled for consistency
            'enable_risk_controls': True,
            'max_notional_pct': trial.suggest_float('max_notional_pct', 0.12, 0.20, step=0.02),
            
            # Transaction costs - slightly higher costs can improve trade selection
            'commission_per_side': trial.suggest_float('commission_per_side', 0.65, 0.85, step=0.05),
            'exchange_fee_per_side': trial.suggest_float('exchange_fee_per_side', 0.05, 0.10, step=0.01),
            'slippage_min': trial.suggest_float('slippage_min', 0.02, 0.04, step=0.005),
            'slippage_pct': trial.suggest_float('slippage_pct', 0.20, 0.30, step=0.025),
        }
    
    def _calculate_winrate_score(self, summary: Dict, params: Dict) -> float:
        """Calculate composite score heavily weighted toward win rate improvement."""
        
        win_rate = summary.get('win_rate', 0.0)
        return_pct = summary.get('return_pct', 0.0)
        max_drawdown = summary.get('max_drawdown', 0.0)
        profit_factor = summary.get('profit_factor', 1.0)
        total_trades = summary.get('total_trades', 0)
        
        # Must have minimum trades to be valid
        if total_trades < 20:
            return 0.0
        
        # Must have positive returns
        if return_pct <= 0:
            return 0.0
        
        # Component scoring (0-100 scale)
        
        # 1. Win Rate Score - This is our primary focus
        baseline_win_rate = 0.375  # Current baseline from testing
        win_rate_improvement = win_rate - baseline_win_rate
        
        # Heavy reward for any improvement, exponential bonus for reaching targets
        if win_rate <= baseline_win_rate:
            win_rate_score = 0.0  # No points for not improving
        elif win_rate < 0.40:
            win_rate_score = (win_rate_improvement / 0.025) * 30  # Up to 30 points for getting to 40%
        elif win_rate < 0.45:
            win_rate_score = 30 + ((win_rate - 0.40) / 0.05) * 40  # Up to 70 points total for getting to 45%
        else:
            win_rate_score = 70 + min(30, (win_rate - 0.45) * 600)  # Exponential bonus above 45%
        
        # 2. Return Score - We still need profitability
        return_score = min(100.0, return_pct / 10.0)  # 1000% return = 100 points
        
        # 3. Drawdown Score - Penalize high drawdowns
        if max_drawdown <= 0.05:  # Less than 5% drawdown
            drawdown_score = 100.0
        elif max_drawdown <= 0.10:  # Less than 10% drawdown  
            drawdown_score = 80.0
        elif max_drawdown <= 0.20:  # Less than 20% drawdown
            drawdown_score = 60.0
        else:
            drawdown_score = max(0.0, 60.0 - (max_drawdown - 0.20) * 1000)
        
        # 4. Profit Factor Score - Confirm we have an edge
        if profit_factor >= 2.5:
            pf_score = 100.0
        elif profit_factor >= 2.0:
            pf_score = 80.0
        elif profit_factor >= 1.5:
            pf_score = 60.0
        else:
            pf_score = max(0.0, (profit_factor - 1.0) * 120)  # Linear from 1.0
        
        # Calculate weighted composite score
        composite_score = (
            self.win_rate_weight * win_rate_score +
            self.return_weight * return_score +
            self.drawdown_weight * drawdown_score +
            self.profit_factor_weight * pf_score
        )
        
        # Special bonuses for exceptional performance
        
        # Big bonus for hitting our target win rate
        if win_rate >= self.target_win_rate:
            target_bonus = 1.0 + min(0.5, (win_rate - self.target_win_rate) * 5.0)  # Up to 50% bonus
            composite_score *= target_bonus
        
        # Bonus for balanced excellence (high win rate AND high returns)
        if win_rate >= 0.40 and return_pct >= 800.0:
            composite_score *= 1.2  # 20% excellence bonus
        
        # Extra bonus for beating 45% win rate threshold (elite performance)
        if win_rate >= 0.45:
            composite_score *= 1.3  # 30% elite bonus
        
        return composite_score
    
    def optimize(
        self, 
        n_trials: int = 35, 
        study_name: str = "hybrid_winrate_optimization"
    ) -> Tuple[optuna.Study, Dict]:
        """Run win-rate focused optimization."""
        
        self.logger.info(f"🎯 Starting hybrid win-rate optimization with {n_trials} trials")
        self.logger.info(f"Focus: Improve win rate from 37.5% baseline to {self.target_win_rate:.1%}")
        
        # Create Optuna study
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42),
            pruner=HyperbandPruner(min_resource=5),
            study_name=study_name
        )
        
        # Run optimization
        study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)
        
        # Find best results
        best_trial = study.best_trial
        best_detailed = None
        
        for result in self.best_results:
            if result['trial_number'] == best_trial.number:
                best_detailed = result
                break
        
        # Log final results
        self.logger.info("🏆 Hybrid Win-Rate Optimization Complete!")
        self.logger.info(f"Best trial: #{best_trial.number}")
        self.logger.info(f"Best composite score: {best_trial.value:.2f}")
        
        if best_detailed:
            summary = best_detailed['summary']
            baseline_win_rate = 0.375
            win_improvement = summary.get('win_rate', 0) - baseline_win_rate
            
            self.logger.info("📊 Best Performance Summary:")
            self.logger.info(f"  Win Rate: {summary.get('win_rate', 0):.1%} (+{win_improvement*100:+.1f}pp)")
            self.logger.info(f"  Target Win Rate: {self.target_win_rate:.1%} {'✅' if summary.get('win_rate', 0) >= self.target_win_rate else '❌'}")
            self.logger.info(f"  Total Return: {summary.get('return_pct', 0):.1f}%")
            self.logger.info(f"  Max Drawdown: {summary.get('max_drawdown', 0):.1%}")
            self.logger.info(f"  Profit Factor: {summary.get('profit_factor', 1):.2f}x")
            self.logger.info(f"  Total Trades: {summary.get('total_trades', 0)}")
        
        # Compile optimization summary
        optimization_summary = {
            'study_name': study_name,
            'n_trials': n_trials,
            'optimization_focus': 'hybrid_winrate_improvement',
            'baseline_win_rate': 0.375,
            'target_win_rate': self.target_win_rate,
            'achieved_target': best_detailed['summary'].get('win_rate', 0) >= self.target_win_rate if best_detailed else False,
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
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            self.logger.info("Cleaned up temporary files")


def main():
    """Run hybrid win-rate optimization."""
    parser = argparse.ArgumentParser(description="Hybrid win-rate optimization")
    parser.add_argument('--decision-table', type=Path, required=True)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--meta', type=Path, required=True)
    parser.add_argument('--trials', type=int, default=35, help="Number of optimization trials")
    parser.add_argument('--target-win-rate', type=float, default=0.42, help="Target win rate (0.42 = 42%)")
    parser.add_argument('--outdir', type=Path, default=Path('results/hybrid_winrate_optimization'))
    parser.add_argument('--study-name', default='hybrid_winrate_opt', help="Optuna study name")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.decision_table.exists():
        raise FileNotFoundError(f"Decision table not found: {args.decision_table}")
    if not args.policy.exists():
        raise FileNotFoundError(f"Policy file not found: {args.policy}")
    if not args.meta.exists():
        raise FileNotFoundError(f"Meta file not found: {args.meta}")
    
    # Create output directory
    args.outdir.mkdir(parents=True, exist_ok=True)
    
    # Create and run optimizer
    optimizer = HybridWinRateOptimizer(
        decision_table_path=args.decision_table,
        policy_path=args.policy,
        meta_path=args.meta,
        target_win_rate=args.target_win_rate
    )
    
    try:
        # Run optimization
        study, summary = optimizer.optimize(n_trials=args.trials, study_name=args.study_name)
        
        # Save comprehensive results
        results_file = args.outdir / 'hybrid_winrate_results.json'
        with open(results_file, 'w') as f:
            # Clean summary for JSON serialization
            clean_summary = json.loads(json.dumps(summary, default=str))
            json.dump({
                'optimization_summary': clean_summary,
                'all_trials': optimizer.best_results,
                'study_stats': {
                    'n_trials': len(study.trials),
                    'n_complete_trials': len([t for t in study.trials if t.state.name == 'COMPLETE']),
                    'best_value': study.best_value,
                    'best_params': study.best_params
                }
            }, f, indent=2, default=str)
        
        # Save Optuna study
        study_file = args.outdir / 'hybrid_winrate_study.pkl'
        optuna.save_study(study, str(study_file))
        
        # Run final validation with best parameters
        if summary['best_performance']:
            final_dir = args.outdir / 'final_hybrid_winrate_backtest'
            final_dir.mkdir(exist_ok=True)
            
            # Run final backtest with best parameters
            best_params = summary['best_params']
            cmd = [
                sys.executable, "Training/hybrid_walkforward_simulation.py",
                "--decision-table", str(args.decision_table),
                "--policy", str(args.policy),
                "--meta", str(args.meta),
                "--outdir", str(final_dir)
            ]
            
            # Add best parameters
            for param_name, param_value in best_params.items():
                if param_name.startswith('base_'):
                    cmd.extend([f"--{param_name.replace('_', '-')}", str(param_value)])
            
            subprocess.run(cmd, check=True)
            
            # Copy results with better names
            if (final_dir / "hybrid_walkforward_summary.json").exists():
                shutil.copy2(final_dir / "hybrid_walkforward_summary.json", 
                           final_dir / "winrate_optimized_summary.json")
            if (final_dir / "hybrid_walkforward_trades.csv").exists():
                shutil.copy2(final_dir / "hybrid_walkforward_trades.csv", 
                           final_dir / "winrate_optimized_trades.csv")
            
            # Print final summary
            print(f"\n✅ Hybrid Win-Rate Optimization Complete!")
            print(f"📊 Results saved to: {args.outdir}")
            print(f"🎯 Final backtest in: {final_dir}")
            print(f"\n📈 Performance Summary:")
            print(f"   Baseline Win Rate: 37.5%")
            print(f"   Target Win Rate: {args.target_win_rate:.1%}")
            print(f"   Achieved Win Rate: {summary['best_performance']['win_rate']:.1%}")
            print(f"   Win Rate Improvement: +{(summary['best_performance']['win_rate'] - 0.375)*100:.1f}pp")
            print(f"   Total Return: {summary['best_performance']['return_pct']:.1f}%")
            print(f"   Max Drawdown: {summary['best_performance']['max_drawdown']:.1%}")
            print(f"   Profit Factor: {summary['best_performance']['profit_factor']:.2f}x")
            print(f"   Target Achieved: {'✅ YES' if summary['achieved_target'] else '❌ NO'}")
        
    finally:
        # Always cleanup
        optimizer.cleanup()


if __name__ == '__main__':
    main()