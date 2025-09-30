#!/usr/bin/env python3
"""
Risk-Adjusted Return Optimizer - Optimize Calmar Ratio (Return/Drawdown)

Goal: Maintain 83.9% win rate while maximizing risk-adjusted returns.
Optimize for Calmar Ratio = Annual Return / Maximum Drawdown

This should find configurations with high returns AND low drawdowns.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
import tempfile
import shutil
import importlib.util

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO)

_ENGINE_MODULE_NAME = "walkforward_shared_engine"
_ENGINE_MODULE = None
_ENGINE_PATH = Path(__file__).with_name("final_optimal_walkforward copy.py")


def _load_engine_module():
    global _ENGINE_MODULE
    if _ENGINE_MODULE is None:
        if not _ENGINE_PATH.exists():
            raise FileNotFoundError(f"Shared walkforward engine not found at {_ENGINE_PATH}")
        spec = importlib.util.spec_from_file_location(_ENGINE_MODULE_NAME, _ENGINE_PATH)
        if spec is None or spec.loader is None:
            raise ImportError("Unable to load shared walkforward engine module")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _ENGINE_MODULE = module
    return _ENGINE_MODULE


class RiskAdjustedReturnOptimizer:
    """
    Risk-adjusted return optimizer with win rate constraint.
    
    Strategy: Maintain 83.9% win rate while maximizing Calmar Ratio (Return/Max Drawdown).
    """
    
    def __init__(
        self,
        decision_table_path: Path,
        policy_path: Path,
        meta_path: Path,
        initial_capital: float = 10_000.0,
        min_win_rate: float = 0.835,        # Hard constraint: must maintain 83.5%+
        min_trades: int = 80,               # Must execute most trades
        max_drawdown_tolerance: float = 0.60,  # Prefer lower drawdowns
        min_return_threshold: float = 1500.0,  # Minimum return to consider (≈15× baseline)
        preprocess_decision_table: bool = True,
    ):
        self.decision_table_path = decision_table_path
        self.policy_path = policy_path
        self.meta_path = meta_path
        self.initial_capital = initial_capital
        self.min_win_rate = min_win_rate
        self.min_trades = min_trades
        self.max_drawdown_tolerance = max_drawdown_tolerance
        self.min_return_threshold = min_return_threshold
        self.preprocess_decision_table = preprocess_decision_table
        
        self.logger = logging.getLogger("risk_adjusted_optimizer")
        mode = "with preprocessing" if preprocess_decision_table else "without preprocessing"
        self.logger.info(
            f"🎯 RISK-ADJUSTED OPTIMIZATION ({mode}): Win Rate ≥ {min_win_rate:.1%}, Maximize Calmar Ratio"
        )
        self.logger.info(f"⚡ Strategy: High returns with low drawdowns (Return/Max Drawdown)")
        
        # Track results
        self.best_results = []
        
        # Create temp directory
        self.temp_dir = Path(tempfile.mkdtemp(prefix="risk_adjusted_"))
        self.logger.info(f"Working directory: {self.temp_dir}")
    
    def objective(self, trial: optuna.Trial) -> float:
        """Maximize risk-adjusted returns (Calmar Ratio) subject to win rate constraint."""
        try:
            # Sample parameters focused on risk-adjusted optimization
            params = self._sample_risk_adjusted_parameters(trial)
            
            # Run simulation
            results = self._run_risk_adjusted_simulation(trial, params)
            
            if results is None:
                return -1000.0  # Heavy penalty for failed trials
            
            # Extract metrics
            win_rate = results.get('win_rate', 0.0)
            total_trades = results.get('total_trades', 0)
            max_drawdown = results.get('max_drawdown', 1.0)
            return_pct = results.get('return_pct', 0.0)
            
            # Hard constraints - if any violated, return heavy penalty
            if win_rate < self.min_win_rate:
                penalty = (self.min_win_rate - win_rate) * 10000
                return max(-1000.0, -penalty)
            
            if total_trades < self.min_trades:
                penalty = (self.min_trades - total_trades) * 50
                return max(-1000.0, -penalty)

            if max_drawdown > self.max_drawdown_tolerance:
                penalty = (max_drawdown - self.max_drawdown_tolerance) * 10_000
                return max(-1000.0, -penalty)

            if return_pct < self.min_return_threshold:
                penalty = (self.min_return_threshold - return_pct) * 5
                return max(-1000.0, -penalty)
            
            # Calculate Calmar Ratio (Return / Max Drawdown)
            # Higher is better - high returns with low drawdowns
            if max_drawdown <= 0.001:  # Avoid division by zero
                calmar_ratio = return_pct * 1000  # Very high score for near-zero drawdown
            else:
                calmar_ratio = return_pct / (max_drawdown * 100)  # Return% / Drawdown%
            
            # Bonus multipliers for exceptional performance
            if win_rate >= 0.85:  # Bonus for exceeding win rate
                calmar_ratio *= 1.1
                
            if total_trades >= 87:  # Bonus for executing all expected trades
                calmar_ratio *= 1.05
                
            if max_drawdown <= 0.20:  # Big bonus for low drawdown
                calmar_ratio *= 1.2
            elif max_drawdown <= 0.15:  # Huge bonus for very low drawdown
                calmar_ratio *= 1.5
            
            # Store results
            trial_result = {
                'trial_number': trial.number,
                'params': params.copy(),
                'results': results,
                'calmar_ratio': calmar_ratio,
                'constraints_satisfied': True
            }
            self.best_results.append(trial_result)
            
            # Log results
            halted_trades = results.get('halted_trades', 0)
            
            self.logger.info(
                f"Trial {trial.number}: "
                f"Win Rate: {win_rate:.1%} ({'✅' if win_rate >= self.min_win_rate else '❌'}), "
                f"Return: {return_pct:.1f}%, "
                f"Max DD: {max_drawdown:.1%}, "
                f"Calmar: {calmar_ratio:.1f}, "
                f"Trades: {total_trades} (halted: {halted_trades})"
            )
            
            return calmar_ratio
            
        except Exception as e:
            self.logger.error(f"Trial {trial.number} failed: {e}")
            return -1000.0
    
    def _sample_risk_adjusted_parameters(self, trial: optuna.Trial) -> Dict:
        """
        Sample parameters optimized for risk-adjusted returns.
        
        Focus on configurations that can achieve high returns with low drawdowns.
        """
        
        # Sample enable flags
        enable_portfolio_stop_loss = trial.suggest_categorical('enable_portfolio_stop_loss', [True, False])
        enable_single_trade_cap = trial.suggest_categorical('enable_single_trade_cap', [True, False])
        enable_market_halt_protection = trial.suggest_categorical('enable_market_halt_protection', [True, False])
        enable_consecutive_loss_breaker = trial.suggest_categorical('enable_consecutive_loss_breaker', [True, False])
        
        # Portfolio stop loss - focus on tighter risk control
        portfolio_stop_loss_pct = (
            trial.suggest_float('portfolio_stop_loss_pct', 0.40, 0.80, step=0.05)  # Tighter range
            if enable_portfolio_stop_loss else 1.0
        )
        
        # Trade size caps - explore both ends
        max_single_trade_notional = (
            trial.suggest_float('max_single_trade_notional', 80_000, 200_000, step=10_000)
            if enable_single_trade_cap else 999_999
        )

        halt_vol_emergency_only = (
            trial.suggest_categorical('halt_vol_emergency_only', [True, False])
            if enable_market_halt_protection else False
        )
        halt_vol_severity_threshold = trial.suggest_float('halt_vol_severity_threshold', 1.0, 3.0, step=0.25)

        # Consecutive losses - explore tighter controls for drawdown management
        max_consecutive_losses = (
            trial.suggest_int('max_consecutive_losses', 15, 50)  # Wider range including tighter controls
            if enable_consecutive_loss_breaker else 999
        )
        
        # Position sizing - explore more granular levels
        enable_position_multiplier = trial.suggest_categorical('enable_position_multiplier', [True, False])
        position_multiplier = (
            trial.suggest_float('position_multiplier', 1.0, 2.5, step=0.1)  # Expanded range
            if enable_position_multiplier else 1.0
        )
        
        # Return filtering for risk management
        enable_return_filter = trial.suggest_categorical('enable_return_filter', [True, False])
        min_expected_return = (
            trial.suggest_float('min_expected_return', 0.0, 0.03, step=0.002)  # Tighter filtering
            if enable_return_filter else 0.0
        )
        
        # NEW: Dynamic position sizing based on recent performance
        enable_dynamic_sizing = trial.suggest_categorical('enable_dynamic_sizing', [True, False])
        lookback_window = (
            trial.suggest_int('lookback_window', 5, 20)  # Look at last N trades
            if enable_dynamic_sizing else 10
        )
        
        # NEW: Volatility-based position sizing
        enable_vol_adjustment = trial.suggest_categorical('enable_vol_adjustment', [True, False])
        vol_lookback = (
            trial.suggest_int('vol_lookback', 10, 30)  # Volatility calculation window
            if enable_vol_adjustment else 20
        )
        
        return {
            # Keep the base bypassed approach that preserves 83.9%
            'approach': 'risk_adjusted_optimization',
            'base_contracts': 10,
            'base_notional': 1000.0,
            'bypass_all_normal_controls': True,
            'preserve_contract_risk_filter': True,
            'holding_period_days': 5,
            'max_notional_pct': 0.10,
            
            # Risk management controls
            'enable_portfolio_stop_loss': enable_portfolio_stop_loss,
            'portfolio_stop_loss_pct': portfolio_stop_loss_pct,
            'enable_single_trade_cap': enable_single_trade_cap,
            'max_single_trade_notional': max_single_trade_notional,
            'enable_market_halt_protection': enable_market_halt_protection,
            'halt_vol_emergency_only': halt_vol_emergency_only,
            'halt_vol_severity_threshold': halt_vol_severity_threshold,
            'enable_consecutive_loss_breaker': enable_consecutive_loss_breaker,
            'max_consecutive_losses': max_consecutive_losses,

            # Return enhancement features
            'enable_position_multiplier': enable_position_multiplier,
            'position_multiplier': position_multiplier,
            'enable_return_filter': enable_return_filter,
            'min_expected_return': min_expected_return,

            # NEW: Advanced risk-adjusted features
            'enable_dynamic_sizing': enable_dynamic_sizing,
            'lookback_window': lookback_window,
            'enable_vol_adjustment': enable_vol_adjustment,
            'vol_lookback': vol_lookback,
            
            # Keep transaction costs the same
            'commission_per_side': 0.65,
            'exchange_fee_per_side': 0.05,
            'slippage_min': 0.02,
            'slippage_pct': 0.20,
        }
    
    def _run_risk_adjusted_simulation(self, trial: optuna.Trial, params: Dict) -> Optional[Dict]:
        """Run the shared walkforward engine directly for this trial."""
        try:
            engine = _load_engine_module()
            meta = engine._load_meta(self.meta_path)

            decision_df = engine.load_decision_table(
                self.decision_table_path,
                meta['action_map'],
                preprocess=self.preprocess_decision_table,
                logger=self.logger,
            )

            states = engine._standardise_states(
                decision_df,
                meta['state_columns'],
                meta['scaler_mean'],
                meta['scaler_scale'],
            )

            algo = engine._load_policy_robust(self.policy_path, meta)
            predicted_actions = algo.predict(states)

            results_df, summary = engine.simulate_optimal_walkforward(
                decision_df,
                predicted_actions,
                meta['action_map'],
                params,
                self.initial_capital,
            )

            trial_dir = self.temp_dir / f"trial_{trial.number}"
            trial_dir.mkdir(exist_ok=True)
            results_df.to_csv(trial_dir / 'risk_adjusted_trades.csv', index=False)
            with (trial_dir / 'risk_adjusted_summary.json').open('w', encoding='utf-8') as fh:
                json.dump(summary, fh, indent=2, default=str)

            return summary

        except Exception as exc:
            self.logger.error(f"Trial {trial.number} simulation error: {exc}")
            return None

    def optimize(
        self, 
        n_trials: int = 100,
        study_name: str = "risk_adjusted_optimization"
    ) -> Tuple[optuna.Study, Dict]:
        """Run risk-adjusted optimization."""
        
        self.logger.info(f"🚀 Starting Risk-Adjusted Optimization with {n_trials} trials")
        self.logger.info(f"🎯 Constraint: Win Rate ≥ {self.min_win_rate:.1%}")
        self.logger.info(f"📈 Objective: Maximize Calmar Ratio (Return/Max Drawdown)")
        
        # Create study
        study = optuna.create_study(
            direction="maximize",  # Maximize Calmar Ratio
            sampler=TPESampler(seed=42),
            pruner=HyperbandPruner(min_resource=20),
            study_name=study_name
        )
        
        # Run optimization
        study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)
        
        # Analyze results
        best_trial = study.best_trial
        best_detailed = None
        
        # Find best valid trial (satisfies constraints)
        valid_trials = [r for r in self.best_results if r.get('constraints_satisfied', False)]
        if valid_trials:
            best_detailed = max(valid_trials, key=lambda x: x['calmar_ratio'])
        
        self.logger.info("🚀 Risk-Adjusted Optimization Complete!")
        self.logger.info(f"Best trial: #{best_trial.number}")
        self.logger.info(f"Best Calmar Ratio: {best_trial.value:.1f}")
        
        if best_detailed:
            results = best_detailed['results']
            
            self.logger.info("📊 Best Risk-Adjusted Performance:")
            self.logger.info(f"  Win Rate: {results.get('win_rate', 0):.1%} (constraint: ≥{self.min_win_rate:.1%})")
            self.logger.info(f"  Return: {results.get('return_pct', 0):.1f}% (vs baseline 1,119.7%)")
            self.logger.info(f"  Max Drawdown: {results.get('max_drawdown', 0):.1%}")
            self.logger.info(f"  Calmar Ratio: {results.get('calmar_ratio', 0):.1f}")
            self.logger.info(f"  Trades: {results.get('total_trades', 0)} (vs baseline 87)")
            self.logger.info(f"  Halted Trades: {results.get('halted_trades', 0)}")
            
            # Compare to pure return optimization
            return_improvement = results.get('return_pct', 0) / 1119.7 - 1.0
            drawdown_improvement = (0.649 - results.get('max_drawdown', 0)) / 0.649
            
            self.logger.info(f"  Return Improvement: {return_improvement*100:+.1f}%")
            self.logger.info(f"  Drawdown Improvement: {drawdown_improvement*100:+.1f}%")
        
        # Compile results
        optimization_summary = {
            'study_name': study_name,
            'n_trials': n_trials,
            'optimization_focus': 'risk_adjusted_calmar_ratio',
            'constraints': {
                'min_win_rate': self.min_win_rate,
                'min_trades': self.min_trades,
                'max_drawdown_tolerance': self.max_drawdown_tolerance,
                'min_return_threshold': self.min_return_threshold
            },
            'baseline_performance': {
                'win_rate': 0.839,
                'return_pct': 1119.7,
                'max_drawdown': 0.649,
                'trades': 87,
                'calmar_ratio': 1119.7 / 64.9
            },
            'best_trial_number': best_trial.number,
            'best_calmar_ratio': best_trial.value,
            'best_params': best_trial.params,
            'best_performance': best_detailed['results'] if best_detailed else None,
            'valid_trials_count': len(valid_trials),
            'total_trials': len(self.best_results)
        }
        
        return study, optimization_summary
    
    def cleanup(self):
        """Clean up temporary files."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            self.logger.info("Cleaned up temporary files")


def main():
    """Run risk-adjusted optimization."""
    parser = argparse.ArgumentParser(description="Risk-adjusted optimization (Calmar Ratio)")
    parser.add_argument('--decision-table', type=Path, required=True)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--meta', type=Path, required=True)
    parser.add_argument('--trials', type=int, default=100, help="Number of optimization trials")
    parser.add_argument('--min-win-rate', type=float, default=0.835, help="Minimum win rate constraint")
    parser.add_argument('--outdir', type=Path, default=Path('results/risk_adjusted_optimization'))
    parser.add_argument('--study-name', default='risk_adjusted_opt', help="Optuna study name")
    parser.add_argument('--skip-preprocessing', action='store_true', help='Use decision table as-is (no forward-label pruning).')
    
    args = parser.parse_args()
    
    # Validate inputs
    for file_path in [args.decision_table, args.policy, args.meta]:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
    
    # Create output directory
    args.outdir.mkdir(parents=True, exist_ok=True)
    
    # Create optimizer
    optimizer = RiskAdjustedReturnOptimizer(
        decision_table_path=args.decision_table,
        policy_path=args.policy,
        meta_path=args.meta,
        min_win_rate=args.min_win_rate,
        preprocess_decision_table=not args.skip_preprocessing,
    )
    
    try:
        # Run optimization
        study, summary = optimizer.optimize(n_trials=args.trials, study_name=args.study_name)
        
        # Save results
        results_file = args.outdir / 'risk_adjusted_results.json'
        with open(results_file, 'w') as f:
            clean_summary = json.loads(json.dumps(summary, default=str))
            json.dump({
                'optimization_summary': clean_summary,
                'all_trials': optimizer.best_results
            }, f, indent=2, default=str)
        
        # Final validation
        if summary['best_performance']:
            final_dir = args.outdir / 'final_risk_adjusted_backtest'
            final_dir.mkdir(exist_ok=True)
            
            # Save best parameters and results
            best_params_file = final_dir / 'risk_adjusted_best_params.json'
            with open(best_params_file, 'w') as f:
                json.dump(summary['best_params'], f, indent=2)
                
            best_results_file = final_dir / 'risk_adjusted_summary.json'
            with open(best_results_file, 'w') as f:
                json.dump(summary['best_performance'], f, indent=2, default=str)
            
            # Print final summary
            results = summary['best_performance']
            baseline_calmar = 1119.7 / 64.9
            achieved_calmar = results.get('calmar_ratio', 0)
            calmar_improvement = (achieved_calmar / baseline_calmar - 1.0) * 100
            
            print(f"\n🚀 Risk-Adjusted Optimization Complete!")
            print(f"📊 Results saved to: {args.outdir}")
            print(f"⚡ Final backtest in: {final_dir}")
            print(f"\n📈 Performance Summary:")
            print(f"   🎯 Constraint: Win Rate ≥ {args.min_win_rate:.1%}")
            print(f"   📊 Achieved: {results.get('win_rate', 0):.1%} win rate, {results.get('return_pct', 0):.1f}% returns")
            print(f"   🛡️ Max Drawdown: {results.get('max_drawdown', 0):.1%} (vs baseline 64.9%)")
            print(f"   📈 Calmar Ratio: {achieved_calmar:.1f} (vs baseline {baseline_calmar:.1f})")
            print(f"   🚀 Calmar Improvement: {calmar_improvement:+.1f}%")
            print(f"   📊 Trades: {results.get('total_trades', 0)} (vs baseline 87)")
            print(f"   🚨 Emergency Halts: {results.get('halted_trades', 0)}")
            
            print(f"\n🎯 Best Configuration:")
            for key, value in summary['best_params'].items():
                if value != False and value != 999999 and value != 1.0 and value != 999:
                    print(f"   {key}: {value}")
        
    finally:
        # Cleanup
        optimizer.cleanup()


if __name__ == '__main__':
    main()
