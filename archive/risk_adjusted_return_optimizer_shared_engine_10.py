#!/usr/bin/env python3
"""
Risk-Adjusted Return Optimizer - Shared Engine Harness (v10)

Goal: Maintain ≥83.9% win rate while minimising drawdown, subject to a
minimum return floor. Objective: max_drawdown + λ·max(0, return_floor - total_return).

Uses the leak-free walkforward engine exported by
``final_optimal_walkforward copy.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import warnings
import tempfile
import shutil
import importlib.util

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
import pandas as pd

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
    """Risk-adjusted return optimizer with win rate and drawdown constraints."""
    
    def __init__(
        self,
        decision_table_path: Path,
        policy_path: Path,
        meta_path: Path,
        initial_capital: float = 10_000.0,
        min_win_rate: float = 0.835,        # Hard constraint: must maintain 83.5%+
        min_trades: int = 80,               # Must execute most trades
        max_drawdown_tolerance: float = 0.60,  # Prefer lower drawdowns
        min_return_threshold: float = 0.15,  # Minimum total return floor (decimal or percent)
        preprocess_decision_table: bool = True,
    ):
        self.decision_table_path = decision_table_path
        self.policy_path = policy_path
        self.meta_path = meta_path
        self.initial_capital = initial_capital
        self.min_win_rate = min_win_rate
        self.min_trades = min_trades
        self.max_drawdown_tolerance = max_drawdown_tolerance
        self.preprocess_decision_table = preprocess_decision_table

        # Interpret return threshold either as decimal or percent from legacy callers.
        if min_return_threshold > 1.0:
            self.return_floor = float(min_return_threshold) / 100.0
        else:
            self.return_floor = float(min_return_threshold)

        self.penalty_lambda = 3.0
        self.prune_drawdown = 0.35
        self.prune_return = 0.05
        
        self.logger = logging.getLogger("risk_adjusted_optimizer")
        mode = "with preprocessing" if preprocess_decision_table else "without preprocessing"
        self.logger.info(
            f"🎯 RISK-ADJUSTED OPTIMIZATION ({mode}): Win Rate ≥ {min_win_rate:.1%}, minimize MDD + λ·shortfall"
        )
        self.logger.info(
            f"⚡ Objective: keep drawdown small while meeting ≥{self.return_floor:.1%} return"
        )
        
        # Track results
        self.best_results = []
        
        # Create temp directory
        self.temp_dir = Path(tempfile.mkdtemp(prefix="risk_adjusted_"))
        self.logger.info(f"Working directory: {self.temp_dir}")
    
    def objective(self, trial: optuna.Trial) -> float:
        """Minimize max drawdown with a return floor, subject to win rate and trade constraints."""
        try:
            params = self._sample_risk_adjusted_parameters(trial)
            simulation = self._run_risk_adjusted_simulation(trial, params)

            if simulation is None:
                raise optuna.TrialPruned()

            results_df, summary = simulation

            equity_series = (
                results_df['equity_curve_value']
                if 'equity_curve_value' in results_df.columns
                else results_df.get('equity_after', pd.Series(dtype=float))
            )
            equity_series = equity_series.fillna(method='ffill').fillna(self.initial_capital)
            if equity_series.empty:
                equity_series = pd.Series([self.initial_capital], name='equity_curve_value')

            running_max = equity_series.cummax()
            drawdowns = (equity_series - running_max) / running_max
            max_drawdown = float(-drawdowns.min()) if not drawdowns.empty else 0.0

            final_equity = float(summary.get('final_capital', self.initial_capital))
            total_return = (final_equity / self.initial_capital) - 1.0
            win_rate = float(summary.get('win_rate', 0.0))
            total_trades = int(summary.get('total_trades', 0))
            halted_trades = int(summary.get('halted_trades', 0))

            if max_drawdown > self.prune_drawdown and total_return < self.prune_return:
                raise optuna.TrialPruned()

            shortfall = max(0.0, self.return_floor - total_return)
            objective_value = max_drawdown + self.penalty_lambda * shortfall

            penalty = 0.0
            constraints_satisfied = True

            if win_rate < self.min_win_rate:
                penalty += (self.min_win_rate - win_rate) * 10.0
                constraints_satisfied = False

            if total_trades < self.min_trades:
                penalty += (self.min_trades - total_trades) * 0.02
                constraints_satisfied = False

            if max_drawdown > self.max_drawdown_tolerance:
                penalty += (max_drawdown - self.max_drawdown_tolerance) * 10.0
                constraints_satisfied = False

            if total_return < self.return_floor:
                constraints_satisfied = False

            objective_value += penalty

            trial.set_user_attr('win_rate', win_rate)
            trial.set_user_attr('total_trades', total_trades)
            trial.set_user_attr('max_drawdown', max_drawdown)
            trial.set_user_attr('total_return', total_return)
            trial.set_user_attr('shortfall', shortfall)
            trial.set_user_attr('objective', objective_value)

            self.best_results.append(
                {
                    'trial_number': trial.number,
                    'objective': objective_value,
                    'params': params.copy(),
                    'summary': summary,
                    'metrics': {
                        'win_rate': win_rate,
                        'total_return': total_return,
                        'max_drawdown': max_drawdown,
                        'total_trades': total_trades,
                        'halted_trades': halted_trades,
                        'shortfall': shortfall,
                    },
                    'constraints_satisfied': constraints_satisfied,
                }
            )

            self.logger.info(
                "Trial %d: obj=%.4f | win=%.1f%% | ret=%.1f%% | mdd=%.1f%% | trades=%d (halted %d)%s",
                trial.number,
                objective_value,
                win_rate * 100,
                total_return * 100,
                max_drawdown * 100,
                total_trades,
                halted_trades,
                " ✅" if constraints_satisfied else "",
            )

            return objective_value

        except optuna.TrialPruned:
            raise
        except Exception as exc:
            self.logger.error(f"Trial {trial.number} failed: {exc}")
            raise optuna.TrialPruned() from exc
    
    def _sample_risk_adjusted_parameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Sample leak-free parameters that the shared engine actually uses."""

        params: Dict[str, Any] = {
            'approach': 'risk_adjusted_vol_target',
            'base_contracts': 10,
            'base_notional': 1000.0,
            'holding_period_days': 5,
            'preserve_contract_risk_filter': True,
            'bypass_all_normal_controls': False,
            'max_notional_pct': trial.suggest_float('max_notional_pct', 0.08, 0.15, step=0.01),
            'commission_per_side': 0.65,
            'exchange_fee_per_side': 0.05,
            'slippage_min': 0.02,
            'slippage_pct': 0.20,
            'enable_single_trade_cap': False,
            'max_single_trade_notional': 999_999,
        }

        # Portfolio-level trailing stop
        params['enable_portfolio_trailing_stop'] = trial.suggest_categorical(
            'enable_portfolio_trailing_stop', [True, False]
        )
        if params['enable_portfolio_trailing_stop']:
            params['portfolio_dd_floor'] = trial.suggest_float('portfolio_dd_floor', 0.10, 0.20, step=0.01)
            params['halt_cooldown_days'] = trial.suggest_int('halt_cooldown_days', 5, 20)
        else:
            params['portfolio_dd_floor'] = 0.20
            params['halt_cooldown_days'] = 0

        # Volatility targeting surface
        params['enable_vol_targeting'] = trial.suggest_categorical('enable_vol_targeting', [True, False])
        if params['enable_vol_targeting']:
            params['target_annual_vol'] = trial.suggest_float('target_annual_vol', 0.10, 0.20, step=0.01)
            params['vol_lookback_days'] = trial.suggest_int('vol_lookback_days', 20, 60, step=5)
            params['vol_scale_min'] = trial.suggest_float('vol_scale_min', 0.30, 0.80, step=0.05)
            params['vol_scale_max'] = trial.suggest_float('vol_scale_max', 0.80, 1.50, step=0.05)
            if params['vol_scale_min'] > params['vol_scale_max']:
                params['vol_scale_min'], params['vol_scale_max'] = params['vol_scale_max'], params['vol_scale_min']
        else:
            params['target_annual_vol'] = 0.15
            params['vol_lookback_days'] = 30
            params['vol_scale_min'] = 0.5
            params['vol_scale_max'] = 1.2

        # Consecutive loss breaker
        params['enable_consecutive_loss_breaker'] = trial.suggest_categorical(
            'enable_consecutive_loss_breaker', [True, False]
        )
        params['max_consecutive_losses'] = (
            trial.suggest_int('max_consecutive_losses', 5, 20)
            if params['enable_consecutive_loss_breaker'] else 999
        )

        # Market halt protection
        params['enable_market_halt_protection'] = trial.suggest_categorical(
            'enable_market_halt_protection', [True, False]
        )
        params['halt_vol_emergency_only'] = False
        params['halt_vol_severity_threshold'] = 2.0
        if params['enable_market_halt_protection']:
            params['halt_vol_emergency_only'] = trial.suggest_categorical(
                'halt_vol_emergency_only', [True, False]
            )
            params['halt_vol_severity_threshold'] = trial.suggest_float(
                'halt_vol_severity_threshold', 1.0, 3.0, step=0.25
            )

        # Contract-level sizing levers
        params['enable_position_multiplier'] = trial.suggest_categorical(
            'enable_position_multiplier', [True, False]
        )
        params['position_multiplier'] = (
            trial.suggest_float('position_multiplier', 1.0, 2.2, step=0.1)
            if params['enable_position_multiplier'] else 1.0
        )

        params['enable_dynamic_sizing'] = trial.suggest_categorical('enable_dynamic_sizing', [True, False])
        params['lookback_window'] = (
            trial.suggest_int('lookback_window', 8, 20)
            if params['enable_dynamic_sizing'] else 12
        )

        params['enable_vol_adjustment'] = trial.suggest_categorical('enable_vol_adjustment', [True, False])
        params['vol_lookback'] = (
            trial.suggest_int('vol_lookback', 12, 30)
            if params['enable_vol_adjustment'] else 20
        )

        # Return and notional filters
        params['enable_return_filter'] = trial.suggest_categorical('enable_return_filter', [True, False])
        params['min_expected_return'] = (
            trial.suggest_float('min_expected_return', 0.0, 0.02, step=0.002)
            if params['enable_return_filter'] else 0.0
        )

        params['max_notional_pct_per_trade'] = trial.suggest_float(
            'max_notional_pct_per_trade', 0.02, 0.05, step=0.005
        )
        params['max_daily_notional_pct'] = trial.suggest_float(
            'max_daily_notional_pct', 0.05, 0.12, step=0.01
        )

        return params
    
    def _run_risk_adjusted_simulation(
        self,
        trial: optuna.Trial,
        params: Dict[str, Any],
    ) -> Optional[Tuple[pd.DataFrame, Dict[str, Any]]]:
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

            return results_df, summary

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
        self.logger.info(
            f"📈 Objective: minimize drawdown while meeting ≥{self.return_floor:.1%} return"
        )
        
        # Create study
        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=42, multivariate=True, group=True),
            pruner=HyperbandPruner(min_resource=20),
            study_name=study_name
        )
        
        # Run optimization
        study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)
        
        # Analyze results
        best_trial = study.best_trial
        best_detailed = None

        valid_trials = [r for r in self.best_results if r.get('constraints_satisfied', False)]
        if valid_trials:
            best_detailed = min(valid_trials, key=lambda x: x['objective'])

        self.logger.info("🚀 Risk-Adjusted Optimization Complete!")
        self.logger.info(f"Best trial overall: #{best_trial.number}")
        self.logger.info(f"Best objective value: {best_trial.value:.4f}")

        if best_detailed:
            metrics = best_detailed['metrics']
            summary_dict = best_detailed['summary']

            achieved_return = float(metrics.get('total_return', 0.0))
            achieved_return_pct = summary_dict.get('return_pct', achieved_return * 100.0)
            achieved_mdd = float(metrics.get('max_drawdown', 0.0))
            achieved_win_rate = float(metrics.get('win_rate', 0.0))
            trades = int(metrics.get('total_trades', 0))
            halted = int(metrics.get('halted_trades', 0))

            calmar_ratio = (
                achieved_return_pct / (achieved_mdd * 100.0)
                if achieved_mdd > 0 else float('inf')
            )

            self.logger.info("📊 Best Constraint-Satisfying Performance:")
            self.logger.info(
                "  Win Rate: %.1f%% (constraint ≥%.1f%%)",
                achieved_win_rate * 100,
                self.min_win_rate * 100,
            )
            self.logger.info(
                "  Return: %.1f%% (floor %.1f%%)",
                achieved_return_pct,
                self.return_floor * 100,
            )
            self.logger.info("  Max Drawdown: %.1f%%", achieved_mdd * 100)
            self.logger.info("  Objective (mdd + λ·shortfall): %.4f", best_detailed['objective'])
            self.logger.info("  Calmar Ratio: %.1f", calmar_ratio)
            self.logger.info("  Trades: %d (halted: %d)", trades, halted)

            baseline_return_pct = 1119.7
            baseline_drawdown = 0.649
            return_improvement = (achieved_return_pct / baseline_return_pct - 1.0) * 100
            drawdown_improvement = (baseline_drawdown - achieved_mdd) / baseline_drawdown * 100

            self.logger.info(f"  Return Improvement vs baseline: {return_improvement:+.1f}%")
            self.logger.info(f"  Drawdown Improvement vs baseline: {drawdown_improvement:+.1f}%")

        optimization_summary = {
            'study_name': study_name,
            'n_trials': n_trials,
            'optimization_focus': 'drawdown_minimization_with_return_floor',
            'objective_definition': 'max_drawdown + lambda * max(0, return_floor - total_return)',
            'lambda_penalty': self.penalty_lambda,
            'return_floor': self.return_floor,
            'constraints': {
                'min_win_rate': self.min_win_rate,
                'min_trades': self.min_trades,
                'max_drawdown_tolerance': self.max_drawdown_tolerance,
            },
            'prune_thresholds': {
                'drawdown': self.prune_drawdown,
                'return': self.prune_return,
            },
            'baseline_performance': {
                'win_rate': 0.839,
                'return_pct': 1119.7,
                'max_drawdown': 0.649,
                'trades': 87,
                'calmar_ratio': 1119.7 / 64.9,
            },
            'best_trial_number': best_detailed['trial_number'] if best_detailed else best_trial.number,
            'best_objective': best_detailed['objective'] if best_detailed else best_trial.value,
            'best_params': best_detailed['params'] if best_detailed else best_trial.params,
            'best_performance': best_detailed['summary'] if best_detailed else None,
            'best_metrics': best_detailed['metrics'] if best_detailed else None,
            'valid_trials_count': len(valid_trials),
            'total_trials': len(self.best_results),
        }
        
        return study, optimization_summary
    
    def cleanup(self):
        """Clean up temporary files."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            self.logger.info("Cleaned up temporary files")


def main():
    """Run risk-adjusted optimization."""
    parser = argparse.ArgumentParser(
        description="Risk-adjusted optimization (drawdown + return floor objective)"
    )
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
            metrics = summary.get('best_metrics', {}) or {}

            achieved_win_rate = results.get('win_rate', metrics.get('win_rate', 0.0))
            achieved_return_pct = results.get('return_pct', metrics.get('total_return', 0.0) * 100.0)
            achieved_mdd = results.get('max_drawdown', metrics.get('max_drawdown', 0.0))
            trades = int(results.get('total_trades', metrics.get('total_trades', 0)))
            halted = int(results.get('halted_trades', metrics.get('halted_trades', 0)))

            achieved_calmar = (
                achieved_return_pct / (achieved_mdd * 100.0)
                if achieved_mdd > 0 else float('inf')
            )

            baseline_calmar = 1119.7 / 64.9
            calmar_improvement = (
                (achieved_calmar / baseline_calmar - 1.0) * 100
                if achieved_calmar not in (0, float('inf')) else float('inf')
            )

            print("\n🚀 Risk-Adjusted Optimization Complete!")
            print(f"📊 Results saved to: {args.outdir}")
            print(f"⚡ Final backtest in: {final_dir}")
            print("\n📈 Performance Summary:")
            print(f"   🎯 Constraint: Win Rate ≥ {args.min_win_rate:.1%}")
            print(f"   📊 Achieved: {achieved_win_rate:.1%} win rate, {achieved_return_pct:.1f}% returns")
            print(f"   🛡️ Max Drawdown: {achieved_mdd:.1%} (vs baseline 64.9%)")
            print(f"   📈 Calmar Ratio: {achieved_calmar:.1f} (vs baseline {baseline_calmar:.1f})")
            if calmar_improvement not in (float('inf'), float('-inf')):
                print(f"   🚀 Calmar Improvement: {calmar_improvement:+.1f}%")
            print(f"   📊 Trades: {trades} (vs baseline 87)")
            print(f"   🚨 Emergency Halts: {halted}")

            print("\n🎯 Best Configuration:")
            interesting_keys = [
                'enable_vol_targeting',
                'target_annual_vol',
                'vol_lookback_days',
                'vol_scale_min',
                'vol_scale_max',
                'enable_portfolio_trailing_stop',
                'portfolio_dd_floor',
                'halt_cooldown_days',
                'enable_consecutive_loss_breaker',
                'max_consecutive_losses',
                'max_notional_pct_per_trade',
                'max_daily_notional_pct',
                'enable_position_multiplier',
                'position_multiplier',
                'enable_dynamic_sizing',
                'lookback_window',
                'enable_vol_adjustment',
                'vol_lookback',
                'enable_return_filter',
                'min_expected_return',
            ]

            for key in interesting_keys:
                if key in summary['best_params']:
                    value = summary['best_params'][key]
                    print(f"   {key}: {value}")
        
    finally:
        # Cleanup
        optimizer.cleanup()


if __name__ == '__main__':
    main()
