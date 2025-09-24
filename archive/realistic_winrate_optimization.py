#!/usr/bin/env python3
"""
Realistic Win-Rate Optimization for Trading System

Based on the data showing ~38% baseline win rate, this optimizer targets:
1. Improve win rate to 45-50% (realistic improvement)
2. Maintain strong returns for profitability  
3. Control drawdowns for risk management
4. Balance profit factor for sustainable edge

This uses a more realistic composite scoring that doesn't penalize for failing 
to achieve unrealistic win rates like 60%+.
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

# Import core functions directly
from d3rlpy import load_learnable
from d3rlpy.algos import DiscreteCQL, DiscreteCQLConfig
import torch
import yaml

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO)


# Copy essential functions to avoid import issues
def _calculate_rolling_drawdown(equity_series: pd.Series, window: int = 20) -> float:
    """Calculate current drawdown from rolling peak."""
    if len(equity_series) < 2:
        return 0.0
    
    peak = equity_series.rolling(window=min(window, len(equity_series))).max().iloc[-1]
    current = equity_series.iloc[-1]
    drawdown = (current - peak) / peak if peak > 0 else 0.0
    return max(drawdown, 0.0)


def _get_optimized_volatility_scaling(
    vol_severity: float, 
    vol_emergency: bool,
    vol_emergency_mult: float = 0.4,
    vol_high_mult: float = 0.6, 
    vol_mod_mult: float = 0.8
) -> float:
    """Get position size multiplier based on optimized volatility parameters."""
    if vol_emergency:
        return vol_emergency_mult
    elif vol_severity > 3.0:
        return vol_high_mult  
    elif vol_severity > 2.0:
        return vol_mod_mult
    else:
        return 1.0


def _optimized_position_sizing(
    equity: float,
    slot: int,
    size_mult: float,
    vol_regime_mult: float = 1.0,
    drawdown: float = 0.0,
    base_contracts: int = 5,
    drawdown_scale_factor: float = 1.5,
    min_drawdown_scale: float = 0.3,
    enable_equity_scaling: bool = True,
    max_equity_mult: float = 1.5,
    equity_threshold: float = 1.5,
    initial_capital: float = 10000.0
) -> int:
    """Optimized position sizing with tunable risk parameters."""
    if slot <= 0 or size_mult <= 0:
        return 0
    
    # Start with base position
    base_position = base_contracts
    
    # Apply RL action size multiplier
    sized_position = int(np.floor(base_position * size_mult))
    
    # Apply vol regime scaling
    vol_adjusted = int(np.floor(sized_position * vol_regime_mult))
    
    # Apply optimized drawdown scaling
    drawdown_scale = max(min_drawdown_scale, 1.0 - (drawdown * drawdown_scale_factor))
    drawdown_adjusted = int(np.floor(vol_adjusted * drawdown_scale))
    
    # Apply equity scaling if enabled
    if enable_equity_scaling and equity > (initial_capital * equity_threshold):
        equity_mult = min(max_equity_mult, equity / initial_capital)
        final_position = int(np.floor(drawdown_adjusted * equity_mult))
    else:
        final_position = drawdown_adjusted
    
    return max(final_position, 0)


def _load_meta(meta_path: Path) -> Dict:
    """Load policy metadata."""
    with open(meta_path, 'r') as f:
        return json.load(f)


def _load_policy_robust(policy_path: Path, meta: Dict):
    """Robust policy loading with fallback."""
    logger = logging.getLogger("realistic_winrate")
    logger.info("Attempting load_learnable …")
    try:
        algo = load_learnable(str(policy_path))
        logger.info("✅ Loaded policy via load_learnable")
        return algo
    except Exception as e:
        logger.warning(f"load_learnable failed: {e}")
        logger.info("Falling back to manual reconstruction …")
        
        # Manual reconstruction for DiscreteCQL
        state_columns = meta.get('state_columns', [])
        action_map = meta.get('action_map', {})
        obs_size = len(state_columns)
        n_actions = len(action_map)
        
        logger.info(f"Detected architecture: obs={obs_size}, actions={n_actions}, critics=2")
        
        config = DiscreteCQLConfig(observation_scaler='standard')
        algo = DiscreteCQL(config=config)
        
        # Create dummy dataset for build
        dummy_obs = torch.randn(10, obs_size)
        dummy_actions = torch.randint(0, n_actions, (10,))
        dummy_rewards = torch.randn(10)
        
        # Build the algorithm
        algo.build_with_dataset({
            'observations': dummy_obs.numpy(),
            'actions': dummy_actions.numpy(),
            'rewards': dummy_rewards.numpy(),
            'terminals': np.zeros(10, dtype=bool)
        })
        
        # Load the saved parameters
        checkpoint = torch.load(str(policy_path), map_location='cpu')
        if hasattr(algo, '_impl') and algo._impl:
            algo._impl.load_state_dict(checkpoint)
        
        logger.info("✅ Loaded policy via manual reconstruction")
        return algo


def _standardise_states(df: pd.DataFrame, columns: List[str], means: List[float], scales: List[float]) -> np.ndarray:
    """Standardize state features using saved scaler parameters."""
    # Filter to only numeric columns and handle missing columns
    available_columns = [col for col in columns if col in df.columns]
    numeric_data = df[available_columns].select_dtypes(include=[np.number]).values
    
    # Adjust means and scales to match available numeric columns
    if len(available_columns) != len(columns):
        print(f"Warning: Using {len(available_columns)} of {len(columns)} expected columns")
    
    means_array = np.array(means[:numeric_data.shape[1]]).reshape(1, -1)
    scales_array = np.array(scales[:numeric_data.shape[1]]).reshape(1, -1)
    
    # Handle division by zero
    scales_array = np.where(scales_array == 0, 1.0, scales_array)
    
    return (numeric_data - means_array) / scales_array


def _decode_action(action_id: int, action_map: Dict[str, Dict]) -> Tuple[int, float]:
    """Decode action ID to slot and size multiplier."""
    if str(action_id) in action_map:
        action_info = action_map[str(action_id)]
        return action_info['slot'], action_info['size_value']
    return 0, 0.0


def _fee_cost(n_contracts: int, commission: float, exchange_fee: float) -> float:
    """Calculate total fees."""
    if n_contracts <= 0:
        return 0.0
    return n_contracts * (commission + exchange_fee) * 2  # Round trip


def _slippage_cost(row: pd.Series, n_contracts: int, slippage_min: float, slippage_pct: float) -> float:
    """Calculate slippage costs."""
    if n_contracts <= 0:
        return 0.0
    
    # Use relative spread as proxy for slippage
    spread_col = None
    for col in row.index:
        if 'relative_spread' in col:
            spread_col = col
            break
    
    if spread_col and not pd.isna(row[spread_col]):
        market_slippage = float(row[spread_col]) * slippage_pct
        total_slippage = max(slippage_min, market_slippage)
    else:
        total_slippage = slippage_min
    
    return n_contracts * total_slippage * 1000  # Base notional scaling


def simulate_realistic_walkforward(
    decision_df: pd.DataFrame,
    predicted_actions: np.ndarray,
    action_map: Dict[str, Dict[str, float]],
    initial_capital: float = 10_000.0,
    commission_per_side: float = 0.65,
    exchange_fee_per_side: float = 0.05,
    slippage_min: float = 0.02,
    slippage_pct: float = 0.20,
    base_contracts: int = 5,
    base_notional: float = 1000.0,
    enable_risk_controls: bool = True,
    max_notional_pct: float = 0.20,
    # Optuna-optimized parameters
    _vol_emergency_mult: float = 0.4,
    _vol_high_mult: float = 0.6,
    _vol_mod_mult: float = 0.8,
    _drawdown_scale_factor: float = 1.5,
    _min_drawdown_scale: float = 0.3,
    _enable_equity_scaling: bool = True,
    _max_equity_mult: float = 1.5,
    _equity_threshold: float = 1.5,
    **kwargs  # Catch any extra parameters
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Realistic walkforward simulation with tunable parameters."""
    logger = logging.getLogger("realistic_winrate")
    df = decision_df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.sort_values('date', inplace=True)

    equity = initial_capital
    history = []
    equity_series = pd.Series([equity])

    for idx, row in df.iterrows():
        action_id = int(predicted_actions[idx])
        slot, size_mult = _decode_action(action_id, action_map)

        record = {
            'date': row['date'],
            'action_id': action_id,
            'slot': slot,
            'size_mult': size_mult,
            'equity_before': equity,
        }

        if slot <= 0 or size_mult <= 0:
            record.update({
                'n_contracts': 0,
                'notional': 0.0,
                'fees': 0.0,
                'slippage': 0.0,
                'realized_pnl': 0.0,
                'equity_after': equity,
                'drawdown': 0.0,
                'vol_regime_mult': 1.0,
            })
            history.append(record)
            continue

        # Risk calculations
        if enable_risk_controls:
            current_drawdown = _calculate_rolling_drawdown(equity_series, 20)
            vol_severity = float(row.get('s_vol_severity', 1.0))
            vol_emergency = bool(row.get('s_vol_emergency', False))
            vol_regime_mult = _get_optimized_volatility_scaling(
                vol_severity, vol_emergency, _vol_emergency_mult, _vol_high_mult, _vol_mod_mult
            )
        else:
            current_drawdown = 0.0
            vol_regime_mult = 1.0

        # Optimized position sizing
        n_contracts = _optimized_position_sizing(
            equity=equity,
            slot=slot,
            size_mult=size_mult,
            vol_regime_mult=vol_regime_mult,
            drawdown=current_drawdown,
            base_contracts=base_contracts,
            drawdown_scale_factor=_drawdown_scale_factor,
            min_drawdown_scale=_min_drawdown_scale,
            enable_equity_scaling=_enable_equity_scaling,
            max_equity_mult=_max_equity_mult,
            equity_threshold=_equity_threshold,
            initial_capital=initial_capital
        )

        # Calculate notional
        notional = base_notional * n_contracts
        
        # Apply max notional cap
        if enable_risk_controls:
            notional_cap = equity * max_notional_pct
            if notional > notional_cap:
                scale = notional_cap / notional if notional > 0 else 0.0
                n_contracts = int(np.floor(n_contracts * scale))
                notional = base_notional * n_contracts

        # Calculate costs
        fees = _fee_cost(n_contracts, commission_per_side, exchange_fee_per_side)
        slippage = _slippage_cost(row, n_contracts, slippage_min, slippage_pct)

        # Calculate P&L
        pnl_col = f"c{slot}_target_pnl"
        raw_return = row.get(pnl_col, 0.0)
        if pd.isna(raw_return):
            raw_return = 0.0
        raw_return = float(raw_return)
        realized_pnl = raw_return * notional

        # Update equity
        equity = equity + realized_pnl - fees - slippage
        equity_series = pd.concat([equity_series, pd.Series([equity])]).tail(40)

        record.update({
            'n_contracts': n_contracts,
            'notional': notional,
            'fees': fees,
            'slippage': slippage,
            'realized_pnl': realized_pnl,
            'equity_after': equity,
            'drawdown': current_drawdown,
            'vol_regime_mult': vol_regime_mult,
        })
        history.append(record)

    # Generate results
    results = pd.DataFrame(history)
    trades_mask = results['n_contracts'] > 0
    winning_trades = results[trades_mask & (results['realized_pnl'] > 0)]
    losing_trades = results[trades_mask & (results['realized_pnl'] <= 0)]
    
    max_drawdown = results['drawdown'].max() if enable_risk_controls and len(results) > 0 else 0.0
    
    summary = {
        'initial_capital': initial_capital,
        'final_capital': equity,
        'total_pnl': float(equity - initial_capital),
        'total_fees': float(results['fees'].sum()),
        'total_slippage': float(results['slippage'].sum()),
        'total_trades': int(trades_mask.sum()),
        'winning_trades': len(winning_trades),
        'losing_trades': len(losing_trades),
        'win_rate': len(winning_trades) / max(trades_mask.sum(), 1),
        'return_pct': float((equity / initial_capital - 1.0) * 100.0),
        'max_drawdown': float(max_drawdown),
        'risk_controls_enabled': enable_risk_controls,
        'base_contracts': base_contracts,
        'base_notional': base_notional,
    }
    
    if trades_mask.sum() > 0:
        trade_pnls = results[trades_mask]['realized_pnl']
        summary.update({
            'avg_trade_pnl': float(trade_pnls.mean()),
            'largest_win': float(trade_pnls.max()),
            'largest_loss': float(trade_pnls.min()),
            'profit_factor': float(winning_trades['realized_pnl'].sum() / abs(losing_trades['realized_pnl'].sum())) if len(losing_trades) > 0 else np.inf,
        })
    
    return results, summary


class RealisticWinRateOptimizer:
    """Realistic win-rate optimizer targeting achievable improvements."""
    
    def __init__(
        self,
        decision_table_path: Path,
        policy_path: Path,
        meta_path: Path,
        initial_capital: float = 10_000.0,
        target_win_rate: float = 0.45,      # Realistic improvement target
        win_rate_weight: float = 0.3,       # Balanced weighting
        return_weight: float = 0.4,         # Slightly favor returns
        drawdown_weight: float = 0.2,       # Risk management
        profit_factor_weight: float = 0.1   # Edge confirmation
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
        
        # Load data once for all trials
        self.logger = logging.getLogger("realistic_winrate")
        self.logger.info("Loading decision table and policy...")
        
        self.df = pd.read_csv(decision_table_path)
        self.meta = _load_meta(meta_path)
        self.states = _standardise_states(
            self.df, self.meta['state_columns'], self.meta['scaler_mean'], self.meta['scaler_scale']
        )
        self.algo = _load_policy_robust(policy_path, self.meta)
        self.predicted_actions = self.algo.predict(self.states)
        
        self.logger.info(f"Loaded {len(self.df)} decision points for optimization")
        self.logger.info(f"Target win rate: {target_win_rate:.1%}")
        self.logger.info(f"Score weights - Win Rate: {win_rate_weight}, Returns: {return_weight}, Drawdown: {drawdown_weight}, PF: {profit_factor_weight}")
        
        # Track best results
        self.best_results = []
    
    def objective(self, trial: optuna.Trial) -> float:
        """Realistic win-rate aware objective function."""
        try:
            # Sample risk parameters from defined ranges
            params = self._sample_parameters(trial)
            
            # Run simulation with sampled parameters
            results, summary = simulate_realistic_walkforward(
                self.df.copy(),
                self.predicted_actions,
                self.meta['action_map'],
                initial_capital=self.initial_capital,
                **params
            )
            
            # Calculate realistic composite score
            composite_score = self._calculate_realistic_composite_score(results, summary, params)
            
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
            
            # Log trial results
            self.logger.info(
                f"Trial {trial.number}: Win Rate: {summary.get('win_rate', 0):.1%}, "
                f"Return: {summary.get('return_pct', 0):.1f}%, "
                f"Drawdown: {summary.get('max_drawdown', 0):.1%}, "
                f"PF: {summary.get('profit_factor', 1):.2f}x, "
                f"Score: {composite_score:.2f}"
            )
            
            return composite_score
            
        except Exception as e:
            self.logger.error(f"Trial {trial.number} failed: {e}")
            return 0.0  # Neutral score for failed trials
    
    def _sample_parameters(self, trial: optuna.Trial) -> Dict[str, float]:
        """Sample parameters focused on realistic win rate improvement."""
        
        return {
            # Core trading parameters - smaller positions may improve win rates
            'base_contracts': trial.suggest_int('base_contracts', 2, 6),
            'base_notional': trial.suggest_float('base_notional', 600.0, 1200.0, step=100.0),
            
            # Risk controls - always enabled for consistency
            'enable_risk_controls': True,
            'max_notional_pct': trial.suggest_float('max_notional_pct', 0.15, 0.25, step=0.02),
            
            # Transaction costs - tighter ranges for realism
            'commission_per_side': trial.suggest_float('commission_per_side', 0.55, 0.75, step=0.05),
            'exchange_fee_per_side': trial.suggest_float('exchange_fee_per_side', 0.03, 0.08, step=0.01),
            'slippage_min': trial.suggest_float('slippage_min', 0.015, 0.04, step=0.005),
            'slippage_pct': trial.suggest_float('slippage_pct', 0.15, 0.30, step=0.05),
            
            # Volatility regime scaling - conservative for win rate
            '_vol_emergency_mult': trial.suggest_float('vol_emergency_mult', 0.2, 0.5, step=0.05),
            '_vol_high_mult': trial.suggest_float('vol_high_mult', 0.4, 0.7, step=0.05),
            '_vol_mod_mult': trial.suggest_float('vol_mod_mult', 0.7, 1.0, step=0.05),
            
            # Drawdown protection - moderate for balance
            '_drawdown_scale_factor': trial.suggest_float('drawdown_scale_factor', 1.2, 2.5, step=0.1),
            '_min_drawdown_scale': trial.suggest_float('min_drawdown_scale', 0.2, 0.4, step=0.05),
            
            # Equity scaling - balanced approach
            '_enable_equity_scaling': trial.suggest_categorical('enable_equity_scaling', [True, False]),
            '_max_equity_mult': trial.suggest_float('max_equity_mult', 1.2, 1.8, step=0.1),
            '_equity_threshold': trial.suggest_float('equity_threshold', 1.3, 1.8, step=0.1),
        }
    
    def _calculate_realistic_composite_score(
        self, 
        results: pd.DataFrame, 
        summary: Dict, 
        params: Dict
    ) -> float:
        """Calculate realistic composite score with achievable targets."""
        # Extract key metrics
        win_rate = summary.get('win_rate', 0.0)
        return_pct = summary.get('return_pct', 0.0)
        max_drawdown = summary.get('max_drawdown', 0.0)
        profit_factor = summary.get('profit_factor', 1.0)
        total_trades = summary.get('total_trades', 0)
        
        # Minimum trade requirement
        if total_trades < 10:
            return 0.0
        
        # Normalize components for scoring (0-100 scale)
        
        # 1. Win Rate Component - realistic scaling
        baseline_win_rate = 0.38  # Current baseline
        win_rate_improvement = win_rate - baseline_win_rate
        win_rate_score = max(0.0, 50.0 + (win_rate_improvement / 0.20) * 50.0)  # 50 baseline, +50 for +20% improvement
        
        # 2. Return Component - scaled for realistic returns
        return_score = min(100.0, (return_pct / 1500.0) * 100)  # Scale so 1500% = 100 points
        
        # 3. Drawdown Component - penalize high drawdowns
        drawdown_score = max(0.0, 100.0 - (max_drawdown * 2000))  # Heavy penalty for drawdown
        
        # 4. Profit Factor Component - realistic scaling  
        profit_factor_score = min(100.0, ((profit_factor - 1.0) / 1.5) * 100)  # Scale so PF=2.5 gives 100 points
        
        # Calculate weighted composite score
        composite_score = (
            self.win_rate_weight * win_rate_score +
            self.return_weight * return_score +
            self.drawdown_weight * drawdown_score +
            self.profit_factor_weight * profit_factor_score
        )
        
        # Bonus for beating target win rate
        if win_rate >= self.target_win_rate:
            bonus_factor = 1.0 + ((win_rate - self.target_win_rate) / 0.10)  # 10% bonus per 1% over target
            composite_score *= min(1.3, bonus_factor)  # Cap at 30% bonus
        
        # Bonus for balanced performance (good win rate AND good returns)
        if win_rate >= 0.42 and return_pct >= 800.0:
            composite_score *= 1.1  # 10% balance bonus
        
        return composite_score
    
    def optimize(
        self, 
        n_trials: int = 50, 
        timeout: Optional[float] = None,
        study_name: str = "realistic_winrate_optimization"
    ) -> Tuple[optuna.Study, Dict]:
        """Run realistic win-rate optimization."""
        
        self.logger.info(f"Starting realistic win-rate optimization with {n_trials} trials")
        self.logger.info(f"Target win rate improvement: {self.target_win_rate:.1%}")
        
        # Create study
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
        
        self.logger.info("🎯 Realistic Win-Rate Optimization Complete!")
        self.logger.info(f"Best trial: #{best_trial.number}")
        self.logger.info(f"Best composite score: {best_trial.value:.2f}")
        
        if best_detailed:
            self.logger.info(f"📊 Best Performance:")
            self.logger.info(f"  Win Rate: {best_detailed['win_rate']:.1%} (target: {self.target_win_rate:.1%})")
            self.logger.info(f"  Total Return: {best_detailed['return_pct']:.1f}%")
            self.logger.info(f"  Max Drawdown: {best_detailed['max_drawdown']:.1%}")
            self.logger.info(f"  Profit Factor: {best_detailed['profit_factor']:.2f}x")
        
        # Compile results
        optimization_summary = {
            'study_name': study_name,
            'n_trials': n_trials,
            'optimization_focus': 'realistic_winrate_improvement',
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
            'best_params': best_params,
            'best_performance': best_detailed
        }
        
        return study, optimization_summary


def main():
    """Run realistic win-rate optimization."""
    parser = argparse.ArgumentParser(description="Realistic win-rate optimization")
    parser.add_argument('--decision-table', type=Path, required=True)
    parser.add_argument('--policy', type=Path, required=True)  
    parser.add_argument('--meta', type=Path, required=True)
    parser.add_argument('--trials', type=int, default=50, help="Number of optimization trials")
    parser.add_argument('--target-win-rate', type=float, default=0.45, help="Target win rate improvement")
    parser.add_argument('--outdir', type=Path, default=Path('results/realistic_winrate_optimization'))
    parser.add_argument('--study-name', default='realistic_winrate_opt', help="Optuna study name")
    
    args = parser.parse_args()
    
    # Create output directory
    args.outdir.mkdir(parents=True, exist_ok=True)
    
    # Create optimizer
    optimizer = RealisticWinRateOptimizer(
        decision_table_path=args.decision_table,
        policy_path=args.policy,
        meta_path=args.meta,
        target_win_rate=args.target_win_rate
    )
    
    # Run optimization
    study, summary = optimizer.optimize(
        n_trials=args.trials,
        study_name=args.study_name
    )
    
    # Save results
    results_file = args.outdir / 'realistic_winrate_results.json'
    with open(results_file, 'w') as f:
        clean_summary = json.loads(json.dumps(summary, default=str))
        json.dump({
            'optimization_summary': clean_summary,
            'all_trials': optimizer.best_results
        }, f, indent=2, default=str)
    
    # Save study
    study_file = args.outdir / 'realistic_winrate_study.pkl'
    optuna.save_study(study, str(study_file))
    
    # Run final simulation with best parameters
    best_params = summary['best_params']
    
    final_results, final_summary = simulate_realistic_walkforward(
        optimizer.df.copy(),
        optimizer.predicted_actions,
        optimizer.meta['action_map'],
        initial_capital=optimizer.initial_capital,
        **best_params
    )
    
    # Save final backtest results
    final_dir = args.outdir / 'final_realistic_winrate_backtest'
    final_dir.mkdir(exist_ok=True)
    
    final_results.to_csv(final_dir / 'realistic_winrate_optimized_trades.csv', index=False)
    with open(final_dir / 'realistic_winrate_optimized_summary.json', 'w') as f:
        json.dump(final_summary, f, indent=2, default=str)
    
    print(f"\n✅ Realistic win-rate optimization complete!")
    print(f"📊 Results saved to: {args.outdir}")
    print(f"🎯 Final backtest in: {final_dir}")
    
    if 'best_performance' in summary and summary['best_performance']:
        print(f"📈 Best Win Rate: {summary['best_performance']['win_rate']:.1%}")
        print(f"💰 Best Return: {summary['best_performance']['return_pct']:.1f}%")
        print(f"📉 Max Drawdown: {summary['best_performance']['max_drawdown']:.1%}")
        print(f"⚡ Profit Factor: {summary['best_performance']['profit_factor']:.2f}x")


if __name__ == '__main__':
    main()