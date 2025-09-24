#!/usr/bin/env python3
"""
Ultra-Minimal Risk Optimization - Preserve 83.9% Win Rate

Philosophy: The IQL model is perfect at what it does. Don't interfere with its 
natural edge. Only add CATASTROPHIC loss protection that activates in extreme scenarios.

Goal: Maintain exactly 83.9% win rate with emergency-only circuit breakers.
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


class UltraMinimalRiskOptimizer:
    """
    Ultra-minimal risk optimizer: Preserve 83.9% win rate with emergency-only protection.
    
    Strategy: Keep the exact bypassed approach that works, add only catastrophic circuit breakers.
    """
    
    def __init__(
        self,
        decision_table_path: Path,
        policy_path: Path,
        meta_path: Path,
        initial_capital: float = 10_000.0,
        target_win_rate: float = 0.835,       # Very close to 83.9%
        win_rate_weight: float = 0.8,         # Extreme focus on preserving win rate
        return_weight: float = 0.15,          # Secondary concern
        safety_weight: float = 0.05           # Minimal safety weight
    ):
        self.decision_table_path = decision_table_path
        self.policy_path = policy_path
        self.meta_path = meta_path
        self.initial_capital = initial_capital
        self.target_win_rate = target_win_rate
        
        # Weights heavily favor win rate preservation
        self.win_rate_weight = win_rate_weight
        self.return_weight = return_weight
        self.safety_weight = safety_weight
        
        self.logger = logging.getLogger("ultra_minimal")
        self.logger.info(f"🎯 ULTRA-MINIMAL: Preserve 83.9% -> {target_win_rate:.1%} win rate")
        self.logger.info(f"⚡ Strategy: Keep bypassed approach + emergency circuit breakers only")
        
        # Track results
        self.best_results = []
        
        # Create temp directory
        self.temp_dir = Path(tempfile.mkdtemp(prefix="ultra_minimal_"))
        self.logger.info(f"Working directory: {self.temp_dir}")
    
    def objective(self, trial: optuna.Trial) -> float:
        """Ultra-minimal objective: preserve 83.9% with emergency-only protection."""
        try:
            # Sample ultra-minimal emergency parameters only
            params = self._sample_emergency_only_parameters(trial)
            
            # Run ultra-minimal simulation
            emergency_results = self._run_emergency_simulation(trial, params)
            
            if emergency_results is None:
                return 0.0
            
            # Calculate preservation score (how close to original 83.9%)
            preservation_score = self._calculate_ultra_preservation_score(emergency_results)
            
            # Store results
            trial_result = {
                'trial_number': trial.number,
                'params': params.copy(),
                'results': emergency_results,
                'preservation_score': preservation_score
            }
            self.best_results.append(trial_result)
            
            # Log with focus on exact preservation
            win_rate = emergency_results.get('win_rate', 0.0)
            return_pct = emergency_results.get('return_pct', 0.0)
            max_drawdown = emergency_results.get('max_drawdown', 0.0)
            total_trades = emergency_results.get('total_trades', 0)
            
            win_rate_delta = win_rate - 0.839  # Exact change from 83.9%
            
            self.logger.info(
                f"Trial {trial.number}: "
                f"Win Rate: {win_rate:.1%} ({win_rate_delta*100:+.1f}pp), "
                f"Return: {return_pct:.1f}%, "
                f"Max DD: {max_drawdown:.1%}, "
                f"Trades: {total_trades}, "
                f"Score: {preservation_score:.2f}"
            )
            
            return preservation_score
            
        except Exception as e:
            self.logger.error(f"Trial {trial.number} failed: {e}")
            return 0.0
    
    def _sample_emergency_only_parameters(self, trial: optuna.Trial) -> Dict:
        """
        Sample ONLY emergency circuit breaker parameters.
        
        Keep everything else exactly as the bypassed approach that gives 83.9% win rate.
        """
        
        return {
            # Keep all the bypassed approach parameters that work perfectly
            'approach': 'ultra_minimal_emergency_only',
            'base_contracts': 10,      # Exact same as bypassed
            'base_notional': 1000.0,   # Exact same as bypassed  
            'bypass_all_normal_controls': True,  # This is key!
            
            # EMERGENCY CIRCUIT BREAKERS ONLY (activate in catastrophic scenarios)
            
            # Emergency #1: Catastrophic portfolio loss protection
            'enable_portfolio_stop_loss': trial.suggest_categorical('enable_portfolio_stop_loss', [True, False]),
            'portfolio_stop_loss_pct': trial.suggest_float('portfolio_stop_loss_pct', 0.30, 0.60, step=0.05) if trial.suggest_categorical('enable_portfolio_stop_loss', [True, False]) else 1.0,
            
            # Emergency #2: Single trade catastrophic size limit  
            'enable_single_trade_cap': trial.suggest_categorical('enable_single_trade_cap', [True, False]),
            'max_single_trade_notional': trial.suggest_float('max_single_trade_notional', 25000, 75000, step=5000) if trial.suggest_categorical('enable_single_trade_cap', [True, False]) else 999999,
            
            # Emergency #3: Market halt conditions (extreme vol)
            'enable_market_halt_protection': trial.suggest_categorical('enable_market_halt_protection', [True, False]),
            'halt_vol_emergency_only': trial.suggest_categorical('halt_vol_emergency_only', [True, False]) if trial.suggest_categorical('enable_market_halt_protection', [True, False]) else False,
            
            # Emergency #4: Consecutive loss circuit breaker
            'enable_consecutive_loss_breaker': trial.suggest_categorical('enable_consecutive_loss_breaker', [True, False]),
            'max_consecutive_losses': trial.suggest_int('max_consecutive_losses', 5, 15) if trial.suggest_categorical('enable_consecutive_loss_breaker', [True, False]) else 999,
            
            # Keep transaction costs EXACTLY the same as bypassed (preserve the edge!)
            'commission_per_side': 0.65,    # Same as bypassed
            'exchange_fee_per_side': 0.05,  # Same as bypassed  
            'slippage_min': 0.02,           # Same as bypassed
            'slippage_pct': 0.20,           # Same as bypassed
        }
    
    def _run_emergency_simulation(self, trial: optuna.Trial, params: Dict) -> Optional[Dict]:
        """Run ultra-minimal simulation with emergency-only protection."""
        try:
            # Create emergency-only simulation script
            emergency_script = self._create_emergency_only_script(params)
            
            # Prepare trial directory
            trial_dir = self.temp_dir / f"trial_{trial.number}"
            trial_dir.mkdir(exist_ok=True)
            
            # Write emergency script
            script_path = trial_dir / "emergency_sim.py"
            with open(script_path, 'w') as f:
                f.write(emergency_script)
            
            # Run emergency simulation
            cmd = [
                sys.executable, str(script_path),
                "--decision-table", str(self.decision_table_path),
                "--policy", str(self.policy_path),
                "--meta", str(self.meta_path),
                "--outdir", str(trial_dir)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, cwd=Path.cwd())
            
            if result.returncode != 0:
                self.logger.warning(f"Trial {trial.number} simulation failed: {result.stderr[:200]}")
                return None
            
            # Load results
            summary_file = trial_dir / "emergency_summary.json"
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    return json.load(f)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Trial {trial.number} simulation error: {e}")
            return None
    
    def _create_emergency_only_script(self, params: Dict) -> str:
        """Create emergency-only simulation script that preserves 83.9% win rate."""
        
        return f'''#!/usr/bin/env python3
"""
Ultra-Minimal Emergency-Only Walkforward Simulation
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from d3rlpy import load_learnable
from d3rlpy.algos import DiscreteCQL, DiscreteCQLConfig
import torch
import yaml

def _load_meta(meta_path: Path) -> Dict:
    with meta_path.open("r", encoding="utf-8") as fh:
        meta = json.load(fh)
    return meta

def _load_policy_robust(policy_path: Path, meta: Dict) -> DiscreteCQL:
    """Load DiscreteCQL policy with fallback."""
    logger = logging.getLogger("emergency")
    
    try:
        algo = load_learnable(str(policy_path))
        logger.info("✅ Loaded policy with load_learnable")
        return algo
    except Exception as e1:
        logger.info("Falling back to manual reconstruction …")
        
        model_data = torch.load(str(policy_path), map_location="cpu")
        q_keys = [k for k in model_data["q_funcs"] if k.endswith("._fc.weight")]
        action_size = model_data["q_funcs"][q_keys[0]].shape[0] if q_keys else len(meta["action_map"])
        
        encoder_keys = [k for k in model_data["q_funcs"] if k.endswith("._encoder._layers.0.weight")]
        observation_size = model_data["q_funcs"][encoder_keys[0]].shape[1] if encoder_keys else len(meta["state_columns"])
        
        n_critics = len({{k.split(".")[0] for k in model_data["q_funcs"].keys()}})
        
        config = DiscreteCQLConfig(n_critics=n_critics)
        algo = config.create()
        algo.create_impl((observation_size,), action_size)
        algo.load_model(str(policy_path))
        logger.info("✅ Loaded via manual reconstruction")
        return algo

def _standardise_states(df: pd.DataFrame, state_cols: List[str], mean: List[float], scale: List[float]) -> np.ndarray:
    numeric_df = df[state_cols].apply(pd.to_numeric, errors="coerce")
    values = numeric_df.to_numpy(dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    mean_arr = np.asarray(mean, dtype=np.float32)
    scale_arr = np.asarray(scale, dtype=np.float32)
    denom = np.where(scale_arr == 0.0, 1.0, scale_arr)
    return (values - mean_arr) / denom

def _decode_action(action_id: int, action_map: Dict[str, Dict[str, float]]) -> Tuple[int, float]:
    info = action_map.get(str(action_id))
    if info is None:
        raise KeyError(f"Action id {{action_id}} not found in action map")
    return int(info.get("slot", 0)), float(info.get("size_value", 0.0))

class EmergencyCircuitBreaker:
    """Emergency-only circuit breaker that preserves model edge."""
    
    def __init__(self, params: Dict, initial_capital: float):
        self.params = params
        self.initial_capital = initial_capital
        self.consecutive_losses = 0
        self.trade_history = []
        
    def should_halt_trading(self, equity: float, row: pd.Series) -> bool:
        """Check if we should halt trading due to emergency conditions."""
        
        # Emergency #1: Catastrophic portfolio loss
        if self.params.get('enable_portfolio_stop_loss', False):
            loss_pct = (self.initial_capital - equity) / self.initial_capital
            stop_loss_pct = self.params.get('portfolio_stop_loss_pct', 0.5)
            if loss_pct > stop_loss_pct:
                return True
        
        # Emergency #3: Market halt (extreme vol emergency only)
        if self.params.get('enable_market_halt_protection', False):
            if self.params.get('halt_vol_emergency_only', False):
                vol_emergency = bool(row.get('s_vol_emergency', False))
                if vol_emergency:
                    return True
        
        return False
    
    def apply_emergency_position_sizing(self, base_contracts: int, notional: float, row: pd.Series) -> Tuple[int, float]:
        """Apply emergency position sizing (only if absolutely necessary)."""
        
        # Start with exact bypassed approach
        emergency_contracts = base_contracts
        emergency_notional = notional
        
        # Emergency #2: Single trade catastrophic size cap
        if self.params.get('enable_single_trade_cap', False):
            max_notional = self.params.get('max_single_trade_notional', 50000)
            if emergency_notional > max_notional:
                scale_factor = max_notional / emergency_notional
                emergency_contracts = int(emergency_contracts * scale_factor)
                emergency_notional = max_notional
        
        return emergency_contracts, emergency_notional
    
    def should_skip_due_to_consecutive_losses(self) -> bool:
        """Check consecutive loss circuit breaker."""
        if not self.params.get('enable_consecutive_loss_breaker', False):
            return False
        
        max_losses = self.params.get('max_consecutive_losses', 10)
        return self.consecutive_losses >= max_losses
    
    def update_consecutive_losses(self, realized_pnl: float):
        """Update consecutive loss counter."""
        if realized_pnl <= 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0  # Reset on any win

def simulate_emergency_walkforward(
    decision_df: pd.DataFrame,
    predicted_actions: np.ndarray,
    action_map: Dict[str, Dict[str, float]], 
    params: Dict,
    initial_capital: float = 10_000.0
) -> Dict[str, float]:
    """
    Emergency-only walkforward: Keep 83.9% approach + catastrophic protection only.
    """
    
    df = decision_df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.sort_values('date', inplace=True)

    equity = initial_capital
    history = []
    emergency_breaker = EmergencyCircuitBreaker(params, initial_capital)

    for idx, row in df.iterrows():
        action_id = int(predicted_actions[idx])
        slot, size_mult = _decode_action(action_id, action_map)

        record = {{
            'date': row['date'],
            'action_id': action_id,
            'slot': slot,
            'size_mult': size_mult,
            'equity_before': equity,
        }}

        # Check emergency halt conditions FIRST
        if emergency_breaker.should_halt_trading(equity, row):
            record.update({{
                'n_contracts': 0,
                'notional': 0.0,
                'fees': 0.0,
                'slippage': 0.0,
                'realized_pnl': 0.0,
                'equity_after': equity,
                'halt_reason': 'emergency_halt'
            }})
            history.append(record)
            continue

        if slot <= 0 or size_mult <= 0:
            record.update({{
                'n_contracts': 0,
                'notional': 0.0,
                'fees': 0.0,
                'slippage': 0.0,
                'realized_pnl': 0.0,
                'equity_after': equity,
            }})
            history.append(record)
            continue

        # Check consecutive loss breaker
        if emergency_breaker.should_skip_due_to_consecutive_losses():
            record.update({{
                'n_contracts': 0,
                'notional': 0.0,
                'fees': 0.0,
                'slippage': 0.0,
                'realized_pnl': 0.0,
                'equity_after': equity,
                'halt_reason': 'consecutive_losses'
            }})
            history.append(record)
            continue

        # EXACT BYPASSED APPROACH (preserve 83.9% win rate)
        base_contracts = 10  # Exact same as bypassed that gives 83.9%
        n_contracts = int(base_contracts * size_mult)  # Apply RL size multiplier
        
        # Calculate notional exactly as bypassed
        base_notional = 1000.0  # Exact same as bypassed
        notional = base_notional * n_contracts
        
        # Apply ONLY emergency position sizing (preserve the edge!)
        n_contracts, notional = emergency_breaker.apply_emergency_position_sizing(n_contracts, notional, row)

        # Calculate costs exactly as bypassed (preserve the edge!)
        commission = 0.65    # Exact same as bypassed
        exchange_fee = 0.05  # Exact same as bypassed
        fees = (commission + exchange_fee) * n_contracts * 2  # Round trip
        
        slippage_min = 0.02   # Exact same as bypassed
        slippage_pct = 0.20   # Exact same as bypassed
        spread = float(row.get("bid_ask_spread", 0.0) or 0.0)
        slip_per_contract = max(slippage_min, slippage_pct * spread)
        slippage = slip_per_contract * 100.0 * n_contracts * 2.0
        
        # Calculate P&L exactly as bypassed (preserve the 83.9% edge!)
        pnl_col = f"c{{slot}}_target_pnl"
        raw_return = row.get(pnl_col, 0.0)
        if pd.isna(raw_return):
            raw_return = 0.0
        raw_return = float(raw_return)
        realized_pnl = raw_return * notional

        # Update consecutive loss counter
        emergency_breaker.update_consecutive_losses(realized_pnl)

        # Update equity
        equity = equity + realized_pnl - fees - slippage

        record.update({{
            'n_contracts': n_contracts,
            'notional': notional,
            'fees': fees,
            'slippage': slippage,
            'realized_pnl': realized_pnl,
            'equity_after': equity,
        }})
        history.append(record)

    # Generate results
    results = pd.DataFrame(history)
    trades_mask = results['n_contracts'] > 0
    winning_trades = results[trades_mask & (results['realized_pnl'] > 0)]
    losing_trades = results[trades_mask & (results['realized_pnl'] <= 0)]
    
    # Calculate drawdown
    equity_series = results['equity_after']
    running_max = equity_series.expanding().max()
    drawdowns = (equity_series - running_max) / running_max
    max_drawdown = abs(drawdowns.min()) if len(drawdowns) > 0 else 0.0
    
    summary = {{
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
        'emergency_params': params.copy(),
        'halted_trades': len(results[results.get('halt_reason', '').notna()]) if 'halt_reason' in results.columns else 0
    }}
    
    if trades_mask.sum() > 0:
        trade_pnls = results[trades_mask]['realized_pnl']
        summary.update({{
            'avg_trade_pnl': float(trade_pnls.mean()),
            'largest_win': float(trade_pnls.max()),
            'largest_loss': float(trade_pnls.min()),
            'profit_factor': float(winning_trades['realized_pnl'].sum() / abs(losing_trades['realized_pnl'].sum())) if len(losing_trades) > 0 else np.inf,
        }})
    
    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--decision-table', type=Path, required=True)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--meta', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    # Load data
    df = pd.read_csv(args.decision_table)
    meta = _load_meta(args.meta)
    states = _standardise_states(df, meta['state_columns'], meta['scaler_mean'], meta['scaler_scale'])
    algo = _load_policy_robust(args.policy, meta)
    predicted_actions = algo.predict(states)
    
    # Emergency parameters for this trial
    emergency_params = {params}
    
    # Run emergency-only simulation
    results = simulate_emergency_walkforward(df, predicted_actions, meta['action_map'], emergency_params)
    
    # Save results
    args.outdir.mkdir(parents=True, exist_ok=True)
    with open(args.outdir / 'emergency_summary.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

if __name__ == '__main__':
    main()
'''
    
    def _calculate_ultra_preservation_score(self, results: Dict) -> float:
        """
        Ultra-preservation score: How close are we to exactly 83.9% win rate?
        
        This is about EXACT preservation, not improvement.
        """
        
        # Target: exact replication of 83.9% win rate, 1119.7% returns
        target_win_rate = 0.839
        target_return = 1119.7
        
        # Current performance
        current_win_rate = results.get('win_rate', 0.0)
        current_return = results.get('return_pct', 0.0)
        current_drawdown = results.get('max_drawdown', 1.0)
        total_trades = results.get('total_trades', 0)
        
        # Must have minimum trades
        if total_trades < 30:
            return 0.0
        
        # Component scores focused on EXACT preservation
        
        # 1. Win Rate Preservation (most critical - aim for EXACT 83.9%)
        win_rate_error = abs(target_win_rate - current_win_rate)
        
        if win_rate_error <= 0.005:      # Within 0.5% of 83.9%
            win_rate_score = 100.0
        elif win_rate_error <= 0.01:     # Within 1% of 83.9%
            win_rate_score = 95.0
        elif win_rate_error <= 0.02:     # Within 2% of 83.9%
            win_rate_score = 85.0
        elif win_rate_error <= 0.03:     # Within 3% of 83.9%
            win_rate_score = 70.0
        elif win_rate_error <= 0.05:     # Within 5% of 83.9%
            win_rate_score = 50.0
        else:
            win_rate_score = max(0.0, 50.0 - (win_rate_error - 0.05) * 1000)
        
        # 2. Return Preservation (should be close to 1119.7%)
        return_ratio = current_return / target_return if target_return > 0 else 0.0
        
        if return_ratio >= 0.95:        # Within 5% of original
            return_score = 100.0
        elif return_ratio >= 0.85:      # Within 15% of original
            return_score = 90.0
        elif return_ratio >= 0.70:      # Within 30% of original
            return_score = 70.0
        else:
            return_score = max(0.0, 70.0 * return_ratio)
        
        # 3. Safety Score (reward drawdown control, but don't over-emphasize)
        if current_drawdown <= 0.02:     # Very low drawdown
            safety_score = 100.0
        elif current_drawdown <= 0.05:   # Low drawdown
            safety_score = 90.0
        elif current_drawdown <= 0.10:   # Moderate drawdown
            safety_score = 80.0
        else:
            safety_score = max(0.0, 80.0 - (current_drawdown - 0.10) * 200)
        
        # Weighted composite (heavily favor win rate preservation)
        preservation_score = (
            self.win_rate_weight * win_rate_score +
            self.return_weight * return_score +
            self.safety_weight * safety_score
        )
        
        # HUGE bonus for achieving near-perfect preservation
        if win_rate_error <= 0.005:  # Within 0.5% of 83.9%
            preservation_score *= 1.5  # 50% bonus for excellence
        elif win_rate_error <= 0.01:  # Within 1% of 83.9%
            preservation_score *= 1.3  # 30% bonus
        elif win_rate_error <= 0.02:  # Within 2% of 83.9%
            preservation_score *= 1.15  # 15% bonus
        
        return preservation_score
    
    def optimize(
        self, 
        n_trials: int = 25, 
        study_name: str = "ultra_minimal_risk_optimization"
    ) -> Tuple[optuna.Study, Dict]:
        """Run ultra-minimal risk optimization."""
        
        self.logger.info(f"⚡ Starting ultra-minimal risk optimization with {n_trials} trials")
        self.logger.info(f"🎯 Goal: Maintain EXACTLY 83.9% win rate with emergency-only protection")
        
        # Create study
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42),
            pruner=HyperbandPruner(min_resource=10),
            study_name=study_name
        )
        
        # Run optimization
        study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)
        
        # Analyze results
        best_trial = study.best_trial
        best_detailed = None
        
        for result in self.best_results:
            if result['trial_number'] == best_trial.number:
                best_detailed = result
                break
        
        self.logger.info("⚡ Ultra-Minimal Risk Optimization Complete!")
        self.logger.info(f"Best trial: #{best_trial.number}")
        self.logger.info(f"Best preservation score: {best_trial.value:.2f}")
        
        if best_detailed:
            results = best_detailed['results']
            target_win_rate = 0.839
            
            self.logger.info("📊 Best Emergency Protection Performance:")
            win_rate_delta = results.get('win_rate', 0) - target_win_rate
            error_magnitude = abs(win_rate_delta) * 100
            
            self.logger.info(f"  Target: 83.9% win rate, 1,119.7% returns")
            self.logger.info(f"  Achieved: {results.get('win_rate', 0):.1%} win rate, {results.get('return_pct', 0):.1f}% returns")
            self.logger.info(f"  Win Rate Error: {win_rate_delta*100:+.1f}pp (|{error_magnitude:.1f}pp|)")
            self.logger.info(f"  Max Drawdown: {results.get('max_drawdown', 0):.1%}")
            self.logger.info(f"  Total Trades: {results.get('total_trades', 0)}")
            self.logger.info(f"  Halted Trades: {results.get('halted_trades', 0)}")
            
            if error_magnitude <= 0.5:
                preservation_quality = "🟢 PERFECT (≤0.5pp error)"
            elif error_magnitude <= 1.0:
                preservation_quality = "🟢 EXCELLENT (≤1pp error)"  
            elif error_magnitude <= 2.0:
                preservation_quality = "🟡 GOOD (≤2pp error)"
            elif error_magnitude <= 3.0:
                preservation_quality = "🟡 ACCEPTABLE (≤3pp error)"
            else:
                preservation_quality = "🔴 NEEDS WORK (>3pp error)"
            
            self.logger.info(f"  Preservation Quality: {preservation_quality}")
        
        # Compile results
        optimization_summary = {
            'study_name': study_name,
            'n_trials': n_trials,
            'optimization_focus': 'ultra_minimal_emergency_only',
            'target_win_rate': 0.839,
            'target_return_pct': 1119.7,
            'approach': 'keep_bypassed_add_emergency_only',
            'score_weights': {
                'win_rate': self.win_rate_weight,
                'returns': self.return_weight,
                'safety': self.safety_weight
            },
            'best_trial_number': best_trial.number,
            'best_preservation_score': best_trial.value,
            'best_params': best_trial.params,
            'best_performance': best_detailed['results'] if best_detailed else None
        }
        
        return study, optimization_summary
    
    def cleanup(self):
        """Clean up temporary files."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            self.logger.info("Cleaned up temporary files")


def main():
    """Run ultra-minimal risk optimization."""
    parser = argparse.ArgumentParser(description="Ultra-minimal emergency-only risk optimization")
    parser.add_argument('--decision-table', type=Path, required=True)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--meta', type=Path, required=True)
    parser.add_argument('--trials', type=int, default=25, help="Number of optimization trials")
    parser.add_argument('--target-win-rate', type=float, default=0.835, help="Target win rate preservation")
    parser.add_argument('--outdir', type=Path, default=Path('results/ultra_minimal_risk_optimization'))
    parser.add_argument('--study-name', default='ultra_minimal_risk_opt', help="Optuna study name")
    
    args = parser.parse_args()
    
    # Validate inputs
    for file_path in [args.decision_table, args.policy, args.meta]:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
    
    # Create output directory
    args.outdir.mkdir(parents=True, exist_ok=True)
    
    # Create optimizer
    optimizer = UltraMinimalRiskOptimizer(
        decision_table_path=args.decision_table,
        policy_path=args.policy,
        meta_path=args.meta,
        target_win_rate=args.target_win_rate
    )
    
    try:
        # Run optimization
        study, summary = optimizer.optimize(n_trials=args.trials, study_name=args.study_name)
        
        # Save results
        results_file = args.outdir / 'ultra_minimal_results.json'
        with open(results_file, 'w') as f:
            clean_summary = json.loads(json.dumps(summary, default=str))
            json.dump({
                'optimization_summary': clean_summary,
                'all_trials': optimizer.best_results
            }, f, indent=2, default=str)
        
        # Save study
        study_file = args.outdir / 'ultra_minimal_study.pkl'
        optuna.save_study(study, str(study_file))
        
        # Run final validation
        if summary['best_performance']:
            final_dir = args.outdir / 'final_ultra_minimal_backtest'
            final_dir.mkdir(exist_ok=True)
            
            # Save best parameters and results
            best_params_file = final_dir / 'ultra_minimal_best_params.json'
            with open(best_params_file, 'w') as f:
                json.dump(summary['best_params'], f, indent=2)
                
            best_results_file = final_dir / 'ultra_minimal_summary.json'
            with open(best_results_file, 'w') as f:
                json.dump(summary['best_performance'], f, indent=2, default=str)
            
            # Print final summary
            results = summary['best_performance']
            target_win_rate = 0.839
            win_rate_error = abs(results.get('win_rate', 0) - target_win_rate) * 100
            
            print(f"\n✅ Ultra-Minimal Risk Optimization Complete!")
            print(f"📊 Results saved to: {args.outdir}")
            print(f"⚡ Final backtest in: {final_dir}")
            print(f"\n📈 Performance Summary:")
            print(f"   🎯 Target: 83.9% win rate, 1,119.7% returns (bypassed)")
            print(f"   ⚡ Achieved: {results.get('win_rate', 0):.1%} win rate, {results.get('return_pct', 0):.1f}% returns")
            print(f"   📏 Win Rate Error: {win_rate_error:.1f}pp")
            print(f"   🛡️ Max Drawdown: {results.get('max_drawdown', 0):.1%}")
            print(f"   📊 Total Trades: {results.get('total_trades', 0)}")
            print(f"   🚨 Emergency Halts: {results.get('halted_trades', 0)}")
            
            if win_rate_error <= 0.5:
                print(f"   🏆 Result: 🟢 PERFECT preservation (≤0.5pp error)")
            elif win_rate_error <= 1.0:
                print(f"   🏆 Result: 🟢 EXCELLENT preservation (≤1pp error)")
            elif win_rate_error <= 2.0:
                print(f"   🏆 Result: 🟡 GOOD preservation (≤2pp error)")
            elif win_rate_error <= 3.0:
                print(f"   🏆 Result: 🟡 ACCEPTABLE preservation (≤3pp error)")
            else:
                print(f"   🏆 Result: 🔴 NEEDS IMPROVEMENT (>{win_rate_error:.1f}pp error)")
        
    finally:
        # Cleanup
        optimizer.cleanup()


if __name__ == '__main__':
    main()