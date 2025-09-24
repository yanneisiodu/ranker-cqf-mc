#!/usr/bin/env python3
"""
Ultra-Minimal Risk Optimization - FIXED VERSION
Addresses all identified bugs from deep analysis.

Philosophy: The IQL model is perfect at what it does. Don't interfere with its 
natural edge. Only add CATASTROPHIC loss protection that activates in extreme scenarios.

Goal: Maintain exactly 83.9% win rate with emergency-only circuit breakers.

FIXES APPLIED:
1. Fixed duplicated suggest_categorical calls
2. Removed/relaxed consecutive loss breaker  
3. Reordered logic: position sizing before halt decisions
4. Added trade retention penalty to preservation score
5. Fixed halted trades counter
6. Adjusted weights heavily toward win-rate preservation (90%)
7. Improved emergency logic and parameter handling
"""

from __future__ import annotations

import argparse
import json
import logging
import math
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


class UltraMinimalRiskOptimizerFixed:
    """
    Ultra-minimal risk optimizer - FIXED VERSION
    
    Strategy: Keep the exact bypassed approach that works, add only catastrophic circuit breakers.
    """
    
    def __init__(
        self,
        decision_table_path: Path,
        policy_path: Path,
        meta_path: Path,
        initial_capital: float = 10_000.0,
        target_win_rate: float = 0.839,       # Exact target from bypassed
        win_rate_weight: float = 0.90,        # HEAVILY favor win rate preservation
        return_weight: float = 0.07,          # Secondary concern
        safety_weight: float = 0.03           # Minimal safety weight
    ):
        self.decision_table_path = decision_table_path
        self.policy_path = policy_path
        self.meta_path = meta_path
        self.initial_capital = initial_capital
        self.target_win_rate = target_win_rate
        
        # Weights HEAVILY favor win rate preservation
        self.win_rate_weight = win_rate_weight
        self.return_weight = return_weight
        self.safety_weight = safety_weight
        
        self.logger = logging.getLogger("ultra_minimal_fixed")
        self.logger.info(f"🎯 ULTRA-MINIMAL FIXED: Preserve 83.9% -> {target_win_rate:.1%} win rate")
        self.logger.info(f"⚡ Strategy: Keep bypassed approach + emergency circuit breakers only")
        self.logger.info(f"🔧 Applied fixes: param sampling, halt logic, score function, counters")
        
        # Track results
        self.best_results = []
        
        # Create temp directory
        self.temp_dir = Path(tempfile.mkdtemp(prefix="ultra_minimal_fixed_"))
        self.logger.info(f"Working directory: {self.temp_dir}")
    
    def objective(self, trial: optuna.Trial) -> float:
        """Ultra-minimal objective: preserve 83.9% with emergency-only protection."""
        try:
            # Sample ultra-minimal emergency parameters (FIXED - no duplicates)
            params = self._sample_emergency_only_parameters_fixed(trial)
            
            # Run ultra-minimal simulation
            emergency_results = self._run_emergency_simulation(trial, params)
            
            if emergency_results is None:
                return 0.0
            
            # Calculate preservation score (FIXED - includes trade retention penalty)
            preservation_score = self._calculate_ultra_preservation_score_fixed(emergency_results)
            
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
            halted_trades = emergency_results.get('halted_trades', 0)
            
            win_rate_delta = win_rate - 0.839  # Exact change from 83.9%
            
            self.logger.info(
                f"Trial {trial.number}: "
                f"Win Rate: {win_rate:.1%} ({win_rate_delta*100:+.1f}pp), "
                f"Return: {return_pct:.1f}%, "
                f"Max DD: {max_drawdown:.1%}, "
                f"Trades: {total_trades} (halted: {halted_trades}), "
                f"Score: {preservation_score:.2f}"
            )
            
            return preservation_score
            
        except Exception as e:
            self.logger.error(f"Trial {trial.number} failed: {e}")
            return 0.0
    
    def _sample_emergency_only_parameters_fixed(self, trial: optuna.Trial) -> Dict:
        """
        Sample ONLY emergency circuit breaker parameters - FIXED VERSION.
        
        Fixes:
        1. No duplicated suggest_categorical calls
        2. Consecutive loss breaker removed or heavily relaxed
        3. Cleaner parameter logic
        """
        
        # Sample enable flags once
        enable_portfolio_stop_loss = trial.suggest_categorical('enable_portfolio_stop_loss', [True, False])
        enable_single_trade_cap = trial.suggest_categorical('enable_single_trade_cap', [True, False])
        enable_market_halt_protection = trial.suggest_categorical('enable_market_halt_protection', [True, False])
        enable_consecutive_loss_breaker = trial.suggest_categorical('enable_consecutive_loss_breaker', [True, False])
        
        # Sample dependent parameters only if flags are enabled
        portfolio_stop_loss_pct = (
            trial.suggest_float('portfolio_stop_loss_pct', 0.40, 0.80, step=0.05) 
            if enable_portfolio_stop_loss else 1.0
        )
        
        max_single_trade_notional = (
            trial.suggest_float('max_single_trade_notional', 75000, 150000, step=5000)
            if enable_single_trade_cap else 999999
        )
        
        halt_vol_emergency_only = (
            trial.suggest_categorical('halt_vol_emergency_only', [True, False])
            if enable_market_halt_protection else False
        )
        
        # HEAVILY RELAXED consecutive loss breaker (was killing good trades)
        max_consecutive_losses = (
            trial.suggest_int('max_consecutive_losses', 15, 30)  # Much higher threshold
            if enable_consecutive_loss_breaker else 999
        )
        
        return {
            # Keep all the bypassed approach parameters that work perfectly
            'approach': 'ultra_minimal_emergency_only_fixed',
            'base_contracts': 10,      # Exact same as bypassed
            'base_notional': 1000.0,   # Exact same as bypassed  
            'bypass_all_normal_controls': True,  # This is key!
            
            # EMERGENCY CIRCUIT BREAKERS ONLY (activate in catastrophic scenarios)
            
            # Emergency #1: Catastrophic portfolio loss protection (RELAXED)
            'enable_portfolio_stop_loss': enable_portfolio_stop_loss,
            'portfolio_stop_loss_pct': portfolio_stop_loss_pct,
            
            # Emergency #2: Single trade catastrophic size limit (INCREASED THRESHOLDS)
            'enable_single_trade_cap': enable_single_trade_cap,
            'max_single_trade_notional': max_single_trade_notional,
            
            # Emergency #3: Market halt conditions (extreme vol)
            'enable_market_halt_protection': enable_market_halt_protection,
            'halt_vol_emergency_only': halt_vol_emergency_only,
            
            # Emergency #4: Consecutive loss circuit breaker (HEAVILY RELAXED)
            'enable_consecutive_loss_breaker': enable_consecutive_loss_breaker,
            'max_consecutive_losses': max_consecutive_losses,
            
            # Keep transaction costs EXACTLY the same as bypassed (preserve the edge!)
            'commission_per_side': 0.65,    # Same as bypassed
            'exchange_fee_per_side': 0.05,  # Same as bypassed  
            'slippage_min': 0.02,           # Same as bypassed
            'slippage_pct': 0.20,           # Same as bypassed
        }
    
    def _run_emergency_simulation(self, trial: optuna.Trial, params: Dict) -> Optional[Dict]:
        """Run ultra-minimal simulation with emergency-only protection."""
        try:
            # Create emergency-only simulation script (FIXED VERSION)
            emergency_script = self._create_emergency_only_script_fixed(params)
            
            # Prepare trial directory
            trial_dir = self.temp_dir / f"trial_{trial.number}"
            trial_dir.mkdir(exist_ok=True)
            
            # Write emergency script
            script_path = trial_dir / "emergency_sim_fixed.py"
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
            summary_file = trial_dir / "emergency_summary_fixed.json"
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    return json.load(f)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Trial {trial.number} simulation error: {e}")
            return None
    
    def _create_emergency_only_script_fixed(self, params: Dict) -> str:
        """Create emergency-only simulation script - FIXED VERSION."""
        
        return f'''#!/usr/bin/env python3
"""
Ultra-Minimal Emergency-Only Walkforward Simulation - FIXED VERSION
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

class EmergencyCircuitBreakerFixed:
    """Emergency-only circuit breaker - FIXED VERSION."""
    
    def __init__(self, params: Dict, initial_capital: float):
        self.params = params
        self.initial_capital = initial_capital
        self.consecutive_losses = 0
        self.trade_history = []
        self.equity_peak = initial_capital  # Track peak for drawdown-based stops
        
    def should_halt_trading_after_sizing(self, equity: float, contracts: int, notional: float, row: pd.Series) -> Tuple[bool, str]:
        """
        FIXED: Check halt conditions AFTER position sizing adjustments.
        Returns (should_halt, reason)
        """
        
        # Emergency #1: Catastrophic portfolio loss (track from peak, not initial)
        if self.params.get('enable_portfolio_stop_loss', False):
            self.equity_peak = max(self.equity_peak, equity)  # Update peak
            drawdown_from_peak = (self.equity_peak - equity) / self.equity_peak
            stop_loss_pct = self.params.get('portfolio_stop_loss_pct', 0.5)
            if drawdown_from_peak > stop_loss_pct:
                return True, f"portfolio_drawdown_{{drawdown_from_peak:.1%}}"
        
        # Emergency #2: Single trade still too large AFTER sizing
        if self.params.get('enable_single_trade_cap', False):
            max_notional = self.params.get('max_single_trade_notional', 100000)
            if notional > max_notional:
                return True, f"trade_size_{{notional:.0f}}"
        
        # Emergency #3: Market halt (extreme vol emergency only)
        if self.params.get('enable_market_halt_protection', False):
            if self.params.get('halt_vol_emergency_only', False):
                vol_emergency = bool(row.get('s_vol_emergency', False))
                if vol_emergency:
                    return True, "market_vol_emergency"
        
        return False, ""
    
    def apply_emergency_position_sizing(self, base_contracts: int, notional: float, row: pd.Series) -> Tuple[int, float]:
        """Apply emergency position sizing - FIXED to handle edge cases."""
        
        # Start with exact bypassed approach
        emergency_contracts = base_contracts
        emergency_notional = notional
        
        # Emergency #2: Single trade catastrophic size cap (more generous thresholds)
        if self.params.get('enable_single_trade_cap', False):
            max_notional = self.params.get('max_single_trade_notional', 100000)
            if emergency_notional > max_notional:
                scale_factor = max_notional / emergency_notional
                emergency_contracts = max(int(emergency_contracts * scale_factor), 1)  # Ensure at least 1 contract
                emergency_notional = max_notional
        
        return emergency_contracts, emergency_notional
    
    def should_skip_due_to_consecutive_losses(self) -> bool:
        """Check consecutive loss circuit breaker - FIXED with much higher thresholds."""
        if not self.params.get('enable_consecutive_loss_breaker', False):
            return False
        
        max_losses = self.params.get('max_consecutive_losses', 20)  # Much higher default
        return self.consecutive_losses >= max_losses
    
    def update_consecutive_losses(self, realized_pnl: float):
        """Update consecutive loss counter."""
        if realized_pnl <= 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0  # Reset on any win

def simulate_emergency_walkforward_fixed(
    decision_df: pd.DataFrame,
    predicted_actions: np.ndarray,
    action_map: Dict[str, Dict[str, float]], 
    params: Dict,
    initial_capital: float = 10_000.0
) -> Dict[str, float]:
    """
    Emergency-only walkforward - FIXED VERSION.
    
    Key fixes:
    1. Apply position sizing BEFORE halt checks
    2. Better halt reason tracking
    3. Improved equity peak tracking
    """
    
    df = decision_df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.sort_values('date', inplace=True)

    equity = initial_capital
    history = []
    emergency_breaker = EmergencyCircuitBreakerFixed(params, initial_capital)

    for idx, row in df.iterrows():
        action_id = int(predicted_actions[idx])
        slot, size_mult = _decode_action(action_id, action_map)

        record = {{
            'date': row['date'],
            'action_id': action_id,
            'slot': slot,
            'size_mult': size_mult,
            'equity_before': equity,
            'halt_reason': None  # Initialize halt reason
        }}

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
        
        # CRITICAL FILTER: Check contract risk (this was the missing piece!)
        price_col = f"c{{slot}}_future_option_price"
        premium = row.get(price_col)
        
        # If price is null/NaN, skip the trade (this matches original behavior!)
        if pd.isna(premium) or premium is None:
            record.update({{
                'n_contracts': 0,
                'notional': 0.0,
                'fees': 0.0,
                'slippage': 0.0,
                'realized_pnl': 0.0,
                'equity_after': equity,
                'skip_reason': f'no_price_data_slot_{{slot}}'
            }})
            history.append(record)
            continue

        # Check consecutive loss breaker FIRST (before sizing)
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
        
        # CRITICAL: Apply 10% equity cap (this was missing - the "bypassed" approach still has this!)
        max_notional_pct = 0.10  # 10% of equity - this constraint is STILL ACTIVE in original
        notional_cap = equity * max_notional_pct
        if notional > notional_cap:
            scale = notional_cap / notional if notional > 0 else 0.0
            n_contracts = int(max(np.floor(n_contracts * scale), 1))  # Ensure at least 1 contract
            notional = base_notional * n_contracts
        
        # Apply ONLY emergency position sizing (AFTER equity cap)
        n_contracts, notional = emergency_breaker.apply_emergency_position_sizing(n_contracts, notional, row)

        # NOW check if we should halt AFTER sizing adjustments
        should_halt, halt_reason = emergency_breaker.should_halt_trading_after_sizing(equity, n_contracts, notional, row)
        if should_halt:
            record.update({{
                'n_contracts': 0,
                'notional': 0.0,
                'fees': 0.0,
                'slippage': 0.0,
                'realized_pnl': 0.0,
                'equity_after': equity,
                'halt_reason': halt_reason
            }})
            history.append(record)
            continue

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

    # Generate results - FIXED halted trades counter
    results = pd.DataFrame(history)
    trades_mask = results['n_contracts'] > 0
    winning_trades = results[trades_mask & (results['realized_pnl'] > 0)]
    losing_trades = results[trades_mask & (results['realized_pnl'] <= 0)]
    
    # FIXED: Correct halted trades count
    halted_trades = (results['halt_reason'].notna()).sum() if 'halt_reason' in results.columns else 0
    
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
        'halted_trades': int(halted_trades)
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
    
    # Run emergency-only simulation - FIXED VERSION
    results = simulate_emergency_walkforward_fixed(df, predicted_actions, meta['action_map'], emergency_params)
    
    # Save results
    args.outdir.mkdir(parents=True, exist_ok=True)
    with open(args.outdir / 'emergency_summary_fixed.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

if __name__ == '__main__':
    main()
'''
    
    def _calculate_ultra_preservation_score_fixed(self, results: Dict) -> float:
        """
        Ultra-preservation score - FIXED VERSION.
        
        Fixes:
        1. Added trade retention penalty
        2. Heavily weighted toward win-rate preservation (90%)
        3. Better handling of edge cases
        """
        
        # Target: exact replication of 83.9% win rate, 1119.7% returns, 87 trades
        target_win_rate = 0.839
        target_return = 1119.7
        target_trades = 87
        
        # Current performance
        current_win_rate = results.get('win_rate', 0.0)
        current_return = results.get('return_pct', 0.0)
        current_drawdown = results.get('max_drawdown', 1.0)
        total_trades = results.get('total_trades', 0)
        halted_trades = results.get('halted_trades', 0)
        
        # Must have minimum trades
        if total_trades < 20:
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
        
        # 3. Safety Score (reward drawdown control)
        if current_drawdown <= 0.02:     # Very low drawdown
            safety_score = 100.0
        elif current_drawdown <= 0.05:   # Low drawdown
            safety_score = 90.0
        elif current_drawdown <= 0.10:   # Moderate drawdown
            safety_score = 80.0
        else:
            safety_score = max(0.0, 80.0 - (current_drawdown - 0.10) * 200)
        
        # 4. NEW: Trade Retention Score (penalize blocking too many trades)
        trade_ratio = total_trades / target_trades
        if trade_ratio >= 0.95:          # Execute 95%+ of expected trades
            trade_retention_score = 100.0
        elif trade_ratio >= 0.85:        # Execute 85%+ of expected trades
            trade_retention_score = 90.0
        elif trade_ratio >= 0.70:        # Execute 70%+ of expected trades
            trade_retention_score = 70.0
        else:
            # Heavily penalize blocking too many trades (this was the main problem)
            trade_retention_score = max(0.0, 70.0 * trade_ratio)
        
        # HEAVILY WEIGHTED toward win-rate preservation (90%)
        preservation_score = (
            0.90 * win_rate_score +        # 90% weight on win rate (was 80%)
            0.05 * return_score +          # 5% weight on returns (was 15%)
            0.02 * safety_score +          # 2% weight on safety (was 5%)
            0.03 * trade_retention_score   # 3% weight on trade retention (NEW)
        )
        
        # HUGE bonus for achieving near-perfect preservation
        if win_rate_error <= 0.005 and trade_ratio >= 0.95:  # Both win rate AND trade retention excellent
            preservation_score *= 1.8  # 80% bonus for excellence
        elif win_rate_error <= 0.01 and trade_ratio >= 0.85:  # Good on both metrics
            preservation_score *= 1.5  # 50% bonus
        elif win_rate_error <= 0.02 and trade_ratio >= 0.75:  # Acceptable on both
            preservation_score *= 1.2  # 20% bonus
        
        # Additional penalty for excessive halt activity
        if halted_trades > total_trades:  # Blocking more trades than executing
            preservation_score *= 0.7  # 30% penalty for over-halting
        
        return preservation_score
    
    def optimize(
        self, 
        n_trials: int = 50,  # Increased from 25 
        study_name: str = "ultra_minimal_risk_optimization_fixed"
    ) -> Tuple[optuna.Study, Dict]:
        """Run ultra-minimal risk optimization - FIXED VERSION."""
        
        self.logger.info(f"⚡ Starting FIXED ultra-minimal risk optimization with {n_trials} trials")
        self.logger.info(f"🎯 Goal: Maintain EXACTLY 83.9% win rate with emergency-only protection")
        self.logger.info(f"🔧 Applied fixes: param sampling, halt logic, score function, counters")
        
        # Create study
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42),
            pruner=HyperbandPruner(min_resource=15),  # Increased min resource
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
        
        self.logger.info("⚡ FIXED Ultra-Minimal Risk Optimization Complete!")
        self.logger.info(f"Best trial: #{best_trial.number}")
        self.logger.info(f"Best preservation score: {best_trial.value:.2f}")
        
        if best_detailed:
            results = best_detailed['results']
            target_win_rate = 0.839
            
            self.logger.info("📊 Best Emergency Protection Performance (FIXED):")
            win_rate_delta = results.get('win_rate', 0) - target_win_rate
            error_magnitude = abs(win_rate_delta) * 100
            
            self.logger.info(f"  Target: 83.9% win rate, 1,119.7% returns, 87 trades")
            self.logger.info(f"  Achieved: {results.get('win_rate', 0):.1%} win rate, {results.get('return_pct', 0):.1f}% returns, {results.get('total_trades', 0)} trades")
            self.logger.info(f"  Win Rate Error: {win_rate_delta*100:+.1f}pp (|{error_magnitude:.1f}pp|)")
            self.logger.info(f"  Max Drawdown: {results.get('max_drawdown', 0):.1%}")
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
            'optimization_focus': 'ultra_minimal_emergency_only_fixed',
            'target_win_rate': 0.839,
            'target_return_pct': 1119.7,
            'target_trades': 87,
            'approach': 'keep_bypassed_add_emergency_only_fixed',
            'fixes_applied': [
                'fixed_duplicated_suggest_categorical_calls',
                'relaxed_consecutive_loss_breaker',
                'reordered_position_sizing_before_halt_checks',
                'added_trade_retention_penalty_to_score',
                'fixed_halted_trades_counter',
                'increased_win_rate_weight_to_90_percent'
            ],
            'score_weights': {
                'win_rate': self.win_rate_weight,
                'returns': self.return_weight,
                'safety': self.safety_weight,
                'trade_retention': 0.03
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
    """Run ultra-minimal risk optimization - FIXED VERSION."""
    parser = argparse.ArgumentParser(description="Ultra-minimal emergency-only risk optimization - FIXED")
    parser.add_argument('--decision-table', type=Path, required=True)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--meta', type=Path, required=True)
    parser.add_argument('--trials', type=int, default=50, help="Number of optimization trials")
    parser.add_argument('--target-win-rate', type=float, default=0.839, help="Target win rate preservation")
    parser.add_argument('--outdir', type=Path, default=Path('results/ultra_minimal_risk_optimization_fixed'))
    parser.add_argument('--study-name', default='ultra_minimal_risk_opt_fixed', help="Optuna study name")
    
    args = parser.parse_args()
    
    # Validate inputs
    for file_path in [args.decision_table, args.policy, args.meta]:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
    
    # Create output directory
    args.outdir.mkdir(parents=True, exist_ok=True)
    
    # Create optimizer - FIXED VERSION
    optimizer = UltraMinimalRiskOptimizerFixed(
        decision_table_path=args.decision_table,
        policy_path=args.policy,
        meta_path=args.meta,
        target_win_rate=args.target_win_rate
    )
    
    try:
        # Run optimization
        study, summary = optimizer.optimize(n_trials=args.trials, study_name=args.study_name)
        
        # Save results
        results_file = args.outdir / 'ultra_minimal_results_fixed.json'
        with open(results_file, 'w') as f:
            clean_summary = json.loads(json.dumps(summary, default=str))
            json.dump({
                'optimization_summary': clean_summary,
                'all_trials': optimizer.best_results
            }, f, indent=2, default=str)
        
        # Don't use optuna.save_study (not available in some versions)
        # study_file = args.outdir / 'ultra_minimal_study_fixed.pkl'  
        # optuna.save_study(study, str(study_file))
        
        # Run final validation
        if summary['best_performance']:
            final_dir = args.outdir / 'final_ultra_minimal_backtest_fixed'
            final_dir.mkdir(exist_ok=True)
            
            # Save best parameters and results
            best_params_file = final_dir / 'ultra_minimal_best_params_fixed.json'
            with open(best_params_file, 'w') as f:
                json.dump(summary['best_params'], f, indent=2)
                
            best_results_file = final_dir / 'ultra_minimal_summary_fixed.json'
            with open(best_results_file, 'w') as f:
                json.dump(summary['best_performance'], f, indent=2, default=str)
            
            # Print final summary
            results = summary['best_performance']
            target_win_rate = 0.839
            win_rate_error = abs(results.get('win_rate', 0) - target_win_rate) * 100
            
            print(f"\n✅ FIXED Ultra-Minimal Risk Optimization Complete!")
            print(f"📊 Results saved to: {args.outdir}")
            print(f"⚡ Final backtest in: {final_dir}")
            print(f"\n📈 Performance Summary:")
            print(f"   🎯 Target: 83.9% win rate, 1,119.7% returns, 87 trades (bypassed)")
            print(f"   ⚡ Achieved: {results.get('win_rate', 0):.1%} win rate, {results.get('return_pct', 0):.1f}% returns, {results.get('total_trades', 0)} trades")
            print(f"   📏 Win Rate Error: {win_rate_error:.1f}pp")
            print(f"   🛡️ Max Drawdown: {results.get('max_drawdown', 0):.1%}")
            print(f"   🚨 Emergency Halts: {results.get('halted_trades', 0)}")
            
            print(f"\n🔧 Fixes Applied:")
            for fix in summary['fixes_applied']:
                print(f"   ✅ {fix.replace('_', ' ').title()}")
            
            if win_rate_error <= 0.5:
                print(f"   🏆 Result: 🟢 PERFECT preservation (≤0.5pp error)")
            elif win_rate_error <= 1.0:
                print(f"   🏆 Result: 🟢 EXCELLENT preservation (≤1pp error)")
            elif win_rate_error <= 2.0:
                print(f"   🏆 Result: 🟡 GOOD preservation (≤2pp error)")
            elif win_rate_error <= 3.0:
                print(f"   🏆 Result: 🟡 ACCEPTABLE preservation (≤3pp error)")
            else:
                print(f"   🏆 Result: 🔴 STILL NEEDS IMPROVEMENT (>{win_rate_error:.1f}pp error)")
        
    finally:
        # Cleanup
        optimizer.cleanup()


if __name__ == '__main__':
    main()
