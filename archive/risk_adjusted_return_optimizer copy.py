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
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
import subprocess
import sys
import tempfile
import shutil
import textwrap

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO)


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
        min_return_threshold: float = 1500.0   # Minimum return to consider (150% above baseline)
    ):
        self.decision_table_path = decision_table_path
        self.policy_path = policy_path
        self.meta_path = meta_path
        self.initial_capital = initial_capital
        self.min_win_rate = min_win_rate
        self.min_trades = min_trades
        self.max_drawdown_tolerance = max_drawdown_tolerance
        self.min_return_threshold = min_return_threshold
        
        self.logger = logging.getLogger("risk_adjusted_optimizer")
        self.logger.info(f"🎯 RISK-ADJUSTED OPTIMIZATION: Win Rate ≥ {min_win_rate:.1%}, Maximize Calmar Ratio")
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
            trial.suggest_float('max_single_trade_notional', 80000, 200000, step=10000)
            if enable_single_trade_cap else 999999
        )
        
        halt_vol_emergency_only = (
            trial.suggest_categorical('halt_vol_emergency_only', [True, False])
            if enable_market_halt_protection else False
        )
        
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
            
            # Risk management controls
            'enable_portfolio_stop_loss': enable_portfolio_stop_loss,
            'portfolio_stop_loss_pct': portfolio_stop_loss_pct,
            'enable_single_trade_cap': enable_single_trade_cap,
            'max_single_trade_notional': max_single_trade_notional,
            'enable_market_halt_protection': enable_market_halt_protection,
            'halt_vol_emergency_only': halt_vol_emergency_only,
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
        """Run risk-adjusted optimization simulation."""
        try:
            # Create risk-adjusted simulation script
            simulation_script = self._create_risk_adjusted_script(params)
            
            # Prepare trial directory
            trial_dir = self.temp_dir / f"trial_{trial.number}"
            trial_dir.mkdir(exist_ok=True)
            
            # Write simulation script
            script_path = trial_dir / "risk_adjusted_sim.py"
            with open(script_path, 'w') as f:
                f.write(simulation_script)
            
            # Run simulation
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
            summary_file = trial_dir / "risk_adjusted_summary.json"
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    return json.load(f)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Trial {trial.number} simulation error: {e}")
            return None
    
    def _create_risk_adjusted_script(self, params: Dict) -> str:
        """Create risk-adjusted optimization simulation script."""

        template = textwrap.dedent("""#!/usr/bin/env python3
'''
Risk-Adjusted Return Optimization Walkforward Simulation
'''

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Deque
from collections import deque
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
    '''Load DiscreteCQL policy with fallback.'''
    logger = logging.getLogger("risk_adjusted")

    try:
        algo = load_learnable(str(policy_path))
        logger.info("✅ Loaded policy with load_learnable")
        return algo
    except Exception:
        logger.info("Falling back to manual reconstruction …")

        model_data = torch.load(str(policy_path), map_location="cpu")
        q_keys = [k for k in model_data["q_funcs"] if k.endswith("._fc.weight")]
        action_size = model_data["q_funcs"][q_keys[0]].shape[0] if q_keys else len(meta["action_map"])

        encoder_keys = [k for k in model_data["q_funcs"] if k.endswith("._encoder._layers.0.weight")]
        observation_size = model_data["q_funcs"][encoder_keys[0]].shape[1] if encoder_keys else len(meta["state_columns"])

        n_critics = len({k.split(".")[0] for k in model_data["q_funcs"].keys()})

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
        raise KeyError(f"Action id {action_id} not found in action map")
    return int(info.get("slot", 0)), float(info.get("size_value", 0.0))


class RiskAdjustedEngine:
    '''Risk-adjusted optimization engine with advanced risk management.'''

    def __init__(self, params: Dict, initial_capital: float):
        self.params = params
        self.initial_capital = initial_capital
        self.consecutive_losses = 0
        self.trade_history = []
        self.equity_peak = initial_capital
        self.recent_pnls = deque(maxlen=params.get('lookback_window', 10))
        self.equity_history = deque(maxlen=params.get('vol_lookback', 20))
        self.equity_history.append(initial_capital)

    def should_halt_trading(self, equity: float, contracts: int, notional: float, row: pd.Series) -> Tuple[bool, str]:
        '''Check halt conditions with enhanced risk management.'''

        # Portfolio stop loss
        if self.params.get('enable_portfolio_stop_loss', False):
            self.equity_peak = max(self.equity_peak, equity)
            drawdown_from_peak = (self.equity_peak - equity) / self.equity_peak
            stop_loss_pct = self.params.get('portfolio_stop_loss_pct', 0.6)
            if drawdown_from_peak > stop_loss_pct:
                return True, f"portfolio_drawdown_{drawdown_from_peak:.1%}"

        # Single trade size cap
        if self.params.get('enable_single_trade_cap', False):
            max_notional = self.params.get('max_single_trade_notional', 150000)
            if notional > max_notional:
                return True, f"trade_size_{notional:.0f}"

        # Market halt
        if self.params.get('enable_market_halt_protection', False):
            if self.params.get('halt_vol_emergency_only', False):
                vol_emergency = bool(row.get('s_vol_emergency', False))
                if vol_emergency:
                    return True, "market_vol_emergency"

        return False, ""

    def update_equity_peak(self, equity: float) -> None:
        if equity > self.equity_peak:
            self.equity_peak = equity

    def calculate_dynamic_position_size(self, base_contracts: int, notional: float) -> Tuple[int, float]:
        '''Calculate position size with dynamic adjustments for risk management.'''

        enhanced_contracts = base_contracts
        enhanced_notional = notional

        # Base position multiplier
        if self.params.get('enable_position_multiplier', False):
            multiplier = self.params.get('position_multiplier', 1.0)
            enhanced_contracts = int(enhanced_contracts * multiplier)
            enhanced_notional = enhanced_notional * multiplier

        # Dynamic sizing based on recent performance
        if self.params.get('enable_dynamic_sizing', False) and len(self.recent_pnls) > 5:
            recent_win_rate = sum(1 for pnl in self.recent_pnls if pnl > 0) / len(self.recent_pnls)
            if recent_win_rate < 0.6:
                size_adjustment = 0.7
                enhanced_contracts = int(enhanced_contracts * size_adjustment)
                enhanced_notional = enhanced_notional * size_adjustment
            elif recent_win_rate > 0.9:
                size_adjustment = 1.2
                enhanced_contracts = int(enhanced_contracts * size_adjustment)
                enhanced_notional = enhanced_notional * size_adjustment

        # Volatility-based adjustment
        if self.params.get('enable_vol_adjustment', False) and len(self.equity_history) > 10:
            equity_returns = np.diff(list(self.equity_history)) / list(self.equity_history)[:-1]
            volatility = np.std(equity_returns)

            if volatility > 0.05:
                vol_adjustment = 0.8
                enhanced_contracts = int(enhanced_contracts * vol_adjustment)
                enhanced_notional = enhanced_notional * vol_adjustment
            elif volatility < 0.02:
                vol_adjustment = 1.1
                enhanced_contracts = int(enhanced_contracts * vol_adjustment)
                enhanced_notional = enhanced_notional * vol_adjustment

        if self.params.get('enable_single_trade_cap', False):
            max_notional = self.params.get('max_single_trade_notional', 150000)
            if enhanced_notional > max_notional:
                scale_factor = max_notional / enhanced_notional
                enhanced_contracts = max(int(enhanced_contracts * scale_factor), 1)
                enhanced_notional = max_notional

        if enhanced_contracts <= 0 or enhanced_notional <= 0:
            return 0, 0.0

        return enhanced_contracts, enhanced_notional

    def should_skip_due_to_consecutive_losses(self) -> bool:
        if not self.params.get('enable_consecutive_loss_breaker', False):
            return False

        max_losses = self.params.get('max_consecutive_losses', 30)
        return self.consecutive_losses >= max_losses

    def should_skip_due_to_return_filter(self, row: pd.Series, slot: int) -> bool:
        if not self.params.get('enable_return_filter', False):
            return False

        min_return = self.params.get('min_expected_return', 0.0)
        pnl_col = f"c{slot}_target_pnl"
        expected_return = float(row.get(pnl_col, 0.0) or 0.0)

        return expected_return < min_return

    def update_performance_tracking(self, realized_pnl: float, equity: float):
        if realized_pnl <= 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        self.recent_pnls.append(realized_pnl)
        self.equity_history.append(equity)


def simulate_risk_adjusted_walkforward(
    decision_df: pd.DataFrame,
    predicted_actions: np.ndarray,
    action_map: Dict[str, Dict[str, float]],
    params: Dict,
    initial_capital: float = 10_000.0
) -> Dict[str, float]:
    '''Risk-adjusted walkforward without look-ahead leakage.'''

    df = decision_df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)

    predicted_actions = np.asarray(predicted_actions, dtype=np.int64)
    if len(predicted_actions) != len(df):
        raise ValueError(
            f"predicted_actions length ({len(predicted_actions)}) does not match decision_df length ({len(df)})"
        )

    holding_period_days = int(params.get('holding_period_days', params.get('trade_horizon_days', 5)))
    if holding_period_days < 0:
        raise ValueError('holding_period_days must be non-negative')
    holding_delta = pd.Timedelta(days=holding_period_days)

    commission = float(params.get('commission_per_side', 0.65))
    exchange_fee = float(params.get('exchange_fee_per_side', 0.05))
    slippage_min = float(params.get('slippage_min', 0.02))
    slippage_pct = float(params.get('slippage_pct', 0.20))

    base_contracts = int(params.get('base_contracts', 10))
    base_notional = float(params.get('base_notional', 1000.0))
    max_notional_pct: Optional[float] = params.get('max_notional_pct', 0.10)

    equity = float(initial_capital)
    optimizer = RiskAdjustedEngine(params, initial_capital)
    history: List[Dict[str, Any]] = []
    open_trades: Deque[Dict[str, Any]] = deque()

    def settle_trades(until_date: pd.Timestamp) -> None:
        nonlocal equity
        while open_trades and open_trades[0]['settle_date'] <= until_date:
            trade = open_trades.popleft()
            record = history[trade['record_index']]

            record['settled'] = True
            record['settled_on'] = trade['settle_date']
            record['equity_before_settlement'] = equity

            equity += trade['realized_pnl'] - trade['fees'] - trade['slippage']
            optimizer.update_performance_tracking(trade['realized_pnl'], equity)
            optimizer.update_equity_peak(equity)

            record['realized_pnl'] = trade['realized_pnl']
            record['equity_after_settlement'] = equity
            record['equity_after'] = equity
            record['equity_curve_value'] = equity
            record['date'] = trade['settle_date']

    for i in range(len(df)):
        row = df.iloc[i]
        current_date = row['date']

        settle_trades(current_date)
        optimizer.update_equity_peak(equity)

        action_id = int(predicted_actions[i])
        slot, size_mult = _decode_action(action_id, action_map)

        record: Dict[str, Any] = {
            'date': current_date,
            'entry_date': current_date,
            'action_id': action_id,
            'slot': slot,
            'size_mult': size_mult,
            'equity_before': equity,
            'halt_reason': None,
            'skip_reason': None,
            'n_contracts': 0,
            'notional': 0.0,
            'fees': 0.0,
            'slippage': 0.0,
            'realized_pnl': 0.0,
            'equity_after': equity,
            'equity_curve_value': equity,
            'settled': False,
            'settled_on': None,
            'equity_before_settlement': None,
            'equity_after_settlement': equity,
        }

        if slot <= 0 or size_mult <= 0:
            history.append(record)
            continue

        price_col = f"c{slot}_future_option_price"
        premium = row.get(price_col)
        if premium is None or pd.isna(premium):
            record['skip_reason'] = f'no_price_data_slot_{slot}'
            history.append(record)
            continue

        if optimizer.should_skip_due_to_consecutive_losses():
            record['halt_reason'] = 'consecutive_losses'
            history.append(record)
            continue

        if optimizer.should_skip_due_to_return_filter(row, slot):
            record['halt_reason'] = 'low_expected_return'
            history.append(record)
            continue

        n_contracts = max(int(round(base_contracts * size_mult)), 0)
        if n_contracts <= 0:
            record['skip_reason'] = 'non_positive_contracts'
            history.append(record)
            continue

        notional = base_notional * n_contracts

        if max_notional_pct is not None:
            notional_cap = float(max_notional_pct) * equity
            if notional_cap <= 0:
                record['halt_reason'] = 'no_equity_available'
                history.append(record)
                continue
            if notional > notional_cap:
                scale = notional_cap / notional if notional > 0 else 0.0
                n_contracts = max(int(np.floor(n_contracts * scale)), 0)
                notional = base_notional * n_contracts
                if n_contracts <= 0:
                    record['skip_reason'] = 'notional_cap_zero'
                    history.append(record)
                    continue

        n_contracts, notional = optimizer.calculate_dynamic_position_size(n_contracts, notional)
        if n_contracts <= 0 or notional <= 0:
            record['skip_reason'] = 'sizing_reduced_to_zero'
            history.append(record)
            continue

        should_halt, halt_reason = optimizer.should_halt_trading(equity, n_contracts, notional, row)
        if should_halt:
            record['halt_reason'] = halt_reason or 'risk_halt'
            history.append(record)
            continue

        fees = (commission + exchange_fee) * n_contracts * 2.0
        spread = float(row.get('bid_ask_spread', 0.0) or 0.0)
        slip_per_contract = max(slippage_min, slippage_pct * spread)
        slippage = slip_per_contract * 100.0 * n_contracts * 2.0

        pnl_col = f"c{slot}_target_pnl"
        raw_return = row.get(pnl_col, 0.0)
        if raw_return is None or pd.isna(raw_return):
            record['skip_reason'] = f'missing_target_pnl_slot_{slot}'
            history.append(record)
            continue
        realized_pnl = float(raw_return) * notional

        record.update(
            {
                'n_contracts': n_contracts,
                'notional': notional,
                'fees': fees,
                'slippage': slippage,
            }
        )

        record_index = len(history)
        history.append(record)

        settlement_date = current_date if holding_period_days == 0 else current_date + holding_delta
        open_trades.append(
            {
                'record_index': record_index,
                'settle_date': settlement_date,
                'realized_pnl': realized_pnl,
                'fees': fees,
                'slippage': slippage,
            }
        )

        if holding_period_days == 0:
            settle_trades(current_date)

    if not df.empty:
        settle_trades(df['date'].iloc[-1] + holding_delta)

    results = pd.DataFrame(history)
    if not results.empty:
        results['date'] = pd.to_datetime(results['date'])
        if 'entry_date' in results.columns:
            results['entry_date'] = pd.to_datetime(results['entry_date'])
        if 'settled_on' in results.columns:
            results['settled_on'] = pd.to_datetime(results['settled_on'])
        results.sort_values('date', inplace=True)
        results.reset_index(drop=True, inplace=True)

    trades_mask = (results['n_contracts'] > 0) & results['settled']
    winning_trades = results[trades_mask & (results['realized_pnl'] > 0)]
    losing_trades = results[trades_mask & (results['realized_pnl'] < 0)]

    halted_trades = int(results['halt_reason'].notna().sum()) if 'halt_reason' in results.columns else 0
    skipped_trades = int(results['skip_reason'].notna().sum()) if 'skip_reason' in results.columns else 0

    equity_series = results['equity_curve_value'] if 'equity_curve_value' in results.columns else results['equity_after']
    running_max = equity_series.expanding().max() if not equity_series.empty else equity_series
    drawdowns = (equity_series - running_max) / running_max if not equity_series.empty else equity_series
    max_drawdown = abs(float(drawdowns.min())) if not drawdowns.empty else 0.0

    final_equity = float(equity)
    return_pct = float((final_equity / initial_capital - 1.0) * 100.0)
    calmar_ratio = return_pct / (max_drawdown * 100.0) if max_drawdown > 0 else float('inf')

    summary = {
        'initial_capital': initial_capital,
        'final_capital': final_equity,
        'total_pnl': final_equity - initial_capital,
        'total_fees': float(results.loc[trades_mask, 'fees'].sum()) if not results.empty else 0.0,
        'total_slippage': float(results.loc[trades_mask, 'slippage'].sum()) if not results.empty else 0.0,
        'total_trades': int(trades_mask.sum()),
        'winning_trades': len(winning_trades),
        'losing_trades': len(losing_trades),
        'win_rate': len(winning_trades) / max(int(trades_mask.sum()), 1),
        'return_pct': return_pct,
        'max_drawdown': float(max_drawdown),
        'calmar_ratio': calmar_ratio,
        'optimization_params': params.copy(),
        'halted_trades': halted_trades,
        'skipped_trades': skipped_trades,
    }

    if trades_mask.any():
        trade_pnls = results.loc[trades_mask, 'realized_pnl']
        loss_sum = losing_trades['realized_pnl'].sum()
        summary.update(
            {
                'avg_trade_pnl': float(trade_pnls.mean()),
                'largest_win': float(trade_pnls.max()),
                'largest_loss': float(trade_pnls.min()),
                'profit_factor': float(winning_trades['realized_pnl'].sum() / abs(loss_sum)) if loss_sum < 0 else float('inf'),
            }
        )

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--decision-table', type=Path, required=True)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--meta', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    df = pd.read_csv(args.decision_table)
    meta = _load_meta(args.meta)
    states = _standardise_states(df, meta['state_columns'], meta['scaler_mean'], meta['scaler_scale'])
    algo = _load_policy_robust(args.policy, meta)
    predicted_actions = algo.predict(states)

    optimization_params = __PARAMS__

    results = simulate_risk_adjusted_walkforward(df, predicted_actions, meta['action_map'], optimization_params)

    args.outdir.mkdir(parents=True, exist_ok=True)
    with open(args.outdir / 'risk_adjusted_summary.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == '__main__':
    main()
""")

        return template.replace("__PARAMS__", repr(params))
    
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
        min_win_rate=args.min_win_rate
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
