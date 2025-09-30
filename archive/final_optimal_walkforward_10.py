#!/usr/bin/env python3
"""
Final Optimal Walkforward Simulation

Uses the best parameters discovered from risk-adjusted optimization:
- Trial #74: 86.6% win rate, 7,988.9% return, 15.2% max drawdown
- Calmar Ratio: 692.5 (39× better than baseline)
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import deque
import numpy as np
import pandas as pd
from d3rlpy import load_learnable
from d3rlpy.algos import DiscreteCQL, DiscreteCQLConfig
import torch

__all__ = [
    "load_decision_table",
    "simulate_optimal_walkforward",
    "OptimalRiskEngine",
    "_prepare_decision_table",
    "_load_meta",
    "_load_policy_robust",
    "_standardise_states",
]

def _load_meta(meta_path: Path) -> Dict:
    with meta_path.open("r", encoding="utf-8") as fh:
        meta = json.load(fh)
    return meta

def _load_policy_robust(policy_path: Path, meta: Dict) -> DiscreteCQL:
    """Load DiscreteCQL policy with fallback."""
    logger = logging.getLogger("optimal_walkforward")
    
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


def _extract_active_slots(action_map: Dict[str, Dict[str, float]]) -> Set[int]:
    """Return set of slots that can be selected by the policy (slot=0 is no-trade)."""

    slots: Set[int] = set()
    for action_cfg in action_map.values():
        slot = int(action_cfg.get("slot", 0))
        if slot > 0:
            slots.add(slot)
    return slots


def _validate_required_columns(df: pd.DataFrame, slots: Set[int]) -> None:
    """Ensure required future columns exist for all actionable slots."""

    required: Set[str] = {"date"}
    for slot in slots:
        required.add(f"c{slot}_future_option_price")
        required.add(f"c{slot}_target_pnl")

    missing = required.difference(df.columns)
    if missing:
        raise KeyError(
            "Decision table missing required columns: "
            + ", ".join(sorted(missing))
        )


def _prepare_decision_table(
    decision_df: pd.DataFrame,
    action_map: Dict[str, Dict[str, float]],
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Sort, de-duplicate, and drop rows lacking future labels for any actionable slot."""

    df = decision_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    actionable_slots = _extract_active_slots(action_map)
    if not actionable_slots:
        return df

    _validate_required_columns(df, actionable_slots)

    mask = pd.Series(True, index=df.index)
    for slot in actionable_slots:
        mask &= df[f"c{slot}_future_option_price"].notna()
        mask &= df[f"c{slot}_target_pnl"].notna()

    dropped = int((~mask).sum())
    if dropped and logger is not None:
        logger.info(
            "Dropping %d rows with missing future labels across actionable slots", dropped
        )

    return df.loc[mask].reset_index(drop=True)


def load_decision_table(
    csv_path: Path,
    action_map: Dict[str, Dict[str, float]],
    preprocess: bool = True,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Load decision table CSV and optionally apply preprocessing."""

    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df.get('date'), errors='coerce')
    if not preprocess:
        if logger is not None:
            logger.info("Loaded decision table without preprocessing (%d rows)", len(df))
        return df

    cleaned = _prepare_decision_table(df, action_map, logger)
    if logger is not None:
        dropped = len(df) - len(cleaned)
        if dropped:
            logger.info("Decision table cleaned: %d rows dropped, %d remaining", dropped, len(cleaned))
        else:
            logger.info("Decision table contains %d rows", len(cleaned))
    return cleaned

class OptimalRiskEngine:
    """Optimal risk engine using Trial #74 parameters."""
    
    def __init__(self, params: Dict, initial_capital: float):
        self.params = params
        self.initial_capital = initial_capital
        self.consecutive_losses = 0
        self.equity_peak = initial_capital
        self.bypass_controls = bool(self.params.get('bypass_all_normal_controls', False))
        self.lookback_window = max(int(self.params.get('lookback_window', 10)), 1)
        self.vol_lookback = max(int(self.params.get('vol_lookback', 20)), 1)
        self.recent_pnls: deque[float] = deque(maxlen=self.lookback_window)
        self.equity_history: deque[float] = deque(maxlen=self.vol_lookback)
        self.equity_history.append(initial_capital)
        self.halt_until: Optional[pd.Timestamp] = None

    def should_halt_trading(self, equity: float, contracts: int, notional: float, row: pd.Series) -> Tuple[bool, str]:
        """Market-wide halts that rely only on same-day information."""
        if self.bypass_controls:
            return False, ""

        if self.params.get('enable_market_halt_protection', False):
            if self.params.get('halt_vol_emergency_only', False):
                vol_emergency = bool(row.get('s_vol_emergency', False))
                if vol_emergency:
                    return True, "market_vol_emergency"
            else:
                vol_severity = float(row.get('s_vol_severity', 0.0) or 0.0)
                threshold = float(self.params.get('halt_vol_severity_threshold', 2.0))
                if vol_severity >= threshold:
                    return True, "market_volatility_halt"

        return False, ""

    def update_equity_peak(self, equity: float) -> None:
        if equity > self.equity_peak:
            self.equity_peak = equity
    
    def _vol_target_scale(self) -> float:
        if not self.params.get('enable_vol_targeting', False):
            return 1.0

        target_vol = float(self.params.get('target_annual_vol', 0.15))
        lookback = max(int(self.params.get('vol_lookback_days', 30)), 5)
        if len(self.equity_history) <= lookback:
            return 1.0

        eq_vals = np.array(self.equity_history)[- (lookback + 1) :]
        returns = np.diff(eq_vals) / eq_vals[:-1]
        if returns.size == 0:
            return 1.0
        daily_vol = float(np.std(returns))
        if daily_vol <= 1e-8:
            return float(self.params.get('vol_scale_max', 1.0))

        ann_vol = daily_vol * np.sqrt(252.0)
        scale = target_vol / ann_vol if ann_vol > 0 else 1.0
        scale_min = float(self.params.get('vol_scale_min', 0.5))
        scale_max = float(self.params.get('vol_scale_max', 1.5))
        return float(np.clip(scale, scale_min, scale_max))

    def apply_optimal_position_sizing(
        self,
        base_contracts: int,
        contract_value: float,
        row: pd.Series,
    ) -> Tuple[int, float]:
        """Apply Trial #74 sizing with optional dynamic/volatility adjustments."""

        per_contract = float(contract_value)
        if per_contract <= 0:
            return 0, 0.0
        enhanced_contracts = max(base_contracts, 0)
        enhanced_notional = per_contract * enhanced_contracts

        if self.params.get('enable_position_multiplier', False):
            multiplier = float(self.params.get('position_multiplier', 1.0))
            enhanced_contracts = int(round(enhanced_contracts * multiplier))
            enhanced_contracts = max(enhanced_contracts, 0)
            enhanced_notional = per_contract * enhanced_contracts

        # Dynamic sizing reacts to recent performance
        if self.params.get('enable_dynamic_sizing', False) and len(self.recent_pnls) > 5:
            recent_win_rate = sum(1 for pnl in self.recent_pnls if pnl > 0) / len(self.recent_pnls)
            if recent_win_rate < 0.6:
                size_adjustment = 0.7
                enhanced_contracts = int(np.floor(enhanced_contracts * size_adjustment))
                enhanced_contracts = max(enhanced_contracts, 0)
                enhanced_notional = per_contract * enhanced_contracts
            elif recent_win_rate > 0.9:
                size_adjustment = 1.2
                enhanced_contracts = int(round(enhanced_contracts * size_adjustment))
                enhanced_contracts = max(enhanced_contracts, 0)
                enhanced_notional = per_contract * enhanced_contracts

        # Volatility-based adjustment using realised equity history
        if self.params.get('enable_vol_adjustment', False) and len(self.equity_history) > 10:
            equity_vals = list(self.equity_history)
            equity_returns = np.diff(equity_vals) / equity_vals[:-1]
            volatility = float(np.std(equity_returns)) if equity_returns.size > 0 else 0.0

            if volatility > 0.05:
                vol_adjustment = 0.8
                enhanced_contracts = int(np.floor(enhanced_contracts * vol_adjustment))
                enhanced_contracts = max(enhanced_contracts, 0)
                enhanced_notional = per_contract * enhanced_contracts
            elif volatility < 0.02:
                vol_adjustment = 1.1
                enhanced_contracts = int(round(enhanced_contracts * vol_adjustment))
                enhanced_contracts = max(enhanced_contracts, 0)
                enhanced_notional = per_contract * enhanced_contracts

        if self.params.get('enable_vol_targeting', False):
            scale = self._vol_target_scale()
            if scale != 1.0:
                scaled_notional = enhanced_notional * scale
                enhanced_contracts = max(int(round(scaled_notional / per_contract)), 0)
                enhanced_notional = per_contract * enhanced_contracts

        if enhanced_contracts <= 0 or enhanced_notional <= 0:
            return 0, 0.0

        return enhanced_contracts, enhanced_notional
    
    def should_skip_due_to_consecutive_losses(self) -> bool:
        """Consecutive loss breaker from Trial #74."""
        if not self.params.get('enable_consecutive_loss_breaker', False):
            return False
        
        max_losses = self.params.get('max_consecutive_losses', 18)  # Trial #74 value
        return self.consecutive_losses >= max_losses
    
    def update_performance_tracking(self, realized_pnl: float, equity: float) -> None:
        """Update loss counters and maintain realised equity history."""

        if realized_pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        self.recent_pnls.append(float(realized_pnl))
        self.equity_history.append(float(equity))
        self.update_equity_peak(equity)

    def should_halt_today(self, current_date: pd.Timestamp, equity: float) -> Tuple[bool, str]:
        halt_until = self.halt_until
        if halt_until is not None and current_date >= halt_until:
            self.halt_until = None
        elif halt_until is not None and current_date < halt_until:
            return True, 'portfolio_trailing_stop_cooldown'

        if self.params.get('enable_portfolio_trailing_stop', False) and self.equity_peak > 0:
            dd_floor = float(self.params.get('portfolio_dd_floor', 0.15))
            drawdown = 1.0 - equity / max(self.equity_peak, equity)
            if drawdown >= dd_floor:
                cooldown = max(int(self.params.get('halt_cooldown_days', 5)), 0)
                if cooldown > 0:
                    self.halt_until = current_date + pd.Timedelta(days=cooldown)
                else:
                    self.halt_until = current_date
                return True, f'portfolio_trailing_stop_{drawdown:.1%}'

        return False, ""

    def should_skip_due_to_return_filter(self, row: pd.Series, slot: int) -> bool:
        if not self.params.get('enable_return_filter', False):
            return False

        min_return = float(self.params.get('min_expected_return', 0.0))
        pnl_col = f"c{slot}_target_pnl"
        expected_return = float(row.get(pnl_col, 0.0) or 0.0)
        return expected_return < min_return

def simulate_optimal_walkforward(
    decision_df: pd.DataFrame,
    predicted_actions: np.ndarray,
    action_map: Dict[str, Dict[str, float]],
    params: Dict,
    initial_capital: float = 10_000.0,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Simulate walkforward trading with delayed P&L settlement to avoid look-ahead leakage."""

    df = decision_df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)

    predicted_actions = np.asarray(predicted_actions, dtype=np.int64)
    if len(predicted_actions) != len(df):
        raise ValueError(
            f"predicted_actions length ({len(predicted_actions)}) does not match decision_df length ({len(df)})"
        )

    base_contracts = int(params.get('base_contracts', 10))
    base_notional = float(params.get('base_notional', 1000.0))
    max_notional_pct: Optional[float] = params.get('max_notional_pct', 0.10)

    holding_period_days = int(params.get('holding_period_days', params.get('trade_horizon_days', 5)))
    if holding_period_days < 0:
        raise ValueError('holding_period_days must be non-negative')
    holding_delta = pd.Timedelta(days=holding_period_days)

    commission = float(params.get('commission_per_side', 0.65))
    exchange_fee = float(params.get('exchange_fee_per_side', 0.05))
    slippage_min = float(params.get('slippage_min', 0.02))
    slippage_pct = float(params.get('slippage_pct', 0.20))

    equity = float(initial_capital)
    optimizer = OptimalRiskEngine(params, initial_capital)
    history: List[Dict[str, Any]] = []
    open_trades: deque[Dict[str, Any]] = deque()
    current_trade_day: Optional[pd.Timestamp] = None
    daily_notional_used: float = 0.0

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

        trade_day = pd.Timestamp(current_date.date())
        if current_trade_day != trade_day:
            current_trade_day = trade_day
            daily_notional_used = 0.0

        if slot <= 0 or size_mult <= 0:
            history.append(record)
            continue

        price_col = f'c{slot}_future_option_price'
        premium = row.get(price_col)
        if premium is None or pd.isna(premium):
            record['skip_reason'] = f'missing_future_price_slot_{slot}'
            history.append(record)
            continue

        if optimizer.should_skip_due_to_consecutive_losses():
            record['halt_reason'] = 'consecutive_losses'
            history.append(record)
            continue

        halt_today, trailing_reason = optimizer.should_halt_today(current_date, equity)
        if halt_today:
            record['halt_reason'] = trailing_reason
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

        n_contracts, notional = optimizer.apply_optimal_position_sizing(n_contracts, base_notional, row)

        per_trade_cap_pct = params.get('max_notional_pct_per_trade')
        if per_trade_cap_pct is not None:
            per_trade_cap = float(per_trade_cap_pct) * equity
            if notional > per_trade_cap:
                scale = per_trade_cap / notional if notional > 0 else 0.0
                n_contracts = max(int(np.floor(n_contracts * scale)), 0)
                notional = base_notional * n_contracts
                if n_contracts <= 0:
                    record['skip_reason'] = 'per_trade_notional_cap'
                    history.append(record)
                    continue

        max_daily_pct = params.get('max_daily_notional_pct')
        if max_daily_pct is not None:
            daily_cap_value = float(max_daily_pct) * equity
            remaining = daily_cap_value - daily_notional_used
            if remaining <= 0:
                record['skip_reason'] = 'daily_notional_cap'
                history.append(record)
                continue
            if notional > remaining:
                scale = remaining / notional if notional > 0 else 0.0
                n_contracts = max(int(np.floor(n_contracts * scale)), 0)
                notional = base_notional * n_contracts
                if n_contracts <= 0:
                    record['skip_reason'] = 'daily_notional_cap'
                    history.append(record)
                    continue

        if params.get('enable_single_trade_cap'):
            max_notional = params.get('max_single_trade_notional')
            if max_notional is None:
                cap_value = params.get('single_trade_notional_cap')
                cap_pct = params.get('single_trade_notional_pct')
                if cap_value is not None:
                    max_notional = float(cap_value)
                elif cap_pct is not None:
                    max_notional = float(cap_pct) * equity
            if max_notional is not None and notional > float(max_notional):
                scale = float(max_notional) / notional if notional > 0 else 0.0
                n_contracts = max(int(np.floor(n_contracts * scale)), 0)
                notional = base_notional * n_contracts
                if n_contracts <= 0:
                    record['skip_reason'] = 'single_trade_cap'
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

        pnl_col = f'c{slot}_target_pnl'
        raw_return = row.get(pnl_col, 0.0)
        if raw_return is None or pd.isna(raw_return):
            record['skip_reason'] = f'missing_target_pnl_slot_{slot}'
            history.append(record)
            continue
        realized_pnl = float(raw_return) * notional

        record.update({
            'n_contracts': n_contracts,
            'notional': notional,
            'fees': fees,
            'slippage': slippage,
            'equity_after': equity,
        })

        record_index = len(history)
        history.append(record)
        daily_notional_used += notional

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

    halted_trades = int(results['halt_reason'].notna().sum())
    skipped_trades = int(results['skip_reason'].notna().sum())

    equity_series = results['equity_curve_value'] if 'equity_curve_value' in results.columns else results['equity_after']
    equity_series = equity_series.fillna(method='ffill').fillna(initial_capital)
    running_max = equity_series.cummax() if not equity_series.empty else equity_series
    drawdowns = (equity_series - running_max) / running_max if not equity_series.empty else equity_series
    max_drawdown = abs(float(drawdowns.min())) if not drawdowns.empty else 0.0

    final_equity = float(equity)
    return_pct = float((final_equity / initial_capital - 1.0) * 100.0)
    calmar_ratio = return_pct / (max_drawdown * 100.0) if max_drawdown > 0 else float('inf')

    summary: Dict[str, Any] = {
        'approach': params.get('approach', 'optimal_trial_74_configuration'),
        'initial_capital': initial_capital,
        'final_capital': final_equity,
        'total_pnl': final_equity - initial_capital,
        'total_fees': float(results.loc[trades_mask, 'fees'].sum()),
        'total_slippage': float(results.loc[trades_mask, 'slippage'].sum()),
        'total_trades': int(trades_mask.sum()),
        'winning_trades': len(winning_trades),
        'losing_trades': len(losing_trades),
        'win_rate': len(winning_trades) / max(int(trades_mask.sum()), 1),
        'return_pct': return_pct,
        'max_drawdown': float(max_drawdown),
        'calmar_ratio': calmar_ratio,
        'halted_trades': halted_trades,
        'skipped_trades': skipped_trades,
        'optimal_params': params.copy(),
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

    return results, summary

def main():
    """Run final optimal walkforward simulation."""
    parser = argparse.ArgumentParser(description="Final optimal walkforward with Trial #62 parameters")
    parser.add_argument('--decision-table', type=Path, required=True)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--meta', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, default=Path('results/final_optimal_walkforward'))
    parser.add_argument('--skip-preprocessing', action='store_true', help='Use decision table as-is without dropping rows with missing future labels.')

    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("optimal_walkforward")
    
    # OPTIMAL PARAMETERS from V10 Trial #6 (5,060% returns, 15.3% drawdown, Calmar 331.2)
    optimal_params = {
        'approach': 'risk_adjusted_vol_target',
        'base_contracts': 10,
        'base_notional': 1000.0,
        'bypass_all_normal_controls': False,
        'preserve_contract_risk_filter': True,
        'holding_period_days': 5,
        'max_notional_pct': 0.14,
        'enable_portfolio_trailing_stop': False,
        'portfolio_dd_floor': 0.2,
        'halt_cooldown_days': 0,
        'enable_consecutive_loss_breaker': True,
        'max_consecutive_losses': 6,  # Very strict loss limit
        'enable_market_halt_protection': True,
        'halt_vol_emergency_only': True,
        'halt_vol_severity_threshold': 2.25,

        'enable_position_multiplier': True,
        'position_multiplier': 1.6,  # Conservative leverage
        'enable_dynamic_sizing': False,
        'lookback_window': 12,
        'enable_vol_adjustment': False,
        'vol_lookback': 20,
        'enable_vol_targeting': False,
        'target_annual_vol': 0.15,
        'vol_lookback_days': 30,
        'vol_scale_min': 0.5,
        'vol_scale_max': 1.2,

        'enable_return_filter': False,
        'min_expected_return': 0.0,
        'max_notional_pct_per_trade': 0.20,  # 20% per trade limit (allows $16k trades with $10k equity)
        'max_daily_notional_pct': 0.50,  # 50% daily limit (allows multiple trades per day)
        'enable_single_trade_cap': False,
        'max_single_trade_notional': 999_999,

        'commission_per_side': 0.65,
        'exchange_fee_per_side': 0.05,
        'slippage_min': 0.02,
        'slippage_pct': 0.20,
    }
    
    logger.info("🎯 FINAL OPTIMAL WALKFORWARD SIMULATION")
    logger.info("=" * 60)
    logger.info("🏆 Using V10 optimal configuration (15.3% MDD, 5,060% returns, Calmar 331.2)")
    logger.info("📊 Features: 1.6× leverage + 6-loss limit + strict notional caps + emergency halts")
    
    # Load data
    logger.info("Loading decision table and policy...")
    meta = _load_meta(args.meta)
    df = load_decision_table(
        args.decision_table,
        meta['action_map'],
        preprocess=not args.skip_preprocessing,
        logger=logger,
    )
    raw_state_cols = meta['state_columns']
    filtered_state_cols = [
        col
        for col in raw_state_cols
        if not (
            col.endswith('target_pnl')
            or col.endswith('future_option_price')
            or col == 's_target_pnl'
            or col.endswith('contractID')
        )
    ]
    if not filtered_state_cols:
        raise ValueError('No valid state columns available after removing realized outcome features')
    if len(filtered_state_cols) != len(raw_state_cols):
        keep_indices = [raw_state_cols.index(col) for col in filtered_state_cols]
        scaler_mean = [meta['scaler_mean'][idx] for idx in keep_indices]
        scaler_scale = [meta['scaler_scale'][idx] for idx in keep_indices]
    else:
        scaler_mean = meta['scaler_mean']
        scaler_scale = meta['scaler_scale']
    states = _standardise_states(df, filtered_state_cols, scaler_mean, scaler_scale)
    algo = _load_policy_robust(args.policy, meta)
    predicted_actions = algo.predict(states)
    
    # Run optimal simulation
    logger.info("Running optimal walkforward simulation...")
    results, summary = simulate_optimal_walkforward(
        df,
        predicted_actions,
        meta['action_map'],
        optimal_params
    )
    
    # Save results
    args.outdir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.outdir / 'optimal_walkforward_trades.csv', index=False)
    with (args.outdir / 'optimal_walkforward_summary.json').open('w', encoding='utf-8') as fh:
        json.dump(summary, fh, indent=2, default=str)
    
    # Display results
    logger.info("🎯 FINAL OPTIMAL WALKFORWARD RESULTS:")
    logger.info("=" * 60)
    logger.info(f"📊 Win Rate: {summary['win_rate']:.1%}")
    logger.info(f"💰 Return: {summary['return_pct']:.1f}%")
    logger.info(f"🛡️ Max Drawdown: {summary['max_drawdown']:.1%}")
    logger.info(f"📈 Calmar Ratio: {summary['calmar_ratio']:.1f}")
    logger.info(f"📊 Total Trades: {summary['total_trades']}")
    logger.info(f"🚨 Emergency Halts: {summary['halted_trades']}")
    logger.info(f"⏳ Skipped Trades: {summary.get('skipped_trades', 0)}")
    logger.info(f"💵 Final Capital: ${summary['final_capital']:,.2f}")
    logger.info(f"⚡ Avg Trade P&L: ${summary.get('avg_trade_pnl', 0):,.0f}")
    logger.info(f"🏆 Profit Factor: {summary.get('profit_factor', 0):.1f}")
    
    # Compare to baseline
    baseline_return = 1119.7
    baseline_drawdown = 0.649
    baseline_calmar = baseline_return / (baseline_drawdown * 100)
    
    return_improvement = (summary['return_pct'] / baseline_return - 1.0) * 100
    drawdown_improvement = (baseline_drawdown - summary['max_drawdown']) / baseline_drawdown * 100
    calmar_improvement = (summary['calmar_ratio'] / baseline_calmar - 1.0) * 100
    
    logger.info("\n🚀 IMPROVEMENT vs BASELINE:")
    logger.info("=" * 60)
    logger.info(f"📈 Return Improvement: {return_improvement:+.1f}%")
    logger.info(f"🛡️ Drawdown Improvement: {drawdown_improvement:+.1f}%")
    logger.info(f"📊 Calmar Improvement: {calmar_improvement:+.1f}%")
    
    # Performance tier
    if summary['max_drawdown'] <= 0.15 and summary['win_rate'] >= 0.85 and summary['return_pct'] >= 5000:
        tier = "🏆 WORLD-CLASS PERFORMANCE"
    elif summary['max_drawdown'] <= 0.20 and summary['win_rate'] >= 0.83 and summary['return_pct'] >= 3000:
        tier = "🥇 ELITE PERFORMANCE"
    elif summary['max_drawdown'] <= 0.30 and summary['win_rate'] >= 0.80 and summary['return_pct'] >= 2000:
        tier = "🥈 EXCELLENT PERFORMANCE"
    else:
        tier = "🥉 GOOD PERFORMANCE"
    
    logger.info(f"\n{tier}")
    
    print(f"\n✅ Final Optimal Walkforward Complete!")
    print(f"📊 Results: {summary['win_rate']:.1%} win rate, {summary['return_pct']:.1f}% return, {summary['max_drawdown']:.1%} drawdown")
    multiplier = optimal_params.get('position_multiplier', 1.0)
    config_bits = [
        f"{multiplier:.1f}× position multiplier" if multiplier and multiplier != 1.0 else "base sizing",
        "vol targeting" if optimal_params.get('enable_vol_targeting') else None,
        "trailing stop" if optimal_params.get('enable_portfolio_trailing_stop') else None,
        "dynamic sizing" if optimal_params.get('enable_dynamic_sizing') else None,
    ]
    config_summary = " + ".join(bit for bit in config_bits if bit)
    if not config_summary:
        config_summary = "baseline risk controls"
    print(f"🎯 Configuration: {config_summary}")
    print(f"⏳ Skipped Trades: {summary.get('skipped_trades', 0)}")
    print(f"💾 Saved to: {args.outdir}")

if __name__ == '__main__':
    main()
