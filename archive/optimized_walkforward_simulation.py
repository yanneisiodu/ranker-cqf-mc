#!/usr/bin/env python3
"""
Optimized Walkforward Simulation with Optuna-tuned Risk Parameters

This version incorporates the flexible risk parameter framework needed for Optuna optimization.
"""

from __future__ import annotations

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


# Copy core functions from hybrid simulation
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
    """
    Optimized position sizing with tunable risk parameters.
    """
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


# Import other necessary functions from hybrid simulation
from hybrid_walkforward_simulation import (
    _load_meta, _load_policy_robust, _standardise_states, _decode_action,
    _fee_cost, _slippage_cost
)


def simulate_optimized_walkforward(
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
    **kwargs  # Catch any extra parameters from Optuna
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Optimized walkforward simulation with Optuna-tuned parameters.
    """
    logger = logging.getLogger("optimized_walkforward")
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


def main() -> None:
    """Run optimized simulation with best parameters."""
    parser = argparse.ArgumentParser(description="Run optimized walkforward simulation")
    parser.add_argument('--decision-table', type=Path, required=True)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--meta', type=Path, required=True)
    parser.add_argument('--params-file', type=Path, help="JSON file with optimized parameters")
    parser.add_argument('--outdir', type=Path, default=Path('optimized_walkforward_results'))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("optimized_walkforward")

    # Load optimized parameters if provided
    if args.params_file and args.params_file.exists():
        with open(args.params_file, 'r') as f:
            optimization_results = json.load(f)
            best_params = optimization_results['optimization_summary']['best_params']
        logger.info(f"Using optimized parameters from {args.params_file}")
    else:
        # Default parameters (equivalent to hybrid)
        best_params = {
            'base_contracts': 5,
            'base_notional': 1000.0,
            'enable_risk_controls': True,
            'max_notional_pct': 0.20,
            'commission_per_side': 0.65,
            'exchange_fee_per_side': 0.05,
            'slippage_min': 0.02,
            'slippage_pct': 0.20,
        }
        logger.info("Using default parameters")

    # Load data
    logger.info("Loading decision table and policy...")
    df = pd.read_csv(args.decision_table)
    meta = _load_meta(args.meta)
    states = _standardise_states(df, meta['state_columns'], meta['scaler_mean'], meta['scaler_scale'])
    algo = _load_policy_robust(args.policy, meta)
    predicted_actions = algo.predict(states)

    # Run simulation
    logger.info("Running optimized walkforward simulation...")
    results, summary = simulate_optimized_walkforward(
        df,
        predicted_actions,
        meta['action_map'],
        **best_params
    )

    # Save results
    args.outdir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.outdir / 'optimized_walkforward_trades.csv', index=False)
    with (args.outdir / 'optimized_walkforward_summary.json').open('w', encoding='utf-8') as fh:
        json.dump(summary, fh, indent=2)

    logger.info("🎯 Optimized Walk-forward Summary:")
    logger.info(f"  Return: {summary['return_pct']:.1f}%")
    logger.info(f"  Win Rate: {summary['win_rate']:.1%}")
    logger.info(f"  Max Drawdown: {summary['max_drawdown']:.1%}")
    logger.info(f"  Total Trades: {summary['total_trades']}")
    logger.info(f"  Profit Factor: {summary.get('profit_factor', 'N/A')}")


if __name__ == '__main__':
    main()