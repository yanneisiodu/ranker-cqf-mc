#!/usr/bin/env python3
"""
Hybrid Walk-forward Simulation: Original Performance + Enhanced Risk Controls

This version bridges the gap between your original 1119% returns and proper risk management
by using realistic parameters while still adding essential risk controls.
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


# Copy the core helper functions from enhanced version
def _calculate_rolling_drawdown(equity_series: pd.Series, window: int = 20) -> float:
    """Calculate current drawdown from rolling peak."""
    if len(equity_series) < 2:
        return 0.0
    
    peak = equity_series.rolling(window=min(window, len(equity_series))).max().iloc[-1]
    current = equity_series.iloc[-1]
    drawdown = (current - peak) / peak if peak > 0 else 0.0
    return max(drawdown, 0.0)


def _get_volatility_regime_multiplier(vol_severity: float, vol_emergency: bool) -> float:
    """Get position size multiplier based on market volatility regime."""
    if vol_emergency:
        return 0.4  # Less aggressive reduction than before
    elif vol_severity > 3.0:
        return 0.6  # High vol regime
    elif vol_severity > 2.0:
        return 0.8  # Moderate vol regime
    else:
        return 1.0  # Normal regime


def _load_meta(meta_path: Path) -> Dict:
    with meta_path.open("r", encoding="utf-8") as fh:
        meta = json.load(fh)
    required = {"state_columns", "scaler_mean", "scaler_scale", "action_map"}
    missing = required - set(meta)
    if missing:
        raise KeyError(f"policy_meta.json missing keys: {missing}")
    return meta


def _load_policy_robust(policy_path: Path, meta: Dict) -> DiscreteCQL:
    """Load DiscreteCQL policy supporting both save() and save_model() formats."""
    logger = logging.getLogger("hybrid_walkforward")

    try:
        logger.info("Attempting load_learnable …")
        algo = load_learnable(str(policy_path))
        logger.info("✅ Loaded policy with load_learnable")
        return algo
    except Exception as e1:
        logger.warning("load_learnable failed: %s", e1)
        logger.info("Falling back to manual reconstruction …")

        try:
            model_data = torch.load(str(policy_path), map_location="cpu")
            if "q_funcs" not in model_data:
                raise ValueError("Unrecognized model format: missing q_funcs")

            # Determine action dimension
            q_keys = [k for k in model_data["q_funcs"] if k.endswith("._fc.weight")]
            if q_keys:
                action_size = model_data["q_funcs"][q_keys[0]].shape[0]
            else:
                action_size = len(meta["action_map"])

            # Determine observation dimension
            encoder_keys = [k for k in model_data["q_funcs"] if k.endswith("._encoder._layers.0.weight")]
            if encoder_keys:
                observation_size = model_data["q_funcs"][encoder_keys[0]].shape[1]
            else:
                observation_size = len(meta["state_columns"])

            # Number of critics stored
            n_critics = len({k.split(".")[0] for k in model_data["q_funcs"].keys()})
            logger.info("Detected architecture: obs=%d, actions=%d, critics=%d", observation_size, action_size, n_critics)

            config_path = policy_path.parent.parent / "config.yaml"
            if config_path.exists():
                with config_path.open("r", encoding="utf-8") as fh:
                    training_cfg = yaml.safe_load(fh)
                config = DiscreteCQLConfig(
                    learning_rate=training_cfg.get("learning_rate", 3e-4),
                    gamma=training_cfg.get("gamma", 0.99),
                    batch_size=training_cfg.get("batch_size", 256),
                    n_critics=n_critics,
                )
            else:
                config = DiscreteCQLConfig(n_critics=n_critics)

            algo = config.create()
            algo.create_impl((observation_size,), action_size)
            algo.load_model(str(policy_path))
            logger.info("✅ Loaded policy via manual reconstruction")
            return algo
        except Exception as e2:
            logger.error("Manual reconstruction failed: %s", e2)
            raise RuntimeError(
                f"Failed to load policy. load_learnable error: {e1}. Manual reconstruction error: {e2}"
            )


def _standardise_states(df: pd.DataFrame, state_cols: List[str], mean: List[float], scale: List[float]) -> np.ndarray:
    numeric_df = df[state_cols].apply(pd.to_numeric, errors="coerce")
    values = numeric_df.to_numpy(dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    mean_arr = np.asarray(mean, dtype=np.float32)
    scale_arr = np.asarray(scale, dtype=np.float32)
    if values.shape[1] != mean_arr.shape[0]:
        raise ValueError("Scaler mean length does not match state dimension")
    denom = np.where(scale_arr == 0.0, 1.0, scale_arr)
    return (values - mean_arr) / denom


def _decode_action(action_id: int, action_map: Dict[str, Dict[str, float]]) -> Tuple[int, float]:
    info = action_map.get(str(action_id))
    if info is None:
        raise KeyError(f"Action id {action_id} not found in action map")
    return int(info.get("slot", 0)), float(info.get("size_value", 0.0))


def _hybrid_position_sizing(
    equity: float,
    slot: int,
    size_mult: float,
    vol_regime_mult: float = 1.0,
    drawdown: float = 0.0,
    base_contracts: int = 5,  # More reasonable base than 10
    equity_scaling: bool = True
) -> int:
    """
    Hybrid position sizing: Original-style base contracts with enhanced risk controls.
    
    This bridges your original success (fixed position sizes) with proper risk management.
    """
    if slot <= 0 or size_mult <= 0:
        return 0
    
    # Start with reasonable base position (like original but smaller)
    base_position = base_contracts
    
    # Apply RL action size multiplier
    sized_position = int(np.floor(base_position * size_mult))
    
    # Apply vol regime scaling (key enhancement!)
    vol_adjusted = int(np.floor(sized_position * vol_regime_mult))
    
    # Apply drawdown scaling (reduce during losses)
    drawdown_scale = max(0.3, 1.0 - (drawdown * 1.5))
    drawdown_adjusted = int(np.floor(vol_adjusted * drawdown_scale))
    
    # Optional: Scale by equity growth (let winners run a bit)
    if equity_scaling and equity > 15000:  # If up 50%+
        equity_mult = min(1.5, equity / 10000)  # Max 1.5x for growth
        final_position = int(np.floor(drawdown_adjusted * equity_mult))
    else:
        final_position = drawdown_adjusted
    
    return max(final_position, 0)


def _hybrid_notional(n_contracts: int, base_notional: float = 1000.0) -> float:
    """Use original-style fixed notional but allow it to be configurable."""
    return base_notional * n_contracts


def _fee_cost(n_contracts: int, commission_per_side: float, exchange_fee_per_side: float) -> float:
    per_side = (commission_per_side + exchange_fee_per_side) * n_contracts
    return 2.0 * per_side


def _slippage_cost(row: pd.Series, n_contracts: int, slip_min: float, slip_pct: float) -> float:
    spread = float(row.get("bid_ask_spread", 0.02) or 0.02)
    slip_per_contract = max(slip_min, slip_pct * spread)
    return slip_per_contract * 100.0 * n_contracts * 2.0


def simulate_hybrid_walkforward(
    decision_df: pd.DataFrame,
    predicted_actions: np.ndarray,
    action_map: Dict[str, Dict[str, float]],
    initial_capital: float = 10_000.0,
    commission_per_side: float = 0.65,
    exchange_fee_per_side: float = 0.05,
    slippage_min: float = 0.02,
    slippage_pct: float = 0.20,
    base_contracts: int = 5,  # Conservative than original 10
    base_notional: float = 1000.0,  # Match original
    enable_risk_controls: bool = True,
    max_notional_pct: float = 0.20,  # 20% max portfolio allocation
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Hybrid walkforward: Original mechanics + essential risk controls.
    """
    logger = logging.getLogger("hybrid_walkforward")
    df = decision_df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.sort_values('date', inplace=True)

    equity = initial_capital
    history = []
    equity_series = pd.Series([equity])
    
    logger.info(f"Hybrid simulation: base_contracts={base_contracts}, base_notional=${base_notional}")
    logger.info(f"Risk controls enabled: {enable_risk_controls}")

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

        # Enhanced risk calculations (if enabled)
        if enable_risk_controls:
            current_drawdown = _calculate_rolling_drawdown(equity_series, 20)
            vol_severity = float(row.get('s_vol_severity', 1.0))
            vol_emergency = bool(row.get('s_vol_emergency', False))
            vol_regime_mult = _get_volatility_regime_multiplier(vol_severity, vol_emergency)
        else:
            current_drawdown = 0.0
            vol_regime_mult = 1.0

        # Hybrid position sizing
        n_contracts = _hybrid_position_sizing(
            equity=equity,
            slot=slot,
            size_mult=size_mult,
            vol_regime_mult=vol_regime_mult,
            drawdown=current_drawdown,
            base_contracts=base_contracts,
            equity_scaling=enable_risk_controls
        )

        # Calculate notional
        notional = _hybrid_notional(n_contracts, base_notional)
        
        # Apply max notional cap
        if enable_risk_controls:
            notional_cap = equity * max_notional_pct
            if notional > notional_cap:
                scale = notional_cap / notional if notional > 0 else 0.0
                n_contracts = int(np.floor(n_contracts * scale))
                notional = _hybrid_notional(n_contracts, base_notional)

        # Calculate costs (same as original)
        fees = _fee_cost(n_contracts, commission_per_side, exchange_fee_per_side)
        slippage = _slippage_cost(row, n_contracts, slippage_min, slippage_pct)

        # Calculate P&L (same as original)
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
    parser = argparse.ArgumentParser(description="Hybrid Walk-forward: Original performance + Enhanced risk controls")
    parser.add_argument('--decision-table', type=Path, required=True)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--meta', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, default=Path('hybrid_walkforward_results'))
    parser.add_argument('--initial-capital', type=float, default=10_000.0)
    parser.add_argument('--commission', type=float, default=0.65)
    parser.add_argument('--exchange-fee', type=float, default=0.05)
    parser.add_argument('--slippage-min', type=float, default=0.02)
    parser.add_argument('--slippage-pct', type=float, default=0.20)
    parser.add_argument('--base-contracts', type=int, default=5, help="Base number of contracts per trade")
    parser.add_argument('--base-notional', type=float, default=1000.0, help="Base notional per contract ($)")
    parser.add_argument('--max-notional-pct', type=float, default=0.20)
    parser.add_argument('--enable-risk-controls', action='store_true', default=True)
    parser.add_argument('--disable-risk-controls', dest='enable_risk_controls', action='store_false')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("hybrid_walkforward")

    logger.info("Loading decision table and policy...")
    df = pd.read_csv(args.decision_table)
    meta = _load_meta(args.meta)
    states = _standardise_states(df, meta['state_columns'], meta['scaler_mean'], meta['scaler_scale'])
    algo = _load_policy_robust(args.policy, meta)
    predicted_actions = algo.predict(states)

    logger.info("Running hybrid walkforward simulation...")
    results, summary = simulate_hybrid_walkforward(
        df,
        predicted_actions,
        meta['action_map'],
        initial_capital=args.initial_capital,
        commission_per_side=args.commission,
        exchange_fee_per_side=args.exchange_fee,
        slippage_min=args.slippage_min,
        slippage_pct=args.slippage_pct,
        base_contracts=args.base_contracts,
        base_notional=args.base_notional,
        enable_risk_controls=args.enable_risk_controls,
        max_notional_pct=args.max_notional_pct,
    )

    # Save results
    args.outdir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.outdir / 'hybrid_walkforward_trades.csv', index=False)
    with (args.outdir / 'hybrid_walkforward_summary.json').open('w', encoding='utf-8') as fh:
        json.dump(summary, fh, indent=2)

    logger.info("🎯 Hybrid Walk-forward Summary:")
    logger.info(f"  Return: {summary['return_pct']:.1f}% (target: 200-600% with better risk control)")
    logger.info(f"  Win Rate: {summary['win_rate']:.1%}")
    logger.info(f"  Max Drawdown: {summary['max_drawdown']:.1%}")
    logger.info(f"  Total Trades: {summary['total_trades']}")
    logger.info(f"  Profit Factor: {summary.get('profit_factor', 'N/A')}")
    logger.info(f"  Risk Controls: {summary['risk_controls_enabled']}")


if __name__ == '__main__':
    main()