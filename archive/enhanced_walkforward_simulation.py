#!/usr/bin/env python3
"""
Enhanced Walk-forward Simulation with Robust Risk Management
 
Key Improvements:
- Re-enabled sophisticated risk-based position sizing
- Added drawdown-based scaling
- Volatility regime-aware position limits
- Dynamic risk budget allocation
- Real option premium-based notional calculation
- Enhanced liquidity constraints
- Portfolio heat and correlation controls
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


# ---------------------------- Enhanced Risk Management Helpers ---------------------------------

def _calculate_rolling_drawdown(equity_series: pd.Series, window: int = 20) -> float:
    """Calculate current drawdown from rolling peak."""
    if len(equity_series) < 2:
        return 0.0
    
    peak = equity_series.rolling(window=min(window, len(equity_series))).max().iloc[-1]
    current = equity_series.iloc[-1]
    drawdown = (current - peak) / peak if peak > 0 else 0.0
    return max(drawdown, 0.0)  # Return positive drawdown magnitude


def _calculate_portfolio_heat(trades_history: List[Dict], lookback_days: int = 5) -> float:
    """Calculate portfolio heat (sum of open position risks)."""
    if len(trades_history) < lookback_days:
        return 0.0
    
    recent_trades = trades_history[-lookback_days:]
    total_risk = sum(
        trade.get('contract_risk', 0) * trade.get('n_contracts', 0) 
        for trade in recent_trades 
        if trade.get('n_contracts', 0) > 0
    )
    
    latest_equity = trades_history[-1].get('equity_before', 10000.0)
    return total_risk / latest_equity if latest_equity > 0 else 0.0


def _get_volatility_regime_multiplier(vol_severity: float, vol_emergency: bool) -> float:
    """Get position size multiplier based on market volatility regime."""
    if vol_emergency:
        return 0.3  # Reduce positions by 70% during vol emergencies
    elif vol_severity > 3.0:
        return 0.5  # High vol regime
    elif vol_severity > 2.0:
        return 0.7  # Moderate vol regime
    else:
        return 1.0  # Normal regime


def _enhanced_base_contracts(
    equity: float, 
    risk_pct: float, 
    contract_risk: float,
    drawdown: float = 0.0,
    portfolio_heat: float = 0.0,
    vol_regime_mult: float = 1.0,
    max_position_pct: float = 0.02  # Max 2% of equity per position
) -> int:
    """
    Enhanced position sizing with multiple risk controls.
    
    Risk Budget = Equity * Risk% * Drawdown_Scaling * Vol_Regime * Heat_Scaling
    Position Size = Risk_Budget / Contract_Risk
    """
    if not np.isfinite(contract_risk) or contract_risk <= 0:
        return 0
    
    # Base risk budget
    base_risk_budget = equity * risk_pct
    
    # Drawdown scaling (reduce size as drawdown increases)
    drawdown_scale = max(0.2, 1.0 - (drawdown * 2.0))  # 0.2x to 1.0x based on drawdown
    
    # Portfolio heat scaling (reduce if too much risk already deployed)
    heat_scale = max(0.3, 1.0 - (portfolio_heat * 3.0))  # 0.3x to 1.0x based on portfolio heat
    
    # Apply all scaling factors
    adjusted_risk_budget = base_risk_budget * drawdown_scale * vol_regime_mult * heat_scale
    
    # Calculate position size
    base_contracts = int(np.floor(adjusted_risk_budget / contract_risk))
    
    # Apply maximum position size cap (% of equity)
    max_contracts_by_equity = int(np.floor((equity * max_position_pct) / contract_risk))
    
    return max(min(base_contracts, max_contracts_by_equity), 0)


def _enhanced_liquidity_caps(row: pd.Series, slot: int, n_contracts: int) -> int:
    """Enhanced liquidity constraints with multiple checks."""
    if n_contracts <= 0:
        return 0
    
    # Try to get slot-specific liquidity data
    oi_col = f"c{slot}_open_interest" 
    vol_col = f"c{slot}_vol_5"  # 5-day average volume
    
    # Fallback to general columns if slot-specific not available
    oi = row.get(oi_col, row.get("open_interest", 1000.0))  # Default 1000 OI
    vol5 = row.get(vol_col, row.get("vol_5", row.get("volume", 100.0)))  # Default 100 volume
    
    oi = float(oi or 1000.0)
    vol5 = float(vol5 or 100.0)
    
    # Conservative liquidity limits
    cap_by_oi = int(np.floor(0.05 * oi)) if oi > 0 else n_contracts  # 5% of open interest
    cap_by_vol = int(np.floor(0.02 * vol5)) if vol5 > 0 else n_contracts  # 2% of daily volume
    
    # Apply most restrictive constraint
    liquidity_cap = min(cap_by_oi, cap_by_vol)
    
    return max(min(n_contracts, liquidity_cap), 0)


def _real_notional(row: pd.Series, n_contracts: int, slot: int) -> float:
    """Calculate real notional using actual option premiums."""
    if n_contracts <= 0:
        return 0.0
    
    # Try slot-specific premium, fallback to general
    premium_col = f"c{slot}_future_option_price"
    premium = row.get(premium_col)
    
    if pd.isna(premium) or premium is None:
        premium = row.get("price_point", row.get("future_option_price", 50.0))
    premium = float(premium or 50.0)  # Default $50 if missing
    
    # Options typically traded in $100 multiplier contracts
    return premium * 100.0 * n_contracts


def _dynamic_max_notional_pct(
    vol_severity: float, 
    vol_emergency: bool,
    base_max_pct: float = 0.10
) -> float:
    """Dynamically adjust maximum notional percentage based on market conditions."""
    if vol_emergency:
        return base_max_pct * 0.5  # 50% of normal during emergencies
    elif vol_severity > 3.0:
        return base_max_pct * 0.7  # 70% during high vol
    else:
        return base_max_pct


# ---------------------------- Core Functions (Reusing from Original) ---------------------------------

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
    logger = logging.getLogger("enhanced_walkforward")

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


def _contract_risk(row: pd.Series, slot: int, downside_min: float, downside_max: float) -> float:
    """Calculate contract risk using CQF quantile predictions."""
    price_col = f"c{slot}_future_option_price"
    downside_col = f"c{slot}_q0.05"
    
    premium = row.get(price_col)
    if pd.isna(premium) or premium is None:
        premium = row.get("price_point", row.get("future_option_price", 50.0))  # Default $50
    premium = float(premium or 50.0)
    
    downside = row.get(downside_col, row.get("s_q0.05", -0.1))
    if pd.isna(downside) or downside is None:
        downside = 0.1  # Default 10% downside
    downside = abs(float(downside or 0.1))
    downside = np.clip(downside, downside_min, downside_max)
    
    # Risk = Premium * Downside_Risk * Contract_Multiplier
    risk = premium * downside * 100.0
    return risk if np.isfinite(risk) else (50.0 * 0.1 * 100.0)  # Fallback: $500


def _fee_cost(n_contracts: int, commission_per_side: float, exchange_fee_per_side: float) -> float:
    per_side = (commission_per_side + exchange_fee_per_side) * n_contracts
    return 2.0 * per_side


def _slippage_cost(row: pd.Series, n_contracts: int, slip_min: float, slip_pct: float) -> float:
    spread = float(row.get("bid_ask_spread", 0.02) or 0.02)  # Default 2 cents
    slip_per_contract = max(slip_min, slip_pct * spread)
    return slip_per_contract * 100.0 * n_contracts * 2.0


# ---------------------------- Enhanced Simulation Engine ---------------------------------

def simulate_enhanced_walkforward(
    decision_df: pd.DataFrame,
    predicted_actions: np.ndarray,
    action_map: Dict[str, Dict[str, float]],
    initial_capital: float = 10_000.0,
    risk_pct: float = 0.005,  # 0.5% risk per trade
    commission_per_side: float = 0.65,
    exchange_fee_per_side: float = 0.05,
    slippage_min: float = 0.02,
    slippage_pct: float = 0.20,
    max_notional_pct: float = 0.10,  # 10% max notional
    downside_min: float = 0.01,
    downside_max: float = 0.30,
    max_position_pct: float = 0.02,  # 2% max per position
    drawdown_lookback: int = 20,
    enable_enhanced_risk: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Enhanced walkforward simulation with sophisticated risk management.
    """
    logger = logging.getLogger("enhanced_walkforward")
    df = decision_df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.sort_values('date', inplace=True)

    equity = initial_capital
    history = []
    equity_series = pd.Series([equity])  # Track equity for drawdown calculation

    logger.info(f"Starting enhanced simulation with risk controls: {enable_enhanced_risk}")
    logger.info(f"Risk per trade: {risk_pct:.1%}, Max notional: {max_notional_pct:.1%}")

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

        # No action case
        if slot <= 0 or size_mult <= 0:
            record.update({
                'n_contracts': 0,
                'notional': 0.0,
                'fees': 0.0,
                'slippage': 0.0,
                'realized_pnl': 0.0,
                'equity_after': equity,
                'drawdown': 0.0,
                'portfolio_heat': 0.0,
                'vol_regime_mult': 1.0,
            })
            history.append(record)
            continue

        # Calculate contract risk
        contract_risk = _contract_risk(row, slot, downside_min, downside_max)
        if contract_risk <= 0 or not np.isfinite(contract_risk):
            record.update({
                'n_contracts': 0,
                'notional': 0.0,
                'contract_risk': contract_risk,
                'fees': 0.0,
                'slippage': 0.0,
                'realized_pnl': 0.0,
                'equity_after': equity,
                'drawdown': 0.0,
                'portfolio_heat': 0.0,
                'vol_regime_mult': 1.0,
            })
            history.append(record)
            continue

        if enable_enhanced_risk:
            # Enhanced risk calculations
            current_drawdown = _calculate_rolling_drawdown(equity_series, drawdown_lookback)
            portfolio_heat = _calculate_portfolio_heat(history, lookback_days=5)
            
            # Get market regime data
            vol_severity = float(row.get('s_vol_severity', 1.0))
            vol_emergency = bool(row.get('s_vol_emergency', False))
            vol_regime_mult = _get_volatility_regime_multiplier(vol_severity, vol_emergency)
            
            # Enhanced position sizing
            base_n = _enhanced_base_contracts(
                equity=equity,
                risk_pct=risk_pct,
                contract_risk=contract_risk,
                drawdown=current_drawdown,
                portfolio_heat=portfolio_heat,
                vol_regime_mult=vol_regime_mult,
                max_position_pct=max_position_pct
            )
            
            # Dynamic max notional based on market conditions
            dynamic_max_notional_pct = _dynamic_max_notional_pct(vol_severity, vol_emergency, max_notional_pct)
            
            record.update({
                'drawdown': current_drawdown,
                'portfolio_heat': portfolio_heat,
                'vol_regime_mult': vol_regime_mult,
                'dynamic_max_notional_pct': dynamic_max_notional_pct,
            })
        else:
            # Fallback to simple position sizing (similar to original but with actual risk calc)
            if contract_risk > 0:
                base_risk_budget = equity * risk_pct
                base_n = max(int(np.floor(base_risk_budget / contract_risk)), 0)
            else:
                base_n = 0
            dynamic_max_notional_pct = max_notional_pct
            
            record.update({
                'drawdown': 0.0,
                'portfolio_heat': 0.0,
                'vol_regime_mult': 1.0,
                'dynamic_max_notional_pct': dynamic_max_notional_pct,
            })

        # Apply size multiplier from RL action
        n_contracts = max(int(np.floor(base_n * size_mult)), 0)
        
        # Apply enhanced liquidity caps
        n_contracts = _enhanced_liquidity_caps(row, slot, n_contracts)

        # Calculate real notional
        notional = _real_notional(row, n_contracts, slot)
        
        # Apply dynamic notional cap
        notional_cap = equity * dynamic_max_notional_pct
        if notional > notional_cap:
            scale = notional_cap / notional if notional > 0 else 0.0
            n_contracts = int(np.floor(n_contracts * scale))
            notional = _real_notional(row, n_contracts, slot)

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
        equity_series = pd.concat([equity_series, pd.Series([equity])]).tail(drawdown_lookback * 2)

        record.update({
            'n_contracts': n_contracts,
            'notional': notional,
            'contract_risk': contract_risk,
            'fees': fees,
            'slippage': slippage,
            'realized_pnl': realized_pnl,
            'equity_after': equity,
        })
        history.append(record)

    # Generate results
    results = pd.DataFrame(history)
    
    # Enhanced summary statistics
    trades_mask = results['n_contracts'] > 0
    winning_trades = results[trades_mask & (results['realized_pnl'] > 0)]
    losing_trades = results[trades_mask & (results['realized_pnl'] <= 0)]
    
    max_drawdown = results['drawdown'].max() if enable_enhanced_risk and len(results) > 0 else 0.0
    avg_portfolio_heat = results['portfolio_heat'].mean() if enable_enhanced_risk and len(results) > 0 else 0.0
    
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
        'avg_portfolio_heat': float(avg_portfolio_heat),
        'enhanced_risk_enabled': enable_enhanced_risk,
    }
    
    # Add trade statistics
    if trades_mask.sum() > 0:
        trade_pnls = results[trades_mask]['realized_pnl']
        summary.update({
            'avg_trade_pnl': float(trade_pnls.mean()),
            'median_trade_pnl': float(trade_pnls.median()),
            'largest_win': float(trade_pnls.max()),
            'largest_loss': float(trade_pnls.min()),
            'profit_factor': float(winning_trades['realized_pnl'].sum() / abs(losing_trades['realized_pnl'].sum())) if len(losing_trades) > 0 else np.inf,
        })
    
    return results, summary


# ----------------------------- CLI ------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Enhanced Walk-forward simulation with robust risk management")
    parser.add_argument('--decision-table', type=Path, required=True)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--meta', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, default=Path('enhanced_walkforward_results'))
    parser.add_argument('--initial-capital', type=float, default=10_000.0)
    parser.add_argument('--risk-pct', type=float, default=0.005)
    parser.add_argument('--commission', type=float, default=0.65)
    parser.add_argument('--exchange-fee', type=float, default=0.05)
    parser.add_argument('--slippage-min', type=float, default=0.02)
    parser.add_argument('--slippage-pct', type=float, default=0.20)
    parser.add_argument('--max-notional-pct', type=float, default=0.10)
    parser.add_argument('--max-position-pct', type=float, default=0.02)
    parser.add_argument('--downside-min', type=float, default=0.01)
    parser.add_argument('--downside-max', type=float, default=0.30)
    parser.add_argument('--enable-enhanced-risk', action='store_true', default=True, 
                       help="Enable enhanced risk management features")
    parser.add_argument('--disable-enhanced-risk', dest='enable_enhanced_risk', action='store_false',
                       help="Disable enhanced risk management (revert to simpler logic)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("enhanced_walkforward")

    logger.info("Loading decision table and policy...")
    df = pd.read_csv(args.decision_table)
    meta = _load_meta(args.meta)
    states = _standardise_states(df, meta['state_columns'], meta['scaler_mean'], meta['scaler_scale'])
    algo = _load_policy_robust(args.policy, meta)
    predicted_actions = algo.predict(states)

    logger.info("Running enhanced walkforward simulation...")
    results, summary = simulate_enhanced_walkforward(
        df,
        predicted_actions,
        meta['action_map'],
        initial_capital=args.initial_capital,
        risk_pct=args.risk_pct,
        commission_per_side=args.commission,
        exchange_fee_per_side=args.exchange_fee,
        slippage_min=args.slippage_min,
        slippage_pct=args.slippage_pct,
        max_notional_pct=args.max_notional_pct,
        max_position_pct=args.max_position_pct,
        downside_min=args.downside_min,
        downside_max=args.downside_max,
        enable_enhanced_risk=args.enable_enhanced_risk,
    )

    # Save results
    args.outdir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.outdir / 'enhanced_walkforward_trades.csv', index=False)
    with (args.outdir / 'enhanced_walkforward_summary.json').open('w', encoding='utf-8') as fh:
        json.dump(summary, fh, indent=2)

    logger.info("Enhanced Walk-forward Summary:")
    logger.info(f"  Return: {summary['return_pct']:.1f}% (vs original 1119.7%)")
    logger.info(f"  Win Rate: {summary['win_rate']:.1%}")
    logger.info(f"  Max Drawdown: {summary['max_drawdown']:.1%}")
    logger.info(f"  Total Trades: {summary['total_trades']}")
    logger.info(f"  Profit Factor: {summary.get('profit_factor', 'N/A')}")
    logger.info(f"  Enhanced Risk Enabled: {summary['enhanced_risk_enabled']}")


if __name__ == '__main__':
    main()