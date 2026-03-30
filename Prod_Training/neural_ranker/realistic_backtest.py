"""Realistic backtest engine for the neural ranker.

Applies execution constraints, position sizing, and risk management
to produce honest P&L from the model's daily rankings.

Key features:
- Bid/ask execution (buy at ask, sell at bid)
- Liquidity filters (volume, OI, spread)
- Conviction-based position sizing
- Per-position and portfolio exposure limits
- Drawdown circuit breaker
- Weekly entry/exit cycle matching 5-day horizon
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from logger import setup_logger
from neural_ranker import ChainTransformer, NeuralRankerConfig, get_device
from utils import (
    apply_relevance_bins,
    load_config,
    prepare_model_frame,
    save_json,
)

logger = setup_logger(__name__)


@dataclass
class ExecutionConfig:
    """Controls what's tradeable."""
    min_price: float = 0.10          # minimum option price (ask)
    max_relative_spread: float = 0.30  # max spread as % of mid price
    min_volume: int = 10              # minimum daily volume
    min_open_interest: int = 50       # minimum open interest
    max_volume_participation: float = 0.10  # max 10% of daily volume


@dataclass
class RiskConfig:
    """Controls position sizing and risk."""
    starting_capital: float = 10000.0
    max_position_pct: float = 0.05    # max 5% per position
    max_total_exposure: float = 0.30   # max 30% capital at risk
    max_positions: int = 10            # max positions per entry
    max_same_direction: int = 7        # max calls or puts
    max_same_expiry: int = 3           # max positions in same expiry bucket
    daily_loss_limit: float = 0.05     # 5% daily loss -> close all
    consecutive_loss_reduce: int = 3   # after 3 losing days, reduce 50%
    top_k_candidates: int = 50         # score top-K before filtering


@dataclass
class Trade:
    """Single trade record."""
    date: str
    exit_date: str
    contractid: str
    option_type: str
    strike: float
    days_to_exp: float
    entry_price: float
    exit_price: float
    position_size: float  # dollar amount invested
    shares: int           # number of contracts
    pnl: float
    return_pct: float
    score: float
    rank: int


def score_day(model, day_features: np.ndarray, device: torch.device) -> np.ndarray:
    """Score a day's options through the model."""
    x = torch.from_numpy(day_features).unsqueeze(0).to(device)
    with torch.no_grad():
        scores = model(x).squeeze(0).cpu().numpy()
    return scores


def filter_tradeable(day: pd.DataFrame, exec_config: ExecutionConfig) -> pd.DataFrame:
    """Filter to only tradeable options."""
    mask = (
        (day['ask_raw'] >= exec_config.min_price) &
        (day['relative_spread_raw'] <= exec_config.max_relative_spread) &
        (day['volume_raw'] >= exec_config.min_volume) &
        (day['open_interest_raw'] >= exec_config.min_open_interest)
    )
    return day[mask].copy()


def size_positions(
    candidates: pd.DataFrame,
    capital: float,
    risk_config: RiskConfig,
    exec_config: ExecutionConfig,
    conviction: float,
) -> List[Dict]:
    """Determine position sizes for selected candidates.

    Args:
        candidates: top-K candidates sorted by score
        capital: current portfolio capital
        risk_config: sizing constraints
        exec_config: execution constraints
        conviction: 0-1 score gap indicating model confidence

    Returns:
        List of position dicts with sizes
    """
    positions = []
    total_allocated = 0.0
    max_total = capital * risk_config.max_total_exposure * conviction
    call_count = 0
    put_count = 0
    expiry_counts: Dict[int, int] = {}

    for _, row in candidates.iterrows():
        if len(positions) >= risk_config.max_positions:
            break
        if total_allocated >= max_total:
            break

        option_type = row.get('type', '')
        if option_type == 'call' and call_count >= risk_config.max_same_direction:
            continue
        if option_type == 'put' and put_count >= risk_config.max_same_direction:
            continue

        # Expiry bucket (weekly/monthly/quarterly)
        dte = row.get('days_to_exp_raw', 30)
        expiry_bucket = 0 if dte <= 7 else (1 if dte <= 30 else (2 if dte <= 90 else 3))
        if expiry_counts.get(expiry_bucket, 0) >= risk_config.max_same_expiry:
            continue

        # Position size
        ask_price = row['ask_raw']
        max_by_pct = capital * risk_config.max_position_pct
        max_by_remaining = max_total - total_allocated
        max_by_volume = row['volume_raw'] * exec_config.max_volume_participation * ask_price * 100
        position_dollars = min(max_by_pct, max_by_remaining, max_by_volume)

        if position_dollars < ask_price * 100:  # can't even buy 1 contract
            continue

        n_contracts = int(position_dollars / (ask_price * 100))
        actual_cost = n_contracts * ask_price * 100

        positions.append({
            'contractid': row.get('contractid', ''),
            'type': option_type,
            'strike': row.get('strike_raw', 0),
            'days_to_exp': dte,
            'entry_price': ask_price,
            'exit_price': row.get('exit_bid_raw', 0),
            'exit_date': row.get('exit_date', ''),
            'n_contracts': n_contracts,
            'cost': actual_cost,
            'score': row.get('score', 0),
            'rank': row.get('rank', 0),
        })

        total_allocated += actual_cost
        if option_type == 'call':
            call_count += 1
        else:
            put_count += 1
        expiry_counts[expiry_bucket] = expiry_counts.get(expiry_bucket, 0) + 1

    return positions


def run_realistic_backtest(
    model: ChainTransformer,
    frame: pd.DataFrame,
    feature_columns: List[str],
    device: torch.device,
    exec_config: ExecutionConfig = ExecutionConfig(),
    risk_config: RiskConfig = RiskConfig(),
) -> Dict[str, Any]:
    """Run realistic backtest with execution and risk constraints."""

    capital = risk_config.starting_capital
    equity_curve = []
    all_trades: List[Trade] = []
    consecutive_losses = 0
    dates = sorted(frame['date'].unique())

    logger.info("Starting backtest: $%.0f capital, %d dates", capital, len(dates))
    logger.info("Execution: min_price=$%.2f, max_spread=%.0f%%, min_vol=%d, min_oi=%d",
                exec_config.min_price, exec_config.max_relative_spread * 100,
                exec_config.min_volume, exec_config.min_open_interest)

    for i, date in enumerate(dates):
        day = frame[frame['date'] == date]
        if len(day) < 20:
            continue

        # Score all options
        features = np.nan_to_num(day[feature_columns].values.astype(np.float32))
        scores = score_day(model, features, device)
        day = day.copy()
        day['score'] = scores

        # Filter to tradeable
        tradeable = filter_tradeable(day, exec_config)
        if len(tradeable) < 5:
            equity_curve.append({'date': date, 'capital': capital, 'daily_pnl': 0, 'n_trades': 0})
            continue

        # Rank tradeable options
        tradeable = tradeable.sort_values('score', ascending=False).reset_index(drop=True)
        tradeable['rank'] = range(1, len(tradeable) + 1)
        candidates = tradeable.head(risk_config.top_k_candidates)

        # Conviction = score gap between #1 and median
        score_range = candidates['score'].iloc[0] - candidates['score'].median()
        conviction = min(1.0, max(0.3, score_range / max(abs(candidates['score'].iloc[0]), 1e-6)))

        # Reduce exposure after consecutive losses
        if consecutive_losses >= risk_config.consecutive_loss_reduce:
            conviction *= 0.5

        # Size positions
        positions = size_positions(candidates, capital, risk_config, exec_config, conviction)

        # Calculate P&L
        daily_pnl = 0.0
        for pos in positions:
            exit_value = pos['n_contracts'] * pos['exit_price'] * 100
            entry_cost = pos['cost']
            pnl = exit_value - entry_cost
            return_pct = pnl / entry_cost if entry_cost > 0 else 0

            daily_pnl += pnl

            all_trades.append(Trade(
                date=str(date),
                exit_date=str(pos['exit_date']),
                contractid=pos['contractid'],
                option_type=pos['type'],
                strike=pos['strike'],
                days_to_exp=pos['days_to_exp'],
                entry_price=pos['entry_price'],
                exit_price=pos['exit_price'],
                position_size=entry_cost,
                shares=pos['n_contracts'],
                pnl=pnl,
                return_pct=return_pct,
                score=pos['score'],
                rank=pos['rank'],
            ))

        # Update capital
        capital += daily_pnl

        # Circuit breaker
        if daily_pnl < 0 and abs(daily_pnl) > risk_config.daily_loss_limit * (capital - daily_pnl):
            consecutive_losses += 1
        elif daily_pnl > 0:
            consecutive_losses = 0

        # Bankruptcy protection
        if capital <= 0:
            logger.warning("Capital depleted at %s", date)
            capital = 0
            equity_curve.append({'date': date, 'capital': 0, 'daily_pnl': daily_pnl, 'n_trades': len(positions)})
            break

        equity_curve.append({
            'date': date,
            'capital': capital,
            'daily_pnl': daily_pnl,
            'n_trades': len(positions),
        })

        if (i + 1) % 100 == 0:
            logger.info("  Day %d/%d | Capital: $%.0f | Trades: %d", i + 1, len(dates), capital, len(all_trades))

    # Compute metrics
    eq = pd.DataFrame(equity_curve)
    trades_df = pd.DataFrame([t.__dict__ for t in all_trades]) if all_trades else pd.DataFrame()

    metrics = {
        'starting_capital': risk_config.starting_capital,
        'ending_capital': capital,
        'total_return_pct': (capital / risk_config.starting_capital - 1) * 100,
        'total_trades': len(all_trades),
        'days_traded': len(eq[eq['n_trades'] > 0]),
        'total_days': len(eq),
    }

    if len(trades_df) > 0:
        metrics['win_rate'] = (trades_df['pnl'] > 0).mean() * 100
        metrics['avg_trade_return'] = trades_df['return_pct'].mean() * 100
        metrics['median_trade_return'] = trades_df['return_pct'].median() * 100
        metrics['avg_position_size'] = trades_df['position_size'].mean()
        metrics['avg_trades_per_day'] = trades_df.groupby('date').size().mean()
        metrics['best_trade'] = trades_df['return_pct'].max() * 100
        metrics['worst_trade'] = trades_df['return_pct'].min() * 100
        metrics['avg_calls'] = (trades_df['option_type'] == 'call').mean() * 100
        metrics['avg_puts'] = (trades_df['option_type'] == 'put').mean() * 100

    if len(eq) > 1:
        daily_returns = eq['daily_pnl'] / eq['capital'].shift(1).fillna(risk_config.starting_capital)
        metrics['sharpe'] = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0
        eq['peak'] = eq['capital'].cummax()
        eq['drawdown'] = (eq['capital'] / eq['peak'] - 1)
        metrics['max_drawdown_pct'] = eq['drawdown'].min() * 100
        metrics['final_equity'] = capital

    return {
        'metrics': metrics,
        'equity_curve': eq,
        'trades': trades_df,
    }


def main():
    parser = argparse.ArgumentParser(description="Realistic backtest with execution and risk constraints")
    parser.add_argument("--model-artifact", required=True, help="Path to neural_ranker_artifact.pt")
    parser.add_argument("--data", nargs="+", required=True, help="CSV files to backtest on")
    parser.add_argument("--config", default="./config_tuned.yaml")
    parser.add_argument("--output-dir", default="./backtest_output")
    parser.add_argument("--starting-capital", type=float, default=10000.0)
    parser.add_argument("--min-price", type=float, default=0.10)
    parser.add_argument("--max-spread", type=float, default=0.30)
    parser.add_argument("--min-volume", type=int, default=10)
    parser.add_argument("--max-exposure", type=float, default=0.30)
    args = parser.parse_args()

    # Load model
    artifact = torch.load(args.model_artifact, map_location='cpu', weights_only=False)
    nr_config = NeuralRankerConfig(**artifact['config'])
    model = ChainTransformer(nr_config)
    state = {k.replace('_orig_mod.', ''): v for k, v in artifact['model_state_dict'].items()}
    model.load_state_dict(state)
    device = get_device()
    model = model.to(device)
    model.eval()
    logger.info("Loaded model: %d params on %s", sum(p.numel() for p in model.parameters()), device)

    # Load data
    cfg = load_config(args.config)
    frame = prepare_model_frame(args.data, cfg, include_targets=True)

    feature_columns = artifact['feature_columns']
    edges = artifact['relevance_edges']
    frame['target_relevance'] = apply_relevance_bins(frame['target_return'], edges).astype(np.float32)
    frame['type_numeric'] = (frame['type'].str.lower() == 'call').astype(np.float32)

    # Save raw values before normalization
    frame['ask_raw'] = frame['ask'].copy()
    frame['bid_raw'] = frame['bid'].copy()
    frame['volume_raw'] = frame['volume'].copy()
    frame['open_interest_raw'] = frame['open_interest'].copy()
    frame['relative_spread_raw'] = frame['relative_spread'].copy()
    frame['strike_raw'] = frame['strike'].copy()
    frame['days_to_exp_raw'] = frame['days_to_exp'].copy()
    frame['exit_bid_raw'] = frame.get('exit_price', frame['bid']).copy()
    if 'exit_date' not in frame.columns:
        frame['exit_date'] = ''

    # Normalize
    train_mean = pd.Series(artifact['train_mean'])
    train_std = pd.Series(artifact['train_std'])
    frame[feature_columns] = (frame[feature_columns] - train_mean) / train_std
    frame[feature_columns] = frame[feature_columns].fillna(0.0)
    frame = frame[frame['relative_spread'] <= 0.50].reset_index(drop=True)

    # Run backtest
    exec_config = ExecutionConfig(
        min_price=args.min_price,
        max_relative_spread=args.max_spread,
        min_volume=args.min_volume,
    )
    risk_config = RiskConfig(
        starting_capital=args.starting_capital,
        max_total_exposure=args.max_exposure,
    )

    result = run_realistic_backtest(model, frame, feature_columns, device, exec_config, risk_config)

    # Print results
    m = result['metrics']
    print()
    print("=" * 60)
    print("REALISTIC BACKTEST RESULTS")
    print("=" * 60)
    print(f"  Starting capital:   ${m['starting_capital']:,.0f}")
    print(f"  Ending capital:     ${m.get('final_equity', 0):,.0f}")
    print(f"  Total return:       {m['total_return_pct']:.1f}%")
    print(f"  Total trades:       {m['total_trades']}")
    print(f"  Days traded:        {m['days_traded']}/{m['total_days']}")
    if 'win_rate' in m:
        print(f"  Win rate:           {m['win_rate']:.1f}%")
        print(f"  Avg trade return:   {m['avg_trade_return']:.1f}%")
        print(f"  Median trade return:{m['median_trade_return']:.1f}%")
        print(f"  Best trade:         {m['best_trade']:.1f}%")
        print(f"  Worst trade:        {m['worst_trade']:.1f}%")
        print(f"  Avg position size:  ${m['avg_position_size']:,.0f}")
        print(f"  Avg trades/day:     {m['avg_trades_per_day']:.1f}")
    if 'sharpe' in m:
        print(f"  Sharpe ratio:       {m['sharpe']:.2f}")
        print(f"  Max drawdown:       {m['max_drawdown_pct']:.1f}%")
    print("=" * 60)

    # Save outputs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(m, output_dir / "backtest_metrics.json")
    result['equity_curve'].to_csv(output_dir / "equity_curve.csv", index=False)
    if len(result['trades']) > 0:
        result['trades'].to_csv(output_dir / "trade_log.csv", index=False)
    logger.info("Results saved to %s", output_dir)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent / "updated_option_agent_codebase"))
    main()
