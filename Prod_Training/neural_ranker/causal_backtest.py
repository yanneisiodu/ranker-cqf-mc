"""Causal backtest using the event-driven simulation engine.

Fixes all known simulation leaks:
  1. P&L settled on exit date, not entry date
  2. Capital reserved on entry, freed on exit
  3. Efficacy features use only matured (settled) trades
  4. Tradability filtered on raw columns before normalization
  5. Exit strategy: take-profit, stop-loss, trailing stop, max hold

Usage:
    python causal_backtest.py \
        --model-artifact /path/to/artifact.pt \
        --data year_2024_data.csv year_2025_data.csv \
        --config config_tuned.yaml \
        --output-dir ./causal_backtest_output
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from logger import setup_logger
from neural_ranker import ChainTransformer, NeuralRankerConfig, get_device
from simulation_engine import (
    ExitStrategy,
    ExecutionConfig,
    RiskConfig,
    SimulationEngine,
    OpenPosition,
    MaturedHistoryQueue,
    filter_tradeable_raw,
)
from utils import (
    apply_relevance_bins,
    load_config,
    prepare_model_frame,
    save_json,
)

logger = setup_logger(__name__)

TOP_M = 40  # score top-M candidates for selection


def run_causal_backtest(
    ranker: ChainTransformer,
    frame: pd.DataFrame,
    feature_columns: List[str],
    device: torch.device,
    exit_strategy: ExitStrategy,
    exec_config: ExecutionConfig,
    risk_config: RiskConfig,
    meta_operator=None,
) -> Dict[str, Any]:
    """Run a fully causal backtest with event-driven settlement."""

    engine = SimulationEngine(exit_strategy, risk_config)
    dates = sorted(frame["date"].unique())

    logger.info("Starting causal backtest: $%.0f capital, %d dates", risk_config.starting_capital, len(dates))
    logger.info("Exit strategy: TP=%.0f%%, SL=%.0f%%, trailing=%.0f%%, max_hold=%d days",
                exit_strategy.take_profit_pct * 100, exit_strategy.stop_loss_pct * 100,
                exit_strategy.trailing_stop_pct * 100, exit_strategy.max_hold_days)

    for i, date in enumerate(dates):
        day = frame[frame["date"] == date]
        if len(day) < 20:
            engine.record_equity(date, 0, [])
            continue

        # ── Step 1: Mark-to-market and settle exits ──────────────────────
        # Build price lookup from today's bid prices (raw)
        price_lookup = dict(zip(day["contractid"], day["bid"]))
        settled_today = engine.step(date, price_lookup)

        # ── Step 2: Score candidates with ranker ─────────────────────────
        # Use normalized features for the ranker
        norm_features = np.nan_to_num(
            day[feature_columns].values.astype(np.float32)
        )
        x = torch.from_numpy(norm_features).unsqueeze(0).to(device)
        with torch.no_grad():
            scores = ranker(x).squeeze(0).cpu().numpy()
        day = day.copy()
        day["score"] = scores

        # ── Step 3: Filter tradeable on RAW columns ──────────────────────
        tradeable = filter_tradeable_raw(day, exec_config)
        if len(tradeable) < 5:
            engine.record_equity(date, 0, settled_today)
            continue

        # ── Step 4: Select candidates ────────────────────────────────────
        top = tradeable.sort_values("score", ascending=False).head(TOP_M)

        # Optional: meta-model filtering
        if meta_operator is not None:
            top = _apply_meta_filter(top, meta_operator, engine.history, day, date)

        # ── Step 5: Size and open positions ──────────────────────────────
        capacity = engine.get_capacity()
        n_new = 0

        if capacity["max_new_positions"] > 0 and capacity["remaining_gross"] > 0:
            call_count = capacity["call_count"]
            put_count = capacity["put_count"]

            for _, row in top.iterrows():
                if n_new >= capacity["max_new_positions"]:
                    break
                if capacity["remaining_gross"] - (n_new * capacity["max_per_position"]) <= 0:
                    break

                side = row.get("type", "")
                if side == "call" and call_count >= risk_config.max_same_direction:
                    continue
                if side == "put" and put_count >= risk_config.max_same_direction:
                    continue

                ask = row.get("ask", 0)
                if ask <= 0:
                    continue

                # Size position
                pos_dollars = min(
                    capacity["max_per_position"],
                    capacity["remaining_gross"] - n_new * capacity["max_per_position"],
                    capacity["available_cash"] - n_new * capacity["max_per_position"],
                    row.get("volume", 100) * exec_config.max_volume_participation * ask * 100,
                )
                if pos_dollars < ask * 100:
                    continue

                n_contracts = int(pos_dollars / (ask * 100))
                cost = n_contracts * ask * 100

                pos = OpenPosition(
                    entry_date=date,
                    contractid=row.get("contractid", ""),
                    option_type=side,
                    strike=row.get("strike", 0),
                    days_to_exp=row.get("days_to_exp", 30),
                    entry_price=ask,
                    n_contracts=n_contracts,
                    cost=cost,
                    score=row["score"],
                    rank=n_new + 1,
                )

                if engine.open_position(pos):
                    n_new += 1
                    if side == "call":
                        call_count += 1
                    else:
                        put_count += 1

        engine.record_equity(date, n_new, settled_today)

        if (i + 1) % 100 == 0:
            logger.info("  Day %d/%d | Equity: $%.0f | Open: %d | Settled: %d | New: %d",
                         i + 1, len(dates), engine.equity,
                         len(engine.open_positions), len(settled_today), n_new)

    # Force-settle any remaining open positions at last day's prices
    if engine.open_positions:
        last_date = dates[-1]
        last_day = frame[frame["date"] == last_date]
        last_prices = dict(zip(last_day["contractid"], last_day["bid"]))

        # Override max_hold to force exit
        for pos in engine.open_positions:
            pos.hold_days = exit_strategy.max_hold_days

        final_settled = engine.step(last_date, last_prices)
        engine.record_equity(last_date, 0, final_settled)
        logger.info("Force-settled %d remaining positions at end of backtest", len(final_settled))

    return engine.get_results()


def _apply_meta_filter(top, meta_operator, history, day, date):
    """Apply selective meta-model filtering if operator is provided."""
    # TODO: integrate with SelectiveOperator using matured history only
    return top


def main():
    parser = argparse.ArgumentParser(description="Causal backtest with event-driven simulation")
    parser.add_argument("--model-artifact", required=True)
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--config", default="./config_tuned.yaml")
    parser.add_argument("--output-dir", default="./causal_backtest_output")
    parser.add_argument("--starting-capital", type=float, default=10000.0)
    # Exit strategy
    parser.add_argument("--take-profit", type=float, default=0.50)
    parser.add_argument("--stop-loss", type=float, default=0.20)
    parser.add_argument("--trailing-stop", type=float, default=0.15)
    parser.add_argument("--max-hold-days", type=int, default=5)
    # Execution
    parser.add_argument("--min-price", type=float, default=0.10)
    parser.add_argument("--max-spread", type=float, default=0.30)
    parser.add_argument("--min-volume", type=int, default=10)
    parser.add_argument("--max-exposure", type=float, default=0.50)
    args = parser.parse_args()

    # Load ranker
    artifact = torch.load(args.model_artifact, map_location="cpu", weights_only=False)
    nr_config = NeuralRankerConfig(**artifact["config"])
    ranker = ChainTransformer(nr_config)
    state = {k.replace("_orig_mod.", ""): v for k, v in artifact["model_state_dict"].items()}
    ranker.load_state_dict(state)
    device = get_device()
    ranker = ranker.to(device)
    ranker.eval()
    logger.info("Loaded ranker: %d params on %s", sum(p.numel() for p in ranker.parameters()), device)

    # Load data
    cfg = load_config(args.config)
    frame = prepare_model_frame(args.data, cfg, include_targets=True)
    logger.info("Loaded: %d rows, %d dates", len(frame), frame["date"].nunique())

    feature_columns = artifact["feature_columns"]
    edges = artifact["relevance_edges"]
    frame["target_relevance"] = apply_relevance_bins(frame["target_return"], edges).astype(np.float32)
    frame["type_numeric"] = (frame["type"].str.lower() == "call").astype(np.float32)

    # Normalize features for ranker — but keep raw columns for execution/simulation
    train_mean = pd.Series(artifact["train_mean"])
    train_std = pd.Series(artifact["train_std"])

    # Save raw columns BEFORE normalization
    raw_cols_to_save = ["ask", "bid", "volume", "open_interest", "relative_spread",
                        "strike", "days_to_exp", "delta", "moneyness"]
    for col in raw_cols_to_save:
        if col in frame.columns:
            frame[f"{col}_raw"] = frame[col].copy()

    # Normalize only the ranker feature columns
    frame[feature_columns] = (frame[feature_columns] - train_mean) / train_std
    frame[feature_columns] = frame[feature_columns].fillna(0.0)

    # Restore raw columns for simulation (execution filters + price lookup)
    for col in raw_cols_to_save:
        raw = f"{col}_raw"
        if raw in frame.columns:
            frame[col] = frame[raw]

    # Configs
    exit_strategy = ExitStrategy(
        take_profit_pct=args.take_profit,
        stop_loss_pct=args.stop_loss,
        trailing_stop_pct=args.trailing_stop,
        max_hold_days=args.max_hold_days,
    )
    exec_config = ExecutionConfig(
        min_price=args.min_price,
        max_relative_spread=args.max_spread,
        min_volume=args.min_volume,
    )
    risk_config = RiskConfig(
        starting_capital=args.starting_capital,
        max_gross_pct=args.max_exposure,
    )

    # Run
    result = run_causal_backtest(ranker, frame, feature_columns, device,
                                 exit_strategy, exec_config, risk_config)

    # Print results
    m = result["metrics"]
    print()
    print("=" * 65)
    print("CAUSAL BACKTEST RESULTS")
    print("=" * 65)
    print(f"  Starting capital:    ${m['starting_capital']:>12,.0f}")
    print(f"  Ending equity:       ${m.get('ending_equity', 0):>12,.0f}")
    print(f"  Total return:        {m['total_return_pct']:>12.1f}%")
    print(f"  Total trades:        {m['total_trades']:>12d}")
    if "win_rate" in m:
        print(f"  Win rate:            {m['win_rate']:>12.1f}%")
        print(f"  Avg trade return:    {m['avg_trade_return']:>12.1f}%")
        print(f"  Median trade return: {m['median_trade_return']:>12.1f}%")
        print(f"  Best trade:          {m['best_trade']:>12.1f}%")
        print(f"  Worst trade:         {m['worst_trade']:>12.1f}%")
        print(f"  Avg hold days:       {m['avg_hold_days']:>12.1f}")
    if "sharpe" in m:
        print(f"  Sharpe ratio:        {m['sharpe']:>12.2f}")
        print(f"  Max drawdown:        {m['max_drawdown_pct']:>12.1f}%")
    print()
    print("  Exit reasons:")
    for reason in ["take_profit", "stop_loss", "trailing_stop", "max_hold", "worthless"]:
        key = f"exit_{reason}_count"
        if key in m:
            print(f"    {reason:15s}: {m[key]:>5d} ({m[f'exit_{reason}_pct']:.0f}%)")
    print()
    for side in ["call", "put"]:
        if f"{side}_trades" in m:
            print(f"  {side.upper():5s}: {m[f'{side}_trades']:>5d} trades, win={m[f'{side}_win_rate']:.0f}%, avg_ret={m[f'{side}_avg_return']:.1f}%")
    print("=" * 65)

    # Save
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_json(m, out / "causal_backtest_metrics.json")
    result["equity_curve"].to_csv(out / "causal_equity_curve.csv", index=False)
    if len(result["trades"]) > 0:
        result["trades"].to_csv(out / "causal_trade_log.csv", index=False)
    logger.info("Results saved to %s", out)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent / "updated_option_agent_codebase"))
    main()
