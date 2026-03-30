"""Backtest with the Selective Meta-Allocator.

Bottom-up: score candidates → meta-model accept/reject → optimize survivors.
Exposure emerges from how many candidates survive, not from a top-down prediction.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
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

# Hard constraints (never overridden)
MAX_EXPOSURE = 0.50
MAX_POSITION_PCT = 0.05
MAX_POSITIONS = 10
MAX_SAME_DIRECTION = 7
MAX_SAME_EXPIRY = 3
VOLUME_PARTICIPATION = 0.10

# Tradability
MIN_PRICE = 0.10
MAX_SPREAD = 0.30
MIN_VOLUME = 10
MIN_OI = 50
TOP_M = 40

CANDIDATE_FEATURES = [
    "score", "score_rank_pct", "score_gap_to_top",
    "score_spread_top_m", "call_ratio_top20",
    "delta", "abs_delta", "moneyness", "days_to_exp",
    "implied_volatility", "relative_spread", "ask_price",
    "volume", "open_interest", "gamma", "theta", "vega",
    "vanna", "charm",
    "spy_rsi", "vix", "spy_momentum", "realized_vol_20d",
    "vrp_20d", "is_bull",
    "call_hit_5d", "put_hit_5d", "basket_hit_5d",
    "consecutive_losses",
]


class SelectiveOperator:
    """Bottom-up selective meta-allocator."""

    def __init__(self, model_dir: str):
        d = Path(model_dir)
        call_art = joblib.load(d / "meta_call.joblib")
        put_art = joblib.load(d / "meta_put.joblib")

        self.call_model = call_art["model"]
        self.call_calibrator = call_art["calibrator"]
        self.call_threshold = call_art["threshold"]
        self.call_features = call_art["features"]

        self.put_model = put_art["model"]
        self.put_calibrator = put_art["calibrator"]
        self.put_threshold = put_art["threshold"]
        self.put_features = put_art["features"]

        logger.info("Loaded selective operator: call_threshold=%.2f, put_threshold=%.2f",
                     self.call_threshold, self.put_threshold)

    def score_candidate(self, features: Dict[str, float], side: str) -> float:
        """Return calibrated P(good trade) for a candidate."""
        if side == "call":
            feat_vec = np.array([[features.get(f, 0) for f in self.call_features]])
            raw = self.call_model.predict_proba(feat_vec)[0, 1]
            return float(np.clip(self.call_calibrator.predict([raw])[0], 1e-6, 1 - 1e-6))
        else:
            feat_vec = np.array([[features.get(f, 0) for f in self.put_features]])
            raw = self.put_model.predict_proba(feat_vec)[0, 1]
            return float(np.clip(self.put_calibrator.predict([raw])[0], 1e-6, 1 - 1e-6))

    def should_accept(self, p_good: float, side: str) -> bool:
        threshold = self.call_threshold if side == "call" else self.put_threshold
        return p_good >= threshold


def run_selective_backtest(
    ranker: ChainTransformer,
    operator: SelectiveOperator,
    frame: pd.DataFrame,
    feature_columns: List[str],
    device: torch.device,
    starting_capital: float = 10000.0,
) -> Dict[str, Any]:

    capital = starting_capital
    equity_curve = []
    all_trades = []
    dates = sorted(frame["date"].unique())

    # Rolling efficacy tracking
    recent_call_returns: List[float] = []
    recent_put_returns: List[float] = []
    recent_basket_returns: List[float] = []

    logger.info("Starting selective backtest: $%.0f, %d dates", capital, len(dates))

    for i, date in enumerate(dates):
        day = frame[frame["date"] == date]
        if len(day) < 20:
            equity_curve.append({"date": date, "capital": capital, "daily_pnl": 0,
                                 "n_trades": 0, "n_accepted": 0, "n_rejected": 0})
            continue

        # Score with ranker
        features = np.nan_to_num(day[feature_columns].values.astype(np.float32))
        x = torch.from_numpy(features).unsqueeze(0).to(device)
        with torch.no_grad():
            scores = ranker(x).squeeze(0).cpu().numpy()
        day = day.copy()
        day["score"] = scores

        # Filter to tradeable
        tradeable = day[
            (day["ask"] >= MIN_PRICE) &
            (day["relative_spread"] <= MAX_SPREAD) &
            (day["volume"] >= MIN_VOLUME) &
            (day["open_interest"] >= MIN_OI)
        ]
        if len(tradeable) < 5:
            equity_curve.append({"date": date, "capital": capital, "daily_pnl": 0,
                                 "n_trades": 0, "n_accepted": 0, "n_rejected": 0})
            continue

        top = tradeable.sort_values("score", ascending=False).head(TOP_M)

        # Market context
        spy_close = day["spy_d_close"].iloc[0] if "spy_d_close" in day.columns else 0
        spy_sma50 = day["spy_d_sma_50"].iloc[0] if "spy_d_sma_50" in day.columns else 0
        spy_rsi = day["spy_d_rsi"].iloc[0] if "spy_d_rsi" in day.columns else 50
        vix = day["vix_d_close"].iloc[0] if "vix_d_close" in day.columns else 20
        spy_mom = day["spy_momentum"].iloc[0] if "spy_momentum" in day.columns else 0
        rvol = day["realized_vol_20d"].iloc[0] if "realized_vol_20d" in day.columns else 0
        vrp = day["vrp_20d"].iloc[0] if "vrp_20d" in day.columns else 0
        is_bull = float(spy_close > spy_sma50) if np.isfinite(spy_close) and np.isfinite(spy_sma50) else 0.5

        # Efficacy features
        call_hit = np.mean([r > 0 for r in recent_call_returns[-25:]]) if len(recent_call_returns) >= 5 else 0.5
        put_hit = np.mean([r > 0 for r in recent_put_returns[-25:]]) if len(recent_put_returns) >= 5 else 0.5
        basket_hit = np.mean([r > 0 for r in recent_basket_returns[-5:]]) if len(recent_basket_returns) >= 1 else 0.5
        consec = 0
        for r in reversed(recent_basket_returns):
            if r < 0:
                consec += 1
            else:
                break

        # Score distribution
        score_top1 = top["score"].iloc[0]
        score_spread = top["score"].iloc[0] - top["score"].iloc[-1] if len(top) > 1 else 0
        call_ratio = (top.head(20)["type"] == "call").mean()

        # ── Per-candidate accept/reject ──────────────────────────────────
        accepted = []
        n_rejected = 0

        for rank, (_, row) in enumerate(top.iterrows(), 1):
            side = row.get("type", "")
            cand_feats = {
                "score": row["score"],
                "score_rank_pct": rank / len(top),
                "score_gap_to_top": score_top1 - row["score"],
                "score_spread_top_m": score_spread,
                "call_ratio_top20": call_ratio,
                "delta": row.get("delta", 0),
                "abs_delta": abs(row.get("delta", 0)),
                "moneyness": row.get("moneyness", 1),
                "days_to_exp": row.get("days_to_exp", 30),
                "implied_volatility": row.get("implied_volatility", 0),
                "relative_spread": row.get("relative_spread", 0),
                "ask_price": row.get("ask", 0),
                "volume": row.get("volume", 0),
                "open_interest": row.get("open_interest", 0),
                "gamma": row.get("gamma", 0),
                "theta": row.get("theta", 0),
                "vega": row.get("vega", 0),
                "vanna": row.get("vanna", 0),
                "charm": row.get("charm", 0),
                "spy_rsi": spy_rsi,
                "vix": vix,
                "spy_momentum": spy_mom,
                "realized_vol_20d": rvol,
                "vrp_20d": vrp,
                "is_bull": is_bull,
                "call_hit_5d": call_hit,
                "put_hit_5d": put_hit,
                "basket_hit_5d": basket_hit,
                "consecutive_losses": consec,
            }

            p_good = operator.score_candidate(cand_feats, side)
            if operator.should_accept(p_good, side):
                accepted.append({
                    "row": row,
                    "p_good": p_good,
                    "rank": rank,
                    "side": side,
                    "utility": row["score"] * p_good,
                })
            else:
                n_rejected += 1

        # ── Constrained optimizer on survivors ───────────────────────────
        if len(accepted) == 0:
            equity_curve.append({"date": date, "capital": capital, "daily_pnl": 0,
                                 "n_trades": 0, "n_accepted": 0, "n_rejected": n_rejected})
            # Still update efficacy
            basket_ret = top.head(20)["target_return"].mean() if "target_return" in top.columns else 0
            if np.isfinite(basket_ret):
                recent_basket_returns.append(basket_ret)
            continue

        # Sort by utility
        accepted.sort(key=lambda x: x["utility"], reverse=True)

        positions = []
        total_spent = 0.0
        call_count = 0
        put_count = 0
        expiry_counts: Dict[int, int] = {}
        max_total = capital * MAX_EXPOSURE

        # Drawdown adjustment
        if len(equity_curve) > 0:
            peak = max(e["capital"] for e in equity_curve)
            dd = 1 - capital / peak if peak > 0 else 0
            if dd > 0.20:
                max_total *= 0.5  # halve exposure during drawdown

        for cand in accepted:
            if len(positions) >= MAX_POSITIONS:
                break
            if total_spent >= max_total:
                break

            row = cand["row"]
            side = cand["side"]
            ask = row.get("ask", 0)
            if ask <= 0:
                continue

            if side == "call" and call_count >= MAX_SAME_DIRECTION:
                continue
            if side == "put" and put_count >= MAX_SAME_DIRECTION:
                continue

            dte = row.get("days_to_exp", 30)
            exp_bucket = 0 if dte <= 7 else (1 if dte <= 30 else (2 if dte <= 90 else 3))
            if expiry_counts.get(exp_bucket, 0) >= MAX_SAME_EXPIRY:
                continue

            pos_dollars = min(
                capital * MAX_POSITION_PCT,
                max_total - total_spent,
                row.get("volume", 100) * VOLUME_PARTICIPATION * ask * 100,
            )
            if pos_dollars < ask * 100:
                continue

            n_contracts = int(pos_dollars / (ask * 100))
            cost = n_contracts * ask * 100

            positions.append({
                "contractid": row.get("contractid", ""),
                "type": side,
                "strike": row.get("strike", 0),
                "days_to_exp": dte,
                "entry_price": ask,
                "exit_price": row.get("exit_price", row.get("bid", 0)),
                "n_contracts": n_contracts,
                "cost": cost,
                "score": row["score"],
                "p_good": cand["p_good"],
                "rank": cand["rank"],
            })

            total_spent += cost
            if side == "call":
                call_count += 1
            else:
                put_count += 1
            expiry_counts[exp_bucket] = expiry_counts.get(exp_bucket, 0) + 1

        # P&L
        daily_pnl = 0.0
        for pos in positions:
            exit_val = pos["n_contracts"] * pos["exit_price"] * 100
            pnl = exit_val - pos["cost"]
            ret = pnl / pos["cost"] if pos["cost"] > 0 else 0
            daily_pnl += pnl

            all_trades.append({
                "date": str(date),
                "contractid": pos["contractid"],
                "type": pos["type"],
                "strike": pos["strike"],
                "days_to_exp": pos["days_to_exp"],
                "entry_price": pos["entry_price"],
                "exit_price": pos["exit_price"],
                "n_contracts": pos["n_contracts"],
                "cost": pos["cost"],
                "pnl": pnl,
                "return_pct": ret,
                "score": pos["score"],
                "p_good": pos["p_good"],
                "rank": pos["rank"],
            })

        capital += daily_pnl
        if capital <= 0:
            logger.warning("Capital depleted at %s", date)
            capital = 0

        equity_curve.append({
            "date": date, "capital": capital, "daily_pnl": daily_pnl,
            "n_trades": len(positions), "n_accepted": len(accepted),
            "n_rejected": n_rejected,
        })

        # Update efficacy
        if "target_return" in top.columns:
            calls_ret = top[top["type"] == "call"]["target_return"].dropna()
            puts_ret = top[top["type"] == "put"]["target_return"].dropna()
            recent_call_returns.extend(calls_ret.tolist())
            recent_put_returns.extend(puts_ret.tolist())
            basket_ret = top.head(20)["target_return"].mean()
            if np.isfinite(basket_ret):
                recent_basket_returns.append(basket_ret)

        if (i + 1) % 100 == 0:
            logger.info("  Day %d/%d | Capital: $%,.0f | Accepted: %d | Trades: %d",
                         i + 1, len(dates), capital, len(accepted), len(all_trades))

    # Metrics
    eq = pd.DataFrame(equity_curve)
    trades_df = pd.DataFrame(all_trades) if all_trades else pd.DataFrame()

    m = {
        "starting_capital": starting_capital,
        "ending_capital": capital,
        "total_return_pct": (capital / starting_capital - 1) * 100,
        "total_trades": len(all_trades),
        "days_traded": len(eq[eq["n_trades"] > 0]),
        "days_sat_out": len(eq[eq["n_trades"] == 0]),
        "total_days": len(eq),
        "avg_accepted_per_day": eq["n_accepted"].mean(),
        "avg_rejected_per_day": eq["n_rejected"].mean(),
    }

    if len(trades_df) > 0:
        m["win_rate"] = (trades_df["pnl"] > 0).mean() * 100
        m["avg_trade_return"] = trades_df["return_pct"].mean() * 100
        m["median_trade_return"] = trades_df["return_pct"].median() * 100
        m["best_trade"] = trades_df["return_pct"].max() * 100
        m["worst_trade"] = trades_df["return_pct"].min() * 100
        m["avg_trades_per_day"] = trades_df.groupby("date").size().mean()
        m["avg_p_good"] = trades_df["p_good"].mean()

        # Side breakdown
        for side in ["call", "put"]:
            st = trades_df[trades_df["type"] == side]
            if len(st) > 0:
                m[f"{side}_trades"] = len(st)
                m[f"{side}_win_rate"] = (st["pnl"] > 0).mean() * 100
                m[f"{side}_avg_return"] = st["return_pct"].mean() * 100

    if len(eq) > 1 and eq["capital"].iloc[0] > 0:
        daily_ret = eq["daily_pnl"] / eq["capital"].shift(1).fillna(starting_capital)
        m["sharpe"] = float(daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0
        eq["peak"] = eq["capital"].cummax()
        eq["drawdown"] = eq["capital"] / eq["peak"] - 1
        m["max_drawdown_pct"] = float(eq["drawdown"].min() * 100)

    return {"metrics": m, "equity_curve": eq, "trades": trades_df}


def main():
    parser = argparse.ArgumentParser(description="Backtest with selective meta-allocator")
    parser.add_argument("--model-artifact", required=True)
    parser.add_argument("--operator-dir", default="./selective_operator_model")
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--config", default="./config_tuned.yaml")
    parser.add_argument("--output-dir", default="./selective_backtest_output")
    parser.add_argument("--starting-capital", type=float, default=10000.0)
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

    # Load operator
    operator = SelectiveOperator(args.operator_dir)

    # Load data
    cfg = load_config(args.config)
    frame = prepare_model_frame(args.data, cfg, include_targets=True)

    feature_columns = artifact["feature_columns"]
    edges = artifact["relevance_edges"]
    frame["target_relevance"] = apply_relevance_bins(frame["target_return"], edges).astype(np.float32)
    frame["type_numeric"] = (frame["type"].str.lower() == "call").astype(np.float32)

    # Save raw values
    for col in ["ask", "bid", "volume", "open_interest", "relative_spread",
                 "strike", "days_to_exp", "delta", "moneyness", "implied_volatility",
                 "spy_d_close", "spy_d_sma_50", "spy_d_rsi", "spy_d_macd_hist",
                 "vix_d_close", "spy_momentum", "realized_vol_5d", "realized_vol_20d",
                 "realized_vol_60d", "vrp_20d", "gamma", "theta", "vega", "vanna", "charm"]:
        if col in frame.columns:
            frame[f"{col}_raw"] = frame[col].copy()

    if "exit_price" not in frame.columns:
        frame["exit_price"] = frame.get("bid", 0)

    # Normalize for ranker
    train_mean = pd.Series(artifact["train_mean"])
    train_std = pd.Series(artifact["train_std"])
    frame[feature_columns] = (frame[feature_columns] - train_mean) / train_std
    frame[feature_columns] = frame[feature_columns].fillna(0.0)
    frame = frame[frame["relative_spread"] <= 0.50].reset_index(drop=True)

    # Restore raw for operator
    for col in ["ask", "bid", "volume", "open_interest", "relative_spread",
                 "strike", "days_to_exp", "delta", "moneyness", "implied_volatility",
                 "spy_d_close", "spy_d_sma_50", "spy_d_rsi", "spy_d_macd_hist",
                 "vix_d_close", "spy_momentum", "realized_vol_5d", "realized_vol_20d",
                 "realized_vol_60d", "vrp_20d", "gamma", "theta", "vega", "vanna", "charm"]:
        raw = f"{col}_raw"
        if raw in frame.columns:
            frame[col] = frame[raw]

    # Run
    result = run_selective_backtest(ranker, operator, frame, feature_columns, device,
                                    starting_capital=args.starting_capital)

    m = result["metrics"]
    print()
    print("=" * 65)
    print("SELECTIVE META-ALLOCATOR BACKTEST RESULTS")
    print("=" * 65)
    print(f"  Starting capital:    ${m['starting_capital']:>12,.0f}")
    print(f"  Ending capital:      ${m.get('ending_capital', 0):>12,.0f}")
    print(f"  Total return:        {m['total_return_pct']:>12.1f}%")
    print(f"  Total trades:        {m['total_trades']:>12d}")
    print(f"  Days traded:         {m['days_traded']:>12d}/{m['total_days']}")
    print(f"  Days sat out:        {m.get('days_sat_out', 0):>12d}")
    print(f"  Avg accepted/day:    {m.get('avg_accepted_per_day', 0):>12.1f}")
    print(f"  Avg rejected/day:    {m.get('avg_rejected_per_day', 0):>12.1f}")
    if "win_rate" in m:
        print(f"  Win rate:            {m['win_rate']:>12.1f}%")
        print(f"  Avg trade return:    {m['avg_trade_return']:>12.1f}%")
        print(f"  Median trade return: {m['median_trade_return']:>12.1f}%")
        print(f"  Best trade:          {m['best_trade']:>12.1f}%")
        print(f"  Worst trade:         {m['worst_trade']:>12.1f}%")
        print(f"  Avg P(good):         {m.get('avg_p_good', 0):>12.3f}")
    if "sharpe" in m:
        print(f"  Sharpe ratio:        {m['sharpe']:>12.2f}")
        print(f"  Max drawdown:        {m['max_drawdown_pct']:>12.1f}%")
    print()
    for side in ["call", "put"]:
        if f"{side}_trades" in m:
            print(f"  {side.upper():5s}: {m[f'{side}_trades']:>5d} trades, win={m[f'{side}_win_rate']:.0f}%, avg_ret={m[f'{side}_avg_return']:.1f}%")
    print("=" * 65)

    # Save
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_json(m, out / "selective_backtest_metrics.json")
    result["equity_curve"].to_csv(out / "selective_equity_curve.csv", index=False)
    if len(result["trades"]) > 0:
        result["trades"].to_csv(out / "selective_trade_log.csv", index=False)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent / "updated_option_agent_codebase"))
    main()
