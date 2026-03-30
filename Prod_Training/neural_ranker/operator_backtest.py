"""Backtest with the Regime-Trust Meta-Allocator.

Integrates all operator components:
1. Day gate → bucketed exposure
2. Bayesian side allocator → call/put budget
3. Candidate tail-cap → reject dangerous positions
4. Constrained basket optimizer → final position selection

Produces realistic P&L with execution constraints.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch

from logger import setup_logger
from neural_ranker import ChainTransformer, NeuralRankerConfig, get_device
from build_operator_dataset import compute_day_features
from utils import (
    apply_relevance_bins,
    load_config,
    prepare_model_frame,
    save_json,
)

logger = setup_logger(__name__)

EXPOSURE_MAP = {0: 0.0, 1: 0.10, 2: 0.20, 3: 0.30, 4: 0.40, 5: 0.50}


class OperatorStack:
    """The full Regime-Trust Meta-Allocator."""

    def __init__(self, operator_dir: str):
        op = Path(operator_dir)

        # Load day gate
        gate_artifact = joblib.load(op / "day_gate.joblib")
        self.gate_model = gate_artifact["model"]
        self.gate_features = gate_artifact["features"]
        self.gate_class_names = gate_artifact["class_names"]

        # Load side allocator
        with open(op / "side_allocator.json") as f:
            self.side_params = json.load(f)

        # Load thresholds
        with open(op / "tail_cap_thresholds.json") as f:
            self.thresholds = json.load(f)

        logger.info("Loaded operator: gate=%d features, side=%d regimes, %d thresholds",
                     len(self.gate_features), len(self.side_params), len(self.thresholds))

    def predict_exposure(self, day_features: Dict[str, float]) -> Tuple[float, str]:
        """Predict exposure bucket from day features."""
        feat_vec = np.array([[day_features.get(f, 0.0) for f in self.gate_features]])
        pred = self.gate_model.predict(feat_vec)[0]
        proba = self.gate_model.predict_proba(feat_vec)[0]
        class_name = self.gate_class_names[pred]

        # Map class to exposure
        if class_name == "sit_out":
            exposure = 0.0
        elif class_name == "normal":
            exposure = 0.10
        elif class_name == "aggressive":
            # Use probability to scale between 0.20 and 0.40
            aggressive_prob = proba[2]
            exposure = 0.20 + 0.20 * min(aggressive_prob, 1.0)
        else:
            exposure = 0.10

        return exposure, class_name

    def get_side_split(self, is_bull: bool) -> Tuple[float, float]:
        """Get call/put allocation from Bayesian posterior."""
        regime = "bull" if is_bull else "bear"
        params = self.side_params.get(regime, {"call_share": 0.5})
        call_share = params.get("call_share", 0.5)

        # Apply guardrails — never more than 80% in one direction
        call_share = np.clip(call_share, 0.20, 0.80)
        return call_share, 1.0 - call_share

    def filter_candidates(
        self,
        candidates: pd.DataFrame,
        is_bull: bool,
    ) -> pd.DataFrame:
        """Apply tail-cap rules to reject dangerous candidates."""
        filtered = candidates.copy()
        rejected_reasons = []

        # Rule 1: Reject deep OTM puts in bull regime
        if is_bull and "delta_raw" in filtered.columns:
            deep_otm_put = (
                (filtered["type"] == "put") &
                (filtered["delta_raw"].abs() < 0.15) &
                (filtered.get("days_to_exp_raw", pd.Series(30)) < 14)
            )
            n_reject = deep_otm_put.sum()
            if n_reject > 0:
                rejected_reasons.append(f"deep_otm_put_bull: {n_reject}")
                filtered = filtered[~deep_otm_put]

        # Rule 2: Reject wide-spread near-expiry
        if "relative_spread_raw" in filtered.columns and "days_to_exp_raw" in filtered.columns:
            wide_short = (
                (filtered["relative_spread_raw"] > 0.25) &
                (filtered["days_to_exp_raw"] < 7)
            )
            n_reject = wide_short.sum()
            if n_reject > 0:
                rejected_reasons.append(f"wide_spread_short_dte: {n_reject}")
                filtered = filtered[~wide_short]

        # Rule 3: Reject very low volume
        if "volume_raw" in filtered.columns:
            low_vol = filtered["volume_raw"] < 10
            n_reject = low_vol.sum()
            if n_reject > 0:
                rejected_reasons.append(f"low_volume: {n_reject}")
                filtered = filtered[~low_vol]

        # Rule 4: Reject very cheap options (penny options)
        if "ask_raw" in filtered.columns:
            cheap = filtered["ask_raw"] < 0.10
            n_reject = cheap.sum()
            if n_reject > 0:
                rejected_reasons.append(f"penny_option: {n_reject}")
                filtered = filtered[~cheap]

        return filtered

    def select_positions(
        self,
        candidates: pd.DataFrame,
        capital: float,
        gross_exposure: float,
        call_share: float,
        put_share: float,
    ) -> List[Dict]:
        """Constrained basket optimizer — select and size positions."""
        if gross_exposure <= 0 or len(candidates) == 0:
            return []

        max_total = capital * gross_exposure
        max_per_position = capital * self.thresholds.get("hard_max_position_pct", 0.05)
        max_positions = int(self.thresholds.get("hard_max_positions", 10))
        max_same_dir = int(self.thresholds.get("hard_max_same_direction", 7))
        max_same_expiry = int(self.thresholds.get("hard_max_same_expiry", 3))

        call_budget = max_total * call_share
        put_budget = max_total * put_share

        positions = []
        call_spent = 0.0
        put_spent = 0.0
        call_count = 0
        put_count = 0
        expiry_counts: Dict[int, int] = {}

        # Sort by score descending
        ranked = candidates.sort_values("score", ascending=False)

        for _, row in ranked.iterrows():
            if len(positions) >= max_positions:
                break

            option_type = row.get("type", "")
            ask = row.get("ask_raw", 0)
            if ask <= 0:
                continue

            # Check direction budget
            if option_type == "call":
                if call_count >= max_same_dir or call_spent >= call_budget:
                    continue
            elif option_type == "put":
                if put_count >= max_same_dir or put_spent >= put_budget:
                    continue

            # Check expiry concentration
            dte = row.get("days_to_exp_raw", 30)
            expiry_bucket = 0 if dte <= 7 else (1 if dte <= 30 else (2 if dte <= 90 else 3))
            if expiry_counts.get(expiry_bucket, 0) >= max_same_expiry:
                continue

            # Size position
            if option_type == "call":
                remaining_budget = min(call_budget - call_spent, max_total - call_spent - put_spent)
            else:
                remaining_budget = min(put_budget - put_spent, max_total - call_spent - put_spent)

            position_dollars = min(max_per_position, remaining_budget)

            # Volume participation limit
            vol = row.get("volume_raw", 100)
            max_by_volume = vol * 0.10 * ask * 100
            position_dollars = min(position_dollars, max_by_volume)

            if position_dollars < ask * 100:
                continue

            n_contracts = int(position_dollars / (ask * 100))
            actual_cost = n_contracts * ask * 100

            positions.append({
                "contractid": row.get("contractid", ""),
                "type": option_type,
                "strike": row.get("strike_raw", row.get("strike", 0)),
                "days_to_exp": dte,
                "entry_price": ask,
                "exit_price": row.get("exit_price", row.get("bid_raw", 0)),
                "exit_date": row.get("exit_date", ""),
                "n_contracts": n_contracts,
                "cost": actual_cost,
                "score": row.get("score", 0),
            })

            if option_type == "call":
                call_spent += actual_cost
                call_count += 1
            else:
                put_spent += actual_cost
                put_count += 1
            expiry_counts[expiry_bucket] = expiry_counts.get(expiry_bucket, 0) + 1

        return positions


def run_operator_backtest(
    model: ChainTransformer,
    operator: OperatorStack,
    frame: pd.DataFrame,
    feature_columns: List[str],
    device: torch.device,
    starting_capital: float = 10000.0,
) -> Dict[str, Any]:
    """Run backtest with the full operator stack."""

    capital = starting_capital
    equity_curve = []
    all_trades = []
    history_df = pd.DataFrame()
    dates = sorted(frame["date"].unique())

    logger.info("Starting operator backtest: $%.0f, %d dates", capital, len(dates))

    for i, date in enumerate(dates):
        day = frame[frame["date"] == date]
        if len(day) < 20:
            equity_curve.append({"date": date, "capital": capital, "daily_pnl": 0,
                                 "n_trades": 0, "gate": "skip", "exposure": 0})
            continue

        # Score with ranker
        features = np.nan_to_num(day[feature_columns].values.astype(np.float32))
        x = torch.from_numpy(features).unsqueeze(0).to(device)
        with torch.no_grad():
            scores = model(x).squeeze(0).cpu().numpy()
        day = day.copy()
        day["score"] = scores

        # ── Operator Layer ──────────────────────────────────────────────

        # 1. Day gate
        day_feats = compute_day_features(day, scores, history_df, top_k=40)
        exposure, gate_class = operator.predict_exposure(day_feats)

        # Drawdown override
        if len(equity_curve) > 0:
            peak = max(e["capital"] for e in equity_curve)
            current_dd = 1 - capital / peak if peak > 0 else 0
            dd_threshold = operator.thresholds.get("hard_drawdown_reduce_threshold", 0.20)
            if current_dd > dd_threshold:
                exposure *= operator.thresholds.get("hard_drawdown_reduce_factor", 0.5)

        if exposure <= 0:
            equity_curve.append({"date": date, "capital": capital, "daily_pnl": 0,
                                 "n_trades": 0, "gate": gate_class, "exposure": 0})
            # Still update history with what would have happened
            basket_return = day.sort_values("score", ascending=False).head(20)["target_return"].mean()
            if np.isfinite(basket_return):
                history_df = pd.concat([history_df, pd.DataFrame([{
                    "date": date, "basket_return": basket_return,
                    "call_return": np.nan, "put_return": np.nan,
                }])], ignore_index=True)
            continue

        # 2. Side allocation
        is_bull = day_feats.get("regime_spy_above_sma50", 0.5) > 0.5
        call_share, put_share = operator.get_side_split(is_bull)

        # 3. Tail-cap filtering
        filtered = operator.filter_candidates(day, is_bull)
        if len(filtered) < 5:
            filtered = day  # fallback

        # 4. Position selection
        positions = operator.select_positions(filtered, capital, exposure, call_share, put_share)

        # Calculate P&L
        daily_pnl = 0.0
        for pos in positions:
            exit_value = pos["n_contracts"] * pos["exit_price"] * 100
            entry_cost = pos["cost"]
            pnl = exit_value - entry_cost

            daily_pnl += pnl
            all_trades.append({
                "date": str(date),
                "exit_date": str(pos["exit_date"]),
                "contractid": pos["contractid"],
                "type": pos["type"],
                "strike": pos["strike"],
                "days_to_exp": pos["days_to_exp"],
                "entry_price": pos["entry_price"],
                "exit_price": pos["exit_price"],
                "n_contracts": pos["n_contracts"],
                "cost": entry_cost,
                "pnl": pnl,
                "return_pct": pnl / entry_cost if entry_cost > 0 else 0,
                "score": pos["score"],
                "gate": gate_class,
                "exposure": exposure,
                "call_share": call_share,
            })

        capital += daily_pnl
        if capital <= 0:
            logger.warning("Capital depleted at %s", date)
            capital = 0

        equity_curve.append({
            "date": date, "capital": capital, "daily_pnl": daily_pnl,
            "n_trades": len(positions), "gate": gate_class, "exposure": exposure,
        })

        # Update history
        top20 = day.sort_values("score", ascending=False).head(20)
        calls = top20[top20["type"] == "call"]
        puts = top20[top20["type"] == "put"]
        history_df = pd.concat([history_df, pd.DataFrame([{
            "date": date,
            "basket_return": top20["target_return"].mean() if "target_return" in top20.columns else 0,
            "call_return": calls["target_return"].mean() if len(calls) > 0 and "target_return" in calls.columns else np.nan,
            "put_return": puts["target_return"].mean() if len(puts) > 0 and "target_return" in puts.columns else np.nan,
        }])], ignore_index=True)

        if (i + 1) % 100 == 0:
            logger.info("  Day %d/%d | Capital: $%,.0f | Gate: %s | Trades: %d",
                         i + 1, len(dates), capital, gate_class, len(all_trades))

    # Compute metrics
    eq = pd.DataFrame(equity_curve)
    trades_df = pd.DataFrame(all_trades) if all_trades else pd.DataFrame()

    metrics = {
        "starting_capital": starting_capital,
        "ending_capital": capital,
        "total_return_pct": (capital / starting_capital - 1) * 100,
        "total_trades": len(all_trades),
        "days_traded": len(eq[eq["n_trades"] > 0]),
        "days_sat_out": len(eq[eq["exposure"] == 0]),
        "total_days": len(eq),
    }

    if len(trades_df) > 0:
        metrics["win_rate"] = (trades_df["pnl"] > 0).mean() * 100
        metrics["avg_trade_return"] = trades_df["return_pct"].mean() * 100
        metrics["median_trade_return"] = trades_df["return_pct"].median() * 100
        metrics["best_trade"] = trades_df["return_pct"].max() * 100
        metrics["worst_trade"] = trades_df["return_pct"].min() * 100
        metrics["avg_trades_per_day"] = trades_df.groupby("date").size().mean()

        # Gate analysis
        for gate in ["sit_out", "normal", "aggressive"]:
            gate_trades = trades_df[trades_df["gate"] == gate]
            if len(gate_trades) > 0:
                metrics[f"gate_{gate}_trades"] = len(gate_trades)
                metrics[f"gate_{gate}_win_rate"] = (gate_trades["pnl"] > 0).mean() * 100
                metrics[f"gate_{gate}_avg_return"] = gate_trades["return_pct"].mean() * 100

    if len(eq) > 1 and eq["capital"].iloc[0] > 0:
        daily_returns = eq["daily_pnl"] / eq["capital"].shift(1).fillna(starting_capital)
        metrics["sharpe"] = float(daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0
        eq["peak"] = eq["capital"].cummax()
        eq["drawdown"] = eq["capital"] / eq["peak"] - 1
        metrics["max_drawdown_pct"] = float(eq["drawdown"].min() * 100)

    return {"metrics": metrics, "equity_curve": eq, "trades": trades_df}


def main():
    parser = argparse.ArgumentParser(description="Backtest with Regime-Trust Meta-Allocator")
    parser.add_argument("--model-artifact", required=True)
    parser.add_argument("--operator-dir", default="./operator_model")
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--config", default="./config_tuned.yaml")
    parser.add_argument("--output-dir", default="./operator_backtest_output")
    parser.add_argument("--starting-capital", type=float, default=10000.0)
    args = parser.parse_args()

    # Load model
    artifact = torch.load(args.model_artifact, map_location="cpu", weights_only=False)
    nr_config = NeuralRankerConfig(**artifact["config"])
    model = ChainTransformer(nr_config)
    state = {k.replace("_orig_mod.", ""): v for k, v in artifact["model_state_dict"].items()}
    model.load_state_dict(state)
    device = get_device()
    model = model.to(device)
    model.eval()

    # Load operator
    operator = OperatorStack(args.operator_dir)

    # Load data
    cfg = load_config(args.config)
    frame = prepare_model_frame(args.data, cfg, include_targets=True)

    feature_columns = artifact["feature_columns"]
    edges = artifact["relevance_edges"]
    frame["target_relevance"] = apply_relevance_bins(frame["target_return"], edges).astype(np.float32)
    frame["type_numeric"] = (frame["type"].str.lower() == "call").astype(np.float32)

    # Save raw values
    for col in ["ask", "bid", "volume", "open_interest", "relative_spread",
                 "strike", "days_to_exp", "delta", "moneyness",
                 "spy_d_close", "spy_d_sma_50", "spy_d_rsi", "spy_d_macd_hist",
                 "vix_d_close", "spy_momentum", "realized_vol_5d",
                 "realized_vol_20d", "realized_vol_60d", "vrp_20d",
                 "implied_volatility"]:
        if col in frame.columns:
            frame[f"{col}_raw"] = frame[col].copy()

    if "exit_price" not in frame.columns:
        frame["exit_price"] = frame.get("bid", 0)

    # Normalize
    train_mean = pd.Series(artifact["train_mean"])
    train_std = pd.Series(artifact["train_std"])
    frame[feature_columns] = (frame[feature_columns] - train_mean) / train_std
    frame[feature_columns] = frame[feature_columns].fillna(0.0)
    frame = frame[frame["relative_spread"] <= 0.50].reset_index(drop=True)

    # Restore raw values for operator features
    for col in ["days_to_exp", "moneyness", "delta", "implied_volatility",
                 "volume", "open_interest", "relative_spread", "ask",
                 "spy_d_close", "spy_d_sma_50", "spy_d_rsi", "spy_d_macd_hist",
                 "vix_d_close", "spy_momentum", "realized_vol_5d",
                 "realized_vol_20d", "realized_vol_60d", "vrp_20d"]:
        raw = f"{col}_raw"
        if raw in frame.columns:
            frame[col] = frame[raw]

    # Run backtest
    result = run_operator_backtest(model, operator, frame, feature_columns, device,
                                   starting_capital=args.starting_capital)

    # Print results
    m = result["metrics"]
    print()
    print("=" * 65)
    print("OPERATOR BACKTEST RESULTS (Regime-Trust Meta-Allocator)")
    print("=" * 65)
    print(f"  Starting capital:    ${m['starting_capital']:>12,.0f}")
    print(f"  Ending capital:      ${m.get('ending_capital', 0):>12,.0f}")
    print(f"  Total return:        {m['total_return_pct']:>12.1f}%")
    print(f"  Total trades:        {m['total_trades']:>12d}")
    print(f"  Days traded:         {m['days_traded']:>12d}/{m['total_days']}")
    print(f"  Days sat out:        {m.get('days_sat_out', 0):>12d}")
    if "win_rate" in m:
        print(f"  Win rate:            {m['win_rate']:>12.1f}%")
        print(f"  Avg trade return:    {m['avg_trade_return']:>12.1f}%")
        print(f"  Median trade return: {m['median_trade_return']:>12.1f}%")
        print(f"  Best trade:          {m['best_trade']:>12.1f}%")
        print(f"  Worst trade:         {m['worst_trade']:>12.1f}%")
    if "sharpe" in m:
        print(f"  Sharpe ratio:        {m['sharpe']:>12.2f}")
        print(f"  Max drawdown:        {m['max_drawdown_pct']:>12.1f}%")
    print()
    print("  Gate breakdown:")
    for gate in ["sit_out", "normal", "aggressive"]:
        key = f"gate_{gate}_trades"
        if key in m:
            print(f"    {gate:12s}: {m[key]:>5d} trades, win={m.get(f'gate_{gate}_win_rate', 0):.0f}%, avg_ret={m.get(f'gate_{gate}_avg_return', 0):.1f}%")
    print("=" * 65)

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(m, output_dir / "operator_backtest_metrics.json")
    result["equity_curve"].to_csv(output_dir / "operator_equity_curve.csv", index=False)
    if len(result["trades"]) > 0:
        result["trades"].to_csv(output_dir / "operator_trade_log.csv", index=False)
    logger.info("Results saved to %s", output_dir)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent / "updated_option_agent_codebase"))
    main()
