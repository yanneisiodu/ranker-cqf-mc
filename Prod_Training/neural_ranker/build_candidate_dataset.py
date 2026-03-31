"""Build candidate-level meta-labeling dataset.

For each day's top-40 tradeable candidates, produces:
- Candidate features (option structure + ranker score + market context)
- Labels: good_trade (1 if return > hurdle), catastrophe (1 if return < -0.80)

This dataset trains the per-trade meta-models that decide accept/reject.
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
from utils import (
    apply_relevance_bins,
    load_config,
    prepare_model_frame,
    save_json,
)

logger = setup_logger(__name__)

# Tradability filters (match realistic_backtest.py)
MIN_PRICE = 0.10
MAX_SPREAD = 0.30
MIN_VOLUME = 10
MIN_OI = 50
TOP_M = 40
RETURN_HURDLE = 0.0  # net positive return after spread
CATASTROPHE_THRESHOLD = -0.80  # lose 80%+ of position


def build_candidate_dataset(
    model_artifact_path: str,
    data_files: list,
    config_file: str,
    output_dir: str,
):
    # Load model
    artifact = torch.load(model_artifact_path, map_location="cpu", weights_only=False)
    nr_config = NeuralRankerConfig(**artifact["config"])
    model = ChainTransformer(nr_config)
    state = {k.replace("_orig_mod.", ""): v for k, v in artifact["model_state_dict"].items()}
    model.load_state_dict(state)
    device = get_device()
    model = model.to(device)
    model.eval()
    logger.info("Loaded model on %s", device)

    # Load data
    cfg = load_config(config_file)
    frame = prepare_model_frame(data_files, cfg, include_targets=True)
    logger.info("Loaded: %d rows, %d dates", len(frame), frame["date"].nunique())

    feature_columns = artifact["feature_columns"]
    edges = artifact["relevance_edges"]
    frame["target_relevance"] = apply_relevance_bins(frame["target_return"], edges).astype(np.float32)
    frame["type_numeric"] = (frame["type"].str.lower() == "call").astype(np.float32)

    # Save raw values before normalization
    raw_save = ["ask", "bid", "volume", "open_interest", "relative_spread",
                "strike", "days_to_exp", "delta", "moneyness", "implied_volatility",
                "spy_d_close", "spy_d_sma_50", "spy_d_rsi", "spy_d_macd_hist",
                "vix_d_close", "spy_momentum", "realized_vol_5d", "realized_vol_20d",
                "realized_vol_60d", "vrp_20d", "gamma", "theta", "vega",
                "vanna", "charm", "vomma"]
    for col in raw_save:
        if col in frame.columns:
            frame[f"{col}_raw"] = frame[col].copy()

    # Normalize features for ranker scoring
    train_mean = pd.Series(artifact["train_mean"])
    train_std = pd.Series(artifact["train_std"])
    frame[feature_columns] = (frame[feature_columns] - train_mean) / train_std
    frame[feature_columns] = frame[feature_columns].fillna(0.0)
    frame = frame[frame["relative_spread"] <= 0.50].reset_index(drop=True)

    # Restore raw values for candidate features
    for col in raw_save:
        raw = f"{col}_raw"
        if raw in frame.columns:
            frame[col] = frame[raw]

    # Rolling efficacy tracking with maturity queue
    # Outcomes are queued by exit_date and only mature when current_date >= exit_date
    recent_call_returns = []
    recent_put_returns = []
    recent_basket_returns = []
    pending_outcomes = []  # list of (exit_date, call_returns, put_returns, basket_return)

    dates = sorted(frame["date"].unique())
    logger.info("Processing %d dates...", len(dates))

    all_candidates = []

    for i, date in enumerate(dates):
        # Mature pending outcomes whose exit_date <= current date
        still_pending = []
        for pend in pending_outcomes:
            if pend["exit_date"] <= date:
                recent_call_returns.extend(pend["call_returns"])
                recent_put_returns.extend(pend["put_returns"])
                if np.isfinite(pend["basket_return"]):
                    recent_basket_returns.append(pend["basket_return"])
            else:
                still_pending.append(pend)
        pending_outcomes = still_pending
        day = frame[frame["date"] == date]
        if len(day) < 20:
            continue

        # Score
        features = np.nan_to_num(day[feature_columns].values.astype(np.float32))
        x = torch.from_numpy(features).unsqueeze(0).to(device)
        with torch.no_grad():
            scores = model(x).squeeze(0).cpu().numpy()
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
            continue

        # Top M candidates
        top = tradeable.sort_values("score", ascending=False).head(TOP_M)

        # Market context (same for all candidates on this day)
        spy_close = day["spy_d_close"].iloc[0] if "spy_d_close" in day.columns else 0
        spy_sma50 = day["spy_d_sma_50"].iloc[0] if "spy_d_sma_50" in day.columns else 0
        spy_rsi = day["spy_d_rsi"].iloc[0] if "spy_d_rsi" in day.columns else 50
        vix = day["vix_d_close"].iloc[0] if "vix_d_close" in day.columns else 20
        spy_momentum = day["spy_momentum"].iloc[0] if "spy_momentum" in day.columns else 0
        rvol_20 = day["realized_vol_20d"].iloc[0] if "realized_vol_20d" in day.columns else 0
        vrp = day["vrp_20d"].iloc[0] if "vrp_20d" in day.columns else 0
        is_bull = float(spy_close > spy_sma50) if np.isfinite(spy_close) and np.isfinite(spy_sma50) else 0.5

        # Rolling efficacy (5-day lookback from matured results)
        call_hit_5d = np.mean([r > 0 for r in recent_call_returns[-25:]]) if len(recent_call_returns) >= 5 else 0.5
        put_hit_5d = np.mean([r > 0 for r in recent_put_returns[-25:]]) if len(recent_put_returns) >= 5 else 0.5
        basket_hit_5d = np.mean([r > 0 for r in recent_basket_returns[-5:]]) if len(recent_basket_returns) >= 1 else 0.5
        consec_losses = 0
        for r in reversed(recent_basket_returns):
            if r < 0:
                consec_losses += 1
            else:
                break

        # Score distribution features
        score_top1 = top["score"].iloc[0]
        score_spread = top["score"].iloc[0] - top["score"].iloc[-1] if len(top) > 1 else 0
        call_ratio_top20 = (top.head(20)["type"] == "call").mean()

        for rank, (_, row) in enumerate(top.iterrows(), 1):
            ret = row.get("target_return", np.nan)

            candidate = {
                "date": str(date),
                "contractid": row.get("contractid", ""),
                "type": row.get("type", ""),
                "rank": rank,

                # Ranker output features
                "score": row["score"],
                "score_rank_pct": rank / len(top),
                "score_gap_to_top": score_top1 - row["score"],
                "score_spread_top_m": score_spread,
                "call_ratio_top20": call_ratio_top20,

                # Option structure features
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

                # Market regime features
                "spy_rsi": spy_rsi,
                "vix": vix,
                "spy_momentum": spy_momentum,
                "realized_vol_20d": rvol_20,
                "vrp_20d": vrp,
                "is_bull": is_bull,

                # Efficacy features
                "call_hit_5d": call_hit_5d,
                "put_hit_5d": put_hit_5d,
                "basket_hit_5d": basket_hit_5d,
                "consecutive_losses": consec_losses,

                # Labels
                "target_return": ret,
                "good_trade": int(ret > RETURN_HURDLE) if np.isfinite(ret) else np.nan,
                "catastrophe": int(ret < CATASTROPHE_THRESHOLD) if np.isfinite(ret) else np.nan,
            }
            all_candidates.append(candidate)

        # Queue outcomes to mature on exit_date (NOT same-day)
        if "target_return" in top.columns and "exit_date" in top.columns:
            exit_date = top["exit_date"].dropna().iloc[0] if len(top["exit_date"].dropna()) > 0 else date + pd.Timedelta(days=7)
            calls = top[top["type"] == "call"]["target_return"].dropna()
            puts = top[top["type"] == "put"]["target_return"].dropna()
            basket_ret = top["target_return"].mean()
            pending_outcomes.append({
                "exit_date": exit_date,
                "call_returns": calls.tolist(),
                "put_returns": puts.tolist(),
                "basket_return": basket_ret,
            })

        if (i + 1) % 100 == 0:
            logger.info("  %d/%d dates, %d candidates", i + 1, len(dates), len(all_candidates))

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(all_candidates)
    df.to_csv(output_path / "candidate_dataset.csv", index=False)

    # Split by side
    calls = df[df["type"] == "call"]
    puts = df[df["type"] == "put"]
    calls.to_csv(output_path / "candidate_calls.csv", index=False)
    puts.to_csv(output_path / "candidate_puts.csv", index=False)

    logger.info("Saved candidate dataset: %d total (%d calls, %d puts)", len(df), len(calls), len(puts))
    logger.info("  Good trade rate (calls): %.1f%%", calls["good_trade"].mean() * 100 if len(calls) > 0 else 0)
    logger.info("  Good trade rate (puts):  %.1f%%", puts["good_trade"].mean() * 100 if len(puts) > 0 else 0)
    logger.info("  Catastrophe rate (calls): %.1f%%", calls["catastrophe"].mean() * 100 if len(calls) > 0 else 0)
    logger.info("  Catastrophe rate (puts):  %.1f%%", puts["catastrophe"].mean() * 100 if len(puts) > 0 else 0)


def main():
    parser = argparse.ArgumentParser(description="Build candidate-level meta-labeling dataset")
    parser.add_argument("--model-artifact", required=True)
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--config", default="./config_tuned.yaml")
    parser.add_argument("--output-dir", default="./candidate_data")
    args = parser.parse_args()

    build_candidate_dataset(args.model_artifact, args.data, args.config, args.output_dir)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent / "updated_option_agent_codebase"))
    main()
