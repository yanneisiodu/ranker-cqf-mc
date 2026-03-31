"""Build the cross-fitted operator dataset from out-of-sample ranker predictions.

For each historical trading day, produces:
1. Day-level features (model trust, market regime, recent efficacy)
2. Candidate-level features (edge, tail risk, option structure)
3. Matured outcomes (did the basket work? which side won?)

This dataset is used to train the Regime-Trust Meta-Allocator.
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
from neural_ranker import ChainTransformer, NeuralRankerConfig, ndcg_at_k, get_device
from utils import (
    apply_relevance_bins,
    load_config,
    prepare_model_frame,
    select_feature_columns,
    save_json,
)

logger = setup_logger(__name__)

# Rolling windows for efficacy features
EFFICACY_WINDOWS = [5, 10, 20]


def compute_day_features(
    day: pd.DataFrame,
    scores: np.ndarray,
    history: pd.DataFrame,
    top_k: int = 40,
) -> Dict[str, float]:
    """Compute day-level features for the operator.

    Features fall into 4 families:
    1. Model-trust: score distribution, coherence
    2. Market-regime: SPY trend, VIX, vol
    3. Recent efficacy: rolling hit rates from matured baskets
    4. Option structure: what the top picks look like
    """
    features = {}
    day = day.copy()
    day["score"] = scores

    # Sort by score
    ranked = day.sort_values("score", ascending=False)
    top = ranked.head(top_k)

    # ── 1. Model-trust features ──────────────────────────────────────────
    features["score_top1"] = float(top["score"].iloc[0])
    features["score_top5_mean"] = float(top.head(5)["score"].mean())
    features["score_top20_mean"] = float(top.head(20)["score"].mean())
    features["score_spread_1_20"] = float(top["score"].iloc[0] - top.head(20)["score"].iloc[-1])
    features["score_spread_1_median"] = float(top["score"].iloc[0] - ranked["score"].median())
    features["score_std_top20"] = float(top.head(20)["score"].std())
    features["score_skew_top20"] = float(top.head(20)["score"].skew()) if len(top) >= 8 else 0.0

    # Call/put skew in top ranks
    top20 = top.head(20)
    n_calls = (top20["type"] == "call").sum()
    n_puts = (top20["type"] == "put").sum()
    features["top20_call_ratio"] = float(n_calls / max(n_calls + n_puts, 1))
    features["top20_put_ratio"] = float(n_puts / max(n_calls + n_puts, 1))

    # ── 2. Market-regime features ────────────────────────────────────────
    # These come from the option chain snapshot (one value per day)
    for col in ["spy_d_close", "spy_d_sma_50", "spy_d_rsi", "spy_d_macd_hist",
                "vix_d_close", "spy_momentum", "realized_vol_5d",
                "realized_vol_20d", "realized_vol_60d", "vrp_20d"]:
        if col in day.columns:
            val = day[col].iloc[0]
            features[f"regime_{col}"] = float(val) if np.isfinite(val) else 0.0

    # SPY trend signals
    if "spy_d_close" in day.columns and "spy_d_sma_50" in day.columns:
        spy = day["spy_d_close"].iloc[0]
        sma = day["spy_d_sma_50"].iloc[0]
        features["regime_spy_above_sma50"] = float(spy > sma) if np.isfinite(spy) and np.isfinite(sma) else 0.5

    # ── 3. Recent efficacy features (from matured history) ───────────────
    if len(history) > 0:
        for window in EFFICACY_WINDOWS:
            recent = history.tail(window)
            if len(recent) == 0:
                features[f"efficacy_{window}d_hit_rate"] = 0.5
                features[f"efficacy_{window}d_mean_return"] = 0.0
                features[f"efficacy_{window}d_call_hit_rate"] = 0.5
                features[f"efficacy_{window}d_put_hit_rate"] = 0.5
                features[f"efficacy_{window}d_catastrophe_rate"] = 0.0
                continue

            features[f"efficacy_{window}d_hit_rate"] = float((recent["basket_return"] > 0).mean())
            features[f"efficacy_{window}d_mean_return"] = float(recent["basket_return"].mean())

            if "call_return" in recent.columns:
                call_valid = recent["call_return"].dropna()
                features[f"efficacy_{window}d_call_hit_rate"] = float((call_valid > 0).mean()) if len(call_valid) > 0 else 0.5
            else:
                features[f"efficacy_{window}d_call_hit_rate"] = 0.5

            if "put_return" in recent.columns:
                put_valid = recent["put_return"].dropna()
                features[f"efficacy_{window}d_put_hit_rate"] = float((put_valid > 0).mean()) if len(put_valid) > 0 else 0.5
            else:
                features[f"efficacy_{window}d_put_hit_rate"] = 0.5

            features[f"efficacy_{window}d_catastrophe_rate"] = float((recent["basket_return"] < -0.5).mean())

        # Consecutive losses
        if len(history) >= 2:
            recent_returns = history["basket_return"].values
            consec = 0
            for r in reversed(recent_returns):
                if r < 0:
                    consec += 1
                else:
                    break
            features["efficacy_consecutive_losses"] = float(consec)
        else:
            features["efficacy_consecutive_losses"] = 0.0

        # Current drawdown from peak
        cumulative = (1 + history["basket_return"]).cumprod()
        peak = cumulative.cummax()
        dd = (cumulative / peak - 1).iloc[-1] if len(cumulative) > 0 else 0.0
        features["efficacy_current_drawdown"] = float(dd)
    else:
        # No history yet — neutral values
        for window in EFFICACY_WINDOWS:
            features[f"efficacy_{window}d_hit_rate"] = 0.5
            features[f"efficacy_{window}d_mean_return"] = 0.0
            features[f"efficacy_{window}d_call_hit_rate"] = 0.5
            features[f"efficacy_{window}d_put_hit_rate"] = 0.5
            features[f"efficacy_{window}d_catastrophe_rate"] = 0.0
        features["efficacy_consecutive_losses"] = 0.0
        features["efficacy_current_drawdown"] = 0.0

    # ── 4. Option-structure features ─────────────────────────────────────
    if "days_to_exp" in top20.columns:
        features["struct_avg_dte"] = float(top20["days_to_exp"].mean())
        features["struct_min_dte"] = float(top20["days_to_exp"].min())
        features["struct_pct_short_dte"] = float((top20["days_to_exp"] < 7).mean())

    if "moneyness" in top20.columns:
        features["struct_avg_moneyness"] = float(top20["moneyness"].mean())
        features["struct_pct_deep_otm"] = float((top20["moneyness"] < 0.95).mean())

    if "delta" in top20.columns:
        features["struct_avg_abs_delta"] = float(top20["delta"].abs().mean())
        features["struct_pct_low_delta"] = float((top20["delta"].abs() < 0.2).mean())

    if "implied_volatility" in top20.columns:
        features["struct_avg_iv"] = float(top20["implied_volatility"].mean())

    if "volume" in top20.columns:
        features["struct_avg_volume"] = float(top20["volume"].mean())

    if "relative_spread" in top20.columns:
        features["struct_avg_spread"] = float(top20["relative_spread"].mean())

    return features


def compute_matured_outcome(
    day: pd.DataFrame,
    scores: np.ndarray,
    top_k: int = 20,
    min_price: float = 0.10,
    max_spread: float = 0.30,
    min_volume: int = 10,
    min_oi: int = 50,
) -> Dict[str, float]:
    """Compute the matured basket outcome for a day's top TRADEABLE picks.

    Applies the same liquidity filters as the realistic backtest.
    """
    day = day.copy()
    day["score"] = scores

    # Filter to tradeable options (matching realistic_backtest.py)
    ask_col = "ask" if "ask_raw" not in day.columns else "ask_raw"
    vol_col = "volume" if "volume_raw" not in day.columns else "volume_raw"
    oi_col = "open_interest" if "open_interest_raw" not in day.columns else "open_interest_raw"
    spread_col = "relative_spread" if "relative_spread_raw" not in day.columns else "relative_spread_raw"

    tradeable = day[
        (day[ask_col] >= min_price) &
        (day[spread_col] <= max_spread) &
        (day[vol_col] >= min_volume) &
        (day[oi_col] >= min_oi)
    ]

    if len(tradeable) < 5:
        tradeable = day  # fallback to all if too few tradeable

    top = tradeable.sort_values("score", ascending=False).head(top_k)

    if "target_return" not in top.columns or top["target_return"].isna().all():
        return {}

    basket_return = top["target_return"].mean()
    calls = top[top["type"] == "call"]
    puts = top[top["type"] == "put"]

    outcome = {
        "basket_return": float(basket_return),
        "basket_win": float(basket_return > 0),
        "basket_hit_rate": float((top["target_return"] > 0).mean()),
        "basket_catastrophe": float((top["target_return"] < -0.5).mean()),
        "basket_best_return": float(top["target_return"].max()),
        "basket_worst_return": float(top["target_return"].min()),
        "call_return": float(calls["target_return"].mean()) if len(calls) > 0 else np.nan,
        "put_return": float(puts["target_return"].mean()) if len(puts) > 0 else np.nan,
        "call_hit_rate": float((calls["target_return"] > 0).mean()) if len(calls) > 0 else np.nan,
        "put_hit_rate": float((puts["target_return"] > 0).mean()) if len(puts) > 0 else np.nan,
        "n_calls": int(len(calls)),
        "n_puts": int(len(puts)),
    }

    # Utility label for gate training
    catastrophe_rate = outcome["basket_catastrophe"]
    concentration = abs(outcome["n_calls"] - outcome["n_puts"]) / max(len(top), 1)
    outcome["utility"] = float(basket_return - 0.5 * catastrophe_rate - 0.2 * concentration)

    # Exposure bucket label based on basket return directly
    # Calibrated to produce a reasonable distribution across buckets
    br = outcome["basket_return"]
    if br > 0.20:
        outcome["optimal_exposure_bucket"] = 5  # 50% — strong day
    elif br > 0.10:
        outcome["optimal_exposure_bucket"] = 4  # 40%
    elif br > 0.03:
        outcome["optimal_exposure_bucket"] = 3  # 30%
    elif br > 0.0:
        outcome["optimal_exposure_bucket"] = 2  # 20%
    elif br > -0.10:
        outcome["optimal_exposure_bucket"] = 1  # 10% — small loss, still tradeable
    else:
        outcome["optimal_exposure_bucket"] = 0  # 0% — sit out, bad day

    return outcome


def build_operator_dataset(
    model_artifact_path: str,
    data_files: list,
    config_file: str,
    output_dir: str,
):
    """Build the full operator dataset by scoring every day with the ranker."""

    # Load model
    artifact = torch.load(model_artifact_path, map_location="cpu", weights_only=False)
    nr_config = NeuralRankerConfig(**artifact["config"])
    model = ChainTransformer(nr_config)
    state = {k.replace("_orig_mod.", ""): v for k, v in artifact["model_state_dict"].items()}
    model.load_state_dict(state)
    device = get_device()
    model = model.to(device)
    model.eval()
    logger.info("Loaded model: %d params on %s", sum(p.numel() for p in model.parameters()), device)

    # Load data
    cfg = load_config(config_file)
    frame = prepare_model_frame(data_files, cfg, include_targets=True)
    logger.info("Loaded: %d rows, %d dates", len(frame), frame["date"].nunique())

    feature_columns = artifact["feature_columns"]
    edges = artifact["relevance_edges"]
    frame["target_relevance"] = apply_relevance_bins(frame["target_return"], edges).astype(np.float32)
    frame["type_numeric"] = (frame["type"].str.lower() == "call").astype(np.float32)

    # Save raw values before normalization
    raw_cols = ["days_to_exp", "moneyness", "delta", "implied_volatility",
                "volume", "open_interest", "relative_spread", "ask",
                "spy_d_close", "spy_d_sma_50",
                "spy_d_rsi", "spy_d_macd_hist", "vix_d_close", "spy_momentum",
                "realized_vol_5d", "realized_vol_20d", "realized_vol_60d", "vrp_20d"]
    for col in raw_cols:
        if col in frame.columns:
            frame[f"{col}_raw"] = frame[col].copy()

    # Normalize
    train_mean = pd.Series(artifact["train_mean"])
    train_std = pd.Series(artifact["train_std"])
    frame[feature_columns] = (frame[feature_columns] - train_mean) / train_std
    frame[feature_columns] = frame[feature_columns].fillna(0.0)
    frame = frame[frame["relative_spread"] <= 0.50].reset_index(drop=True)

    # Use raw columns for operator features
    for col in raw_cols:
        raw_col = f"{col}_raw"
        if raw_col in frame.columns:
            frame[col] = frame[raw_col]

    # Score each day and build dataset
    dates = sorted(frame["date"].unique())
    logger.info("Scoring %d dates...", len(dates))

    day_rows = []
    history_df = pd.DataFrame()
    pending_outcomes = []  # maturity queue: outcomes settle on exit_date

    for i, date in enumerate(dates):
        day = frame[frame["date"] == date]
        if len(day) < 20:
            continue

        # Score with ranker
        features = np.nan_to_num(day[feature_columns].values.astype(np.float32))
        x = torch.from_numpy(features).unsqueeze(0).to(device)
        with torch.no_grad():
            scores = model(x).squeeze(0).cpu().numpy()

        # Compute day features using matured history
        day_feats = compute_day_features(day, scores, history_df, top_k=40)
        day_feats["date"] = str(date)

        # Mature pending outcomes whose exit_date <= current date
        still_pending = []
        for pend in pending_outcomes:
            if pend["exit_date"] <= date:
                history_df = pd.concat([history_df, pd.DataFrame([{
                    "date": pend["entry_date"],
                    "basket_return": pend["basket_return"],
                    "call_return": pend.get("call_return", np.nan),
                    "put_return": pend.get("put_return", np.nan),
                }])], ignore_index=True)
            else:
                still_pending.append(pend)
        pending_outcomes = still_pending

        # Compute outcome (labels) but queue to exit_date
        outcome = compute_matured_outcome(day, scores, top_k=20)
        if outcome:
            day_feats.update(outcome)
            exit_date = day["exit_date"].dropna().iloc[0] if "exit_date" in day.columns and len(day["exit_date"].dropna()) > 0 else date + pd.Timedelta(days=7)
            pending_outcomes.append({
                "exit_date": exit_date,
                "entry_date": date,
                "basket_return": outcome["basket_return"],
                "call_return": outcome.get("call_return", np.nan),
                "put_return": outcome.get("put_return", np.nan),
            })

        day_rows.append(day_feats)

        if (i + 1) % 100 == 0:
            logger.info("  %d/%d dates processed", i + 1, len(dates))

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset = pd.DataFrame(day_rows)
    dataset_path = output_path / "operator_dataset.csv"
    dataset.to_csv(dataset_path, index=False)
    logger.info("Saved operator dataset: %d rows, %d columns to %s",
                len(dataset), len(dataset.columns), dataset_path)

    # Summary stats
    if "basket_return" in dataset.columns:
        logger.info("Dataset summary:")
        logger.info("  Days: %d", len(dataset))
        logger.info("  Basket win rate: %.1f%%", (dataset["basket_return"] > 0).mean() * 100)
        logger.info("  Mean basket return: %.2f%%", dataset["basket_return"].mean() * 100)
        logger.info("  Exposure bucket distribution:")
        if "optimal_exposure_bucket" in dataset.columns:
            for bucket in range(6):
                pct = (dataset["optimal_exposure_bucket"] == bucket).mean() * 100
                logger.info("    Bucket %d (%d%%): %.1f%% of days", bucket, bucket * 10, pct)

    return str(dataset_path)


def main():
    parser = argparse.ArgumentParser(description="Build operator dataset from ranker predictions")
    parser.add_argument("--model-artifact", required=True)
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--config", default="./config_tuned.yaml")
    parser.add_argument("--output-dir", default="./operator_data")
    args = parser.parse_args()

    build_operator_dataset(args.model_artifact, args.data, args.config, args.output_dir)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent / "updated_option_agent_codebase"))
    main()
