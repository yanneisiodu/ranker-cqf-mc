"""Train the Regime-Balanced Rolling Expert Ensemble.

Trains three experts with different windows and recency weights,
then trains a gate to fuse them.

Usage:
    python train_ensemble.py \
        --data year_2019_data.csv ... year_2025_data.csv \
        --val-data year_2025_data.csv \
        --config config_tuned.yaml \
        --output-dir ./ensemble_output \
        --recent-months 18 \
        --core-months 36 \
        --stress-replay-pct 0.10
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from logger import setup_logger
from neural_ranker import ChainTransformer, NeuralRankerConfig, get_device, ndcg_at_k
from train_neural_ranker import PrebuiltDataset, collate_chains, evaluate
from regime_ensemble import (
    assign_regime_bucket,
    is_stress_day,
    recency_weight,
    train_expert,
    train_gate,
)
from torch.utils.data import DataLoader
from utils import (
    apply_relevance_bins,
    compute_relevance_bins,
    load_config,
    prepare_model_frame,
    save_json,
    select_feature_columns,
    summarize_frame,
)

logger = setup_logger(__name__)


def build_groups_from_frame(
    frame: pd.DataFrame,
    feature_columns: List[str],
    edges: np.ndarray,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[pd.Timestamp], List[Dict]]:
    """Build daily groups + per-day metadata from a DataFrame."""
    frame["target_relevance"] = apply_relevance_bins(frame["target_return"], edges).astype(np.float32)
    frame["type_numeric"] = (frame["type"].str.lower() == "call").astype(np.float32)

    if "relative_spread" in frame.columns:
        frame = frame[frame["relative_spread"] <= 0.50]

    groups = []
    dates = []
    day_meta = []

    for date in sorted(frame["date"].unique()):
        day = frame[frame["date"] == date]
        if len(day) < 2:
            continue
        feats = np.nan_to_num(day[feature_columns].values.astype(np.float32))
        rels = day["target_relevance"].values.astype(np.float32)
        groups.append((feats, rels))
        dates.append(date)

        regime = assign_regime_bucket(day)
        stress = is_stress_day(day)
        day_meta.append({
            "date": date,
            "regime_bucket": regime,
            "is_stress": stress,
            "spy_momentum": day["spy_momentum"].iloc[0] if "spy_momentum" in day.columns else 0,
            "vix": day["vix_d_close"].iloc[0] if "vix_d_close" in day.columns else 20,
            "realized_vol_20d": day["realized_vol_20d"].iloc[0] if "realized_vol_20d" in day.columns else 0,
            "vrp_20d": day["vrp_20d"].iloc[0] if "vrp_20d" in day.columns else 0,
            "spy_rsi": day["spy_d_rsi"].iloc[0] if "spy_d_rsi" in day.columns else 50,
            "n_options": len(day),
        })

    return groups, dates, day_meta


def select_window(
    groups: List, dates: List, day_meta: List,
    end_date: pd.Timestamp, months: int,
) -> Tuple[List, np.ndarray]:
    """Select groups within a rolling window and compute recency weights."""
    start_date = end_date - pd.DateOffset(months=months)
    mask = [(d >= start_date and d <= end_date) for d in dates]
    selected = [g for g, m in zip(groups, mask) if m]
    n = len(selected)
    weights = np.array([recency_weight(i, n) for i in range(n)])
    return selected, weights


def select_stress_replay(
    groups: List, dates: List, day_meta: List,
    end_date: pd.Timestamp, recent_months: int = 30, replay_pct: float = 0.10,
) -> Tuple[List, np.ndarray]:
    """Build E_stress: recent data + curated stress replay."""
    start_recent = end_date - pd.DateOffset(months=recent_months)

    recent_groups = []
    stress_replay = []

    for g, d, meta in zip(groups, dates, day_meta):
        if d >= start_recent and d <= end_date:
            recent_groups.append(g)
        elif meta["is_stress"]:
            stress_replay.append(g)

    # Sample replay to target percentage
    n_recent = len(recent_groups)
    n_replay = max(1, int(n_recent * replay_pct / (1 - replay_pct)))
    if len(stress_replay) > n_replay:
        indices = np.random.choice(len(stress_replay), size=n_replay, replace=False)
        stress_replay = [stress_replay[i] for i in indices]

    # Combine: recent first, then replay
    combined = recent_groups + stress_replay
    n = len(combined)

    # Weights: recent get recency decay, replay gets lower fixed weight
    weights = np.ones(n)
    for i in range(len(recent_groups)):
        weights[i] = recency_weight(i, len(recent_groups))
    for i in range(len(recent_groups), n):
        weights[i] = 0.5  # replay weight

    logger.info("  E_stress: %d recent + %d stress replay = %d total", len(recent_groups), len(stress_replay), n)
    return combined, weights


def train_ensemble(
    data_files: Sequence[str],
    val_files: Sequence[str],
    config_file: str,
    output_dir: str,
    recent_months: int = 18,
    core_months: int = 36,
    stress_replay_pct: float = 0.10,
    epochs: int = 50,
    nrows: Optional[int] = None,
    base_artifact_path: Optional[str] = None,
):
    config = load_config(config_file)
    device = get_device()
    nr_config = NeuralRankerConfig.from_config(config)

    # Load all training data
    logger.info("Loading training data...")
    train_frame = prepare_model_frame(data_files, config, include_targets=True, nrows=nrows)
    logger.info("Train: %s", summarize_frame(train_frame))

    logger.info("Loading validation data...")
    val_frame = prepare_model_frame(val_files, config, include_targets=True, nrows=nrows)
    logger.info("Val: %s", summarize_frame(val_frame))

    # Feature setup
    feature_columns, _, _ = select_feature_columns(train_frame, config)
    num_features = [c for c in feature_columns if c != "type"] + ["type_numeric"]

    for frame in [train_frame, val_frame]:
        frame["type_numeric"] = (frame["type"].str.lower() == "call").astype(np.float32)

    edges = compute_relevance_bins(train_frame["target_return"], n_bins=5)

    # Normalize
    train_mean = train_frame[num_features].mean()
    train_std = train_frame[num_features].std().replace(0, 1)
    for frame in [train_frame, val_frame]:
        frame[num_features] = (frame[num_features] - train_mean) / train_std
        frame[num_features] = frame[num_features].fillna(0.0)

    # Build groups
    logger.info("Building daily groups...")
    all_groups, all_dates, all_meta = build_groups_from_frame(train_frame, num_features, edges)
    val_groups, val_dates, val_meta = build_groups_from_frame(val_frame, num_features, edges)
    logger.info("Train: %d days, Val: %d days", len(all_groups), len(val_groups))

    # Determine end of training period
    end_date = max(all_dates)
    logger.info("Training end date: %s", end_date)

    # Model config
    actual_config = NeuralRankerConfig(
        input_dim=len(num_features),
        embed_dim=nr_config.embed_dim,
        n_heads=nr_config.n_heads,
        n_layers=nr_config.n_layers,
        dropout=nr_config.dropout,
        mlp_hidden=nr_config.mlp_hidden,
        learning_rate=nr_config.learning_rate,
        weight_decay=nr_config.weight_decay,
        warmup_epochs=nr_config.warmup_epochs,
        epochs=epochs,
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load base model weights for warm-starting
    base_state = None
    if base_artifact_path:
        import torch as _torch
        base_artifact = _torch.load(base_artifact_path, map_location="cpu", weights_only=False)
        base_state = base_artifact["model_state_dict"]
        logger.info("Loaded base model for warm-starting from %s", base_artifact_path)

    # ── Train E_recent ──────────────────────────────────────────────────
    recent_groups, recent_weights = select_window(all_groups, all_dates, all_meta, end_date, recent_months)
    logger.info("E_recent: %d days (last %d months)", len(recent_groups), recent_months)
    recent_state, recent_ndcg, recent_epoch = train_expert(
        "E_recent", recent_groups, recent_weights, val_groups, actual_config, device, epochs=epochs, base_state=base_state,
    )
    torch.save({"state_dict": recent_state, "config": asdict(actual_config), "name": "recent",
                "ndcg": recent_ndcg, "epoch": recent_epoch},
               out / "expert_recent.pt")

    # ── Train E_core ────────────────────────────────────────────────────
    core_groups, core_weights = select_window(all_groups, all_dates, all_meta, end_date, core_months)
    logger.info("E_core: %d days (last %d months)", len(core_groups), core_months)
    core_state, core_ndcg, core_epoch = train_expert(
        "E_core", core_groups, core_weights, val_groups, actual_config, device, epochs=epochs, base_state=base_state,
    )
    torch.save({"state_dict": core_state, "config": asdict(actual_config), "name": "core",
                "ndcg": core_ndcg, "epoch": core_epoch},
               out / "expert_core.pt")

    # ── Train E_stress ──────────────────────────────────────────────────
    stress_groups, stress_weights = select_stress_replay(
        all_groups, all_dates, all_meta, end_date,
        recent_months=core_months, replay_pct=stress_replay_pct,
    )
    stress_state, stress_ndcg, stress_epoch = train_expert(
        "E_stress", stress_groups, stress_weights, val_groups, actual_config, device, epochs=epochs, base_state=base_state,
    )
    torch.save({"state_dict": stress_state, "config": asdict(actual_config), "name": "stress",
                "ndcg": stress_ndcg, "epoch": stress_epoch},
               out / "expert_stress.pt")

    # ── Train Gate ──────────────────────────────────────────────────────
    logger.info("Scoring all experts on validation for gate training...")

    # Score each val day with each expert
    expert_daily_ndcgs = {}
    for name, state in [("recent", recent_state), ("core", core_state), ("stress", stress_state)]:
        model = ChainTransformer(actual_config).to(device)
        model.load_state_dict(state)
        model.eval()

        ndcgs = []
        for feats, rels in val_groups:
            x = torch.from_numpy(feats).unsqueeze(0).to(device)
            pm = torch.zeros(1, feats.shape[0], dtype=torch.bool, device=device)
            with torch.no_grad():
                scores = model(x, padding_mask=pm).squeeze(0).cpu().numpy()
            ndcgs.append(ndcg_at_k(scores, rels, k=20))
        expert_daily_ndcgs[name] = np.array(ndcgs)
        del model; gc.collect()

    # Day features for gate
    gate_features = []
    for meta in val_meta:
        gate_features.append([
            meta.get("spy_momentum", 0),
            meta.get("vix", 20),
            meta.get("realized_vol_20d", 0),
            meta.get("vrp_20d", 0),
            meta.get("spy_rsi", 50),
            meta.get("regime_bucket", 0),
            meta.get("n_options", 0),
        ])
    gate_features = np.array(gate_features, dtype=np.float32)

    gate_artifact = train_gate(expert_daily_ndcgs, gate_features, out)

    # Save metadata
    summary = {
        "experts": {
            "recent": {"ndcg": recent_ndcg, "epoch": recent_epoch, "days": len(recent_groups)},
            "core": {"ndcg": core_ndcg, "epoch": core_epoch, "days": len(core_groups)},
            "stress": {"ndcg": stress_ndcg, "epoch": stress_epoch, "days": len(stress_groups)},
        },
        "config": asdict(actual_config),
        "feature_columns": num_features,
        "relevance_edges": edges.tolist(),
        "train_mean": train_mean.to_dict(),
        "train_std": train_std.to_dict(),
        "recent_months": recent_months,
        "core_months": core_months,
    }
    save_json(summary, out / "ensemble_summary.json")
    logger.info("Ensemble training complete. Saved to %s", out)

    # Also save normalization info as a separate artifact for inference
    torch.save({
        "feature_columns": num_features,
        "relevance_edges": edges,
        "train_mean": train_mean.to_dict(),
        "train_std": train_std.to_dict(),
        "config": asdict(actual_config),
    }, out / "ensemble_meta.pt")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Train regime-balanced expert ensemble")
    parser.add_argument("--data", nargs="+", required=True, help="Training CSV files")
    parser.add_argument("--val-data", nargs="+", required=True, help="Validation CSV files")
    parser.add_argument("--config", default="./config_tuned.yaml")
    parser.add_argument("--output-dir", default="./ensemble_output")
    parser.add_argument("--recent-months", type=int, default=18)
    parser.add_argument("--core-months", type=int, default=36)
    parser.add_argument("--stress-replay-pct", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--base-artifact", default=None, help="Base model for warm-starting experts")
    args = parser.parse_args()

    train_ensemble(
        data_files=args.data,
        val_files=args.val_data,
        config_file=args.config,
        output_dir=args.output_dir,
        recent_months=args.recent_months,
        core_months=args.core_months,
        stress_replay_pct=args.stress_replay_pct,
        epochs=args.epochs,
        nrows=args.nrows,
        base_artifact_path=args.base_artifact,
    )


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent / "updated_option_agent_codebase"))
    main()
