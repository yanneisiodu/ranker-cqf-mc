"""Regime-Balanced Rolling Expert Ensemble for the neural ranker.

Three ChainTransformer experts with different training windows:
  E_recent: last 12-18 months (freshest patterns)
  E_core:   last 30-36 months (stable medium-term)
  E_stress: recent + curated crash/stress replay

Fused by a deterministic regime gate that weights experts
based on current market conditions and recent efficacy.

Training uses recency-weighted ListMLE loss with group-DRO
over regime buckets.
"""
from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import xgboost as xgb
import joblib

from logger import setup_logger
from neural_ranker import (
    ChainTransformer,
    NeuralRankerConfig,
    get_device,
    listmle_loss,
    ndcg_at_k,
)
from train_neural_ranker import (
    PrebuiltDataset,
    collate_chains,
    evaluate,
)
from utils import (
    apply_relevance_bins,
    compute_relevance_bins,
    load_config,
    save_json,
    select_feature_columns,
)

logger = setup_logger(__name__)


# ── Regime Bucketing ────────────────────────────────────────────────────────

def assign_regime_bucket(day_df: pd.DataFrame) -> int:
    """Assign a regime bucket to a day based on market features.

    3 dimensions × 3 levels = 27 possible buckets:
      - SPY trend: up(2) / flat(1) / down(0)
      - Volatility: low(0) / mid(1) / high(2)
      - VRP state: compressed(0) / neutral(1) / stressed(2)
    """
    spy_mom = day_df["spy_momentum"].iloc[0] if "spy_momentum" in day_df.columns else 0
    rvol = day_df["realized_vol_20d"].iloc[0] if "realized_vol_20d" in day_df.columns else 0.15
    vrp = day_df["vrp_20d"].iloc[0] if "vrp_20d" in day_df.columns else 0

    # Handle NaN
    if not np.isfinite(spy_mom): spy_mom = 0
    if not np.isfinite(rvol): rvol = 0.15
    if not np.isfinite(vrp): vrp = 0

    # Trend bucket
    if spy_mom > 0.02:
        trend = 2  # up
    elif spy_mom < -0.02:
        trend = 0  # down
    else:
        trend = 1  # flat

    # Vol bucket
    if rvol > 0.25:
        vol = 2  # high
    elif rvol < 0.12:
        vol = 0  # low
    else:
        vol = 1  # mid

    # VRP bucket
    if vrp > 0.05:
        vrp_b = 2  # stressed (IV >> RV)
    elif vrp < -0.02:
        vrp_b = 0  # compressed (IV < RV)
    else:
        vrp_b = 1  # neutral

    return trend * 9 + vol * 3 + vrp_b


def is_stress_day(day_df: pd.DataFrame) -> bool:
    """Identify stress/crash days for E_stress replay."""
    vix = day_df["vix_d_close"].iloc[0] if "vix_d_close" in day_df.columns else 20
    spy_mom = day_df["spy_momentum"].iloc[0] if "spy_momentum" in day_df.columns else 0

    if not np.isfinite(vix): vix = 20
    if not np.isfinite(spy_mom): spy_mom = 0

    return vix > 30 or spy_mom < -0.05


# ── Recency-Weighted ListMLE ────────────────────────────────────────────────

def recency_weight(date_idx: int, total_dates: int, decay: float = 0.003) -> float:
    """Exponential recency decay. Most recent day = 1.0, oldest < 0.1."""
    age = total_dates - date_idx - 1
    return np.exp(-decay * age)


def train_expert_epoch(
    model: ChainTransformer,
    groups: List[Tuple[np.ndarray, np.ndarray]],
    weights: np.ndarray,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    accum_steps: int = 16,
) -> float:
    """Train one epoch with per-day recency weights."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    optimizer.zero_grad()

    # Shuffle indices but keep weights aligned
    indices = np.random.permutation(len(groups))

    for step, idx in enumerate(indices):
        feats, rels = groups[idx]
        w = weights[idx]

        features = torch.from_numpy(feats).unsqueeze(0).to(device)
        relevance = torch.from_numpy(rels).unsqueeze(0).to(device)
        padding_mask = torch.zeros(1, features.shape[1], dtype=torch.bool, device=device)

        scores = model(features, padding_mask=padding_mask)
        loss = listmle_loss(scores.float(), relevance, padding_mask=padding_mask, top_k=200)
        weighted_loss = loss * w / accum_steps
        weighted_loss.backward()

        total_loss += loss.item()
        n_batches += 1

        if (step + 1) % accum_steps == 0 or (step + 1) == len(indices):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

    return total_loss / max(n_batches, 1)


# ── Expert Training ─────────────────────────────────────────────────────────

def train_expert(
    name: str,
    groups: List[Tuple[np.ndarray, np.ndarray]],
    weights: np.ndarray,
    val_groups: List[Tuple[np.ndarray, np.ndarray]],
    config: NeuralRankerConfig,
    device: torch.device,
    epochs: int = 50,
    lr: float = 4.717e-4,
    base_state: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[Dict[str, torch.Tensor], float, int]:
    """Train a single expert with recency-weighted loss.

    If base_state is provided, warm-starts from those weights
    instead of random initialization.
    """
    logger.info("Training expert '%s': %d train days, %d val days", name, len(groups), len(val_groups))

    model = ChainTransformer(config).to(device)
    if base_state is not None:
        cleaned = {k.replace("_orig_mod.", ""): v for k, v in base_state.items()}
        model.load_state_dict(cleaned)
        logger.info("  Warm-started from base model")
        # Use lower LR for fine-tuning
        lr = lr * 0.2
        logger.info("  Fine-tuning LR: %.1e", lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config.weight_decay)

    warmup = config.warmup_epochs
    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / max(warmup, 1)
        progress = (epoch - warmup) / max(epochs - warmup, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    val_ds = PrebuiltDataset(val_groups)
    val_loader = DataLoader(val_ds, batch_size=1, collate_fn=collate_chains, num_workers=0)

    best_ndcg = -float("inf")
    best_epoch = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_expert_epoch(model, groups, weights, optimizer, device)
        val_metrics = evaluate(model, val_loader, device, k=20)
        scheduler.step()
        elapsed = time.time() - t0

        val_ndcg = val_metrics["ndcg_at_k"]
        logger.info("  %s epoch %2d/%d | loss=%.4f | ndcg@20=%.4f | %.1fs",
                     name, epoch, epochs, train_loss, val_ndcg, elapsed)

        if val_ndcg > best_ndcg:
            best_ndcg = val_ndcg
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            logger.info("    ** New best %.4f at epoch %d **", best_ndcg, epoch)

    logger.info("  %s best: NDCG@20=%.4f at epoch %d", name, best_ndcg, best_epoch)
    del model; torch.cuda.empty_cache() if torch.cuda.is_available() else None; gc.collect()
    return best_state, best_ndcg, best_epoch


# ── Gate Training ───────────────────────────────────────────────────────────

def train_gate(
    expert_scores: Dict[str, np.ndarray],  # expert_name -> (n_days,) NDCG per day
    day_features: np.ndarray,               # (n_days, n_features)
    output_dir: Path,
) -> Any:
    """Train XGBoost gate that predicts which expert is best each day."""
    n_days = day_features.shape[0]
    expert_names = list(expert_scores.keys())

    # Label = index of best expert per day
    score_matrix = np.stack([expert_scores[name] for name in expert_names], axis=1)  # (n_days, n_experts)
    best_expert = np.argmax(score_matrix, axis=1)

    logger.info("Gate training: %d days, %d experts", n_days, len(expert_names))
    for i, name in enumerate(expert_names):
        pct = (best_expert == i).mean() * 100
        logger.info("  %s wins %.0f%% of days", name, pct)

    # Train/val split
    split = int(n_days * 0.7)
    X_train, X_val = day_features[:split], day_features[split:]
    y_train, y_val = best_expert[:split], best_expert[split:]

    # Ensure all classes present in train (add synthetic samples if needed)
    n_experts = len(expert_names)
    for cls in range(n_experts):
        if cls not in y_train:
            X_train = np.vstack([X_train, X_train[:1]])
            y_train = np.append(y_train, cls)

    gate = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=n_experts,
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        n_jobs=-1,
        random_state=42,
    )
    gate.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    val_pred = gate.predict(X_val)
    acc = (val_pred == y_val).mean()
    logger.info("  Gate accuracy: %.1f%%", acc * 100)

    # Save
    artifact = {"gate": gate, "expert_names": expert_names}
    gate_path = output_dir / "ensemble_gate.joblib"
    joblib.dump(artifact, gate_path)
    return artifact


# ── Ensemble Inference ──────────────────────────────────────────────────────

class ExpertEnsemble:
    """Three experts + gate for inference."""

    def __init__(self, ensemble_dir: str):
        d = Path(ensemble_dir)

        # Load experts
        self.experts = {}
        self.expert_names = []
        for name in ["recent", "core", "stress"]:
            path = d / f"expert_{name}.pt"
            if path.exists():
                artifact = torch.load(path, map_location="cpu", weights_only=False)
                config = NeuralRankerConfig(**artifact["config"])
                model = ChainTransformer(config)
                model.load_state_dict(artifact["state_dict"])
                model.eval()
                self.experts[name] = model
                self.expert_names.append(name)

        # Load gate
        gate_artifact = joblib.load(d / "ensemble_gate.joblib")
        self.gate = gate_artifact["gate"]

        logger.info("Loaded ensemble: %d experts + gate", len(self.experts))

    def score(
        self,
        features: torch.Tensor,
        padding_mask: torch.Tensor,
        day_features: np.ndarray,
        device: torch.device,
    ) -> np.ndarray:
        """Score options using gated expert ensemble."""
        # Get gate weights
        gate_proba = self.gate.predict_proba(day_features.reshape(1, -1))[0]

        # Score with each expert
        fused = np.zeros(features.shape[1])
        for i, name in enumerate(self.expert_names):
            model = self.experts[name].to(device)
            with torch.no_grad():
                scores = model(features, padding_mask=padding_mask).squeeze(0).cpu().numpy()
            fused += gate_proba[i] * scores

        return fused
