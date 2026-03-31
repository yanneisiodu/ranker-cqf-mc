"""Elastic Weight Consolidation (EWC) fine-tuning for the neural ranker.

Loads a pre-trained ranker (e.g. 2018-2023), computes Fisher information
on the old training data, then fine-tunes on new data (e.g. 2024-2025)
with a penalty that protects important old-task weights.

Usage:
    python ewc_finetune.py \
        --base-artifact ./neural_ranker_artifact.pt \
        --old-data year_2023_data.csv \
        --new-data year_2024_data.csv year_2025_data.csv \
        --val-data year_2025_data.csv \
        --config ./config_tuned.yaml \
        --output-dir ./ewc_output
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from logger import setup_logger
from neural_ranker import (
    ChainTransformer,
    NeuralRankerConfig,
    get_device,
    listmle_loss,
    ndcg_at_k,
)
from train_neural_ranker import (
    DailyChainDataset,
    PrebuiltDataset,
    collate_chains,
    evaluate,
)
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


def compute_fisher(
    model: ChainTransformer,
    loader: DataLoader,
    device: torch.device,
    n_samples: int = 200,
) -> Dict[str, torch.Tensor]:
    """Compute diagonal Fisher information matrix.

    Approximates importance of each weight by averaging squared gradients
    over a sample of the old training data.
    """
    model.eval()
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}

    count = 0
    for features, relevance, padding_mask in loader:
        if count >= n_samples:
            break

        features = features.to(device)
        relevance = relevance.to(device)
        padding_mask = padding_mask.to(device)

        model.zero_grad()
        scores = model(features, padding_mask=padding_mask)
        loss = listmle_loss(scores.float(), relevance, padding_mask=padding_mask, top_k=200)
        loss.backward()

        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher[n] += p.grad.detach() ** 2

        count += 1

    # Average
    for n in fisher:
        fisher[n] /= max(count, 1)

    logger.info("Computed Fisher over %d samples, %d parameters", count, len(fisher))
    return fisher


def ewc_penalty(
    model: ChainTransformer,
    fisher: Dict[str, torch.Tensor],
    old_params: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Compute EWC penalty: Σ Fisher(w) * (w - w_old)²"""
    penalty = torch.tensor(0.0, device=next(model.parameters()).device)
    for n, p in model.named_parameters():
        if n in fisher and n in old_params:
            penalty += (fisher[n] * (p - old_params[n]) ** 2).sum()
    return penalty


def finetune_with_ewc(
    base_artifact_path: str,
    old_data_files: Sequence[str],
    new_data_files: Sequence[str],
    val_data_files: Sequence[str],
    config_file: Optional[str] = None,
    output_dir: Optional[str] = None,
    ewc_lambda: float = 1000.0,
    finetune_lr: float = 5e-5,
    finetune_epochs: int = 20,
    fisher_samples: int = 200,
    nrows: Optional[int] = None,
) -> Dict[str, Any]:
    """Fine-tune a pre-trained ranker with EWC protection."""

    config = load_config(config_file)
    device = get_device()
    logger.info("Device: %s", device)

    # Load base model
    artifact = torch.load(base_artifact_path, map_location="cpu", weights_only=False)
    nr_config = NeuralRankerConfig(**artifact["config"])
    model = ChainTransformer(nr_config).to(device)
    state = {k.replace("_orig_mod.", ""): v for k, v in artifact["model_state_dict"].items()}
    model.load_state_dict(state)
    logger.info("Loaded base model: %d params", sum(p.numel() for p in model.parameters()))

    feature_columns = artifact["feature_columns"]
    edges = artifact["relevance_edges"]
    train_mean = pd.Series(artifact["train_mean"])
    train_std = pd.Series(artifact["train_std"])

    def prepare_frame(data_files):
        frame = prepare_model_frame(data_files, config, include_targets=True, nrows=nrows)
        frame["type_numeric"] = (frame["type"].str.lower() == "call").astype(np.float32)
        frame["target_relevance"] = apply_relevance_bins(frame["target_return"], edges).astype(np.float32)
        frame[feature_columns] = (frame[feature_columns] - train_mean) / train_std
        frame[feature_columns] = frame[feature_columns].fillna(0.0)
        return frame

    # ── Step 1: Compute Fisher on old data ───────────────────────────────
    logger.info("Loading old data for Fisher computation...")
    old_frame = prepare_frame(old_data_files)
    old_ds = DailyChainDataset(old_frame, feature_columns, max_spread=0.50)
    old_loader = DataLoader(old_ds, batch_size=1, shuffle=True, collate_fn=collate_chains, num_workers=0)
    logger.info("Old data: %d days", len(old_ds))

    fisher = compute_fisher(model, old_loader, device, n_samples=fisher_samples)

    # Save old params as anchor
    old_params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}

    # ── Step 2: Prepare new data for fine-tuning ─────────────────────────
    logger.info("Loading new data for fine-tuning...")
    new_frame = prepare_frame(new_data_files)
    new_ds = DailyChainDataset(new_frame, feature_columns, max_spread=0.50)
    new_loader = DataLoader(new_ds, batch_size=1, shuffle=True, collate_fn=collate_chains, num_workers=0)
    logger.info("New data: %d days", len(new_ds))

    logger.info("Loading validation data...")
    val_frame = prepare_frame(val_data_files)
    val_ds = DailyChainDataset(val_frame, feature_columns, max_spread=0.50)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_chains, num_workers=0)
    logger.info("Val data: %d days", len(val_ds))

    # ── Step 3: Fine-tune with EWC ───────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=finetune_lr, weight_decay=nr_config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=finetune_epochs, eta_min=1e-6)

    logger.info("Fine-tuning: lr=%.1e, ewc_lambda=%.0f, epochs=%d", finetune_lr, ewc_lambda, finetune_epochs)

    best_ndcg = -float("inf")
    best_epoch = 0
    best_state = None
    history = []

    # Evaluate base model before fine-tuning
    base_metrics = evaluate(model, val_loader, device, k=20)
    logger.info("Base model val NDCG@20 = %.4f", base_metrics["ndcg_at_k"])

    for epoch in range(1, finetune_epochs + 1):
        t0 = time.time()
        model.train()
        total_rank_loss = 0.0
        total_ewc_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()

        accum_steps = 16
        for step, (features, relevance, padding_mask) in enumerate(new_loader):
            features = features.to(device)
            relevance = relevance.to(device)
            padding_mask = padding_mask.to(device)

            scores = model(features, padding_mask=padding_mask)
            rank_loss = listmle_loss(scores.float(), relevance, padding_mask=padding_mask, top_k=200)
            ewc_loss_val = ewc_penalty(model, fisher, old_params)
            loss = (rank_loss + ewc_lambda * ewc_loss_val) / accum_steps
            loss.backward()

            total_rank_loss += rank_loss.item()
            total_ewc_loss += ewc_loss_val.item()
            n_batches += 1

            if (step + 1) % accum_steps == 0 or (step + 1) == len(new_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

        scheduler.step()

        val_metrics = evaluate(model, val_loader, device, k=20)
        elapsed = time.time() - t0
        val_ndcg = val_metrics["ndcg_at_k"]
        lr = optimizer.param_groups[0]["lr"]

        logger.info(
            "Epoch %2d/%d | rank=%.4f ewc=%.6f | ndcg@1=%.3f @5=%.3f @10=%.3f @20=%.3f | lr=%.1e | %.1fs",
            epoch, finetune_epochs,
            total_rank_loss / max(n_batches, 1),
            total_ewc_loss / max(n_batches, 1),
            val_metrics.get("ndcg_at_1", 0),
            val_metrics.get("ndcg_at_5", 0),
            val_metrics.get("ndcg_at_10", 0),
            val_ndcg, lr, elapsed,
        )

        history.append({
            "epoch": epoch,
            "rank_loss": total_rank_loss / max(n_batches, 1),
            "ewc_loss": total_ewc_loss / max(n_batches, 1),
            "val_ndcg": val_ndcg,
            "lr": lr,
        })

        if val_ndcg > best_ndcg:
            best_ndcg = val_ndcg
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            logger.info("  ** New best NDCG@20 = %.4f at epoch %d **", best_ndcg, epoch)

    # Save
    if best_state is not None:
        model.load_state_dict(best_state)

    root = Path(output_dir or "./ewc_output")
    root.mkdir(parents=True, exist_ok=True)

    ewc_artifact = {
        "model_state_dict": best_state,
        "config": asdict(nr_config),
        "feature_columns": feature_columns,
        "relevance_edges": edges,
        "train_mean": artifact["train_mean"],
        "train_std": artifact["train_std"],
        "best_epoch": best_epoch,
        "best_ndcg": best_ndcg,
        "ewc_lambda": ewc_lambda,
        "finetune_lr": finetune_lr,
        "base_ndcg": base_metrics["ndcg_at_k"],
    }
    artifact_path = root / "ewc_ranker_artifact.pt"
    torch.save(ewc_artifact, artifact_path)
    logger.info("Saved EWC artifact to %s", artifact_path)

    save_json({
        "base_ndcg": base_metrics["ndcg_at_k"],
        "best_ndcg": best_ndcg,
        "best_epoch": best_epoch,
        "ewc_lambda": ewc_lambda,
        "finetune_lr": finetune_lr,
        "improvement": best_ndcg - base_metrics["ndcg_at_k"],
        "history": history,
    }, root / "ewc_metrics.json")

    logger.info("Base NDCG: %.4f -> EWC NDCG: %.4f (improvement: %+.4f)",
                base_metrics["ndcg_at_k"], best_ndcg, best_ndcg - base_metrics["ndcg_at_k"])

    return {
        "artifact_path": str(artifact_path),
        "base_ndcg": base_metrics["ndcg_at_k"],
        "best_ndcg": best_ndcg,
        "best_epoch": best_epoch,
    }


def finetune_ewc_from_loaders(
    base_artifact_path: str,
    old_loader: DataLoader,
    new_loader: DataLoader,
    val_loader: DataLoader,
    output_dir: Optional[str] = None,
    ewc_lambda: float = 1000.0,
    finetune_lr: float = 5e-5,
    finetune_epochs: int = 20,
    fisher_samples: int = 200,
) -> Dict[str, Any]:
    """EWC fine-tuning from pre-built DataLoaders (no CSV processing)."""

    device = get_device()
    logger.info("Device: %s", device)

    artifact = torch.load(base_artifact_path, map_location="cpu", weights_only=False)
    nr_config = NeuralRankerConfig(**artifact["config"])
    model = ChainTransformer(nr_config).to(device)
    state = {k.replace("_orig_mod.", ""): v for k, v in artifact["model_state_dict"].items()}
    model.load_state_dict(state)
    logger.info("Loaded base model: %d params", sum(p.numel() for p in model.parameters()))

    feature_columns = artifact["feature_columns"]
    edges = artifact["relevance_edges"]

    # Step 1: Fisher on old data
    fisher = compute_fisher(model, old_loader, device, n_samples=fisher_samples)
    old_params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}

    # Step 2: Evaluate base model
    base_metrics = evaluate(model, val_loader, device, k=20)
    logger.info("Base model val NDCG@20 = %.4f", base_metrics["ndcg_at_k"])

    # Step 3: Fine-tune
    optimizer = torch.optim.AdamW(model.parameters(), lr=finetune_lr, weight_decay=nr_config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=finetune_epochs, eta_min=1e-6)

    logger.info("Fine-tuning: lr=%.1e, ewc_lambda=%.0f, epochs=%d", finetune_lr, ewc_lambda, finetune_epochs)

    best_ndcg = -float("inf")
    best_epoch = 0
    best_state = None
    history = []

    accum_steps = 16
    for epoch in range(1, finetune_epochs + 1):
        t0 = time.time()
        model.train()
        total_rank_loss = 0.0
        total_ewc_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()

        for step, (features, relevance, padding_mask) in enumerate(new_loader):
            features = features.to(device)
            relevance = relevance.to(device)
            padding_mask = padding_mask.to(device)

            scores = model(features, padding_mask=padding_mask)
            rank_loss = listmle_loss(scores.float(), relevance, padding_mask=padding_mask, top_k=200)
            ewc_loss_val = ewc_penalty(model, fisher, old_params)
            loss = (rank_loss + ewc_lambda * ewc_loss_val) / accum_steps
            loss.backward()

            total_rank_loss += rank_loss.item()
            total_ewc_loss += ewc_loss_val.item()
            n_batches += 1

            if (step + 1) % accum_steps == 0 or (step + 1) == len(new_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

        scheduler.step()
        val_metrics = evaluate(model, val_loader, device, k=20)
        elapsed = time.time() - t0
        val_ndcg = val_metrics["ndcg_at_k"]

        logger.info(
            "Epoch %2d/%d | rank=%.4f ewc=%.6f | ndcg@1=%.3f @5=%.3f @10=%.3f @20=%.3f | lr=%.1e | %.1fs",
            epoch, finetune_epochs,
            total_rank_loss / max(n_batches, 1),
            total_ewc_loss / max(n_batches, 1),
            val_metrics.get("ndcg_at_1", 0), val_metrics.get("ndcg_at_5", 0),
            val_metrics.get("ndcg_at_10", 0), val_ndcg,
            optimizer.param_groups[0]["lr"], elapsed,
        )

        history.append({
            "epoch": epoch,
            "rank_loss": total_rank_loss / max(n_batches, 1),
            "ewc_loss": total_ewc_loss / max(n_batches, 1),
            "val_ndcg": val_ndcg,
        })

        if val_ndcg > best_ndcg:
            best_ndcg = val_ndcg
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            logger.info("  ** New best NDCG@20 = %.4f at epoch %d **", best_ndcg, epoch)

    # Save
    if best_state is not None:
        model.load_state_dict(best_state)

    root = Path(output_dir or "./ewc_output")
    root.mkdir(parents=True, exist_ok=True)

    ewc_artifact = {
        "model_state_dict": best_state,
        "config": asdict(nr_config),
        "feature_columns": feature_columns,
        "relevance_edges": edges,
        "train_mean": artifact["train_mean"],
        "train_std": artifact["train_std"],
        "best_epoch": best_epoch,
        "best_ndcg": best_ndcg,
        "ewc_lambda": ewc_lambda,
        "finetune_lr": finetune_lr,
        "base_ndcg": base_metrics["ndcg_at_k"],
    }
    artifact_path = root / "ewc_ranker_artifact.pt"
    torch.save(ewc_artifact, artifact_path)
    logger.info("Saved EWC artifact to %s", artifact_path)

    save_json({
        "base_ndcg": base_metrics["ndcg_at_k"],
        "best_ndcg": best_ndcg,
        "best_epoch": best_epoch,
        "improvement": best_ndcg - base_metrics["ndcg_at_k"],
        "history": history,
    }, root / "ewc_metrics.json")

    logger.info("Base NDCG: %.4f -> EWC NDCG: %.4f (improvement: %+.4f)",
                base_metrics["ndcg_at_k"], best_ndcg, best_ndcg - base_metrics["ndcg_at_k"])

    return {
        "artifact_path": str(artifact_path),
        "base_ndcg": base_metrics["ndcg_at_k"],
        "best_ndcg": best_ndcg,
        "best_epoch": best_epoch,
    }


def main():
    parser = argparse.ArgumentParser(description="EWC fine-tuning for neural ranker")
    parser.add_argument("--base-artifact", required=True, help="Pre-trained model artifact")
    parser.add_argument("--old-data", nargs="+", required=True, help="Old training data (for Fisher)")
    parser.add_argument("--new-data", nargs="+", required=True, help="New data to fine-tune on")
    parser.add_argument("--val-data", nargs="+", required=True, help="Validation data")
    parser.add_argument("--config", default="./config_tuned.yaml")
    parser.add_argument("--output-dir", default="./ewc_output")
    parser.add_argument("--ewc-lambda", type=float, default=1000.0)
    parser.add_argument("--finetune-lr", type=float, default=5e-5)
    parser.add_argument("--finetune-epochs", type=int, default=20)
    parser.add_argument("--fisher-samples", type=int, default=200)
    parser.add_argument("--nrows", type=int, default=None)
    args = parser.parse_args()

    result = finetune_with_ewc(
        base_artifact_path=args.base_artifact,
        old_data_files=args.old_data,
        new_data_files=args.new_data,
        val_data_files=args.val_data,
        config_file=args.config,
        output_dir=args.output_dir,
        ewc_lambda=args.ewc_lambda,
        finetune_lr=args.finetune_lr,
        finetune_epochs=args.finetune_epochs,
        fisher_samples=args.fisher_samples,
        nrows=args.nrows,
    )
    logger.info("EWC fine-tuning complete: %s", result)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent / "updated_option_agent_codebase"))
    main()
