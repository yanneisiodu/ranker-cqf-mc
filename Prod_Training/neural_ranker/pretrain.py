"""Self-supervised pretraining for the option chain transformer.

Masks 15% of options in each day's chain and trains the model to
reconstruct the missing features from the remaining options.
Teaches the model option chain structure before fine-tuning on ranking.

Usage:
    # Local
    python pretrain.py --data year_2018_data.csv ... year_2025_data.csv \
        --config config_tuned.yaml --output-dir ./pretrain_output

    # Cloud Run (via entrypoint.sh MODE=pretrain)
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
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from logger import setup_logger
from neural_ranker import ChainTransformer, NeuralRankerConfig, get_device
from train_neural_ranker import PrebuiltDataset, collate_chains
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


class MaskedChainDataset(Dataset):
    """Dataset that randomly masks 15% of options per chain for reconstruction."""

    def __init__(self, groups: List[Tuple[np.ndarray, np.ndarray]], mask_ratio: float = 0.15, seed: int = 42):
        self.groups = groups
        self.mask_ratio = mask_ratio
        self.rng = np.random.RandomState(seed)

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features, _ = self.groups[idx]  # ignore relevance for pretraining
        n_options = features.shape[0]

        # Randomly select options to mask
        n_mask = max(1, int(n_options * self.mask_ratio))
        mask_indices = self.rng.choice(n_options, size=n_mask, replace=False)

        # Create masked input — zero out masked options
        masked_features = features.copy()
        masked_features[mask_indices] = 0.0

        # Create mask tensor (True = masked)
        mask = np.zeros(n_options, dtype=bool)
        mask[mask_indices] = True

        return (
            torch.from_numpy(masked_features),
            torch.from_numpy(features),  # reconstruction target
            torch.from_numpy(mask),
        )


def collate_masked_chains(batch):
    """Pad variable-length masked chains."""
    masked_list, target_list, mask_list = zip(*batch)
    max_len = max(m.shape[0] for m in masked_list)
    input_dim = masked_list[0].shape[1]

    padded_masked = torch.zeros(len(batch), max_len, input_dim)
    padded_target = torch.zeros(len(batch), max_len, input_dim)
    reconstruction_mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    padding_mask = torch.ones(len(batch), max_len, dtype=torch.bool)

    for i, (m, t, rm) in enumerate(zip(masked_list, target_list, mask_list)):
        n = m.shape[0]
        padded_masked[i, :n] = m
        padded_target[i, :n] = t
        reconstruction_mask[i, :n] = rm
        padding_mask[i, :n] = False

    return padded_masked, padded_target, reconstruction_mask, padding_mask


class ReconstructionHead(nn.Module):
    """Predicts masked option features from transformer embeddings."""

    def __init__(self, embed_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def pretrain_one_epoch(
    model: ChainTransformer,
    recon_head: ReconstructionHead,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    recon_head.train()
    total_loss = 0.0
    n_batches = 0

    for masked_features, target_features, recon_mask, padding_mask in loader:
        masked_features = masked_features.to(device)
        target_features = target_features.to(device)
        recon_mask = recon_mask.to(device)
        padding_mask = padding_mask.to(device)

        # Forward through transformer encoder + reconstruction head
        # No AMP — reconstruction loss requires float32 throughout
        embeddings = model.encoder(masked_features)
        embeddings = model.transformer(embeddings, src_key_padding_mask=padding_mask)
        predictions = recon_head(embeddings)

        # MSE loss on masked positions only
        mask_3d = recon_mask.unsqueeze(-1).expand_as(predictions)
        masked_preds = predictions[mask_3d]
        masked_targets = target_features[mask_3d]

        loss = nn.functional.mse_loss(masked_preds, masked_targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(recon_head.parameters()), max_norm=1.0
        )
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_reconstruction(
    model: ChainTransformer,
    recon_head: ReconstructionHead,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    recon_head.eval()
    total_loss = 0.0
    n_batches = 0

    for masked_features, target_features, recon_mask, padding_mask in loader:
        masked_features = masked_features.to(device)
        target_features = target_features.to(device)
        recon_mask = recon_mask.to(device)
        padding_mask = padding_mask.to(device)

        embeddings = model.encoder(masked_features)
        embeddings = model.transformer(embeddings, src_key_padding_mask=padding_mask)
        predictions = recon_head(embeddings)

        mask_3d = recon_mask.unsqueeze(-1).expand_as(predictions)
        masked_preds = predictions[mask_3d]
        masked_targets = target_features[mask_3d]

        loss = nn.functional.mse_loss(masked_preds, masked_targets)
        total_loss += loss.item()
        n_batches += 1

    return {"recon_loss": total_loss / max(n_batches, 1)}


def pretrain_from_groups(
    train_groups: List[Tuple[np.ndarray, np.ndarray]],
    val_groups: List[Tuple[np.ndarray, np.ndarray]],
    num_features: List[str],
    config: Dict[str, Any],
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Pretrain the transformer encoder on masked option reconstruction."""
    nr_config = NeuralRankerConfig.from_config(config)
    device = get_device()
    logger.info("Using device: %s", device)

    torch.manual_seed(nr_config.seed)
    np.random.seed(nr_config.seed)

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
        epochs=nr_config.epochs,
        patience=nr_config.patience,
        seed=nr_config.seed,
    )

    logger.info("Config: input_dim=%d, embed=%d, heads=%d, layers=%d",
                actual_config.input_dim, actual_config.embed_dim,
                actual_config.n_heads, actual_config.n_layers)
    logger.info("Train: %d days, Val: %d days", len(train_groups), len(val_groups))

    # Datasets with masking
    train_ds = MaskedChainDataset(train_groups, mask_ratio=0.15)
    val_ds = MaskedChainDataset(val_groups, mask_ratio=0.15, seed=99)

    use_cuda = device.type == "cuda"
    loader_kwargs = {
        "batch_size": 1,
        "num_workers": 4 if use_cuda else 2,
        "pin_memory": use_cuda,
        "persistent_workers": True,
    }
    train_loader = DataLoader(train_ds, shuffle=True, collate_fn=collate_masked_chains, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, collate_fn=collate_masked_chains, **loader_kwargs)

    # Model + reconstruction head
    model = ChainTransformer(actual_config).to(device)
    recon_head = ReconstructionHead(actual_config.embed_dim, actual_config.input_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in recon_head.parameters())
    logger.info("Total parameters (encoder + recon head): %s", f"{n_params:,}")

    # Note: torch.compile is NOT used for pretraining because we access
    # model.encoder and model.transformer directly (not through model.forward).
    # torch.compile wraps the model and breaks direct submodule access.

    all_params = list(model.parameters()) + list(recon_head.parameters())
    optimizer = torch.optim.AdamW(
        all_params,
        lr=actual_config.learning_rate,
        weight_decay=actual_config.weight_decay,
    )

    warmup = actual_config.warmup_epochs
    pretrain_epochs = actual_config.epochs

    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / max(warmup, 1)
        progress = (epoch - warmup) / max(pretrain_epochs - warmup, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    history = []

    for epoch in range(1, pretrain_epochs + 1):
        t0 = time.time()
        train_loss = pretrain_one_epoch(model, recon_head, train_loader, optimizer, device)
        val_metrics = evaluate_reconstruction(model, recon_head, val_loader, device)
        scheduler.step()
        elapsed = time.time() - t0

        val_loss = val_metrics["recon_loss"]
        lr = optimizer.param_groups[0]["lr"]
        logger.info(
            "Pretrain %3d/%d | train_loss=%.4f | val_loss=%.4f | lr=%.2e | %.1fs",
            epoch, pretrain_epochs, train_loss, val_loss, lr, elapsed,
        )

        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_loss": val_loss, "lr": lr, "elapsed_s": elapsed,
        })

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            # Save only the transformer encoder weights (not recon head)
            best_state = {}
            for k, v in model.state_dict().items():
                best_state[k] = v.cpu().clone()
            logger.info("  ** New best val_loss = %.4f at epoch %d **", best_loss, epoch)

        if epoch - best_epoch >= actual_config.patience:
            logger.info("Early stopping at epoch %d (patience=%d)", epoch, actual_config.patience)
            break

    # Save pretrained encoder weights
    root = Path(output_dir or "./pretrain_output")
    root.mkdir(parents=True, exist_ok=True)

    artifact = {
        "encoder_state_dict": best_state,
        "config": asdict(actual_config),
        "feature_columns": num_features,
        "best_epoch": best_epoch,
        "best_loss": best_loss,
    }
    artifact_path = root / "pretrained_encoder.pt"
    torch.save(artifact, artifact_path)
    logger.info("Saved pretrained encoder to %s", artifact_path)

    save_json({
        "best_epoch": best_epoch, "best_loss": best_loss,
        "n_params": n_params, "device": str(device), "history": history,
    }, root / "pretrain_metrics.json")

    return {
        "artifact_path": str(artifact_path),
        "best_loss": best_loss,
        "best_epoch": best_epoch,
    }


def pretrain_from_files(
    data_files: Sequence[str],
    config_file: Optional[str] = None,
    output_dir: Optional[str] = None,
    nrows: Optional[int] = None,
) -> Dict[str, Any]:
    """File-based entry point for local execution."""
    config = load_config(config_file)
    logger.info("Loading data...")
    frame = prepare_model_frame(data_files, config, include_targets=True, nrows=nrows)
    logger.info("Loaded: %s", summarize_frame(frame))

    feature_columns, _, _ = select_feature_columns(frame, config)
    num_features = [c for c in feature_columns if c != "type"] + ["type_numeric"]
    frame["type_numeric"] = (frame["type"].str.lower() == "call").astype(np.float32)

    # Compute relevance (needed for group format compatibility)
    edges = compute_relevance_bins(frame["target_return"], n_bins=5)
    frame["target_relevance"] = apply_relevance_bins(frame["target_return"], edges).astype(np.float32)

    # Normalize
    train_mean = frame[num_features].mean()
    train_std = frame[num_features].std().replace(0, 1)
    frame[num_features] = (frame[num_features] - train_mean) / train_std
    frame[num_features] = frame[num_features].fillna(0.0)

    # Liquidity filter
    if "relative_spread" in frame.columns:
        frame = frame[frame["relative_spread"] <= 0.50].reset_index(drop=True)

    # Build groups
    groups = []
    for date in sorted(frame["date"].unique()):
        day = frame[frame["date"] == date]
        if len(day) < 2:
            continue
        feats = np.nan_to_num(day[num_features].values.astype(np.float32))
        rels = day["target_relevance"].values.astype(np.float32)
        groups.append((feats, rels))

    # Split 90/10 for pretrain train/val
    split = int(len(groups) * 0.9)
    train_groups = groups[:split]
    val_groups = groups[split:]

    return pretrain_from_groups(train_groups, val_groups, num_features, config, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain option chain transformer")
    parser.add_argument("--data", nargs="+", required=True, help="CSV files")
    parser.add_argument("--config", default="./config_tuned.yaml")
    parser.add_argument("--output-dir", default="./pretrain_output")
    parser.add_argument("--nrows", type=int, default=None)
    args = parser.parse_args()

    result = pretrain_from_files(
        args.data, config_file=args.config,
        output_dir=args.output_dir, nrows=args.nrows,
    )
    logger.info("Pretraining complete: %s", result)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent / "updated_option_agent_codebase"))
    main()
