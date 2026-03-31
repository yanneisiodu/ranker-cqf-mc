"""Training script for the listwise neural option ranker.

Two-stage approach:
    1. XGBoost pre-ranks the full chain (~9K options/day)
    2. Transformer re-ranks the top-K candidates (~200/day)

Walk-forward training with purged splits. MPS-accelerated on Apple Silicon.

Usage:
    python train_neural_ranker.py \
        --train-data year_2019_data.csv year_2020_data.csv ... year_2023_data.csv \
        --val-data year_2024_data.csv \
        --config config_tuned.yaml \
        --output-dir ./neural_output
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from logger import setup_logger
from neural_ranker import (
    ChainTransformer,
    NeuralRankerConfig,
    get_device,
    listmle_loss,
    ndcg_at_k,
)
from utils import (
    build_preprocessor,
    compute_relevance_bins,
    apply_relevance_bins,
    load_config,
    prepare_model_frame,
    save_json,
    select_feature_columns,
    summarize_frame,
)

logger = setup_logger(__name__)


# ── dataset ─────────────────────────────────────────────────────────────────

class DailyChainDataset(Dataset):
    """Each sample is one day's full option chain after liquidity filtering.

    Filters by spread <= max_spread to remove illiquid junk while retaining
    99.8% of actual top-20 winners. No XGBoost pre-filtering needed.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        feature_columns: List[str],
    ):
        """Assumes raw liquidity filtering has already been applied upstream."""
        self.feature_columns = feature_columns
        self.dates = sorted(frame["date"].unique())
        logger.info("DailyChainDataset: %d rows, %d dates (pre-filtered)", len(frame), len(self.dates))

        # Group by date
        self.groups: List[Tuple[np.ndarray, np.ndarray]] = []
        for date in self.dates:
            mask = frame["date"] == date
            day_df = frame[mask]
            if len(day_df) < 2:
                continue
            features = day_df[feature_columns].to_numpy(dtype=np.float32)
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            relevance = day_df["target_relevance"].to_numpy(dtype=np.float32)
            self.groups.append((features, relevance))

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        features, relevance = self.groups[idx]
        return torch.from_numpy(features), torch.from_numpy(relevance)


def collate_chains(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    """Pad variable-length chains to the longest in the batch."""
    features_list, relevance_list = zip(*batch)
    max_len = max(f.shape[0] for f in features_list)
    input_dim = features_list[0].shape[1]

    padded_features = torch.zeros(len(batch), max_len, input_dim)
    padded_relevance = torch.zeros(len(batch), max_len)
    padding_mask = torch.ones(len(batch), max_len, dtype=torch.bool)

    for i, (f, r) in enumerate(zip(features_list, relevance_list)):
        n = f.shape[0]
        padded_features[i, :n] = f
        padded_relevance[i, :n] = r
        padding_mask[i, :n] = False

    return padded_features, padded_relevance, padding_mask


# ── training loop ───────────────────────────────────────────────────────────

def train_one_epoch(
    model: ChainTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    accum_steps: int = 16,
    recon_head: Optional[nn.Module] = None,
    recon_weight: float = 0.1,
    mask_ratio: float = 0.15,
) -> Dict[str, float]:
    model.train()
    if recon_head is not None:
        recon_head.train()
    total_rank_loss = 0.0
    total_recon_loss = 0.0
    n_batches = 0
    optimizer.zero_grad()

    for step, (features, relevance, padding_mask) in enumerate(loader):
        features = features.to(device)
        relevance = relevance.to(device)
        padding_mask = padding_mask.to(device)

        # Ranking loss
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            scores = model(features, padding_mask=padding_mask)
        rank_loss = listmle_loss(scores.float(), relevance, padding_mask=padding_mask, top_k=200)

        # Multi-task: reconstruction loss as regularizer
        if recon_head is not None:
            # Mask random options and reconstruct
            B, S, D = features.shape
            n_mask = max(1, int(S * mask_ratio))
            mask_idx = torch.randint(0, S, (B, n_mask), device=device)
            masked_input = features.clone()
            for b in range(B):
                masked_input[b, mask_idx[b]] = 0.0

            embeddings = model.encoder(masked_input)
            embeddings = model.transformer(embeddings, src_key_padding_mask=padding_mask)
            predictions = recon_head(embeddings)

            # MSE on masked positions only
            recon_loss = 0.0
            for b in range(B):
                idx = mask_idx[b]
                recon_loss = recon_loss + nn.functional.mse_loss(
                    predictions[b, idx], features[b, idx]
                )
            recon_loss = recon_loss / B
            loss = (rank_loss + recon_weight * recon_loss) / accum_steps
            total_recon_loss += recon_loss.item()
        else:
            loss = rank_loss / accum_steps

        loss.backward()
        total_rank_loss += rank_loss.item()
        n_batches += 1

        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            all_params = list(model.parameters())
            if recon_head is not None:
                all_params += list(recon_head.parameters())
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

    return {
        "rank_loss": total_rank_loss / max(n_batches, 1),
        "recon_loss": total_recon_loss / max(n_batches, 1) if recon_head else 0.0,
    }


@torch.no_grad()
def evaluate(
    model: ChainTransformer,
    loader: DataLoader,
    device: torch.device,
    k: int = 20,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    ndcg_lists: Dict[int, List[float]] = {1: [], 5: [], 10: [], 20: []}

    for features, relevance, padding_mask in loader:
        features = features.to(device)
        relevance = relevance.to(device)
        padding_mask = padding_mask.to(device)

        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            scores = model(features, padding_mask=padding_mask)
        loss = listmle_loss(scores.float(), relevance, padding_mask=padding_mask, top_k=200)
        total_loss += loss.item()
        n_batches += 1

        scores_np = scores.cpu().numpy()
        relevance_np = relevance.cpu().numpy()
        mask_np = padding_mask.cpu().numpy()

        for i in range(scores_np.shape[0]):
            valid = ~mask_np[i]
            if valid.sum() < 2:
                continue
            s = scores_np[i][valid]
            r = relevance_np[i][valid]
            for kk in ndcg_lists:
                val = ndcg_at_k(s, r, k=kk)
                if not np.isnan(val):
                    ndcg_lists[kk].append(val)

    result = {"loss": total_loss / max(n_batches, 1)}
    for kk, vals in ndcg_lists.items():
        result[f"ndcg_at_{kk}"] = float(np.mean(vals)) if vals else float("nan")
    result["ndcg_at_k"] = result["ndcg_at_20"]  # backward compat
    result["ndcg_std"] = float(np.std(ndcg_lists[20])) if ndcg_lists[20] else float("nan")
    result["n_days"] = len(ndcg_lists[20])
    return result


def train_neural_ranker_from_frames(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    config: Dict[str, Any],
    output_dir: Optional[str] = None,
    nrows: Optional[int] = None,
) -> Dict[str, Any]:
    """Core training logic — accepts pre-loaded DataFrames."""
    nr_config = NeuralRankerConfig.from_config(config)
    device = get_device()
    logger.info("Using device: %s", device)

    torch.manual_seed(nr_config.seed)
    np.random.seed(nr_config.seed)

    if nrows:
        train_frame = train_frame.head(nrows)
        val_frame = val_frame.head(nrows)

    logger.info("Train: %s", summarize_frame(train_frame))
    logger.info("Val: %s", summarize_frame(val_frame))

    # Feature columns
    feature_columns, numerical_features, categorical_features = select_feature_columns(train_frame, config)
    # For neural net, use only numerical features (no categorical encoding needed — type is encoded numerically)
    num_features = [c for c in feature_columns if c != "type"]

    # Add type as numeric (call=1, put=0)
    for frame in [train_frame, val_frame]:
        frame["type_numeric"] = (frame["type"].str.lower() == "call").astype(np.float32)
    num_features = num_features + ["type_numeric"]

    # Compute relevance bins from training data
    edges = compute_relevance_bins(train_frame["target_return"], n_bins=5)
    train_frame["target_relevance"] = apply_relevance_bins(train_frame["target_return"], edges).astype(np.float32)
    val_frame["target_relevance"] = apply_relevance_bins(val_frame["target_return"], edges).astype(np.float32)

    # Normalize features using training stats
    train_mean = train_frame[num_features].mean()
    train_std = train_frame[num_features].std().replace(0, 1)
    for frame in [train_frame, val_frame]:
        frame[num_features] = (frame[num_features] - train_mean) / train_std
        frame[num_features] = frame[num_features].fillna(0.0)

    # Update config with actual input dim
    actual_config = NeuralRankerConfig(
        input_dim=len(num_features),
        embed_dim=nr_config.embed_dim,
        n_heads=nr_config.n_heads,
        n_layers=nr_config.n_layers,
        dropout=nr_config.dropout,
        mlp_hidden=nr_config.mlp_hidden,
        top_k_candidates=nr_config.top_k_candidates,
        learning_rate=nr_config.learning_rate,
        weight_decay=nr_config.weight_decay,
        batch_dates=nr_config.batch_dates,
        epochs=nr_config.epochs,
        patience=nr_config.patience,
        seed=nr_config.seed,
    )
    logger.info("Config: input_dim=%d, embed=%d, heads=%d, layers=%d",
                actual_config.input_dim, actual_config.embed_dim,
                actual_config.n_heads, actual_config.n_layers)

    # Build datasets — full chain with liquidity filter only
    logger.info("Building datasets (full chain, spread<=50%% filter)...")
    # Raw liquidity filter BEFORE normalization
    from simulation_engine import filter_tradeable_raw, ExecutionConfig
    exec_cfg = ExecutionConfig.from_config(config)
    train_frame = filter_tradeable_raw(train_frame, exec_cfg)
    val_frame = filter_tradeable_raw(val_frame, exec_cfg)
    logger.info("After raw liquidity filter: train=%d, val=%d", len(train_frame), len(val_frame))

    train_ds = DailyChainDataset(train_frame, num_features)
    val_ds = DailyChainDataset(val_frame, num_features)
    logger.info("Train: %d days, Val: %d days", len(train_ds), len(val_ds))

    # batch_size=1 (one day's chain per forward pass) since chains are ~7K options
    # Gradient accumulation simulates larger effective batch size
    accum_steps = actual_config.batch_dates  # accumulate over this many days
    use_cuda = device.type == "cuda"
    loader_kwargs = {
        "batch_size": 1,
        "collate_fn": collate_chains,
        "num_workers": 4 if use_cuda else 2,
        "pin_memory": use_cuda,
        "persistent_workers": True,
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    # Build model
    model = ChainTransformer(actual_config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %s", f"{n_params:,}")

    # Compile model for CUDA (1.3-1.5x speedup via kernel fusion)
    if device.type == "cuda":
        import os
        use_compile = os.getenv("TORCH_COMPILE", "true").lower() in ("true", "1", "yes")
        if use_compile:
            try:
                model = torch.compile(model)
                logger.info("Model compiled with torch.compile (CUDA)")
            except Exception as e:
                logger.warning("torch.compile failed, using eager mode: %s", e)
        else:
            logger.info("torch.compile disabled via TORCH_COMPILE env var")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=actual_config.learning_rate,
        weight_decay=actual_config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=actual_config.epochs,
        eta_min=1e-6,
    )

    # Training loop with early stopping
    best_ndcg = -float("inf")
    best_epoch = 0
    best_state = None
    history: List[Dict[str, Any]] = []

    for epoch in range(1, actual_config.epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, accum_steps=accum_steps)
        train_loss = train_metrics["rank_loss"]
        val_metrics = evaluate(model, val_loader, device, k=20)
        scheduler.step()
        elapsed = time.time() - t0

        val_ndcg = val_metrics["ndcg_at_k"]
        lr = optimizer.param_groups[0]["lr"]

        logger.info(
            "Epoch %3d/%d | loss=%.4f/%.4f | ndcg@1=%.3f @5=%.3f @10=%.3f @20=%.3f | lr=%.2e | %.1fs",
            epoch, actual_config.epochs, train_loss, val_metrics["loss"],
            val_metrics.get("ndcg_at_1", 0), val_metrics.get("ndcg_at_5", 0),
            val_metrics.get("ndcg_at_10", 0), val_ndcg, lr, elapsed,
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_ndcg_at_k": val_ndcg,
            "val_ndcg_std": val_metrics["ndcg_std"],
            "val_n_days": val_metrics["n_days"],
            "lr": lr,
            "elapsed_s": elapsed,
        })

        if val_ndcg > best_ndcg:
            best_ndcg = val_ndcg
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            logger.info("  ** New best NDCG@20 = %.4f at epoch %d **", best_ndcg, epoch)

        if epoch - best_epoch >= actual_config.patience:
            logger.info("Early stopping at epoch %d (patience=%d)", epoch, actual_config.patience)
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation
    final_metrics = evaluate(model, val_loader, device, k=20)
    logger.info("Final val NDCG@20 = %.4f (from epoch %d)", final_metrics["ndcg_at_k"], best_epoch)

    # Save artifacts
    root = Path(output_dir or "./neural_output")
    root.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model_state_dict": best_state,
        "config": asdict(actual_config),
        "feature_columns": num_features,
        "relevance_edges": edges,
        "train_mean": train_mean.to_dict(),
        "train_std": train_std.to_dict(),
        "best_epoch": best_epoch,
        "best_ndcg": best_ndcg,
    }
    artifact_path = root / "neural_ranker_artifact.pt"
    torch.save(artifact, artifact_path)
    logger.info("Saved artifact to %s", artifact_path)

    metrics = {
        "best_epoch": best_epoch,
        "best_ndcg_at_k": best_ndcg,
        "final_val": final_metrics,
        "n_params": n_params,
        "device": str(device),
        "history": history,
    }
    save_json(metrics, root / "neural_ranker_metrics.json")

    return {
        "artifact_path": str(artifact_path),
        "best_ndcg": best_ndcg,
        "best_epoch": best_epoch,
        "n_params": n_params,
    }


class PrebuiltDataset(Dataset):
    """Dataset from pre-built (features, relevance) numpy arrays."""

    def __init__(self, groups: List[Tuple[np.ndarray, np.ndarray]]):
        self.groups = groups

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        features, relevance = self.groups[idx]
        return torch.from_numpy(features), torch.from_numpy(relevance)


def train_neural_ranker_from_datasets(
    train_groups: List[Tuple[np.ndarray, np.ndarray]],
    val_groups: List[Tuple[np.ndarray, np.ndarray]],
    num_features: List[str],
    train_mean: "pd.Series",
    train_std: "pd.Series",
    relevance_edges: np.ndarray,
    config: Dict[str, Any],
    output_dir: Optional[str] = None,
    pretrained_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Memory-efficient entry point — accepts pre-built daily groups.

    If pretrained_path is provided, loads pretrained encoder weights
    before fine-tuning on ranking.
    """
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
        top_k_candidates=nr_config.top_k_candidates,
        learning_rate=nr_config.learning_rate,
        weight_decay=nr_config.weight_decay,
        batch_dates=nr_config.batch_dates,
        epochs=nr_config.epochs,
        patience=nr_config.patience,
        seed=nr_config.seed,
    )
    logger.info("Config: input_dim=%d, embed=%d, heads=%d, layers=%d",
                actual_config.input_dim, actual_config.embed_dim,
                actual_config.n_heads, actual_config.n_layers)
    logger.info("Train: %d days, Val: %d days", len(train_groups), len(val_groups))

    train_ds = PrebuiltDataset(train_groups)
    val_ds = PrebuiltDataset(val_groups)

    use_cuda = device.type == "cuda"
    accum_steps = actual_config.batch_dates
    loader_kwargs = {
        "batch_size": 1,
        "collate_fn": collate_chains,
        "num_workers": 4 if use_cuda else 2,
        "pin_memory": use_cuda,
        "persistent_workers": True,
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    model = ChainTransformer(actual_config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %s", f"{n_params:,}")

    # Load pretrained encoder weights if available
    if pretrained_path:
        pretrained = torch.load(pretrained_path, map_location="cpu", weights_only=False)
        encoder_state = pretrained["encoder_state_dict"]
        missing, unexpected = model.load_state_dict(encoder_state, strict=False)
        logger.info("Loaded pretrained encoder from %s", pretrained_path)
        logger.info("  Missing keys (expected — ranking head): %s", missing)
        if unexpected:
            logger.warning("  Unexpected keys: %s", unexpected)

    if device.type == "cuda":
        import os
        use_compile = os.getenv("TORCH_COMPILE", "true").lower() in ("true", "1", "yes")
        if use_compile:
            try:
                model = torch.compile(model)
                logger.info("Model compiled with torch.compile (CUDA)")
            except Exception as e:
                logger.warning("torch.compile failed, using eager mode: %s", e)
        else:
            logger.info("torch.compile disabled via TORCH_COMPILE env var")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=actual_config.learning_rate,
        weight_decay=actual_config.weight_decay,
    )

    # Warmup + cosine schedule
    warmup = actual_config.warmup_epochs
    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / max(warmup, 1)
        progress = (epoch - warmup) / max(actual_config.epochs - warmup, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_ndcg = -float("inf")
    best_epoch = 0
    best_state = None
    history = []

    for epoch in range(1, actual_config.epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, accum_steps=accum_steps)
        train_loss = train_metrics["rank_loss"]
        val_metrics = evaluate(model, val_loader, device, k=20)
        scheduler.step()
        elapsed = time.time() - t0

        val_ndcg = val_metrics["ndcg_at_k"]
        lr = optimizer.param_groups[0]["lr"]
        logger.info(
            "Epoch %3d/%d | loss=%.4f/%.4f | ndcg@1=%.3f @5=%.3f @10=%.3f @20=%.3f | lr=%.2e | %.1fs",
            epoch, actual_config.epochs, train_loss, val_metrics["loss"],
            val_metrics.get("ndcg_at_1", 0), val_metrics.get("ndcg_at_5", 0),
            val_metrics.get("ndcg_at_10", 0), val_ndcg, lr, elapsed,
        )
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_metrics["loss"], "val_ndcg_at_k": val_ndcg,
                        "lr": lr, "elapsed_s": elapsed})

        if val_ndcg > best_ndcg:
            best_ndcg = val_ndcg
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            logger.info("  ** New best NDCG@20 = %.4f at epoch %d **", best_ndcg, epoch)

        if epoch - best_epoch >= actual_config.patience:
            logger.info("Early stopping at epoch %d (patience=%d)", epoch, actual_config.patience)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    root = Path(output_dir or "./neural_output")
    root.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model_state_dict": best_state,
        "config": asdict(actual_config),
        "feature_columns": num_features,
        "relevance_edges": relevance_edges,
        "train_mean": train_mean.to_dict(),
        "train_std": train_std.to_dict(),
        "best_epoch": best_epoch,
        "best_ndcg": best_ndcg,
    }
    artifact_path = root / "neural_ranker_artifact.pt"
    torch.save(artifact, artifact_path)
    logger.info("Saved artifact to %s", artifact_path)

    save_json({"best_epoch": best_epoch, "best_ndcg_at_k": best_ndcg,
               "n_params": n_params, "device": str(device), "history": history},
              root / "neural_ranker_metrics.json")

    return {"artifact_path": str(artifact_path), "best_ndcg": best_ndcg,
            "best_epoch": best_epoch, "n_params": n_params}


def train_neural_ranker(
    train_files: Sequence[str],
    val_files: Sequence[str],
    config_file: Optional[str] = None,
    output_dir: Optional[str] = None,
    nrows: Optional[int] = None,
) -> Dict[str, Any]:
    """File-based entry point — loads CSVs, processes features, then trains."""
    config = load_config(config_file)
    logger.info("Loading training data...")
    train_frame = prepare_model_frame(train_files, config, include_targets=True, nrows=nrows)
    logger.info("Loading validation data...")
    val_frame = prepare_model_frame(val_files, config, include_targets=True, nrows=nrows)
    return train_neural_ranker_from_frames(train_frame, val_frame, config, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a listwise neural option ranker")
    parser.add_argument("--train-data", nargs="+", required=True, help="Training CSV files")
    parser.add_argument("--val-data", nargs="+", required=True, help="Validation CSV files")
    parser.add_argument("--config", default="./config_tuned.yaml", help="Path to YAML config")
    parser.add_argument("--output-dir", default="./neural_output", help="Output directory")
    parser.add_argument("--nrows", type=int, default=None, help="Row cap for debugging")
    args = parser.parse_args()

    result = train_neural_ranker(
        args.train_data,
        args.val_data,
        config_file=args.config,
        output_dir=args.output_dir,
        nrows=args.nrows,
    )
    logger.info("Training complete: %s", result)


if __name__ == "__main__":
    # Also add the updated_option_agent_codebase to path for prod_train_ranker imports
    sys.path.insert(0, str(Path(__file__).parent.parent / "updated_option_agent_codebase"))
    main()
