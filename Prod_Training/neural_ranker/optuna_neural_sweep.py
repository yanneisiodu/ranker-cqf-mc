"""Optuna hyperparameter sweep for the neural ranker.

Runs on Cloud Run GPU. Each trial trains to early stopping (~2-5 min).
Results saved to GCS as JSON for cross-job aggregation.

Parallel jobs can run different search regions simultaneously.
"""
from __future__ import annotations

import gc
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
import torch
from torch.utils.data import DataLoader

from cloud_config import CloudConfig
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
    train_one_epoch,
)
from utils import (
    apply_relevance_bins,
    compute_relevance_bins,
    load_config,
    save_json,
    select_feature_columns,
)

logger = setup_logger(__name__)


def load_datasets(config: Dict) -> Tuple[List, List, List[str], "pd.Series", "pd.Series", np.ndarray]:
    """Load parquet data year-by-year into pre-built groups. Memory efficient."""
    cache_dir = CloudConfig.DATA_DIR + "/cache"

    # Get feature columns from first year
    first_year = CloudConfig.train_years_list()[0]
    sample = pd.read_parquet(f"{cache_dir}/year_{first_year}_prepared.parquet", columns=["date"])
    sample = pd.read_parquet(f"{cache_dir}/year_{first_year}_prepared.parquet")
    feature_columns, _, _ = select_feature_columns(sample, config)
    num_features = [c for c in feature_columns if c != "type"] + ["type_numeric"]
    del sample; gc.collect()

    # Compute stats across all training years
    logger.info("Computing feature statistics...")
    sums = None
    sq_sums = None
    total_n = 0
    all_returns = []
    for year in CloudConfig.train_years_list():
        df = pd.read_parquet(f"{cache_dir}/year_{year}_prepared.parquet")
        df["type_numeric"] = (df["type"].str.lower() == "call").astype(np.float32)
        vals = df[num_features].fillna(0).values.astype(np.float64)
        if sums is None:
            sums = vals.sum(axis=0)
            sq_sums = (vals ** 2).sum(axis=0)
        else:
            sums += vals.sum(axis=0)
            sq_sums += (vals ** 2).sum(axis=0)
        total_n += len(vals)
        all_returns.append(df["target_return"].values)
        del df, vals; gc.collect()

    train_mean = pd.Series(sums / total_n, index=num_features)
    train_std = pd.Series(np.sqrt(sq_sums / total_n - (sums / total_n) ** 2), index=num_features).replace(0, 1)
    edges = compute_relevance_bins(pd.Series(np.concatenate(all_returns)), n_bins=5)
    del all_returns, sums, sq_sums; gc.collect()
    logger.info("Stats computed over %d rows", total_n)

    # Build training groups
    logger.info("Building training dataset...")
    from simulation_engine import filter_tradeable_raw, ExecutionConfig as _EC
    exec_cfg = _EC.from_config(config)
    train_groups = []
    for year in CloudConfig.train_years_list():
        df = pd.read_parquet(f"{cache_dir}/year_{year}_prepared.parquet")
        df["type_numeric"] = (df["type"].str.lower() == "call").astype(np.float32)
        df["target_relevance"] = apply_relevance_bins(df["target_return"], edges).astype(np.float32)
        df = filter_tradeable_raw(df, exec_cfg)  # filter on RAW before normalization
        df[num_features] = (df[num_features] - train_mean) / train_std
        df[num_features] = df[num_features].fillna(0.0)
        for date in sorted(df["date"].unique()):
            day = df[df["date"] == date]
            if len(day) < 2:
                continue
            train_groups.append((
                np.nan_to_num(day[num_features].values.astype(np.float32)),
                day["target_relevance"].values.astype(np.float32),
            ))
        del df; gc.collect()
    logger.info("Train: %d days", len(train_groups))

    # Build validation groups
    logger.info("Building validation dataset...")
    val_groups = []
    for year in CloudConfig.val_years_list():
        df = pd.read_parquet(f"{cache_dir}/year_{year}_prepared.parquet")
        df["type_numeric"] = (df["type"].str.lower() == "call").astype(np.float32)
        df["target_relevance"] = apply_relevance_bins(df["target_return"], edges).astype(np.float32)
        df = filter_tradeable_raw(df, exec_cfg)  # filter on RAW before normalization
        df[num_features] = (df[num_features] - train_mean) / train_std
        df[num_features] = df[num_features].fillna(0.0)
        for date in sorted(df["date"].unique()):
            day = df[df["date"] == date]
            if len(day) < 2:
                continue
            val_groups.append((
                np.nan_to_num(day[num_features].values.astype(np.float32)),
                day["target_relevance"].values.astype(np.float32),
            ))
        del df; gc.collect()
    logger.info("Val: %d days", len(val_groups))

    return train_groups, val_groups, num_features, train_mean, train_std, edges


def neural_objective(
    trial: optuna.Trial,
    train_groups: List,
    val_groups: List,
    num_features: List[str],
    device: torch.device,
) -> float:
    """Single Optuna trial: build model, train to early stopping, return best NDCG."""

    # Suggest hyperparameters
    params = {
        "embed_dim": trial.suggest_categorical("embed_dim", [64, 128, 256]),
        "n_heads": trial.suggest_categorical("n_heads", [4, 8]),
        "n_layers": trial.suggest_int("n_layers", 2, 4),
        "dropout": trial.suggest_float("dropout", 0.05, 0.35, step=0.05),
        "mlp_hidden": trial.suggest_categorical("mlp_hidden", [128, 256, 512]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
        "warmup_epochs": trial.suggest_int("warmup_epochs", 0, 5),
    }

    config = NeuralRankerConfig(
        input_dim=len(num_features),
        embed_dim=params["embed_dim"],
        n_heads=params["n_heads"],
        n_layers=params["n_layers"],
        dropout=params["dropout"],
        mlp_hidden=params["mlp_hidden"],
        learning_rate=params["learning_rate"],
        weight_decay=params["weight_decay"],
        warmup_epochs=params["warmup_epochs"],
        listmle_top_k=200,
        epochs=30,
        patience=6,
    )

    train_ds = PrebuiltDataset(train_groups)
    val_ds = PrebuiltDataset(val_groups)

    use_cuda = device.type == "cuda"
    loader_kwargs = {
        "batch_size": 1,
        "collate_fn": collate_chains,
        "num_workers": 2 if use_cuda else 0,
        "pin_memory": use_cuda,
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    model = ChainTransformer(config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Warmup + cosine schedule
    warmup = params["warmup_epochs"]

    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / max(warmup, 1)
        progress = (epoch - warmup) / max(config.epochs - warmup, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_ndcg = -float("inf")
    best_epoch = 0
    accum_steps = 16

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, accum_steps=accum_steps, listmle_top_k=config.listmle_top_k)
        val_metrics = evaluate(model, val_loader, device, k=20, listmle_top_k=config.listmle_top_k)
        scheduler.step()

        val_ndcg = val_metrics["ndcg_at_k"]

        # Report to Optuna for pruning
        trial.report(val_ndcg, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        if val_ndcg > best_ndcg:
            best_ndcg = val_ndcg
            best_epoch = epoch

        if epoch - best_epoch >= config.patience:
            break

    # Cleanup GPU memory
    del model, optimizer, train_loader, val_loader, train_ds
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()

    logger.info("  Trial %d: NDCG=%.4f epoch=%d | %s",
                trial.number, best_ndcg, best_epoch,
                {k: round(v, 4) if isinstance(v, float) else v for k, v in params.items()})

    return best_ndcg


def run_optuna_sweep():
    config = load_config(CloudConfig.CONFIG_PATH)
    device = get_device()
    logger.info("Device: %s", device)

    # Load data once
    train_groups, val_groups, num_features, train_mean, train_std, edges = load_datasets(config)

    # Create study with job-specific seed
    n_trials = int(CloudConfig.EPOCHS)  # reuse EPOCHS env var for trial count
    seed = CloudConfig.OPTUNA_SEED
    logger.info("Optuna seed: %d", seed)
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=2),
        sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True),
    )

    logger.info("Starting Optuna sweep: %d trials", n_trials)

    study.optimize(
        lambda trial: neural_objective(trial, train_groups, val_groups, num_features, device),
        n_trials=n_trials,
        show_progress_bar=False,
    )

    # Save results
    output_dir = Path(CloudConfig.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for t in study.trials:
        entry = {
            "number": t.number,
            "value": t.value,
            "state": t.state.name,
            "params": t.params,
        }
        results.append(entry)

    results.sort(key=lambda r: r.get("value") or 0, reverse=True)

    save_json(results, output_dir / "optuna_sweep_results.json")

    # Log top 5
    logger.info("")
    logger.info("=== TOP 5 RESULTS ===")
    for r in results[:5]:
        if r["value"] is not None:
            logger.info("  NDCG=%.4f | %s", r["value"], r["params"])

    logger.info("")
    logger.info("Best: NDCG=%.4f | %s", study.best_value, study.best_params)

    # Save best config
    best = study.best_params
    save_json({"best_ndcg": study.best_value, "best_params": best}, output_dir / "optuna_best_params.json")


if __name__ == "__main__":
    run_optuna_sweep()
