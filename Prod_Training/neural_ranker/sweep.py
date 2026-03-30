"""Hyperparameter sweep for neural ranker on Cloud Run GPU.

Runs a grid search over dropout, weight_decay, and learning_rate.
Each combo trains to early stopping (~15 epochs × 8s = ~2 min).
Results saved to GCS as a JSON summary.
"""
from __future__ import annotations

import itertools
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

from cloud_config import CloudConfig
from logger import setup_logger
from neural_ranker import ChainTransformer, NeuralRankerConfig
from train_neural_ranker import train_neural_ranker_from_frames
from utils import load_config, prepare_model_frame, summarize_frame

logger = setup_logger(__name__)

# Sweep grid
GRID = {
    "dropout": [0.05, 0.10, 0.15, 0.20],
    "weight_decay": [1e-4, 5e-4, 1e-3],
    "learning_rate": [3e-4, 1e-3],
}


def run_sweep():
    config = load_config(CloudConfig.CONFIG_PATH)
    cache_dir = CloudConfig.DATA_DIR + "/cache"

    # Load pre-processed parquet
    logger.info("Loading parquet data...")
    train_frames = []
    for year in CloudConfig.train_years_list():
        path = f"{cache_dir}/year_{year}_prepared.parquet"
        train_frames.append(pd.read_parquet(path))
    train_frame = pd.concat(train_frames, ignore_index=True)

    val_frames = []
    for year in CloudConfig.val_years_list():
        path = f"{cache_dir}/year_{year}_prepared.parquet"
        val_frames.append(pd.read_parquet(path))
    val_frame = pd.concat(val_frames, ignore_index=True)

    logger.info("Train: %d rows, Val: %d rows", len(train_frame), len(val_frame))

    # Generate all combinations
    keys = list(GRID.keys())
    combos = list(itertools.product(*GRID.values()))
    logger.info("Running %d hyperparameter combinations", len(combos))

    results: List[Dict[str, Any]] = []

    for i, values in enumerate(combos, 1):
        params = dict(zip(keys, values))
        logger.info("=== Trial %d/%d: %s ===", i, len(combos), params)

        # Override config with sweep params
        sweep_config = config.copy()
        sweep_config["neural_ranker"] = {
            **sweep_config.get("neural_ranker", {}),
            **params,
        }

        t0 = time.time()
        try:
            result = train_neural_ranker_from_frames(
                train_frame=train_frame,
                val_frame=val_frame,
                config=sweep_config,
                output_dir=f"{CloudConfig.OUTPUT_DIR}/sweep_{i:03d}",
            )
            # Free GPU memory between trials
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            elapsed = time.time() - t0
            entry = {
                "trial": i,
                **params,
                "best_ndcg": result["best_ndcg"],
                "best_epoch": result["best_epoch"],
                "elapsed_s": round(elapsed, 1),
            }
            logger.info("  Result: NDCG=%.4f epoch=%d (%.0fs)", result["best_ndcg"], result["best_epoch"], elapsed)
        except Exception as e:
            elapsed = time.time() - t0
            entry = {
                "trial": i,
                **params,
                "best_ndcg": float("nan"),
                "best_epoch": 0,
                "elapsed_s": round(elapsed, 1),
                "error": str(e),
            }
            logger.error("  Failed: %s", e)

        results.append(entry)

    # Sort by NDCG
    results.sort(key=lambda r: r.get("best_ndcg", 0), reverse=True)

    # Save results
    output_path = Path(CloudConfig.OUTPUT_DIR) / "sweep_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print top 5
    logger.info("")
    logger.info("=== TOP 5 RESULTS ===")
    for r in results[:5]:
        logger.info(
            "  NDCG=%.4f | dropout=%.2f lr=%.1e wd=%.1e | epoch=%d (%.0fs)",
            r["best_ndcg"], r["dropout"], r["learning_rate"], r["weight_decay"],
            r["best_epoch"], r["elapsed_s"],
        )

    logger.info("")
    logger.info("Best: %s", results[0])
    logger.info("Sweep results saved to %s", output_path)


if __name__ == "__main__":
    run_sweep()
