"""Optuna hyperparameter tuner for the options trading pipeline.

Two-phase, multi-fidelity tuning with parallel trials and early pruning.

Usage:
    # Phase 1: coarse search on 25% of dates (fast)
    python optuna_tuner.py --model ranker --phase 1 \
        --data year_2023_data.csv year_2024_data.csv

    # Phase 2: refine on full data around Phase 1 winners
    python optuna_tuner.py --model ranker --phase 2 \
        --data year_2023_data.csv year_2024_data.csv

    # Same for meta and return models
"""
from __future__ import annotations

import argparse
import copy
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import warnings

import numpy as np
import optuna
import pandas as pd

warnings.filterwarnings("ignore", message="The reported value is ignored")
import xgboost as xgb
import yaml
from optuna.storages.journal import JournalFileBackend, JournalStorage
from optuna_integration.xgboost import XGBoostPruningCallback
from sklearn.metrics import brier_score_loss, mean_absolute_error

from logger import setup_logger
from prod_train_ranker import (
    RankerConfig,
    _daily_ndcg,
    _ranker_model,
    generate_oof_ranker_features,
    fit_ranker_artifact,
    predict_ranker_scores,
)
from prod_meta_labeler import MetaLabelerConfig
from prod_log_return_predictor import LogReturnConfig
from utils import (
    PurgedWalkForwardSplit,
    apply_relevance_bins,
    build_preprocessor,
    compute_relevance_bins,
    daily_top_k_mean_return,
    group_sizes_by_date,
    load_config,
    make_sample_by_dates,
    prepare_model_frame,
    save_json,
    select_feature_columns,
    split_train_calibration,
    validate_purged_split,
)

logger = setup_logger(__name__)

# ── hardware constants ──────────────────────────────────────────────────────
N_CORES = os.cpu_count() or 10
DEFAULT_N_WORKERS = 4
THREADS_PER_WORKER = max(1, N_CORES // DEFAULT_N_WORKERS)

PHASE1_TRIALS = 100
PHASE2_TRIALS = 20
PHASE1_DATE_FRACTION = 0.25


# ── search spaces ───────────────────────────────────────────────────────────

def _narrow(center: float, lo: float, hi: float, shrink: float = 0.3) -> tuple:
    """Narrow a range around a center value for Phase 2."""
    span = hi - lo
    new_lo = max(lo, center - span * shrink)
    new_hi = min(hi, center + span * shrink)
    return new_lo, new_hi


def suggest_ranker_params(trial: optuna.Trial, phase: int, center: Optional[Dict] = None) -> Dict[str, Any]:
    if phase == 1 or center is None:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "relevance_bins": trial.suggest_int("relevance_bins", 3, 7),
        }
    c = center
    lo_est, hi_est = _narrow(c["n_estimators"], 100, 800)
    lo_lr, hi_lr = _narrow(c["learning_rate"], 0.01, 0.3, shrink=0.4)
    return {
        "n_estimators": trial.suggest_int("n_estimators", max(100, int(lo_est)), min(800, int(hi_est)), step=25),
        "learning_rate": trial.suggest_float("learning_rate", max(0.01, lo_lr), min(0.3, hi_lr), log=True),
        "max_depth": trial.suggest_int("max_depth", max(3, c["max_depth"] - 1), min(8, c["max_depth"] + 1)),
        "subsample": trial.suggest_float("subsample", *_narrow(c["subsample"], 0.5, 1.0)),
        "colsample_bytree": trial.suggest_float("colsample_bytree", *_narrow(c["colsample_bytree"], 0.5, 1.0)),
        "reg_alpha": trial.suggest_float("reg_alpha", *_narrow(c["reg_alpha"], 1e-3, 10.0, shrink=0.4), log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", *_narrow(c["reg_lambda"], 1e-3, 10.0, shrink=0.4), log=True),
        "relevance_bins": c["relevance_bins"],  # fix structural param in Phase 2
    }


def suggest_meta_params(trial: optuna.Trial, phase: int, center: Optional[Dict] = None) -> Dict[str, Any]:
    if phase == 1 or center is None:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
    c = center
    lo_est, hi_est = _narrow(c["n_estimators"], 100, 800)
    lo_lr, hi_lr = _narrow(c["learning_rate"], 0.01, 0.3, shrink=0.4)
    return {
        "n_estimators": trial.suggest_int("n_estimators", max(100, int(lo_est)), min(800, int(hi_est)), step=25),
        "learning_rate": trial.suggest_float("learning_rate", max(0.01, lo_lr), min(0.3, hi_lr), log=True),
        "max_depth": trial.suggest_int("max_depth", max(3, c["max_depth"] - 1), min(8, c["max_depth"] + 1)),
        "subsample": trial.suggest_float("subsample", *_narrow(c["subsample"], 0.5, 1.0)),
        "colsample_bytree": trial.suggest_float("colsample_bytree", *_narrow(c["colsample_bytree"], 0.5, 1.0)),
        "reg_alpha": trial.suggest_float("reg_alpha", *_narrow(c["reg_alpha"], 1e-3, 10.0, shrink=0.4), log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", *_narrow(c["reg_lambda"], 1e-3, 10.0, shrink=0.4), log=True),
    }


def suggest_return_params(trial: optuna.Trial, phase: int, center: Optional[Dict] = None) -> Dict[str, Any]:
    if phase == 1 or center is None:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=25),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        }
    c = center
    lo_est, hi_est = _narrow(c["n_estimators"], 50, 500)
    lo_lr, hi_lr = _narrow(c["learning_rate"], 0.01, 0.3, shrink=0.4)
    return {
        "n_estimators": trial.suggest_int("n_estimators", max(50, int(lo_est)), min(500, int(hi_est)), step=25),
        "learning_rate": trial.suggest_float("learning_rate", max(0.01, lo_lr), min(0.3, hi_lr), log=True),
        "max_depth": trial.suggest_int("max_depth", max(2, c["max_depth"] - 1), min(6, c["max_depth"] + 1)),
        "subsample": trial.suggest_float("subsample", *_narrow(c["subsample"], 0.5, 1.0)),
    }


# ── objective functions ─────────────────────────────────────────────────────

def ranker_objective(
    trial: optuna.Trial,
    frame: pd.DataFrame,
    config: Dict[str, Any],
    feature_columns: Sequence[str],
    phase: int,
    center: Optional[Dict] = None,
) -> float:
    """Walk-forward CV for XGBRanker with pruning. Maximizes mean NDCG@k."""
    params = suggest_ranker_params(trial, phase, center)
    cfg = config.copy()
    cfg["ranker"] = {**cfg.get("ranker", {}), **params}
    ranker_cfg = RankerConfig.from_config(cfg)

    working = frame.copy().sort_values(["date", "contractid"]).reset_index(drop=True)
    numerical_features = [col for col in feature_columns if col != "type"]
    categorical_features = [col for col in feature_columns if col == "type"]

    unique_dates = working["date"].nunique()
    effective_min_train_days = min(ranker_cfg.min_fold_train_days, max(2, unique_dates // 2 - 1))
    effective_n_splits = min(ranker_cfg.n_splits, max(1, unique_dates - effective_min_train_days - 1))
    splitter = PurgedWalkForwardSplit(
        n_splits=effective_n_splits,
        purge_days=ranker_cfg.purge_days,
        min_train_days=effective_min_train_days,
    )

    fold_ndcgs: List[float] = []
    for fold_number, (train_idx, test_idx) in enumerate(splitter.split(working["date"]), start=1):
        train_df = working.iloc[train_idx].copy()
        test_df = working.iloc[test_idx].copy()

        preprocessor = build_preprocessor(numerical_features, categorical_features)
        edges = compute_relevance_bins(train_df["target_return"], n_bins=ranker_cfg.relevance_bins)
        train_df["target_relevance"] = apply_relevance_bins(train_df["target_return"], edges)
        test_df["target_relevance"] = apply_relevance_bins(test_df["target_return"], edges)

        X_train = preprocessor.fit_transform(train_df[list(feature_columns)])
        X_test = preprocessor.transform(test_df[list(feature_columns)])
        y_train = train_df["target_relevance"].to_numpy(dtype=float)
        y_test = test_df["target_relevance"].to_numpy(dtype=float)
        train_groups = group_sizes_by_date(train_df)
        test_groups = group_sizes_by_date(test_df)

        pruning_cb = XGBoostPruningCallback(trial, f"validation_0-ndcg@{ranker_cfg.top_k_eval}")
        model = xgb.XGBRanker(
            objective="rank:ndcg",
            eval_metric=f"ndcg@{ranker_cfg.top_k_eval}",
            tree_method="hist",
            random_state=ranker_cfg.random_state,
            n_estimators=ranker_cfg.n_estimators,
            learning_rate=ranker_cfg.learning_rate,
            max_depth=ranker_cfg.max_depth,
            subsample=ranker_cfg.subsample,
            colsample_bytree=ranker_cfg.colsample_bytree,
            reg_alpha=ranker_cfg.reg_alpha,
            reg_lambda=ranker_cfg.reg_lambda,
            n_jobs=THREADS_PER_WORKER,
            early_stopping_rounds=30,
            callbacks=[pruning_cb],
        )
        model.fit(
            X_train, y_train,
            group=train_groups,
            eval_set=[(X_test, y_test)],
            eval_group=[test_groups],
            verbose=False,
        )

        preds = model.predict(X_test)
        test_scored = test_df[["date", "contractid", "target_return", "target_relevance"]].copy()
        test_scored["ranker_score"] = preds
        fold_ndcg = _daily_ndcg(test_scored, ranker_cfg.top_k_eval)
        fold_ndcgs.append(fold_ndcg)

    return float(np.mean(fold_ndcgs))


def meta_objective(
    trial: optuna.Trial,
    frame: pd.DataFrame,
    config: Dict[str, Any],
    feature_columns: Sequence[str],
    phase: int,
    center: Optional[Dict] = None,
) -> float:
    """Walk-forward CV for meta-labeler classifier. Minimizes Brier score."""
    params = suggest_meta_params(trial, phase, center)
    meta_cfg_dict = {**config.get("meta_labeler", {}), **params}

    working = frame.copy().sort_values(["date", "contractid"]).reset_index(drop=True)
    model_features = list(feature_columns) + ["ranker_score", "ranker_rank", "ranker_percentile"]
    numerical_features = [col for col in model_features if col != "type"]
    categorical_features = [col for col in model_features if col == "type"]

    n_splits = int(meta_cfg_dict.get("n_splits", 4))
    purge_days = int(meta_cfg_dict.get("purge_days", 5))
    unique_dates = working["date"].nunique()
    effective_min_train = min(10, max(2, unique_dates // 2 - 1))
    effective_n_splits = min(n_splits, max(1, unique_dates - effective_min_train - 1))
    splitter = PurgedWalkForwardSplit(
        n_splits=effective_n_splits,
        purge_days=purge_days,
        min_train_days=effective_min_train,
    )

    fold_briers: List[float] = []
    for fold_number, (train_idx, test_idx) in enumerate(splitter.split(working["date"]), start=1):
        train_df = working.iloc[train_idx].copy()
        test_df = working.iloc[test_idx].copy()

        preprocessor = build_preprocessor(numerical_features, categorical_features)
        X_train = preprocessor.fit_transform(train_df[model_features])
        X_test = preprocessor.transform(test_df[model_features])
        y_train = train_df["target_profitable"].to_numpy(dtype=int)
        y_test = test_df["target_profitable"].to_numpy(dtype=int)

        positives = float(y_train.sum())
        negatives = float(len(y_train) - positives)
        spw = max(1.0, negatives / positives) if positives > 0 else 1.0

        pruning_cb = XGBoostPruningCallback(trial, "validation_0-logloss")
        model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=42,
            n_estimators=int(params["n_estimators"]),
            learning_rate=float(params["learning_rate"]),
            max_depth=int(params["max_depth"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
            reg_alpha=float(params["reg_alpha"]),
            reg_lambda=float(params["reg_lambda"]),
            scale_pos_weight=spw,
            n_jobs=THREADS_PER_WORKER,
            early_stopping_rounds=30,
            callbacks=[pruning_cb],
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        raw_prob = model.predict_proba(X_test)[:, 1]
        brier = float(brier_score_loss(y_test, raw_prob))
        fold_briers.append(brier)

    return float(np.mean(fold_briers))


def return_objective(
    trial: optuna.Trial,
    frame: pd.DataFrame,
    config: Dict[str, Any],
    feature_columns: Sequence[str],
    phase: int,
    center: Optional[Dict] = None,
) -> float:
    """Walk-forward CV for return predictor. Minimizes MAE on signed log returns."""
    params = suggest_return_params(trial, phase, center)

    working = frame.copy().sort_values(["date", "contractid"]).reset_index(drop=True)
    model_features = list(feature_columns) + ["ranker_score", "ranker_rank", "ranker_percentile"]
    numerical_features = [col for col in model_features if col != "type"]
    categorical_features = [col for col in model_features if col == "type"]

    n_splits = int(config.get("return_model", {}).get("n_splits", 4))
    purge_days = int(config.get("return_model", {}).get("purge_days", 5))
    unique_dates = working["date"].nunique()
    effective_min_train = min(10, max(2, unique_dates // 2 - 1))
    effective_n_splits = min(n_splits, max(1, unique_dates - effective_min_train - 1))
    splitter = PurgedWalkForwardSplit(
        n_splits=effective_n_splits,
        purge_days=purge_days,
        min_train_days=effective_min_train,
    )

    fold_maes: List[float] = []
    for fold_number, (train_idx, test_idx) in enumerate(splitter.split(working["date"]), start=1):
        train_df = working.iloc[train_idx].copy()
        test_df = working.iloc[test_idx].copy()

        preprocessor = build_preprocessor(numerical_features, categorical_features)
        X_train = preprocessor.fit_transform(train_df[model_features])
        X_test = preprocessor.transform(test_df[model_features])
        y_train = train_df["target_signed_log_return"].to_numpy(dtype=float)
        y_test = test_df["target_signed_log_return"].to_numpy(dtype=float)

        # Mean model with pruning callback
        pruning_cb = XGBoostPruningCallback(trial, "validation_0-rmse")
        mid_model = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=int(params["n_estimators"]),
            learning_rate=float(params["learning_rate"]),
            max_depth=int(params["max_depth"]),
            subsample=float(params["subsample"]),
            random_state=42,
            tree_method="hist",
            n_jobs=THREADS_PER_WORKER,
            early_stopping_rounds=30,
            callbacks=[pruning_cb],
        )
        mid_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        mid_pred = mid_model.predict(X_test)
        mae = float(mean_absolute_error(y_test, mid_pred))
        fold_maes.append(mae)

    return float(np.mean(fold_maes))


# ── study helpers ───────────────────────────────────────────────────────────

def create_study(model_name: str, phase: int, output_dir: Path, direction: str) -> optuna.Study:
    study_name = f"{model_name}_phase{phase}"
    journal_path = output_dir / f"{study_name}.journal"
    storage = JournalStorage(JournalFileBackend(str(journal_path)))
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=1,
        max_resource=4,
        reduction_factor=3,
    )
    sampler = optuna.samplers.TPESampler(seed=42, multivariate=True)
    return optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=direction,
        pruner=pruner,
        sampler=sampler,
        load_if_exists=True,
    )


def load_phase1_best(model_name: str, output_dir: Path) -> Dict[str, Any]:
    journal_path = output_dir / f"{model_name}_phase1.journal"
    if not journal_path.exists():
        raise FileNotFoundError(
            f"Phase 1 results not found at {journal_path}. Run --phase 1 first."
        )
    storage = JournalStorage(JournalFileBackend(str(journal_path)))
    study = optuna.load_study(
        study_name=f"{model_name}_phase1",
        storage=storage,
    )
    logger.info("Loaded Phase 1 best params: %s (value=%.4f)", study.best_params, study.best_value)
    return study.best_params


def save_best_params(study: optuna.Study, model_name: str, output_dir: Path) -> Path:
    """Save best params as a YAML file matching config.yaml structure."""
    config_key = {"ranker": "ranker", "meta": "meta_labeler", "return": "return_model"}[model_name]
    best = study.best_params
    output = {config_key: {k: v for k, v in best.items()}}
    path = output_dir / f"{model_name}_best_params.yaml"
    with open(path, "w") as f:
        yaml.dump(output, f, default_flow_style=False, sort_keys=False)
    logger.info("Saved best params to %s", path)

    # Also save full study summary
    summary = {
        "best_value": study.best_value,
        "best_params": best,
        "n_trials": len(study.trials),
        "n_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        "n_complete": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
    }
    save_json(summary, output_dir / f"{model_name}_study_summary.json")
    return path


# ── pre-computation ─────────────────────────────────────────────────────────

def precompute_ranker_features(
    frame: pd.DataFrame,
    config: Dict[str, Any],
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    """Run ranker OOF once and augment frame with ranker features."""
    logger.info("Pre-computing OOF ranker features for meta/return tuning...")
    ranker_cfg = RankerConfig.from_config(config)
    oof, fold_metrics = generate_oof_ranker_features(frame, ranker_cfg, feature_columns)
    augmented = pd.concat([frame, oof], axis=1).dropna(subset=["ranker_score"]).reset_index(drop=True)
    mean_ndcg = np.mean([f["ndcg_at_k"] for f in fold_metrics])
    logger.info("Ranker OOF: %d rows, mean NDCG@k=%.4f", len(augmented), mean_ndcg)
    return augmented


# ── main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter tuning for the options trading pipeline"
    )
    parser.add_argument("--model", choices=["ranker", "meta", "return"], required=True)
    parser.add_argument("--phase", type=int, choices=[1, 2], required=True)
    parser.add_argument("--data", nargs="+", required=True, help="Tuning CSV file(s)")
    parser.add_argument("--config", default="./config.yaml", help="Path to YAML config")
    parser.add_argument("--output-dir", default="./optuna_output", help="Output directory")
    parser.add_argument("--n-trials", type=int, default=None, help="Override trial count")
    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_WORKERS, help="Parallel workers")
    parser.add_argument("--nrows", type=int, default=None, help="Row cap for debugging")
    args = parser.parse_args()

    global THREADS_PER_WORKER
    THREADS_PER_WORKER = max(1, N_CORES // args.n_jobs)

    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and prepare data
    logger.info("Loading data...")
    frame = prepare_model_frame(args.data, config, include_targets=True, nrows=args.nrows)
    logger.info("Loaded %d rows, %d dates", len(frame), frame["date"].nunique())

    # Phase 1: subsample to 25% of dates for speed
    if args.phase == 1:
        n_dates = max(20, int(frame["date"].nunique() * PHASE1_DATE_FRACTION))
        frame = make_sample_by_dates(frame, n_dates)
        logger.info("Phase 1: subsampled to %d dates, %d rows", frame["date"].nunique(), len(frame))

    feature_columns, _, _ = select_feature_columns(frame, config)
    n_trials = args.n_trials or (PHASE1_TRIALS if args.phase == 1 else PHASE2_TRIALS)

    # Load Phase 1 best params for Phase 2
    center = None
    if args.phase == 2:
        center = load_phase1_best(args.model, output_dir)

    # Pre-compute ranker features for meta/return
    if args.model in ("meta", "return"):
        frame = precompute_ranker_features(frame, config, feature_columns)

    # Create study and optimize
    direction_map = {"ranker": "maximize", "meta": "minimize", "return": "minimize"}
    study = create_study(args.model, args.phase, output_dir, direction_map[args.model])

    objective_map = {
        "ranker": ranker_objective,
        "meta": meta_objective,
        "return": return_objective,
    }
    objective_fn = objective_map[args.model]

    logger.info(
        "Starting Optuna: model=%s phase=%d trials=%d workers=%d threads_per_worker=%d",
        args.model, args.phase, n_trials, args.n_jobs, THREADS_PER_WORKER,
    )

    study.optimize(
        lambda trial: objective_fn(trial, frame, config, feature_columns, args.phase, center),
        n_trials=n_trials,
        n_jobs=args.n_jobs,
        show_progress_bar=True,
    )

    # Save results
    params_path = save_best_params(study, args.model, output_dir)
    complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    logger.info(
        "Done: best_value=%.4f | complete=%d pruned=%d | params saved to %s",
        study.best_value, complete, pruned, params_path,
    )


if __name__ == "__main__":
    main()
