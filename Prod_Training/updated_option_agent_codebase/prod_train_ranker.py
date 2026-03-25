from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import ndcg_score

from logger import setup_logger
from utils import (
    PurgedWalkForwardSplit,
    apply_relevance_bins,
    build_preprocessor,
    compute_relevance_bins,
    daily_top_k_mean_return,
    get_output_dir,
    group_sizes_by_date,
    load_config,
    prepare_model_frame,
    save_json,
    select_feature_columns,
    summarize_frame,
    validate_purged_split,
)

logger = setup_logger(__name__)


@dataclass(frozen=True)
class RankerConfig:
    horizon_days: int = 5
    n_splits: int = 4
    purge_days: int = 5
    min_fold_train_days: int = 10
    n_estimators: int = 300
    learning_rate: float = 0.05
    max_depth: int = 4
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    top_k_eval: int = 20
    relevance_bins: int = 5
    random_state: int = 42

    @classmethod
    def from_config(cls, config: Dict[str, object]) -> "RankerConfig":
        cfg = config.get("ranker", {})
        horizon_days = int(config.get("data", {}).get("horizon_days", cfg.get("horizon_days", 5)))
        return cls(
            horizon_days=horizon_days,
            n_splits=int(cfg.get("n_splits", 4)),
            purge_days=int(cfg.get("purge_days", horizon_days)),
            min_fold_train_days=int(cfg.get("min_fold_train_days", 10)),
            n_estimators=int(cfg.get("n_estimators", 300)),
            learning_rate=float(cfg.get("learning_rate", 0.05)),
            max_depth=int(cfg.get("max_depth", 4)),
            subsample=float(cfg.get("subsample", 0.8)),
            colsample_bytree=float(cfg.get("colsample_bytree", 0.8)),
            reg_alpha=float(cfg.get("reg_alpha", 0.1)),
            reg_lambda=float(cfg.get("reg_lambda", 1.0)),
            top_k_eval=int(cfg.get("top_k_eval", 20)),
            relevance_bins=int(cfg.get("relevance_bins", 5)),
        )


def _ranker_model(config: RankerConfig) -> xgb.XGBRanker:
    return xgb.XGBRanker(
        objective="rank:ndcg",
        eval_metric=f"ndcg@{config.top_k_eval}",
        tree_method="hist",
        random_state=config.random_state,
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        max_depth=config.max_depth,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        reg_alpha=config.reg_alpha,
        reg_lambda=config.reg_lambda,
        n_jobs=4,
    )


def _daily_ndcg(scored: pd.DataFrame, k: int) -> float:
    values: List[float] = []
    for _, group in scored.groupby("date"):
        if len(group) < 2:
            continue
        truth = group["target_relevance"].to_numpy(dtype=float).reshape(1, -1)
        pred = group["ranker_score"].to_numpy(dtype=float).reshape(1, -1)
        if np.allclose(truth, truth[:, :1]):
            continue
        values.append(float(ndcg_score(truth, pred, k=min(k, len(group)))))
    return float(np.mean(values)) if values else float("nan")


def fit_ranker_artifact(frame: pd.DataFrame, config: RankerConfig, feature_columns: Optional[Sequence[str]] = None) -> Dict[str, object]:
    working = frame.copy().sort_values(["date", "contractid"]).reset_index(drop=True)
    if feature_columns is None:
        feature_columns, numerical_features, categorical_features = select_feature_columns(working, load_config(None))
    else:
        feature_columns = list(feature_columns)
        categorical_features = [col for col in feature_columns if col == "type"]
        numerical_features = [col for col in feature_columns if col != "type"]

    preprocessor = build_preprocessor(numerical_features, categorical_features)
    edges = compute_relevance_bins(working["target_return"], n_bins=config.relevance_bins)
    working["target_relevance"] = apply_relevance_bins(working["target_return"], edges)

    X = preprocessor.fit_transform(working[list(feature_columns)])
    model = _ranker_model(config)
    groups = group_sizes_by_date(working)
    model.fit(X, working["target_relevance"].to_numpy(dtype=float), group=groups, verbose=False)
    return {
        "artifact_type": "ranker",
        "feature_columns": list(feature_columns),
        "numerical_features": list(numerical_features),
        "categorical_features": list(categorical_features),
        "preprocessor": preprocessor,
        "model": model,
        "relevance_edges": edges,
        "config": asdict(config),
        "train_summary": summarize_frame(working),
    }


def predict_ranker_scores(artifact: Dict[str, object], frame: pd.DataFrame) -> pd.DataFrame:
    feature_columns = artifact["feature_columns"]
    preprocessor = artifact["preprocessor"]
    model = artifact["model"]

    scored = frame.copy().sort_values(["date", "contractid"]).reset_index(drop=True)
    X = preprocessor.transform(scored[feature_columns])
    scored["ranker_score"] = model.predict(X)
    scored["ranker_rank"] = scored.groupby("date")["ranker_score"].rank(ascending=False, method="first")
    group_size = scored.groupby("date")["ranker_score"].transform("count")
    scored["ranker_percentile"] = 1.0 - (scored["ranker_rank"] - 1.0) / group_size.clip(lower=1)
    return scored


def generate_oof_ranker_features(frame: pd.DataFrame, config: RankerConfig, feature_columns: Sequence[str]) -> Tuple[pd.DataFrame, List[Dict[str, object]]]:
    working = frame.copy().sort_values(["date", "contractid"]).reset_index(drop=True)
    numerical_features = [col for col in feature_columns if col != "type"]
    categorical_features = [col for col in feature_columns if col == "type"]
    unique_dates = working["date"].nunique()
    effective_min_train_days = min(config.min_fold_train_days, max(2, unique_dates // 2 - 1))
    effective_n_splits = min(config.n_splits, max(1, unique_dates - effective_min_train_days - 1))
    splitter = PurgedWalkForwardSplit(
        n_splits=effective_n_splits,
        purge_days=config.purge_days,
        min_train_days=effective_min_train_days,
    )

    oof = pd.DataFrame(index=working.index, data={
        "ranker_score": np.nan,
        "ranker_rank": np.nan,
        "ranker_percentile": np.nan,
    })
    fold_metrics: List[Dict[str, object]] = []

    for fold_number, (train_idx, test_idx) in enumerate(splitter.split(working["date"]), start=1):
        train_df = working.iloc[train_idx].copy()
        test_df = working.iloc[test_idx].copy()
        validate_purged_split(train_df["date"], test_df["date"], config.purge_days)

        preprocessor = build_preprocessor(numerical_features, categorical_features)
        edges = compute_relevance_bins(train_df["target_return"], n_bins=config.relevance_bins)
        train_df["target_relevance"] = apply_relevance_bins(train_df["target_return"], edges)
        test_df["target_relevance"] = apply_relevance_bins(test_df["target_return"], edges)

        X_train = preprocessor.fit_transform(train_df[list(feature_columns)])
        X_test = preprocessor.transform(test_df[list(feature_columns)])

        model = _ranker_model(config)
        model.fit(X_train, train_df["target_relevance"].to_numpy(dtype=float), group=group_sizes_by_date(train_df), verbose=False)
        preds = model.predict(X_test)

        test_scored = test_df[["date", "contractid", "target_return", "target_relevance"]].copy()
        test_scored["ranker_score"] = preds
        test_scored["ranker_rank"] = test_scored.groupby("date")["ranker_score"].rank(ascending=False, method="first")
        counts = test_scored.groupby("date")["ranker_score"].transform("count")
        test_scored["ranker_percentile"] = 1.0 - (test_scored["ranker_rank"] - 1.0) / counts.clip(lower=1)

        oof.loc[test_idx, ["ranker_score", "ranker_rank", "ranker_percentile"]] = test_scored[
            ["ranker_score", "ranker_rank", "ranker_percentile"]
        ].to_numpy()

        fold_metrics.append(
            {
                "fold": fold_number,
                "train_rows": int(len(train_df)),
                "test_rows": int(len(test_df)),
                "train_end": str(train_df["date"].max()),
                "test_start": str(test_df["date"].min()),
                "test_end": str(test_df["date"].max()),
                "ndcg_at_k": _daily_ndcg(test_scored, config.top_k_eval),
                "top_k_mean_return": daily_top_k_mean_return(test_scored, "ranker_score", "target_return", k=min(5, config.top_k_eval)),
            }
        )

    return oof, fold_metrics


def train_ranker(
    data_files: Sequence[str],
    config_file: Optional[str] = None,
    output_dir: Optional[str] = None,
    nrows: Optional[int] = None,
) -> Dict[str, object]:
    config = load_config(config_file)
    ranker_config = RankerConfig.from_config(config)
    frame = prepare_model_frame(data_files, config, include_targets=True, nrows=nrows)
    feature_columns, _, _ = select_feature_columns(frame, config)

    logger.info("Training ranker on %s", summarize_frame(frame))
    oof_features, fold_metrics = generate_oof_ranker_features(frame, ranker_config, feature_columns)
    final_artifact = fit_ranker_artifact(frame, ranker_config, feature_columns)

    scored_oof = pd.concat([frame[["date", "contractid", "target_return"]], oof_features], axis=1)
    valid_oof = scored_oof.dropna(subset=["ranker_score"]).copy()
    if len(valid_oof):
        edges = final_artifact["relevance_edges"]
        valid_oof["target_relevance"] = apply_relevance_bins(valid_oof["target_return"], edges)
        summary_metrics = {
            "oof_rows": int(len(valid_oof)),
            "mean_ndcg_at_k": _daily_ndcg(valid_oof, ranker_config.top_k_eval),
            "oof_top_5_mean_return": daily_top_k_mean_return(valid_oof, "ranker_score", "target_return", k=5),
        }
    else:
        summary_metrics = {"oof_rows": 0, "mean_ndcg_at_k": float("nan"), "oof_top_5_mean_return": float("nan")}

    root = get_output_dir(config, output_dir)
    artifact_path = root / "ranker_artifact.joblib"
    metrics_path = root / "ranker_metrics.json"
    oof_path = root / "ranker_oof_predictions.csv"

    final_artifact["metrics"] = {"folds": fold_metrics, "summary": summary_metrics}
    joblib.dump(final_artifact, artifact_path)
    save_json(final_artifact["metrics"], metrics_path)
    valid_oof.to_csv(oof_path, index=False)

    logger.info("Saved ranker artifact to %s", artifact_path)
    return {
        "artifact_path": str(artifact_path),
        "metrics_path": str(metrics_path),
        "oof_path": str(oof_path),
        "metrics": final_artifact["metrics"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a leakage-safe XGBoost options ranker")
    parser.add_argument("--data", nargs="+", required=True, help="CSV file(s) with historical option snapshots")
    parser.add_argument("--config", default="./config.yaml", help="Path to YAML config")
    parser.add_argument("--output-dir", default=None, help="Directory for trained artifacts")
    parser.add_argument("--nrows", type=int, default=None, help="Optional row cap for quick smoke runs")
    args = parser.parse_args()

    result = train_ranker(args.data, config_file=args.config, output_dir=args.output_dir, nrows=args.nrows)
    logger.info("Training complete: %s", result)


if __name__ == "__main__":
    main()
