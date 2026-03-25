from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, precision_score, recall_score, roc_auc_score

from logger import setup_logger
from prod_train_ranker import RankerConfig, fit_ranker_artifact, generate_oof_ranker_features, predict_ranker_scores
from utils import (
    build_preprocessor,
    get_output_dir,
    load_config,
    prepare_model_frame,
    save_json,
    select_feature_columns,
    split_train_calibration,
    summarize_frame,
)

logger = setup_logger(__name__)


@dataclass(frozen=True)
class MetaLabelerConfig:
    horizon_days: int = 5
    n_estimators: int = 300
    learning_rate: float = 0.05
    max_depth: int = 4
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    calibration_days: int = 5
    n_splits: int = 4
    purge_days: int = 5
    random_state: int = 42

    @classmethod
    def from_config(cls, config: Dict[str, object]) -> "MetaLabelerConfig":
        cfg = config.get("meta_labeler", {})
        horizon_days = int(config.get("data", {}).get("horizon_days", cfg.get("horizon_days", 5)))
        return cls(
            horizon_days=horizon_days,
            n_estimators=int(cfg.get("n_estimators", 300)),
            learning_rate=float(cfg.get("learning_rate", 0.05)),
            max_depth=int(cfg.get("max_depth", 4)),
            subsample=float(cfg.get("subsample", 0.8)),
            colsample_bytree=float(cfg.get("colsample_bytree", 0.8)),
            reg_alpha=float(cfg.get("reg_alpha", 0.1)),
            reg_lambda=float(cfg.get("reg_lambda", 1.0)),
            calibration_days=int(cfg.get("calibration_days", 5)),
            n_splits=int(cfg.get("n_splits", 4)),
            purge_days=int(cfg.get("purge_days", horizon_days)),
        )


def _classifier(config: MetaLabelerConfig, scale_pos_weight: float) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=config.random_state,
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        max_depth=config.max_depth,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        reg_alpha=config.reg_alpha,
        reg_lambda=config.reg_lambda,
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
    )


def _fit_calibrator(raw_prob: np.ndarray, y_true: np.ndarray) -> Optional[IsotonicRegression]:
    unique_labels = np.unique(y_true)
    if len(unique_labels) < 2:
        return None
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_prob, y_true)
    return calibrator


def _apply_calibrator(calibrator: Optional[IsotonicRegression], raw_prob: np.ndarray) -> np.ndarray:
    if calibrator is None:
        return np.clip(raw_prob, 1e-6, 1 - 1e-6)
    return np.clip(calibrator.predict(raw_prob), 1e-6, 1 - 1e-6)


def _classification_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    pred = (prob >= threshold).astype(int)
    metrics: Dict[str, float] = {
        "brier": float(brier_score_loss(y_true, prob)),
        "log_loss": float(log_loss(y_true, prob, labels=[0, 1])),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "average_precision": float(average_precision_score(y_true, prob)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, prob))
    except ValueError:
        metrics["roc_auc"] = float("nan")
    return metrics


def _augment_with_honest_ranker_features(
    frame: pd.DataFrame,
    base_feature_columns: Sequence[str],
    config: Dict[str, object],
    ranker_artifact_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, object], Dict[str, object]]:
    meta_cfg = MetaLabelerConfig.from_config(config)
    ranker_cfg = RankerConfig.from_config(config)

    train_idx, calibration_idx = split_train_calibration(frame["date"], meta_cfg.calibration_days, meta_cfg.purge_days)
    train_df = frame.iloc[train_idx].copy().reset_index(drop=True)
    calibration_df = frame.iloc[calibration_idx].copy().reset_index(drop=True)

    honest_ranker_train, fold_metrics = generate_oof_ranker_features(train_df, ranker_cfg, base_feature_columns)
    train_df = pd.concat([train_df, honest_ranker_train], axis=1).dropna(subset=["ranker_score"]).reset_index(drop=True)

    if ranker_artifact_path:
        ranker_artifact = joblib.load(ranker_artifact_path)
    else:
        ranker_artifact = fit_ranker_artifact(frame, ranker_cfg, base_feature_columns)

    calibration_scored = predict_ranker_scores(ranker_artifact, calibration_df)
    combined = pd.concat([train_df, calibration_scored], axis=0, ignore_index=True)
    diagnostics = {
        "train_rows_after_oof": int(len(train_df)),
        "calibration_rows": int(len(calibration_scored)),
        "ranker_cv": fold_metrics,
    }
    return combined, ranker_artifact, diagnostics


def fit_meta_labeler_artifact(
    frame: pd.DataFrame,
    config: Dict[str, object],
    ranker_artifact_path: Optional[str] = None,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    meta_cfg = MetaLabelerConfig.from_config(config)
    base_feature_columns, numerical_features, categorical_features = select_feature_columns(frame, config)

    augmented, ranker_artifact, ranker_diagnostics = _augment_with_honest_ranker_features(
        frame,
        base_feature_columns,
        config,
        ranker_artifact_path=ranker_artifact_path,
    )
    meta_features = list(base_feature_columns) + ["ranker_score", "ranker_rank", "ranker_percentile"]
    numerical_plus_ranker = list(numerical_features) + ["ranker_score", "ranker_rank", "ranker_percentile"]

    train_idx, calibration_idx = split_train_calibration(augmented["date"], meta_cfg.calibration_days, meta_cfg.purge_days)
    train_df = augmented.iloc[train_idx].copy().reset_index(drop=True)
    calibration_df = augmented.iloc[calibration_idx].copy().reset_index(drop=True)

    scale_pos_weight = 1.0
    positives = float(train_df["target_profitable"].sum())
    negatives = float(len(train_df) - positives)
    if positives > 0:
        scale_pos_weight = max(1.0, negatives / positives)

    preprocessor = build_preprocessor(numerical_plus_ranker, categorical_features)
    X_train = preprocessor.fit_transform(train_df[meta_features])
    X_cal = preprocessor.transform(calibration_df[meta_features])

    model = _classifier(meta_cfg, scale_pos_weight)
    model.fit(X_train, train_df["target_profitable"].to_numpy(dtype=int), verbose=False)

    raw_cal = model.predict_proba(X_cal)[:, 1]
    calibrator = _fit_calibrator(raw_cal, calibration_df["target_profitable"].to_numpy(dtype=int))
    cal_prob = _apply_calibrator(calibrator, raw_cal)

    metrics = _classification_metrics(calibration_df["target_profitable"].to_numpy(dtype=int), cal_prob)
    metrics.update(
        {
            "train_rows": int(len(train_df)),
            "calibration_rows": int(len(calibration_df)),
            "calibration_positive_rate": float(calibration_df["target_profitable"].mean()),
        }
    )

    artifact = {
        "artifact_type": "meta_labeler",
        "feature_columns": meta_features,
        "base_feature_columns": list(base_feature_columns),
        "numerical_features": numerical_plus_ranker,
        "categorical_features": list(categorical_features),
        "preprocessor": preprocessor,
        "model": model,
        "calibrator": calibrator,
        "config": asdict(meta_cfg),
        "ranker_artifact_path": ranker_artifact_path,
        "train_summary": summarize_frame(frame),
        "metrics": metrics,
        "ranker_diagnostics": ranker_diagnostics,
    }
    return artifact, ranker_artifact


def predict_meta_probabilities(
    artifact: Dict[str, object],
    frame: pd.DataFrame,
    ranker_artifact: Dict[str, object],
) -> pd.DataFrame:
    scored = predict_ranker_scores(ranker_artifact, frame)
    feature_columns = artifact["feature_columns"]
    X = artifact["preprocessor"].transform(scored[feature_columns])
    raw_prob = artifact["model"].predict_proba(X)[:, 1]
    scored["prob_profit_raw"] = raw_prob
    scored["prob_profit"] = _apply_calibrator(artifact.get("calibrator"), raw_prob)
    return scored


def train_meta_labeler(
    data_files: Sequence[str],
    config_file: Optional[str] = None,
    output_dir: Optional[str] = None,
    ranker_artifact_path: Optional[str] = None,
    nrows: Optional[int] = None,
) -> Dict[str, object]:
    config = load_config(config_file)
    frame = prepare_model_frame(data_files, config, include_targets=True, nrows=nrows)
    logger.info("Training meta-labeler on %s", summarize_frame(frame))

    artifact, ranker_artifact = fit_meta_labeler_artifact(frame, config, ranker_artifact_path=ranker_artifact_path)

    root = get_output_dir(config, output_dir)
    artifact_path = root / "meta_labeler_artifact.joblib"
    metrics_path = root / "meta_labeler_metrics.json"
    ranker_path = root / "ranker_artifact_from_meta.joblib"

    joblib.dump(artifact, artifact_path)
    save_json({"metrics": artifact["metrics"], "ranker_diagnostics": artifact["ranker_diagnostics"]}, metrics_path)
    if ranker_artifact_path is None:
        joblib.dump(ranker_artifact, ranker_path)
        ranker_artifact_path = str(ranker_path)

    artifact["ranker_artifact_path"] = ranker_artifact_path
    joblib.dump(artifact, artifact_path)
    logger.info("Saved meta-labeler artifact to %s", artifact_path)
    return {
        "artifact_path": str(artifact_path),
        "metrics_path": str(metrics_path),
        "ranker_artifact_path": ranker_artifact_path,
        "metrics": artifact["metrics"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a leakage-safe meta-labeler for option profitability")
    parser.add_argument("--data", nargs="+", required=True, help="CSV file(s) with historical option snapshots")
    parser.add_argument("--config", default="./config.yaml", help="Path to YAML config")
    parser.add_argument("--output-dir", default=None, help="Directory for trained artifacts")
    parser.add_argument("--ranker-artifact", default=None, help="Optional trained ranker artifact")
    parser.add_argument("--nrows", type=int, default=None, help="Optional row cap for quick smoke runs")
    args = parser.parse_args()

    result = train_meta_labeler(
        args.data,
        config_file=args.config,
        output_dir=args.output_dir,
        ranker_artifact_path=args.ranker_artifact,
        nrows=args.nrows,
    )
    logger.info("Training complete: %s", result)


if __name__ == "__main__":
    main()
