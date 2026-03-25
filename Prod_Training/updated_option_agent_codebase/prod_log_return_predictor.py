from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_pinball_loss, mean_squared_error

from logger import setup_logger
from prod_train_ranker import RankerConfig, fit_ranker_artifact, generate_oof_ranker_features, predict_ranker_scores
from utils import (
    build_preprocessor,
    get_output_dir,
    inverse_signed_log1p,
    load_config,
    prepare_model_frame,
    save_json,
    select_feature_columns,
    split_train_calibration,
    summarize_frame,
)

logger = setup_logger(__name__)


@dataclass(frozen=True)
class LogReturnConfig:
    horizon_days: int = 5
    n_estimators: int = 120
    learning_rate: float = 0.05
    max_depth: int = 3
    subsample: float = 0.8
    calibration_days: int = 5
    n_splits: int = 4
    purge_days: int = 5
    random_state: int = 42

    @classmethod
    def from_config(cls, config: Dict[str, object]) -> "LogReturnConfig":
        cfg = config.get("return_model", {})
        horizon_days = int(config.get("data", {}).get("horizon_days", cfg.get("horizon_days", 5)))
        return cls(
            horizon_days=horizon_days,
            n_estimators=int(cfg.get("n_estimators", 120)),
            learning_rate=float(cfg.get("learning_rate", 0.05)),
            max_depth=int(cfg.get("max_depth", 3)),
            subsample=float(cfg.get("subsample", 0.8)),
            calibration_days=int(cfg.get("calibration_days", 5)),
            n_splits=int(cfg.get("n_splits", 4)),
            purge_days=int(cfg.get("purge_days", horizon_days)),
        )


def _make_regressor(config: LogReturnConfig, objective: str, alpha: Optional[float] = None) -> xgb.XGBRegressor:
    params = {
        "objective": objective,
        "n_estimators": config.n_estimators,
        "learning_rate": config.learning_rate,
        "max_depth": config.max_depth,
        "subsample": config.subsample,
        "random_state": config.random_state,
        "tree_method": "hist",
        "n_jobs": -1,
    }
    if alpha is not None:
        params["quantile_alpha"] = alpha
    return xgb.XGBRegressor(**params)


def _augment_with_honest_ranker_features(
    frame: pd.DataFrame,
    base_feature_columns: Sequence[str],
    config: Dict[str, object],
    ranker_artifact_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, object], Dict[str, object]]:
    ret_cfg = LogReturnConfig.from_config(config)
    ranker_cfg = RankerConfig.from_config(config)

    train_idx, calibration_idx = split_train_calibration(frame["date"], ret_cfg.calibration_days, ret_cfg.purge_days)
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


def fit_return_artifact(
    frame: pd.DataFrame,
    config: Dict[str, object],
    ranker_artifact_path: Optional[str] = None,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    ret_cfg = LogReturnConfig.from_config(config)
    base_feature_columns, numerical_features, categorical_features = select_feature_columns(frame, config)

    augmented, ranker_artifact, ranker_diagnostics = _augment_with_honest_ranker_features(
        frame,
        base_feature_columns,
        config,
        ranker_artifact_path=ranker_artifact_path,
    )
    model_features = list(base_feature_columns) + ["ranker_score", "ranker_rank", "ranker_percentile"]
    numerical_plus_ranker = list(numerical_features) + ["ranker_score", "ranker_rank", "ranker_percentile"]

    train_idx, calibration_idx = split_train_calibration(augmented["date"], ret_cfg.calibration_days, ret_cfg.purge_days)
    train_df = augmented.iloc[train_idx].copy().reset_index(drop=True)
    calibration_df = augmented.iloc[calibration_idx].copy().reset_index(drop=True)

    preprocessor = build_preprocessor(numerical_plus_ranker, categorical_features)
    X_train = preprocessor.fit_transform(train_df[model_features])
    X_cal = preprocessor.transform(calibration_df[model_features])

    low_model = _make_regressor(ret_cfg, objective="reg:quantileerror", alpha=0.10)
    mid_model = _make_regressor(ret_cfg, objective="reg:squarederror")
    high_model = _make_regressor(ret_cfg, objective="reg:quantileerror", alpha=0.90)

    target = train_df["target_signed_log_return"].to_numpy(dtype=float)
    low_model.fit(X_train, target)
    mid_model.fit(X_train, target)
    high_model.fit(X_train, target)

    low_pred = low_model.predict(X_cal)
    mid_pred = mid_model.predict(X_cal)
    high_pred = high_model.predict(X_cal)
    low_pred, mid_pred, high_pred = np.sort(np.vstack([low_pred, mid_pred, high_pred]), axis=0)

    cal_true = calibration_df["target_signed_log_return"].to_numpy(dtype=float)
    metrics = {
        "mae_signed_log": float(mean_absolute_error(cal_true, mid_pred)),
        "rmse_signed_log": float(np.sqrt(mean_squared_error(cal_true, mid_pred))),
        "pinball_q10": float(mean_pinball_loss(cal_true, low_pred, alpha=0.10)),
        "pinball_q90": float(mean_pinball_loss(cal_true, high_pred, alpha=0.90)),
        "interval_80_coverage": float(np.mean((cal_true >= low_pred) & (cal_true <= high_pred))),
        "train_rows": int(len(train_df)),
        "calibration_rows": int(len(calibration_df)),
        "sign_accuracy": float(np.mean((np.sign(mid_pred) == np.sign(cal_true)).astype(float))),
    }

    artifact = {
        "artifact_type": "return_distribution",
        "feature_columns": model_features,
        "base_feature_columns": list(base_feature_columns),
        "numerical_features": numerical_plus_ranker,
        "categorical_features": list(categorical_features),
        "preprocessor": preprocessor,
        "models": {"q10": low_model, "mean": mid_model, "q90": high_model},
        "config": asdict(ret_cfg),
        "ranker_artifact_path": ranker_artifact_path,
        "train_summary": summarize_frame(frame),
        "metrics": metrics,
        "ranker_diagnostics": ranker_diagnostics,
    }
    return artifact, ranker_artifact


def predict_return_distribution(
    artifact: Dict[str, object],
    frame: pd.DataFrame,
    ranker_artifact: Dict[str, object],
) -> pd.DataFrame:
    scored = predict_ranker_scores(ranker_artifact, frame)
    X = artifact["preprocessor"].transform(scored[artifact["feature_columns"]])
    low = artifact["models"]["q10"].predict(X)
    mean = artifact["models"]["mean"].predict(X)
    high = artifact["models"]["q90"].predict(X)
    low, mean, high = np.sort(np.vstack([low, mean, high]), axis=0)

    scored["pred_signed_log_q10"] = low
    scored["pred_signed_log_mean"] = mean
    scored["pred_signed_log_q90"] = high
    scored["pred_return_q10"] = inverse_signed_log1p(low)
    scored["expected_return"] = inverse_signed_log1p(mean)
    scored["pred_return_q90"] = inverse_signed_log1p(high)
    scored["uncertainty"] = scored["pred_return_q90"] - scored["pred_return_q10"]
    return scored


def train_log_return_predictor(
    data_files: Sequence[str],
    config_file: Optional[str] = None,
    output_dir: Optional[str] = None,
    ranker_artifact_path: Optional[str] = None,
    nrows: Optional[int] = None,
) -> Dict[str, object]:
    config = load_config(config_file)
    frame = prepare_model_frame(data_files, config, include_targets=True, nrows=nrows)
    logger.info("Training return distribution model on %s", summarize_frame(frame))

    artifact, ranker_artifact = fit_return_artifact(frame, config, ranker_artifact_path=ranker_artifact_path)

    root = get_output_dir(config, output_dir)
    artifact_path = root / "return_distribution_artifact.joblib"
    metrics_path = root / "return_distribution_metrics.json"
    ranker_path = root / "ranker_artifact_from_return.joblib"

    joblib.dump(artifact, artifact_path)
    save_json({"metrics": artifact["metrics"], "ranker_diagnostics": artifact["ranker_diagnostics"]}, metrics_path)
    if ranker_artifact_path is None:
        joblib.dump(ranker_artifact, ranker_path)
        ranker_artifact_path = str(ranker_path)

    artifact["ranker_artifact_path"] = ranker_artifact_path
    joblib.dump(artifact, artifact_path)
    logger.info("Saved return distribution artifact to %s", artifact_path)
    return {
        "artifact_path": str(artifact_path),
        "metrics_path": str(metrics_path),
        "ranker_artifact_path": ranker_artifact_path,
        "metrics": artifact["metrics"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a leakage-safe return distribution model for options")
    parser.add_argument("--data", nargs="+", required=True, help="CSV file(s) with historical option snapshots")
    parser.add_argument("--config", default="./config.yaml", help="Path to YAML config")
    parser.add_argument("--output-dir", default=None, help="Directory for trained artifacts")
    parser.add_argument("--ranker-artifact", default=None, help="Optional trained ranker artifact")
    parser.add_argument("--nrows", type=int, default=None, help="Optional row cap for quick smoke runs")
    args = parser.parse_args()

    result = train_log_return_predictor(
        args.data,
        config_file=args.config,
        output_dir=args.output_dir,
        ranker_artifact_path=args.ranker_artifact,
        nrows=args.nrows,
    )
    logger.info("Training complete: %s", result)


if __name__ == "__main__":
    main()
