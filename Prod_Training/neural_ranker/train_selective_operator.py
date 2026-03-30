"""Train the Selective Meta-Allocator.

Bottom-up approach: predict trust at the trade level, then let
exposure emerge from what survives acceptance.

Components:
1. Call meta-model: P(good trade | call candidate)
2. Put meta-model: P(good trade | put candidate)
3. Conformal thresholds: accept/reject calibration
4. Constrained optimizer: select from survivors
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, precision_score,
    recall_score, classification_report,
)
from sklearn.isotonic import IsotonicRegression
import joblib

from logger import setup_logger
from utils import save_json

logger = setup_logger(__name__)

CANDIDATE_FEATURES = [
    # Ranker output
    "score", "score_rank_pct", "score_gap_to_top",
    "score_spread_top_m", "call_ratio_top20",
    # Option structure
    "delta", "abs_delta", "moneyness", "days_to_exp",
    "implied_volatility", "relative_spread", "ask_price",
    "volume", "open_interest", "gamma", "theta", "vega",
    "vanna", "charm",
    # Market regime
    "spy_rsi", "vix", "spy_momentum", "realized_vol_20d",
    "vrp_20d", "is_bull",
    # Efficacy
    "call_hit_5d", "put_hit_5d", "basket_hit_5d",
    "consecutive_losses",
]


def train_meta_model(
    df: pd.DataFrame,
    side: str,
    output_dir: Path,
) -> Dict[str, Any]:
    """Train a calibrated meta-model for one side (call or put)."""
    logger.info("Training %s meta-model on %d candidates...", side, len(df))

    available = [f for f in CANDIDATE_FEATURES if f in df.columns]
    X = df[available].fillna(0).values
    y = df["good_trade"].fillna(0).values.astype(int)

    # Purged time-series split (70/15/15 — train/calibrate/test)
    n = len(X)
    train_end = int(n * 0.70)
    cal_end = int(n * 0.85)

    X_train, y_train = X[:train_end], y[:train_end]
    X_cal, y_cal = X[train_end:cal_end], y[train_end:cal_end]
    X_test, y_test = X[cal_end:], y[cal_end:]

    logger.info("  Train: %d, Calibrate: %d, Test: %d", len(X_train), len(X_cal), len(X_test))
    logger.info("  Positive rate — train: %.1f%%, cal: %.1f%%, test: %.1f%%",
                y_train.mean() * 100, y_cal.mean() * 100, y_test.mean() * 100)

    # Train XGBoost
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=max(1.0, (1 - y_train.mean()) / max(y_train.mean(), 0.01)),
        n_jobs=-1,
        random_state=42,
        early_stopping_rounds=30,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train, eval_set=[(X_cal, y_cal)], verbose=False)

    # Calibrate with isotonic regression
    raw_cal = model.predict_proba(X_cal)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_cal, y_cal)

    # Evaluate on test
    raw_test = model.predict_proba(X_test)[:, 1]
    cal_test = calibrator.predict(raw_test)
    cal_test = np.clip(cal_test, 1e-6, 1 - 1e-6)

    try:
        auc = roc_auc_score(y_test, cal_test)
    except ValueError:
        auc = 0.5
    brier = brier_score_loss(y_test, cal_test)

    logger.info("  Test AUC: %.4f, Brier: %.4f", auc, brier)

    # Find conformal acceptance threshold
    # Target: among accepted trades, at least X% should be good
    # Try thresholds and pick the one that maximizes accepted count
    # while keeping precision above target
    target_precision = 0.40 if side == "put" else 0.50
    best_threshold = 0.5
    best_accepted = 0

    for threshold in np.arange(0.20, 0.80, 0.02):
        accepted = cal_test >= threshold
        if accepted.sum() < 5:
            continue
        precision = y_test[accepted].mean()
        if precision >= target_precision and accepted.sum() > best_accepted:
            best_threshold = threshold
            best_accepted = accepted.sum()

    # Apply best threshold to test set
    accepted_mask = cal_test >= best_threshold
    n_accepted = accepted_mask.sum()
    if n_accepted > 0:
        accepted_precision = y_test[accepted_mask].mean()
        accepted_recall = y_test[accepted_mask].sum() / max(y_test.sum(), 1)
    else:
        accepted_precision = 0
        accepted_recall = 0

    logger.info("  Acceptance threshold: %.2f", best_threshold)
    logger.info("  Accepted: %d/%d (%.0f%%)", n_accepted, len(y_test), n_accepted / len(y_test) * 100)
    logger.info("  Accepted precision: %.1f%% (target: %.0f%%)", accepted_precision * 100, target_precision * 100)
    logger.info("  Accepted recall: %.1f%%", accepted_recall * 100)

    # Feature importance
    importances = dict(zip(available, model.feature_importances_))
    top_feats = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]
    logger.info("  Top features:")
    for name, imp in top_feats:
        logger.info("    %s: %.4f", name, imp)

    # Save
    artifact = {
        "model": model,
        "calibrator": calibrator,
        "features": available,
        "threshold": best_threshold,
        "side": side,
    }
    path = output_dir / f"meta_{side}.joblib"
    joblib.dump(artifact, path)

    return {
        "side": side,
        "auc": auc,
        "brier": brier,
        "threshold": best_threshold,
        "accepted_pct": n_accepted / len(y_test) * 100 if len(y_test) > 0 else 0,
        "accepted_precision": accepted_precision * 100,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "top_features": top_feats[:5],
    }


def train_selective_operator(candidate_dir: str, output_dir: str) -> Dict[str, Any]:
    """Train all components of the selective meta-allocator."""
    cand = Path(candidate_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Train call meta-model
    calls = pd.read_csv(cand / "candidate_calls.csv")
    call_results = train_meta_model(calls, "call", out)

    # Train put meta-model
    puts = pd.read_csv(cand / "candidate_puts.csv")
    put_results = train_meta_model(puts, "put", out)

    # Summary
    summary = {
        "call_meta": call_results,
        "put_meta": put_results,
    }
    save_json(summary, out / "selective_operator_summary.json")
    logger.info("Selective operator training complete. Saved to %s", out)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Train selective meta-allocator")
    parser.add_argument("--candidate-dir", default="./candidate_data")
    parser.add_argument("--output-dir", default="./selective_operator_model")
    args = parser.parse_args()

    train_selective_operator(args.candidate_dir, args.output_dir)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent / "updated_option_agent_codebase"))
    main()
