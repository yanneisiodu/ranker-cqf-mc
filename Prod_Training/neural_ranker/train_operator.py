"""Train the Regime-Trust Meta-Allocator.

Components:
1. Day gate model (XGBoost) — predicts exposure bucket
2. Bayesian side allocator — call/put budget from posterior
3. Candidate tail-cap — thresholds from data
4. Constrained basket optimizer — position selection

All trained on the operator dataset built by build_operator_dataset.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import TimeSeriesSplit
import joblib

from logger import setup_logger
from utils import save_json

logger = setup_logger(__name__)

# ── Feature groups ──────────────────────────────────────────────────────────

MODEL_TRUST_FEATURES = [
    "score_top1", "score_top5_mean", "score_top20_mean",
    "score_spread_1_20", "score_spread_1_median",
    "score_std_top20", "score_skew_top20",
    "top20_call_ratio", "top20_put_ratio",
]

REGIME_FEATURES = [
    "regime_spy_d_close", "regime_spy_d_sma_50", "regime_spy_d_rsi",
    "regime_spy_d_macd_hist", "regime_vix_d_close", "regime_spy_momentum",
    "regime_realized_vol_5d", "regime_realized_vol_20d",
    "regime_realized_vol_60d", "regime_vrp_20d",
    "regime_spy_above_sma50",
]

EFFICACY_FEATURES = [
    "efficacy_5d_hit_rate", "efficacy_5d_mean_return",
    "efficacy_5d_call_hit_rate", "efficacy_5d_put_hit_rate",
    "efficacy_5d_catastrophe_rate",
    "efficacy_10d_hit_rate", "efficacy_10d_mean_return",
    "efficacy_10d_call_hit_rate", "efficacy_10d_put_hit_rate",
    "efficacy_10d_catastrophe_rate",
    "efficacy_20d_hit_rate", "efficacy_20d_mean_return",
    "efficacy_20d_call_hit_rate", "efficacy_20d_put_hit_rate",
    "efficacy_20d_catastrophe_rate",
    "efficacy_consecutive_losses", "efficacy_current_drawdown",
]

STRUCTURE_FEATURES = [
    "struct_avg_dte", "struct_min_dte", "struct_pct_short_dte",
    "struct_avg_moneyness", "struct_pct_deep_otm",
    "struct_avg_abs_delta", "struct_pct_low_delta",
    "struct_avg_iv", "struct_avg_volume", "struct_avg_spread",
]

ALL_FEATURES = MODEL_TRUST_FEATURES + REGIME_FEATURES + EFFICACY_FEATURES + STRUCTURE_FEATURES


# ── 1. Day Gate Model ───────────────────────────────────────────────────────

def train_day_gate(df: pd.DataFrame, output_dir: Path) -> Dict[str, Any]:
    """Train XGBoost classifier to predict optimal exposure bucket.

    Uses purged time-series CV to avoid leakage.
    """
    logger.info("Training day gate model...")

    # Select features present in dataset
    available = [f for f in ALL_FEATURES if f in df.columns]
    logger.info("  Features available: %d/%d", len(available), len(ALL_FEATURES))

    X = df[available].fillna(0).values
    y = df["optimal_exposure_bucket"].values

    # Simplify to 3 classes for more signal: sit_out (0), normal (1), aggressive (2+)
    y_simple = np.where(y == 0, 0, np.where(y <= 1, 1, 2))
    class_names = ["sit_out", "normal", "aggressive"]
    logger.info("  Class distribution: %s", dict(zip(class_names, np.bincount(y_simple, minlength=3))))

    # Date-level purged split
    from utils import split_train_cal_test_by_date
    if "date" not in df.columns:
        raise ValueError("Operator dataset must have a 'date' column")
    train_idx, cal_idx, test_idx = split_train_cal_test_by_date(
        df["date"], cal_sessions=20, test_sessions=40, purge_sessions=5,
    )
    X_train, y_train = X[train_idx], y_simple[train_idx]
    X_val, y_val = X[cal_idx], y_simple[cal_idx]

    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=42,
        early_stopping_rounds=20,
        eval_metric="mlogloss",
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # Evaluate
    val_pred = model.predict(X_val)
    val_proba = model.predict_proba(X_val)
    acc = accuracy_score(y_val, val_pred)
    logger.info("  Validation accuracy: %.1f%%", acc * 100)
    logger.info("\n%s", classification_report(y_val, val_pred, target_names=class_names))

    # Feature importance
    importances = dict(zip(available, model.feature_importances_))
    top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:15]
    logger.info("  Top features:")
    for name, imp in top_features:
        logger.info("    %s: %.4f", name, imp)

    # Analyze: what does the gate do on the val set?
    val_df = df.iloc[train_size:].copy()
    val_df["gate_pred"] = val_pred
    val_df["gate_class"] = [class_names[p] for p in val_pred]

    for cls_idx, cls_name in enumerate(class_names):
        mask = val_df["gate_pred"] == cls_idx
        if mask.sum() > 0:
            mean_ret = val_df.loc[mask, "basket_return"].mean()
            win_rate = (val_df.loc[mask, "basket_return"] > 0).mean()
            logger.info("  Gate=%s: %d days, mean_return=%.1f%%, win_rate=%.0f%%",
                        cls_name, mask.sum(), mean_ret * 100, win_rate * 100)

    # Save
    artifact = {
        "model": model,
        "features": available,
        "class_names": class_names,
    }
    gate_path = output_dir / "day_gate.joblib"
    joblib.dump(artifact, gate_path)
    logger.info("  Saved day gate to %s", gate_path)

    return {
        "accuracy": acc,
        "features": available,
        "top_features": top_features[:10],
    }


# ── 2. Bayesian Side Allocator ─────────────────────────────────────────────

def train_side_allocator(df: pd.DataFrame, output_dir: Path) -> Dict[str, Any]:
    """Build Bayesian call/put allocator from matured outcomes.

    Maintains Beta posteriors for call and put efficacy,
    conditioned on regime (bull/bear).
    """
    logger.info("Building Bayesian side allocator...")

    # Split by regime
    if "regime_spy_above_sma50" in df.columns:
        bull_mask = df["regime_spy_above_sma50"] > 0.5
    else:
        bull_mask = pd.Series(True, index=df.index)

    results = {}
    for regime, mask in [("bull", bull_mask), ("bear", ~bull_mask)]:
        subset = df[mask]
        if len(subset) < 5:
            results[regime] = {"call_alpha": 1, "call_beta": 1, "put_alpha": 1, "put_beta": 1}
            continue

        # Count wins/losses for calls and puts
        call_wins = subset["call_hit_rate"].dropna()
        put_wins = subset["put_hit_rate"].dropna()

        # Beta prior: alpha=1, beta=1 (uniform)
        call_alpha = 1 + (call_wins > 0.5).sum()
        call_beta = 1 + (call_wins <= 0.5).sum()
        put_alpha = 1 + (put_wins > 0.5).sum()
        put_beta = 1 + (put_wins <= 0.5).sum()

        call_posterior_mean = call_alpha / (call_alpha + call_beta)
        put_posterior_mean = put_alpha / (put_alpha + put_beta)

        # Normalize to get allocation
        total = call_posterior_mean + put_posterior_mean
        call_share = call_posterior_mean / total if total > 0 else 0.5

        results[regime] = {
            "call_alpha": int(call_alpha),
            "call_beta": int(call_beta),
            "put_alpha": int(put_alpha),
            "put_beta": int(put_beta),
            "call_posterior_mean": round(call_posterior_mean, 4),
            "put_posterior_mean": round(put_posterior_mean, 4),
            "call_share": round(call_share, 4),
            "n_days": len(subset),
        }
        logger.info("  %s regime (%d days): call_share=%.1f%%, call_posterior=%.3f, put_posterior=%.3f",
                     regime, len(subset), call_share * 100,
                     call_posterior_mean, put_posterior_mean)

    side_path = output_dir / "side_allocator.json"
    save_json(results, side_path)
    logger.info("  Saved side allocator to %s", side_path)
    return results


# ── 3. Candidate Tail-Cap Thresholds ────────────────────────────────────────

def compute_tail_cap_thresholds(df: pd.DataFrame, output_dir: Path) -> Dict[str, Any]:
    """Compute data-driven thresholds for candidate rejection.

    Identifies option characteristics associated with catastrophic losses.
    """
    logger.info("Computing tail-cap thresholds...")

    thresholds = {}

    # Days with catastrophic losses (>50% of basket lost)
    if "basket_worst_return" in df.columns:
        catastrophe_days = df[df["basket_worst_return"] < -0.90]
        normal_days = df[df["basket_worst_return"] >= -0.90]

        logger.info("  Catastrophe days: %d (%.1f%%)", len(catastrophe_days), len(catastrophe_days) / len(df) * 100)

        # Compare structure features between catastrophe and normal days
        for feat in STRUCTURE_FEATURES:
            if feat in df.columns:
                cat_mean = catastrophe_days[feat].mean()
                norm_mean = normal_days[feat].mean()
                if np.isfinite(cat_mean) and np.isfinite(norm_mean) and norm_mean != 0:
                    ratio = cat_mean / norm_mean
                    if abs(ratio - 1) > 0.2:
                        logger.info("    %s: catastrophe=%.3f, normal=%.3f (ratio=%.2f)",
                                    feat, cat_mean, norm_mean, ratio)

    # Thresholds based on percentiles of losing days
    losing_days = df[df["basket_return"] < -0.10]
    if len(losing_days) > 10:
        thresholds["max_pct_deep_otm"] = float(losing_days["struct_pct_deep_otm"].quantile(0.80)) if "struct_pct_deep_otm" in df.columns else 1.0
        thresholds["max_pct_low_delta"] = float(losing_days["struct_pct_low_delta"].quantile(0.80)) if "struct_pct_low_delta" in df.columns else 1.0
        thresholds["max_pct_short_dte"] = float(losing_days["struct_pct_short_dte"].quantile(0.80)) if "struct_pct_short_dte" in df.columns else 1.0
        thresholds["min_avg_volume"] = float(losing_days["struct_avg_volume"].quantile(0.20)) if "struct_avg_volume" in df.columns else 0.0
    else:
        thresholds = {
            "max_pct_deep_otm": 0.50,
            "max_pct_low_delta": 0.60,
            "max_pct_short_dte": 0.80,
            "min_avg_volume": 50,
        }

    # Hard guardrails (never overridden)
    thresholds["hard_max_exposure"] = 0.50
    thresholds["hard_max_position_pct"] = 0.05
    thresholds["hard_max_positions"] = 10
    thresholds["hard_max_same_direction"] = 7
    thresholds["hard_max_same_expiry"] = 3
    thresholds["hard_daily_loss_limit"] = 0.05
    thresholds["hard_drawdown_reduce_threshold"] = 0.20
    thresholds["hard_drawdown_reduce_factor"] = 0.5

    thresholds_path = output_dir / "tail_cap_thresholds.json"
    save_json(thresholds, thresholds_path)
    logger.info("  Saved thresholds to %s", thresholds_path)
    return thresholds


# ── Main ────────────────────────────────────────────────────────────────────

def train_operator(dataset_path: str, output_dir: str) -> Dict[str, Any]:
    """Train all operator components."""
    df = pd.read_csv(dataset_path)
    logger.info("Loaded operator dataset: %d rows, %d columns", len(df), len(df.columns))

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Day gate
    gate_results = train_day_gate(df, out)

    # 2. Side allocator
    side_results = train_side_allocator(df, out)

    # 3. Tail-cap thresholds
    thresholds = compute_tail_cap_thresholds(df, out)

    # Summary
    summary = {
        "gate": gate_results,
        "side_allocator": side_results,
        "thresholds": thresholds,
        "dataset_size": len(df),
    }
    save_json(summary, out / "operator_summary.json")
    logger.info("Operator training complete. Artifacts saved to %s", out)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Train the Regime-Trust Meta-Allocator")
    parser.add_argument("--dataset", default="./operator_data/operator_dataset.csv")
    parser.add_argument("--output-dir", default="./operator_model")
    args = parser.parse_args()

    train_operator(args.dataset, args.output_dir)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent / "updated_option_agent_codebase"))
    main()
