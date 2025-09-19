#!/usr/bin/env python3
"""Integrated inference pipeline: Ranker → CQF → Stress MC → Trade recommendations."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import joblib
import logging
import numpy as np
import pandas as pd

# Local imports (training utilities)
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR / "Training") not in sys.path:
    sys.path.insert(0, str(BASE_DIR / "Training"))

from utils import load_config, preprocess_data  # noqa: E402
from prod_train_ranker import (  # noqa: E402
    calculate_target,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_LOOKAHEAD_DAYS,
)
from prod_cqf import OptimalCQF  # noqa: E402
from regime_tools import add_regime_features, add_realized_vol_features  # noqa: E402
from inference.prod_stress_mc import EnhancedStressMC, StressConfig  # noqa: E402


K_VALUES = [1, 5, 10, 20]

logger = logging.getLogger("inference_pipeline")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def calculate_precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    if k <= 0:
        return 0.0
    idx = np.argsort(scores)[::-1][:k]
    return float(np.mean(y_true[idx] > 0))


def ndcg_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    if k <= 0:
        return 0.0
    gains = (2 ** y_true) - 1
    order = np.argsort(scores)[::-1]
    gains_sorted = gains[order][:k]
    dcg = np.sum(gains_sorted / np.log2(np.arange(2, k + 2)))
    ideal = np.sort(gains)[::-1][:k]
    idcg = np.sum(ideal / np.log2(np.arange(2, k + 2)))
    return float(dcg / max(idcg, 1e-12))


def evaluate_ranker_metrics(df: pd.DataFrame) -> Dict[str, float]:
    df_sorted = df.sort_values("date")
    metrics = {}
    for date, group in df_sorted.groupby("date"):
        y_true = group['target_relevance_int'].values
        pred = group['ranker_score'].values
        for k in K_VALUES:
            actual_k = min(k, len(group))
            ndcg = ndcg_at_k(y_true, pred, actual_k)
            prec = calculate_precision_at_k(y_true, pred, actual_k)
            metrics.setdefault(f'ndcg@{k}', []).append(ndcg)
            metrics.setdefault(f'precision@{k}', []).append(prec)

    return {f'mean_{m}': float(np.nanmean(vals)) for m, vals in metrics.items()}


def evaluate_cqf_coverage(cqf: OptimalCQF, df: pd.DataFrame, preds: pd.DataFrame,
                           horizon_days: int = 30) -> Dict[str, float]:
    dates = sorted(df['date'].unique())[-horizon_days:]
    mask = df['date'].isin(dates) & df['target_pnl'].notna()
    y_true = df.loc[mask, 'target_pnl'].values.astype(float)
    metrics = cqf.evaluate_coverage(
        y_true,
        {col: preds.loc[mask, col].values for col in preds.columns}
    )
    return metrics


def main():
    p = argparse.ArgumentParser(description="Integrated inference pipeline")
    p.add_argument('--raw-data', default=str(BASE_DIR / 'year_2024_data.csv'))
    p.add_argument('--config', default=str(BASE_DIR / 'config.yaml'))
    p.add_argument('--ranker-model', default=str(BASE_DIR / 'model_output/xgboost_ranker_2022_2022_fixed_params_20250918_185021.joblib'))
    p.add_argument('--ranker-features', default=str(BASE_DIR / 'model_output/xgb_feature_names_2022_2022_20250918_185021.pkl'))
    p.add_argument('--sharpe-edges', default=str(BASE_DIR / 'model_output/sharpe_qcut_edges_2022_2022_20250918_185021.pkl'))
    p.add_argument('--cqf-model', default=str(BASE_DIR / 'model_output/optimal_cqf_step8.joblib'))
    p.add_argument('--top-n', type=int, default=1000)
    p.add_argument('--output-dir', default=str(BASE_DIR / 'inference_output'))
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading raw data: %s", args.raw_data)
    raw_df = pd.read_csv(args.raw_data, low_memory=False)
    raw_df['date'] = pd.to_datetime(raw_df['date'], errors='coerce')

    config = load_config(args.config)

    logger.info("Preprocessing data via shared utils")
    processed_df, _ = preprocess_data(raw_df, config, scaler=None)
    processed_df = add_regime_features(processed_df)
    processed_df = add_realized_vol_features(processed_df)

    logger.info("Computing target Sharpe for diagnostics")
    target_df = calculate_target(processed_df.copy(), TARGET_LOOKAHEAD_DAYS)

    logger.info("Computing delta-hedged PnL targets for CQF monitoring")
    temp_cqf = OptimalCQF()
    pnl_df = temp_cqf.calculate_delta_hedged_pnl(processed_df.copy(), TARGET_LOOKAHEAD_DAYS)
    target_df = target_df.merge(
        pnl_df[['contractID', 'date', 'target_pnl', 'future_option_price']],
        on=['contractID', 'date'],
        how='left'
    )

    # Map to integer relevance using training quantiles
    edges = joblib.load(args.sharpe_edges)
    q1, q2, q3 = edges[1], edges[2], edges[3]
    target_df['target_relevance_int'] = pd.cut(
        target_df['target_5d_sharpe'],
        bins=edges,
        labels=[0, 1, 2, 3]
    ).astype(int)

    # Ranker inference
    ranker_model = joblib.load(args.ranker_model)
    feature_list = joblib.load(args.ranker_features)
    X_ranker = target_df[['date'] + feature_list].copy()
    if 'type' in X_ranker.columns:
        X_ranker['type'] = X_ranker['type'].astype(str)
    scores = ranker_model.predict(X_ranker[feature_list])
    target_df['ranker_score'] = scores

    metrics = evaluate_ranker_metrics(target_df)
    logger.info("Ranker metrics: %s", metrics)

    top_n = target_df.nlargest(args.top_n, 'ranker_score').copy()
    top_n.to_csv(output_dir / 'ranker_candidates.csv', index=False)

    # CQF inference
    cqf_artifact = joblib.load(args.cqf_model)
    cqf = OptimalCQF()
    cqf.models = cqf_artifact['models']
    cqf.preprocessor = cqf_artifact['preprocessor']
    cqf.feature_names = cqf_artifact['feature_names']
    cqf.conformal_adjustments = cqf_artifact['conformal_adjustments']
    cqf.conformal_calibrator = cqf_artifact['conformal_calibrator']
    cqf.page_hinkley = cqf_artifact['page_hinkley']
    cqf.evt_adjuster = cqf_artifact['evt_adjuster']
    cqf.prob_calibrator = cqf_artifact['prob_calibrator']
    cqf.quantiles = cqf_artifact['quantiles']
    cqf.horizon = cqf_artifact.get('horizon', cqf.horizon)

    quantile_preds = cqf.predict_quantiles(top_n, apply_conformal=True)
    decision_feats = cqf.calculate_decision_features(quantile_preds)
    price_preds = cqf.convert_to_price_predictions(top_n, quantile_preds)

    cqf_output = top_n[['contractID', 'date', 'ranker_score', 'target_pnl', 'future_option_price']].copy()
    cqf_output = pd.concat([cqf_output, quantile_preds, decision_feats[['expected_return', 'utility', 'prob_profit']]], axis=1)
    cqf_output = pd.concat([cqf_output, price_preds], axis=1)
    cqf_output.to_csv(output_dir / 'cqf_predictions.csv', index=False)

    coverage_metrics = evaluate_cqf_coverage(cqf, top_n, quantile_preds)
    logger.info("CQF coverage metrics: %s", coverage_metrics)

    # Stress MC
    stress_cfg = StressConfig(n_paths=5000, risk_aversion=0.5, min_prob_profit=0.45, max_downside_var=0.15)
    mc = EnhancedStressMC(stress_cfg)

    mc_inputs = cqf_output.copy()
    for col in ['last_raw', 'last', 'delta', 'gamma', 'vega', 'theta', 'moneyness']:
        if col not in mc_inputs.columns and col in top_n.columns:
            mc_inputs[col] = top_n[col]

    stress_results = mc.rank_contracts(mc_inputs, spy_history=None)
    stress_results.to_csv(output_dir / 'stress_metrics.csv', index=False)

    top_trades = stress_results.head(100).copy()
    recommendations = top_trades[['contractID', 'utility_score', 'expected_pnl', 'var_95', 'cvar_95', 'prob_profit']]
    recommendations.to_csv(output_dir / 'trade_recommendations.csv', index=False)

    logger.info("Saved final trade recommendations: %s", output_dir / 'trade_recommendations.csv')


if __name__ == '__main__':
    try:
        if str(BASE_DIR / "Training") not in sys.path:
            sys.path.insert(0, str(BASE_DIR / "Training"))
        from logger import setup_logger as training_setup_logger  # type: ignore
        training_setup_logger("inference_pipeline")
    except Exception:
        pass
    main()
