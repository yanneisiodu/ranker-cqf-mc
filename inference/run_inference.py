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
import inference.llm_stress as stress_basic  # noqa: E402
import inference.llm_stress2 as stress_agent  # noqa: E402
from inference.llm_client import OpenAILLMClient  # noqa: E402


K_VALUES = [1, 5, 10, 20]

logger = logging.getLogger("inference_pipeline")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _load_env_value(path: Path, key: str) -> Optional[str]:
    """Return value for `key` from a simple KEY=VALUE .env file if present."""
    if not path.exists():
        return None
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue
        if '=' not in line:
            continue
        name, value = line.split('=', 1)
        if name.strip() == key:
            return value.strip().strip("\"'")
    return None


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
    if y_true.size == 0:
        logger.warning("Coverage evaluation skipped: no rows with target_pnl available in the last %d day(s).", horizon_days)
        return {}
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
    p.add_argument('--stress-mode', choices=['mc', 'llm', 'shadow'], default='mc',
                   help="Which stress engine output to persist")
    p.add_argument('--llm-provider', choices=['openai', 'none'], default='openai',
                   help="LLM backend to use when stress-mode requires it")
    p.add_argument('--llm-model', default='gpt-4.1-mini',
                   help="LLM model identifier (provider dependent)")
    p.add_argument('--llm-api-key', default=None,
                   help="Optional API key (otherwise reads OPENAI_API_KEY env var)")
    p.add_argument('--llm-log-scenarios', action='store_true',
                   help="Include per-underlying scenario logs in LLM outputs")
    p.add_argument('--llm-engine', choices=['basic', 'agent'], default='basic',
                   help="LLM stress module to use (basic two-piece vs ultra agent)")
    p.add_argument('--min-prob-profit', type=float, default=0.45,
                   help="Minimum probability of profit threshold for stress gating")
    p.add_argument('--llm-max-groups', type=int, default=10,
                   help="Maximum number of LLM prompts; fallback to MC if exceeded")
    p.add_argument('--llm-max-contracts', type=int, default=20,
                   help="Upper bound on contracts evaluated when LLM mode is active")
    args = p.parse_args()

    if args.llm_api_key is None and args.llm_provider == 'openai':
        env_key = _load_env_value(Path(__file__).resolve().parent / '.env', 'OPENAI_API_KEY')
        if env_key:
            args.llm_api_key = env_key

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading raw data: %s", args.raw_data)
    raw_df = pd.read_csv(args.raw_data, low_memory=False)
    raw_df['date'] = pd.to_datetime(raw_df['date'], errors='coerce')
    raw_spy_cols = ['date'] + [c for c in raw_df.columns if c.startswith('spy_')]
    raw_spy_frame = raw_df[raw_spy_cols].drop_duplicates('date') if len(raw_spy_cols) > 1 else None
    spy_history_full = None
    if 'date' in raw_df.columns and 'spy_d_close' in raw_df.columns:
        spy_history_full = (
            raw_df[['date', 'spy_d_close']]
            .dropna()
            .drop_duplicates(subset='date')
            .rename(columns={'spy_d_close': 'close'})
        )
        spy_history_full['date'] = pd.to_datetime(spy_history_full['date'], errors='coerce')
        spy_history_full = spy_history_full.dropna(subset=['date'])

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

    effective_top_n = args.top_n
    if args.stress_mode in ('llm', 'shadow') and args.llm_max_contracts is not None:
        effective_top_n = min(args.top_n, args.llm_max_contracts)
        if effective_top_n < args.top_n:
            logger.info("Reducing top-N from %d to %d due to LLM max contracts setting.", args.top_n, effective_top_n)

    top_n = target_df.nlargest(effective_top_n, 'ranker_score').copy()

    import re

    def _extract_root(cid: str) -> str:
        match = re.match(r"[A-Za-z]+", str(cid))
        return match.group(0) if match else str(cid)

    if 'underlying' not in top_n.columns:
        top_n['underlying'] = top_n['contractID'].map(_extract_root)
    if 'underlying_symbol' not in top_n.columns:
        top_n['underlying_symbol'] = top_n['contractID'].map(_extract_root)

    if raw_spy_frame is not None:
        top_n = top_n.merge(raw_spy_frame, on='date', how='left', suffixes=('', '_raw'))

    # Enrich with SPY/underlying columns so the stress layer can calibrate shocks
    spy_cols = [
        col for col in processed_df.columns
        if col.startswith('spy_') or col in {'underlying_price', 'spot', 'spot_price', 'underlying_last'}
    ]
    extra_cols = [c for c in spy_cols if c not in top_n.columns]
    if extra_cols:
        enrich_cols = ['contractID', 'date'] + extra_cols
        top_n = top_n.merge(
            processed_df[enrich_cols],
            on=['contractID', 'date'],
            how='left'
        )
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
    lookback = 252
    if spy_history_full is not None and not spy_history_full.empty:
        lookback = min(lookback, max(30, len(spy_history_full) - 1))
    stress_cfg = StressConfig(n_paths=5000, risk_aversion=0.5, min_prob_profit=args.min_prob_profit, max_downside_var=0.15, lookback_days=lookback)
    mc = EnhancedStressMC(stress_cfg)

    stress_module = stress_agent if args.llm_engine == 'agent' else stress_basic

    ScenarioConfig = stress_module.ScenarioConfig
    LLMStressConfig = stress_module.LLMStressConfig
    ScenarioValidator = stress_module.ScenarioValidator
    ScenarioEngine = stress_module.ScenarioEngine
    LLMStressEngine = stress_module.LLMStressEngine

    llm_scenario_config = ScenarioConfig(group_key='underlying')
    llm_validator = ScenarioValidator(llm_scenario_config)
    llm_stress_config = LLMStressConfig(
        risk_aversion=stress_cfg.risk_aversion,
        min_prob_profit=args.min_prob_profit,
        max_downside_var=stress_cfg.max_downside_var,
        skew_bonus=0.0,
        horizon_days=stress_cfg.horizon_days_for_theta,
    )
    llm_client = None
    if args.stress_mode in ('llm', 'shadow') and args.llm_provider == 'openai':
        try:
            # GPT-5 doesn't support temperature parameter
            request_kwargs = {} if args.llm_model.startswith('gpt-5') else {'temperature': 0}
            llm_client = OpenAILLMClient(api_key=args.llm_api_key, model=args.llm_model, request_kwargs=request_kwargs)
        except Exception as exc:
            logger.warning('LLM client initialisation failed (%s). Falling back to MC scenarios.', exc)
            llm_client = None

    llm_engine = LLMStressEngine(ScenarioEngine(llm_scenario_config, llm_client=llm_client), llm_validator, llm_stress_config)

    mc_inputs = cqf_output.copy()
    extra_cols = [
        'last_raw', 'last', 'delta', 'gamma', 'vega', 'theta', 'moneyness',
        'underlying', 'underlying_symbol', 'underlying_price',
        'spy_d_close', 'spy_d_open', 'spy_d_high', 'spy_d_low',
        'spy_d_close_raw', 'spy_d_open_raw', 'spy_d_high_raw', 'spy_d_low_raw'
    ]
    for col in extra_cols:
        if col not in mc_inputs.columns and col in top_n.columns:
            mc_inputs[col] = top_n[col]

    stress_mode = args.stress_mode

    spy_history = None
    close_col = 'spy_d_close_raw' if 'spy_d_close_raw' in mc_inputs.columns else 'spy_d_close'
    if 'date' in mc_inputs.columns and close_col in mc_inputs.columns:
        spy_history = (
            mc_inputs[['date', close_col]]
            .dropna()
            .drop_duplicates(subset='date')
            .rename(columns={close_col: 'close'})
        )
        spy_history['date'] = pd.to_datetime(spy_history['date'], errors='coerce')
        spy_history = spy_history.dropna(subset=['date'])

    # Use spy_history if it exists and is not empty, otherwise use spy_history_full
    selected_spy_history = spy_history if spy_history is not None and not spy_history.empty else spy_history_full
    stress_results_mc = mc.rank_contracts(mc_inputs, spy_history=selected_spy_history)

    llm_results = None
    if stress_mode in ('llm', 'shadow'):
        regime_series = top_n.get('vol_severity') if 'vol_severity' in top_n.columns else None
        regime_label = 'high_vol' if regime_series is not None and float(regime_series.fillna(0).mean()) > 2.0 else 'normal'
        llm_context = {
            'regime': regime_label,
            'date': str(top_n['date'].max()) if 'date' in top_n.columns else None,
            'analogs': '',
            'use_agent': args.llm_engine == 'agent',
        }
        group_key = llm_scenario_config.group_key if llm_scenario_config.group_key in mc_inputs.columns else None
        if group_key is None:
            for cand in ('underlying', 'underlying_symbol', 'ticker', 'root', 'contractID'):
                if cand in mc_inputs.columns:
                    group_key = cand
                    break
            else:
                group_key = 'contractID'
        num_groups = mc_inputs[group_key].nunique()
        logger.info("LLM stress engine will request %d prompt(s) grouped by '%s'", int(num_groups), group_key)
        if args.llm_max_groups is not None and num_groups > args.llm_max_groups:
            logger.warning(
                "Skipping LLM stress: grouping produced %d prompts which exceeds the cap of %d.",
                int(num_groups), args.llm_max_groups
            )
        else:
            llm_results = llm_engine.evaluate(mc_inputs, context=llm_context, store_scenarios=args.llm_log_scenarios)
            llm_results = llm_results.merge(mc_inputs[['contractID']], on='contractID', how='left')
            llm_results = llm_results.sort_values('utility_score', ascending=False).reset_index(drop=True)
            llm_results['rank'] = np.arange(1, len(llm_results) + 1)

    if stress_mode == 'llm' and llm_results is not None:
        stress_results = llm_results.copy()
    else:
        stress_results = stress_results_mc.copy()

    if llm_results is not None:
        llm_path = output_dir / 'stress_metrics_llm.csv'
        llm_results.to_csv(llm_path, index=False)
        logger.info('Saved LLM stress metrics: %s', llm_path)

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
