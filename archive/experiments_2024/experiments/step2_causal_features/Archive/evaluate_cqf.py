#!/usr/bin/env python3
"""
Standalone evaluator for CQF models produced by cqf2.py

What it does
------------
- Loads a saved model artifact (joblib) from cqf2.py
- Loads raw eval CSV, runs the same preprocessing pipeline (via utils.preprocess_data)
- Recomputes delta-hedged PnL targets (identical to cqf2 logic, including SPY alignment)
- Generates quantile predictions with saved preprocessor + models (+ conformal adjustments)
- Computes decision features (expected_return, prob_profit, utility, etc.)
- Evaluates:
    * Pinball loss per quantile
    * Coverage per quantile + 90% interval coverage
    * Interval score (Winkler) for 90% PI
    * qCRPS (quantile-CRPS over provided quantiles; scaled partial CRPS)
    * VaR backtests (Kupiec POF & Christoffersen independence) for 5% lower tail
    * Brier score for profit-probability calibration (p = prob_profit, y = 1{PnL>0})
    * Cross-sectional Top-K ranking quality (NDCG@K, Precision@K), averaged by date
    * Correlation (Pearson/Spearman) between expected_return and realized PnL

Outputs
-------
- metrics.json          # overall metrics
- ndcg_by_date.csv      # per-date NDCG/Precision (for dates with enough contracts)
- predictions.csv       # per-row predictions + decision features + realized PnL
- console logging of summary metrics

Usage
-----
python evaluate_cqf.py \
  --model model_output/optimal_cqf.joblib \
  --eval-data year_2023_data.csv \
  --config config.yaml \
  --output-dir eval_out \
  --horizon 5 \
  --k-values 5,10,20
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# Optional SciPy for p-values; script works without it (p-values=None)
try:
    from scipy.stats import chi2
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

# Make sure we can import the user's utils.py (same one used by cqf2.py)
# Add current dir to path for convenience
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from typing import Any
import joblib

# Import existing preprocessing from the project
try:
    from utils import load_config, preprocess_data
except Exception as e:
    print("ERROR: Could not import utils.load_config/preprocess_data. "
          "Ensure utils.py is importable. Details:", e, file=sys.stderr)
    raise

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("cqf_eval")

# -------------------------- Target Computation (as in cqf2) --------------------------

def compute_delta_hedged_pnl(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """
    Delta-hedged P&L target, aligned like cqf2 (with target_date and SPY alignment).
    """
    price_col = 'last_raw' if 'last_raw' in df.columns else 'last'
    if 'contractID' not in df.columns or price_col not in df.columns:
        logger.error("Missing 'contractID' or price column for PnL calculation")
        df['target_pnl'] = np.nan
        return df

    df = df.sort_values(['contractID', 'date']).reset_index(drop=True)

    # Forward option price and target date
    g = df.groupby('contractID', sort=False)
    future_option_price = g[price_col].shift(-horizon)
    df['target_date'] = g['date'].shift(-horizon)

    spy_col = 'spy_d_close'
    if spy_col in df.columns:
        spy_daily = df[['date', spy_col]].drop_duplicates('date').sort_values('date').copy()
        spy_daily['spy_fwd'] = spy_daily[spy_col].shift(-horizon)
        df = df.merge(spy_daily[['date', 'spy_fwd']], on='date', how='left')

        option_pnl = (future_option_price - df[price_col]) / df[price_col]
        underlying_pnl = (df['spy_fwd'] - df[spy_col]) / df[spy_col]

        if 'delta' in df.columns:
            df['target_pnl'] = option_pnl + df['delta'] * (-underlying_pnl)
        else:
            logger.warning("Delta missing; using raw option return for target")
            df['target_pnl'] = option_pnl
    else:
        logger.warning("SPY column missing; using raw option return for target")
        df['target_pnl'] = (future_option_price - df[price_col]) / df[price_col]

    df['target_pnl'] = df['target_pnl'].replace([np.inf, -np.inf], np.nan)
    before = len(df)
    df = df.dropna(subset=['target_pnl']).copy()
    logger.info(f"Target computed. Dropped {before - len(df)} rows with invalid target.")
    return df


# -------------------------- Prediction Helpers --------------------------

def predict_quantiles(artifact: Dict[str, Any], df: pd.DataFrame, apply_conformal: bool = True) -> pd.DataFrame:
    """
    Use saved models + preprocessor to produce quantile predictions.
    """
    feature_names = artifact['feature_names']
    X = df[feature_names]
    preprocessor = artifact['preprocessor']
    X_scaled = preprocessor.transform(X)

    preds = {}
    # Ensure deterministic order using saved quantiles (floats like 0.05, 0.5, 0.95)
    for q in artifact['quantiles']:
        model = artifact['models'][q]
        arr = model.predict(X_scaled)
        
        # Ensure 1D array (flatten if needed)
        if arr.ndim > 1:
            arr = arr.flatten()

        if apply_conformal:
            adj = artifact.get('conformal_adjustments', {})
            if q == 0.05 and 'lower' in adj:
                arr = arr - adj['lower']
            if q == 0.95 and 'upper' in adj:
                arr = arr + adj['upper']

        preds[f"q{q:.2f}"] = arr

    # Enforce monotonicity q05 <= q50 <= q95
    if all(k in preds for k in ['q0.05', 'q0.50', 'q0.95']):
        q05 = preds['q0.05']
        q50 = np.maximum(preds['q0.50'], q05)
        q95 = np.maximum(preds['q0.95'], q50)
        q50 = np.minimum(q50, q95)
        q05 = np.minimum(q05, q50)
        preds['q0.05'] = q05
        preds['q0.50'] = q50
        preds['q0.95'] = q95

    return pd.DataFrame(preds, index=df.index)





# -------------------------- Metrics --------------------------

def pinball_loss(y: np.ndarray, yhat: np.ndarray, alpha: float) -> float:
    e = y - yhat
    return np.mean(np.maximum(alpha * e, (alpha - 1) * e))


def coverage(y: np.ndarray, yhat: np.ndarray, alpha: float) -> float:
    """For lower quantiles alpha<=0.5: P(y >= yhat) expected ~ 1-alpha.
       For upper quantiles alpha>0.5:  P(y <= yhat) expected ~ alpha.
    """
    if alpha <= 0.5:
        return float(np.mean(y >= yhat))
    else:
        return float(np.mean(y <= yhat))


def interval_coverage(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean((y >= lower) & (y <= upper)))


def interval_score(y: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float = 0.10) -> float:
    """Winkler/Interval score for central (1-alpha) interval."""
    width = upper - lower
    below = (lower - y) * (y < lower)
    above = (y - upper) * (y > upper)
    return float(np.mean(width + (2.0 / alpha) * below + (2.0 / alpha) * above))


def qcrps(y: np.ndarray, quantile_preds: Dict[float, np.ndarray]) -> float:
    """
    Quantile-CRPS: approximate CRPS by integrating pinball loss across the
    provided alphas only (partial range), then scale by the alpha span so
    metrics are comparable as you keep the same alphas set.

    CRPS(F,y) = 2 * ∫_0^1 pinball_alpha(y, Q(alpha)) d alpha
    Approximate with trapezoidal rule on provided alphas.
    """
    alphas = sorted(quantile_preds.keys())
    if len(alphas) < 2:
        return float('nan')
    pin = {a: np.mean(np.maximum(a * (y - quantile_preds[a]), (a - 1) * (y - quantile_preds[a]))) for a in alphas}
    area = 0.0
    for i in range(len(alphas) - 1):
        a0, a1 = alphas[i], alphas[i + 1]
        y0, y1 = pin[a0], pin[a1]
        area += 0.5 * (y0 + y1) * (a1 - a0)
    # scale to [alpha_min, alpha_max] span
    span = alphas[-1] - alphas[0]
    return float(2.0 * area / (span if span > 0 else 1.0))


def kupiec_pof(y: np.ndarray, var_pred: np.ndarray, alpha: float = 0.05, side: str = "lower") -> Dict[str, Optional[float]]:
    """
    Kupiec POF (unconditional coverage) test.
    side: "lower" checks y < VaR (lower tail), "upper" checks y > VaR (upper tail).
    Returns LR statistic and p-value (if SciPy available), plus breach rate.
    """
    if side == "lower":
        breaches = (y < var_pred).astype(int)
        p0 = alpha
    else:
        breaches = (y > var_pred).astype(int)
        p0 = alpha

    x = int(breaches.sum())
    n = int(len(breaches))
    if n == 0:
        return {"breaches": x, "n": n, "rate": None, "LR_pof": None, "p_value": None}

    phat = x / n if n > 0 else 0.0
    
    # Edge cases: when phat is 0 or 1, or when p0 is 0 or 1
    if phat == 0.0 or phat == 1.0 or p0 == 0.0 or p0 == 1.0:
        LR_pof = np.inf if phat != p0 else 0.0
        pval = 0.0 if SCIPY_AVAILABLE and LR_pof == np.inf else (1.0 if SCIPY_AVAILABLE else None)
    else:
        # Safe calculation avoiding log(0)
        eps = 1e-15
        num = max(eps, (1 - p0) ** (n - x) * (p0 ** x))
        den = max(eps, (1 - phat) ** (n - x) * (phat ** x))
        LR_pof = -2.0 * np.log(num / den)
        if SCIPY_AVAILABLE:
            pval = 1 - chi2.cdf(LR_pof, df=1)
        else:
            pval = None

    return {
        "breaches": x,
        "n": n,
        "rate": (x / n) if n > 0 else None,
        "LR_pof": float(LR_pof) if np.isfinite(LR_pof) else None,
        "p_value": float(pval) if (pval is not None and np.isfinite(pval)) else None
    }


def christoffersen_independence(y: np.ndarray, var_pred: np.ndarray, alpha: float = 0.05, side: str = "lower") -> Dict[str, Optional[float]]:
    """
    Christoffersen independence test for VaR exceptions.
    Build 2x2 transition matrix for exception indicator (It in {0,1}).
    """
    if side == "lower":
        I = (y < var_pred).astype(int)
    else:
        I = (y > var_pred).astype(int)

    if len(I) < 2:
        return {"LR_ind": None, "p_value": None}

    I0 = I[:-1]
    I1 = I[1:]

    n00 = int(((I0 == 0) & (I1 == 0)).sum())
    n01 = int(((I0 == 0) & (I1 == 1)).sum())
    n10 = int(((I0 == 1) & (I1 == 0)).sum())
    n11 = int(((I0 == 1) & (I1 == 1)).sum())

    n0 = n00 + n01
    n1 = n10 + n11

    # Probabilities
    pi01 = n01 / n0 if n0 > 0 else 0.0
    pi11 = n11 / n1 if n1 > 0 else 0.0
    pi = (n01 + n11) / (n0 + n1) if (n0 + n1) > 0 else 0.0

    # Likelihoods (avoid log(0) with tiny eps)
    eps = 1e-12
    L1 = ((1 - pi01 + eps) ** n00) * ((pi01 + eps) ** n01) * ((1 - pi11 + eps) ** n10) * ((pi11 + eps) ** n11)
    L0 = ((1 - pi + eps) ** (n00 + n10)) * ((pi + eps) ** (n01 + n11))

    LR_ind = -2.0 * np.log(L0 / L1) if L0 > 0 and L1 > 0 else np.inf
    if SCIPY_AVAILABLE and np.isfinite(LR_ind):
        pval = 1 - chi2.cdf(LR_ind, df=1)
    else:
        pval = None

    return {"LR_ind": float(LR_ind) if np.isfinite(LR_ind) else None,
            "p_value": float(pval) if (pval is not None and np.isfinite(pval)) else None}


# -------------------------- Ranking Metrics (cross-sectional) --------------------------

def dcg_at_k(relevances: np.ndarray, k: int) -> float:
    rel = relevances[:k]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    return float(np.sum((2**rel - 1) * discounts))


def ndcg_at_k(y_true: np.ndarray, order_pred: np.ndarray, k: int) -> float:
    """
    y_true: realized values used as "relevance" (we clip negatives to 0).
    order_pred: indices that sort items by predicted score descending.
    """
    gains = np.clip(y_true, 0.0, None)
    ideal_order = np.argsort(-gains)
    dcg = dcg_at_k(gains[order_pred], k)
    idcg = dcg_at_k(gains[ideal_order], k)
    return float(dcg / idcg) if idcg > 0 else np.nan


def precision_at_k(y_true: np.ndarray, order_pred: np.ndarray, k: int) -> float:
    """Binary label = 1 if realized PnL > 0."""
    labels = (y_true > 0).astype(int)
    topk = labels[order_pred][:k]
    return float(np.mean(topk)) if k > 0 else np.nan


def cross_sectional_rank_metrics(df: pd.DataFrame, k_list: List[int]) -> pd.DataFrame:
    """
    Compute per-date NDCG@K and Precision@K using expected_return for ranking.
    Returns per-date metrics and logs the overall averages.
    """
    if 'date' not in df.columns:
        logger.warning("No 'date' column; computing aggregate (single group) ranking metrics.")
        df = df.assign(date='ALL')

    rows = []
    for date, g in df.groupby('date'):
        if len(g) < max(k_list):
            continue
        y = g['target_pnl'].values
        # Sorting index by predicted expected_return (desc)
        order = np.argsort(-g['expected_return'].values)

        row = {'date': date, 'n': len(g)}
        for k in k_list:
            row[f'ndcg@{k}'] = ndcg_at_k(y, order, k)
            row[f'precision@{k}'] = precision_at_k(y, order, k)
        rows.append(row)

    out = pd.DataFrame(rows).sort_values('date')
    return out


# -------------------------- Correlations --------------------------

def correlations(df: pd.DataFrame) -> Dict[str, float]:
    out = {}
    if 'expected_return' in df.columns and 'target_pnl' in df.columns:
        x = df['expected_return'].values
        y = df['target_pnl'].values
        if len(x) > 2 and np.std(x) > 0 and np.std(y) > 0:
            out['pearson_exp_vs_real'] = float(np.corrcoef(x, y)[0, 1])
            # Spearman
            rx = pd.Series(x).rank().values
            ry = pd.Series(y).rank().values
            out['spearman_exp_vs_real'] = float(np.corrcoef(rx, ry)[0, 1])
        else:
            out['pearson_exp_vs_real'] = np.nan
            out['spearman_exp_vs_real'] = np.nan
    return out


# -------------------------- Brier score --------------------------

def brier_score(prob: np.ndarray, y: np.ndarray) -> float:
    """Brier on binary y in {0,1}"""
    return float(np.mean((prob - y) ** 2))


# -------------------------- Main --------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved CQF model artifact.")
    parser.add_argument("--model", required=True, help="Path to joblib artifact saved by cqf2.py")
    parser.add_argument("--eval-data", required=True, help="Raw CSV for evaluation")
    parser.add_argument("--config", default="config.yaml", help="Config YAML for preprocess_data")
    parser.add_argument("--output-dir", default="eval_out", help="Directory for outputs")
    parser.add_argument("--horizon", type=int, default=5, help="Prediction horizon used for targets")
    parser.add_argument("--k-values", default="5,10,20", help="Comma-separated K values for NDCG/Precision")
    parser.add_argument("--apply-conformal", action="store_true", help="Apply conformal adjustments when predicting")
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load artifact
    logger.info(f"Loading model artifact: {args.model}")
    artifact = joblib.load(args.model)
    required_keys = ['models', 'preprocessor', 'feature_names', 'quantiles']
    for k in required_keys:
        if k not in artifact:
            raise ValueError(f"Model artifact missing key: {k}")

    # Load and preprocess eval data
    logger.info(f"Loading eval data: {args.eval_data}")
    cfg = load_config(args.config)
    raw = pd.read_csv(args.eval_data, low_memory=False)
    raw['date'] = pd.to_datetime(raw['date'], errors='coerce')
    raw = raw.dropna(subset=['date']).copy()

    # Friendly contractID handling (as in cqf2 main)
    if 'contractID' not in raw.columns:
        if 'contract_id' in raw.columns:
            raw = raw.rename(columns={'contract_id': 'contractID'})
        elif 'option_symbol' in raw.columns:
            raw = raw.rename(columns={'option_symbol': 'contractID'})
    raw['contractID'] = raw['contractID'].astype(str)

    raw = raw.sort_values(['date', 'contractID']).reset_index(drop=True)

    # Project preprocessing (feature engineering)
    proc, _ = preprocess_data(raw, cfg, scaler=None)

    # Keep only rows that have all required features BEFORE computing targets
    features = artifact['feature_names']
    missing_feats = [f for f in features if f not in proc.columns]
    if missing_feats:
        raise ValueError(f"Missing features in eval data after preprocessing: {missing_feats}")

    # Remove rows with missing features to ensure alignment
    proc = proc.dropna(subset=features)
    
    # Compute target (delta-hedged PnL), same as cqf2
    proc_with_targets = compute_delta_hedged_pnl(proc, horizon=args.horizon)

    # Predict quantiles ONLY on rows that have valid targets
    qdf = predict_quantiles(artifact, proc_with_targets, apply_conformal=args.apply_conformal)

    # Decision features on quantiles (no separate DataFrame, add columns to qdf)
    if all(c in qdf.columns for c in ['q0.05', 'q0.50', 'q0.95']):
        q05, q50, q95 = qdf['q0.05'], qdf['q0.50'], qdf['q0.95']
        qdf['expected_return'] = (q05 + 4 * q50 + q95) / 6.0
        qdf['downside_risk'] = np.abs(np.minimum(q05, 0))
        qdf['upside_potential'] = np.maximum(q95, 0)
        qdf['uncertainty'] = q95 - q05

        prob_profit_raw = np.where(
            q95 <= 0, 0.0,
            np.where(q05 >= 0, 1.0, 0.5 + 0.45 * (q50 / (q95 - q05 + 1e-8)))
        )
        qdf['prob_profit'] = np.clip(prob_profit_raw, 0.0, 1.0)
        risk_penalty = 0.5
        qdf['utility'] = qdf['expected_return'] - risk_penalty * qdf['downside_risk']

    # Combine with base data - use shared index
    base_cols = ['contractID', 'date', 'target_pnl']
    preds = proc_with_targets[base_cols].copy()
    for col in qdf.columns:
        preds[col] = qdf[col]

    # ----------- Scalar forecast metrics -----------
    y = preds['target_pnl'].values
    metrics = {}

    # Pinball + coverage per quantile
    quantile_map = {}
    for q in artifact['quantiles']:
        key = f"q{q:.2f}"
        arr = preds[key].values
        # Ensure 1D array (flatten if needed for metrics)
        if arr.ndim > 1:
            arr = arr.flatten()
        
        quantile_map[q] = arr
        metrics[f'pinball_{key}'] = pinball_loss(y, quantile_map[q], q)
        metrics[f'coverage_{key}'] = coverage(y, quantile_map[q], q)

    # 90% interval metrics (if 0.05 / 0.95 available)
    if (0.05 in quantile_map) and (0.95 in quantile_map):
        lower = quantile_map[0.05]
        upper = quantile_map[0.95]
        metrics['interval90_coverage'] = interval_coverage(y, lower, upper)
        metrics['interval90_score'] = interval_score(y, lower, upper, alpha=0.10)

        # VaR backtests for lower tail
        kup = kupiec_pof(y, lower, alpha=0.05, side="lower")
        chrst = christoffersen_independence(y, lower, alpha=0.05, side="lower")
        for k, v in kup.items():
            metrics[f'kupiec_{k}'] = v
        for k, v in chrst.items():
            metrics[f'christoffersen_{k}'] = v

        # Realized CVaR_95 (severity of bad outcomes)
        breaches = y < lower
        if breaches.any():
            metrics['realized_cvar_95'] = float(np.mean(y[breaches]))
        else:
            metrics['realized_cvar_95'] = np.nan

    # qCRPS on provided quantiles (scaled partial CRPS)
    metrics['qcrps'] = qcrps(y, quantile_map)

    # Brier score for profit-prob calibration
    if 'prob_profit' in preds.columns:
        metrics['brier_prob_profit'] = brier_score(preds['prob_profit'].values, (y > 0).astype(int))
    else:
        metrics['brier_prob_profit'] = np.nan

    # ----------- Ranking metrics (cross-sectional by date) -----------
    k_list = [int(x) for x in args.k_values.split(',') if x.strip()]
    rank_df = cross_sectional_rank_metrics(preds, k_list=k_list)
    rank_df_out = outdir / "ndcg_by_date.csv"
    rank_df.to_csv(rank_df_out, index=False)

    # Aggregate ranking metrics
    for k in k_list:
        metrics[f'avg_ndcg@{k}'] = float(rank_df[f'ndcg@{k}'].mean()) if f'ndcg@{k}' in rank_df else np.nan
        metrics[f'avg_precision@{k}'] = float(rank_df[f'precision@{k}'].mean()) if f'precision@{k}' in rank_df else np.nan

    # Correlations
    metrics.update(correlations(preds))

    # Save predictions
    preds_out = outdir / "predictions.csv"
    preds.to_csv(preds_out, index=False)

    # Save metrics
    metrics_out = outdir / "metrics.json"
    with open(metrics_out, "w") as f:
        json.dump(metrics, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating,)) else o)

    # Console summary
    logger.info("=== Forecast Accuracy ===")
    for q in artifact['quantiles']:
        key = f"q{q:.2f}"
        logger.info(f"Pinball {key}: {metrics[f'pinball_{key}']:.6f} | Coverage {key}: {metrics[f'coverage_{key}']:.3f}")

    if 'interval90_coverage' in metrics:
        logger.info(f"Interval90 Coverage: {metrics['interval90_coverage']:.3f} | Interval Score: {metrics['interval90_score']:.6f}")
        logger.info(f"Kupiec breaches: {metrics.get('kupiec_breaches')} / {metrics.get('kupiec_n')}, rate={metrics.get('kupiec_rate')}")
        if SCIPY_AVAILABLE:
            logger.info(f"Kupiec LR={metrics.get('kupiec_LR_pof')} p={metrics.get('kupiec_p_value')}")
            logger.info(f"Christoffersen LR={metrics.get('christoffersen_LR_ind')} p={metrics.get('christoffersen_p_value')}")

    logger.info(f"qCRPS (scaled-partial): {metrics['qcrps']:.6f}")
    logger.info(f"Brier(prob_profit): {metrics['brier_prob_profit']:.6f}")

    logger.info("=== Ranking Quality (cross-sectional by date, using expected_return) ===")
    for k in k_list:
        logger.info(f"avg NDCG@{k}: {metrics.get(f'avg_ndcg@{k}', np.nan)}  |  avg Precision@{k}: {metrics.get(f'avg_precision@{k}', np.nan)}")

    logger.info("=== Correlations ===")
    logger.info(f"Pearson(expected vs realized): {metrics.get('pearson_exp_vs_real')}")
    logger.info(f"Spearman(expected vs realized): {metrics.get('spearman_exp_vs_real')}")

    logger.info(f"Saved: {metrics_out}")
    logger.info(f"Saved: {rank_df_out}")
    logger.info(f"Saved: {preds_out}")
    logger.info("Done.")

if __name__ == "__main__":
    main()
