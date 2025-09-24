#!/usr/bin/env python3
"""
Rolling CV Optuna Tuner for CQF - Institutional-Grade Hyperparameter Optimization

Advanced features:
- Rolling time-based cross-validation (3 folds) to prevent regime overfitting  
- Composite objective: pinball + coverage penalties + 2x VaR tail penalty
- Spearman rank correlation bonus for trading signal preservation
- Leak-proof TRAIN→TUNE→CALIB→EVAL per fold with guard bands

Usage:
  python optuna_tune_cqf.py \
    --train-data year_2022_data.csv \
    --config config.yaml \
    --trials 40 \
    --folds 3 \
    --study-name cqf_rolling_cv \
    --out-best model_output/rolling_cv_best_params.json
"""

import argparse, logging, warnings
from pathlib import Path
from datetime import timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.metrics import mean_pinball_loss
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Reuse your pipeline & features
from cqf2 import OptimalCQF, load_and_prepare_data

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("optuna_cqf")


# -------------------------------
# Helpers (metrics & folds)
# -------------------------------
def _pinballs(y, preds: Dict[str, np.ndarray]) -> Dict[str, float]:
    return {
        "q005": mean_pinball_loss(y, preds["q0.05"], alpha=0.05),
        "q050": mean_pinball_loss(y, preds["q0.50"], alpha=0.50),
        "q095": mean_pinball_loss(y, preds["q0.95"], alpha=0.95),
    }

def _coverage(y, preds: Dict[str, np.ndarray]) -> Dict[str, float]:
    cov = {}
    cov["q0.05"] = float(np.mean(y >= preds["q0.05"]))           # should be ~0.95
    cov["q0.50"] = float(np.mean(y <= preds["q0.50"]))           # median: ~0.50 (upper-side defn)
    cov["q0.95"] = float(np.mean(y <= preds["q0.95"]))           # should be ~0.95
    cov["interval90"] = float(np.mean((y >= preds["q0.05"]) & (y <= preds["q0.95"])))
    return cov

def _expected_return(preds: Dict[str, np.ndarray]) -> np.ndarray:
    # Simpson's rule (same as in cqf2.py)
    return (preds["q0.05"] + 4.0 * preds["q0.50"] + preds["q0.95"]) / 6.0

def _spearman_rank(y, preds):
    try:
        er = _expected_return(preds)
        r, _ = spearmanr(er, y)
        return 0.0 if np.isnan(r) else float(r)
    except Exception:
        return 0.0

def _composite_objective(y, preds) -> float:
    """
    Lower is better.
    Components:
      - avg pinball loss (primary)
      - coverage penalties for q5, median, q95, and interval90
      - 2x penalty for q05 undercoverage (VaR problem focus)
      - small reward for rank quality (Spearman)
    """
    pb = _pinballs(y, preds)
    cov = _coverage(y, preds)
    
    # Pinball
    pb_avg = (pb["q005"] + pb["q050"] + pb["q095"]) / 3.0

    # Coverage target errors
    err_q05 = abs(cov["q0.05"] - 0.95)
    # For median we computed P{Y <= q50}; target is 0.50
    err_q50 = abs(cov["q0.50"] - 0.50)
    err_q95 = abs(cov["q0.95"] - 0.95)
    err_int = abs(cov["interval90"] - 0.90)

    # Heavier penalty on lower-tail undercoverage (dangerous in options)
    under_q05 = max(0.95 - cov["q0.05"], 0.0)
    tail_pen = 2.0 * under_q05  # 2x penalty for VaR breaches

    # Rank bonus
    rank = _spearman_rank(y, preds)
    rank_bonus = 0.05 * max(rank, 0.0)  # only reward positive rank

    # Weights (risk-first approach)
    cov_w = 0.5
    int_w = 0.7

    composite = (
        pb_avg
        + cov_w * (err_q05 + err_q50 + err_q95)
        + int_w * err_int
        + tail_pen
        - rank_bonus
    )
    return float(composite)

def build_preprocessor() -> Pipeline:
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])

def feature_subset(df: pd.DataFrame) -> List[str]:
    # Mirror cqf2.py feature selection
    cqf_features = [
        "delta","gamma","theta","vega",
        "price_roll_mean_5","price_roll_mean_20",
        "price_roll_std_5","price_roll_std_20",
        "price_roll_zscore_5","price_roll_zscore_20",
        "iv_roll_mean_5","iv_roll_mean_20",
        "moneyness","days_to_exp","implied_volatility",
        "mispricing_ratio","risk_adjusted_signal",
        "relative_spread","option_volume_oi_ratio",
        "spy_d_close","vix_d_close","iv_vix_ratio","spy_momentum",
    ]
    avail = [c for c in cqf_features if c in df.columns]
    avail = [c for c in avail if df[c].notna().sum() > 0.5 * len(df)]
    return avail

def make_folds(df: pd.DataFrame,
               horizon: int,
               n_folds: int = 3,
               tune_days: int = 30,
               calib_days: int = 30,
               eval_days: int = 30) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Rolling time-based folds:
      [-------- TRAIN --------][-- TUNE --][-- CALIB --][-- EVAL --]  x n_folds
    Guard bands are handled by keeping the TUNE/CALIB/EVAL windows strictly after TRAIN.
    """
    d = df.sort_values("date").reset_index(drop=True)
    max_date = d["date"].max()
    folds = []
    for i in range(n_folds):
        eval_end = max_date - timedelta(days=i * eval_days)
        eval_start = eval_end - timedelta(days=eval_days)

        calib_end = eval_start
        calib_start = calib_end - timedelta(days=calib_days)

        tune_end = calib_start
        tune_start = tune_end - timedelta(days=tune_days)

        # TRAIN ends strictly before TUNE start minus guard
        guard = timedelta(days=horizon)
        train_end = tune_start - guard

        train = d[d["date"] < train_end]
        tune = d[(d["date"] >= tune_start) & (d["date"] < tune_end)]
        calib = d[(d["date"] >= calib_start) & (d["date"] < calib_end)]
        eval_ = d[(d["date"] >= eval_start) & (d["date"] < eval_end)]

        if min(map(len, [train, tune, calib, eval_])) < 200:
            # Skip tiny folds
            continue
        folds.append((train.copy(), tune.copy(), calib.copy(), eval_.copy()))
    return folds

def train_quantile_models(X_tr, y_tr, X_tu, y_tu, params, quantiles=(0.05, 0.5, 0.95), seed=42):
    models = {}
    for q in quantiles:
        model = xgb.XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=q,
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            min_child_weight=params["min_child_weight"],
            gamma=params["gamma"],
            reg_alpha=params["reg_alpha"],
            reg_lambda=params["reg_lambda"],
            tree_method="hist",
            n_jobs=-1,
            random_state=seed,
            early_stopping_rounds=30,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_tu, y_tu)], verbose=False)
        models[q] = model
    return models

def conformal_adjustments(models: Dict[float, xgb.XGBRegressor],
                          X_cal: np.ndarray,
                          y_cal: np.ndarray,
                          alpha: float = 0.10) -> Dict[str, float]:
    adj = {}
    if 0.05 in models and 0.95 in models:
        lp = models[0.05].predict(X_cal)
        up = models[0.95].predict(X_cal)
        lower_scores = lp - y_cal
        upper_scores = y_cal - up
        n = len(y_cal)
        ql = np.ceil((n + 1) * (1 - alpha)) / n
        adj["lower"] = float(np.quantile(lower_scores, ql))
        adj["upper"] = float(np.quantile(upper_scores, ql))
    return adj

def predict_quantiles(models, X, adj: Dict[str, float]):
    out = {}
    for q, m in models.items():
        p = m.predict(X)
        if q == 0.05 and "lower" in adj:
            p = p - adj["lower"]
        if q == 0.95 and "upper" in adj:
            p = p + adj["upper"]
        out[f"q{q:.2f}"] = p
    # enforce monotonicity
    if all(k in out for k in ["q0.05", "q0.50", "q0.95"]):
        q05, q50, q95 = out["q0.05"], out["q0.50"], out["q0.95"]
        q50 = np.maximum(q50, q05)
        q95 = np.maximum(q95, q50)
        q50 = np.minimum(q50, q95)
        q05 = np.minimum(q05, q50)
        out["q0.05"], out["q0.50"], out["q0.95"] = q05, q50, q95
    return out

# -------------------------------
# Optuna objective
# -------------------------------
def build_objective(df: pd.DataFrame, horizon: int, n_folds: int):
    folds = make_folds(df, horizon=horizon, n_folds=n_folds, tune_days=30, calib_days=30, eval_days=30)
    features = feature_subset(df)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "n_estimators": trial.suggest_int("n_estimators", 250, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.01, 10.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        }

        composite_scores = []

        for (train, tune, calib, eval_) in folds:
            # Preprocess (fit on TRAIN only)
            pre = build_preprocessor()
            X_tr = pre.fit_transform(train[features]); y_tr = train["target_pnl"].values
            X_tu = pre.transform(tune[features]);     y_tu = tune["target_pnl"].values
            X_ca = pre.transform(calib[features]);    y_ca = calib["target_pnl"].values
            X_ev = pre.transform(eval_[features]);    y_ev = eval_["target_pnl"].values

            # Train quantile models with early stopping (on TUNE)
            models = train_quantile_models(X_tr, y_tr, X_tu, y_tu, params, seed=42)

            # Conformal on CALIB
            adj = conformal_adjustments(models, X_ca, y_ca, alpha=0.10)

            # Predict on EVAL
            preds = predict_quantiles(models, X_ev, adj)

            # Score
            score = _composite_objective(y_ev, preds)
            composite_scores.append(score)

            # Pruning hint
            trial.report(np.mean(composite_scores), step=len(composite_scores))
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(composite_scores))

    return objective, features


# -------------------------------
# Main
# -------------------------------
def main():
    ap = argparse.ArgumentParser(description="Optuna hyperparameter tuning for CQF (quantile XGBoost + conformal)")
    ap.add_argument("--train-data", required=True, help="CSV used for tuning (same format as cqf2)")
    ap.add_argument("--config", default="config.yaml", help="config.yaml used by utils/preprocess")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--study-name", default="cqf_optuna")
    ap.add_argument("--storage", default=None, help="e.g., sqlite:///cqf_optuna.db")
    ap.add_argument("--out-best", default="best_params.json")
    args = ap.parse_args()

    # Load & preprocess with your existing code (includes causal features, etc.)
    data = load_and_prepare_data(args.train_data, args.config)

    # Compute targets with your current definition
    cqf_tmp = OptimalCQF(horizon=args.horizon)
    data = cqf_tmp.calculate_delta_hedged_pnl(data, args.horizon)

    # Build objective
    objective, features = build_objective(data, horizon=args.horizon, n_folds=args.folds)

    # Study (TPE + pruning)
    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=10)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=1)
    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=args.storage,
        load_if_exists=bool(args.storage),
    )
    log.info(f"Starting optimization: trials={args.trials}, folds={args.folds}, horizon={args.horizon}")
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    log.info(f"Best value: {study.best_value:.6f}")
    log.info(f"Best params: {study.best_params}")

    # Save best params
    out = Path(args.out_best)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.Series(study.best_params).to_json(out, indent=2)
    log.info(f"Saved best params to {out}")

if __name__ == "__main__":
    main()
