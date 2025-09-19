#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calibrated Quantile Forecasting (CQF) for 3d forward returns.

- Trains q05/q50/q95 quantile models on your Step-2 preprocessed feature table
- GroupKFold by 'date' to avoid cross-day leakage
- Coverage calibration via bucketed residual-quantile offsets
- Optional Step-2 ranker score as a conditioning feature
- Deterministic decision features at inference

Usage:
  # Train
  python cqf.py train \
    --train-csv ../../year_2022_data.csv \
    --config-file config.yaml \
    --utils-module utils \
    --horizon 3 \
    --artifact model_output/cqf_2022_q05_q50_q95.joblib \
    --features-pkl experiments/step2_causal_features/model_output/xgb_feature_names_2022_2022_*.pkl \
    --step2-model experiments/step2_causal_features/model_output/xgboost_ranker_2022_2022_*.joblib \
    --step2-features experiments/step2_causal_features/model_output/xgb_feature_names_2022_2022_*.pkl \
    --bucket-cols moneyness_bucket,tenor_bucket

  # Predict on OOS
  python cqf.py predict \
    --eval-csv ../../year_2023_data.csv \
    --config-file config.yaml \
    --utils-module utils \
    --artifact model_output/cqf_2022_q05_q50_q95.joblib \
    --horizon 3 \
    --out-csv model_output/cqf_preds_2023.csv \
    --hurdle 0.02 \
    --stress-gap -0.03 \
    --add-step2-score
"""
from __future__ import annotations
import argparse, logging, sys, os, math, json, warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import joblib
from dataclasses import dataclass, asdict, field
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_pinball_loss
from sklearn.ensemble import GradientBoostingRegressor

# Optional deps (graceful fallback if unavailable)
try:
    from sklearn.isotonic import IsotonicRegression  # type: ignore
except Exception:  # pragma: no cover
    IsotonicRegression = None  # type: ignore

try:
    import xgboost as xgb  # type: ignore
except Exception:  # pragma: no cover
    xgb = None  # type: ignore

# --------------------------- Logging -----------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOG = logging.getLogger("cqf")

# --------------------------- Utilities ---------------------------------------
def import_utils(module_name: str):
    """Import your project utils (must expose load_config, preprocess_data)."""
    try:
        mod = __import__(module_name, fromlist=["*"])
        if not hasattr(mod, "load_config") or not hasattr(mod, "preprocess_data"):
            raise ImportError("utils module must expose load_config & preprocess_data")
        return mod
    except Exception as e:
        raise RuntimeError(f"Failed to import utils module '{module_name}': {e}")

def compute_forward_return(df: pd.DataFrame, horizon: int, price_col: str = "px_eval") -> pd.Series:
    """Purely forward, contract-wise return. No lookahead leakage."""
    if "contractID" not in df.columns:
        raise ValueError("contractID column required")
    rets = np.full(len(df), np.nan, dtype=np.float64)
    for _, gdf in df.groupby("contractID", sort=False):
        s = gdf[price_col].to_numpy(dtype=float)
        fwd = np.roll(s, -horizon); fwd[-horizon:] = np.nan
        ok = (s > 0) & np.isfinite(s) & np.isfinite(fwd)
        r = np.full_like(s, np.nan, dtype=float)
        r[ok] = (fwd[ok] - s[ok]) / s[ok]
        rets[gdf.index] = r
    return pd.Series(rets, index=df.index, name=f"fwd_ret_{horizon}d")

def build_eval_price(df: pd.DataFrame) -> pd.Series:
    """Use mid if sane, else last."""
    bid = df.get("bid"); ask = df.get("ask"); last = df.get("last")
    if bid is None or ask is None:
        return last.copy()
    valid = (bid > 0) & (ask > 0) & (bid <= ask)
    mid = (bid + ask) / 2.0
    px = pd.Series(np.where(valid, mid, np.nan), index=df.index)
    return px.where(np.isfinite(px), last)

def pick_feature_cols(df: pd.DataFrame, features_pkl: Optional[str]) -> List[str]:
    ban = {"contractID","date","asof_date","symbol","underlying","option_symbol",
           "fwd_ret_3d","fwd_ret_5d","fwd_ret_10d","px_eval","target","target_relevance_int"}
    if features_pkl and Path(features_pkl).exists():
        fs = joblib.load(features_pkl)
        # Keep intersection in case preprocessing changed
        return [c for c in fs if c in df.columns and c not in ban]
    # Fallback: heuristics
    return [c for c in df.columns if c not in ban and np.issubdtype(df[c].dtype, np.number)]

def add_step2_score_if_given(df_proc: pd.DataFrame,
                             step2_model_path: Optional[str],
                             step2_features_pkl: Optional[str]) -> Optional[pd.Series]:
    if not step2_model_path or not Path(step2_model_path).exists():
        return None
    model = joblib.load(step2_model_path)
    feat_names = None
    if step2_features_pkl and Path(step2_features_pkl).exists():
        feat_names = joblib.load(step2_features_pkl)
        feat_names = [c for c in feat_names if c in df_proc.columns]
    else:
        feat_names = [c for c in df_proc.columns if np.issubdtype(df_proc[c].dtype, np.number)]
    X = df_proc[feat_names].copy()
    # XGB rankers sometimes choke on object dtypes
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = X[col].astype(str)
    s = model.predict(X)
    # flip sign if your ranker scores "lower is better" — adjust if needed:
    score_pos = -s
    return pd.Series(score_pos, index=df_proc.index, name="step2_score_pos")

# --------------------------- Data classes ------------------------------------
@dataclass
class QuantileSpec:
    alpha: float
    n_estimators: int = 400
    max_depth: int = 3
    learning_rate: float = 0.05
    subsample: float = 1.0
    min_samples_leaf: int = 20

@dataclass
class CQFArtifact:
    version: str
    horizon: int
    price_col: str
    feature_cols: List[str]
    bucket_cols: List[str]
    quantiles: List[float]
    models: Dict[str, GradientBoostingRegressor]
    calib_offsets: Dict[str, Dict[str, float]]  # {qkey: {bucket_key: delta}}
    metadata: Dict[str, str]

# --------------------------- Core CQF ----------------------------------------
class CQF:
    def __init__(self,
                 q_specs: List[QuantileSpec],
                 bucket_cols: Optional[List[str]] = None):
        self.q_specs = q_specs
        self.bucket_cols = bucket_cols or []
        self.models: Dict[str, GradientBoostingRegressor] = {}
        self.calib_offsets: Dict[str, Dict[str, float]] = {}

    @staticmethod
    def _bucket_key(df: pd.DataFrame, bucket_cols: List[str]) -> pd.Series:
        if not bucket_cols:
            return pd.Series(["__GLOBAL__"] * len(df), index=df.index)
        # Fill NaN with 'NA' strings for robust grouping
        b = df[bucket_cols].astype("string").fillna("NA")
        return b.apply(lambda r: "|".join(map(str, r.values)), axis=1)

    def fit(self,
            df_proc: pd.DataFrame,
            y: pd.Series,
            feature_cols: List[str],
            groups: pd.Series,
            random_state: int = 42):
        """Fit q05/q50/q95 models and build OOF residual-based calibration per bucket."""
        X = df_proc[feature_cols].copy()
        # Guard: keep only numeric columns for GBM
        X = X.select_dtypes(include=[np.number])
        # Track the actually used feature columns for persistence/inference
        self.feature_cols_used = list(X.columns)
        # init models by alpha
        for spec in self.q_specs:
            m = GradientBoostingRegressor(
                loss="quantile",
                alpha=spec.alpha,
                n_estimators=spec.n_estimators,
                max_depth=spec.max_depth,
                learning_rate=spec.learning_rate,
                subsample=spec.subsample,
                min_samples_leaf=spec.min_samples_leaf,
                random_state=random_state
            )
            self.models[f"q{int(round(100*spec.alpha)):02d}"] = m

        # OOF predictions for calibration
        oof_preds: Dict[str, np.ndarray] = {k: np.full(len(X), np.nan) for k in self.models.keys()}
        gkf = GroupKFold(n_splits=5)
        for fold, (tr, va) in enumerate(gkf.split(X, y, groups=groups)):
            LOG.info("Training fold %d ...", fold+1)
            Xtr, ytr = X.iloc[tr], y.iloc[tr]
            Xva = X.iloc[va]
            for key, m in self.models.items():
                m_fold = clone(m)
                m_fold.fit(Xtr, ytr)
                oof_preds[key][va] = m_fold.predict(Xva)
        # Train final models on all data
        for key, m in self.models.items():
            m.fit(X, y)

        # Calibration offsets per bucket & quantile: δ_α(b) = Q_α(y - q̂)
        buckets = self._bucket_key(df_proc, self.bucket_cols)
        for key in self.models.keys():
            resid = y.to_numpy() - oof_preds[key]
            df_cal = pd.DataFrame({"bucket": buckets, "resid": resid})
            offs: Dict[str, float] = {}
            for b, g in df_cal.groupby("bucket", sort=False):
                r = g["resid"].to_numpy()
                alpha = int(key[1:]) / 100.0
                offs[b] = float(np.nanquantile(r, alpha))
            self.calib_offsets[key] = offs
        LOG.info("Calibration offsets learned for %d buckets per quantile.", len(set(buckets)))

        # Post-calibration OOF coverage diagnostics
        try:
            uniq = buckets.astype("string").fillna("NA").to_numpy()
            # Precompute per-key offsets vector
            def cal_oof_for(key: str) -> np.ndarray:
                offs_map = self.calib_offsets.get(key, {})
                default_off = offs_map.get("__GLOBAL__", 0.0)
                bucket_offs = {b: offs_map.get(b, default_off) for b in np.unique(uniq)}
                offs_vec = np.array([bucket_offs.get(b, default_off) for b in uniq], dtype=float)
                return oof_preds[key] + offs_vec

            if all(k in oof_preds for k in ["q05","q50","q95"]):
                q05c = cal_oof_for("q05")
                q50c = cal_oof_for("q50")
                q95c = cal_oof_for("q95")
                yv = y.to_numpy()
                p_lo = float(np.nanmean(yv <= q05c))
                p_md = float(np.nanmean(yv <= q50c))
                p_hi = float(np.nanmean(yv >= q95c))
                d_lo = abs(p_lo - (list(self.models.keys()) and 0.05))  # expected ~ alpha for q05
                d_md = abs(p_md - 0.5)
                d_hi = abs(p_hi - (list(self.models.keys()) and 0.05))  # expected ~ alpha for q95
                LOG.info("OOF post-cal coverage: P(y<=q05)=%.3f (|Δ|≈%.3f) P(y<=q50)=%.3f (|Δ|≈%.3f) P(y>=q95)=%.3f (|Δ|≈%.3f)", p_lo, d_lo, p_md, d_md, p_hi, d_hi)
                if d_lo > 0.02 or d_md > 0.02 or d_hi > 0.02:
                    LOG.warning("Post-cal OOF coverage deviates >2pp (lo=%.3f md=%.3f hi=%.3f)", d_lo, d_md, d_hi)
        except Exception as e:
            LOG.warning("Coverage diagnostics failed: %s", e)

        # Diagnostics
        for key in self.models.keys():
            pred = oof_preds[key]
            alpha = int(key[1:]) / 100.0
            cvr = float(np.nanmean(y.to_numpy() <= pred))
            mpbl = mean_pinball_loss(y, pred, alpha=alpha)
            LOG.info("OOF q%s coverage=%.3f (target=%.3f) pinball=%.6f", key, cvr, alpha, mpbl)

    def _apply_calibration(self, qkey: str, raw_pred: np.ndarray, buckets: pd.Series) -> np.ndarray:
        offs_map = self.calib_offsets.get(qkey, {})
        default_off = offs_map.get("__GLOBAL__", 0.0)
        out = np.empty_like(raw_pred, dtype=float)
        # Vectorized map: build array of offsets
        uniq = buckets.astype("string").fillna("NA").to_numpy()
        # Precompute per-bucket offset
        bucket_offs = {b: offs_map.get(b, default_off) for b in np.unique(uniq)}
        offs_vec = np.array([bucket_offs.get(b, default_off) for b in uniq], dtype=float)
        out[:] = raw_pred + offs_vec
        return out

    def predict_quantiles(self, df_proc: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        use_cols = getattr(self, "feature_cols_used", feature_cols)
        use_cols = [c for c in use_cols if c in df_proc.columns]
        X = df_proc[use_cols].copy()
        # numeric-only at inference as well
        X = X.select_dtypes(include=[np.number])
        if X.shape[1] == 0:
            raise ValueError("No usable numeric features found for prediction.")
        buckets = self._bucket_key(df_proc, self.bucket_cols)
        preds = {}
        for key, m in self.models.items():
            raw = m.predict(X)
            cal = self._apply_calibration(key, raw, buckets)
            preds[key] = cal
        qdf = pd.DataFrame(preds, index=df_proc.index)
        # Enforce monotonicity q05 <= q50 <= q95 if present
        try:
            if {"q05","q50","q95"}.issubset(set(qdf.columns)):
                q05 = qdf["q05"].to_numpy()
                q50 = np.maximum(qdf["q50"].to_numpy(), q05)
                q95 = np.maximum(qdf["q95"].to_numpy(), q50)
                # repair any inversions conservatively
                q50 = np.minimum(q50, q95)
                q05 = np.minimum(q05, q50)
                qdf["q05"], qdf["q50"], qdf["q95"] = q05, q50, q95
        except Exception:
            pass
        return qdf

# --------------------------- Decision helpers --------------------------------
def ev_from_quantiles(q05: np.ndarray, q50: np.ndarray, q95: np.ndarray) -> np.ndarray:
    """
    Simple EV proxy from three quantiles.
    Intuition: skew-aware Simpson-esque average; conservative on tails.
    """
    return (q05 + 4.0*q50 + q95) / 6.0

def pop_above_hurdle(q05: np.ndarray, q50: np.ndarray, q95: np.ndarray, hurdle: float) -> np.ndarray:
    """
    Approximate P(Y > h). Piecewise-linear between known quantiles.
    """
    res = np.zeros_like(q50, dtype=float)
    # Below q05 -> near 0; above q95 -> near 1
    res[hurdle <= q05] = 1.0
    res[hurdle >= q95] = 0.0
    mid_mask = (hurdle > q05) & (hurdle < q95)
    # Linearly interpolate using two segments (q05->q50, q50->q95)
    left = (hurdle >= q05) & (hurdle <= q50)
    right = (hurdle >= q50) & (hurdle <= q95)
    # On left segment, probability mass above hurdle decreases from ~1 at q05 to ~0.5 at q50
    res[left] = 1.0 - 0.5 * (hurdle[left] - q05[left]) / np.maximum(q50[left] - q05[left], 1e-12)
    # On right segment, mass above hurdle decreases from 0.5 at q50 to 0 at q95
    res[right] = 0.5 * (q95[right] - hurdle[right]) / np.maximum(q95[right] - q50[right], 1e-12)
    return res

def stress_ev(ev: np.ndarray, gap: float = -0.03) -> np.ndarray:
    """Very simple stress: apply a fixed downside gap to EV."""
    return ev + gap

# --------------------------- Enhanced CQF (Mapping from scores) ---------------
# The following section implements an alternative, light-weight CQF that
# maps within-day score percentiles to forward-return quantiles with
# optional isotonic calibration, conformal offsets, and position sizing.

@dataclass
class EnhancedCqfConfig:
    # Data columns
    date_col: str = "asof_date"
    id_col: str = "contractID"
    score_col: str = "score"
    ret_col: str = "target_fwd_return_3d"

    # Windowing
    window_days: int = 120
    min_window_days: int = 30
    adaptive_window: bool = True
    calib_frac: float = 0.25

    # Binning
    nbins: int = 50
    min_count_per_bin: int = 30
    use_adaptive_bins: bool = True

    # Calibration
    alpha: float = 0.05
    use_isotonic: bool = True

    # Costs / sizing / limits
    tcost_bps: float = 10.0
    hurdle_bps: float = 5.0
    max_positions: Optional[int] = 50
    cap_per_underlying: int = 5
    sizing_method: str = "kelly"  # "kelly", "risk_parity", "equal", "uncertainty"
    kelly_fraction: float = 0.25

    # Feature context (optional)
    feature_cols: List[str] = field(default_factory=list)

    # Output options
    output_diagnostics: bool = True


@dataclass
class EnhancedCqfMapping:
    bins: np.ndarray
    grid: np.ndarray
    q05_grid: np.ndarray
    q50_grid: np.ndarray
    q95_grid: np.ndarray
    alpha: float
    adj_lower: float
    adj_upper: float
    score_flipped: bool = False
    feature_stats: Optional[Dict[str, Dict[str, float]]] = None
    calibration_metrics: Dict[str, float] = field(default_factory=dict)
    config: EnhancedCqfConfig = field(default_factory=EnhancedCqfConfig)
    # Optional isotonic calibrators
    iso_lower: Any = None
    iso_median: Any = None
    iso_upper: Any = None

    def predict_quantiles(self, score_pct: np.ndarray, features: Optional[pd.DataFrame] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        q05 = np.interp(score_pct, self.grid, self.q05_grid)
        q50 = np.interp(score_pct, self.grid, self.q50_grid)
        q95 = np.interp(score_pct, self.grid, self.q95_grid)
        # Isotonic calibration if available
        if self.iso_lower is not None:
            try:
                q05 = self.iso_lower.predict(q05)
            except Exception:
                pass
        if self.iso_median is not None:
            try:
                q50 = self.iso_median.predict(q50)
            except Exception:
                pass
        if self.iso_upper is not None:
            try:
                q95 = self.iso_upper.predict(q95)
            except Exception:
                pass
        # Conformal
        if np.isfinite(self.adj_lower):
            q05 = q05 - self.adj_lower
        if np.isfinite(self.adj_upper):
            q95 = q95 + self.adj_upper
        # Optional feature-based widening (example: high IV widens)
        if features is not None and self.feature_stats is not None:
            if "options_data_implied_volatility" in features.columns and "options_data_implied_volatility" in self.feature_stats:
                iv = features["options_data_implied_volatility"].to_numpy()
                s = self.feature_stats["options_data_implied_volatility"]
                iv_z = (iv - s.get("mean", 0.0)) / (s.get("std", 1e-6) + 1e-6)
                mult = 1.0 + 0.2 * np.clip(iv_z - 1.0, 0.0, 2.0)
                width = (q95 - q05) * mult
                center = 0.5 * (q95 + q05)
                q05 = center - 0.5 * width
                q95 = center + 0.5 * width
        return q05, q50, q95


def _spearmanr_safe(x: np.ndarray, y: np.ndarray) -> float:
    try:
        # Use pandas ranking to compute Spearman via Pearson on ranks
        xr = pd.Series(x).rank(method="average").to_numpy()
        yr = pd.Series(y).rank(method="average").to_numpy()
        if np.std(xr) < 1e-12 or np.std(yr) < 1e-12:
            return np.nan
        return float(np.corrcoef(xr, yr)[0, 1])
    except Exception:
        return np.nan


def _analyze_score_direction(df: pd.DataFrame, cfg: EnhancedCqfConfig) -> Tuple[pd.DataFrame, bool, Dict[str, float]]:
    cors: List[float] = []
    for _, g in df.groupby(cfg.date_col):
        if g.shape[0] < 5:
            continue
        s = g[cfg.score_col].to_numpy()
        r = g[cfg.ret_col].to_numpy()
        mask = np.isfinite(s) & np.isfinite(r)
        if mask.sum() < 5:
            continue
        cors.append(_spearmanr_safe(s[mask], r[mask]))
    med = float(np.nanmedian(cors)) if cors else 0.0
    flipped = med < 0
    if flipped:
        df = df.copy()
        df[cfg.score_col] = -df[cfg.score_col]
    stats = {"median_correlation": med, "n_days_analyzed": float(len(cors))}
    LOG.info("Score direction: median Spearman=%.4f flipped=%s", med, flipped)
    return df, flipped, stats


def _daily_percentile(df: pd.DataFrame, date_col: str, score_col: str) -> pd.Series:
    def _rank_to_pct(s: pd.Series) -> pd.Series:
        if len(s) <= 1:
            return pd.Series([0.5] * len(s), index=s.index)
        r = s.rank(method="average")
        return (r - 1) / (len(r) - 1)
    return df.groupby(date_col)[score_col].transform(_rank_to_pct)


def _compute_bin_quantiles(pcts: np.ndarray, rets: np.ndarray, bins: np.ndarray, quantiles: List[float], min_count: int) -> Dict[float, np.ndarray]:
    out: Dict[float, np.ndarray] = {q: np.full(len(bins) - 1, np.nan) for q in quantiles}
    valid = np.isfinite(rets)
    if valid.any():
        global_q = {q: float(np.nanquantile(rets[valid], q)) for q in quantiles}
    else:
        global_q = {q: 0.0 for q in quantiles}
    for i in range(len(bins) - 1):
        mask = (pcts >= bins[i]) & (pcts < bins[i + 1])
        bin_vals = rets[mask]
        bin_vals = bin_vals[np.isfinite(bin_vals)]
        if bin_vals.size >= min_count:
            for q in quantiles:
                out[q][i] = float(np.nanquantile(bin_vals, q))
    for q in quantiles:
        s = pd.Series(out[q])
        s = s.fillna(method="ffill").fillna(method="bfill").fillna(global_q[q])
        if len(s) > 3:
            s = s.rolling(3, center=True, min_periods=1).mean()
        out[q] = s.to_numpy()
    return out


def _fit_isotonic(calib_df: pd.DataFrame, grid: np.ndarray, qmap: Dict[float, np.ndarray], cfg: EnhancedCqfConfig) -> Tuple[Any, Any, Any]:
    if IsotonicRegression is None or calib_df.shape[0] < 100:
        return None, None, None
    try:
        pcts = calib_df["score_pct"].to_numpy()
        y = calib_df[cfg.ret_col].to_numpy()
        mask = np.isfinite(pcts) & np.isfinite(y)
        q05p = np.interp(pcts[mask], grid, qmap[cfg.alpha])
        q50p = np.interp(pcts[mask], grid, qmap[0.5])
        q95p = np.interp(pcts[mask], grid, qmap[1 - cfg.alpha])
        iso_l = IsotonicRegression(out_of_bounds="clip").fit(q05p, y[mask])
        iso_m = IsotonicRegression(out_of_bounds="clip").fit(q50p, y[mask])
        iso_u = IsotonicRegression(out_of_bounds="clip").fit(q95p, y[mask])
        return iso_l, iso_m, iso_u
    except Exception as e:  # pragma: no cover
        warnings.warn(f"Isotonic calibration failed: {e}")
        return None, None, None


def _conformal_adjust(calib_df: pd.DataFrame, grid: np.ndarray, qmap: Dict[float, np.ndarray], cfg: EnhancedCqfConfig) -> Tuple[float, float]:
    if calib_df.shape[0] < cfg.min_count_per_bin:
        return 0.0, 0.0
    pcts = calib_df["score_pct"].to_numpy()
    y = calib_df[cfg.ret_col].to_numpy()
    mask = np.isfinite(pcts) & np.isfinite(y)
    if mask.sum() == 0:
        return 0.0, 0.0
    q05p = np.interp(pcts[mask], grid, qmap[cfg.alpha])
    q95p = np.interp(pcts[mask], grid, qmap[1 - cfg.alpha])
    lower_scores = q05p - y[mask]
    upper_scores = y[mask] - q95p
    if lower_scores.size == 0 or upper_scores.size == 0:
        return 0.0, 0.0
    return float(np.nanquantile(lower_scores, 1 - cfg.alpha)), float(np.nanquantile(upper_scores, 1 - cfg.alpha))


def _eval_calibration(calib_df: pd.DataFrame, grid: np.ndarray, qmap: Dict[float, np.ndarray], cfg: EnhancedCqfConfig) -> Dict[str, float]:
    if calib_df.shape[0] < 10:
        return {"coverage_lower": np.nan, "coverage_upper": np.nan}
    pcts = calib_df["score_pct"].to_numpy()
    y = calib_df[cfg.ret_col].to_numpy()
    mask = np.isfinite(pcts) & np.isfinite(y)
    q05p = np.interp(pcts[mask], grid, qmap[cfg.alpha])
    q95p = np.interp(pcts[mask], grid, qmap[1 - cfg.alpha])
    cov_lo = float(np.mean(y[mask] >= q05p))
    cov_hi = float(np.mean(y[mask] <= q95p))
    return {
        "coverage_lower": cov_lo,
        "coverage_upper": cov_hi,
        "expected_coverage": 1.0 - cfg.alpha,
        "interval_width": float(np.mean(q95p - q05p)),
        "calibration_error": float(abs(cov_lo - (1.0 - cfg.alpha))),
    }


def _select_window(df: pd.DataFrame, date_col: str, ret_col: str, min_days: int, default_days: int) -> int:
    windows = [30, 60, 90, 120, 150, 180]
    stats_list: List[Tuple[int, float]] = []
    if df.empty:
        return default_days
    max_date = pd.to_datetime(df[date_col]).max()
    for w in windows:
        if w < min_days:
            continue
        min_date = max_date - pd.Timedelta(days=w - 1)
        wdf = df[df[date_col] >= min_date]
        if wdf.shape[0] < 100:
            continue
        gd = wdf.groupby(date_col)[ret_col].agg(['mean', 'std'])
        if gd.shape[0] < 3:
            continue
        stability = float(gd['std'].std() / (gd['mean'].std() + 1e-6))
        stats_list.append((w, stability))
    if not stats_list:
        return default_days
    best = min(stats_list, key=lambda t: t[1])[0]
    LOG.info("EnhancedCQF: selected window=%d days", best)
    return best


def fit_enhanced_cqf(hist_df: pd.DataFrame, cfg: EnhancedCqfConfig) -> EnhancedCqfMapping:
    if hist_df is None or hist_df.empty:
        raise ValueError("hist_df is empty")
    df = hist_df.copy()
    if cfg.date_col not in df.columns:
        raise ValueError(f"Missing date column '{cfg.date_col}'")
    df[cfg.date_col] = pd.to_datetime(df[cfg.date_col], errors="coerce")
    df = df.dropna(subset=[cfg.date_col, cfg.score_col, cfg.ret_col])
    # pick window
    if cfg.adaptive_window:
        window_days = _select_window(df, cfg.date_col, cfg.ret_col, cfg.min_window_days, cfg.window_days)
    else:
        window_days = cfg.window_days
    max_date = df[cfg.date_col].max()
    min_date = max_date - pd.Timedelta(days=window_days - 1)
    dfw = df[df[cfg.date_col] >= min_date].copy()
    if dfw.empty:
        raise ValueError("No data in selected window")
    # Detect score direction and percentiles
    dfw, flipped, _ = _analyze_score_direction(dfw, cfg)
    dfw["score_pct"] = _daily_percentile(dfw, cfg.date_col, cfg.score_col)
    # Time split into train/calib by days
    days = sorted(dfw[cfg.date_col].unique())
    split_idx = int(len(days) * (1.0 - cfg.calib_frac))
    split_idx = max(5, min(split_idx, len(days) - 5))
    train_days = set(days[:split_idx])
    calib_days = set(days[split_idx:])
    tr = dfw[dfw[cfg.date_col].isin(train_days)]
    ca = dfw[dfw[cfg.date_col].isin(calib_days)]
    # Bins
    if cfg.use_adaptive_bins:
        bins = np.quantile(tr["score_pct"].to_numpy(), np.linspace(0, 1, cfg.nbins + 1))
        bins[0], bins[-1] = 0.0, 1.0
    else:
        bins = np.linspace(0.0, 1.0, cfg.nbins + 1)
    grid = 0.5 * (bins[:-1] + bins[1:])
    qmap = _compute_bin_quantiles(tr["score_pct"].to_numpy(), tr[cfg.ret_col].to_numpy(), bins, [cfg.alpha, 0.5, 1 - cfg.alpha], cfg.min_count_per_bin)
    # Optional isotonic
    iso_l, iso_m, iso_u = (None, None, None)
    if cfg.use_isotonic:
        iso_l, iso_m, iso_u = _fit_isotonic(ca, grid, qmap, cfg)
    # Conformal offsets
    adj_l, adj_u = _conformal_adjust(ca, grid, qmap, cfg)
    # Feature stats (if any)
    feat_stats: Optional[Dict[str, Dict[str, float]]] = None
    if cfg.feature_cols:
        avail = [c for c in cfg.feature_cols if c in dfw.columns]
        if avail:
            feat_stats = {c: {"mean": float(dfw[c].mean()), "std": float(dfw[c].std())} for c in avail}
    metrics = _eval_calibration(ca, grid, qmap, cfg)
    mapping = EnhancedCqfMapping(
        bins=bins,
        grid=grid,
        q05_grid=qmap[cfg.alpha],
        q50_grid=qmap[0.5],
        q95_grid=qmap[1 - cfg.alpha],
        alpha=cfg.alpha,
        adj_lower=adj_l,
        adj_upper=adj_u,
        score_flipped=flipped,
        feature_stats=feat_stats,
        calibration_metrics=metrics,
        config=cfg,
        iso_lower=iso_l,
        iso_median=iso_m,
        iso_upper=iso_u,
    )
    LOG.info("EnhancedCQF fitted. coverage_lo=%.3f coverage_hi=%.3f width=%.5f", metrics.get("coverage_lower", float('nan')), metrics.get("coverage_upper", float('nan')), metrics.get("interval_width", float('nan')))
    return mapping


class TransactionCostModel:
    def __init__(self, base_cost_bps: float = 10.0):
        self.base_cost = base_cost_bps / 10000.0
    def estimate_cost(self, features: Optional[pd.DataFrame], spread_col: str = "trading_metrics_spread_pct", oi_col: str = "options_data_open_interest") -> np.ndarray:
        if features is None or features.empty:
            return np.full(0, self.base_cost)
        n = len(features)
        costs = np.full(n, self.base_cost, dtype=float)
        if spread_col in features.columns:
            costs += 0.5 * np.clip(features[spread_col].to_numpy(dtype=float), 0.0, 1.0)
        if oi_col in features.columns:
            oi = features[oi_col].to_numpy(dtype=float)
            mult = np.where(oi < 100, 2.0, np.where(oi < 1000, 1.5, 1.0))
            costs *= mult
        return costs


class PositionSizer:
    @staticmethod
    def kelly(q05: np.ndarray, q50: np.ndarray, q95: np.ndarray, fraction: float = 0.25) -> np.ndarray:
        p_win = (q50 > 0).astype(float)
        avg_win = np.where(q95 > 0, q95, q50)
        avg_loss = np.abs(np.minimum(q05, 0))
        b = avg_win / (avg_loss + 1e-6)
        k = (p_win * b - (1 - p_win)) / (b + 1e-6)
        return fraction * np.clip(k, 0.0, 1.0)
    @staticmethod
    def risk_parity(q05: np.ndarray, q50: np.ndarray, q95: np.ndarray) -> np.ndarray:
        risk = q95 - q05
        w = 1.0 / (risk + 1e-6)
        s = w.sum()
        return w / s if s > 0 else np.zeros_like(w)
    @staticmethod
    def uncertainty(q05: np.ndarray, q50: np.ndarray, q95: np.ndarray) -> np.ndarray:
        unc = q95 - q05
        w = np.clip(q50, 0.0, None) / (unc**2 + 1e-6)
        s = w.sum()
        return w / s if s > 0 else np.zeros_like(w)


def apply_enhanced_cqf(mapping: EnhancedCqfMapping, today_df: pd.DataFrame, features_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    cfg = mapping.config
    df = today_df.copy()
    df[cfg.date_col] = pd.to_datetime(df[cfg.date_col], errors="coerce")
    df = df.dropna(subset=[cfg.date_col, cfg.score_col])
    if mapping.score_flipped:
        df[cfg.score_col] = -df[cfg.score_col]
    df["score_pct"] = _daily_percentile(df, cfg.date_col, cfg.score_col)
    q05, q50, q95 = mapping.predict_quantiles(df["score_pct"].to_numpy(), features_df)
    out = df[[cfg.date_col, cfg.id_col, cfg.score_col]].copy()
    out["score_pct"], out["q05"], out["q50"], out["q95"] = df["score_pct"].to_numpy(), q05, q50, q95
    out["uncertainty"] = out["q95"] - out["q05"]
    tcosts = TransactionCostModel(cfg.tcost_bps).estimate_cost(features_df if features_df is not None else df)
    if tcosts.size != len(out):
        tcosts = np.full(len(out), cfg.tcost_bps / 10000.0)
    hurdle = cfg.hurdle_bps / 10000.0
    out["pass_gate"] = (out["q05"] > -tcosts) & (out["q50"] > hurdle)
    # sizing
    if cfg.sizing_method == "kelly":
        w = PositionSizer.kelly(out["q05"].to_numpy(), out["q50"].to_numpy(), out["q95"].to_numpy(), cfg.kelly_fraction)
    elif cfg.sizing_method == "risk_parity":
        w = PositionSizer.risk_parity(out["q05"].to_numpy(), out["q50"].to_numpy(), out["q95"].to_numpy())
    elif cfg.sizing_method == "uncertainty":
        w = PositionSizer.uncertainty(out["q05"].to_numpy(), out["q50"].to_numpy(), out["q95"].to_numpy())
    else:
        w = np.ones(len(out), dtype=float)
        s = w.sum(); w = w / s if s > 0 else w
    w = w * out["pass_gate"].to_numpy().astype(float)
    out["raw_weight"] = w
    # caps
    if cfg.max_positions and cfg.max_positions > 0:
        for d in out[cfg.date_col].unique():
            m = out[cfg.date_col] == d
            if m.sum() > cfg.max_positions:
                keep = out.loc[m].nlargest(cfg.max_positions, "q50").index
                out.loc[m & ~out.index.isin(keep), "raw_weight"] = 0.0
    # normalize per-day
    out["weight"] = 0.0
    for d in out[cfg.date_col].unique():
        m = out[cfg.date_col] == d
        s = float(out.loc[m, "raw_weight"].sum())
        if s > 0:
            out.loc[m, "weight"] = out.loc[m, "raw_weight"] / s
    # diagnostics
    if cfg.output_diagnostics:
        out["tcost"] = np.where(len(tcosts) == len(out), tcosts, cfg.tcost_bps / 10000.0)
        out["expected_sharpe"] = out["q50"] / (out["uncertainty"] + 1e-6)
        out["confidence"] = 1.0 - out["uncertainty"]
    return out


def _load_model_predict_scores(model_path: str, X: pd.DataFrame) -> np.ndarray:
    # Try joblib first (sklearn/xgb wrapper)
    try:
        mdl = joblib.load(model_path)
        pred = mdl.predict(X)
        return np.asarray(pred, dtype=float)
    except Exception:
        pass
    # Try raw XGBoost
    if xgb is not None:
        try:
            booster = xgb.Booster()
            booster.load_model(model_path)
            dm = xgb.DMatrix(X)
            return booster.predict(dm)
        except Exception:
            pass
    raise RuntimeError(f"Unable to load model at {model_path} for scoring")


def integrate_with_xgboost(xgb_model_path: str, data_df: pd.DataFrame, cfg: EnhancedCqfConfig) -> pd.DataFrame:
    cols = [c for c in data_df.columns if c not in {cfg.id_col, cfg.date_col, 'rank_label'} and not c.startswith(('target_', 'fwd_', 'fwd_ret_'))]
    X = data_df[cols].copy()
    for c in X.columns:
        if X[c].dtype == object:
            X[c] = X[c].astype(str)
    X = X.fillna(0.0)
    scores = _load_model_predict_scores(xgb_model_path, X)
    df = data_df.copy()
    df[cfg.score_col] = scores
    if cfg.date_col not in df.columns and 'date' in df.columns:
        df[cfg.date_col] = df['date']
    if cfg.id_col not in df.columns and 'contractID' in df.columns:
        df[cfg.id_col] = df['contractID']
    return df


class CQFBacktester:
    def __init__(self, cfg: EnhancedCqfConfig):
        self.cfg = cfg
    def backtest(self, data_df: pd.DataFrame, mapping: EnhancedCqfMapping, refit_frequency: int = 30) -> pd.DataFrame:
        df = data_df.sort_values(self.cfg.date_col).copy()
        dates = sorted(pd.to_datetime(df[self.cfg.date_col]).unique())
        all_out: List[pd.DataFrame] = []
        last_refit = dates[0] if dates else None
        current_mapping = mapping
        for d in dates:
            if last_refit is not None and (pd.to_datetime(d) - pd.to_datetime(last_refit)).days >= refit_frequency:
                lb = pd.to_datetime(d) - pd.Timedelta(days=self.cfg.window_days)
                hist = df[(pd.to_datetime(df[self.cfg.date_col]) >= lb) & (pd.to_datetime(df[self.cfg.date_col]) < pd.to_datetime(d))]
                if hist.shape[0] > 100:
                    try:
                        current_mapping = fit_enhanced_cqf(hist, self.cfg)
                        last_refit = d
                    except Exception as e:
                        LOG.warning("Refit failed at %s: %s", d, e)
            today = df[pd.to_datetime(df[self.cfg.date_col]) == pd.to_datetime(d)]
            if today.empty:
                continue
            dec = apply_enhanced_cqf(current_mapping, today)
            dec["backtest_date"] = pd.to_datetime(d)
            all_out.append(dec)
        if not all_out:
            return pd.DataFrame()
        res = pd.concat(all_out, ignore_index=True)
        if self.cfg.ret_col in df.columns:
            res = res.merge(df[[self.cfg.date_col, self.cfg.id_col, self.cfg.ret_col]], on=[self.cfg.date_col, self.cfg.id_col], how="left")
            res["position_return"] = res.get("weight", 0.0) * res[self.cfg.ret_col]
            daily = res.groupby("backtest_date")["position_return"].sum()
            if daily.std(ddof=0) > 0:
                sharpe = float(np.sqrt(252.0) * daily.mean() / (daily.std(ddof=0) + 1e-12))
            else:
                sharpe = float('nan')
            # max drawdown
            cum = (1.0 + daily).cumprod()
            running_max = cum.cummax()
            dd = (cum - running_max) / (running_max + 1e-12)
            LOG.info("Backtest: Sharpe=%.3f MaxDD=%.2f%%", sharpe, 100.0 * float(dd.min()))
        return res

# --------------------------- CLI plumbing ------------------------------------
def ensure_date_group(df: pd.DataFrame) -> pd.Series:
    if "asof_date" in df.columns:
        g = pd.to_datetime(df["asof_date"], errors="coerce")
    elif "date" in df.columns:
        g = pd.to_datetime(df["date"], errors="coerce")
    else:
        raise ValueError("date/asof_date column required for GroupKFold")
    if g.isna().any():
        raise ValueError("date/asof_date contains NaN after parsing")
    return g.dt.strftime("%Y-%m-%d")

def load_and_preprocess(csv_path: str, cfg_file: str, utils_module: str) -> pd.DataFrame:
    utils = import_utils(utils_module)
    df = pd.read_csv(csv_path, low_memory=False)
    # Normalize schema
    if "contractID" not in df.columns:
        if "contract_id" in df.columns:
            df = df.rename(columns={"contract_id":"contractID"})
        elif "option_symbol" in df.columns:
            df = df.rename(columns={"option_symbol":"contractID"})
    if "date" not in df.columns:
        raise ValueError("Input must contain 'date'")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    # Build eval price
    df["px_eval"] = build_eval_price(df)
    cfg = utils.load_config(cfg_file)
    df_proc, _ = utils.preprocess_data(df.copy(), cfg, scaler=None)
    return df, df_proc

def save_artifact(path: str, art: CQFArtifact):
    obj = {
        "version": art.version,
        "horizon": art.horizon,
        "price_col": art.price_col,
        "feature_cols": art.feature_cols,
        "bucket_cols": art.bucket_cols,
        "quantiles": art.quantiles,
        "models": art.models,
        "calib_offsets": art.calib_offsets,
        "metadata": art.metadata,
    }
    joblib.dump(obj, path)
    LOG.info("Saved CQF artifact → %s", path)

def load_artifact(path: str) -> CQFArtifact:
    obj = joblib.load(path)
    art = CQFArtifact(
        version=obj["version"],
        horizon=obj["horizon"],
        price_col=obj["price_col"],
        feature_cols=list(obj["feature_cols"]),
        bucket_cols=list(obj["bucket_cols"]),
        quantiles=list(obj["quantiles"]),
        models=obj["models"],
        calib_offsets=obj["calib_offsets"],
        metadata=obj.get("metadata", {}),
    )
    return art

# --------------------------- Commands ----------------------------------------
def cmd_train(args):
    LOG.info("=== CQF TRAIN ===")
    raw_df, dfp = load_and_preprocess(args.train_csv, args.config_file, args.utils_module)
    # Target
    target = compute_forward_return(raw_df.sort_values(["contractID","date"]).reset_index(drop=True),
                                    horizon=args.horizon,
                                    price_col="px_eval")
    dfp = dfp.reindex(target.index)  # align
    # Keep rows with valid target
    mask = np.isfinite(target.to_numpy())
    dfp = dfp.loc[mask].reset_index(drop=True)
    target = target.loc[mask].reset_index(drop=True)
    raw_df = raw_df.loc[mask].reset_index(drop=True)

    # Feature columns
    feature_cols = pick_feature_cols(dfp, args.features_pkl)

    # Optional Step-2 score as feature (OOS within train by fold)
    if args.step2_model:
        # Build OOF step2 score to avoid leakage into CQF training
        gdates = ensure_date_group(raw_df)
        gkf = GroupKFold(n_splits=5)
        oof_s2 = np.full(len(dfp), np.nan, dtype=float)
        for fold, (tr, va) in enumerate(gkf.split(dfp, target, groups=gdates)):
            LOG.info("Step2 OOF scoring fold %d ...", fold+1)
            # Fit preprocess is already done; we only need predictions
            s2 = add_step2_score_if_given(dfp.iloc[tr], args.step2_model, args.step2_features)
            # Train data just to align columns, we ignore
            # Predict on VA using the *same* model artifact (OK: model is frozen)
            s2_va = add_step2_score_if_given(dfp.iloc[va], args.step2_model, args.step2_features)
            oof_s2[va] = s2_va.to_numpy()
        dfp["step2_score_pos"] = oof_s2
        feature_cols = ["step2_score_pos"] + feature_cols

    # Buckets (optional)
    bucket_cols = []
    if args.bucket_cols:
        bucket_cols = [c.strip() for c in args.bucket_cols.split(",") if c.strip() in dfp.columns]

    # GroupKFold groups
    groups = ensure_date_group(raw_df)

    # Specs
    q_specs = [
        QuantileSpec(alpha=0.05, n_estimators=500, max_depth=3, learning_rate=0.05, min_samples_leaf=40),
        QuantileSpec(alpha=0.50, n_estimators=600, max_depth=3, learning_rate=0.05, min_samples_leaf=40),
        QuantileSpec(alpha=0.95, n_estimators=500, max_depth=3, learning_rate=0.05, min_samples_leaf=40),
    ]
    cqf = CQF(q_specs=q_specs, bucket_cols=bucket_cols)
    cqf.fit(dfp, target, feature_cols, groups)

    # Train-time diagnostics
    pred_q = cqf.predict_quantiles(dfp, feature_cols)
    # Coverage after calibration
    for key in pred_q.columns:
        alpha = int(key[1:]) / 100.0
        cov = float(np.mean(target.to_numpy() <= pred_q[key].to_numpy()))
        pin = mean_pinball_loss(target, pred_q[key], alpha=alpha)
        LOG.info("POST-CAL q%s coverage=%.3f pinball=%.6f", key, cov, pin)

    # Save artifact
    art = CQFArtifact(
        version="1.0.0",
        horizon=args.horizon,
        price_col="px_eval",
        feature_cols=feature_cols,
        bucket_cols=bucket_cols,
        quantiles=[0.05, 0.50, 0.95],
        models=cqf.models,
        calib_offsets=cqf.calib_offsets,
        metadata={
            "train_csv": str(Path(args.train_csv).resolve()),
            "features_pkl": str(args.features_pkl) if args.features_pkl else "",
            "used_step2_score": str(bool(args.step2_model)),
        }
    )
    Path(args.artifact).parent.mkdir(parents=True, exist_ok=True)
    save_artifact(args.artifact, art)
    LOG.info("Train done. Samples used: %d", len(dfp))

def cmd_predict(args):
    LOG.info("=== CQF PREDICT ===")
    art = load_artifact(args.artifact)
    raw_df, dfp = load_and_preprocess(args.eval_csv, args.config_file, args.utils_module)

    # Optionally add Step-2 score now (single pass; inference can safely use the frozen model)
    if args.add_step2_score:
        s2 = add_step2_score_if_given(dfp, args.step2_model, args.step2_features)
        if s2 is not None:
            dfp["step2_score_pos"] = s2
            if "step2_score_pos" not in art.feature_cols:
                art.feature_cols = ["step2_score_pos"] + art.feature_cols

    # Align features
    missing = [c for c in art.feature_cols if c not in dfp.columns]
    if missing:
        LOG.warning("Missing feature columns at inference: %s", missing)
    Xcols = [c for c in art.feature_cols if c in dfp.columns]
    if not Xcols:
        raise ValueError("No usable features found at inference. Check artifact feature_cols vs eval schema.")

    # Rehydrate CQF
    cqf = CQF(q_specs=[], bucket_cols=art.bucket_cols)
    cqf.models = art.models
    cqf.calib_offsets = art.calib_offsets

    qdf = cqf.predict_quantiles(dfp, Xcols)
    q05 = qdf["q05"].to_numpy() if "q05" in qdf else qdf.get("q5", pd.Series(np.nan, index=qdf.index)).to_numpy()
    q50 = qdf["q50"].to_numpy()
    q95 = qdf["q95"].to_numpy()

    # Decision features
    ev = ev_from_quantiles(q05, q50, q95)
    pop = pop_above_hurdle(q05, q50, q95, hurdle=args.hurdle)
    sev = stress_ev(ev, gap=args.stress_gap)

    out = pd.DataFrame({
        "contractID": raw_df.get("contractID", pd.Series(index=dfp.index, dtype=str)).astype(str).values,
        "date": pd.to_datetime(raw_df["date"]).dt.strftime("%Y-%m-%d").values,
        "q05": q05, "q50": q50, "q95": q95,
        "ev": ev, "pop_gt_hurdle": pop, "stress_ev": sev
    })
    # (Optional) derive a deterministic utility score to rank candidates
    lam = float(args.tail_lambda)
    out["utility"] = out["ev"] - lam * np.maximum(0.0, -out["q05"])  # penalize left tail
    out = out.sort_values(["date", "utility"], ascending=[True, False]).reset_index(drop=True)

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    LOG.info("Wrote %s (%d rows)", args.out_csv, len(out))

# --------------------------- Main -------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Calibrated Quantile Forecasting (CQF)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("train", help="Train quantile models and calibration")
    pt.add_argument("--train-csv", required=True)
    pt.add_argument("--config-file", required=True)
    pt.add_argument("--utils-module", default="utils", help="Python module name for your utils.py")
    pt.add_argument("--horizon", type=int, default=3)
    pt.add_argument("--artifact", required=True, help="Output .joblib")
    pt.add_argument("--features-pkl", default=None, help="Optional feature list PKL (Step-2)")
    pt.add_argument("--step2-model", default=None, help="Optional: path to Step-2 ranker .joblib")
    pt.add_argument("--step2-features", default=None, help="Optional: Step-2 feature list PKL")
    pt.add_argument("--bucket-cols", default="", help="Comma-separated bucket columns present after preprocess")

    pp = sub.add_parser("predict", help="Predict quantiles + decision features")
    pp.add_argument("--eval-csv", required=True)
    pp.add_argument("--config-file", required=True)
    pp.add_argument("--utils-module", default="utils")
    pp.add_argument("--artifact", required=True)
    pp.add_argument("--horizon", type=int, default=3)
    pp.add_argument("--out-csv", required=True)
    pp.add_argument("--hurdle", type=float, default=0.02)
    pp.add_argument("--stress-gap", type=float, default=-0.03)
    pp.add_argument("--tail_lambda", type=float, default=0.5, help="Penalty weight for left tail in utility")
    pp.add_argument("--add-step2-score", action="store_true", help="Compute Step-2 score at inference")
    pp.add_argument("--step2-model", default=None)
    pp.add_argument("--step2-features", default=None)

    # Enhanced CQF subcommands
    pe = sub.add_parser("fit", help="Fit Enhanced CQF mapping from scores to quantiles")
    pe.add_argument("--data", required=True, help="Path to historical data CSV/parquet")
    pe.add_argument("--config", help="Path to config JSON with EnhancedCqfConfig fields")
    pe.add_argument("--output", required=True, help="Output .joblib mapping path")

    pae = sub.add_parser("apply", help="Apply Enhanced CQF mapping to generate decisions")
    pae.add_argument("--model", required=True, help="Path to saved Enhanced CQF mapping .joblib")
    pae.add_argument("--data", required=True, help="Path to today's data CSV/parquet")
    pae.add_argument("--output", required=True, help="Output decisions CSV/parquet")

    pb = sub.add_parser("backtest", help="Backtest Enhanced CQF strategy with periodic refits")
    pb.add_argument("--data", required=True, help="Path to historical data CSV/parquet")
    pb.add_argument("--config", help="Path to config JSON with EnhancedCqfConfig fields")
    pb.add_argument("--output", required=True, help="Output results CSV/parquet")
    pb.add_argument("--refit-freq", type=int, default=30)

    pi = sub.add_parser("integrate", help="Inject XGBoost scores, fit and apply Enhanced CQF")
    pi.add_argument("--xgb-model", required=True, help="Path to XGBoost or joblib model")
    pi.add_argument("--data", required=True, help="Path to data CSV/parquet")
    pi.add_argument("--output", required=True, help="Output decisions CSV/parquet")

    args = p.parse_args()
    if args.cmd == "train":
        cmd_train(args)
    elif args.cmd == "predict":
        cmd_predict(args)
    elif args.cmd == "fit":
        cfg = EnhancedCqfConfig()
        if getattr(args, "config", None):
            with open(args.config, "r") as f:
                cfg_dict = json.load(f)
            cfg = EnhancedCqfConfig(**cfg_dict)
        # Load data
        data = pd.read_csv(args.data) if args.data.endswith('.csv') else pd.read_parquet(args.data)
        mapping = fit_enhanced_cqf(data, cfg)
        joblib.dump(mapping, args.output)
        LOG.info("Enhanced CQF mapping saved → %s", args.output)
    elif args.cmd == "apply":
        mapping = joblib.load(args.model)
        data = pd.read_csv(args.data) if args.data.endswith('.csv') else pd.read_parquet(args.data)
        dec = apply_enhanced_cqf(mapping, data)
        if args.output.endswith('.csv'):
            dec.to_csv(args.output, index=False)
        else:
            dec.to_parquet(args.output, index=False)
        LOG.info("Decisions saved → %s", args.output)
    elif args.cmd == "backtest":
        cfg = EnhancedCqfConfig()
        if getattr(args, "config", None):
            with open(args.config, "r") as f:
                cfg_dict = json.load(f)
            cfg = EnhancedCqfConfig(**cfg_dict)
        data = pd.read_csv(args.data) if args.data.endswith('.csv') else pd.read_parquet(args.data)
        # initial fit on first window
        data[cfg.date_col] = pd.to_datetime(data[cfg.date_col], errors="coerce")
        start = data[cfg.date_col].min()
        init_cut = start + pd.Timedelta(days=cfg.window_days)
        init_hist = data[data[cfg.date_col] <= init_cut]
        mapping = fit_enhanced_cqf(init_hist, cfg)
        bt = CQFBacktester(cfg)
        res = bt.backtest(data, mapping, getattr(args, "refit_freq", 30))
        if args.output.endswith('.csv'):
            res.to_csv(args.output, index=False)
        else:
            res.to_parquet(args.output, index=False)
        LOG.info("Backtest results saved → %s", args.output)
    elif args.cmd == "integrate":
        cfg = EnhancedCqfConfig()
        data = pd.read_csv(args.data) if args.data.endswith('.csv') else pd.read_parquet(args.data)
        data = integrate_with_xgboost(args.xgb_model, data, cfg)
        mapping = fit_enhanced_cqf(data, cfg)
        dec = apply_enhanced_cqf(mapping, data)
        if args.output.endswith('.csv'):
            dec.to_csv(args.output, index=False)
        else:
            dec.to_parquet(args.output, index=False)
        LOG.info("Integrated decisions saved → %s", args.output)

if __name__ == "__main__":
    main()
