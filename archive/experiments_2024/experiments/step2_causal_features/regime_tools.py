#!/usr/bin/env python3
"""
Regime-aware calibration + drift monitoring for CQF.

Includes:
- AdaptiveConformalCalibrator (grouped/weighted/rolling conformal)
- PageHinkley drift detector (fast online change detection)
- EVTTailAdjuster (GPD-based tail widening with safe fallback)

Dependencies: numpy, pandas
Optional: scipy (for GPD). If unavailable, falls back gracefully.
"""

from __future__ import annotations
import math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

# ----- optional scipy for EVT -----
try:
    from scipy.stats import genpareto
    _SCIPY_OK = True
except Exception:
    _SCIPY_OK = False


# ---------- utilities ----------

def _weighted_quantile(v: np.ndarray,
                       w: np.ndarray,
                       q: float) -> float:
    """Weighted quantile in [0,1]. Supports NaNs and zero/neg weights."""
    v = np.asarray(v, dtype=float)
    w = np.asarray(w, dtype=float)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if mask.sum() == 0:
        return float('nan')
    v, w = v[mask], w[mask]
    order = np.argsort(v)
    v, w = v[order], w[order]
    cum_w = np.cumsum(w)
    cutoff = q * w.sum()
    idx = np.searchsorted(cum_w, cutoff, side='right')
    idx = min(max(idx, 0), len(v) - 1)
    return float(v[idx])


def _exp_decay_weights(dates: pd.Series,
                       lam: float) -> np.ndarray:
    """Exponential recency weights, lam in (0,1]. lam ~ 0.97..0.995 typical."""
    if dates is None or dates.isnull().all():
        return np.ones(len(dates) if dates is not None else 0, dtype=float)
    d = pd.to_datetime(dates)
    max_d = pd.to_datetime(d.max())
    age = (max_d - d).dt.days.clip(lower=0).astype(float)
    return np.power(lam, age)


def _make_bins_vix(vix: pd.Series) -> pd.Series:
    """Simple, robust VIX bins. Adjust cut points as needed."""
    # bins: [0,15), [15,20), [20,30), [30, inf)
    cuts = [-np.inf, 15, 20, 30, np.inf]
    return pd.cut(vix.fillna(vix.median()), bins=cuts, labels=False, include_lowest=True)


def _make_bins_dte(dte: pd.Series) -> pd.Series:
    """DTE bins: 0-7, 8-21, 22-45, 46+."""
    cuts = [-np.inf, 7, 21, 45, np.inf]
    return pd.cut(dte.fillna(dte.median()), bins=cuts, labels=False, include_lowest=True)


def _scores_one_sided(y: np.ndarray,
                      lower_pred: Optional[np.ndarray],
                      upper_pred: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Conformal one-sided nonnegative scores:
      S_lo = max(q_lo(x) - y, 0)  -> widen lower by subtracting quantile(S_lo)
      S_up = max(y - q_up(x), 0)  -> widen upper by adding quantile(S_up)
    """
    S_lo = None
    S_up = None
    if lower_pred is not None:
        S_lo = np.maximum(lower_pred - y, 0.0)
    if upper_pred is not None:
        S_up = np.maximum(y - upper_pred, 0.0)
    return S_lo, S_up


# ---------- Adaptive Conformal Calibrator ----------

@dataclass
class AdaptiveConformalCalibrator:
    """
    Grouped/Weighted/Rolling conformal calibration.

    - Supports one-sided adjustments for lower and upper quantiles
    - Supports grouping by regime covariates (e.g., VIX bin × DTE bin)
    - Supports exponential recency weighting to track regime shifts
    - Safe fallbacks if a group's sample is small

    Usage:
      cal = AdaptiveConformalCalibrator(alpha=0.10)
      cal.fit(calib_df, pred_lower, pred_median, pred_upper,
              vix=calib_df['vix_d_close'], dte=calib_df['days_to_exp'], date=calib_df['date'])
      adj = cal.adjust(df, pred_lower, pred_median, pred_upper,
                       vix=df['vix_d_close'], dte=df['days_to_exp'], date=df['date'])
    """
    alpha: float = 0.10                     # central interval target, e.g., 0.10 for 90%
    alpha_lo: Optional[float] = None        # if None, uses alpha/2
    alpha_up: Optional[float] = None        # if None, uses alpha/2
    use_groups: bool = True
    min_group_n: int = 300
    recency_lambda: float = 0.99            # exponential decay λ (closer to 1=slower decay)
    global_floor_n: int = 1000              # min N for global fallback
    median_debias: bool = True              # recenter q50 by median residual (grouped)
    # saved state (properly initialized)
    _group_adj: Optional[Dict[Tuple[int, int], Tuple[float, float]]] = None
    _global_adj: Optional[Tuple[float, float]] = None
    _group_medbias: Optional[Dict[Tuple[int, int], float]] = None
    _global_medbias: float = 0.0
    _last_bins: Optional[Dict[str, List[float]]] = None
    
    def __post_init__(self):
        """Initialize mutable fields to avoid dataclass sharing issues."""
        if self._group_adj is None:
            self._group_adj = {}
        if self._global_adj is None:
            self._global_adj = (0.0, 0.0)
        if self._group_medbias is None:
            self._group_medbias = {}
        if self._last_bins is None:
            self._last_bins = {}

    def fit(self,
            calib_df: pd.DataFrame,
            pred_lower: np.ndarray,
            pred_median: Optional[np.ndarray],
            pred_upper: np.ndarray,
            vix: Optional[pd.Series] = None,
            dte: Optional[pd.Series] = None,
            date: Optional[pd.Series] = None) -> "AdaptiveConformalCalibrator":

        y = calib_df['target_pnl'].values.astype(float)
        q_lo = pred_lower if pred_lower is not None else None
        q_md = pred_median if pred_median is not None else None
        q_up = pred_upper if pred_upper is not None else None

        a_lo = self.alpha_lo if self.alpha_lo is not None else self.alpha / 2.0
        a_up = self.alpha_up if self.alpha_up is not None else self.alpha / 2.0

        # one-sided nonnegative scores
        S_lo, S_up = _scores_one_sided(y, q_lo, q_up)

        # recency weights
        w_time = _exp_decay_weights(date, self.recency_lambda) if date is not None else np.ones_like(y)

        # group bins
        if self.use_groups and vix is not None and dte is not None:
            g_vix = _make_bins_vix(vix)
            g_dte = _make_bins_dte(dte)
            groups = pd.DataFrame({'g_vix': g_vix, 'g_dte': g_dte})
        else:
            groups = pd.DataFrame({'g_vix': pd.Series([0]*len(y)), 'g_dte': pd.Series([0]*len(y))})

        # compute adjustments
        self._group_adj = {}
        self._group_medbias = {}

        # global fallbacks
        lo_global = _weighted_quantile(S_lo, w_time, 1.0 - a_lo) if S_lo is not None else 0.0
        up_global = _weighted_quantile(S_up, w_time, 1.0 - a_up) if S_up is not None else 0.0
        self._global_adj = (float(lo_global if math.isfinite(lo_global) else 0.0),
                            float(up_global if math.isfinite(up_global) else 0.0))

        # median bias (q50) - group-wise for better regime adaptation
        if q_md is not None and self.median_debias:
            resid_md = (q_md - y)  # positive means q50 too high (optimistic)
            md_global = _weighted_quantile(resid_md, w_time, 0.5)
            self._global_medbias = float(md_global if math.isfinite(md_global) else 0.0)
        else:
            self._global_medbias = 0.0

        # grouped
        df_scores = pd.DataFrame({
            'S_lo': S_lo if S_lo is not None else np.zeros_like(y),
            'S_up': S_up if S_up is not None else np.zeros_like(y),
            'w': w_time,
            'g_vix': groups['g_vix'].values,
            'g_dte': groups['g_dte'].values,
        })
        if q_md is not None:
            df_scores['resid_md'] = (q_md - y)
        else:
            df_scores['resid_md'] = 0.0

        for (iv, idte), sub in df_scores.groupby(['g_vix', 'g_dte']):
            n = len(sub)
            if n < self.min_group_n:
                continue
            lo = _weighted_quantile(sub['S_lo'].values, sub['w'].values, 1.0 - a_lo)
            up = _weighted_quantile(sub['S_up'].values, sub['w'].values, 1.0 - a_up)
            lo = float(lo if math.isfinite(lo) else 0.0)
            up = float(up if math.isfinite(up) else 0.0)
            self._group_adj[(int(iv), int(idte))] = (lo, up)

            if self.median_debias:
                medb = _weighted_quantile(sub['resid_md'].values, sub['w'].values, 0.5)
                self._group_medbias[(int(iv), int(idte))] = float(medb if math.isfinite(medb) else 0.0)

        return self

    def _lookup_adj(self, vix_bin: Optional[int], dte_bin: Optional[int]) -> Tuple[float, float, float]:
        """Get (lower_adj, upper_adj, med_bias) with fallback to global."""
        if vix_bin is not None and dte_bin is not None:
            key = (int(vix_bin), int(dte_bin))
            if key in self._group_adj:
                lo, up = self._group_adj[key]
                medb = self._group_medbias.get(key, self._global_medbias)
                return lo, up, medb
        lo, up = self._global_adj
        return lo, up, self._global_medbias

    def adjust(self,
               df: pd.DataFrame,
               pred_lower: np.ndarray,
               pred_median: Optional[np.ndarray],
               pred_upper: np.ndarray,
               vix: Optional[pd.Series] = None,
               dte: Optional[pd.Series] = None,
               date: Optional[pd.Series] = None) -> Dict[str, np.ndarray]:
        """
        Apply stored adjustments to predictions. Returns dict with adjusted arrays.
        """
        n = len(df)
        out_lo = pred_lower.copy() if pred_lower is not None else None
        out_md = pred_median.copy() if pred_median is not None else None
        out_up = pred_upper.copy() if pred_upper is not None else None

        if self.use_groups and vix is not None and dte is not None:
            vix_bins = _make_bins_vix(vix)
            dte_bins = _make_bins_dte(dte)
        else:
            vix_bins = pd.Series([0]*n)
            dte_bins = pd.Series([0]*n)

        for i in range(n):
            lo_adj, up_adj, medb = self._lookup_adj(vix_bins.iloc[i], dte_bins.iloc[i])
            if out_lo is not None:
                out_lo[i] = out_lo[i] - lo_adj
            if out_up is not None:
                out_up[i] = out_up[i] + up_adj
            if out_md is not None and self.median_debias:
                out_md[i] = out_md[i] - medb

        # enforce monotonicity
        if out_lo is not None and out_md is not None:
            out_md = np.maximum(out_md, out_lo)
        if out_up is not None:
            if out_md is not None:
                out_up = np.maximum(out_up, out_md)
            if out_lo is not None:
                out_up = np.maximum(out_up, out_lo)

        return {
            'q0.05': out_lo if out_lo is not None else None,
            'q0.50': out_md if out_md is not None else None,
            'q0.95': out_up if out_up is not None else None
        }


# ---------- EVT Tail Adjuster (optional, layered on top) ----------

@dataclass
class EVTTailAdjuster:
    """
    VIX-adaptive EVT patch for tails on calibration scores S_lo and S_up.
    Applies heavy tail protection during regime stress, moderate during stable periods. 

    Typical usage:
      evt = EVTTailAdjuster(base_alpha=0.005, stress_alpha=0.02)
      evt.fit(S_lo, S_up, mean_vix=18.5)  # vectors from calibration + VIX context
      lo_inc, up_inc = evt.increments()
    """
    tail_thresh: float = 0.80      # use top 20% of scores as exceedances  
    base_alpha: float = 0.005      # baseline tail mass for stable periods
    stress_alpha: float = 0.02     # heavy tail mass for regime stress (VIX>25)
    vix_threshold: float = 25.0    # VIX level triggering stress mode
    lo_inc_: float = 0.0
    up_inc_: float = 0.0
    _adaptive_alpha: float = 0.005  # computed based on VIX context

    def _fit_one_side(self, S: np.ndarray) -> float:
        S = np.asarray(S, dtype=float)
        S = S[np.isfinite(S)]
        S = S[S > 0]
        if len(S) < 100:
            return 0.0
        u = np.quantile(S, self.tail_thresh)
        exceed = S[S > u] - u
        if len(exceed) < 50:
            return 0.0

        if _SCIPY_OK:
            # Fit GPD to exceedances
            try:
                c, loc, scale = genpareto.fit(exceed, floc=0.0)
                # quantile for extra tail mass (VIX-adaptive)
                q = genpareto.ppf(1.0 - self._adaptive_alpha, c, loc=0.0, scale=scale)
                q = float(q if math.isfinite(q) and q > 0 else 0.0)
                return u + q
            except Exception:
                # GPD fit failed, fallback to empirical
                pass
        
        # fallback: empirical (very robust)
        q = np.quantile(S, min(0.999, 1.0 - self._adaptive_alpha))
        return float(q if math.isfinite(q) else 0.0)

    def fit(self, S_lo: Optional[np.ndarray], S_up: Optional[np.ndarray], mean_vix: float = 20.0) -> "EVTTailAdjuster":
        """
        Fit EVT with VIX-adaptive conservatism.
        """
        # Adaptive alpha based on VIX stress level
        if mean_vix >= self.vix_threshold:
            # High stress: use full protection
            self._adaptive_alpha = self.stress_alpha
        else:
            # Stable/moderate: interpolate between base and stress
            stress_factor = max(0.0, (mean_vix - 15.0) / (self.vix_threshold - 15.0))
            self._adaptive_alpha = self.base_alpha + stress_factor * (self.stress_alpha - self.base_alpha)
        
        self.lo_inc_ = self._fit_one_side(S_lo) if S_lo is not None else 0.0
        self.up_inc_ = self._fit_one_side(S_up) if S_up is not None else 0.0
        return self

    def increments(self) -> Tuple[float, float]:
        # These are absolute scores (>= 0) to further widen the band.
        return float(self.lo_inc_), float(self.up_inc_)


# ---------- Page-Hinkley drift detector ----------

@dataclass
class PageHinkley:
    """
    Online drift detector for residual streams.

    Trigger when cumulative mean deviation exceeds threshold.
    - delta: magnitude of allowed changes around mean (tolerance)
    - lambda_: detection threshold
    - alpha: forgetting factor for running mean
    """
    delta: float = 0.005
    lambda_: float = 50.0
    alpha: float = 0.99

    # internal state
    mean_t: float = 0.0
    min_mean: float = 0.0
    cum_sum: float = 0.0
    n: int = 0
    change_detected: bool = False

    def update(self, x: float) -> bool:
        self.n += 1
        if self.n == 1:
            self.mean_t = x
            self.min_mean = x
            self.cum_sum = 0.0
            self.change_detected = False
            return False

        # exponentially-smoothed mean
        self.mean_t = self.alpha * self.mean_t + (1 - self.alpha) * x
        self.cum_sum = self.cum_sum + (x - self.mean_t - self.delta)
        self.min_mean = min(self.min_mean, self.cum_sum)

        stat = self.cum_sum - self.min_mean
        self.change_detected = stat > self.lambda_
        return self.change_detected

    def reset(self):
        self.mean_t = 0.0
        self.min_mean = 0.0
        self.cum_sum = 0.0
        self.n = 0
        self.change_detected = False


# ---------- Regime Feature Engineering ----------

def add_realized_vol_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 6: Add realized volatility and vol-of-vol features for enhanced regime detection.
    These features supplement VIX-based detection to catch surprise events.
    """
    df = df.copy()
    
    # Realized volatility (rolling std of returns)
    if 'spy_d_close' in df.columns:
        # Calculate returns
        df['spy_returns'] = df['spy_d_close'].pct_change()
        
        # Realized volatility (annualized, 20-day window)
        df['realized_vol_20d'] = df['spy_returns'].rolling(20, min_periods=10).std() * np.sqrt(252)
        
        # Vol-of-vol (volatility of volatility - key for detecting surprise shocks)
        df['vol_of_vol_20d'] = df['realized_vol_20d'].rolling(20, min_periods=10).std()
        
        # Step 7: More sensitive emergency detection (95th percentile instead of 99th)
        df['vol_emergency'] = (df['realized_vol_20d'] > df['realized_vol_20d'].rolling(252, min_periods=100).quantile(0.95)).astype(int)
        
        # Step 7: Vol-of-vol severity multiplier for Black Swan scaling
        if 'vol_of_vol_20d' in df.columns:
            vol_of_vol_baseline = df['vol_of_vol_20d'].rolling(252, min_periods=100).quantile(0.8)
            df['vol_severity'] = np.clip(df['vol_of_vol_20d'] / vol_of_vol_baseline.fillna(1.0), 1.0, 50.0)  # Cap at 50x
        else:
            df['vol_severity'] = 1.0
        
        # Vol acceleration (rate of vol change)
        df['vol_acceleration'] = df['realized_vol_20d'].diff(5)  # 5-day vol change
        
    return df

def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add regime-aware features to capture structural changes.
    Modifies df in-place and returns it.
    """
    df = df.copy()
    
    # VIX regime classification
    if 'vix_d_close' in df.columns:
        df['vix_regime'] = pd.cut(df['vix_d_close'], 
                                  bins=[0, 15, 25, 100], 
                                  labels=['low', 'medium', 'high'])
        df['vix_regime'] = df['vix_regime'].cat.codes  # Convert to numeric
    
    # Volatility clustering (rolling volatility of volatility)
    if 'implied_volatility' in df.columns:
        df['vol_cluster'] = df.groupby('contractID')['implied_volatility'].transform(
            lambda x: x.rolling(20, min_periods=5).std()
        )
        df['vol_cluster'] = df['vol_cluster'].fillna(0)
    
    # Market stress indicator  
    if 'vix_d_close' in df.columns and 'vol_cluster' in df.columns:
        # Rolling quantile for vol_cluster threshold
        vol_thresh = df['vol_cluster'].rolling(252, min_periods=50).quantile(0.8)
        vol_thresh = vol_thresh.fillna(df['vol_cluster'].quantile(0.8))
        
        # Step 6: Enhanced stress detection with realized vol
        vix_stress = (df['vix_d_close'] > 20).astype(int)
        cluster_stress = (df['vol_cluster'] > vol_thresh).astype(int)
        
        # Add realized vol emergency detection
        vol_emergency = df['vol_emergency'].fillna(0).astype(int) if 'vol_emergency' in df.columns else 0
        vol_spike = (df['vol_of_vol_20d'] > df['vol_of_vol_20d'].rolling(252, min_periods=50).quantile(0.95)).astype(int) if 'vol_of_vol_20d' in df.columns else 0
        
        df['stress_score'] = vix_stress + cluster_stress + vol_emergency + vol_spike
    
    return df


# ---------- Regime Detection Utilities ----------

def detect_regime_shift(df: pd.DataFrame, window: int = 60) -> pd.Series:
    """
    Detect regime shifts using variance and autocorrelation.
    Returns boolean series indicating regime shift periods.
    """
    if 'vix_d_close' not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
    
    # Rolling volatility of volatility (vol-of-vol)
    vix_rolling_std = df['vix_d_close'].rolling(window, min_periods=window//2).std()
    vol_of_vol = vix_rolling_std.rolling(window//2, min_periods=window//4).std()
    
    # Regime shift signal
    regime_threshold = vol_of_vol.quantile(0.9, interpolation='nearest')
    regime_threshold = regime_threshold if pd.notna(regime_threshold) else vol_of_vol.median()
    
    return vol_of_vol > regime_threshold


# ---------- Utility for time-decay sample weights ----------

def calculate_time_decay_weights(dates: pd.Series, decay_lambda: float = 0.995) -> np.ndarray:
    """
    Calculate time-decay sample weights for training.
    More recent samples get higher weights.
    
    Args:
        dates: Series of dates
        decay_lambda: Decay factor (0.99-0.999 typical)
        
    Returns:
        Array of sample weights
    """
    if dates.isnull().all():
        return np.ones(len(dates))
    
    max_date = dates.max()
    days_old = (max_date - dates).dt.days.clip(lower=0)
    weights = np.power(decay_lambda, days_old)
    
    # Normalize to mean=1 for stability
    weights = weights / weights.mean()
    return weights.values
