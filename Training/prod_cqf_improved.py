#!/usr/bin/env python3
"""
Optimal CQF (Calibrated Quantile Forecasting) Implementation - v3 (Improved)

Improvements over v2:
- Frozen dataclass for config (immutable, type-safe)
- Extracted repeated patterns into reusable helpers
- Added caching for expensive preprocessing operations
- Unified conformal calibration flow (stable/stress handled cleanly)
- Full type hints throughout
- ~30% code reduction while maintaining all functionality
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_pinball_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils import load_config, preprocess_data
from logger import setup_logger
from regime_tools import (
    add_regime_features,
    add_realized_vol_features,
    calculate_time_decay_weights,
    AdaptiveConformalCalibrator,
    EVTTailAdjuster,
    _scores_one_sided,
)

warnings.filterwarnings('ignore')
logger = setup_logger(__name__, level=logging.INFO)


# ===== Transformation Utilities =====

def log_modulus_transform(y: np.ndarray) -> np.ndarray:
    """Transform target to log space: sign(y) * log(1 + |y|)"""
    return np.sign(y) * np.log1p(np.abs(y))


def inverse_log_modulus(y_log: np.ndarray) -> np.ndarray:
    """Inverse transform: sign(y') * (exp(|y'|) - 1)"""
    return np.sign(y_log) * np.expm1(np.abs(y_log))


# ===== Configuration =====

@dataclass(frozen=True)
class CQFConfig:
    """Immutable configuration for CQF model."""

    # Time Decay
    time_decay_lambda: float = 0.995

    # Data Splits
    test_days: int = 90
    val_days: int = 60
    val_calib_split: float = 0.5

    # Transformation
    use_log_transform: bool = True

    # Feature Selection
    min_feature_coverage: float = 0.5

    # XGBoost Hyperparameters
    n_estimators: int = 464
    max_depth: int = 4
    learning_rate: float = 0.030991
    min_child_weight: float = 1.167711
    subsample: float = 0.700948
    colsample_bytree: float = 0.767209
    reg_alpha: float = 0.015122
    reg_lambda: float = 0.051630
    gamma: float = 0.738906
    max_bin: int = 256
    tree_method: str = 'hist'
    early_stopping_rounds: int = 30

    # Sample Weighting
    tail_percentile: float = 0.9
    tail_weight_multiplier: float = 2.0

    # Conformal Calibration
    min_group_size: int = 200
    min_calib_samples: int = 100
    safety_factor: float = 1.2
    conformal_alpha: float = 0.1

    # Regime Detection
    stable_vix_threshold: float = 0.8
    vol_of_vol_threshold: float = 1.0
    severe_stress_threshold: float = 1.5
    black_swan_threshold: float = 5.0
    emergency_threshold: float = 10.0
    vol_lookback_days: int = 20

    # EVT
    evt_max_multiplier: float = 0.5
    evt_tail_thresh: float = 0.70
    evt_base_alpha: float = 0.005
    evt_vix_threshold: float = 20.0
    evt_min_samples: int = 100
    evt_min_exceed: int = 50

    # Emergency Widening
    emergency_multiplier_base: float = 1.0
    emergency_multiplier_rate: float = 0.2
    emergency_multiplier_max: float = 5.0

    # Probability Classifier
    prob_n_estimators: int = 50
    prob_max_depth: int = 3
    prob_learning_rate: float = 0.0130
    prob_min_child_weight: int = 2
    prob_subsample: float = 0.8110
    prob_colsample: float = 0.6820
    prob_reg_alpha: float = 4.8910
    prob_reg_lambda: float = 3.6791
    prob_gamma: float = 2.2547

    # Decision
    risk_penalty: float = 0.5

    # Quality Gates
    max_under_coverage: float = 0.10
    max_over_coverage: float = 0.15
    q05_max_error: float = 0.12
    q50_max_error: float = 0.20
    q95_max_error: float = 0.15
    q50_min_correlation: float = 0.15
    q50_min_spearman: float = 0.20
    q50_max_median_bias: float = 0.10

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


# ===== Feature Definitions =====

CQF_FEATURES: Tuple[str, ...] = (
    'delta', 'gamma', 'theta', 'vega',
    'price_roll_mean_5', 'price_roll_mean_20',
    'price_roll_std_5', 'price_roll_std_20',
    'price_roll_zscore_5', 'price_roll_zscore_20',
    'iv_roll_mean_5', 'iv_roll_mean_20',
    'moneyness', 'days_to_exp', 'implied_volatility',
    'mispricing_ratio', 'risk_adjusted_signal',
    'relative_spread', 'option_volume_oi_ratio',
    'spy_d_close', 'vix_d_close', 'iv_vix_ratio', 'spy_momentum',
    'vix_regime', 'vol_cluster', 'stress_score',
    'realized_vol_20d', 'vol_of_vol_20d', 'vol_emergency', 'vol_acceleration', 'vol_severity',
)


# ===== Helper Functions =====

def _validate_columns(df: pd.DataFrame, required: List[str], phase: str) -> None:
    """Validate required columns exist."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{phase} data missing required columns: {missing}")


def _compute_bin_edges(series: pd.Series, n_bins: int = 5) -> List[float]:
    """Compute quantile-based bin edges."""
    if series is None or series.dropna().empty:
        return []
    values = series.dropna().astype(float).values
    if values.size < n_bins:
        return []
    edges = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)))
    if edges.size < 2:
        return []
    edges = edges.tolist()
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def _assign_bins(values: pd.Series, edges: List[float]) -> np.ndarray:
    """Assign values to bins."""
    if not edges:
        return np.full(len(values), -1, dtype=int)
    arr = values.astype(float).to_numpy()
    bins = np.digitize(arr, edges[1:-1], right=True)
    bins[np.isnan(arr)] = -1
    return bins.astype(int)


def _safe_scalar(value: Any) -> float:
    """Convert value to scalar float safely."""
    if value is None:
        return 0.0
    arr = np.asarray(value)
    if arr.size == 0:
        return 0.0
    return float(arr.flat[0]) if np.isfinite(arr.flat[0]) else 0.0


def _standardize_scores(scores: Optional[np.ndarray]) -> Tuple[Optional[np.ndarray], float]:
    """Standardize EVT scores for robust tail fitting."""
    if scores is None:
        return None, 1.0
    arr = np.clip(np.asarray(scores, dtype=float), 0.0, None)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return arr, 1.0
    positive = finite[finite > 0]
    scale = np.median(positive) if positive.size > 0 else np.std(finite)
    scale = scale if np.isfinite(scale) and scale > 1e-6 else 1.0
    return arr / scale, scale


# ===== Main CQF Class =====

class OptimalCQF:
    """Calibrated Quantile Forecasting for options P&L prediction."""

    def __init__(
        self,
        quantiles: Tuple[float, ...] = (0.05, 0.5, 0.95),
        horizon: int = 5,
        config: Optional[CQFConfig] = None,
        random_state: int = 42,
    ):
        self.quantiles = quantiles
        self.horizon = horizon
        self.config = config or CQFConfig()
        self.random_state = random_state

        # Model state
        self.models: Dict[float, xgb.XGBRegressor] = {}
        self.preprocessor: Optional[Pipeline] = None
        self.feature_names: List[str] = []
        self.conformal_adjustments: Dict[str, Any] = {}
        self.conformal_calibrator: Optional[AdaptiveConformalCalibrator] = None
        self.evt_adjuster: Optional[EVTTailAdjuster] = None
        self.prob_classifier: Optional[xgb.XGBClassifier] = None
        self.prob_isotonic: Optional[IsotonicRegression] = None
        self._calib_df: Optional[pd.DataFrame] = None
        self._vol_of_vol_threshold: Optional[float] = None

        # Cache for preprocessed data
        self._transform_cache: Dict[int, np.ndarray] = {}

    # ===== Data Preparation =====

    def calculate_delta_hedged_pnl(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate delta-hedged P&L targets."""
        logger.info(f"Calculating delta-hedged PnL (horizon={self.horizon}d)")

        price_col = 'last_raw' if 'last_raw' in df.columns else 'last'
        _validate_columns(df, ['contractID', price_col], "PnL calculation")

        df = df.sort_values(['contractID', 'date']).reset_index(drop=True)
        grp = df.groupby('contractID')

        future_price = grp[price_col].shift(-self.horizon)
        df['future_option_price'] = future_price
        df['target_date'] = grp['date'].shift(-self.horizon)

        option_pnl = (future_price - df[price_col]) / df[price_col]

        # Delta hedging
        if 'spy_d_close' in df.columns and 'delta' in df.columns:
            spy_daily = df[['date', 'spy_d_close']].drop_duplicates('date').sort_values('date')
            spy_daily['spy_fwd'] = spy_daily['spy_d_close'].shift(-self.horizon)
            df = df.merge(spy_daily[['date', 'spy_fwd']], on='date', how='left')
            underlying_pnl = (df['spy_fwd'] - df['spy_d_close']) / df['spy_d_close']
            df['target_pnl'] = option_pnl + df['delta'] * (-underlying_pnl)
        else:
            logger.warning("Missing SPY or delta, using raw option returns")
            df['target_pnl'] = option_pnl

        df['option_return'] = option_pnl
        df['target_pnl'] = df['target_pnl'].replace([np.inf, -np.inf], np.nan)

        initial = len(df)
        df = df.dropna(subset=['target_pnl'])
        logger.info(f"Dropped {initial - len(df)} rows with invalid targets")

        # Store raw target before transform
        df['target_pnl_raw'] = df['target_pnl'].copy()

        if self.config.use_log_transform:
            df['target_pnl'] = log_modulus_transform(df['target_pnl'].values)
            logger.info(f"Applied log-modulus transform: "
                       f"[{df['target_pnl_raw'].min():.2f}, {df['target_pnl_raw'].max():.2f}] → "
                       f"[{df['target_pnl'].min():.2f}, {df['target_pnl'].max():.2f}]")

        return df.sort_values('date')

    def create_time_splits(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create time-based train/validation/test splits."""
        logger.info("Creating time-based splits")

        df = df.sort_values('date')
        max_date = df['date'].max()

        test_start = max_date - timedelta(days=self.config.test_days)
        val_start = test_start - timedelta(days=self.config.val_days)

        if 'target_date' in df.columns:
            train_mask = df['target_date'] < val_start
            val_mask = (df['date'] >= val_start) & (df['target_date'] < test_start)
            test_mask = df['date'] >= test_start
        else:
            guard = timedelta(days=self.horizon)
            train_mask = df['date'] < (val_start - guard)
            val_mask = (df['date'] >= val_start) & (df['date'] < (test_start - guard))
            test_mask = df['date'] >= test_start

        train_df = df[train_mask].reset_index(drop=True)
        val_df = df[val_mask].reset_index(drop=True)
        test_df = df[test_mask].reset_index(drop=True)

        logger.info(f"Splits - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        return train_df, val_df, test_df

    def create_preprocessor(self, df: pd.DataFrame) -> Pipeline:
        """Create preprocessing pipeline."""
        available = [f for f in CQF_FEATURES if f in df.columns]
        self.feature_names = [
            f for f in available
            if df[f].notna().mean() > self.config.min_feature_coverage
        ]
        logger.info(f"Selected {len(self.feature_names)} features")

        self.preprocessor = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ])
        return self.preprocessor

    def _transform_features(self, X: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """Transform features with optional caching."""
        cache_key = id(X)
        if not fit and cache_key in self._transform_cache:
            return self._transform_cache[cache_key]

        if fit:
            result = self.preprocessor.fit_transform(X)
        else:
            result = self.preprocessor.transform(X)

        self._transform_cache[cache_key] = result
        return result

    # ===== Model Training =====

    def train_quantile_models(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame
    ) -> Dict[str, float]:
        """Train XGBoost quantile models."""
        logger.info("Training quantile models")

        # Validate XGBoost version
        xgb_major = int(str(xgb.__version__).split('.')[0])
        if xgb_major < 2:
            raise RuntimeError(f"XGBoost >= 2.0 required. Installed: {xgb.__version__}")

        # Split validation for tuning vs calibration
        split_idx = int(len(val_df) * self.config.val_calib_split)
        tune_df = val_df.iloc[:split_idx].reset_index(drop=True)
        self._calib_df = val_df.iloc[split_idx:].reset_index(drop=True)
        logger.info(f"Split validation: Tune={len(tune_df)}, Calib={len(self._calib_df)}")

        X_train = train_df[self.feature_names]
        y_train = train_df['target_pnl']
        X_tune = tune_df[self.feature_names]
        y_tune = tune_df['target_pnl']

        X_train_scaled = self._transform_features(X_train, fit=True)
        X_tune_scaled = self._transform_features(X_tune)

        # Compute sample weights
        sample_weights = self._compute_sample_weights(train_df, y_train)

        metrics = {}
        for q in self.quantiles:
            logger.info(f"Training Q{q:.2f}")
            model = self._create_quantile_model(q)

            fit_kwargs = {
                'X': X_train_scaled,
                'y': y_train,
                'sample_weight': sample_weights,
                'eval_set': [(X_tune_scaled, y_tune)],
                'verbose': False,
            }

            # Early stopping only for tail quantiles
            if q != 0.5:
                model.set_params(early_stopping_rounds=self.config.early_stopping_rounds)

            model.fit(**fit_kwargs)
            self.models[q] = model

            # Evaluate
            y_pred = model.predict(X_tune_scaled)
            q_metrics = self._evaluate_quantile(q, y_tune.values, y_pred)
            metrics.update(q_metrics)

            self._log_feature_importance(model, q)

        return metrics

    def _create_quantile_model(self, quantile: float) -> xgb.XGBRegressor:
        """Create XGBoost model for specific quantile."""
        base_params = {
            'n_estimators': self.config.n_estimators,
            'max_depth': self.config.max_depth,
            'learning_rate': self.config.learning_rate,
            'min_child_weight': self.config.min_child_weight,
            'subsample': self.config.subsample,
            'colsample_bytree': self.config.colsample_bytree,
            'reg_alpha': self.config.reg_alpha,
            'reg_lambda': self.config.reg_lambda,
            'gamma': self.config.gamma,
            'max_bin': self.config.max_bin,
            'tree_method': self.config.tree_method,
            'n_jobs': -1,
            'random_state': self.random_state,
        }

        # Q0.50 uses L2 loss for better correlation
        if quantile == 0.5:
            return xgb.XGBRegressor(objective='reg:squarederror', **base_params)
        else:
            return xgb.XGBRegressor(
                objective='reg:quantileerror',
                quantile_alpha=quantile,
                **base_params,
            )

    def _compute_sample_weights(
        self, df: pd.DataFrame, y: pd.Series
    ) -> np.ndarray:
        """Compute combined time-decay and tail weights."""
        # Tail weighting
        tail_thresh = np.quantile(np.abs(y), self.config.tail_percentile)
        tail_weights = 1.0 + self.config.tail_weight_multiplier * np.minimum(
            1.0, np.abs(y) / (tail_thresh + 1e-12)
        )

        # Time decay
        if 'date' in df.columns:
            time_weights = calculate_time_decay_weights(
                df['date'], self.config.time_decay_lambda
            )
            return tail_weights * time_weights
        return tail_weights

    def _evaluate_quantile(
        self, q: float, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate quantile predictions."""
        metrics = {}

        if q == 0.5:
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            metrics['q0.50_rmse'] = rmse
            logger.info(f"Q0.50 - RMSE: {rmse:.4f}")

            # Additional Q0.50 diagnostics
            if self.config.use_log_transform:
                y_pred_raw = inverse_log_modulus(y_pred)
                y_true_raw = inverse_log_modulus(y_true)
            else:
                y_pred_raw, y_true_raw = y_pred, y_true

            corr_pearson = np.corrcoef(y_pred_raw, y_true_raw)[0, 1]
            corr_spearman, _ = spearmanr(y_pred_raw, y_true_raw)
            median_bias = np.median(y_pred_raw - y_true_raw)

            metrics['q0.50_pearson'] = corr_pearson
            metrics['q0.50_spearman'] = corr_spearman
            metrics['q0.50_median_bias'] = median_bias

            logger.info(f"Q0.50 - Pearson: {corr_pearson:.4f}, Spearman: {corr_spearman:.4f}")

            if corr_spearman < self.config.q50_min_spearman:
                logger.warning(f"Q0.50 Spearman ({corr_spearman:.4f}) below threshold")
        else:
            pinball = mean_pinball_loss(y_true, y_pred, alpha=q)
            metrics[f'q{q:.2f}_pinball'] = pinball
            logger.info(f"Q{q:.2f} - Pinball: {pinball:.4f}")

        return metrics

    def _log_feature_importance(self, model: xgb.XGBRegressor, q: float) -> None:
        """Log top features for a quantile model."""
        if not hasattr(model, 'feature_importances_'):
            return
        imp = model.feature_importances_
        indices = np.argsort(imp)[::-1][:10]
        logger.info(f"Top features for Q{q:.2f}:")
        for i, idx in enumerate(indices):
            logger.info(f"  {i+1}. {self.feature_names[idx]}: {imp[idx]:.4f}")

    # ===== Conformal Calibration (Unified) =====

    def calculate_conformal_adjustments(self) -> Dict[str, Any]:
        """Calculate conformal adjustments (auto-detects stable vs stress)."""
        logger.info("Calculating conformal adjustments")

        if self._calib_df is None:
            raise ValueError("No calibration data. Call train_quantile_models first.")

        if len(self._calib_df) < self.config.min_calib_samples:
            logger.warning("Insufficient calibration samples, using safety margin")
            return self._calibrate_baseline(safety=self.config.safety_factor)

        is_stable = self._detect_stable_period()

        if is_stable:
            return self._calibrate_baseline()
        else:
            return self._calibrate_stress()

    def _detect_stable_period(self) -> bool:
        """Detect if calibration period is stable."""
        df = self._calib_df
        lookback = self.config.vol_lookback_days

        # VIX stability
        vix_stress = 1.0
        if 'vix_d_close' in df.columns and len(df) >= lookback:
            vix = df['vix_d_close']
            baseline = vix.std()
            if baseline > 1e-6:
                vix_stress = vix.iloc[-lookback:].std() / baseline

        # Vol emergency
        vol_emergency = False
        if 'vol_emergency' in df.columns:
            vol_emergency = df['vol_emergency'].iloc[-lookback:].sum() > 0

        # Vol-of-vol
        vol_of_vol_stress = 1.0
        if 'vol_of_vol_20d' in df.columns:
            recent = df['vol_of_vol_20d'].iloc[-lookback:].mean()
            baseline = df['vol_of_vol_20d'].quantile(0.8)
            self._vol_of_vol_threshold = float(df['vol_of_vol_20d'].quantile(0.85))
            if pd.notna(baseline) and baseline > 1e-6:
                vol_of_vol_stress = recent / baseline

        # Severity
        severity = 1.0
        if 'vol_severity' in df.columns:
            severity = df['vol_severity'].iloc[-lookback:].max()

        is_stable = (
            vix_stress <= self.config.stable_vix_threshold
            and not vol_emergency
            and vol_of_vol_stress <= self.config.vol_of_vol_threshold
            and severity <= self.config.severe_stress_threshold
        )

        logger.info(f"Regime: {'STABLE' if is_stable else 'STRESS'} "
                   f"(VIX={vix_stress:.2f}, Emergency={vol_emergency}, "
                   f"Vol-of-Vol={vol_of_vol_stress:.2f}, Severity={severity:.1f}x)")

        return is_stable

    def _calibrate_baseline(self, safety: float = 1.0) -> Dict[str, Any]:
        """Simple conformal calibration for stable periods."""
        logger.info("Using baseline conformal calibration")

        df = self._calib_df
        X = df[self.feature_names]
        y = df['target_pnl'].values
        X_scaled = self._transform_features(X)

        preds = {q: m.predict(X_scaled) for q, m in self.models.items()}

        adjustments: Dict[str, Any] = {}
        alpha = self.config.conformal_alpha

        if 0.05 in preds and 0.95 in preds:
            lower_scores = preds[0.05] - y
            upper_scores = y - preds[0.95]

            n = len(y)
            k = max(1, min(int(np.ceil((n + 1) * (1 - alpha))), n))

            adjustments['lower'] = float(np.partition(lower_scores, k-1)[k-1]) * safety
            adjustments['upper'] = float(np.partition(upper_scores, k-1)[k-1]) * safety

            # Log coverage
            cov = np.mean((y >= preds[0.05]) & (y <= preds[0.95]))
            adj_cov = np.mean(
                (y >= preds[0.05] - adjustments['lower']) &
                (y <= preds[0.95] + adjustments['upper'])
            )
            logger.info(f"Coverage: {cov:.1%} → {adj_cov:.1%}")

        # Median bias
        if 0.5 in preds:
            adjustments['median_bias'] = float(np.median(preds[0.5] - y))

        self.conformal_calibrator = None
        self.evt_adjuster = None
        self.conformal_adjustments = adjustments
        return adjustments

    def _calibrate_stress(self) -> Dict[str, Any]:
        """Adaptive conformal + EVT for stress periods."""
        logger.info("Using adaptive conformal + EVT")

        df = self._calib_df
        X = df[self.feature_names]
        y = df['target_pnl'].values
        X_scaled = self._transform_features(X)

        preds = {q: m.predict(X_scaled) for q, m in self.models.items()}

        vix = df.get('vix_d_close')
        dte = df.get('days_to_exp')
        date = df.get('date')

        # Compute severity
        severity = 1.0
        if 'vol_severity' in df.columns:
            severity = np.clip(
                df['vol_severity'].iloc[-self.config.vol_lookback_days:].max(),
                1.0, self.config.black_swan_threshold
            )

        # Adaptive alpha
        alpha = min(
            self.config.conformal_alpha * (1 + severity * 0.1),
            0.3
        )

        # Fit adaptive calibrator
        self.conformal_calibrator = AdaptiveConformalCalibrator(
            alpha=alpha,
            use_groups=True,
            min_group_n=200,
            recency_lambda=self.config.time_decay_lambda,
            median_debias=True,
        )

        self.conformal_calibrator.fit(
            df, preds.get(0.05), preds.get(0.5), preds.get(0.95),
            vix=vix, dte=dte, date=date,
        )

        # Apply EVT
        adjusted = self.conformal_calibrator.adjust(
            df, preds.get(0.05), preds.get(0.5), preds.get(0.95),
            vix=vix, dte=dte, date=date,
        )

        self._apply_evt(y, adjusted, preds, severity, vix)

        # Store baseline adjustments for fallback
        adjustments = {'adaptive_conformal': True, 'evt_applied': True}
        if 0.05 in preds and 0.95 in preds:
            lower_scores = preds[0.05] - y
            upper_scores = y - preds[0.95]
            n = len(y)
            k = max(1, min(int(np.ceil((n + 1) * (1 - self.config.conformal_alpha))), n))
            adjustments['lower'] = float(np.partition(lower_scores, k-1)[k-1])
            adjustments['upper'] = float(np.partition(upper_scores, k-1)[k-1])

        if 0.5 in preds:
            adjustments['median_bias'] = float(np.median(preds[0.5] - y))

        self.conformal_adjustments = adjustments
        return adjustments

    def _apply_evt(
        self, y: np.ndarray, adjusted: Dict, preds: Dict, severity: float, vix: Optional[pd.Series]
    ) -> None:
        """Apply EVT tail adjustments."""
        try:
            adj_lo = adjusted.get('q0.05', preds.get(0.05))
            adj_up = adjusted.get('q0.95', preds.get(0.95))

            y_lo, _ = _scores_one_sided(y, lower_pred=adj_lo, side='lower')
            _, y_up = _scores_one_sided(y, upper_pred=adj_up, side='upper')

            z_lo, scale_lo = _standardize_scores(y_lo)
            z_up, scale_up = _standardize_scores(y_up)

            mean_vix = float(vix.mean()) if vix is not None else self.config.evt_vix_threshold
            stress_alpha = min(0.02 * severity, 0.10)

            self.evt_adjuster = EVTTailAdjuster(
                tail_thresh=self.config.evt_tail_thresh,
                base_alpha=self.config.evt_base_alpha,
                stress_alpha=stress_alpha,
                vix_threshold=self.config.evt_vix_threshold,
                min_samples=self.config.evt_min_samples,
                min_exceed=self.config.evt_min_exceed,
            )
            self.evt_adjuster.fit(z_lo, z_up, mean_vix=mean_vix)

            lo_evt_z, up_evt_z = self.evt_adjuster.increments()
            lo_evt = _safe_scalar(lo_evt_z) * scale_lo
            up_evt = _safe_scalar(up_evt_z) * scale_up

            # Cap EVT adjustments
            band = np.maximum(preds.get(0.95, 0) - preds.get(0.05, 0), 1e-6)
            max_cap = float(np.median(band)) * self.config.evt_max_multiplier
            lo_evt = float(np.clip(lo_evt, 0, max_cap))
            up_evt = float(np.clip(up_evt, 0, max_cap))

            logger.info(f"EVT adjustments: Lower +{lo_evt:.4f}, Upper +{up_evt:.4f}")

            # Apply to calibrator
            if self.conformal_calibrator and self.conformal_calibrator._global_adj:
                old_lo, old_up = self.conformal_calibrator._global_adj
                self.conformal_calibrator._global_adj = (old_lo + lo_evt, old_up + up_evt)

        except Exception as e:
            logger.warning(f"EVT failed: {e}")
            self.evt_adjuster = None

    # ===== Probability Classifier =====

    def fit_probability_classifier(self) -> None:
        """Train probability classifier for P(profit > 0)."""
        logger.info("Training probability classifier")

        if self._calib_df is None:
            raise ValueError("No calibration data")

        try:
            qdf = self.predict_quantiles(self._calib_df, apply_conformal=True)
            X_prob, names = self._build_prob_features(qdf, self._calib_df)
            y_binary = (self._calib_df['target_pnl'].values > 0).astype(int)

            logger.info(f"Training on {len(X_prob):,} samples, win rate: {y_binary.mean():.1%}")

            self.prob_classifier = xgb.XGBClassifier(
                n_estimators=self.config.prob_n_estimators,
                max_depth=self.config.prob_max_depth,
                learning_rate=self.config.prob_learning_rate,
                min_child_weight=self.config.prob_min_child_weight,
                subsample=self.config.prob_subsample,
                colsample_bytree=self.config.prob_colsample,
                reg_alpha=self.config.prob_reg_alpha,
                reg_lambda=self.config.prob_reg_lambda,
                gamma=self.config.prob_gamma,
                objective='binary:logistic',
                random_state=self.random_state,
                n_jobs=-1,
            )
            self.prob_classifier.fit(X_prob, y_binary)

            # Isotonic calibration
            prob_raw = self.prob_classifier.predict_proba(X_prob)[:, 1]
            self.prob_isotonic = IsotonicRegression(out_of_bounds='clip')
            self.prob_isotonic.fit(prob_raw, y_binary)

            brier = np.mean((prob_raw - y_binary) ** 2)
            prob_cal = self.prob_isotonic.transform(prob_raw)
            brier_cal = np.mean((prob_cal - y_binary) ** 2)

            logger.info(f"Brier score: {brier:.4f} → {brier_cal:.4f} (after isotonic)")

        except Exception as e:
            logger.warning(f"Probability classifier failed: {e}")
            self.prob_classifier = None

    def _build_prob_features(
        self, qdf: pd.DataFrame, raw_df: pd.DataFrame
    ) -> Tuple[np.ndarray, List[str]]:
        """Build feature matrix for probability classifier."""
        q05 = qdf['q0.05'].values
        q95 = qdf['q0.95'].values
        width = q95 - q05

        def get_col(name: str, default: float = 0.0) -> np.ndarray:
            return raw_df[name].values if name in raw_df.columns else np.full(len(raw_df), default)

        moneyness = get_col('moneyness')
        delta = get_col('delta')
        dte = get_col('days_to_exp')
        iv = get_col('implied_volatility')
        vix = get_col('vix_d_close')

        features = [
            q05, q95, width,
            moneyness, delta, dte, iv, vix,
            delta * moneyness,
            q05 * iv,
            width * vix,
        ]
        names = [
            'q0.05', 'q0.95', 'width',
            'moneyness', 'delta', 'dte', 'iv', 'vix',
            'delta_x_moneyness', 'q05_x_iv', 'width_x_vix',
        ]

        return np.column_stack(features), names

    # ===== Prediction =====

    def predict_quantiles(
        self, df: pd.DataFrame, apply_conformal: bool = True
    ) -> pd.DataFrame:
        """Predict quantiles with optional conformal adjustment."""
        X = df[self.feature_names]
        X_scaled = self._transform_features(X)

        raw_preds = {q: m.predict(X_scaled) for q, m in self.models.items()}

        if apply_conformal and self.conformal_adjustments:
            preds = self._apply_conformal(df, raw_preds)
        else:
            preds = {f'q{q:.2f}': p for q, p in raw_preds.items()}

        # Inverse transform if needed
        if self.config.use_log_transform:
            preds = {k: inverse_log_modulus(v) for k, v in preds.items()}

        # Enforce monotonicity
        if all(k in preds for k in ['q0.05', 'q0.50', 'q0.95']):
            q05 = preds['q0.05']
            q50 = np.clip(preds['q0.50'], q05, None)
            q95 = np.clip(preds['q0.95'], q50, None)
            q50 = np.clip(q50, None, q95)
            q05 = np.clip(q05, None, q50)
            preds['q0.05'], preds['q0.50'], preds['q0.95'] = q05, q50, q95

        return pd.DataFrame(preds, index=df.index)

    def _apply_conformal(
        self, df: pd.DataFrame, raw_preds: Dict[float, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Apply conformal adjustments to predictions."""
        # Detect current stress level
        use_adaptive = False
        if self.conformal_calibrator is not None:
            vol_emergency = df.get('vol_emergency', pd.Series([0])).iloc[-1] if len(df) > 0 else False
            vol_of_vol_high = False
            if 'vol_of_vol_20d' in df.columns and self._vol_of_vol_threshold:
                current = df['vol_of_vol_20d'].iloc[-1] if len(df) > 0 else 0
                vol_of_vol_high = current > self._vol_of_vol_threshold

            severity = df['vol_severity'].iloc[-1] if 'vol_severity' in df.columns and len(df) > 0 else 1.0
            use_adaptive = vol_emergency or vol_of_vol_high or severity > self.config.black_swan_threshold

        if use_adaptive:
            vix = df.get('vix_d_close')
            dte = df.get('days_to_exp')
            date = df.get('date')

            adjusted = self.conformal_calibrator.adjust(
                df, raw_preds.get(0.05), raw_preds.get(0.5), raw_preds.get(0.95),
                vix=vix, dte=dte, date=date,
            )
            preds = {
                'q0.05': adjusted.get('q0.05', raw_preds.get(0.05)),
                'q0.50': adjusted.get('q0.50', raw_preds.get(0.5)),
                'q0.95': adjusted.get('q0.95', raw_preds.get(0.95)),
            }

            # Emergency widening for black swan
            severity = df['vol_severity'].iloc[-1] if 'vol_severity' in df.columns and len(df) > 0 else 1.0
            if severity > self.config.emergency_threshold:
                mult = min(
                    self.config.emergency_multiplier_base +
                    (severity - self.config.emergency_threshold) * self.config.emergency_multiplier_rate,
                    self.config.emergency_multiplier_max
                )
                width = preds['q0.95'] - preds['q0.05']
                mid = preds['q0.50']
                preds['q0.05'] = mid - width * mult / 2
                preds['q0.95'] = mid + width * mult / 2
                logger.warning(f"Black swan: {mult:.1f}x interval multiplier")
        else:
            preds = {}
            for q, pred in raw_preds.items():
                if q == 0.05 and 'lower' in self.conformal_adjustments:
                    preds['q0.05'] = pred - self.conformal_adjustments['lower']
                elif q == 0.95 and 'upper' in self.conformal_adjustments:
                    preds['q0.95'] = pred + self.conformal_adjustments['upper']
                else:
                    preds[f'q{q:.2f}'] = pred

        return preds

    def calculate_decision_features(
        self, qdf: pd.DataFrame, raw_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Calculate decision features from quantile predictions."""
        result = qdf.copy()

        if not all(c in qdf.columns for c in ['q0.05', 'q0.50', 'q0.95']):
            return result

        q05, q50, q95 = qdf['q0.05'], qdf['q0.50'], qdf['q0.95']

        result['downside_risk'] = np.abs(np.minimum(q05, 0))
        result['upside_potential'] = np.maximum(q95, 0)
        result['uncertainty'] = q95 - q05

        # Probability of profit
        if self.prob_classifier is not None and raw_df is not None:
            try:
                X_prob, _ = self._build_prob_features(qdf, raw_df)
                prob_raw = self.prob_classifier.predict_proba(X_prob)[:, 1]
                result['prob_profit'] = (
                    self.prob_isotonic.transform(prob_raw)
                    if self.prob_isotonic else prob_raw
                )
            except Exception:
                result['prob_profit'] = np.where(q95 <= 0, 0.0, np.where(q05 >= 0, 1.0, 0.5))
        else:
            result['prob_profit'] = np.where(q95 <= 0, 0.0, np.where(q05 >= 0, 1.0, 0.5))

        # Blended expected return
        tail_exp = result['prob_profit'] * q95 + (1 - result['prob_profit']) * q05
        central_exp = (q05 + 2*q50 + q95) / 4.0
        result['expected_return'] = 0.5 * tail_exp + 0.5 * central_exp
        result['utility'] = result['expected_return'] - self.config.risk_penalty * result['downside_risk']

        return result

    # ===== Evaluation =====

    def evaluate_coverage(
        self, y_true: np.ndarray, predictions: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """Evaluate quantile coverage."""
        metrics = {}

        for qstr, y_pred in predictions.items():
            q = float(qstr[1:])
            if q <= 0.5:
                coverage = np.mean(y_true >= y_pred)
                expected = 1 - q
            else:
                coverage = np.mean(y_true <= y_pred)
                expected = q

            metrics[f'{qstr}_coverage'] = coverage
            metrics[f'{qstr}_coverage_error'] = abs(coverage - expected)
            metrics[f'{qstr}_pinball'] = mean_pinball_loss(y_true, y_pred, alpha=q)

        if 'q0.05' in predictions and 'q0.95' in predictions:
            interval_cov = np.mean(
                (y_true >= predictions['q0.05']) & (y_true <= predictions['q0.95'])
            )
            metrics['interval_90_coverage'] = interval_cov
            metrics['interval_90_error'] = abs(interval_cov - 0.90)

        return metrics

    def check_quality_gates(self, metrics: Dict[str, float]) -> bool:
        """Check if model meets quality standards."""
        interval = metrics.get('interval_90_coverage', 0.0)
        error = 0.90 - interval

        interval_pass = (
            error <= self.config.max_under_coverage and
            error >= -self.config.max_over_coverage
        )

        thresholds = {
            0.05: self.config.q05_max_error,
            0.50: self.config.q50_max_error,
            0.95: self.config.q95_max_error,
        }

        quantile_pass = all(
            metrics.get(f'q{q:.2f}_coverage_error', 1.0) <= thresh
            for q, thresh in thresholds.items()
        )

        passed = interval_pass and quantile_pass
        logger.info(f"Quality Gates: {'PASSED' if passed else 'FAILED'}")
        logger.info(f"  Interval: {interval:.1%} (target 90%)")

        return passed

    # ===== Serialization =====

    def save_model(self, path: str, metadata: Optional[Dict] = None) -> str:
        """Save model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        artifact = {
            'models': self.models,
            'preprocessor': self.preprocessor,
            'feature_names': self.feature_names,
            'conformal_adjustments': self.conformal_adjustments,
            'conformal_calibrator': self.conformal_calibrator,
            'evt_adjuster': self.evt_adjuster,
            'prob_classifier': self.prob_classifier,
            'prob_isotonic': self.prob_isotonic,
            'quantiles': self.quantiles,
            'horizon': self.horizon,
            'config': self.config.to_dict(),
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'metadata': metadata or {},
            'version': 'v3',
        }

        joblib.dump(artifact, path)
        logger.info(f"Model saved to: {path}")
        return path

    @classmethod
    def load_model(cls, path: str) -> 'OptimalCQF':
        """Load model from disk."""
        artifact = joblib.load(path)

        config = CQFConfig(**artifact.get('config', {}))
        cqf = cls(
            quantiles=artifact['quantiles'],
            horizon=artifact['horizon'],
            config=config,
        )

        cqf.models = artifact['models']
        cqf.preprocessor = artifact['preprocessor']
        cqf.feature_names = artifact['feature_names']
        cqf.conformal_adjustments = artifact['conformal_adjustments']
        cqf.conformal_calibrator = artifact.get('conformal_calibrator')
        cqf.evt_adjuster = artifact.get('evt_adjuster')
        cqf.prob_classifier = artifact.get('prob_classifier')
        cqf.prob_isotonic = artifact.get('prob_isotonic')

        logger.info(f"Model loaded from: {path}")
        return cqf


# ===== Data Loading =====

def load_and_prepare_data(data_file: str, config_file: str = "config.yaml") -> pd.DataFrame:
    """Load and prepare data with regime features."""
    logger.info(f"Loading: {data_file}")

    config = load_config(config_file)
    df = pd.read_csv(data_file, low_memory=False)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])

    # Normalize contract ID column
    if 'contractID' not in df.columns:
        for alt in ['contract_id', 'option_symbol']:
            if alt in df.columns:
                df = df.rename(columns={alt: 'contractID'})
                break

    df['contractID'] = df['contractID'].astype(str)
    df = df.sort_values(['date', 'contractID']).reset_index(drop=True)

    df, _ = preprocess_data(df, config, scaler=None)
    df = add_regime_features(df)
    df = add_realized_vol_features(df)

    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


# ===== Main =====

def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="CQF Training v3")
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--eval-data", required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="model_output/cqf_v3.joblib")
    parser.add_argument("--horizon", type=int, default=5)
    args = parser.parse_args()

    try:
        cqf = OptimalCQF(horizon=args.horizon)

        # Load and prepare
        train_data = load_and_prepare_data(args.train_data, args.config)
        train_data = cqf.calculate_delta_hedged_pnl(train_data)
        train_df, val_df, _ = cqf.create_time_splits(train_data)

        # Train
        cqf.create_preprocessor(train_df)
        cqf.train_quantile_models(train_df, val_df)
        cqf.calculate_conformal_adjustments()
        cqf.fit_probability_classifier()

        # Evaluate
        eval_data = load_and_prepare_data(args.eval_data, args.config)
        eval_data = cqf.calculate_delta_hedged_pnl(eval_data)

        qpreds = cqf.predict_quantiles(eval_data, apply_conformal=True)
        preds_dict = {col: qpreds[col].values for col in qpreds.columns}
        metrics = cqf.evaluate_coverage(eval_data['target_pnl_raw'].values, preds_dict)

        if cqf.check_quality_gates(metrics):
            cqf.save_model(args.output, {'train_data': args.train_data, 'metrics': metrics})
            logger.info("Training complete")
            return 0
        else:
            logger.error("Quality gates failed")
            return 1

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
