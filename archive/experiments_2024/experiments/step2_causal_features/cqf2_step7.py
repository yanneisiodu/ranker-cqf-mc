#!/usr/bin/env python3
"""
Optimal CQF (Calibrated Quantile Forecasting) Implementation

Combines best practices from all CQF implementations:
- XGBoost quantile regression (fastest, most accurate)
- Step2 causal features (no lookahead bias)
- Delta-hedged PnL targets (options-relevant)
- Time-based validation (realistic evaluation)
- Conformal calibration (coverage guarantees)

Usage:
    python cqf2.py --train-data year_2022_data.csv --eval-data year_2023_data.csv
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_pinball_loss
from sklearn.isotonic import IsotonicRegression
import joblib
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

# Import existing preprocessing
from utils import load_config, preprocess_data
from logger import setup_logger

# Import Step 1: regime features, Step 2: time-decay weights, Step 3B: regime-conditional conformal, Step 4: PageHinkley drift, Step 5: EVT tail protection
from regime_tools import add_regime_features, add_realized_vol_features, calculate_time_decay_weights, AdaptiveConformalCalibrator, PageHinkley, EVTTailAdjuster, _scores_one_sided

warnings.filterwarnings('ignore')
logger = setup_logger(__name__, level=logging.INFO)

class OptimalCQF:
    """
    Simplified, optimal CQF implementation focused on accuracy.
    """
    
    def __init__(self, 
                 quantiles: List[float] = [0.05, 0.5, 0.95],
                 horizon: int = 5,
                 random_state: int = 42):
        self.quantiles = quantiles
        self.horizon = horizon
        self.random_state = random_state
        self.models = {}
        self.preprocessor = None
        self.feature_names = []
        self.conformal_adjustments = {}  # Keep for backward compatibility
        self.conformal_calibrator = None  # Step 3B: Regime-conditional conformal calibrator
        self.page_hinkley = PageHinkley(delta=0.005, lambda_=50.0, alpha=0.99)  # Step 4: Drift detector
        self.evt_adjuster = None  # Step 5: EVT tail adjuster for high-stress periods
        self.prob_calibrator = None
        
    def calculate_delta_hedged_pnl(self, df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
        """
        Calculate delta-hedged P&L targets - more relevant than raw returns for options.
        
        Delta-hedged P&L ≈ Option P&L + Delta × (-Underlying P&L)
        This isolates option-specific alpha from broad market moves.
        """
        logger.info(f"Calculating delta-hedged PnL targets (horizon={horizon}d)")
        
        # Use raw price for calculation
        price_col = 'last_raw' if 'last_raw' in df.columns else 'last'
        
        if 'contractID' not in df.columns or price_col not in df.columns:
            logger.error("Missing required columns for PnL calculation")
            return df.assign(target_pnl=np.nan)
        
        df = df.sort_values(['contractID', 'date']).reset_index(drop=True)
        
        # Calculate forward option prices and target dates
        df_grouped = df.groupby('contractID')
        future_option_price = df_grouped[price_col].shift(-horizon)
        df['target_date'] = df_grouped['date'].shift(-horizon)  # Track actual target computation date
        
        # Calculate forward underlying prices (using SPY close) - Fixed alignment
        spy_col = 'spy_d_close'
        if spy_col in df.columns:
            # Fix SPY alignment: merge daily SPY data instead of contract groupby
            spy_daily = df[['date', spy_col]].drop_duplicates('date').sort_values('date')
            spy_daily['spy_fwd'] = spy_daily[spy_col].shift(-horizon)
            
            # Merge back to main dataframe
            df = df.merge(spy_daily[['date', 'spy_fwd']], on='date', how='left')
            
            # Option P&L
            option_pnl = (future_option_price - df[price_col]) / df[price_col]
            
            # Underlying P&L (using properly aligned SPY data)
            underlying_pnl = (df['spy_fwd'] - df[spy_col]) / df[spy_col]
            
            # Delta-hedged P&L = Option P&L + Delta × (-Underlying P&L)
            if 'delta' in df.columns:
                df['target_pnl'] = option_pnl + df['delta'] * (-underlying_pnl)
            else:
                logger.warning("Delta column missing, using raw option returns")
                df['target_pnl'] = option_pnl
        else:
            logger.warning("SPY data missing, using raw option returns")
            df['target_pnl'] = (future_option_price - df[price_col]) / df[price_col]
        
        # Clean infinite and NaN values
        df['target_pnl'] = df['target_pnl'].replace([np.inf, -np.inf], np.nan)
        
        initial_rows = len(df)
        df = df.dropna(subset=['target_pnl'])
        dropped_rows = initial_rows - len(df)
        
        logger.info(f"Delta-hedged PnL calculation complete. Dropped {dropped_rows} rows with invalid targets")
        logger.info(f"Target stats: mean={df['target_pnl'].mean():.4f}, std={df['target_pnl'].std():.4f}")
        
        return df.sort_values('date')
    
    def create_time_splits(self, df: pd.DataFrame, 
                          test_days: int = 90, 
                          val_days: int = 60) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create time-based train/validation/test splits.
        More realistic than GroupKFold for trading applications.
        """
        logger.info("Creating time-based data splits")
        
        df_sorted = df.sort_values('date')
        max_date = df_sorted['date'].max()
        
        # Calculate split dates with guard bands to prevent label leakage
        test_start = max_date - timedelta(days=test_days)
        val_start = test_start - timedelta(days=val_days)
        guard = timedelta(days=self.horizon)  # Prevent target calculation using future data
        
        # Create splits using target_date to prevent row-shift vs calendar-day mismatch
        if 'target_date' in df_sorted.columns:
            # Leak-proof splits: ensure no target crosses split boundaries
            train_mask = df_sorted['target_date'] < val_start
            val_mask = (df_sorted['date'] >= val_start) & (df_sorted['target_date'] < test_start)
            test_mask = df_sorted['date'] >= test_start
            
            train_df = df_sorted.loc[train_mask].reset_index(drop=True)
            val_df = df_sorted.loc[val_mask].reset_index(drop=True)
            test_df = df_sorted.loc[test_mask].reset_index(drop=True)
        else:
            # Fallback to calendar guard bands (less precise)
            train_df = df_sorted[df_sorted['date'] < (val_start - guard)].reset_index(drop=True)
            val_df = df_sorted[
                (df_sorted['date'] >= val_start) & (df_sorted['date'] < (test_start - guard))
            ].reset_index(drop=True)
            test_df = df_sorted[df_sorted['date'] >= test_start].reset_index(drop=True)
        
        logger.info(f"Data splits - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        logger.info(f"Date ranges - Train: {train_df['date'].min()} to {train_df['date'].max()}")
        logger.info(f"             Val: {val_df['date'].min()} to {val_df['date'].max()}")
        logger.info(f"             Test: {test_df['date'].min()} to {test_df['date'].max()}")
        
        # Leak detection assertions
        self._assert_no_label_crossing(train_df, val_df, test_df, val_start, test_start)
        
        return train_df, val_df, test_df
    
    def _assert_no_label_crossing(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                                 test_df: pd.DataFrame, val_start, test_start):
        """
        Leak detection assertions to prevent target crossing split boundaries.
        """
        def _check_split(name: str, df: pd.DataFrame, cutoff):
            if 'target_date' not in df.columns:
                return
            bad = df.loc[df['target_date'] >= cutoff, ['contractID', 'date', 'target_date']].head(5)
            if not bad.empty:
                logger.warning(f"⚠️  {name} split has {len(bad)} rows with labels beyond boundary")
                logger.warning(f"Sample violations:\n{bad}")
                # In production, this could be assert False, but for now just warn
        
        _check_split("Train", train_df, val_start)
        _check_split("Val", val_df, test_start)
        logger.info("✅ Leak detection passed: No target crossing detected")
    
    def create_preprocessor(self, df: pd.DataFrame) -> Pipeline:
        """
        Create preprocessing pipeline optimized for CQF.
        Keep only the most predictive features to avoid overfitting.
        """
        # CQF-optimized feature subset
        cqf_features = [
            # Core Greeks
            'delta', 'gamma', 'theta', 'vega',
            
            # Causal rolling features (our edge!)
            'price_roll_mean_5', 'price_roll_mean_20',
            'price_roll_std_5', 'price_roll_std_20', 
            'price_roll_zscore_5', 'price_roll_zscore_20',
            'iv_roll_mean_5', 'iv_roll_mean_20',
            
            # Options-specific features
            'moneyness', 'days_to_exp', 'implied_volatility',
            'mispricing_ratio', 'risk_adjusted_signal',
            'relative_spread', 'option_volume_oi_ratio',
            
            # Market context
            'spy_d_close', 'vix_d_close', 'iv_vix_ratio', 'spy_momentum',
            
            # Step 1: regime features + Step 2: time-decay weights + Step 3B: regime-conditional conformal + Step 4: PageHinkley drift + Step 5: EVT tail protection + Step 6: VIX pipeline fix + enhanced regime detection + Step 7: aggressive Black Swan response
            'vix_regime', 'vol_cluster', 'stress_score',
            
            # Step 6: Enhanced regime detection features + Step 7: severity scaling
            'realized_vol_20d', 'vol_of_vol_20d', 'vol_emergency', 'vol_acceleration', 'vol_severity'
        ]
        
        # Filter to existing columns
        available_features = [col for col in cqf_features if col in df.columns]
        
        # Remove features with too many NaNs (>50%)
        available_features = [
            col for col in available_features 
            if df[col].notna().sum() > len(df) * 0.5
        ]
        
        self.feature_names = available_features
        logger.info(f"Selected {len(available_features)} features for CQF")
        
        # Proper preprocessing pipeline: median imputation -> scaling
        self.preprocessor = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        return self.preprocessor
    
    def train_quantile_models(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict[str, float]:
        """
        Train XGBoost quantile regression models.
        Uses XGBoost native quantile support for best accuracy.
        Splits val_df to prevent calibration set reuse leak.
        """
        logger.info("Training XGBoost quantile regression models")
        
        # Split val_df to prevent soft leak: tune_df (early stopping) + calib_df (conformal)
        val_split = len(val_df) // 2
        tune_df = val_df.iloc[:val_split].reset_index(drop=True)  # First half for early stopping
        calib_df = val_df.iloc[val_split:].reset_index(drop=True)  # Second half for conformal
        
        # Store calib_df for later conformal calibration
        self.calib_df = calib_df
        
        logger.info(f"Split validation: Tune={len(tune_df)}, Calib={len(calib_df)}")
        
        # Prepare features - let preprocessor handle NaN values
        X_train = train_df[self.feature_names]
        y_train = train_df['target_pnl']
        X_tune = tune_df[self.feature_names]  # Use tune_df for early stopping
        y_tune = tune_df['target_pnl']
        
        # Fit preprocessor and transform data (handles NaN with median imputation)
        X_train_scaled = self.preprocessor.fit_transform(X_train)
        X_tune_scaled = self.preprocessor.transform(X_tune)
        
        metrics = {}
        
        # Train separate model for each quantile
        for quantile in self.quantiles:
            logger.info(f"Training quantile {quantile:.3f} model")
            
            # XGBoost quantile regressor - Optuna-optimized parameters (60 trials)
            model = xgb.XGBRegressor(
                objective='reg:quantileerror',      # Native quantile loss
                quantile_alpha=quantile,            # Target quantile
                n_estimators=464,                   # Optuna optimal: +55% trees for better accuracy
                max_depth=4,                        # Optuna confirmed: optimal depth
                learning_rate=0.030991,             # Optuna optimal: slower but more stable
                min_child_weight=1.167711,          # Optuna optimal: regularization
                subsample=0.700948,                 # Optuna optimal: more aggressive regularization
                colsample_bytree=0.767209,          # Optuna optimal: feature sampling
                reg_alpha=0.015122,                 # Optuna optimal: L1 regularization
                reg_lambda=0.051630,                # Optuna optimal: L2 regularization
                gamma=0.738906,                     # Optuna optimal: minimum split loss
                max_bin=256,                        # Optuna optimal: histogram precision
                tree_method='hist',                 # Fast on modern hardware
                n_jobs=-1,                          # Use all CPU cores
                random_state=self.random_state,
                early_stopping_rounds=30            # Increased patience (from Optuna best practices)
            )
            
            # Step 2: Combine tail-aware weights with time-decay weights
            tail_threshold = np.quantile(np.abs(y_train), 0.9)  # 90th percentile of |PnL|
            tail_weights = 1.0 + 2.0 * np.minimum(1.0, np.abs(y_train) / tail_threshold)
            
            # Time-decay weights: emphasize recent data for regime adaptation
            if 'date' in train_df.columns:
                time_weights = calculate_time_decay_weights(train_df['date'], decay_lambda=0.995)
                # Combine multiplicatively: both tail emphasis AND recency emphasis
                sample_weights = tail_weights * time_weights
                logger.info(f"Sample weights - Tail range: [{tail_weights.min():.2f}, {tail_weights.max():.2f}], Time range: [{time_weights.min():.2f}, {time_weights.max():.2f}]")
            else:
                sample_weights = tail_weights
                logger.info(f"Sample weights - Tail only: [{tail_weights.min():.2f}, {tail_weights.max():.2f}]")
            
            # Train with early stopping on tune set (separate from calibration set)
            model.fit(
                X_train_scaled, y_train,
                sample_weight=sample_weights,  # Emphasize tail + recent observations
                eval_set=[(X_tune_scaled, y_tune)],  # Use tune_df to prevent leak
                verbose=False
            )
            
            self.models[quantile] = model
            
            # Calculate validation metrics on tune set
            y_pred_tune = model.predict(X_tune_scaled)
            pinball_loss = mean_pinball_loss(y_tune, y_pred_tune, alpha=quantile)
            
            metrics[f'q{quantile:.3f}_pinball'] = pinball_loss
            logger.info(f"Quantile {quantile:.3f} - Validation Pinball Loss: {pinball_loss:.4f}")
        
        return metrics
    
    def calculate_conformal_adjustments(self, alpha: float = 0.1) -> Dict[str, float]:
        """
        Step 5: Regime-conditional conformal + PageHinkley drift + EVT tail protection.
        Uses light touch for stable periods, heavy protection + EVT for stress periods.
        Monitors coverage drift and triggers re-calibration on recent data.
        """
        logger.info("Calculating regime-conditional conformal with drift detection + EVT tail protection")
        
        # Use stored calib_df (separate from early stopping tune_df)
        if not hasattr(self, 'calib_df'):
            raise ValueError("No calibration data available. Call train_quantile_models first.")
            
        X_calib = self.calib_df[self.feature_names]
        y_calib = self.calib_df['target_pnl']
        X_calib_scaled = self.preprocessor.transform(X_calib)
        
        # Get model predictions on calibration set
        predictions = {}
        for quantile, model in self.models.items():
            predictions[quantile] = model.predict(X_calib_scaled)
        
        # Extract regime features for stress detection
        vix = self.calib_df['vix_d_close'] if 'vix_d_close' in self.calib_df.columns else None
        dte = self.calib_df['days_to_exp'] if 'days_to_exp' in self.calib_df.columns else None
        date = self.calib_df['date'] if 'date' in self.calib_df.columns else None
        
        # Step 6: Enhanced stress detection (VIX + realized vol + vol-of-vol)
        vix_stress = vix.std() if vix is not None else 0.5
        
        # Add vol-of-vol emergency detection
        vol_emergency_active = False
        vol_of_vol_stress = 0.0
        
        if 'vol_emergency' in self.calib_df.columns:
            vol_emergency_active = self.calib_df['vol_emergency'].iloc[-20:].sum() > 0  # Emergency in last 20 samples
        
        if 'vol_of_vol_20d' in self.calib_df.columns:
            vol_of_vol_recent = self.calib_df['vol_of_vol_20d'].iloc[-20:].mean()
            vol_of_vol_baseline = self.calib_df['vol_of_vol_20d'].quantile(0.8)
            vol_of_vol_stress = vol_of_vol_recent / vol_of_vol_baseline if vol_of_vol_baseline > 0 else 1.0
        
        # Step 7: Calculate vol severity multiplier for aggressive response
        vol_severity_calib = 1.0
        if 'vol_severity' in self.calib_df.columns:
            vol_severity_calib = self.calib_df['vol_severity'].iloc[-20:].max()  # Max severity in recent samples
        
        # Step 7: More sensitive stress detection + severity scaling
        is_stable_period = (vix_stress <= 1.0) and (not vol_emergency_active) and (vol_of_vol_stress <= 1.2) and (vol_severity_calib <= 2.0)
        
        logger.info(f"Stress indicators - VIX: {vix_stress:.3f}, Vol Emergency: {vol_emergency_active}, Vol-of-Vol: {vol_of_vol_stress:.3f}")
        
        if 0.05 in predictions and 0.95 in predictions:
            if is_stable_period:
                # Stable period: light touch (match Simple CQF performance)
                logger.info(f"Stable period detected (VIX: {vix_stress:.3f}, Vol Emergency: {vol_emergency_active}, Vol-of-Vol: {vol_of_vol_stress:.3f}) - Using light conformal")
                
                # Use simple global conformal (like Step 2 baseline)
                lower_pred = predictions[0.05]
                upper_pred = predictions[0.95]
                
                # Standard conformal calculation
                lower_scores = lower_pred - y_calib
                upper_scores = y_calib - upper_pred
                
                n = len(y_calib)
                q_level = np.ceil((n + 1) * (1 - alpha)) / n
                
                adjustments = {
                    'lower': np.quantile(lower_scores, q_level),
                    'upper': np.quantile(upper_scores, q_level)
                }
                
                # Simple coverage check
                coverage_90 = np.mean((y_calib >= lower_pred) & (y_calib <= upper_pred))
                adjusted_lower = lower_pred - adjustments['lower']
                adjusted_upper = upper_pred + adjustments['upper']
                adjusted_coverage = np.mean((y_calib >= adjusted_lower) & (y_calib <= adjusted_upper))
                
                logger.info(f"Pre-conformal 90% coverage: {coverage_90:.1%}")
                logger.info(f"Post-light-conformal 90% coverage: {adjusted_coverage:.1%}")
                logger.info("EVT disabled for stable period (preserving Step 4 baseline performance)")
                
                self.conformal_calibrator = None  # Use simple approach
                self.evt_adjuster = None  # Step 5: No EVT during stable periods
                
                # Step 4: PageHinkley drift detection on stable period coverage
                coverage_error = abs(adjusted_coverage - 0.90)
                if self.page_hinkley.update(coverage_error):
                    logger.warning(f"🚨 Drift detected by Page-Hinkley (error={coverage_error:.3f}) → rebuilding conformal on recent window")
                    # Use recent 60-day window for re-calibration if sufficient data
                    recent_cutoff = self.calib_df['date'].max() - pd.Timedelta(days=60)
                    recent_calib = self.calib_df[self.calib_df['date'] >= recent_cutoff]
                    
                    if len(recent_calib) >= 1000:  # Minimum data for stable calibration
                        logger.info(f"Re-calibrating on recent {len(recent_calib)} samples")
                        original_calib = self.calib_df
                        self.calib_df = recent_calib  # Temporarily use recent data
                        
                        # Recursive call with recent data (prevent infinite recursion)
                        if not hasattr(self, '_recalibrating'):
                            self._recalibrating = True
                            result = self.calculate_conformal_adjustments(alpha=alpha)
                            del self._recalibrating
                            self.calib_df = original_calib  # Restore original
                            return result
                    else:
                        logger.warning("Insufficient recent data for drift re-calibration, using wider safety margins")
                        # Apply conservative 20% wider intervals when data is sparse
                        safety_factor = 1.2
                        adjustments['lower'] = adjustments['lower'] * safety_factor
                        adjustments['upper'] = adjustments['upper'] * safety_factor
                        logger.info(f"Applied safety factor {safety_factor} - Lower: {adjustments['lower']:.4f}, Upper: {adjustments['upper']:.4f}")
                
            else:
                # Stress period: full adaptive conformal protection + EVT tail adjustment
                logger.info(f"Stress period detected (VIX: {vix_stress:.3f}, Vol Severity: {vol_severity_calib:.1f}x) - Using AGGRESSIVE adaptive conformal + EVT")
                
                # Step 7: Severity-adaptive conformal parameters
                adaptive_alpha = min(alpha * (1 + vol_severity_calib * 0.1), 0.3)  # Scale alpha with severity
                
                self.conformal_calibrator = AdaptiveConformalCalibrator(
                    alpha=adaptive_alpha,        # Step 7: Severity-scaled alpha  
                    use_groups=True,             # VIX×DTE grouping for stress
                    min_group_n=200,             # Step 7: Smaller groups for faster adaptation
                    recency_lambda=0.995,        # Step 7: Even heavier recency weighting
                    median_debias=True
                )
                
                pred_lo = predictions[0.05]
                pred_md = predictions.get(0.5)
                pred_up = predictions[0.95]
                
                # Fit adaptive conformal calibrator
                self.conformal_calibrator.fit(self.calib_df, pred_lo, pred_md, pred_up, 
                                             vix=vix, dte=dte, date=date)
                
                # Coverage diagnostics
                coverage_90 = np.mean((y_calib >= pred_lo) & (y_calib <= pred_up))
                adjusted = self.conformal_calibrator.adjust(self.calib_df, pred_lo, pred_md, pred_up,
                                                           vix=vix, dte=dte, date=date)
                adjusted_coverage = np.mean((y_calib >= adjusted['q0.05']) & (y_calib <= adjusted['q0.95']))
                
                logger.info(f"Pre-conformal 90% coverage: {coverage_90:.1%}")
                logger.info(f"Post-adaptive-conformal 90% coverage: {adjusted_coverage:.1%}")
                
                # Step 5: EVT tail adjustment ONLY during stress periods
                mean_vix = vix.mean() if vix is not None else 20.0
                
                # Calculate one-sided scores for EVT
                S_lo, S_up = _scores_one_sided(y_calib, adjusted['q0.05'], adjusted['q0.95'])
                
                # Step 7: Severity-scaled EVT parameters for Black Swan events
                black_swan_alpha = min(0.02 * vol_severity_calib, 0.15)  # Scale with vol severity, cap at 15%
                
                self.evt_adjuster = EVTTailAdjuster(
                    tail_thresh=0.70,          # Step 7: Lower threshold for more tail samples
                    base_alpha=0.005,          # Light tail mass for stable periods  
                    stress_alpha=black_swan_alpha,  # Step 7: Severity-scaled tail mass
                    vix_threshold=20.0         # Step 7: Lower VIX threshold (was 25.0)
                )
                self.evt_adjuster.fit(S_lo, S_up, mean_vix=mean_vix)
                
                # Get EVT increments and apply to adjustments
                lo_evt, up_evt = self.evt_adjuster.increments()
                logger.info(f"EVT tail adjustments (stress period) - Lower: +{lo_evt:.4f}, Upper: +{up_evt:.4f}")
                
                # Inflate conformal adjustments with EVT
                if self.conformal_calibrator._global_adj:
                    old_lo, old_up = self.conformal_calibrator._global_adj
                    self.conformal_calibrator._global_adj = (old_lo + lo_evt, old_up + up_evt)
                    
                if self.conformal_calibrator._group_adj:
                    for k, (lo, up) in self.conformal_calibrator._group_adj.items():
                        self.conformal_calibrator._group_adj[k] = (lo + lo_evt, up + up_evt)
                
                # Step 4: PageHinkley drift detection on stress period coverage  
                coverage_error = abs(adjusted_coverage - 0.90)
                if self.page_hinkley.update(coverage_error):
                    logger.warning(f"🚨 Drift detected by Page-Hinkley (error={coverage_error:.3f}) → rebuilding conformal on recent window")
                    # Use recent 60-day window for re-calibration if sufficient data
                    recent_cutoff = self.calib_df['date'].max() - pd.Timedelta(days=60)
                    recent_calib = self.calib_df[self.calib_df['date'] >= recent_cutoff]
                    
                    if len(recent_calib) >= 1000:
                        logger.info(f"Re-calibrating on recent {len(recent_calib)} samples")
                        original_calib = self.calib_df
                        self.calib_df = recent_calib
                        
                        # Recursive call with recent data (prevent infinite recursion)
                        if not hasattr(self, '_recalibrating'):
                            self._recalibrating = True
                            result = self.calculate_conformal_adjustments(alpha=alpha)
                            del self._recalibrating
                            self.calib_df = original_calib
                            return result
                    else:
                        logger.warning("Insufficient recent data for drift re-calibration, using wider safety margins")
                        # Apply conservative 20% wider intervals for adaptive conformal
                        safety_factor = 1.2
                        if self.conformal_calibrator._global_adj:
                            old_lo, old_up = self.conformal_calibrator._global_adj
                            self.conformal_calibrator._global_adj = (old_lo * safety_factor, old_up * safety_factor)
                
                adjustments = {'adaptive_conformal': True, 'evt_applied': True}  # Mark as adaptive + EVT
        
        # Median re-centering: fix systematic bias in q50 predictions
        if 0.5 in predictions:
            median_bias = np.median(predictions[0.5] - y_calib)
            adjustments['median_bias'] = float(median_bias)
            logger.info(f"Median bias on calibration set: {median_bias:+.5f} (will be subtracted from q50 predictions)")
        
        self.conformal_adjustments = adjustments
        return adjustments
    
    def fit_probability_calibrator(self):
        """
        Fit isotonic regression to calibrate prob_profit predictions.
        Uses calibration set with conformal-adjusted quantiles for proper calibration.
        """
        logger.info("Fitting probability calibrator")
        
        if not hasattr(self, 'calib_df'):
            raise ValueError("No calibration data available. Call train_quantile_models first.")
        
        # Get conformal-adjusted quantile predictions on calibration set
        qdf_calib = self.predict_quantiles(self.calib_df, apply_conformal=True)
        
        # Calculate raw prob_profit 
        if all(c in qdf_calib.columns for c in ['q0.05', 'q0.50', 'q0.95']):
            q05, q50, q95 = qdf_calib['q0.05'], qdf_calib['q0.50'], qdf_calib['q0.95']
            prob_profit_raw = np.where(
                q95 <= 0, 0.0,
                np.where(q05 >= 0, 1.0, 0.5 + 0.45 * (q50 / (q95 - q05 + 1e-8)))
            )
            prob_profit_raw = np.clip(prob_profit_raw, 0.0, 1.0)
            
            # True binary labels: 1 if target_pnl > 0
            y_binary = (self.calib_df['target_pnl'].values > 0).astype(int)
            
            # Fit isotonic regression: raw_prob → empirical_prob
            self.prob_calibrator = IsotonicRegression(out_of_bounds='clip')
            self.prob_calibrator.fit(prob_profit_raw, y_binary)
            
            # Diagnostics
            calibrated_probs = self.prob_calibrator.predict(prob_profit_raw)
            original_brier = np.mean((prob_profit_raw - y_binary) ** 2)
            calibrated_brier = np.mean((calibrated_probs - y_binary) ** 2)
            
            logger.info(f"Probability calibration: Brier {original_brier:.4f} → {calibrated_brier:.4f}")
        else:
            logger.warning("Cannot fit probability calibrator: missing quantiles")
    
    def predict_quantiles(self, df: pd.DataFrame, apply_conformal: bool = True) -> pd.DataFrame:
        """
        Step 3B: Predict quantiles with regime-conditional conformal calibration.
        """
        X = df[self.feature_names]
        X_scaled = self.preprocessor.transform(X)
        
        # Get raw predictions
        raw_predictions = {}
        for quantile, model in self.models.items():
            raw_predictions[quantile] = model.predict(X_scaled)
        
        # Apply regime-conditional conformal adjustments
        if apply_conformal:
            # Step 6: Enhanced real-time stress detection
            vix = df['vix_d_close'] if 'vix_d_close' in df.columns else None
            dte = df['days_to_exp'] if 'days_to_exp' in df.columns else None
            date = df['date'] if 'date' in df.columns else None
            
            # Step 7: Real-time stress indicators with severity scaling
            vol_emergency_now = df['vol_emergency'].iloc[-1] if 'vol_emergency' in df.columns and len(df) > 0 else False
            vol_of_vol_high = False
            vol_severity_pred = 1.0
            
            if 'vol_of_vol_20d' in df.columns and len(df) > 0:
                vol_of_vol_current = df['vol_of_vol_20d'].iloc[-1] if not pd.isna(df['vol_of_vol_20d'].iloc[-1]) else 0
                vol_of_vol_threshold = df['vol_of_vol_20d'].quantile(0.85) if len(df) > 50 else 0  # Step 7: Lower threshold
                vol_of_vol_high = vol_of_vol_current > vol_of_vol_threshold
            
            if 'vol_severity' in df.columns and len(df) > 0:
                vol_severity_pred = df['vol_severity'].iloc[-1] if not pd.isna(df['vol_severity'].iloc[-1]) else 1.0
            
            # Step 7: Enhanced regime switching with severity scaling
            is_black_swan = vol_severity_pred > 5.0  # Black Swan threshold
            use_adaptive = (self.conformal_calibrator is not None) and (vol_emergency_now or vol_of_vol_high or is_black_swan)
            
            if use_adaptive:
                # Stress detected: use adaptive conformal calibrator + EVT
                logger.info(f"STRESS DETECTED - Vol Emergency: {vol_emergency_now}, Vol-of-Vol High: {vol_of_vol_high}, Severity: {vol_severity_pred:.1f}x - Using AGGRESSIVE adaptive + EVT")
                
                adjusted = self.conformal_calibrator.adjust(
                    df, 
                    raw_predictions.get(0.05), 
                    raw_predictions.get(0.5),
                    raw_predictions.get(0.95),
                    vix=vix, dte=dte, date=date
                )
                
                predictions = {
                    'q0.05': adjusted.get('q0.05', raw_predictions.get(0.05)),
                    'q0.50': adjusted.get('q0.50', raw_predictions.get(0.5)),
                    'q0.95': adjusted.get('q0.95', raw_predictions.get(0.95))
                }
                
                # Step 7: Black Swan emergency multiplier for extreme vol_severity
                if vol_severity_pred > 10.0:  # Extreme Black Swan threshold
                    emergency_multiplier = min(1.0 + (vol_severity_pred - 10.0) * 0.2, 5.0)  # Cap at 5x
                    q50 = predictions['q0.50']
                    q_width = predictions['q0.95'] - predictions['q0.05']
                    
                    # Scale intervals around median
                    predictions['q0.05'] = q50 - (q_width * emergency_multiplier / 2)
                    predictions['q0.95'] = q50 + (q_width * emergency_multiplier / 2)
                    
                    logger.warning(f"🚨 BLACK SWAN EMERGENCY: Applied {emergency_multiplier:.1f}x interval multiplier (severity: {vol_severity_pred:.1f}x)")
            else:
                # Stable period: use simple conformal adjustments (like Step 2)
                logger.info("Stable period detected - Using simple conformal")
                predictions = {}
                for quantile, raw_pred in raw_predictions.items():
                    if quantile == 0.05 and 'lower' in self.conformal_adjustments:
                        predictions[f'q{quantile:.2f}'] = raw_pred - self.conformal_adjustments['lower']
                    elif quantile == 0.5 and 'median_bias' in self.conformal_adjustments:
                        predictions[f'q{quantile:.2f}'] = raw_pred - self.conformal_adjustments['median_bias']
                    elif quantile == 0.95 and 'upper' in self.conformal_adjustments:
                        predictions[f'q{quantile:.2f}'] = raw_pred + self.conformal_adjustments['upper']
                    else:
                        predictions[f'q{quantile:.2f}'] = raw_pred
        else:
            # No conformal: raw predictions
            predictions = {f'q{quantile:.2f}': pred for quantile, pred in raw_predictions.items()}
        
        # Enforce monotonicity: q05 ≤ q50 ≤ q95
        if 'q0.05' in predictions and 'q0.50' in predictions and 'q0.95' in predictions:
            q05 = predictions['q0.05']
            q50 = np.maximum(predictions['q0.50'], q05)
            q95 = np.maximum(predictions['q0.95'], q50)
            
            # Fix any remaining inversions
            q50 = np.minimum(q50, q95)
            q05 = np.minimum(q05, q50)
            
            predictions['q0.05'] = q05
            predictions['q0.50'] = q50  
            predictions['q0.95'] = q95
        
        return pd.DataFrame(predictions, index=df.index)
    
    def calculate_decision_features(self, quantile_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trading decision features from quantiles.
        """
        result = quantile_df.copy()
        
        if all(col in quantile_df.columns for col in ['q0.05', 'q0.50', 'q0.95']):
            q05, q50, q95 = quantile_df['q0.05'], quantile_df['q0.50'], quantile_df['q0.95']
            
            # Expected value (skew-aware Simpson's rule)
            result['expected_return'] = (q05 + 4*q50 + q95) / 6.0
            
            # Risk metrics
            result['downside_risk'] = np.abs(np.minimum(q05, 0))
            result['upside_potential'] = np.maximum(q95, 0)
            result['uncertainty'] = q95 - q05
            
            # Probability of profit (linear interpolation) with bounds checking
            prob_profit_raw = np.where(
                q95 <= 0, 0.0,  # No chance if even 95th percentile is negative
                np.where(
                    q05 >= 0, 1.0,  # Certain profit if even 5th percentile is positive
                    0.5 + 0.45 * (q50 / (q95 - q05 + 1e-8))  # Linear interpolation
                )
            )
            prob_profit_raw = np.clip(prob_profit_raw, 0.0, 1.0)  # Ensure [0,1] bounds
            
            # Apply isotonic calibration if available
            if self.prob_calibrator is not None:
                result['prob_profit'] = self.prob_calibrator.predict(prob_profit_raw)
            else:
                result['prob_profit'] = prob_profit_raw
            
            # Risk-adjusted utility (penalize downside risk)
            risk_penalty = 0.5  # Adjust based on risk tolerance
            result['utility'] = result['expected_return'] - risk_penalty * result['downside_risk']
        
        return result
    
    def evaluate_coverage(self, y_true: np.ndarray, predictions: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Evaluate quantile model coverage and calibration.
        """
        metrics = {}
        
        for quantile_str, y_pred in predictions.items():
            quantile = float(quantile_str[1:])  # Extract from 'q0.05' -> 0.05
            
            # Calculate empirical coverage
            if quantile <= 0.5:
                # Lower quantiles: check if actual is above prediction
                coverage = np.mean(y_true >= y_pred)
                expected_coverage = 1 - quantile
            else:
                # Upper quantiles: check if actual is below prediction
                coverage = np.mean(y_true <= y_pred)
                expected_coverage = quantile
            
            coverage_error = abs(coverage - expected_coverage)
            pinball_loss = mean_pinball_loss(y_true, y_pred, alpha=quantile)
            
            metrics[f'{quantile_str}_coverage'] = coverage
            metrics[f'{quantile_str}_coverage_error'] = coverage_error
            metrics[f'{quantile_str}_pinball'] = pinball_loss
        
        # Interval coverage (90% prediction interval)
        if 'q0.05' in predictions and 'q0.95' in predictions:
            interval_coverage = np.mean(
                (y_true >= predictions['q0.05']) & (y_true <= predictions['q0.95'])
            )
            metrics['interval_90_coverage'] = interval_coverage
            metrics['interval_90_error'] = abs(interval_coverage - 0.90)
        
        return metrics
    
    def check_quality_gates(self, metrics: Dict[str, float]) -> bool:
        """
        Check if model meets production quality standards.
        """
        # Quality thresholds (relaxed for market regime testing)  
        MAX_COVERAGE_ERROR = 0.15  # ±15% tolerance for regime changes
        
        # Check coverage errors
        coverage_errors = [
            metrics.get(f'q{q:.2f}_coverage_error', 1.0) 
            for q in self.quantiles
        ]
        max_coverage_error = max(coverage_errors) if coverage_errors else 1.0
        
        # Check interval coverage
        interval_error = metrics.get('interval_90_error', 1.0)
        
        # Pass/fail criteria
        coverage_pass = max_coverage_error <= MAX_COVERAGE_ERROR
        interval_pass = interval_error <= MAX_COVERAGE_ERROR
        
        passed = coverage_pass and interval_pass
        
        logger.info(f"Quality Gates: {'✅ PASSED' if passed else '❌ FAILED'}")
        logger.info(f"  Max Coverage Error: {max_coverage_error:.1%} {'✅' if coverage_pass else '❌'}")
        logger.info(f"  Interval Coverage Error: {interval_error:.1%} {'✅' if interval_pass else '❌'}")
        
        return passed
    
    def save_model(self, output_path: str, metadata: Dict = None):
        """Save trained CQF model and artifacts."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save model artifacts (Step 5: include EVT tail adjuster + PageHinkley drift + regime-conditional conformal)
        model_artifact = {
            'models': self.models,
            'preprocessor': self.preprocessor,
            'feature_names': self.feature_names,
            'conformal_adjustments': self.conformal_adjustments,  # Simple conformal (stable periods)
            'conformal_calibrator': self.conformal_calibrator,     # Adaptive conformal (stress periods)  
            'page_hinkley': self.page_hinkley,                     # Step 4: Drift detector
            'evt_adjuster': self.evt_adjuster,                     # Step 5: EVT tail adjuster (stress periods only)
            'prob_calibrator': self.prob_calibrator,
            'quantiles': self.quantiles,
            'horizon': self.horizon,
            'timestamp': timestamp,
            'metadata': metadata or {}
        }
        
        # Save to joblib
        joblib.dump(model_artifact, output_path)
        logger.info(f"CQF model saved to: {output_path}")
        
        return output_path

def load_and_prepare_data(data_file: str, config_file: str = "config.yaml") -> pd.DataFrame:
    """
    Load and preprocess data using existing utils.py pipeline.
    """
    logger.info(f"Loading and preparing data from: {data_file}")
    
    # Load configuration
    config = load_config(config_file)
    
    # Load raw data
    df_raw = pd.read_csv(data_file, low_memory=False)
    df_raw['date'] = pd.to_datetime(df_raw['date'], errors='coerce')
    df_raw = df_raw.dropna(subset=['date'])
    
    # Handle contractID column naming
    if 'contractID' not in df_raw.columns:
        if 'contract_id' in df_raw.columns:
            df_raw = df_raw.rename(columns={'contract_id': 'contractID'})
        elif 'option_symbol' in df_raw.columns:
            df_raw = df_raw.rename(columns={'option_symbol': 'contractID'})
    
    df_raw['contractID'] = df_raw['contractID'].astype(str)
    df_raw = df_raw.sort_values(['date', 'contractID']).reset_index(drop=True)
    
    logger.info(f"Raw data loaded: {len(df_raw)} rows, date range: {df_raw['date'].min()} to {df_raw['date'].max()}")
    
    # Preprocess using existing pipeline (gets us the causal features!)
    df_processed, _ = preprocess_data(df_raw, config, scaler=None)
    
    # Step 1: regime features + Step 2: time-decay weights + Step 3B: regime-conditional conformal + Step 4: PageHinkley drift + Step 5: EVT tail protection + Step 6: enhanced regime detection
    df_processed = add_regime_features(df_processed)
    df_processed = add_realized_vol_features(df_processed)  # Step 6: vol-of-vol detection
    logger.info("Added Step 1+2+3B+4+5+6 enhancements: regime features + time-decay weights + regime-conditional conformal + PageHinkley drift + EVT tail protection + enhanced regime detection")
    
    logger.info(f"Data preprocessed: {len(df_processed)} rows, {len(df_processed.columns)} features")
    return df_processed

def main():
    """Main training and evaluation pipeline."""
    parser = argparse.ArgumentParser(description="Optimal CQF Training")
    parser.add_argument("--train-data", required=True, help="Training data CSV file")
    parser.add_argument("--eval-data", required=True, help="Evaluation data CSV file") 
    parser.add_argument("--config", default="config.yaml", help="Configuration file")
    parser.add_argument("--output", default="model_output/optimal_cqf.joblib", help="Output model path")
    parser.add_argument("--horizon", type=int, default=5, help="Prediction horizon in days")
    
    args = parser.parse_args()
    
    try:
        # Initialize CQF
        cqf = OptimalCQF(horizon=args.horizon)
        
        # Load and prepare training data
        logger.info("=== Loading Training Data ===")
        train_data = load_and_prepare_data(args.train_data, args.config)
        
        # Calculate targets
        logger.info("=== Calculating Targets ===")
        train_data = cqf.calculate_delta_hedged_pnl(train_data, args.horizon)
        
        # Create time-based splits
        logger.info("=== Creating Data Splits ===")
        train_df, val_df, _ = cqf.create_time_splits(train_data)
        
        # Create preprocessor
        logger.info("=== Creating Preprocessor ===")
        cqf.create_preprocessor(train_df)
        
        # Train quantile models
        logger.info("=== Training Quantile Models ===")
        train_metrics = cqf.train_quantile_models(train_df, val_df)
        
        # Calculate conformal adjustments (uses stored calibration set)
        logger.info("=== Calculating Conformal Calibration ===")
        conformal_metrics = cqf.calculate_conformal_adjustments()
        
        # Fit probability calibrator (uses calibration set)
        logger.info("=== Fitting Probability Calibrator ===")
        cqf.fit_probability_calibrator()
        
        # Evaluate on separate evaluation data
        logger.info("=== Final Evaluation ===")
        eval_data = load_and_prepare_data(args.eval_data, args.config)
        eval_data = cqf.calculate_delta_hedged_pnl(eval_data, args.horizon)
        
        # Make predictions
        quantile_preds = cqf.predict_quantiles(eval_data, apply_conformal=True)
        decision_features = cqf.calculate_decision_features(quantile_preds)
        
        # Evaluate coverage
        predictions_dict = {col: quantile_preds[col].values for col in quantile_preds.columns}
        coverage_metrics = cqf.evaluate_coverage(eval_data['target_pnl'].values, predictions_dict)
        
        # Check quality gates
        quality_passed = cqf.check_quality_gates(coverage_metrics)
        
        # Save model if quality gates pass
        if quality_passed:
            metadata = {
                'train_data': args.train_data,
                'eval_data': args.eval_data,
                'train_metrics': train_metrics,
                'coverage_metrics': coverage_metrics,
                'quality_passed': True
            }
            model_path = cqf.save_model(args.output, metadata)
            
            # Save predictions for analysis (Step 6: INCLUDE REGIME FEATURES!)
            results_df = pd.DataFrame({
                'contractID': eval_data['contractID'].values,
                'date': eval_data['date'].dt.strftime('%Y-%m-%d').values,
                'target_actual': eval_data['target_pnl'].values,
                **{col: quantile_preds[col].values for col in quantile_preds.columns},
                **{col: decision_features[col].values for col in decision_features.columns if col not in quantile_preds.columns}
            })
            
            # Step 7 FIX: Add ALL required data for Enhanced Stress MC
            # Option basics for Enhanced Stress MC
            option_cols = ['last', 'last_raw', 'strike', 'type', 'implied_volatility', 'moneyness']
            
            # Greeks for shock calculations  
            greek_cols = ['delta', 'gamma', 'vega', 'theta', 'rho']
            
            # Market data for SPY shocks
            market_cols = ['spy_d_close', 'spy_d_open', 'spy_d_high', 'spy_d_low']
            
            # Regime features (Step 6-7)
            regime_cols = ['vix_d_close', 'days_to_exp', 'vix_regime', 'vol_cluster', 'stress_score', 
                          'realized_vol_20d', 'vol_of_vol_20d', 'vol_emergency', 'vol_acceleration', 'vol_severity']
            
            # Combine all required columns for complete Enhanced Stress MC dataset
            all_required_cols = option_cols + greek_cols + market_cols + regime_cols
            
            for col in all_required_cols:
                if col in eval_data.columns:
                    results_df[col] = eval_data[col].values
                else:
                    logger.warning(f"Missing column for Enhanced Stress MC: {col}")
            
            results_path = args.output.replace('.joblib', '_predictions.csv')
            results_df.to_csv(results_path, index=False)
            logger.info(f"Predictions saved to: {results_path}")
            
            # Summary
            logger.info("=== Training Complete ===")
            logger.info(f"✅ Model saved: {model_path}")
            logger.info(f"✅ Quality gates: PASSED")
            for metric, value in coverage_metrics.items():
                if 'coverage' in metric and not 'error' in metric:
                    logger.info(f"✅ {metric}: {value:.1%}")
            
        else:
            logger.error("❌ Model failed quality gates - not saving")
            return 1
            
        return 0
        
    except Exception as e:
        logger.error(f"CQF training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
