"""
Meta-Labeling Model for Options Trading

Based on López de Prado's meta-labeling approach from "Advances in Financial Machine Learning":
- Primary model (Ranker) provides direction/selection
- Meta-labeler predicts probability that the primary signal will be profitable
- Output: calibrated P(profit) used for position sizing

Pipeline: RANKER -> META-LABELER -> KELLY SIZER -> EXECUTION

Author: Generated with Claude Code
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import IsotonicRegression
from sklearn.metrics import (
    brier_score_loss, log_loss, roc_auc_score,
    precision_score, recall_score, f1_score,
    precision_recall_curve, average_precision_score
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib
import logging
import os
import argparse
from datetime import datetime
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field

from utils import load_config, preprocess_data

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True)
class MetaLabelerConfig:
    """Immutable configuration for meta-labeler training."""
    # Target settings
    horizon_days: int = 5
    profit_threshold: float = 0.0  # PnL > threshold = profitable

    # Training settings
    n_cv_splits: int = 5
    purge_days: int = 5  # Must be >= horizon_days
    early_stopping_rounds: int = 50

    # Model hyperparameters (tuned for binary classification)
    learning_rate: float = 0.05
    max_depth: int = 4
    n_estimators: int = 500
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    gamma: float = 0.1
    reg_alpha: float = 0.01
    reg_lambda: float = 0.1
    scale_pos_weight: float = 1.0  # Will be adjusted based on class imbalance

    # Top-K selection (simulate ranker output)
    top_k_per_day: int = 20

    # Quality gates
    min_auc: float = 0.55
    max_brier: float = 0.25
    min_precision_at_50: float = 0.4  # Precision when predicting top 50% as profitable


# Features used by meta-labeler
NUMERICAL_FEATURES = [
    # Option characteristics
    'days_to_exp', 'strike', 'last', 'bid', 'ask', 'volume', 'open_interest',
    'implied_volatility', 'delta', 'gamma', 'theta', 'vega', 'rho',
    'moneyness', 'relative_spread', 'bid_ask_spread',

    # Market regime features
    'spy_d_close', 'spy_d_SMA_50', 'spy_d_RSI', 'spy_d_MACD_Hist',
    'vix_d_close', 'spy_momentum',

    # Derived features
    'ofi', 'price_change_1d', 'iv_change_1d', 'zero_day_premium',
    'option_volume_oi_ratio', 'mispricing_ratio', 'risk_adjusted_signal',
    'iv_vix_ratio',

    # Rolling features (if available)
    'price__mean', 'price__standard_deviation',
]

# Meta-features: Ranker's output gets added to this
META_FEATURES = [
    'ranker_score',      # Raw ranker prediction
    'ranker_rank',       # Rank within day (1 = best)
    'ranker_percentile', # Percentile within day
]


# =============================================================================
# Data Processing
# =============================================================================

class PurgedTimeSeriesSplit:
    """Time series CV with purging to prevent lookahead bias."""

    def __init__(self, n_splits: int = 5, purge_days: int = 5):
        self.n_splits = n_splits
        self.purge_days = purge_days

    def split(self, dates: pd.Series):
        dates = pd.Series(dates).reset_index(drop=True)
        unique_dates = np.sort(dates.unique())
        n_dates = len(unique_dates)
        fold_size = n_dates // (self.n_splits + 1)

        for fold in range(self.n_splits):
            train_end = (fold + 1) * fold_size
            test_start = train_end
            test_end = min(test_start + fold_size, n_dates)

            train_dates = unique_dates[:train_end]
            test_dates = unique_dates[test_start:test_end]

            # Apply purge
            if self.purge_days > 0 and len(test_dates) > 0:
                purge_cutoff = pd.Timestamp(test_dates[0]) - pd.Timedelta(days=self.purge_days)
                train_mask = dates.apply(lambda x: pd.Timestamp(x) < purge_cutoff)
            else:
                train_mask = dates.isin(train_dates)

            test_mask = dates.isin(test_dates)

            train_idx = dates.index[train_mask].tolist()
            test_idx = dates.index[test_mask].tolist()

            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx


def load_raw_data(file_path: str) -> pd.DataFrame:
    """Load and prepare raw data."""
    logger.info(f"Loading: {file_path}")
    df = pd.read_csv(file_path, low_memory=False)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])

    # Standardize contract ID column
    if 'contractID' not in df.columns:
        for col in ['contract_id', 'option_symbol']:
            if col in df.columns:
                df = df.rename(columns={col: 'contractID'})
                break
        else:
            raise ValueError("Missing contract identifier column")

    df['contractID'] = df['contractID'].astype(str)
    return df.sort_values(['date', 'contractID']).reset_index(drop=True)


def calculate_delta_hedged_pnl(
    df: pd.DataFrame,
    horizon_days: int = 5
) -> pd.DataFrame:
    """
    Calculate delta-hedged P&L as the target.

    This represents the actual profit/loss from holding the option
    while maintaining a delta hedge.
    """
    price_col = 'last_raw' if 'last_raw' in df.columns else 'last'
    delta_col = 'delta'
    spy_col = 'spy_d_close'

    df = df.sort_values(['contractID', 'date']).copy()

    # Future option price
    df['future_price'] = df.groupby('contractID')[price_col].shift(-horizon_days)

    # Future SPY price for delta hedge
    df['future_spy'] = df.groupby('contractID')[spy_col].shift(-horizon_days)

    # Delta-hedged P&L: Option P&L - Delta * SPY P&L
    option_pnl = df['future_price'] - df[price_col]
    spy_pnl = df['future_spy'] - df[spy_col]

    # Normalize by option price for percentage P&L
    df['pnl_pct'] = np.where(
        df[price_col] > 0,
        (option_pnl - df[delta_col] * spy_pnl) / df[price_col],
        np.nan
    )

    # Raw P&L for Kelly calculations
    df['pnl_raw'] = option_pnl - df[delta_col] * spy_pnl

    return df.dropna(subset=['pnl_pct', 'pnl_raw'])


def create_binary_labels(
    df: pd.DataFrame,
    threshold: float = 0.0
) -> pd.DataFrame:
    """
    Create binary profit labels.

    y = 1 if PnL > threshold (profitable)
    y = 0 if PnL <= threshold (not profitable)
    """
    df = df.copy()
    df['profitable'] = (df['pnl_pct'] > threshold).astype(int)

    win_rate = df['profitable'].mean()
    logger.info(f"Win rate at threshold {threshold}: {win_rate:.1%}")

    return df


def add_ranker_features(
    df: pd.DataFrame,
    ranker_model_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Add ranker-related meta-features to the dataframe.

    These features (rank, percentile within day) provide useful context
    for the meta-labeler about relative option quality.

    CRITICAL: NO LEAKAGE - only uses features available at prediction time.

    Args:
        df: DataFrame with features
        ranker_model_path: Optional path to trained ranker model
    """
    df = df.copy()

    if ranker_model_path and os.path.exists(ranker_model_path):
        logger.info(f"Loading ranker from: {ranker_model_path}")
        ranker_artifacts = joblib.load(ranker_model_path)
        raise NotImplementedError("Ranker integration pending - using Greeks-based scoring")

    # === NO LEAKAGE: Use Greeks-based scoring ===
    # This provides useful meta-features about relative option quality within each day

    if all(col in df.columns for col in ['delta', 'gamma', 'theta', 'moneyness']):
        logger.info("Adding Greeks-based ranking features (no leakage)")

        # Normalize each component
        delta_vals = df['delta'].abs().fillna(0)
        delta_score = (delta_vals - delta_vals.mean()) / (delta_vals.std() + 1e-8)

        price_col = 'last_raw' if 'last_raw' in df.columns else 'last'
        theta_vals = -df['theta'].fillna(0) / (df[price_col].fillna(1) + 0.01)
        theta_score = (theta_vals - theta_vals.mean()) / (theta_vals.std() + 1e-8)

        moneyness_vals = 1 - df['moneyness'].abs().fillna(0.5)
        moneyness_score = (moneyness_vals - moneyness_vals.mean()) / (moneyness_vals.std() + 1e-8)

        # Combine into ranker score
        df['ranker_score'] = (
            0.4 * delta_score +
            0.3 * theta_score +
            0.3 * moneyness_score
        )
    else:
        # Fallback: use IV if available
        if 'implied_volatility' in df.columns:
            iv_vals = df['implied_volatility'].fillna(df['implied_volatility'].median())
            df['ranker_score'] = (iv_vals - iv_vals.mean()) / (iv_vals.std() + 1e-8)
        else:
            logger.warning("No ranking features available - using zeros")
            df['ranker_score'] = 0.0

    # Add ranking meta-features (useful for model to understand relative quality)
    df['ranker_rank'] = df.groupby('date')['ranker_score'].rank(
        ascending=False, method='average'
    )
    df['ranker_percentile'] = df.groupby('date')['ranker_score'].rank(pct=True)

    # Log statistics
    logger.info(f"Ranker score stats: mean={df['ranker_score'].mean():.4f}, std={df['ranker_score'].std():.4f}")

    return df


# =============================================================================
# Meta-Labeler Model
# =============================================================================

class MetaLabeler:
    """
    Meta-labeling classifier following López de Prado's approach.

    Predicts P(profit | ranker selected this option).
    Uses isotonic regression for probability calibration.
    """

    def __init__(self, config: MetaLabelerConfig):
        self.config = config
        self.classifier = None
        self.calibrator = None
        self.scaler = None
        self.imputer = None
        self.feature_names = None
        self.metrics = {}

    def _prepare_features(
        self,
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, List[str]]:
        """Prepare feature matrix from dataframe."""
        # Get available features
        available_features = []
        for f in NUMERICAL_FEATURES + META_FEATURES:
            if f in df.columns:
                available_features.append(f)

        self.feature_names = available_features
        X = df[available_features].values

        return X, available_features

    def fit(
        self,
        df: pd.DataFrame,
        eval_df: Optional[pd.DataFrame] = None
    ) -> 'MetaLabeler':
        """
        Train the meta-labeler on historical data.

        Trains on ALL data (not just ranker's selections) to:
        1. Learn general patterns between features and profitability
        2. Maximize training data utilization
        3. Generalize to any ranker's selections at inference time

        Args:
            df: Training data with features and profit labels
            eval_df: Optional held-out evaluation data
        """
        logger.info("Training meta-labeler on ALL data...")

        # Train on ALL data to learn general profitability patterns
        # At inference, model will predict P(profit) for ranker's selections
        df_train = df.copy()
        logger.info(f"Training on {len(df_train)} samples (all data)")

        X, feature_names = self._prepare_features(df_train)
        y = df_train['profitable'].values

        # Handle class imbalance
        pos_weight = (1 - y.mean()) / max(y.mean(), 1e-6)
        logger.info(f"Class balance - Profitable: {y.mean():.1%}, scale_pos_weight: {pos_weight:.2f}")

        # Preprocessing
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()

        X_imputed = self.imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_imputed)

        # Split for calibration
        dates = df_train['date'].values
        unique_dates = np.sort(np.unique(dates))
        n_dates = len(unique_dates)

        calib_start = int(n_dates * 0.8)
        calib_dates = unique_dates[calib_start:]
        train_mask = ~np.isin(dates, calib_dates)

        X_train = X_scaled[train_mask]
        y_train = y[train_mask]
        X_calib = X_scaled[~train_mask]
        y_calib = y[~train_mask]

        logger.info(f"Train: {len(X_train)}, Calibration: {len(X_calib)}")

        # Train XGBoost classifier
        self.classifier = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            learning_rate=self.config.learning_rate,
            max_depth=self.config.max_depth,
            n_estimators=self.config.n_estimators,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            gamma=self.config.gamma,
            reg_alpha=self.config.reg_alpha,
            reg_lambda=self.config.reg_lambda,
            scale_pos_weight=pos_weight,
            early_stopping_rounds=self.config.early_stopping_rounds,
            random_state=42,
            n_jobs=-1,
        )

        self.classifier.fit(
            X_train, y_train,
            eval_set=[(X_calib, y_calib)],
            verbose=False
        )

        # Get raw probabilities
        prob_calib = self.classifier.predict_proba(X_calib)[:, 1]

        # Calibrate with isotonic regression
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.calibrator.fit(prob_calib, y_calib)

        # Evaluate calibration improvement
        brier_raw = brier_score_loss(y_calib, prob_calib)
        prob_calibrated = self.calibrator.predict(prob_calib)
        brier_calibrated = brier_score_loss(y_calib, prob_calibrated)

        logger.info(f"Brier score: {brier_raw:.4f} -> {brier_calibrated:.4f} (after isotonic)")

        # Calculate metrics on calibration set
        self._calculate_metrics(y_calib, prob_calibrated, prefix='calib')

        # Evaluate on held-out data if provided
        if eval_df is not None:
            self._evaluate(eval_df)

        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict calibrated probability of profit.

        Returns:
            Array of P(profit) for each row
        """
        X, _ = self._prepare_features(df)
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)

        prob_raw = self.classifier.predict_proba(X_scaled)[:, 1]
        prob_calibrated = self.calibrator.predict(prob_raw)

        return prob_calibrated

    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        prefix: str = ''
    ) -> Dict[str, float]:
        """Calculate classification metrics."""
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = {
            f'{prefix}_auc': roc_auc_score(y_true, y_prob),
            f'{prefix}_brier': brier_score_loss(y_true, y_prob),
            f'{prefix}_log_loss': log_loss(y_true, y_prob),
            f'{prefix}_precision': precision_score(y_true, y_pred, zero_division=0),
            f'{prefix}_recall': recall_score(y_true, y_pred, zero_division=0),
            f'{prefix}_f1': f1_score(y_true, y_pred, zero_division=0),
            f'{prefix}_avg_precision': average_precision_score(y_true, y_prob),
        }

        self.metrics.update(metrics)

        logger.info(f"Metrics ({prefix}):")
        logger.info(f"  AUC: {metrics[f'{prefix}_auc']:.4f}")
        logger.info(f"  Brier: {metrics[f'{prefix}_brier']:.4f}")
        logger.info(f"  Precision: {metrics[f'{prefix}_precision']:.4f}")
        logger.info(f"  Recall: {metrics[f'{prefix}_recall']:.4f}")
        logger.info(f"  F1: {metrics[f'{prefix}_f1']:.4f}")

        return metrics

    def _evaluate(self, df: pd.DataFrame) -> Dict[str, float]:
        """Evaluate on held-out data (all data, not just selections)."""
        logger.info("Evaluating on held-out data...")

        # Evaluate on ALL data since model trained on all data
        y_true = df['profitable'].values
        y_prob = self.predict_proba(df)

        return self._calculate_metrics(y_true, y_prob, prefix='eval')

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from classifier."""
        importance = self.classifier.feature_importances_
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

    def check_quality_gates(self) -> bool:
        """Check if model passes quality gates."""
        passed = True

        # Check AUC
        auc_key = 'eval_auc' if 'eval_auc' in self.metrics else 'calib_auc'
        if self.metrics.get(auc_key, 0) < self.config.min_auc:
            logger.warning(f"FAILED: AUC {self.metrics[auc_key]:.4f} < {self.config.min_auc}")
            passed = False

        # Check Brier score
        brier_key = 'eval_brier' if 'eval_brier' in self.metrics else 'calib_brier'
        if self.metrics.get(brier_key, 1) > self.config.max_brier:
            logger.warning(f"FAILED: Brier {self.metrics[brier_key]:.4f} > {self.config.max_brier}")
            passed = False

        if passed:
            logger.info("Quality gates: PASSED")
        else:
            logger.warning("Quality gates: FAILED")

        return passed


# =============================================================================
# Training Pipeline
# =============================================================================

def train_meta_labeler(
    train_path: str,
    eval_path: Optional[str],
    output_path: str,
    config: MetaLabelerConfig,
    ranker_path: Optional[str] = None,
) -> MetaLabeler:
    """
    Full training pipeline for meta-labeler.

    Args:
        train_path: Path to training data CSV
        eval_path: Path to evaluation data CSV
        output_path: Where to save the trained model
        config: Training configuration
        ranker_path: Optional path to trained ranker model

    Returns:
        Trained MetaLabeler instance
    """
    # Load and preprocess training data
    df_raw = load_raw_data(train_path)

    cfg = load_config("config.yaml")
    df_processed, _ = preprocess_data(df_raw, cfg, scaler=None)

    logger.info(f"Loaded {len(df_processed)} rows, {len(df_processed.columns)} columns")

    # Calculate target (delta-hedged PnL)
    logger.info(f"Calculating delta-hedged PnL (horizon={config.horizon_days}d)")
    df_with_pnl = calculate_delta_hedged_pnl(df_processed, config.horizon_days)

    # Create binary labels
    df_labeled = create_binary_labels(df_with_pnl, config.profit_threshold)

    # Add ranker meta-features (useful context, but train on ALL data)
    df_with_features = add_ranker_features(
        df_labeled,
        ranker_model_path=ranker_path
    )

    # Log class distribution
    n_profitable = df_with_features['profitable'].sum()
    n_total = len(df_with_features)
    logger.info(f"Training data: {n_profitable} profitable / {n_total} total ({n_profitable/n_total:.1%})")

    # Load eval data if provided
    eval_df = None
    if eval_path:
        eval_raw = load_raw_data(eval_path)
        eval_processed, _ = preprocess_data(eval_raw, cfg, scaler=None)
        eval_with_pnl = calculate_delta_hedged_pnl(eval_processed, config.horizon_days)
        eval_labeled = create_binary_labels(eval_with_pnl, config.profit_threshold)
        eval_df = add_ranker_features(
            eval_labeled,
            ranker_model_path=ranker_path
        )

    # Train meta-labeler on ALL data
    model = MetaLabeler(config)
    model.fit(df_with_features, eval_df)

    # Check quality gates
    passed = model.check_quality_gates()

    # Show feature importance
    importance = model.get_feature_importance()
    logger.info("Top features:")
    for _, row in importance.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")

    # Save model
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    artifacts = {
        'classifier': model.classifier,
        'calibrator': model.calibrator,
        'scaler': model.scaler,
        'imputer': model.imputer,
        'feature_names': model.feature_names,
        'config': config,
        'metrics': model.metrics,
        'training_date': datetime.now().isoformat(),
    }

    joblib.dump(artifacts, output_path)
    logger.info(f"Model saved to: {output_path}")

    return model


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train Meta-Labeler for options trading"
    )
    parser.add_argument(
        '--train-data',
        type=str,
        required=True,
        help='Path to training data CSV'
    )
    parser.add_argument(
        '--eval-data',
        type=str,
        default=None,
        help='Path to evaluation data CSV'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='model_output/meta_labeler.joblib',
        help='Output path for model artifacts'
    )
    parser.add_argument(
        '--ranker-model',
        type=str,
        default=None,
        help='Path to trained ranker model (optional)'
    )
    parser.add_argument(
        '--horizon',
        type=int,
        default=5,
        help='P&L horizon in days'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=20,
        help='Number of top options ranker selects per day'
    )

    args = parser.parse_args()

    config = MetaLabelerConfig(
        horizon_days=args.horizon,
        top_k_per_day=args.top_k,
    )

    train_meta_labeler(
        train_path=args.train_data,
        eval_path=args.eval_data,
        output_path=args.output,
        config=config,
        ranker_path=args.ranker_model,
    )


if __name__ == '__main__':
    main()
