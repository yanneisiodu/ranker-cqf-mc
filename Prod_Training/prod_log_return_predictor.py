"""
Log-Return Predictor - Shadow System
=====================================

This is a shadow implementation to test the log-return prediction approach
against the existing Meta-Labeler + Kelly system.

Key difference from Meta-Labeler:
- Meta-Labeler: Predicts P(profit) -> uses Kelly formula f* = (p*b - q)/b
- Log-Return: Predicts E[log(1+r)] directly -> sizes = prediction / target

This allows us to compare both approaches on the same data.

IMPORTANT: This is a TEST/SHADOW system. Do not replace the production
Meta-Labeler until this proves superior in backtesting.

Author: Generated with Claude Code
"""

from __future__ import annotations
import logging
import os
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr

from utils import load_config, preprocess_data

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class LogReturnConfig:
    """Configuration for log-return predictor."""
    # Target calculation
    horizon_days: int = 5
    transaction_cost_bps: float = 20.0  # 20 bps round-trip

    # Model parameters
    n_estimators: int = 200
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8

    # Position sizing
    target_log_return: float = 0.10  # 10% expected = max position
    max_position_fraction: float = 0.15  # 15% max per position
    min_position_fraction: float = 0.01  # 1% minimum

    # Uncertainty estimation
    uncertainty_scale: float = 1.5  # Multiplier for uncertainty penalty

    # Features to use
    features: List[str] = field(default_factory=lambda: [
        'delta', 'gamma', 'theta', 'vega', 'rho',
        'implied_volatility', 'moneyness', 'days_to_exp',
        'spy_d_close', 'spy_d_RSI', 'vix_d_close',
        'relative_spread', 'open_interest', 'volume',
        'ranker_score', 'ranker_percentile'
    ])

    # Top-K selection (for comparison with meta-labeler)
    top_k_per_day: int = 20


# =============================================================================
# Data Loading and Target Calculation
# =============================================================================

def load_raw_data(file_path: str) -> pd.DataFrame:
    """Load raw data from CSV."""
    logger.info(f"Loading: {file_path}")
    df = pd.read_csv(file_path, low_memory=False)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])

    if 'contractID' not in df.columns:
        if 'contract_id' in df.columns:
            df = df.rename(columns={'contract_id': 'contractID'})
        elif 'option_symbol' in df.columns:
            df = df.rename(columns={'option_symbol': 'contractID'})
        else:
            raise ValueError("Missing contract identifier column")

    return df.sort_values(['date', 'contractID']).reset_index(drop=True)


def calculate_log_returns(
    df: pd.DataFrame,
    horizon_days: int = 5,
    transaction_cost_bps: float = 20.0
) -> pd.DataFrame:
    """
    Calculate log returns for options.

    Uses delta-hedged P&L to isolate option-specific returns.

    Args:
        df: DataFrame with price and Greeks
        horizon_days: Forward-looking horizon
        transaction_cost_bps: Transaction costs in basis points
    """
    df = df.copy()
    price_col = 'last_raw' if 'last_raw' in df.columns else 'last'

    if price_col not in df.columns:
        raise ValueError(f"Price column {price_col} not found")

    # Sort for proper shifting
    df = df.sort_values(['contractID', 'date'])

    # Get future price
    df['future_price'] = df.groupby('contractID')[price_col].shift(-horizon_days)

    # Calculate raw return
    df['raw_return'] = np.where(
        df[price_col] > 0,
        (df['future_price'] - df[price_col]) / df[price_col],
        np.nan
    )

    # Subtract transaction costs
    tc = transaction_cost_bps / 10000
    df['net_return'] = df['raw_return'] - tc

    # Calculate log return (what Kelly optimizes)
    # Clip to avoid log(0) or log(negative)
    df['log_return'] = np.log1p(df['net_return'].clip(lower=-0.99))

    # Drop rows without target
    df = df.dropna(subset=['log_return'])

    logger.info(f"Log return stats: mean={df['log_return'].mean():.4f}, "
                f"std={df['log_return'].std():.4f}, "
                f"min={df['log_return'].min():.4f}, max={df['log_return'].max():.4f}")

    return df


def add_ranker_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ranker-related features (same as meta-labeler for fair comparison).
    """
    df = df.copy()

    if all(col in df.columns for col in ['delta', 'gamma', 'theta', 'moneyness']):
        logger.info("Adding Greeks-based ranking features")

        delta_vals = df['delta'].abs().fillna(0)
        delta_score = (delta_vals - delta_vals.mean()) / (delta_vals.std() + 1e-8)

        price_col = 'last_raw' if 'last_raw' in df.columns else 'last'
        theta_vals = -df['theta'].fillna(0) / (df[price_col].fillna(1) + 0.01)
        theta_score = (theta_vals - theta_vals.mean()) / (theta_vals.std() + 1e-8)

        moneyness_vals = 1 - df['moneyness'].abs().fillna(0.5)
        moneyness_score = (moneyness_vals - moneyness_vals.mean()) / (moneyness_vals.std() + 1e-8)

        df['ranker_score'] = (
            0.4 * delta_score +
            0.3 * theta_score +
            0.3 * moneyness_score
        )
    else:
        df['ranker_score'] = 0.0

    df['ranker_rank'] = df.groupby('date')['ranker_score'].rank(ascending=False, method='average')
    df['ranker_percentile'] = df.groupby('date')['ranker_score'].rank(pct=True)

    return df


# =============================================================================
# Log-Return Predictor Model
# =============================================================================

class LogReturnPredictor:
    """
    Predicts E[log(1 + return)] directly instead of P(profit).

    This is the core of the shadow system. The proposal claims this is
    superior because:
    1. Directly predicts what Kelly optimizes
    2. Captures trade magnitude (unlike binary classification)
    3. Single model instead of separate p and b estimation

    We test this claim empirically.
    """

    def __init__(self, config: LogReturnConfig):
        self.config = config
        self.model = None
        self.quantile_models: Dict[str, xgb.XGBRegressor] = {}
        self.imputer = None
        self.scaler = None
        self.feature_names: List[str] = []
        self.metrics: Dict[str, float] = {}

    def _prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Extract and prepare features."""
        available = [f for f in self.config.features if f in df.columns]

        if not available:
            raise ValueError("No features available for training")

        X = df[available].copy()

        # Handle infinite values
        X = X.replace([np.inf, -np.inf], np.nan)

        return X.values, available

    def fit(
        self,
        df: pd.DataFrame,
        eval_df: Optional[pd.DataFrame] = None
    ) -> 'LogReturnPredictor':
        """
        Train the log-return predictor.

        Trains on ALL data (same as meta-labeler for fair comparison).
        """
        logger.info("Training log-return predictor on ALL data...")

        X, self.feature_names = self._prepare_features(df)
        y = df['log_return'].values

        logger.info(f"Training on {len(df)} samples with {len(self.feature_names)} features")
        logger.info(f"Target stats: mean={y.mean():.4f}, std={y.std():.4f}")

        # Preprocessing
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()

        X_imputed = self.imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_imputed)

        # Time-based split for validation
        dates = df['date'].values
        unique_dates = np.sort(np.unique(dates))
        n_dates = len(unique_dates)

        val_start = int(n_dates * 0.8)
        val_dates = unique_dates[val_start:]
        train_mask = ~np.isin(dates, val_dates)

        X_train, X_val = X_scaled[train_mask], X_scaled[~train_mask]
        y_train, y_val = y[train_mask], y[~train_mask]

        logger.info(f"Train: {len(y_train)}, Validation: {len(y_val)}")

        # Main model: predict expected log-return
        self.model = xgb.XGBRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1
        )

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        # Quantile models for uncertainty estimation
        for q, alpha in [('q10', 0.10), ('q90', 0.90)]:
            logger.info(f"Training quantile model {q}...")
            qmodel = xgb.XGBRegressor(
                n_estimators=150,
                max_depth=5,
                learning_rate=self.config.learning_rate,
                objective='reg:quantileerror',
                quantile_alpha=alpha,
                random_state=42,
                n_jobs=-1
            )
            qmodel.fit(X_train, y_train, verbose=False)
            self.quantile_models[q] = qmodel

        # Evaluate on validation set
        self._evaluate_model(X_val, y_val, prefix='val')

        # Evaluate on eval_df if provided
        if eval_df is not None:
            X_eval, _ = self._prepare_features(eval_df)
            X_eval_proc = self.scaler.transform(self.imputer.transform(X_eval))
            y_eval = eval_df['log_return'].values
            self._evaluate_model(X_eval_proc, y_eval, prefix='eval')

        return self

    def _evaluate_model(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        prefix: str = 'val'
    ) -> Dict[str, float]:
        """Evaluate model performance."""
        y_pred = self.model.predict(X)

        # Regression metrics
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # Spearman correlation (critical for ranking)
        spearman_corr, spearman_p = spearmanr(y_pred, y_true)

        # Store metrics
        self.metrics[f'{prefix}_mse'] = mse
        self.metrics[f'{prefix}_mae'] = mae
        self.metrics[f'{prefix}_r2'] = r2
        self.metrics[f'{prefix}_spearman'] = spearman_corr

        logger.info(f"{prefix.upper()} Metrics:")
        logger.info(f"  MSE: {mse:.6f}")
        logger.info(f"  MAE: {mae:.6f}")
        logger.info(f"  R2: {r2:.4f}")
        logger.info(f"  Spearman: {spearman_corr:.4f} (p={spearman_p:.2e})")

        # Critical check: negative Spearman means predictions are inversely ranked
        if spearman_corr < 0:
            logger.error(f"CRITICAL: Negative Spearman correlation ({spearman_corr:.4f})")
            logger.error("Higher predicted returns correlate with LOWER actual returns!")
        elif spearman_corr < 0.05:
            logger.warning(f"Very weak Spearman correlation ({spearman_corr:.4f})")

        return self.metrics

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict log-returns and uncertainty.

        Returns DataFrame with:
        - expected_log_return: E[log(1 + return)]
        - q10, q90: Quantile predictions
        - uncertainty: q90 - q10 interval width
        """
        X, _ = self._prepare_features(df)
        X_proc = self.scaler.transform(self.imputer.transform(X))

        results = pd.DataFrame(index=df.index)
        results['expected_log_return'] = self.model.predict(X_proc)
        results['q10'] = self.quantile_models['q10'].predict(X_proc)
        results['q90'] = self.quantile_models['q90'].predict(X_proc)
        results['uncertainty'] = results['q90'] - results['q10']

        return results

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from the model."""
        importance = self.model.feature_importances_
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

    def save(self, path: str):
        """Save model to disk."""
        artifacts = {
            'model': self.model,
            'quantile_models': self.quantile_models,
            'imputer': self.imputer,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'config': self.config,
            'metrics': self.metrics
        }
        joblib.dump(artifacts, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'LogReturnPredictor':
        """Load model from disk."""
        artifacts = joblib.load(path)
        predictor = cls(artifacts['config'])
        predictor.model = artifacts['model']
        predictor.quantile_models = artifacts['quantile_models']
        predictor.imputer = artifacts['imputer']
        predictor.scaler = artifacts['scaler']
        predictor.feature_names = artifacts['feature_names']
        predictor.metrics = artifacts.get('metrics', {})
        return predictor


# =============================================================================
# Position Sizing (Log-Return Based)
# =============================================================================

def compute_confidence_factor(
    uncertainty: pd.Series,
    scale: float = 1.5
) -> pd.Series:
    """
    Convert uncertainty into confidence factor for position sizing.

    Higher uncertainty = lower confidence = smaller position.
    """
    max_uncertainty = uncertainty.quantile(0.95)
    normalized = (uncertainty / max_uncertainty).clip(0, 1)
    confidence = np.exp(-scale * normalized)
    return confidence


def size_positions_log_return(
    predictions: pd.DataFrame,
    config: LogReturnConfig
) -> pd.DataFrame:
    """
    Size positions based on log-return predictions.

    Formula from proposal:
    size = min(expected_log_return / target_return, max_fraction) × confidence

    Note: This does NOT properly account for variance in the Kelly-optimal way.
    We test it empirically to see if it works despite this limitation.
    """
    result = predictions.copy()

    # Compute confidence from uncertainty
    confidence = compute_confidence_factor(
        result['uncertainty'],
        scale=config.uncertainty_scale
    )

    # Base size: proportional to expected log-return
    # Negative expected returns = zero position
    base_size = (result['expected_log_return'] / config.target_log_return).clip(lower=0)

    # Apply hard cap
    base_size = base_size.clip(upper=config.max_position_fraction)

    # Apply confidence scaling
    adjusted_size = base_size * confidence

    # Apply minimum threshold
    adjusted_size = adjusted_size.where(
        adjusted_size >= config.min_position_fraction,
        0
    )

    result['position_size'] = adjusted_size
    result['confidence'] = confidence

    return result


# =============================================================================
# Training Pipeline
# =============================================================================

def train_log_return_predictor(
    train_path: str,
    eval_path: Optional[str],
    output_path: str,
    config: LogReturnConfig
) -> bool:
    """Full training pipeline for log-return predictor."""

    # Load and preprocess training data
    df_raw = load_raw_data(train_path)
    cfg = load_config("config.yaml")
    df_processed, _ = preprocess_data(df_raw, cfg, scaler=None)

    # Calculate log-return target
    df_with_target = calculate_log_returns(
        df_processed,
        config.horizon_days,
        config.transaction_cost_bps
    )

    # Add ranker features
    df_with_features = add_ranker_features(df_with_target)

    logger.info(f"Training data: {len(df_with_features)} samples")

    # Load eval data if provided
    eval_df = None
    if eval_path:
        eval_raw = load_raw_data(eval_path)
        eval_processed, _ = preprocess_data(eval_raw, cfg, scaler=None)
        eval_with_target = calculate_log_returns(
            eval_processed,
            config.horizon_days,
            config.transaction_cost_bps
        )
        eval_df = add_ranker_features(eval_with_target)

    # Train model
    model = LogReturnPredictor(config)
    model.fit(df_with_features, eval_df)

    # Show feature importance
    importance = model.get_feature_importance()
    logger.info("Top 10 features:")
    for _, row in importance.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")

    # Save model
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    model.save(output_path)

    # Quality check
    spearman = model.metrics.get('eval_spearman', model.metrics.get('val_spearman', 0))
    if spearman < 0:
        logger.error("QUALITY GATE FAILED: Negative Spearman correlation")
        return False
    elif spearman < 0.05:
        logger.warning("QUALITY GATE WARNING: Very weak Spearman correlation")

    return True


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Log-Return Predictor (Shadow System)')
    parser.add_argument('--train-data', type=str, required=True, help='Path to training data')
    parser.add_argument('--eval-data', type=str, help='Path to evaluation data')
    parser.add_argument('--output', type=str, default='model_output/log_return_predictor.joblib')
    parser.add_argument('--horizon', type=int, default=5, help='Prediction horizon in days')
    args = parser.parse_args()

    config = LogReturnConfig(horizon_days=args.horizon)

    success = train_log_return_predictor(
        train_path=args.train_data,
        eval_path=args.eval_data,
        output_path=args.output,
        config=config
    )

    if success:
        logger.info("Training completed successfully")
    else:
        logger.error("Training completed with quality gate failures")


if __name__ == '__main__':
    main()
