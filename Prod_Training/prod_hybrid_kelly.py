"""
Hybrid Kelly Sizer - Best of Both Worlds
=========================================

Combines:
1. Meta-Labeler: Calibrated P(profit) predictions
2. Log-Return Regressor: Expected return magnitude predictions
3. Proper Kelly: f* = (p*b - q) / b with full risk management

Theory:
- Kelly criterion needs: probability of win (p) and win/loss ratio (b)
- Meta-Labeler gives us calibrated p
- Log-Return Regressor gives us E[return], from which we derive b
- Combined, we get proper Kelly with both probability AND magnitude

This should outperform both individual approaches:
- Better than Meta-Labeler alone: captures trade magnitude
- Better than Log-Return alone: proper risk management via Kelly

Author: Generated with Claude Code
"""

from __future__ import annotations
import logging
import os
import sys
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib

# Add current directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import local modules at module level (required for pickle/joblib unpickling)
from utils import load_config, preprocess_data
from prod_log_return_predictor import (
    LogReturnPredictor, LogReturnConfig,
    load_raw_data, calculate_log_returns, add_ranker_features
)
from prod_meta_labeler import MetaLabeler, MetaLabelerConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class HybridKellyConfig:
    """Configuration for hybrid Kelly sizer.

    Optimized via Optuna (150 trials) on 2024 data, validated on 2025.
    """
    # Kelly parameters - optimized from 0.25 to 0.48
    kelly_fraction: float = 0.48  # Half Kelly (optimized)

    # Position limits - optimized for higher returns
    min_position_pct: float = 0.0125  # 1.25% minimum (optimized)
    max_position_pct: float = 0.106   # 10.6% maximum per position (optimized)
    max_portfolio_risk: float = 0.70  # 70% max total allocation (optimized)

    # Probability thresholds - optimized
    min_prob_to_trade: float = 0.40   # Don't trade if P(win) < 40% (optimized)
    high_conviction_threshold: float = 0.69  # Boost size above this (optimized)

    # Return magnitude thresholds
    min_expected_return: float = 0.025  # 2.5% minimum expected return (optimized)

    # Uncertainty adjustment
    uncertainty_penalty: float = 1.07  # Scale down by uncertainty (optimized)

    # Win/loss ratio bounds
    min_win_loss_ratio: float = 0.39  # Don't trade if b < 0.39 (optimized)
    max_win_loss_ratio: float = 5.0   # Cap b at 5 to avoid extreme bets (optimized)


# =============================================================================
# Core Kelly Calculation
# =============================================================================

def kelly_criterion(
    prob_win: float,
    win_loss_ratio: float,
    fraction: float = 0.25
) -> float:
    """
    Calculate Kelly optimal fraction.

    f* = (p * b - q) / b

    where:
        p = probability of winning
        q = probability of losing (1 - p)
        b = win/loss ratio (how much you win vs how much you lose)

    Args:
        prob_win: Probability of winning (0-1)
        win_loss_ratio: Ratio of win amount to loss amount
        fraction: Kelly fraction (0.25 = quarter Kelly)

    Returns:
        Optimal bet fraction (can be negative if edge is negative)
    """
    if prob_win <= 0 or prob_win >= 1:
        return 0.0
    if win_loss_ratio <= 0:
        return 0.0

    q = 1 - prob_win
    kelly = (prob_win * win_loss_ratio - q) / win_loss_ratio

    return kelly * fraction


def estimate_win_loss_ratio(
    expected_return: float,
    prob_win: float,
    uncertainty: float = 0.0
) -> float:
    """
    Estimate win/loss ratio from expected return and probability.

    E[return] = p * win_amount - q * loss_amount

    Assuming symmetric wins/losses for simplicity:
    E[return] = p * b * L - q * L = L * (p*b - q)

    If we assume L = 1 (unit loss):
    b = (E[return] + q) / p

    But we also know from log-return that trades have asymmetric payoffs.
    We use the expected return magnitude as a proxy for the average
    win amount, adjusted by uncertainty.

    Args:
        expected_return: Expected log return from regressor
        prob_win: Probability of winning from classifier
        uncertainty: Prediction uncertainty (q90 - q10)

    Returns:
        Estimated win/loss ratio
    """
    if prob_win <= 0 or prob_win >= 1:
        return 0.0

    # Convert log return to simple return for ratio calculation
    # E[log(1+r)] ≈ E[r] - 0.5*Var[r] for small returns
    # For larger returns, exp(log_return) - 1 = simple_return
    simple_return = np.exp(expected_return) - 1

    # If expected return is negative, ratio should be low
    if simple_return <= 0:
        return 0.0

    # Estimate average win and loss amounts
    # Assume: wins are proportional to expected_return when positive
    # losses are proportional to uncertainty (downside risk)

    q = 1 - prob_win

    # Method: Use expected return as win magnitude
    # and uncertainty as a proxy for loss magnitude
    if uncertainty > 0:
        # Higher uncertainty = higher potential loss
        avg_loss = max(0.05, uncertainty * 0.5)  # At least 5% loss
    else:
        avg_loss = 0.10  # Default 10% loss

    # Win amount: expected return scaled by probability
    # If we expect 10% return with 60% probability,
    # the conditional win must be higher
    if prob_win > 0:
        avg_win = simple_return / prob_win + avg_loss * q / prob_win
        avg_win = max(avg_win, simple_return)  # At least the expected return
    else:
        avg_win = simple_return

    # Win/loss ratio
    if avg_loss > 0:
        ratio = avg_win / avg_loss
    else:
        ratio = 1.0

    return ratio


# =============================================================================
# Hybrid Kelly Sizer
# =============================================================================

class HybridKellySizer:
    """
    Combines Meta-Labeler probabilities with Log-Return predictions
    for optimal Kelly-based position sizing.
    """

    def __init__(self, config: HybridKellyConfig = None):
        self.config = config or HybridKellyConfig()

    def size_positions(
        self,
        df: pd.DataFrame,
        prob_column: str = 'prob_profit',
        return_column: str = 'expected_log_return',
        uncertainty_column: str = 'uncertainty'
    ) -> pd.DataFrame:
        """
        Calculate position sizes using hybrid Kelly.

        Args:
            df: DataFrame with probability and return predictions
            prob_column: Column with P(profit) from Meta-Labeler
            return_column: Column with E[log(1+r)] from Log-Return model
            uncertainty_column: Column with prediction uncertainty

        Returns:
            DataFrame with position sizes added
        """
        df = df.copy()
        cfg = self.config

        # Validate required columns
        if prob_column not in df.columns:
            raise ValueError(f"Missing probability column: {prob_column}")
        if return_column not in df.columns:
            raise ValueError(f"Missing return column: {return_column}")

        # Get predictions
        probs = df[prob_column].values
        returns = df[return_column].values

        # Handle uncertainty (use default if not provided)
        if uncertainty_column in df.columns:
            uncertainties = df[uncertainty_column].values
        else:
            uncertainties = np.full(len(df), 0.5)  # Default uncertainty

        # Calculate position sizes
        position_sizes = np.zeros(len(df))
        win_loss_ratios = np.zeros(len(df))
        kelly_fractions = np.zeros(len(df))

        for i in range(len(df)):
            p = probs[i]
            exp_ret = returns[i]
            unc = uncertainties[i]

            # Gate 1: Minimum probability
            if p < cfg.min_prob_to_trade:
                continue

            # Gate 2: Minimum expected return
            if exp_ret < cfg.min_expected_return:
                continue

            # Estimate win/loss ratio
            b = estimate_win_loss_ratio(exp_ret, p, unc)

            # Gate 3: Win/loss ratio bounds
            if b < cfg.min_win_loss_ratio:
                continue
            b = min(b, cfg.max_win_loss_ratio)  # Cap to avoid extreme bets

            win_loss_ratios[i] = b

            # Calculate Kelly fraction
            kelly = kelly_criterion(p, b, cfg.kelly_fraction)
            kelly_fractions[i] = kelly

            # Gate 4: Kelly must be positive
            if kelly <= 0:
                continue

            # Apply uncertainty penalty
            if unc > 0:
                # Higher uncertainty = reduce position
                uncertainty_factor = np.exp(-cfg.uncertainty_penalty * unc)
                kelly *= uncertainty_factor

            # Apply conviction boost
            if p >= cfg.high_conviction_threshold:
                conviction_boost = 1.0 + (p - cfg.high_conviction_threshold) * 0.5
                kelly *= conviction_boost

            # Enforce position limits
            size = np.clip(kelly, cfg.min_position_pct, cfg.max_position_pct)
            position_sizes[i] = size

        # Store results
        df['win_loss_ratio'] = win_loss_ratios
        df['kelly_fraction'] = kelly_fractions
        df['position_size'] = position_sizes

        # Log statistics
        n_trades = (position_sizes > 0).sum()
        total_alloc = position_sizes.sum()

        logger.info(f"Hybrid Kelly sizing complete:")
        logger.info(f"  Positions sized: {n_trades}")
        logger.info(f"  Total allocation: {total_alloc:.2%}")
        if n_trades > 0:
            logger.info(f"  Avg position size: {position_sizes[position_sizes > 0].mean():.4f}")
            logger.info(f"  Avg win/loss ratio: {win_loss_ratios[win_loss_ratios > 0].mean():.2f}")

        return df

    def size_portfolio(
        self,
        df: pd.DataFrame,
        prob_column: str = 'prob_profit',
        return_column: str = 'expected_log_return',
        uncertainty_column: str = 'uncertainty',
        date_column: str = 'date'
    ) -> pd.DataFrame:
        """
        Size positions with portfolio-level risk management.

        Applies daily allocation caps and diversification.
        """
        df = self.size_positions(df, prob_column, return_column, uncertainty_column)
        cfg = self.config

        # Apply daily portfolio cap
        def cap_daily_allocation(group):
            total = group['position_size'].sum()
            if total > cfg.max_portfolio_risk:
                scale = cfg.max_portfolio_risk / total
                group['position_size'] = group['position_size'] * scale
            return group

        if date_column in df.columns:
            df = df.groupby(date_column, group_keys=False).apply(cap_daily_allocation)

        return df


# =============================================================================
# Evaluation Pipeline
# =============================================================================

def evaluate_hybrid_kelly(
    meta_labeler_path: str,
    log_return_path: str,
    eval_data_path: str,
    output_dir: str,
    config: HybridKellyConfig = None,
    ranker_model_path: str = None
) -> Dict:
    """
    Full evaluation of hybrid Kelly approach.

    Loads both models, combines predictions, and runs backtest.

    Args:
        meta_labeler_path: Path to trained Meta-Labeler model
        log_return_path: Path to trained Log-Return Predictor
        eval_data_path: Path to evaluation data CSV
        output_dir: Directory for output files
        config: Hybrid Kelly configuration
        ranker_model_path: Optional path to trained ranker model (if None, uses heuristic)
    """
    os.makedirs(output_dir, exist_ok=True)

    if config is None:
        config = HybridKellyConfig()

    # Load models
    logger.info(f"Loading Meta-Labeler from: {meta_labeler_path}")
    meta_artifacts = joblib.load(meta_labeler_path)

    logger.info(f"Loading Log-Return Predictor from: {log_return_path}")
    log_return_model = LogReturnPredictor.load(log_return_path)

    # Load ranker model if provided
    ranker_pipeline = None
    ranker_feature_cols = None
    if ranker_model_path and os.path.exists(ranker_model_path):
        logger.info(f"Loading trained Ranker from: {ranker_model_path}")
        ranker_pipeline = joblib.load(ranker_model_path)
        # Try to load feature names (support both old and new naming conventions)
        feature_path = ranker_model_path.replace('xgboost_ranker2_', 'xgb_feature_names_').replace('.joblib', '.pkl')
        if not os.path.exists(feature_path):
            # Try simpler naming: ranker.joblib -> ranker_feature_names.pkl
            feature_path = ranker_model_path.replace('.joblib', '_feature_names.pkl')
        if os.path.exists(feature_path):
            ranker_feature_cols = joblib.load(feature_path)
            logger.info(f"  Loaded {len(ranker_feature_cols)} ranker features")
        else:
            logger.warning(f"  Could not find ranker feature names file")
    else:
        logger.info("Using heuristic-based ranker scoring (no trained model provided)")

    # Load and prepare evaluation data
    logger.info(f"Loading evaluation data from: {eval_data_path}")
    df_raw = load_raw_data(eval_data_path)

    cfg = load_config("config.yaml")
    df_processed, _ = preprocess_data(df_raw, cfg, scaler=None)

    # Calculate log returns for actual P&L
    lr_config = LogReturnConfig()
    df_with_returns = calculate_log_returns(
        df_processed,
        lr_config.horizon_days,
        lr_config.transaction_cost_bps
    )

    # Add ranker features (heuristic or ML-based)
    if ranker_pipeline is not None and ranker_feature_cols is not None:
        # Use trained ranker model
        df_with_features = add_ranker_features(df_with_returns)  # Add base features first

        # Score with trained ranker
        available_cols = [c for c in ranker_feature_cols if c in df_with_features.columns]
        missing_cols = set(ranker_feature_cols) - set(available_cols)
        if missing_cols:
            logger.warning(f"Missing ranker features: {missing_cols}")
            # Fill missing columns with 0
            for col in missing_cols:
                df_with_features[col] = 0

        # Get ranker predictions
        X_ranker = df_with_features[ranker_feature_cols].copy()
        if 'type' in X_ranker.columns:
            X_ranker['type'] = X_ranker['type'].astype(str)

        ranker_scores = ranker_pipeline.predict(X_ranker)
        df_with_features['ranker_score'] = ranker_scores
        logger.info(f"  Applied trained ranker: score range [{ranker_scores.min():.3f}, {ranker_scores.max():.3f}]")
    else:
        # Use heuristic-based scoring
        df_with_features = add_ranker_features(df_with_returns)

    logger.info(f"Evaluating on {len(df_with_features)} samples")

    # Get Meta-Labeler predictions
    logger.info("Getting Meta-Labeler predictions...")
    meta_config = meta_artifacts.get('config', MetaLabelerConfig())
    meta_labeler = MetaLabeler(meta_config)
    meta_labeler.classifier = meta_artifacts['classifier']
    meta_labeler.calibrator = meta_artifacts['calibrator']
    meta_labeler.scaler = meta_artifacts['scaler']
    meta_labeler.imputer = meta_artifacts['imputer']
    meta_labeler.feature_names = meta_artifacts['feature_names']

    df_with_features['prob_profit'] = meta_labeler.predict_proba(df_with_features)

    # Get Log-Return predictions
    logger.info("Getting Log-Return predictions...")
    log_return_preds = log_return_model.predict(df_with_features)
    df_with_features['expected_log_return'] = log_return_preds['expected_log_return']
    df_with_features['uncertainty'] = log_return_preds['uncertainty']
    df_with_features['q10'] = log_return_preds['q10']
    df_with_features['q90'] = log_return_preds['q90']

    # Filter to top-K per day (same as other evaluations for fair comparison)
    df_with_features['ranker_rank_daily'] = df_with_features.groupby('date')['ranker_score'].rank(
        ascending=False, method='first'
    )
    eval_df = df_with_features[df_with_features['ranker_rank_daily'] <= 20].copy()
    logger.info(f"Backtest on {len(eval_df)} top-K selections")

    # Apply Hybrid Kelly sizing
    sizer = HybridKellySizer(config)
    sized_df = sizer.size_portfolio(eval_df)

    # Run backtest
    logger.info("\n" + "="*60)
    logger.info("BACKTEST RESULTS")
    logger.info("="*60)

    trades_df = sized_df[sized_df['position_size'] > 0].copy()

    if len(trades_df) == 0:
        logger.warning("No trades to backtest!")
        return {'n_trades': 0}

    # Calculate trade returns
    trades_df['trade_return'] = trades_df['position_size'] * trades_df['net_return']

    # Daily returns
    daily_returns = trades_df.groupby('date')['trade_return'].sum()

    # Build equity curve
    initial_capital = 100000
    equity = [initial_capital]
    for ret in daily_returns:
        equity.append(equity[-1] * (1 + ret))
    equity_curve = pd.Series(equity[1:], index=daily_returns.index)

    # Metrics
    total_return = (equity_curve.iloc[-1] / initial_capital - 1) * 100

    if daily_returns.std() > 0:
        sharpe = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
    else:
        sharpe = 0.0

    running_max = equity_curve.expanding().max()
    drawdown = (running_max - equity_curve) / running_max
    max_dd = drawdown.max() * 100

    wins = trades_df[trades_df['net_return'] > 0]
    losses = trades_df[trades_df['net_return'] <= 0]
    win_rate = len(wins) / len(trades_df) * 100 if len(trades_df) > 0 else 0

    avg_win = wins['net_return'].mean() * 100 if len(wins) > 0 else 0
    avg_loss = abs(losses['net_return'].mean()) * 100 if len(losses) > 0 else 0

    gross_profit = wins['trade_return'].sum() if len(wins) > 0 else 0
    gross_loss = abs(losses['trade_return'].sum()) if len(losses) > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    logger.info(f"  Total Return: {total_return:.1f}%")
    logger.info(f"  Sharpe Ratio: {sharpe:.2f}")
    logger.info(f"  Max Drawdown: {max_dd:.1f}%")
    logger.info(f"  Win Rate: {win_rate:.1f}%")
    logger.info(f"  Avg Win: {avg_win:.2f}%")
    logger.info(f"  Avg Loss: {avg_loss:.2f}%")
    logger.info(f"  Profit Factor: {profit_factor:.2f}")
    logger.info(f"  Number of Trades: {len(trades_df)}")

    # Quality check
    logger.info("\n" + "="*60)
    logger.info("QUALITY GATES")
    logger.info("="*60)

    failures = []
    if sharpe < 0.5:
        failures.append(f"Sharpe {sharpe:.2f} < 0.5")
    if max_dd > 30:
        failures.append(f"Max DD {max_dd:.1f}% > 30%")
    if profit_factor < 1.2:
        failures.append(f"Profit Factor {profit_factor:.2f} < 1.2")

    if not failures:
        logger.info("  STATUS: PASSED")
    else:
        logger.warning("  STATUS: FAILED")
        for f in failures:
            logger.warning(f"    - {f}")

    # Save results
    results = {
        'backtest': {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'n_trades': len(trades_df),
        },
        'quality_gates': {
            'passed': len(failures) == 0,
            'failures': failures,
        },
        'config': config,
    }

    results_path = os.path.join(output_dir, 'hybrid_kelly_results.joblib')
    joblib.dump(results, results_path)
    logger.info(f"\nResults saved to: {results_path}")

    trades_path = os.path.join(output_dir, 'hybrid_kelly_trades.csv')
    trades_df.to_csv(trades_path, index=False)
    logger.info(f"Trades saved to: {trades_path}")

    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Hybrid Kelly: Meta-Labeler + Log-Return"
    )
    parser.add_argument(
        '--meta-labeler',
        type=str,
        required=True,
        help='Path to trained Meta-Labeler model'
    )
    parser.add_argument(
        '--log-return',
        type=str,
        required=True,
        help='Path to trained Log-Return Predictor'
    )
    parser.add_argument(
        '--eval-data',
        type=str,
        required=True,
        help='Path to evaluation data CSV'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='hybrid_output',
        help='Directory for output files'
    )
    parser.add_argument(
        '--kelly-fraction',
        type=float,
        default=0.25,
        help='Kelly fraction (default: 0.25 = quarter Kelly)'
    )
    parser.add_argument(
        '--max-position',
        type=float,
        default=0.05,
        help='Max position size (default: 0.05 = 5%%)'
    )
    parser.add_argument(
        '--max-portfolio',
        type=float,
        default=0.40,
        help='Max portfolio allocation (default: 0.40 = 40%%)'
    )
    parser.add_argument(
        '--ranker',
        type=str,
        default=None,
        help='Path to trained ranker model (optional, uses heuristic if not provided)'
    )

    args = parser.parse_args()

    config = HybridKellyConfig(
        kelly_fraction=args.kelly_fraction,
        max_position_pct=args.max_position,
        max_portfolio_risk=args.max_portfolio,
    )

    evaluate_hybrid_kelly(
        meta_labeler_path=args.meta_labeler,
        log_return_path=args.log_return,
        eval_data_path=args.eval_data,
        output_dir=args.output_dir,
        config=config,
        ranker_model_path=args.ranker,
    )


if __name__ == '__main__':
    main()
