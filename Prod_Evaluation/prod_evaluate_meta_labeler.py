"""
Evaluation Script for Meta-Labeler + Kelly Pipeline

Evaluates:
1. Meta-labeler classification performance (AUC, Brier, calibration)
2. Kelly sizing effectiveness
3. Full pipeline backtest (P&L, Sharpe, Drawdown)
4. Comparison metrics vs baseline strategies

Pipeline: RANKER -> META-LABELER -> KELLY SIZER -> EXECUTION

Author: Generated with Claude Code
"""

import pandas as pd
import numpy as np
import joblib
import argparse
import logging
import os
from datetime import datetime
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

from sklearn.metrics import (
    roc_auc_score, brier_score_loss, log_loss,
    precision_score, recall_score, f1_score,
    precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report
)
from sklearn.calibration import calibration_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils import load_config, preprocess_data
from prod_meta_labeler import (
    MetaLabeler, MetaLabelerConfig,
    load_raw_data, calculate_delta_hedged_pnl,
    create_binary_labels, add_ranker_features
)
from prod_kelly_sizer import KellySizer, KellyConfig, kelly_criterion

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Evaluation Metrics
# =============================================================================

def evaluate_classification(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Comprehensive classification metrics.

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        threshold: Classification threshold

    Returns:
        Dict of metric names to values
    """
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        'auc_roc': roc_auc_score(y_true, y_prob),
        'brier_score': brier_score_loss(y_true, y_prob),
        'log_loss': log_loss(y_true, y_prob),
        'avg_precision': average_precision_score(y_true, y_prob),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'accuracy': np.mean(y_true == y_pred),
    }

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['true_positive_rate'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0

    return metrics


def evaluate_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> Dict[str, float]:
    """
    Probability calibration metrics.

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for calibration curve

    Returns:
        Dict with calibration metrics
    """
    # Calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)

    # Expected Calibration Error (ECE)
    # Note: calibration_curve may return fewer bins than requested
    actual_bins = len(prob_pred)
    bin_edges = np.linspace(0, 1, actual_bins + 1)
    bin_counts = np.histogram(y_prob, bins=bin_edges)[0]
    ece = np.sum(np.abs(prob_true - prob_pred) * bin_counts / len(y_prob))

    # Maximum Calibration Error
    mce = np.max(np.abs(prob_true - prob_pred)) if len(prob_true) > 0 else 0

    return {
        'ece': ece,
        'mce': mce,
        'calibration_slope': np.polyfit(prob_pred, prob_true, 1)[0] if len(prob_pred) > 1 else 1.0,
    }


# =============================================================================
# Backtest Engine
# =============================================================================

@dataclass
class BacktestResult:
    """Container for backtest results."""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    n_trades: int
    trades_df: pd.DataFrame
    equity_curve: pd.Series


def backtest_pipeline(
    df: pd.DataFrame,
    prob_column: str = 'prob_profit',
    pnl_column: str = 'pnl_pct',
    date_column: str = 'date',
    kelly_config: KellyConfig = None,
    initial_capital: float = 100000
) -> BacktestResult:
    """
    Backtest the meta-labeler + Kelly pipeline.

    Args:
        df: DataFrame with probability predictions and actual PnL
        prob_column: Column with predicted probabilities
        pnl_column: Column with actual P&L
        date_column: Column with trade dates
        kelly_config: Kelly configuration
        initial_capital: Starting capital

    Returns:
        BacktestResult with performance metrics
    """
    if kelly_config is None:
        kelly_config = KellyConfig()

    sizer = KellySizer(kelly_config)

    # Size positions
    df = df.copy()
    sized_df = sizer.size_portfolio(df, prob_column=prob_column)

    # Filter to trades we actually take
    trades_df = sized_df[sized_df['position_size'] > 0].copy()

    if len(trades_df) == 0:
        logger.warning("No trades to backtest!")
        return BacktestResult(
            total_return=0.0, sharpe_ratio=0.0, max_drawdown=0.0,
            win_rate=0.0, avg_win=0.0, avg_loss=0.0, profit_factor=0.0,
            n_trades=0, trades_df=pd.DataFrame(),
            equity_curve=pd.Series([initial_capital])
        )

    # Calculate trade returns
    trades_df['trade_return'] = trades_df['position_size'] * trades_df[pnl_column]

    # Group by date for daily returns
    daily_returns = trades_df.groupby(date_column)['trade_return'].sum()

    # Build equity curve
    equity = [initial_capital]
    for ret in daily_returns:
        equity.append(equity[-1] * (1 + ret))
    equity_curve = pd.Series(equity[1:], index=daily_returns.index)

    # Calculate metrics
    total_return = (equity_curve.iloc[-1] / initial_capital - 1) * 100

    # Sharpe ratio (annualized, assuming 252 trading days)
    if daily_returns.std() > 0:
        sharpe = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
    else:
        sharpe = 0.0

    # Max drawdown
    running_max = equity_curve.expanding().max()
    drawdown = (running_max - equity_curve) / running_max
    max_dd = drawdown.max() * 100

    # Win rate
    wins = trades_df[trades_df[pnl_column] > 0]
    losses = trades_df[trades_df[pnl_column] <= 0]
    win_rate = len(wins) / len(trades_df) * 100 if len(trades_df) > 0 else 0

    # Average win/loss
    avg_win = wins[pnl_column].mean() * 100 if len(wins) > 0 else 0
    avg_loss = abs(losses[pnl_column].mean()) * 100 if len(losses) > 0 else 0

    # Profit factor
    gross_profit = wins['trade_return'].sum() if len(wins) > 0 else 0
    gross_loss = abs(losses['trade_return'].sum()) if len(losses) > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    return BacktestResult(
        total_return=total_return,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=profit_factor,
        n_trades=len(trades_df),
        trades_df=trades_df,
        equity_curve=equity_curve
    )


# =============================================================================
# Quality Gates
# =============================================================================

@dataclass
class QualityGates:
    """Quality gate thresholds."""
    min_auc: float = 0.55
    max_brier: float = 0.25
    max_ece: float = 0.10
    min_sharpe: float = 0.5
    max_drawdown: float = 30.0  # percent
    min_profit_factor: float = 1.2


def check_quality_gates(
    classification_metrics: Dict[str, float],
    calibration_metrics: Dict[str, float],
    backtest_result: BacktestResult,
    gates: QualityGates = None
) -> Tuple[bool, List[str]]:
    """
    Check if model passes quality gates.

    Returns:
        Tuple of (passed: bool, failures: List[str])
    """
    if gates is None:
        gates = QualityGates()

    failures = []

    # Classification gates
    if classification_metrics['auc_roc'] < gates.min_auc:
        failures.append(f"AUC {classification_metrics['auc_roc']:.4f} < {gates.min_auc}")

    if classification_metrics['brier_score'] > gates.max_brier:
        failures.append(f"Brier {classification_metrics['brier_score']:.4f} > {gates.max_brier}")

    # Calibration gates
    if calibration_metrics['ece'] > gates.max_ece:
        failures.append(f"ECE {calibration_metrics['ece']:.4f} > {gates.max_ece}")

    # Backtest gates
    if backtest_result.sharpe_ratio < gates.min_sharpe:
        failures.append(f"Sharpe {backtest_result.sharpe_ratio:.2f} < {gates.min_sharpe}")

    if backtest_result.max_drawdown > gates.max_drawdown:
        failures.append(f"Max DD {backtest_result.max_drawdown:.1f}% > {gates.max_drawdown}%")

    if backtest_result.profit_factor < gates.min_profit_factor:
        failures.append(f"Profit Factor {backtest_result.profit_factor:.2f} < {gates.min_profit_factor}")

    passed = len(failures) == 0
    return passed, failures


# =============================================================================
# Visualization
# =============================================================================

def plot_evaluation_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    backtest_result: BacktestResult,
    output_path: str
) -> None:
    """Generate evaluation plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. ROC Curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ax = axes[0, 0]
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC={roc_auc_score(y_true, y_prob):.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Calibration Plot
    ax = axes[0, 1]
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    ax.plot(prob_pred, prob_true, 'bo-', linewidth=2, markersize=8, label='Model')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect')
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('True Probability')
    ax.set_title('Calibration Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Equity Curve
    ax = axes[1, 0]
    if len(backtest_result.equity_curve) > 0:
        ax.plot(backtest_result.equity_curve.values, 'g-', linewidth=2)
        ax.axhline(y=backtest_result.equity_curve.iloc[0], color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Trading Day')
    ax.set_ylabel('Portfolio Value ($)')
    ax.set_title(f'Equity Curve (Return: {backtest_result.total_return:.1f}%)')
    ax.grid(True, alpha=0.3)

    # 4. Probability Distribution
    ax = axes[1, 1]
    ax.hist(y_prob[y_true == 1], bins=30, alpha=0.6, label='Profitable', color='green')
    ax.hist(y_prob[y_true == 0], bins=30, alpha=0.6, label='Not Profitable', color='red')
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Count')
    ax.set_title('Probability Distribution by Outcome')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Evaluation plots saved to: {output_path}")


# =============================================================================
# Main Evaluation Function
# =============================================================================

def evaluate_meta_labeler(
    model_path: str,
    eval_data_path: str,
    output_dir: str,
    kelly_config: KellyConfig = None,
    ranker_path: Optional[str] = None
) -> Dict[str, any]:
    """
    Full evaluation of meta-labeler + Kelly pipeline.

    Args:
        model_path: Path to trained meta-labeler
        eval_data_path: Path to evaluation data
        output_dir: Directory for output files
        kelly_config: Kelly sizer configuration
        ranker_path: Optional path to ranker model

    Returns:
        Dict with all evaluation results
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    logger.info(f"Loading model from: {model_path}")
    artifacts = joblib.load(model_path)

    meta_config = artifacts.get('config', MetaLabelerConfig())
    model = MetaLabeler(meta_config)
    model.classifier = artifacts['classifier']
    model.calibrator = artifacts['calibrator']
    model.scaler = artifacts['scaler']
    model.imputer = artifacts['imputer']
    model.feature_names = artifacts['feature_names']

    # Load and prepare evaluation data
    logger.info(f"Loading evaluation data from: {eval_data_path}")
    df_raw = load_raw_data(eval_data_path)

    cfg = load_config("config.yaml")
    df_processed, _ = preprocess_data(df_raw, cfg, scaler=None)

    # Calculate P&L
    df_with_pnl = calculate_delta_hedged_pnl(df_processed, meta_config.horizon_days)
    df_labeled = create_binary_labels(df_with_pnl, meta_config.profit_threshold)

    # Add ranker features
    df_with_features = add_ranker_features(
        df_labeled,
        ranker_model_path=ranker_path
    )

    # Get predictions on ALL data
    logger.info(f"Evaluating on {len(df_with_features)} samples (all data)")
    y_true_all = df_with_features['profitable'].values
    y_prob_all = model.predict_proba(df_with_features)
    df_with_features['prob_profit'] = y_prob_all

    # For backtest: select top-K per day based on ranker score
    # This simulates real-world usage where ranker selects options
    df_with_features['ranker_rank_daily'] = df_with_features.groupby('date')['ranker_score'].rank(
        ascending=False, method='first'
    )
    eval_df = df_with_features[df_with_features['ranker_rank_daily'] <= meta_config.top_k_per_day].copy()
    logger.info(f"Backtest on {len(eval_df)} ranker top-K selections")

    # Get predictions for selected subset
    y_true = eval_df['profitable'].values
    y_prob = eval_df['prob_profit'].values

    # === Classification Metrics ===
    logger.info("\n" + "="*60)
    logger.info("CLASSIFICATION METRICS")
    logger.info("="*60)

    class_metrics = evaluate_classification(y_true, y_prob)
    for name, value in class_metrics.items():
        logger.info(f"  {name}: {value:.4f}")

    # === Calibration Metrics ===
    logger.info("\n" + "="*60)
    logger.info("CALIBRATION METRICS")
    logger.info("="*60)

    calib_metrics = evaluate_calibration(y_true, y_prob)
    for name, value in calib_metrics.items():
        logger.info(f"  {name}: {value:.4f}")

    # === Backtest Results ===
    logger.info("\n" + "="*60)
    logger.info("BACKTEST RESULTS")
    logger.info("="*60)

    if kelly_config is None:
        kelly_config = KellyConfig()

    backtest = backtest_pipeline(
        eval_df,
        prob_column='prob_profit',
        pnl_column='pnl_pct',
        kelly_config=kelly_config
    )

    logger.info(f"  Total Return: {backtest.total_return:.1f}%")
    logger.info(f"  Sharpe Ratio: {backtest.sharpe_ratio:.2f}")
    logger.info(f"  Max Drawdown: {backtest.max_drawdown:.1f}%")
    logger.info(f"  Win Rate: {backtest.win_rate:.1f}%")
    logger.info(f"  Avg Win: {backtest.avg_win:.2f}%")
    logger.info(f"  Avg Loss: {backtest.avg_loss:.2f}%")
    logger.info(f"  Profit Factor: {backtest.profit_factor:.2f}")
    logger.info(f"  Number of Trades: {backtest.n_trades}")

    # === Baseline Comparison ===
    logger.info("\n" + "="*60)
    logger.info("BASELINE COMPARISON")
    logger.info("="*60)

    # Equal weight baseline (all top-K, equal size)
    equal_weight_size = kelly_config.max_position_pct
    eval_df_baseline = eval_df.copy()
    eval_df_baseline['position_size'] = equal_weight_size
    eval_df_baseline['trade_return'] = eval_df_baseline['position_size'] * eval_df_baseline['pnl_pct']

    baseline_return = eval_df_baseline['trade_return'].sum() * 100
    baseline_win_rate = (eval_df_baseline['pnl_pct'] > 0).mean() * 100

    logger.info(f"  Equal Weight Baseline:")
    logger.info(f"    Return: {baseline_return:.1f}%")
    logger.info(f"    Win Rate: {baseline_win_rate:.1f}%")

    logger.info(f"  Kelly Improvement:")
    logger.info(f"    Return Delta: {backtest.total_return - baseline_return:+.1f}%")

    # === Quality Gates ===
    logger.info("\n" + "="*60)
    logger.info("QUALITY GATES")
    logger.info("="*60)

    gates = QualityGates()
    passed, failures = check_quality_gates(class_metrics, calib_metrics, backtest, gates)

    if passed:
        logger.info("  STATUS: PASSED")
    else:
        logger.warning("  STATUS: FAILED")
        for f in failures:
            logger.warning(f"    - {f}")

    # === Generate Plots ===
    plot_path = os.path.join(output_dir, 'evaluation_report.png')
    plot_evaluation_report(y_true, y_prob, backtest, plot_path)

    # === Save Results ===
    results = {
        'classification': class_metrics,
        'calibration': calib_metrics,
        'backtest': {
            'total_return': backtest.total_return,
            'sharpe_ratio': backtest.sharpe_ratio,
            'max_drawdown': backtest.max_drawdown,
            'win_rate': backtest.win_rate,
            'profit_factor': backtest.profit_factor,
            'n_trades': backtest.n_trades,
        },
        'baseline': {
            'return': baseline_return,
            'win_rate': baseline_win_rate,
        },
        'quality_gates': {
            'passed': passed,
            'failures': failures,
        },
        'evaluation_date': datetime.now().isoformat(),
        'data_file': eval_data_path,
        'model_file': model_path,
    }

    results_path = os.path.join(output_dir, 'evaluation_results.joblib')
    joblib.dump(results, results_path)
    logger.info(f"\nResults saved to: {results_path}")

    # Save trades for analysis
    trades_path = os.path.join(output_dir, 'backtest_trades.csv')
    backtest.trades_df.to_csv(trades_path, index=False)
    logger.info(f"Trades saved to: {trades_path}")

    return results


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Meta-Labeler + Kelly Pipeline"
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained meta-labeler model'
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
        default='evaluation_output',
        help='Directory for output files'
    )
    parser.add_argument(
        '--ranker-model',
        type=str,
        default=None,
        help='Path to trained ranker model (optional)'
    )
    parser.add_argument(
        '--kelly-fraction',
        type=float,
        default=0.25,
        help='Kelly fraction for sizing'
    )

    args = parser.parse_args()

    kelly_config = KellyConfig(kelly_fraction=args.kelly_fraction)

    evaluate_meta_labeler(
        model_path=args.model,
        eval_data_path=args.eval_data,
        output_dir=args.output_dir,
        kelly_config=kelly_config,
        ranker_path=args.ranker_model,
    )


if __name__ == '__main__':
    main()
