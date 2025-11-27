"""
Evaluation Script for Log-Return Predictor (Shadow System)

This script evaluates the log-return prediction approach and provides
a direct comparison with the Meta-Labeler + Kelly system.

Key Comparisons:
1. Regression metrics (R2, MAE, Spearman correlation)
2. Backtest performance (Return, Sharpe, Drawdown)
3. Head-to-head comparison with Meta-Labeler

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

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils import load_config, preprocess_data
from prod_log_return_predictor import (
    LogReturnPredictor, LogReturnConfig,
    load_raw_data, calculate_log_returns,
    add_ranker_features, size_positions_log_return
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Evaluation Metrics
# =============================================================================

def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Comprehensive regression metrics for log-return prediction.

    Args:
        y_true: True log returns
        y_pred: Predicted log returns

    Returns:
        Dict of metric names to values
    """
    # Standard regression metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Spearman correlation (critical for ranking)
    spearman_corr, spearman_p = spearmanr(y_pred, y_true)

    # Direction accuracy (does prediction get sign right?)
    direction_correct = np.mean(np.sign(y_pred) == np.sign(y_true))

    # Positive return accuracy (if we predict positive, is it positive?)
    pred_positive = y_pred > 0
    if pred_positive.sum() > 0:
        positive_precision = (y_true[pred_positive] > 0).mean()
    else:
        positive_precision = 0.0

    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'spearman': spearman_corr,
        'spearman_p': spearman_p,
        'direction_accuracy': direction_correct,
        'positive_precision': positive_precision,
    }

    return metrics


def evaluate_quantile_coverage(
    y_true: np.ndarray,
    q10: np.ndarray,
    q90: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate quantile prediction coverage.

    Good uncertainty estimation should have ~80% of true values
    falling between q10 and q90.
    """
    within_interval = ((y_true >= q10) & (y_true <= q90)).mean()
    below_q10 = (y_true < q10).mean()
    above_q90 = (y_true > q90).mean()

    # Interval width (average uncertainty)
    avg_interval_width = np.mean(q90 - q10)

    return {
        'coverage_80': within_interval,
        'below_q10': below_q10,
        'above_q90': above_q90,
        'avg_interval_width': avg_interval_width,
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


def backtest_log_return_pipeline(
    df: pd.DataFrame,
    position_column: str = 'position_size',
    return_column: str = 'net_return',
    date_column: str = 'date',
    initial_capital: float = 100000
) -> BacktestResult:
    """
    Backtest the log-return predictor pipeline.

    Args:
        df: DataFrame with position sizes and actual returns
        position_column: Column with position sizes
        return_column: Column with actual returns
        date_column: Column with trade dates
        initial_capital: Starting capital

    Returns:
        BacktestResult with performance metrics
    """
    df = df.copy()

    # Filter to trades we actually take
    trades_df = df[df[position_column] > 0].copy()

    if len(trades_df) == 0:
        logger.warning("No trades to backtest!")
        return BacktestResult(
            total_return=0.0, sharpe_ratio=0.0, max_drawdown=0.0,
            win_rate=0.0, avg_win=0.0, avg_loss=0.0, profit_factor=0.0,
            n_trades=0, trades_df=pd.DataFrame(),
            equity_curve=pd.Series([initial_capital])
        )

    # Calculate trade returns
    trades_df['trade_return'] = trades_df[position_column] * trades_df[return_column]

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
    wins = trades_df[trades_df[return_column] > 0]
    losses = trades_df[trades_df[return_column] <= 0]
    win_rate = len(wins) / len(trades_df) * 100 if len(trades_df) > 0 else 0

    # Average win/loss
    avg_win = wins[return_column].mean() * 100 if len(wins) > 0 else 0
    avg_loss = abs(losses[return_column].mean()) * 100 if len(losses) > 0 else 0

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
# Comparison with Meta-Labeler
# =============================================================================

def load_meta_labeler_results(results_path: str) -> Optional[Dict]:
    """Load previous meta-labeler evaluation results for comparison."""
    try:
        results = joblib.load(results_path)
        return results
    except Exception as e:
        logger.warning(f"Could not load meta-labeler results: {e}")
        return None


def print_comparison(
    log_return_backtest: BacktestResult,
    meta_labeler_results: Optional[Dict]
) -> None:
    """Print head-to-head comparison."""
    logger.info("\n" + "="*70)
    logger.info("HEAD-TO-HEAD COMPARISON: Log-Return vs Meta-Labeler")
    logger.info("="*70)

    metrics = [
        ('Total Return (%)', 'total_return'),
        ('Sharpe Ratio', 'sharpe_ratio'),
        ('Max Drawdown (%)', 'max_drawdown'),
        ('Win Rate (%)', 'win_rate'),
        ('Profit Factor', 'profit_factor'),
        ('Number of Trades', 'n_trades'),
    ]

    if meta_labeler_results:
        ml_backtest = meta_labeler_results.get('backtest', {})

        logger.info(f"{'Metric':<25} {'Log-Return':>15} {'Meta-Labeler':>15} {'Winner':>12}")
        logger.info("-" * 70)

        for name, key in metrics:
            lr_val = getattr(log_return_backtest, key)
            ml_val = ml_backtest.get(key, 0)

            # Determine winner (lower is better for drawdown)
            if key == 'max_drawdown':
                winner = 'Log-Return' if lr_val < ml_val else 'Meta-Labeler'
            else:
                winner = 'Log-Return' if lr_val > ml_val else 'Meta-Labeler'

            if key in ['total_return', 'max_drawdown', 'win_rate']:
                logger.info(f"{name:<25} {lr_val:>14.1f}% {ml_val:>14.1f}% {winner:>12}")
            elif key == 'n_trades':
                logger.info(f"{name:<25} {lr_val:>15d} {ml_val:>15d} {winner:>12}")
            else:
                logger.info(f"{name:<25} {lr_val:>15.2f} {ml_val:>15.2f} {winner:>12}")
    else:
        logger.info("Meta-labeler results not available for comparison.")
        logger.info(f"\nLog-Return Results:")
        for name, key in metrics:
            val = getattr(log_return_backtest, key)
            if key in ['total_return', 'max_drawdown', 'win_rate']:
                logger.info(f"  {name}: {val:.1f}%")
            elif key == 'n_trades':
                logger.info(f"  {name}: {val}")
            else:
                logger.info(f"  {name}: {val:.2f}")


# =============================================================================
# Visualization
# =============================================================================

def plot_evaluation_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    q10: np.ndarray,
    q90: np.ndarray,
    backtest_result: BacktestResult,
    output_path: str
) -> None:
    """Generate evaluation plots for log-return predictor."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Predicted vs Actual
    ax = axes[0, 0]
    ax.scatter(y_pred, y_true, alpha=0.3, s=10)
    ax.plot([-0.5, 0.5], [-0.5, 0.5], 'r--', linewidth=2, label='Perfect')
    ax.set_xlabel('Predicted Log Return')
    ax.set_ylabel('Actual Log Return')
    ax.set_title(f'Predicted vs Actual (R²={r2_score(y_true, y_pred):.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)

    # 2. Uncertainty Calibration
    ax = axes[0, 1]
    residuals = y_true - y_pred
    quantiles = np.linspace(0.05, 0.95, 19)
    actual_coverage = []
    for q in quantiles:
        lower = np.percentile(y_pred - y_true, (1-q)*100/2)
        upper = np.percentile(y_pred - y_true, 100 - (1-q)*100/2)
        covered = ((residuals >= lower) & (residuals <= upper)).mean()
        actual_coverage.append(covered)
    ax.plot(quantiles, actual_coverage, 'b-o', linewidth=2, markersize=4, label='Model')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect')
    ax.set_xlabel('Expected Coverage')
    ax.set_ylabel('Actual Coverage')
    ax.set_title('Uncertainty Calibration')
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

    # 4. Prediction Distribution by Outcome
    ax = axes[1, 1]
    profitable = y_true > 0
    ax.hist(y_pred[profitable], bins=30, alpha=0.6, label='Profitable', color='green')
    ax.hist(y_pred[~profitable], bins=30, alpha=0.6, label='Not Profitable', color='red')
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Predicted Log Return')
    ax.set_ylabel('Count')
    ax.set_title('Prediction Distribution by Outcome')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Evaluation plots saved to: {output_path}")


def plot_comparison_chart(
    log_return_result: BacktestResult,
    meta_labeler_result: Optional[Dict],
    output_path: str
) -> None:
    """Generate comparison bar chart."""
    if meta_labeler_result is None:
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    ml_bt = meta_labeler_result.get('backtest', {})

    # 1. Returns
    ax = axes[0]
    returns = [log_return_result.total_return, ml_bt.get('total_return', 0)]
    bars = ax.bar(['Log-Return', 'Meta-Labeler'], returns, color=['blue', 'orange'])
    ax.set_ylabel('Total Return (%)')
    ax.set_title('Return Comparison')
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    for bar, val in zip(bars, returns):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.1f}%', ha='center', va='bottom')

    # 2. Sharpe
    ax = axes[1]
    sharpes = [log_return_result.sharpe_ratio, ml_bt.get('sharpe_ratio', 0)]
    bars = ax.bar(['Log-Return', 'Meta-Labeler'], sharpes, color=['blue', 'orange'])
    ax.set_ylabel('Sharpe Ratio')
    ax.set_title('Sharpe Comparison')
    for bar, val in zip(bars, sharpes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.2f}', ha='center', va='bottom')

    # 3. Win Rate and Profit Factor
    ax = axes[2]
    x = np.arange(2)
    width = 0.35
    win_rates = [log_return_result.win_rate, ml_bt.get('win_rate', 0)]
    pf = [min(log_return_result.profit_factor, 10), min(ml_bt.get('profit_factor', 0), 10)]

    ax.bar(x - width/2, win_rates, width, label='Win Rate (%)', color='green', alpha=0.7)
    ax.bar(x + width/2, pf, width, label='Profit Factor (capped 10)', color='purple', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(['Log-Return', 'Meta-Labeler'])
    ax.set_title('Win Rate & Profit Factor')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Comparison chart saved to: {output_path}")


# =============================================================================
# Quality Gates
# =============================================================================

@dataclass
class LogReturnQualityGates:
    """Quality gate thresholds for log-return predictor."""
    min_spearman: float = 0.05  # Minimum rank correlation
    max_mae: float = 0.20  # Maximum mean absolute error
    min_direction_accuracy: float = 0.52  # Better than random
    min_sharpe: float = 0.5
    max_drawdown: float = 30.0  # percent
    min_profit_factor: float = 1.2


def check_quality_gates(
    regression_metrics: Dict[str, float],
    backtest_result: BacktestResult,
    gates: LogReturnQualityGates = None
) -> Tuple[bool, List[str]]:
    """
    Check if model passes quality gates.

    Returns:
        Tuple of (passed: bool, failures: List[str])
    """
    if gates is None:
        gates = LogReturnQualityGates()

    failures = []

    # Regression gates
    if regression_metrics['spearman'] < gates.min_spearman:
        failures.append(f"Spearman {regression_metrics['spearman']:.4f} < {gates.min_spearman}")

    if regression_metrics['mae'] > gates.max_mae:
        failures.append(f"MAE {regression_metrics['mae']:.4f} > {gates.max_mae}")

    if regression_metrics['direction_accuracy'] < gates.min_direction_accuracy:
        failures.append(f"Direction Acc {regression_metrics['direction_accuracy']:.4f} < {gates.min_direction_accuracy}")

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
# Main Evaluation Function
# =============================================================================

def evaluate_log_return_predictor(
    model_path: str,
    eval_data_path: str,
    output_dir: str,
    config: LogReturnConfig = None,
    meta_labeler_results_path: Optional[str] = None
) -> Dict:
    """
    Full evaluation of log-return predictor pipeline.

    Args:
        model_path: Path to trained log-return predictor
        eval_data_path: Path to evaluation data
        output_dir: Directory for output files
        config: Log-return configuration
        meta_labeler_results_path: Path to meta-labeler results for comparison

    Returns:
        Dict with all evaluation results
    """
    os.makedirs(output_dir, exist_ok=True)

    if config is None:
        config = LogReturnConfig()

    # Load model
    logger.info(f"Loading model from: {model_path}")
    predictor = LogReturnPredictor.load(model_path)

    # Load and prepare evaluation data
    logger.info(f"Loading evaluation data from: {eval_data_path}")
    df_raw = load_raw_data(eval_data_path)

    cfg = load_config("config.yaml")
    df_processed, _ = preprocess_data(df_raw, cfg, scaler=None)

    # Calculate log returns for target
    df_with_target = calculate_log_returns(
        df_processed,
        config.horizon_days,
        config.transaction_cost_bps
    )

    # Add ranker features
    df_with_features = add_ranker_features(df_with_target)

    logger.info(f"Evaluating on {len(df_with_features)} samples")

    # Get predictions
    predictions = predictor.predict(df_with_features)
    df_with_features = pd.concat([df_with_features, predictions], axis=1)

    y_true = df_with_features['log_return'].values
    y_pred = df_with_features['expected_log_return'].values
    q10 = df_with_features['q10'].values
    q90 = df_with_features['q90'].values

    # === Regression Metrics ===
    logger.info("\n" + "="*60)
    logger.info("REGRESSION METRICS")
    logger.info("="*60)

    reg_metrics = evaluate_regression(y_true, y_pred)
    for name, value in reg_metrics.items():
        if 'p' in name.lower():
            logger.info(f"  {name}: {value:.2e}")
        else:
            logger.info(f"  {name}: {value:.4f}")

    # === Quantile Coverage ===
    logger.info("\n" + "="*60)
    logger.info("UNCERTAINTY QUANTIFICATION")
    logger.info("="*60)

    quant_metrics = evaluate_quantile_coverage(y_true, q10, q90)
    for name, value in quant_metrics.items():
        logger.info(f"  {name}: {value:.4f}")

    # === Position Sizing ===
    logger.info("\n" + "="*60)
    logger.info("POSITION SIZING")
    logger.info("="*60)

    # Apply position sizing
    sized_df = size_positions_log_return(df_with_features[predictions.columns], config)
    df_with_features['position_size'] = sized_df['position_size']
    df_with_features['confidence'] = sized_df['confidence']

    # Filter to top-K per day (for fair comparison with meta-labeler)
    df_with_features['ranker_rank_daily'] = df_with_features.groupby('date')['ranker_score'].rank(
        ascending=False, method='first'
    )
    eval_df = df_with_features[df_with_features['ranker_rank_daily'] <= config.top_k_per_day].copy()

    logger.info(f"Position sizing on {len(eval_df)} top-K selections")
    logger.info(f"  Positions taken: {(eval_df['position_size'] > 0).sum()}")
    logger.info(f"  Avg position size: {eval_df[eval_df['position_size'] > 0]['position_size'].mean():.4f}")

    # === Backtest Results ===
    logger.info("\n" + "="*60)
    logger.info("BACKTEST RESULTS")
    logger.info("="*60)

    backtest = backtest_log_return_pipeline(
        eval_df,
        position_column='position_size',
        return_column='net_return'
    )

    logger.info(f"  Total Return: {backtest.total_return:.1f}%")
    logger.info(f"  Sharpe Ratio: {backtest.sharpe_ratio:.2f}")
    logger.info(f"  Max Drawdown: {backtest.max_drawdown:.1f}%")
    logger.info(f"  Win Rate: {backtest.win_rate:.1f}%")
    logger.info(f"  Avg Win: {backtest.avg_win:.2f}%")
    logger.info(f"  Avg Loss: {backtest.avg_loss:.2f}%")
    logger.info(f"  Profit Factor: {backtest.profit_factor:.2f}")
    logger.info(f"  Number of Trades: {backtest.n_trades}")

    # === Load Meta-Labeler Results for Comparison ===
    meta_labeler_results = None
    if meta_labeler_results_path:
        meta_labeler_results = load_meta_labeler_results(meta_labeler_results_path)

    # === Head-to-Head Comparison ===
    print_comparison(backtest, meta_labeler_results)

    # === Quality Gates ===
    logger.info("\n" + "="*60)
    logger.info("QUALITY GATES")
    logger.info("="*60)

    gates = LogReturnQualityGates()
    passed, failures = check_quality_gates(reg_metrics, backtest, gates)

    if passed:
        logger.info("  STATUS: PASSED")
    else:
        logger.warning("  STATUS: FAILED")
        for f in failures:
            logger.warning(f"    - {f}")

    # === Generate Plots ===
    plot_path = os.path.join(output_dir, 'log_return_evaluation.png')
    plot_evaluation_report(y_true, y_pred, q10, q90, backtest, plot_path)

    if meta_labeler_results:
        comparison_path = os.path.join(output_dir, 'comparison_chart.png')
        plot_comparison_chart(backtest, meta_labeler_results, comparison_path)

    # === Save Results ===
    results = {
        'regression': reg_metrics,
        'uncertainty': quant_metrics,
        'backtest': {
            'total_return': backtest.total_return,
            'sharpe_ratio': backtest.sharpe_ratio,
            'max_drawdown': backtest.max_drawdown,
            'win_rate': backtest.win_rate,
            'profit_factor': backtest.profit_factor,
            'n_trades': backtest.n_trades,
        },
        'quality_gates': {
            'passed': passed,
            'failures': failures,
        },
        'evaluation_date': datetime.now().isoformat(),
        'data_file': eval_data_path,
        'model_file': model_path,
    }

    results_path = os.path.join(output_dir, 'log_return_results.joblib')
    joblib.dump(results, results_path)
    logger.info(f"\nResults saved to: {results_path}")

    # Save trades for analysis
    trades_path = os.path.join(output_dir, 'log_return_trades.csv')
    backtest.trades_df.to_csv(trades_path, index=False)
    logger.info(f"Trades saved to: {trades_path}")

    # === Final Summary ===
    logger.info("\n" + "="*70)
    logger.info("SHADOW SYSTEM EVALUATION SUMMARY")
    logger.info("="*70)
    logger.info(f"Log-Return Predictor Performance:")
    logger.info(f"  - Spearman Correlation: {reg_metrics['spearman']:.4f}")
    logger.info(f"  - Direction Accuracy: {reg_metrics['direction_accuracy']:.2%}")
    logger.info(f"  - Total Return: {backtest.total_return:.1f}%")
    logger.info(f"  - Win Rate: {backtest.win_rate:.1f}%")
    logger.info(f"  - Quality Gates: {'PASSED' if passed else 'FAILED'}")

    if meta_labeler_results:
        ml_bt = meta_labeler_results.get('backtest', {})
        logger.info(f"\nMeta-Labeler Performance (for comparison):")
        logger.info(f"  - Total Return: {ml_bt.get('total_return', 0):.1f}%")
        logger.info(f"  - Win Rate: {ml_bt.get('win_rate', 0):.1f}%")

        # Recommendation
        lr_better = (
            backtest.total_return > ml_bt.get('total_return', 0) and
            backtest.sharpe_ratio > ml_bt.get('sharpe_ratio', 0)
        )
        logger.info(f"\nRECOMMENDATION: {'Consider switching to Log-Return' if lr_better else 'Keep Meta-Labeler'}")

    return results


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Log-Return Predictor (Shadow System)"
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained log-return predictor model'
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
        '--meta-labeler-results',
        type=str,
        default=None,
        help='Path to meta-labeler results for comparison'
    )
    parser.add_argument(
        '--horizon',
        type=int,
        default=5,
        help='Prediction horizon in days'
    )

    args = parser.parse_args()

    config = LogReturnConfig(horizon_days=args.horizon)

    evaluate_log_return_predictor(
        model_path=args.model,
        eval_data_path=args.eval_data,
        output_dir=args.output_dir,
        config=config,
        meta_labeler_results_path=args.meta_labeler_results,
    )


if __name__ == '__main__':
    main()
