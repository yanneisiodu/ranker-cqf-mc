"""
Optuna Hyperparameter Optimization for Hybrid Kelly
====================================================

Uses Bayesian optimization to find optimal position sizing parameters
that maximize risk-adjusted returns.

Author: Generated with Claude Code
"""

from __future__ import annotations
import logging
import os
import sys
import argparse
import json
from dataclasses import dataclass, asdict
from typing import Dict, Optional

import numpy as np
import pandas as pd
import joblib
import optuna
from optuna.samplers import TPESampler

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import load_config, preprocess_data
from prod_log_return_predictor import (
    LogReturnPredictor, LogReturnConfig,
    load_raw_data, calculate_log_returns, add_ranker_features
)
from prod_meta_labeler import MetaLabeler, MetaLabelerConfig
from prod_hybrid_kelly import (
    HybridKellyConfig, HybridKellySizer,
    kelly_criterion, estimate_win_loss_ratio
)

logging.basicConfig(
    level=logging.WARNING,  # Reduce noise during optimization
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress optuna info logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


# =============================================================================
# Global Data Cache (loaded once, reused across trials)
# =============================================================================

class DataCache:
    """Cache loaded data and models to avoid reloading each trial."""

    def __init__(self):
        self.eval_df = None
        self.meta_labeler = None
        self.log_return_model = None
        self.is_loaded = False

    def load(
        self,
        meta_labeler_path: str,
        log_return_path: str,
        eval_data_path: str,
        ranker_model_path: str = None
    ):
        """Load all data and models once."""
        if self.is_loaded:
            return

        print("Loading models and data (one-time)...")

        # Load models
        meta_artifacts = joblib.load(meta_labeler_path)
        self.log_return_model = LogReturnPredictor.load(log_return_path)

        # Setup meta-labeler
        meta_config = meta_artifacts.get('config', MetaLabelerConfig())
        self.meta_labeler = MetaLabeler(meta_config)
        self.meta_labeler.classifier = meta_artifacts['classifier']
        self.meta_labeler.calibrator = meta_artifacts['calibrator']
        self.meta_labeler.scaler = meta_artifacts['scaler']
        self.meta_labeler.imputer = meta_artifacts['imputer']
        self.meta_labeler.feature_names = meta_artifacts['feature_names']

        # Load ranker if provided
        ranker_pipeline = None
        ranker_feature_cols = None
        if ranker_model_path and os.path.exists(ranker_model_path):
            ranker_pipeline = joblib.load(ranker_model_path)
            feature_path = ranker_model_path.replace('.joblib', '_feature_names.pkl')
            if os.path.exists(feature_path):
                ranker_feature_cols = joblib.load(feature_path)

        # Load and prepare data
        df_raw = load_raw_data(eval_data_path)
        cfg = load_config("config.yaml")
        df_processed, _ = preprocess_data(df_raw, cfg, scaler=None)

        # Calculate log returns
        lr_config = LogReturnConfig()
        df_with_returns = calculate_log_returns(
            df_processed,
            lr_config.horizon_days,
            lr_config.transaction_cost_bps
        )

        # Add ranker features
        if ranker_pipeline is not None and ranker_feature_cols is not None:
            df_with_features = add_ranker_features(df_with_returns)

            # Fill missing columns
            for col in ranker_feature_cols:
                if col not in df_with_features.columns:
                    df_with_features[col] = 0

            # Score with ranker
            X_ranker = df_with_features[ranker_feature_cols].copy()
            if 'type' in X_ranker.columns:
                X_ranker['type'] = X_ranker['type'].astype(str)

            ranker_scores = ranker_pipeline.predict(X_ranker)
            df_with_features['ranker_score'] = ranker_scores
        else:
            df_with_features = add_ranker_features(df_with_returns)

        # Get predictions
        df_with_features['prob_profit'] = self.meta_labeler.predict_proba(df_with_features)

        log_return_preds = self.log_return_model.predict(df_with_features)
        df_with_features['expected_log_return'] = log_return_preds['expected_log_return']
        df_with_features['uncertainty'] = log_return_preds['uncertainty']

        # Filter to top-K per day
        df_with_features['ranker_rank_daily'] = df_with_features.groupby('date')['ranker_score'].rank(
            ascending=False, method='first'
        )
        self.eval_df = df_with_features[df_with_features['ranker_rank_daily'] <= 20].copy()

        print(f"Data loaded: {len(self.eval_df)} samples ready for optimization")
        self.is_loaded = True


# Global cache instance
DATA_CACHE = DataCache()


# =============================================================================
# Fast Backtest Function
# =============================================================================

def run_backtest(df: pd.DataFrame, config: HybridKellyConfig) -> Dict:
    """
    Run backtest with given config and return metrics.

    This is a streamlined version optimized for speed.
    """
    df = df.copy()
    cfg = config

    # Get predictions
    probs = df['prob_profit'].values
    returns = df['expected_log_return'].values
    uncertainties = df['uncertainty'].values

    # Calculate position sizes (vectorized where possible)
    position_sizes = np.zeros(len(df))

    for i in range(len(df)):
        p = probs[i]
        exp_ret = returns[i]
        unc = uncertainties[i]

        # Gates
        if p < cfg.min_prob_to_trade:
            continue
        if exp_ret < cfg.min_expected_return:
            continue

        # Win/loss ratio
        b = estimate_win_loss_ratio(exp_ret, p, unc)
        if b < cfg.min_win_loss_ratio:
            continue
        b = min(b, cfg.max_win_loss_ratio)

        # Kelly
        kelly = kelly_criterion(p, b, cfg.kelly_fraction)
        if kelly <= 0:
            continue

        # Uncertainty penalty
        if unc > 0:
            kelly *= np.exp(-cfg.uncertainty_penalty * unc)

        # Conviction boost
        if p >= cfg.high_conviction_threshold:
            kelly *= 1.0 + (p - cfg.high_conviction_threshold) * 0.5

        # Position limits
        size = np.clip(kelly, cfg.min_position_pct, cfg.max_position_pct)
        position_sizes[i] = size

    df['position_size'] = position_sizes

    # Apply daily portfolio cap
    def cap_daily(group):
        total = group['position_size'].sum()
        if total > cfg.max_portfolio_risk:
            group['position_size'] = group['position_size'] * (cfg.max_portfolio_risk / total)
        return group

    df = df.groupby('date', group_keys=False).apply(cap_daily)

    # Filter to actual trades
    trades_df = df[df['position_size'] > 0].copy()

    if len(trades_df) == 0:
        return {
            'total_return': -100,
            'sharpe': 0,
            'max_drawdown': 100,
            'win_rate': 0,
            'profit_factor': 0,
            'n_trades': 0,
        }

    # Calculate returns
    trades_df['trade_return'] = trades_df['position_size'] * trades_df['net_return']
    daily_returns = trades_df.groupby('date')['trade_return'].sum()

    # Equity curve
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

    gross_profit = wins['trade_return'].sum() if len(wins) > 0 else 0
    gross_loss = abs(losses['trade_return'].sum()) if len(losses) > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    return {
        'total_return': total_return,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'n_trades': len(trades_df),
    }


# =============================================================================
# Optuna Objective Function
# =============================================================================

def create_objective(objective_type: str = 'balanced'):
    """
    Create Optuna objective function.

    Objective types:
    - 'return': Maximize total return
    - 'sharpe': Maximize Sharpe ratio
    - 'balanced': Maximize return * sharpe / (1 + drawdown)
    - 'calmar': Maximize return / max_drawdown
    """

    def objective(trial: optuna.Trial) -> float:
        # Sample hyperparameters
        config = HybridKellyConfig(
            kelly_fraction=trial.suggest_float('kelly_fraction', 0.10, 0.50),
            min_position_pct=trial.suggest_float('min_position_pct', 0.001, 0.02),
            max_position_pct=trial.suggest_float('max_position_pct', 0.03, 0.15),
            max_portfolio_risk=trial.suggest_float('max_portfolio_risk', 0.20, 0.70),
            min_prob_to_trade=trial.suggest_float('min_prob_to_trade', 0.40, 0.60),
            high_conviction_threshold=trial.suggest_float('high_conviction_threshold', 0.55, 0.75),
            min_expected_return=trial.suggest_float('min_expected_return', 0.005, 0.05),
            uncertainty_penalty=trial.suggest_float('uncertainty_penalty', 0.5, 2.0),
            min_win_loss_ratio=trial.suggest_float('min_win_loss_ratio', 0.3, 1.0),
            max_win_loss_ratio=trial.suggest_float('max_win_loss_ratio', 5.0, 15.0),
        )

        # Run backtest
        results = run_backtest(DATA_CACHE.eval_df, config)

        # Calculate objective
        total_return = results['total_return']
        sharpe = results['sharpe']
        max_dd = results['max_drawdown']
        n_trades = results['n_trades']

        # Penalize if too few trades (unrealistic)
        if n_trades < 50:
            return -1000

        # Penalize extreme drawdown
        if max_dd > 50:
            return -500

        if objective_type == 'return':
            return total_return
        elif objective_type == 'sharpe':
            return sharpe
        elif objective_type == 'calmar':
            if max_dd > 0:
                return total_return / max_dd
            return total_return
        else:  # balanced
            # Maximize: return * sharpe / (1 + drawdown)
            if max_dd >= 0:
                score = total_return * max(sharpe, 0) / (1 + max_dd / 10)
            else:
                score = total_return * max(sharpe, 0)
            return score

    return objective


# =============================================================================
# Main Optimization
# =============================================================================

def optimize_kelly_params(
    meta_labeler_path: str,
    log_return_path: str,
    eval_data_path: str,
    ranker_model_path: str = None,
    n_trials: int = 100,
    objective_type: str = 'balanced',
    output_dir: str = 'optimization_output'
) -> Dict:
    """
    Run Optuna optimization to find best Kelly parameters.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    DATA_CACHE.load(
        meta_labeler_path=meta_labeler_path,
        log_return_path=log_return_path,
        eval_data_path=eval_data_path,
        ranker_model_path=ranker_model_path
    )

    print(f"\n{'='*60}")
    print(f"OPTUNA OPTIMIZATION")
    print(f"{'='*60}")
    print(f"Objective: {objective_type}")
    print(f"Trials: {n_trials}")
    print(f"{'='*60}\n")

    # Create study
    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        study_name='hybrid_kelly_optimization'
    )

    # Optimize
    objective = create_objective(objective_type)
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
        n_jobs=1  # Sequential for stability
    )

    # Best parameters
    best_params = study.best_params
    best_value = study.best_value

    print(f"\n{'='*60}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"Best objective value: {best_value:.2f}")
    print(f"\nBest parameters:")
    for name, value in best_params.items():
        print(f"  {name}: {value:.4f}")

    # Run final backtest with best params
    best_config = HybridKellyConfig(**best_params)
    final_results = run_backtest(DATA_CACHE.eval_df, best_config)

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS WITH OPTIMIZED PARAMS")
    print(f"{'='*60}")
    print(f"  Total Return: {final_results['total_return']:.1f}%")
    print(f"  Sharpe Ratio: {final_results['sharpe']:.2f}")
    print(f"  Max Drawdown: {final_results['max_drawdown']:.1f}%")
    print(f"  Win Rate: {final_results['win_rate']:.1f}%")
    print(f"  Profit Factor: {final_results['profit_factor']:.2f}")
    print(f"  Number of Trades: {final_results['n_trades']}")

    # Save results
    output = {
        'best_params': best_params,
        'best_objective': best_value,
        'final_results': final_results,
        'objective_type': objective_type,
        'n_trials': n_trials,
    }

    # Save as JSON
    json_path = os.path.join(output_dir, 'best_params.json')
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\nBest parameters saved to: {json_path}")

    # Save study
    study_path = os.path.join(output_dir, 'optuna_study.pkl')
    joblib.dump(study, study_path)
    print(f"Study saved to: {study_path}")

    return output


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Optimize Hybrid Kelly parameters with Optuna"
    )
    parser.add_argument(
        '--meta-labeler',
        type=str,
        default='model_output/meta_labeler.joblib',
        help='Path to Meta-Labeler model'
    )
    parser.add_argument(
        '--log-return',
        type=str,
        default='model_output/log_return_predictor.joblib',
        help='Path to Log-Return Predictor'
    )
    parser.add_argument(
        '--eval-data',
        type=str,
        default='../Data/year_2024_data.csv',
        help='Path to evaluation data (optimize on this)'
    )
    parser.add_argument(
        '--ranker',
        type=str,
        default='model_output/ranker.joblib',
        help='Path to ranker model'
    )
    parser.add_argument(
        '--n-trials',
        type=int,
        default=100,
        help='Number of optimization trials'
    )
    parser.add_argument(
        '--objective',
        type=str,
        default='balanced',
        choices=['return', 'sharpe', 'balanced', 'calmar'],
        help='Optimization objective'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='optimization_output',
        help='Output directory'
    )

    args = parser.parse_args()

    optimize_kelly_params(
        meta_labeler_path=args.meta_labeler,
        log_return_path=args.log_return,
        eval_data_path=args.eval_data,
        ranker_model_path=args.ranker,
        n_trials=args.n_trials,
        objective_type=args.objective,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
