#!/usr/bin/env python3
"""
Standalone CQF Model Evaluation Script

Evaluates a trained OptimalCQF model on test data with comprehensive metrics:
- Coverage accuracy (primary metric for quantile models)
- Pinball loss for each quantile
- Price prediction accuracy
- Regime-aware performance analysis
- Quality gate validation
"""

import pandas as pd
import numpy as np
import joblib
import logging
import os
import time
import argparse
import glob
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from sklearn.metrics import mean_pinball_loss
import sys

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_config, preprocess_data
from logger import setup_logger
from prod_cqf import OptimalCQF

# --- Configuration & Constants ---
logger = setup_logger(__name__, level=logging.INFO)

DEFAULT_QUANTILES = [0.05, 0.5, 0.95]
COVERAGE_TARGETS = {
    '90%_interval': (0.05, 0.95, 0.90),
    '80%_interval': (0.10, 0.90, 0.80), 
    '50%_interval': (0.25, 0.75, 0.50)
}

# --- Helper Functions for Model Discovery ---
def find_latest_cqf_model(model_dir: str = "../model_output") -> Optional[str]:
    """Find the latest CQF model file."""
    pattern = os.path.join(model_dir, "*cqf*.joblib")
    model_files = glob.glob(pattern)
    
    if not model_files:
        logger.error(f"No CQF model files found matching pattern: {pattern}")
        return None
    
    # Sort by modification time, get latest
    latest_model = max(model_files, key=os.path.getmtime)
    return latest_model

# --- Data Loading and Preparation ---
def load_and_prepare_eval_data(data_file: str, config_file: str) -> pd.DataFrame:
    """Load and preprocess evaluation data."""
    t_start = time.time()
    logger.info(f"Loading evaluation data from: {data_file}")
    
    try:
        # Load raw data
        df = pd.read_csv(data_file, low_memory=False)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        logger.info(f"Loaded {len(df)} raw rows. Date range: {df['date'].min()} to {df['date'].max()}")
        
        # Handle contract ID column
        if 'contractID' not in df.columns:
            if 'contract_id' in df.columns:
                df = df.rename(columns={'contract_id': 'contractID'})
            elif 'option_symbol' in df.columns:
                df = df.rename(columns={'option_symbol': 'contractID'})
            else:
                raise ValueError("Missing contract identifier column")
        
        df['contractID'] = df['contractID'].astype(str)
        df = df.sort_values(['date', 'contractID']).reset_index(drop=True)
        
        # Preprocess using utils
        config = load_config(config_file)
        if not config:
            raise ValueError("Failed to load configuration")
        
        df_processed, _ = preprocess_data(df, config, scaler=None)
        logger.info(f"Preprocessing complete. Shape: {df_processed.shape}")
        
        # Add regime features if missing (required by CQF models)
        from regime_tools import add_regime_features, add_realized_vol_features
        
        logger.info("Adding regime features...")
        df_processed = add_realized_vol_features(df_processed)
        df_processed = add_regime_features(df_processed)
        logger.info(f"Regime features added. Final shape: {df_processed.shape}")
        
        t_end = time.time()
        logger.info(f"Data loading and preprocessing finished in {t_end - t_start:.2f} seconds")
        return df_processed
        
    except Exception as e:
        logger.error(f"Error loading evaluation data: {e}", exc_info=True)
        raise

# --- Core Evaluation Functions ---
def calculate_coverage_metrics(y_true: np.ndarray, predictions: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Calculate coverage accuracy for quantile predictions."""
    metrics = {}
    
    # Check each coverage interval
    for interval_name, (lower_q, upper_q, target_coverage) in COVERAGE_TARGETS.items():
        lower_col = f'q{lower_q:.2f}'
        upper_col = f'q{upper_q:.2f}'
        
        if lower_col in predictions and upper_col in predictions:
            lower_pred = predictions[lower_col]
            upper_pred = predictions[upper_col]
            
            # Calculate actual coverage
            in_interval = (y_true >= lower_pred) & (y_true <= upper_pred)
            actual_coverage = np.mean(in_interval)
            
            metrics[f'{interval_name}_coverage'] = actual_coverage
            metrics[f'{interval_name}_target'] = target_coverage
            metrics[f'{interval_name}_error'] = abs(actual_coverage - target_coverage)
            
            logger.info(f"{interval_name}: {actual_coverage:.1%} coverage (target: {target_coverage:.1%})")
    
    return metrics

def calculate_pinball_metrics(y_true: np.ndarray, predictions: Dict[str, np.ndarray], 
                            quantiles: List[float]) -> Dict[str, float]:
    """Calculate pinball loss for each quantile."""
    metrics = {}
    
    for q in quantiles:
        col_name = f'q{q:.2f}'
        if col_name in predictions:
            y_pred = predictions[col_name]
            pinball = mean_pinball_loss(y_true, y_pred, alpha=q)
            metrics[f'pinball_q{q:.2f}'] = pinball
            logger.info(f"Pinball loss Q{q:.2f}: {pinball:.6f}")
    
    return metrics

def calculate_regime_performance(eval_df: pd.DataFrame, predictions: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Analyze performance across different market regimes."""
    metrics = {}
    
    if 'vix_d_close' not in eval_df.columns:
        logger.warning("VIX data not available for regime analysis")
        return metrics
    
    # Define VIX regimes
    vix_regimes = pd.cut(eval_df['vix_d_close'], 
                        bins=[0, 15, 25, 100], 
                        labels=['low_vix', 'medium_vix', 'high_vix'])
    
    # Calculate coverage by regime
    if 'q0.05' in predictions and 'q0.95' in predictions:
        y_true = eval_df['target_pnl'].values
        in_interval = (y_true >= predictions['q0.05']) & (y_true <= predictions['q0.95'])
        
        for regime in ['low_vix', 'medium_vix', 'high_vix']:
            mask = (vix_regimes == regime)
            if mask.sum() > 0:
                regime_coverage = np.mean(in_interval[mask])
                metrics[f'coverage_{regime}'] = regime_coverage
                logger.info(f"Coverage {regime}: {regime_coverage:.1%} ({mask.sum()} samples)")
    
    # Calculate stress score performance if available
    if 'stress_score' in eval_df.columns and 'q0.50' in predictions:
        y_true = eval_df['target_pnl'].values
        y_pred_median = predictions['q0.50']
        
        # Performance by stress level
        for stress_level in [0, 1, 2, 3, 4]:
            mask = (eval_df['stress_score'] == stress_level)
            if mask.sum() > 0:
                mse = np.mean((y_true[mask] - y_pred_median[mask]) ** 2)
                metrics[f'mse_stress_{stress_level}'] = mse
                logger.info(f"MSE stress level {stress_level}: {mse:.6f} ({mask.sum()} samples)")
    
    return metrics

def check_quality_gates(coverage_metrics: Dict[str, float]) -> bool:
    """Check if model passes quality gates for production use."""
    gates_passed = True
    
    # Primary gate: 90% interval should have 85-95% coverage
    if '90%_interval_coverage' in coverage_metrics:
        coverage = coverage_metrics['90%_interval_coverage']
        if coverage < 0.85 or coverage > 0.95:
            logger.error(f"❌ Quality gate FAILED: 90% interval coverage {coverage:.1%} outside [85%, 95%]")
            gates_passed = False
        else:
            logger.info(f"✅ Quality gate PASSED: 90% interval coverage {coverage:.1%}")
    
    # Secondary gates for other intervals
    for interval in ['80%_interval', '50%_interval']:
        if f'{interval}_error' in coverage_metrics:
            error = coverage_metrics[f'{interval}_error']
            if error > 0.10:  # Allow 10% error tolerance
                logger.warning(f"⚠️  Quality gate WARNING: {interval} error {error:.1%} > 10%")
            else:
                logger.info(f"✅ Quality gate PASSED: {interval} error {error:.1%}")
    
    return gates_passed

# --- Main Evaluation Function ---
def evaluate_cqf_model(model_file: str, eval_data_file: str, config_file: str,
                      sample_size: Optional[int] = None, verbose: bool = True) -> bool:
    """Evaluate CQF model on test data."""
    
    if verbose:
        logger.info("=== CQF Model Evaluation ===")
        logger.info(f"Model: {os.path.basename(model_file)}")
        logger.info(f"Data: {os.path.basename(eval_data_file)}")
    
    try:
        # Load model
        logger.info("Loading CQF model...")
        saved_data = joblib.load(model_file)
        logger.info(f"✅ Model data loaded successfully: {type(saved_data)}")
        
        # Handle different save formats
        if isinstance(saved_data, dict):
            # Create OptimalCQF instance and restore from saved state
            logger.info("Reconstructing OptimalCQF from saved dictionary...")
            
            # Extract metadata to determine model parameters
            if 'quantiles' in saved_data:
                quantiles = saved_data['quantiles']
            else:
                quantiles = DEFAULT_QUANTILES
                logger.warning(f"Quantiles not found in saved data, using default: {quantiles}")
            
            if 'horizon' in saved_data:
                horizon = saved_data['horizon']
            else:
                horizon = 5
                logger.warning(f"Horizon not found in saved data, using default: {horizon}")
            
            # Create new OptimalCQF instance
            cqf_model = OptimalCQF(quantiles=quantiles, horizon=horizon)
            
            # Restore saved components
            if 'models' in saved_data:
                cqf_model.models = saved_data['models']
                logger.info(f"Restored {len(cqf_model.models)} quantile models")
            
            if 'preprocessor' in saved_data:
                cqf_model.preprocessor = saved_data['preprocessor']
                logger.info("Restored preprocessor")
            
            if 'feature_names' in saved_data:
                cqf_model.feature_names = saved_data['feature_names']
                logger.info(f"Restored {len(cqf_model.feature_names)} feature names")
            
            if 'conformal_adjustments' in saved_data:
                cqf_model.conformal_adjustments = saved_data['conformal_adjustments']
                logger.info("Restored conformal adjustments")
            
            if 'prob_calibrator' in saved_data:
                cqf_model.prob_calibrator = saved_data['prob_calibrator']
                logger.info("Restored probability calibrator")
            
            logger.info("✅ OptimalCQF successfully reconstructed from saved data")
            
        elif hasattr(saved_data, 'predict_quantiles'):
            # Already an OptimalCQF instance
            cqf_model = saved_data
            logger.info("✅ OptimalCQF instance loaded directly")
        else:
            raise ValueError(f"Unsupported model format: {type(saved_data)}")
        
        # Load and prepare evaluation data
        logger.info("Loading evaluation data...")
        eval_df = load_and_prepare_eval_data(eval_data_file, config_file)
        
        if len(eval_df) == 0:
            logger.error("No evaluation data available")
            return False
        
        # Calculate delta-hedged PnL targets BEFORE sampling
        # (Need temporal continuity for future price lookups)
        logger.info("Calculating delta-hedged PnL targets...")
        eval_df = cqf_model.calculate_delta_hedged_pnl(eval_df, horizon=cqf_model.horizon)
        
        # Remove rows with invalid targets
        eval_df = eval_df.dropna(subset=['target_pnl'])
        logger.info(f"After target calculation: {len(eval_df)} valid samples")
        
        # Sample data if requested (after target calculation)
        if sample_size and len(eval_df) > sample_size:
            eval_df = eval_df.sample(n=sample_size, random_state=42)
            logger.info(f"Sampled {sample_size} rows for evaluation")
        
        if 'target_pnl' not in eval_df.columns or eval_df['target_pnl'].isna().all():
            logger.error("Failed to calculate valid targets")
            return False
        
        valid_targets = eval_df['target_pnl'].dropna()
        logger.info(f"✅ Targets calculated for {len(valid_targets)} samples")
        logger.info(f"Target stats: mean={valid_targets.mean():.4f}, std={valid_targets.std():.4f}")
        
        # Make quantile predictions
        logger.info("Generating quantile predictions...")
        quantile_preds = cqf_model.predict_quantiles(eval_df, apply_conformal=True)
        
        if quantile_preds.empty:
            logger.error("Failed to generate predictions")
            return False
        
        logger.info(f"✅ Predictions generated: {quantile_preds.shape}")
        
        # Prepare data for evaluation
        # Align predictions with valid targets
        eval_df_clean = eval_df.dropna(subset=['target_pnl'])
        quantile_preds_clean = quantile_preds.loc[eval_df_clean.index]
        y_true = eval_df_clean['target_pnl'].values
        
        predictions_dict = {col: quantile_preds_clean[col].values for col in quantile_preds_clean.columns}
        
        logger.info(f"Evaluation on {len(y_true)} samples")
        
        # Calculate coverage metrics
        logger.info("\n=== Coverage Analysis ===")
        coverage_metrics = calculate_coverage_metrics(y_true, predictions_dict)
        
        # Calculate pinball metrics
        logger.info("\n=== Pinball Loss Analysis ===")
        pinball_metrics = calculate_pinball_metrics(y_true, predictions_dict, cqf_model.quantiles)
        
        # Regime performance analysis
        logger.info("\n=== Regime Performance Analysis ===")
        regime_metrics = calculate_regime_performance(eval_df_clean, predictions_dict)
        
        # Quality gates
        logger.info("\n=== Quality Gate Validation ===")
        quality_passed = check_quality_gates(coverage_metrics)
        
        # Summary
        all_metrics = {**coverage_metrics, **pinball_metrics, **regime_metrics}
        
        if verbose:
            logger.info("\n=== EVALUATION SUMMARY ===")
            logger.info(f"✅ Model: {os.path.basename(model_file)}")
            logger.info(f"✅ Samples evaluated: {len(y_true)}")
            logger.info(f"✅ Quality gates: {'PASSED' if quality_passed else 'FAILED'}")
            
            # Key metrics
            if '90%_interval_coverage' in coverage_metrics:
                logger.info(f"✅ Main coverage (90%): {coverage_metrics['90%_interval_coverage']:.1%}")
            
            if 'pinball_q0.50' in pinball_metrics:
                logger.info(f"✅ Median pinball loss: {pinball_metrics['pinball_q0.50']:.6f}")
        
        return quality_passed
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}", exc_info=True)
        return False

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained CQF model")
    
    # Model specification
    parser.add_argument("--model-file", type=str, help="Path to trained CQF model (.joblib)")
    parser.add_argument("--eval-data-file", type=str, required=True, help="Evaluation data CSV file")
    parser.add_argument("--config-file", type=str, default="config.yaml", help="Configuration file")
    
    # Auto-discovery
    parser.add_argument("--model-dir", type=str, default="../model_output",
                       help="Directory to auto-discover CQF model files")
    parser.add_argument("--auto-discover", action="store_true",
                       help="Automatically find latest CQF model in model-dir")
    
    # Evaluation options
    parser.add_argument("--sample-size", type=int, help="Sample size for evaluation")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    
    args = parser.parse_args()
    
    # Auto-discovery logic
    if args.auto_discover or not args.model_file:
        logger.info("Auto-discovering CQF model files...")
        model_file = find_latest_cqf_model(args.model_dir)
        
        if not model_file:
            logger.error("Could not find CQF model file. Please specify manually.")
            sys.exit(1)
        
        args.model_file = model_file
        logger.info(f"✅ Auto-discovered model: {os.path.basename(args.model_file)}")
    
    # Validate required files exist
    required_files = [args.model_file, args.eval_data_file, args.config_file]
    for file_path in required_files:
        if not os.path.exists(file_path):
            logger.error(f"Required file not found: {file_path}")
            sys.exit(1)
    
    # Run evaluation
    success = evaluate_cqf_model(
        model_file=args.model_file,
        eval_data_file=args.eval_data_file,
        config_file=args.config_file,
        sample_size=args.sample_size,
        verbose=not args.quiet
    )
    
    logger.info(f"\n🏁 CQF Evaluation {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()