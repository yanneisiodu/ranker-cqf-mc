#!/usr/bin/env python3
"""
Integrated Options Trading System

Combines:
1. Step 7 CQF (Calibrated Quantile Forecasting) for risk quantification
2. XGBoost Ranking Model for option selection  
3. Enhanced Stress Monte Carlo for scenario analysis

Usage:
    python integrated_trading_system.py --eval-data data_combined/eval_2024_2025.csv --output trading_recommendations.csv
"""

import pandas as pd
import numpy as np
import joblib
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

# Import existing modules
from utils import load_config, preprocess_data
from logger import setup_logger
from regime_tools import add_regime_features, add_realized_vol_features

warnings.filterwarnings('ignore')
logger = setup_logger(__name__, level=logging.INFO)

class IntegratedTradingSystem:
    """
    Complete Options Trading System integrating CQF risk assessment with ranking-based selection.
    """
    
    def __init__(self, cqf_model_path: str, ranking_model_path: str, 
                 ranking_features_path: str, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        
        # Load CQF model (Step 7 - Black Swan protection)
        logger.info(f"Loading CQF model from: {cqf_model_path}")
        cqf_artifact = joblib.load(cqf_model_path)
        
        self.cqf_models = cqf_artifact['models']
        self.cqf_preprocessor = cqf_artifact['preprocessor'] 
        self.cqf_feature_names = cqf_artifact['feature_names']
        self.conformal_adjustments = cqf_artifact.get('conformal_adjustments', {})
        self.conformal_calibrator = cqf_artifact.get('conformal_calibrator')
        self.evt_adjuster = cqf_artifact.get('evt_adjuster')
        self.prob_calibrator = cqf_artifact.get('prob_calibrator')
        
        logger.info(f"✅ CQF model loaded: {len(self.cqf_models)} quantiles, {len(self.cqf_feature_names)} features")
        
        # Load XGBoost Ranking model
        logger.info(f"Loading XGBoost Ranking model from: {ranking_model_path}")
        self.ranking_model = joblib.load(ranking_model_path)
        
        logger.info(f"Loading ranking features from: {ranking_features_path}")
        self.ranking_feature_names = joblib.load(ranking_features_path)
        
        logger.info(f"✅ Ranking model loaded: {len(self.ranking_feature_names)} features")
        
        # Model metadata
        self.cqf_quantiles = [0.05, 0.5, 0.95]
        self.horizon = 5
        
    def preprocess_data(self, data_file: str) -> pd.DataFrame:
        """
        Complete preprocessing pipeline combining CQF and ranking features.
        """
        logger.info(f"Loading and preprocessing data from: {data_file}")
        
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
        
        logger.info(f"Raw data: {len(df_raw)} rows, date range: {df_raw['date'].min()} to {df_raw['date'].max()}")
        
        # Apply existing preprocessing (gets causal features)
        df_processed, _ = preprocess_data(df_raw, self.config, scaler=None)
        
        # Add regime-aware features (Step 6-7 enhancements)
        df_processed = add_regime_features(df_processed)
        df_processed = add_realized_vol_features(df_processed)
        
        logger.info(f"Preprocessed data: {len(df_processed)} rows, {len(df_processed.columns)} features")
        return df_processed
        
    def calculate_delta_hedged_pnl(self, df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
        """
        Calculate delta-hedged P&L targets for CQF evaluation.
        """
        logger.info(f"Calculating delta-hedged PnL targets (horizon={horizon}d)")
        
        price_col = 'last_raw' if 'last_raw' in df.columns else 'last'
        
        if 'contractID' not in df.columns or price_col not in df.columns:
            logger.error("Missing required columns for PnL calculation")
            return df.assign(target_pnl=np.nan)
        
        df = df.sort_values(['contractID', 'date']).reset_index(drop=True)
        
        # Calculate forward option prices and target dates
        df_grouped = df.groupby('contractID')
        future_option_price = df_grouped[price_col].shift(-horizon)
        df['target_date'] = df_grouped['date'].shift(-horizon)
        
        # Calculate forward underlying prices (using SPY close)
        spy_col = 'spy_d_close'
        if spy_col in df.columns:
            spy_daily = df[['date', spy_col]].drop_duplicates('date').sort_values('date')
            spy_daily['spy_fwd'] = spy_daily[spy_col].shift(-horizon)
            df = df.merge(spy_daily[['date', 'spy_fwd']], on='date', how='left')
            
            option_pnl = (future_option_price - df[price_col]) / df[price_col]
            underlying_pnl = (df['spy_fwd'] - df[spy_col]) / df[spy_col]
            
            if 'delta' in df.columns:
                df['target_pnl'] = option_pnl + df['delta'] * (-underlying_pnl)
            else:
                df['target_pnl'] = option_pnl
        else:
            df['target_pnl'] = (future_option_price - df[price_col]) / df[price_col]
        
        # Clean infinite and NaN values
        df['target_pnl'] = df['target_pnl'].replace([np.inf, -np.inf], np.nan)
        
        initial_rows = len(df)
        df = df.dropna(subset=['target_pnl'])
        dropped_rows = initial_rows - len(df)
        
        logger.info(f"PnL calculation complete. Dropped {dropped_rows} rows, target stats: mean={df['target_pnl'].mean():.4f}")
        return df.sort_values('date')
    
    def predict_cqf_quantiles(self, df: pd.DataFrame, apply_conformal: bool = True) -> pd.DataFrame:
        """
        Generate CQF quantile predictions with Step 7 regime-adaptive calibration.
        """
        logger.info("Generating CQF quantile predictions")
        
        # Filter to CQF features
        available_cqf_features = [col for col in self.cqf_feature_names if col in df.columns]
        if len(available_cqf_features) != len(self.cqf_feature_names):
            missing = set(self.cqf_feature_names) - set(available_cqf_features)
            logger.warning(f"Missing CQF features: {missing}")
        
        X = df[available_cqf_features]
        X_scaled = self.cqf_preprocessor.transform(X)
        
        # Get raw predictions
        raw_predictions = {}
        for quantile, model in self.cqf_models.items():
            raw_predictions[quantile] = model.predict(X_scaled)
        
        # Apply regime-adaptive conformal calibration
        if apply_conformal:
            vix = df['vix_d_close'] if 'vix_d_close' in df.columns else None
            dte = df['days_to_exp'] if 'days_to_exp' in df.columns else None
            date = df['date'] if 'date' in df.columns else None
            
            # Step 7: Real-time stress detection with severity scaling
            vol_emergency_now = df['vol_emergency'].iloc[-1] if 'vol_emergency' in df.columns and len(df) > 0 else False
            vol_severity_pred = df['vol_severity'].iloc[-1] if 'vol_severity' in df.columns and len(df) > 0 else 1.0
            
            vol_of_vol_high = False
            if 'vol_of_vol_20d' in df.columns and len(df) > 0:
                vol_of_vol_current = df['vol_of_vol_20d'].iloc[-1] if not pd.isna(df['vol_of_vol_20d'].iloc[-1]) else 0
                vol_of_vol_threshold = df['vol_of_vol_20d'].quantile(0.85) if len(df) > 50 else 0
                vol_of_vol_high = vol_of_vol_current > vol_of_vol_threshold
            
            is_black_swan = vol_severity_pred > 5.0
            use_adaptive = (self.conformal_calibrator is not None) and (vol_emergency_now or vol_of_vol_high or is_black_swan)
            
            if use_adaptive:
                logger.info(f"STRESS DETECTED - Emergency: {vol_emergency_now}, Vol-of-Vol: {vol_of_vol_high}, Severity: {vol_severity_pred:.1f}x - Using adaptive conformal")
                
                adjusted = self.conformal_calibrator.adjust(
                    df, raw_predictions.get(0.05), raw_predictions.get(0.5), 
                    raw_predictions.get(0.95), vix=vix, dte=dte, date=date
                )
                
                predictions = {
                    'q0.05': adjusted.get('q0.05', raw_predictions.get(0.05)),
                    'q0.50': adjusted.get('q0.50', raw_predictions.get(0.5)),
                    'q0.95': adjusted.get('q0.95', raw_predictions.get(0.95))
                }
                
                # Step 7: Black Swan emergency multiplier
                if vol_severity_pred > 10.0:
                    emergency_multiplier = min(1.0 + (vol_severity_pred - 10.0) * 0.2, 5.0)
                    q50 = predictions['q0.50']
                    q_width = predictions['q0.95'] - predictions['q0.05']
                    
                    predictions['q0.05'] = q50 - (q_width * emergency_multiplier / 2)
                    predictions['q0.95'] = q50 + (q_width * emergency_multiplier / 2)
                    
                    logger.warning(f"🚨 BLACK SWAN EMERGENCY: Applied {emergency_multiplier:.1f}x interval multiplier")
            else:
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
            predictions = {f'q{quantile:.2f}': pred for quantile, pred in raw_predictions.items()}
        
        # Enforce monotonicity
        if 'q0.05' in predictions and 'q0.50' in predictions and 'q0.95' in predictions:
            q05 = predictions['q0.05']
            q50 = np.maximum(predictions['q0.50'], q05)
            q95 = np.maximum(predictions['q0.95'], q50)
            q50 = np.minimum(q50, q95)
            q05 = np.minimum(q05, q50)
            
            predictions['q0.05'], predictions['q0.50'], predictions['q0.95'] = q05, q50, q95
        
        return pd.DataFrame(predictions, index=df.index)
    
    def calculate_risk_features(self, quantile_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive risk and decision features from CQF quantiles.
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
            
            # Probability of profit with isotonic calibration
            prob_profit_raw = np.where(
                q95 <= 0, 0.0,
                np.where(q05 >= 0, 1.0, 0.5 + 0.45 * (q50 / (q95 - q05 + 1e-8)))
            )
            prob_profit_raw = np.clip(prob_profit_raw, 0.0, 1.0)
            
            if self.prob_calibrator is not None:
                result['prob_profit'] = self.prob_calibrator.predict(prob_profit_raw)
            else:
                result['prob_profit'] = prob_profit_raw
            
            # Risk-adjusted utility
            risk_penalty = 0.5
            result['utility'] = result['expected_return'] - risk_penalty * result['downside_risk']
            
            # Regime-aware risk scaling
            if 'vol_severity' in quantile_df.columns:
                vol_severity = quantile_df['vol_severity'].fillna(1.0)
                result['regime_adjusted_risk'] = result['downside_risk'] * np.sqrt(vol_severity)
                result['regime_adjusted_utility'] = result['expected_return'] - risk_penalty * result['regime_adjusted_risk']
        
        return result
    
    def generate_ranking_scores(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate XGBoost ranking scores for option selection.
        """
        logger.info("Generating XGBoost ranking scores")
        
        # Filter to ranking features
        available_ranking_features = [col for col in self.ranking_feature_names if col in df.columns]
        if len(available_ranking_features) != len(self.ranking_feature_names):
            missing = set(self.ranking_feature_names) - set(available_ranking_features)
            logger.warning(f"Missing ranking features: {missing}")
        
        X_ranking = df[available_ranking_features]
        
        # Handle any remaining NaNs
        X_ranking = X_ranking.fillna(X_ranking.median())
        
        # Generate ranking scores
        ranking_scores = self.ranking_model.predict(X_ranking)
        
        logger.info(f"Ranking scores: mean={ranking_scores.mean():.4f}, std={ranking_scores.std():.4f}")
        return pd.Series(ranking_scores, index=df.index, name='ranking_score')
    
    def generate_trading_recommendations(self, df: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
        """
        Generate final trading recommendations combining CQF risk assessment and ranking.
        """
        logger.info(f"Generating top {top_n} trading recommendations")
        
        # Get CQF quantile predictions
        cqf_quantiles = self.predict_cqf_quantiles(df, apply_conformal=True)
        
        # Calculate comprehensive risk features
        risk_features = self.calculate_risk_features(cqf_quantiles)
        
        # Generate ranking scores
        ranking_scores = self.generate_ranking_scores(df)
        
        # Combine all features
        recommendations = pd.concat([
            df[['contractID', 'date', 'last', 'strike', 'type', 'days_to_exp', 'implied_volatility', 'delta', 'vega']],
            cqf_quantiles,
            risk_features,
            ranking_scores
        ], axis=1)
        
        # Add regime context
        if 'vol_severity' in df.columns:
            recommendations['vol_severity'] = df['vol_severity']
        if 'vol_emergency' in df.columns:
            recommendations['vol_emergency'] = df['vol_emergency']
        
        # Filter recommendations by quality criteria
        quality_mask = (
            (recommendations['prob_profit'] > 0.45) &  # Reasonable profit probability
            (recommendations['uncertainty'] < recommendations['uncertainty'].quantile(0.8)) &  # Manageable uncertainty
            (recommendations['downside_risk'] < recommendations['expected_return'].abs() * 2)  # Risk-controlled
        )
        
        quality_recs = recommendations[quality_mask].copy()
        logger.info(f"Quality filtered: {len(quality_recs)} / {len(recommendations)} recommendations")
        
        if len(quality_recs) == 0:
            logger.warning("No recommendations passed quality filters")
            return recommendations.nlargest(top_n, 'ranking_score')
        
        # Select top recommendations by combined score
        # Combine ranking score with risk-adjusted utility
        quality_recs['combined_score'] = (
            0.6 * quality_recs['ranking_score'] + 
            0.4 * quality_recs['regime_adjusted_utility'].fillna(quality_recs['utility'])
        )
        
        final_recommendations = quality_recs.nlargest(top_n, 'combined_score')
        
        # Add recommendation metadata
        final_recommendations['recommendation_rank'] = range(1, len(final_recommendations) + 1)
        final_recommendations['timestamp'] = datetime.now()
        
        logger.info(f"Final recommendations: {len(final_recommendations)} top options selected")
        return final_recommendations
    
    def evaluate_system_performance(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate integrated system performance.
        """
        logger.info("Evaluating integrated system performance")
        
        # Calculate targets for evaluation
        df_with_targets = self.calculate_delta_hedged_pnl(df, self.horizon)
        valid_mask = df_with_targets['target_pnl'].notna()
        df_valid = df_with_targets[valid_mask].reset_index(drop=True)
        
        if len(df_valid) == 0:
            logger.error("No valid targets for evaluation")
            return {}
        
        # Get CQF predictions
        cqf_preds = self.predict_cqf_quantiles(df_valid, apply_conformal=True)
        
        # Calculate coverage metrics
        target = df_valid['target_pnl'].values
        q05, q95 = cqf_preds['q0.05'].values, cqf_preds['q0.95'].values
        
        coverage_90 = np.mean((target >= q05) & (target <= q95))
        width = np.mean(q95 - q05)
        
        # Get ranking performance
        ranking_scores = self.generate_ranking_scores(df_valid)
        
        # Generate recommendations for evaluation
        recommendations = self.generate_trading_recommendations(df_valid, top_n=100)
        
        metrics = {
            'cqf_coverage_90': coverage_90 * 100,
            'cqf_interval_width': width,
            'total_recommendations': len(recommendations),
            'high_prob_profit_pct': (recommendations['prob_profit'] > 0.6).mean() * 100,
            'avg_expected_return': recommendations['expected_return'].mean(),
            'avg_ranking_score': ranking_scores.mean(),
            'regime_stress_detected': (df_valid['vol_emergency'].sum() > 0) if 'vol_emergency' in df_valid.columns else False
        }
        
        logger.info("System Performance Metrics:")
        for key, value in metrics.items():
            if isinstance(value, bool):
                logger.info(f"  {key}: {value}")
            else:
                logger.info(f"  {key}: {value:.2f}")
        
        return metrics

def main():
    """Main execution pipeline for integrated trading system."""
    parser = argparse.ArgumentParser(description="Integrated Options Trading System")
    parser.add_argument("--eval-data", required=True, help="Evaluation data CSV file")
    parser.add_argument("--cqf-model", default="model_output/cqf_step7_black_swan_test.joblib", help="CQF model path")
    parser.add_argument("--ranking-model", default="model_output/xgboost_ranker_2019_2023_fixed_params_20250901_163328.joblib", help="Ranking model path")
    parser.add_argument("--ranking-features", default="model_output/xgb_feature_names_2019_2023_20250901_163328.pkl", help="Ranking features path")
    parser.add_argument("--config", default="config.yaml", help="Configuration file")
    parser.add_argument("--output", default="trading_recommendations.csv", help="Output recommendations file")
    parser.add_argument("--top-n", type=int, default=50, help="Number of top recommendations")
    
    args = parser.parse_args()
    
    try:
        # Initialize integrated trading system
        logger.info("=== Initializing Integrated Trading System ===")
        trading_system = IntegratedTradingSystem(
            cqf_model_path=args.cqf_model,
            ranking_model_path=args.ranking_model,
            ranking_features_path=args.ranking_features,
            config_path=args.config
        )
        
        # Load and preprocess evaluation data
        logger.info("=== Loading and Preprocessing Data ===")
        eval_data = trading_system.preprocess_data(args.eval_data)
        
        # Evaluate system performance
        logger.info("=== Evaluating System Performance ===")
        performance_metrics = trading_system.evaluate_system_performance(eval_data)
        
        # Generate trading recommendations
        logger.info("=== Generating Trading Recommendations ===")
        recommendations = trading_system.generate_trading_recommendations(eval_data, top_n=args.top_n)
        
        # Save results
        recommendations.to_csv(args.output, index=False)
        logger.info(f"✅ Trading recommendations saved to: {args.output}")
        
        # Save performance report
        performance_report = {
            'timestamp': datetime.now().isoformat(),
            'eval_data': args.eval_data,
            'models': {
                'cqf_model': args.cqf_model,
                'ranking_model': args.ranking_model
            },
            'performance_metrics': performance_metrics,
            'recommendation_count': len(recommendations)
        }
        
        report_path = args.output.replace('.csv', '_performance_report.json')
        import json
        with open(report_path, 'w') as f:
            json.dump(performance_report, f, indent=2, default=str)
        
        logger.info(f"✅ Performance report saved to: {report_path}")
        
        # Summary
        logger.info("=== Integration Complete ===")
        logger.info(f"✅ CQF Coverage: {performance_metrics.get('cqf_coverage_90', 0):.1f}%")
        logger.info(f"✅ Recommendations: {len(recommendations)} top options")
        logger.info(f"✅ Avg Expected Return: {performance_metrics.get('avg_expected_return', 0):.4f}")
        logger.info(f"✅ High Probability Trades: {performance_metrics.get('high_prob_profit_pct', 0):.1f}%")
        
        return 0
        
    except Exception as e:
        logger.error(f"Integration failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
