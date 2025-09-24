#!/usr/bin/env python3
"""
Test script to verify the XGBoost model functionality
"""

import joblib
import pandas as pd
import numpy as np
import sys

def test_xgboost_model():
    print("🔍 Testing XGBoost Model Functionality")
    print("=" * 50)
    
    try:
        # Load the XGBoost model (use the latest trained model)
        print("Loading XGBoost model...")
        model = joblib.load('model_output/xgboost_ranker_2022_2024_optuna_tuned_20250829_182605.joblib')
        print(f"✅ Model loaded successfully")
        print(f"Model type: {type(model)}")
        print(f"Pipeline steps: {model.steps}")
        
        # Load features
        features = joblib.load('model_output/xgb_feature_names_2022_2024_20250829_182605.pkl')
        print(f"✅ Features loaded: {len(features)} features")
        
        # Create varied test data to check if model produces different outputs
        print("\nCreating varied test data...")
        
        test_cases = [
            # Case 1: High IV, short expiry, ITM put
            {
                'days_to_exp': 7.0,
                'strike': 460.0,
                'last': 12.5,
                'bid': 12.3,
                'ask': 12.7,
                'volume': 500,
                'open_interest': 2000,
                'implied_volatility': 0.45,  # High IV
                'delta': -0.85,  # Deep ITM
                'gamma': 0.001,
                'theta': -0.15,
                'vega': 0.05,
                'rho': -0.2,
                'spy_d_close': 450.0,
                'vix_d_close': 35.0,  # High VIX
                'moneyness': 1.02,
                'relative_spread': 0.015,
                'bid_ask_spread': 0.4,
                'ofi': 1.5,
                'price_change_1d': 0.25,
                'iv_change_1d': 0.05,
                'zero_day_premium': 0.0,
                'option_volume_oi_ratio': 0.25,
                'mispricing_ratio': 1.1,
                'risk_adjusted_signal': 2.5,
                'iv_vix_ratio': 1.3,
                'spy_momentum': -0.02,
                'price__mean': 10.0,
                'price__standard_deviation': 2.0,
                'type': 'P'
            },
            # Case 2: Low IV, long expiry, OTM put
            {
                'days_to_exp': 90.0,
                'strike': 430.0,
                'last': 2.1,
                'bid': 2.0,
                'ask': 2.2,
                'volume': 50,
                'open_interest': 800,
                'implied_volatility': 0.15,  # Low IV
                'delta': -0.15,  # OTM
                'gamma': 0.008,
                'theta': -0.01,
                'vega': 0.25,
                'rho': -0.08,
                'spy_d_close': 450.0,
                'vix_d_close': 12.0,  # Low VIX
                'moneyness': 0.96,
                'relative_spread': 0.05,
                'bid_ask_spread': 0.2,
                'ofi': -0.5,
                'price_change_1d': -0.1,
                'iv_change_1d': -0.02,
                'zero_day_premium': 0.0,
                'option_volume_oi_ratio': 0.06,
                'mispricing_ratio': 0.95,
                'risk_adjusted_signal': -1.0,
                'iv_vix_ratio': 1.25,
                'spy_momentum': 0.01,
                'price__mean': 2.5,
                'price__standard_deviation': 0.3,
                'type': 'P'
            },
            # Case 3: Medium case
            {
                'days_to_exp': 30.0,
                'strike': 450.0,
                'last': 5.5,
                'bid': 5.4,
                'ask': 5.6,
                'volume': 200,
                'open_interest': 1200,
                'implied_volatility': 0.25,
                'delta': -0.5,
                'gamma': 0.01,
                'theta': -0.05,
                'vega': 0.15,
                'rho': -0.1,
                'spy_d_close': 450.0,
                'vix_d_close': 20.0,
                'moneyness': 1.0,
                'relative_spread': 0.02,
                'bid_ask_spread': 0.2,
                'ofi': 0.0,
                'price_change_1d': 0.0,
                'iv_change_1d': 0.0,
                'zero_day_premium': 0.0,
                'option_volume_oi_ratio': 0.17,
                'mispricing_ratio': 1.0,
                'risk_adjusted_signal': 0.0,
                'iv_vix_ratio': 1.25,
                'spy_momentum': 0.0,
                'price__mean': 5.0,
                'price__standard_deviation': 0.8,
                'type': 'P'
            }
        ]
        
        test_df = pd.DataFrame(test_cases)
        print(f"✅ Test data created with shape: {test_df.shape}")
        print("\nTest case characteristics:")
        print("Case 1: High IV, short expiry, deep ITM")
        print("Case 2: Low IV, long expiry, OTM") 
        print("Case 3: Medium case baseline")
        
        # Test prediction
        print("\n🧪 Testing predictions...")
        predictions = model.predict(test_df)
        print(f"✅ Predictions successful! Shape: {predictions.shape}")
        print(f"Predictions: {predictions}")
        print(f"Prediction stats:")
        print(f"  Min: {predictions.min():.6f}")
        print(f"  Max: {predictions.max():.6f}")
        print(f"  Mean: {predictions.mean():.6f}")
        print(f"  Std: {predictions.std():.6f}")
        
        # Check for variation
        unique_predictions = len(set(np.round(predictions, 6)))
        print(f"\nUnique predictions (rounded to 6 decimals): {unique_predictions}")
        
        if unique_predictions == 1:
            print("🚨 PROBLEM: All predictions are identical!")
            print("   This suggests the model is not working properly.")
        elif unique_predictions == len(predictions):
            print("✅ GOOD: All predictions are different")
            print("   Model is producing varied outputs for different inputs.")
        else:
            print(f"⚠️  PARTIAL: {unique_predictions}/{len(predictions)} unique predictions")
            print("   Some variation but not fully distinct.")
        
        # Test with extreme cases
        print("\n🧪 Testing with extreme cases...")
        extreme_cases = [
            # Extreme case 1: Very high IV, very short expiry
            dict(test_cases[0], **{
                'days_to_exp': 1.0,
                'implied_volatility': 0.80,
                'vix_d_close': 50.0,
                'delta': -0.95
            }),
            # Extreme case 2: Very low IV, very long expiry
            dict(test_cases[1], **{
                'days_to_exp': 365.0,
                'implied_volatility': 0.08,
                'vix_d_close': 8.0,
                'delta': -0.05
            })
        ]
        
        extreme_df = pd.DataFrame(extreme_cases)
        extreme_predictions = model.predict(extreme_df)
        print(f"Extreme predictions: {extreme_predictions}")
        
        # Compare with original predictions
        all_predictions = np.concatenate([predictions, extreme_predictions])
        total_unique = len(set(np.round(all_predictions, 6)))
        print(f"\nTotal unique predictions across all tests: {total_unique}/{len(all_predictions)}")
        
        if total_unique <= 2:
            print("🚨 CRITICAL: Model shows very limited variation")
            print("   Likely returning constant or near-constant values")
        else:
            print(f"✅ Model shows reasonable variation: {total_unique} distinct outputs")
            
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_xgboost_model()
    sys.exit(0 if success else 1) 