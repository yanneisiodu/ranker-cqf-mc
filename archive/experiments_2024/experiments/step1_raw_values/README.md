# Step 1: Raw Values Fix

**Objective**: Ensure all return-like features and targets are computed on raw values, not scaled values.

## Problem

The original pipeline had a critical issue where:
1. `utils.preprocess_data()` scales price and IV columns using `RobustScaler`
2. Feature engineering functions (`price_change_1d`, `iv_change_1d`) computed returns on these scaled values
3. Target calculation (`target_5d_sharpe`) computed Sharpe ratios on scaled prices
4. This distorted return magnitudes and degraded model generalization

## Solution

### Changes Made

1. **utils.py `_scale_data()` function**:
   - Added preservation of raw columns before scaling
   - Creates `last_raw` and `implied_volatility_raw` copies before any transformation
   - These columns bypass all scaling operations

2. **utils.py `_engineer_features()` function**:
   - Updated to use `last_raw` (if available) instead of `last` for `price_change_1d`
   - Updated to use `implied_volatility_raw` (if available) instead of `implied_volatility` for `iv_change_1d`
   - Maintains backward compatibility by falling back to scaled columns if raw columns not found

3. **train_xgboost_ranking_model.py `calculate_target()` function**:
   - Updated to use `last_raw` (if available) instead of `last` for target calculation
   - All Sharpe ratio computations now use unscaled price data
   - Maintains backward compatibility

4. **train_xgboost_ranking_model.py Step 3.5**:
   - Updated basic feature engineering to use raw columns
   - Added logging to show which columns are being used

5. **config.yaml**:
   - Added explanatory comments about why `last` and `implied_volatility` remain in `numerical_cols_to_scale`
   - The scaled versions are preserved for other models that may benefit from normalization

## Files

- `utils.py` - Modified preprocessing pipeline with raw value preservation
- `train_xgboost_ranking_model.py` - Modified training script to use raw values
- `config.yaml` - Added explanatory comments
- `validate_raw_returns.py` - Validation script demonstrating the fix
- `logger.py` - Dependency copied for standalone operation

## Validation

Run the validation script to verify the fix:

```bash
cd experiments/step1_raw_values
python validate_raw_returns.py
```

**Expected output**:
- ✓ Raw columns preserved during preprocessing 
- ✓ Feature engineering uses raw columns when available
- ✓ Target calculation uses raw columns when available
- ✓ Demonstrates meaningful difference between scaled and raw returns

## Testing

To test the full pipeline with the raw values fix:

```bash
# Smoke test: Train on 2022 only with no Optuna trials
cd experiments/step1_raw_values
python train_xgboost_ranking_model.py --start-year 2022 --end-year 2022 --trials 0
```

This should run end-to-end without breakages and produce a model trained on properly computed returns and targets.

## Backward Compatibility

- All existing feature and target column names are preserved
- Pipeline gracefully falls back to scaled columns if raw columns are not available
- No breaking changes to downstream evaluation or inference code

## Next Steps

After validation, this fix addresses the [CRITICAL] item #1 from MODEL_FIXES.md and enables moving to the next critical fix (eliminating tsfresh lookahead leakage).
