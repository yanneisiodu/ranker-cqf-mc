# Step 2: Causal Rolling Features

## Status: COMPLETED ✅

### What This Step Fixes

This experiment addresses **[CRITICAL] Fix #2** from MODEL_FIXES.md:
- **Problem**: tsfresh features introduce lookahead bias by computing entity-level features over the full time series
- **Solution**: Replace with causal rolling features computed strictly from past data

### Key Changes Made

1. **Removed tsfresh Dependencies**
   - Disabled tsfresh imports and functionality in `utils.py`
   - Removed `_add_tsfresh_features()` function
   - Updated config to set `use_tsfresh: false`

2. **Added Causal Rolling Features**
   - New `_add_causal_rolling_features()` function computes rolling statistics
   - Features computed per-contractID using only past data (no future leakage)
   - Proper sorting by ['contractID', 'date'] ensures causal ordering

3. **New Rolling Features Added**
   - **Price features** (using `last_raw`):
     - `price_roll_mean_{5,20}`: Rolling mean
     - `price_roll_std_{5,20}`: Rolling standard deviation  
     - `price_roll_min_{5,20}`: Rolling minimum
     - `price_roll_max_{5,20}`: Rolling maximum
     - `price_roll_zscore_{5,20}`: Z-score (current - mean) / (std + ε)
   - **Implied Volatility features** (using `implied_volatility_raw`):
     - `iv_roll_mean_{5,20}`: Rolling mean of IV
     - `iv_roll_std_{5,20}`: Rolling standard deviation of IV
   - **Volume/OI features**:
     - `vol_roll_mean_{5,20}`: Rolling mean of volume
     - `oi_roll_mean_{5,20}`: Rolling mean of open interest
     - `vol_oi_ratio`: volume / (open_interest + ε)

4. **Rolling Window Configuration**
   - Windows: 5 and 20 periods
   - `min_periods`: Set to max(1, window//4) for partial window calculations
   - NaN handling: Fill with 0 for early rows where windows aren't full

### Files Modified

- `utils.py`: 
  - Replaced `_add_tsfresh_features()` with `_add_causal_rolling_features()`
  - Updated preprocessing pipeline to use causal features
  - Removed tsfresh imports
- `config.yaml`: 
  - Set `use_tsfresh: false`, `use_causal_rolling: true`
  - Added new rolling features to `numerical_cols_to_scale`
  - Documented all new features with comments

### Validation

The experiment includes `validate_causal_features.py` which:
- ✅ Confirms no tsfresh columns (e.g., `price__mean`) are present
- ✅ Validates that rolling features exist and follow expected naming
- ✅ Tests causal behavior by manual spot-checking rolling calculations
- ✅ Confirms `last_raw` preservation from Step 1
- ✅ Ensures pipeline integrity (no NaN explosions, reasonable feature counts)

### Usage

```bash
# Run validation first
python validate_causal_features.py

# Train model (smoke test)
python train_xgboost_ranking_model.py --start_year 2022 --end_year 2022 --trials 0

# Full evaluation on out-of-sample data  
python train_xgboost_ranking_model.py --start_year 2022 --end_year 2022 --trials 10
# Then evaluate on 2025 data
```

### Impact vs Step 1

✅ **Eliminated**: Lookahead bias from tsfresh features  
✅ **Added**: 19+ new causal rolling features  
✅ **Maintained**: Raw value computations from Step 1  
✅ **Improved**: True causal feature computation with no future information leakage  

### Expected Behavior

- **Metrics may decrease initially** due to removal of lookahead information
- **Out-of-sample performance should improve** due to elimination of overfitting
- **Model generalization should be more robust** in true forward-testing scenarios

This critical fix ensures that all features respect the temporal ordering constraint, preventing the model from "cheating" with future information during training.

### Dependencies

This step builds on Step 1 (raw value computations) and requires:
- Raw value preservation (`last_raw`, `implied_volatility_raw`)
- All feature engineering using raw values
- Target computation on unscaled data

The pipeline is backward compatible but `use_tsfresh` must be set to `false` to avoid conflicts.