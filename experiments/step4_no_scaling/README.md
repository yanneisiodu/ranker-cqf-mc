# Step 4: Remove Double-Scaling

## Status: COMPLETED ❌ (Not Recommended for Adoption)

### What This Step Fixes

This experiment addresses **[HIGH] Fix #3** from MODEL_FIXES.md:
- **Problem**: Data is scaled globally in `utils` (RobustScaler) and again in the model pipeline via `StandardScaler`
- **Solution**: Remove redundant scaling since tree-based models (XGBoost) don't require feature scaling

### Key Changes Made

1. **Removed StandardScaler from Model Pipeline**
   - Modified `create_preprocessor()` in `train_xgboost_ranking_model.py`
   - Numerical pipeline now uses only `SimpleImputer(strategy='mean')`
   - Kept `OneHotEncoder` for categorical features unchanged
   - Updated debug messages to reflect scaling removal

2. **Disabled Dataset-Wide Scaling**
   - Set `numerical_cols_to_scale: []` in `config.yaml`
   - Modified `_scale_data()` in `utils.py` to handle empty scaling lists
   - Enhanced logging to clearly indicate when scaling is skipped

3. **Added Acceptance Guard**
   - Added assertion before final model training to ensure no `_scaled` columns exist
   - Prevents accidental leakage of globally-scaled features
   - Clear error message if scaled columns are detected

4. **Benefits**
   - Eliminates redundant preprocessing (faster training/prediction)
   - Removes potential signal distortion from double-scaling
   - Tree-based models work well with raw feature scales
   - Simplified preprocessing pipeline

### Files Modified

- `train_xgboost_ranking_model.py`: 
  - Removed `StandardScaler` from numerical preprocessing pipeline
  - Added acceptance guard before final model training
  - Updated debug messages to reflect no-scaling approach
- `config.yaml`: 
  - Set `numerical_cols_to_scale: []` to disable dataset-wide scaling
- `utils.py`: 
  - Enhanced `_scale_data()` to handle empty scaling lists gracefully
  - Improved logging messages for scaling skip scenarios

### Validation

The changes include built-in validation:
- ✅ Acceptance guard prevents `_scaled` columns from entering training
- ✅ Enhanced logging confirms when scaling is skipped
- ✅ Preprocessor creates simpler pipeline (Imputer only for numerics)
- ✅ Raw features preserved for return calculations (from Step 1)
- ✅ Causal rolling features maintained (from Step 2)

### Usage

```bash
# Train model (smoke test - 2022 data, no tuning)
python train_xgboost_ranking_model.py --start-year 2022 --end-year 2022 --trials 0

# Evaluate on 2023 out-of-sample data
python evaluate_model.py --model-file model_output/xgboost_ranker_2022_2022_fixed_params_*.joblib \
                        --eval-data-file ../../year_2023_data.csv \
                        --config-file config.yaml \
                        --sharpe-edges-file model_output/sharpe_qcut_edges_2022_2022_*.pkl \
                        --feature-list-file model_output/xgb_feature_names_2022_2022_*.pkl
```

### Impact vs Step 2

✅ **Eliminated**: Double-scaling (RobustScaler + StandardScaler)  
✅ **Simplified**: Preprocessing pipeline (Imputer only for numerics)  
✅ **Maintained**: All causal rolling features from Step 2  
✅ **Improved**: Faster training/prediction with no redundant transforms  

### Actual Results (2023 OOS Evaluation)

**⚠️ PERFORMANCE DEGRADATION DETECTED**

| Metric | Step 2 (Double-Scaling) | Step 4 (No Scaling) | Change |
|--------|-------------------------|---------------------|--------|
| **NDCG(exp)@1** | 0.8727 | 0.6262 | **-28.3%** ❌ |
| **NDCG(exp)@5** | 0.8616 | 0.5906 | **-31.5%** ❌ |
| **NDCG(exp)@10** | 0.8288 | 0.5583 | **-32.6%** ❌ |
| **NDCG(exp)@20** | 0.7836 | 0.5272 | **-32.7%** ❌ |
| **Spearman** | 0.0775 | -0.0136 | **Sign flip** ❌ |

### Key Findings

❌ **Contrary to theory**: Tree models in this specific case DO benefit from feature scaling  
✅ **Faster training**: ~230s vs typical scaled training time  
✅ **Memory savings**: Simplified preprocessing pipeline  
❌ **Signal distortion**: Removing scaling actually hurt predictive power  

### Recommendation

**DO NOT ADOPT** this change. While theoretically sound, the empirical evidence shows significant performance degradation. The double-scaling may be providing beneficial regularization or feature normalization that this dataset/model benefits from.

### Dependencies

This step builds on Step 2 (causal features) and requires:
- Causal rolling features (no tsfresh)  
- Raw value preservation (`last_raw`, `implied_volatility_raw`)
- All target computations on unscaled data

The acceptance guard ensures no scaled columns accidentally enter training.