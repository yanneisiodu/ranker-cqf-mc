# Step 3: Per-Day Realized Forward Returns Labeling

## Status: REJECTED ❌

### What This Step Fixes

This experiment addresses **[CRITICAL] Fix #8** from MODEL_FIXES.md:
- **Problem**: Sharpe-based global quantile labels use dataset-wide statistics that may leak future information
- **Solution**: Replace with per-day rank labels derived from realized forward returns over h=3 days

### Key Changes Made

1. **Replaced Sharpe-Based Target System**
   - Removed global quantile-based target calculation (`target_5d_sharpe`)
   - Removed quantile edges saving/loading system
   - Replaced with per-day realized forward return ranking

2. **Implemented Per-Day Realized Returns Labeling**
   - New `make_daily_rank_labels()` function in `utils.py`
   - Computes forward returns over h=3 days using shift(-h) on raw prices
   - Creates per-day percentile rankings: [0.0, 0.50, 0.75, 0.90, 0.97, 1.0]
   - Bins into ordinal labels [0, 1, 2, 3, 4] for ranking objective

3. **Enhanced Coverage and Data Quality**
   - **Coverage gates**: Drop rows without valid forward returns
   - **Day filtering**: Remove single-row days before label creation
   - **Causality enforcement**: Only `shift(-h)` used for target, features remain causal
   - **Group integrity**: Assert group sizes match data after filtering

4. **Removed Dataset-Wide Scaling**
   - Set `numerical_cols_to_scale: []` to prevent leakage
   - Preprocessor uses only SimpleImputer, no StandardScaler
   - Maintains raw value computations for target creation

### Files Modified

- `utils.py`: 
  - Added `make_daily_rank_labels()` function for per-day ranking
  - Maintains all causal rolling features from Step 2
- `train_xgboost_ranking_model.py`:
  - Removed Sharpe-based target calculation and quantile binning
  - Replaced with `make_daily_rank_labels()` call after preprocessing
  - Updated preprocessor to use only SimpleImputer (no scaling)
  - Added group integrity assertions
- `evaluate_model.py`:
  - Replaced Sharpe-based evaluation targets with realized return labels
  - Removed sharpe-edges-file argument requirement
  - Uses same `make_daily_rank_labels()` for consistent evaluation
- `config.yaml`: 
  - Added `label_horizon_days: 3` and `rank_label_bins` parameters
  - Set `numerical_cols_to_scale: []` to disable dataset-wide scaling

### Validation

The experiment includes validation checks:
- ✅ **Causality**: `shift(-h)` only used for target; features use `shift(1)` and past-only windows
- ✅ **Coverage**: Log overall coverage (non-NaN target fraction) and days retained after min 2 rows
- ✅ **Group integrity**: Assert per-day group sizes match training data rows; assert data sorted by date
- ✅ **Raw values**: Ensure `last_raw` exists and is used for target computation
- ✅ **No leakage**: No dataset-wide scaling; SimpleImputer only in preprocessor

### Usage

```bash
# Train model with realized return labels (smoke test)
python train_xgboost_ranking_model.py --start-year 2022 --end-year 2022 --trials 0

# Evaluate on out-of-sample data (2023)
python evaluate_model.py --model-file model_output/xgboost_ranker_2022_2022_fixed_params_*.joblib --eval-data-file ../../year_2023_data.csv --config-file config.yaml --feature-list-file model_output/xgb_feature_names_*.pkl

# Full optimization
python train_xgboost_ranking_model.py --start-year 2022 --end-year 2022 --trials 10
```

### Impact vs Step 2

✅ **Eliminated**: Global quantile-based target leakage  
✅ **Replaced**: Sharpe ratio targets with per-day realized return rankings  
✅ **Maintained**: All causal rolling features from Step 2  
✅ **Improved**: True per-day ranking without cross-temporal information leakage  

### Final Decision Gate Results ❌

**Step2 Baseline (2023):**
- NDCG@20: 0.8317, Precision@5: 0.9637, Precision@10: 0.9444

**Step3 Results (2023):**  
- NDCG@20: 0.3174 (-0.5143), Precision@5: 0.3733 (-0.5904), Precision@10: 0.3583 (-0.5861)

**Step3 Results (2025):**
- NDCG@20: 0.3604 (-0.4713), Precision@5: 0.4585 (-0.5052), Precision@10: 0.4537 (-0.4907)

**Decision Gate Criteria:** Accept if NDCG@20 improves by ≥+0.05 on both datasets OR ≥+0.03 with precision improvements ≥+0.03  
**Result:** **FAILED** - All metrics show massive decreases (-0.47 to -0.59) instead of improvements

### Conclusion

Despite comprehensive optimization efforts including:
✅ Early stopping with recent-day validation  
✅ Manual grid search over 48 hyperparameter combinations  
✅ Midprice preference (99.9% bid/ask coverage)  
✅ Configurable binning strategies  

The per-day realized forward returns labeling approach **fundamentally underperforms** the Sharpe-based global quantile approach from step2. The ~50-60% performance degradation indicates this is not a hyperparameter or implementation issue, but a conceptual limitation of the approach.

### Target Labeling Process

1. **Forward Return Computation**: `(price_t+h - price_t) / price_t` using `last_raw`
2. **Coverage Filtering**: Drop rows without valid h-day forward returns
3. **Day Filtering**: Remove days with <2 contracts (can't rank single items)
4. **Per-Day Ranking**: Rank within each day using percentile bins [0.0, 0.50, 0.75, 0.90, 0.97, 1.0]
5. **Ordinal Labels**: Map to [0, 1, 2, 3, 4] for XGBoost ranking objective

### Dependencies

This step builds on Step 2 (causal rolling features) and Step 1 (raw value computations):
- Raw value preservation (`last_raw` column for target computation)
- Causal rolling features with no future leakage
- No dataset-wide scaling to prevent cross-temporal contamination