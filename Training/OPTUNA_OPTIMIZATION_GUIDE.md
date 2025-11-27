# Optuna Hyperparameter Optimization Guide

This guide walks you through optimizing the probability classifier hyperparameters using Optuna to improve MACE (Mean Absolute Calibration Error) from 19% to ~12-15%.

---

## Overview

**Current Performance (v4)**:
- MACE: 19.0%
- Brier: 0.2817
- Bias: -12.7%

**Target After Optimization**:
- MACE: <15% (realistic), <12% (ambitious)
- Brier: <0.26
- Bias: <10%

---

## Step 1: Run Optuna Optimization

### Quick Run (100 trials, ~1 hour)

```bash
cd /Users/chinonsoisiodu/Documents/Projects/ranker_cqf_mc/Training
source /Users/chinonsoisiodu/Documents/global_python_env/bin/activate

python optimize_prob_classifier.py \
  --data ../Data/train_2021_2023.csv \
  --config config.yaml \
  --n-trials 100 \
  --timeout 3600 \
  --n-splits 3 \
  --output optuna_study_prob_classifier.db
```

### Thorough Run (200 trials, ~2 hours)

```bash
python optimize_prob_classifier.py \
  --n-trials 200 \
  --timeout 7200 \
  --n-splits 4
```

### What It Does

1. **Loads data** from the v4 predictions CSV (967K samples)
2. **Creates time-series CV splits** (3-4 folds)
3. **Optimizes hyperparameters**:
   - `n_estimators`: [50, 300]
   - `max_depth`: [2, 6]
   - `learning_rate`: [0.01, 0.3]
   - `min_child_weight`: [1, 20]
   - `subsample`: [0.6, 1.0]
   - `colsample_bytree`: [0.6, 1.0]
   - `reg_alpha`: [0.0, 5.0]
   - `reg_lambda`: [0.0, 5.0]
   - `gamma`: [0.0, 5.0]
4. **Evaluates** using MACE on held-out validation sets
5. **Saves** best parameters to `model_output/best_prob_classifier_params.json`

---

## Step 2: Review Results

```bash
python apply_optimized_params.py
```

This will show:
- Best MACE achieved
- Best hyperparameters
- Instructions to update `prod_cqf.py`

---

## Step 3: Update Configuration

Edit [prod_cqf.py](prod_cqf.py#L116-L125) and replace the probability classifier config with optimized values:

```python
# ===== Probability Classifier (OPTIMIZED by Optuna) =====
PROB_CLASSIFIER_N_ESTIMATORS = <best_value>
PROB_CLASSIFIER_MAX_DEPTH = <best_value>
PROB_CLASSIFIER_LEARNING_RATE = <best_value>
PROB_CLASSIFIER_MIN_CHILD_WEIGHT = <best_value>
PROB_CLASSIFIER_SUBSAMPLE = <best_value>
PROB_CLASSIFIER_COLSAMPLE = <best_value>
PROB_CLASSIFIER_REG_ALPHA = <best_value>
PROB_CLASSIFIER_REG_LAMBDA = <best_value>
PROB_CLASSIFIER_GAMMA = <best_value>
```

---

## Step 4: Retrain with Optimized Parameters

```bash
python prod_cqf.py \
  --train-data ../Data/train_2021_2023.csv \
  --eval-data ../Data/year_2024_data.csv \
  --config config.yaml \
  --output model_output/optimal_cqf_v6_optimized.joblib \
  --horizon 5
```

---

## Step 5: Validate Improvement

Compare v4 (baseline) vs v6 (optimized):

```bash
python -c "
import pandas as pd
import numpy as np

df_v4 = pd.read_csv('model_output/optimal_cqf_v4_classifier_predictions.csv')
df_v6 = pd.read_csv('model_output/optimal_cqf_v6_optimized_predictions.csv')

for name, df in [('v4 Baseline', df_v4), ('v6 Optimized', df_v6)]:
    actual = (df['target_actual'] > 0).astype(int)
    pred = df['prob_profit']

    brier = np.mean((pred - actual) ** 2)

    # MACE
    total_error = 0
    buckets = 0
    for i in range(10):
        lower, upper = i * 0.1, (i + 1) * 0.1
        mask = (pred >= lower) & (pred < upper)
        if mask.sum() > 100:
            expected = (lower + upper) / 2
            actual_wr = actual[mask].mean()
            total_error += abs(actual_wr - expected)
            buckets += 1

    mace = total_error / buckets
    bias = pred.mean() - actual.mean()

    print(f'{name:15} | MACE: {mace*100:5.1f}% | Brier: {brier:.4f} | Bias: {bias*100:+6.1f}%')
"
```

---

## Troubleshooting

### Issue: Optuna can't find data

**Solution**: Make sure you've run training at least once to generate the predictions CSV:

```bash
ls -lh model_output/optimal_cqf_v4_classifier_predictions.csv
```

If missing, run:
```bash
python prod_cqf.py --train-data ../Data/train_2021_2023.csv --eval-data ../Data/year_2024_data.csv --config config.yaml --output model_output/optimal_cqf_v4_classifier.joblib --horizon 5
```

### Issue: Optimization is too slow

**Solutions**:
1. Reduce `--n-trials` to 50
2. Reduce `--n-splits` to 2
3. Use `--timeout 1800` (30 minutes) for quick test

### Issue: MACE not improving

**Possible causes**:
1. **Inherent noise**: Delta-hedged P&L may have ~12% noise floor
2. **Feature limitations**: Current features may not capture all predictive signals
3. **Distribution shift**: 2024 test data has different characteristics than 2021-2023

**Next steps**:
1. Check feature importance: Which features are most predictive?
2. Add interaction features: `delta * moneyness`, `q0.05 * implied_vol`, etc.
3. Consider regime-specific models: Separate classifiers for high-VIX vs low-VIX

---

## Expected Outcomes

### Conservative (50-100 trials)
- **MACE**: 16-18% (small improvement)
- **Time**: 30-60 minutes
- **Value**: Quick validation that optimization helps

### Moderate (100-200 trials)
- **MACE**: 13-16% (good improvement)
- **Time**: 1-2 hours
- **Value**: Production-ready calibration

### Ambitious (300+ trials)
- **MACE**: 12-14% (best achievable)
- **Time**: 3-4 hours
- **Value**: Approaching noise floor

---

## File Outputs

1. **optuna_study_prob_classifier.db**: SQLite database with all trial results
2. **model_output/best_prob_classifier_params.json**: Best hyperparameters
3. **model_output/optimal_cqf_v6_optimized.joblib**: Retrained model
4. **model_output/optimal_cqf_v6_optimized_predictions.csv**: Predictions for analysis

---

## Advanced: Resume Interrupted Optimization

Optuna saves progress to the database, so you can resume:

```bash
# Run for 1 hour
python optimize_prob_classifier.py --n-trials 100 --timeout 3600

# Later, add 50 more trials
python optimize_prob_classifier.py --n-trials 50 --timeout 1800
```

The study will continue from where it left off.

---

## Advanced: Visualize Optimization

```python
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances

# Load study
study = optuna.load_study(
    study_name='prob_classifier_optimization',
    storage='sqlite:///optuna_study_prob_classifier.db'
)

# Plot optimization history
fig1 = plot_optimization_history(study)
fig1.write_html('optuna_history.html')

# Plot parameter importances
fig2 = plot_param_importances(study)
fig2.write_html('optuna_param_importance.html')

print(f"✅ Saved visualization to optuna_history.html and optuna_param_importance.html")
```

---

## Summary

**Do this**:
1. Run `optimize_prob_classifier.py` with 100 trials (~1 hour)
2. Review results with `apply_optimized_params.py`
3. Update `prod_cqf.py` config
4. Retrain model → `optimal_cqf_v6_optimized.joblib`
5. Compare v4 vs v6 performance

**Expected gain**: MACE from 19% → 13-15% (20-30% improvement)

**Realistic best**: MACE ~12% (delta-hedged P&L noise floor)
