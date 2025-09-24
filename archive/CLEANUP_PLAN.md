# Codebase Cleanup Plan

## 🗂️ Current State Analysis
- **Total directories**: ~50+
- **Redundant artifacts**: Multiple model versions, duplicate results
- **Active development**: Scattered across experiments/, Training/, and root
- **Archive candidates**: Old experiments, duplicate models, logs

## 🎯 Cleanup Strategy

### 1. IMMEDIATE DELETIONS (Safe to Remove)
```bash
# Remove duplicate test evaluation directories
rm -rf 2024_backtest/eval_cql_robust_test/
rm -rf 2024_backtest/eval_cql_robust_test2/

# Remove training logs (can be regenerated)
rm -rf d3rlpy_logs/
rm -f iql_training_2023.log

# Remove old inference outputs (superseded by 2024_backtest)
rm -rf cql_2023_inference/
rm -rf inference_output_june2023_llm_only/

# Remove duplicate policy artifacts (keep _fixed versions)
rm -f final_iql_training_2023/discrete_cql_policy.d3
rm -f final_iql_training_2023/policy_meta.json
rm -rf iql_cql_artifacts/  # Duplicate of final_iql_training_2023/
```

### 2. ARCHIVE EXPERIMENTAL WORK
```bash
# Create archive directory
mkdir -p archive/

# Move completed experiments
mv experiments/ archive/experiments_2024/
mv papers/ archive/research_papers/

# Move old model artifacts
mkdir -p archive/legacy_models/
mv model_output/sharpe_qcut_edges_2021_2023_* archive/legacy_models/
mv model_output/xgb_feature_names_2021_2023_* archive/legacy_models/
mv model_output/xgboost_ranker2_2021_2023_* archive/legacy_models/
# Keep only 2022_2022 production models and optimal_cqf_step8.joblib
```

### 3. CONSOLIDATE OPTIMIZATION ARTIFACTS
```bash
# Move optimization experiments to archive
mv optimized_behavior_policy_fixed/ archive/
mv optuna_100_trials_fixed/ archive/
mv improve_behavior_policy.py archive/
mv optimize_behavior_policy.py archive/
mv train_improved_cql.py archive/
mv optuna_tune_cqf.py archive/
```

### 4. ORGANIZE PRODUCTION STRUCTURE
```
📁 Production Structure (After Cleanup):
├── 🎯 Training/                 # Core training pipeline
│   ├── iql_pipeline.py         # Main training script
│   ├── evaluate_cql_policy.py  # Robust evaluation
│   ├── prod_*.py               # Production modules
│   └── utils.py                # Shared utilities
├── 📊 2024_backtest/           # Latest results
│   ├── *.md                    # Analysis docs
│   ├── decision_table.*        # 2024 data
│   └── eval_cql/              # Final evaluation
├── 🤖 final_iql_training_2023/ # Production models
│   ├── *_fixed.*              # Working artifacts
│   └── inference_outputs/     # Model outputs
├── 📦 model_output/            # Production models only
│   ├── optimal_cqf_step8.joblib
│   ├── *_2022_2022_*_185021.*  # Core ranker model
│   └── (remove older versions)
├── 💾 data/                    # Raw data (new dir)
│   └── year_*.csv             # Move all year data here
├── 📚 archive/                 # Archived work
│   ├── experiments_2024/
│   ├── research_papers/
│   ├── legacy_models/
│   └── optimization_work/
└── 📋 docs/                   # Current documentation
```

## 🏆 Expected Savings
- **Directories**: ~50 → ~10 (80% reduction)
- **Disk Space**: ~2-3GB → ~500MB (75% reduction)
- **Files**: ~500+ → ~100 (80% reduction)

## ✅ Production-Ready Structure
After cleanup, the codebase will have:
- **Clear separation**: Production vs Archive vs Data
- **Single source of truth**: One working model per type
- **Maintainable**: Easy to find current vs historical work
- **Git-friendly**: Smaller repository size
- **Documented**: Clear purpose for each directory

## 🔧 Implementation Commands
Would you like me to execute this cleanup plan step by step?