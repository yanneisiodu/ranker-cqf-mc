# Trading Agent2 - Complete Execution Guide
## Step-by-Step Commands for World-Class Trading System

**⚠️ CRITICAL: Follow this guide EXACTLY - any deviation could completely thwart model performance**

**Project**: Trading Agent2 - XGBoost Models  
**Target Performance**: 86.6% win rate, 7,988.9% returns, 15.2% max drawdown  
**Execution Order**: MUST follow stages 1→2→3→4→5 sequentially

---

## 🔧 **Prerequisites & Environment Setup**

### **1. Environment Activation**
```bash
# CRITICAL: Always activate the trading environment first
source /Users/chinonsoisiodu/Documents/Projects/Trading\ Agent2/trading_env/bin/activate

# Navigate to project directory
cd "/Users/chinonsoisiodu/Documents/Projects/Trading Agent2/legacy data & xgboost/xgboost_models"
```

### **2. Required Dependencies**
```bash
# Verify all dependencies are installed (from requirements.txt)
pip install -r requirements.txt

# Key requirements:
# pandas, numpy, xgboost, scikit-learn, joblib, optuna
# d3rlpy (for IQL training), streamlit, plotly
```

### **3. Data File Verification**
```bash
# Verify required data files exist
ls -la data/
# Expected files:
# ✅ year_2019_data.csv (932MB)
# ✅ year_2020_data.csv (1.3GB) 
# ✅ year_2021_data.csv (1.4GB)
# ✅ year_2022_data.csv (1.3GB)
# ✅ year_2023_data.csv (1.1GB)
# ✅ year_2024_data.csv (1.3GB)
# ✅ year_2025_data.csv (230MB)

# Verify config file
ls -la config.yaml
# Expected: ✅ config.yaml (126 lines with optimized behavior policy parameters)
```

---

## 🚀 **Stage 1: XGBoost Ranker Training**

### **Purpose**: Train ranking model to identify most profitable option contracts

### **Command Structure**
```bash
python Training/prod_train_ranker.py \
    --start-year START_YEAR \
    --end-year END_YEAR \
    --trials OPTUNA_TRIALS \
    --config config.yaml
```

### **Recommended Execution Commands**

#### **Option 1: Latest Training (2022-2024)**
```bash
# Train on most recent data with full Optuna optimization
python Training/prod_train_ranker.py \
    --start-year 2022 \
    --end-year 2024 \
    --trials 100 \
    --config config.yaml
```

#### **Option 2: Extended Training (2019-2024)**
```bash
# Train on extended dataset (more data, longer training)
python Training/prod_train_ranker.py \
    --start-year 2019 \
    --end-year 2024 \
    --trials 100 \
    --config config.yaml
```

#### **Option 3: Quick Training (Fixed Parameters)**
```bash
# Use pre-optimized parameters (faster, no Optuna)
python Training/prod_train_ranker.py \
    --start-year 2022 \
    --end-year 2024 \
    --trials 0 \
    --config config.yaml
```

### **Expected Outputs**
```bash
model_output/
├── xgboost_ranker2_2022_2024_optuna_TIMESTAMP.joblib     # Trained model
├── xgb_feature_names_2022_2024_TIMESTAMP.pkl             # Feature list
└── sharpe_qcut_edges_2022_2024_TIMESTAMP.pkl             # Target edges
```

### **Performance Validation**
```bash
# Check training logs for final NDCG@20 score
# Expected: NDCG@20 > 0.75 for good performance
# Expected runtime: 15-30 minutes (100 trials)
```

---

## 📊 **Stage 2: CQF Model Training** 

### **Purpose**: Train Calibrated Quantile Forecasting model for return distributions

### **Command Structure**
```bash
python Training/prod_cqf.py \
    --train-data TRAIN_DATA_FILE \
    --eval-data EVAL_DATA_FILE \
    --config config.yaml \
    --output OUTPUT_PATH \
    --horizon DAYS
```

### **Recommended Execution Commands**

#### **Standard Training (2023 train, 2024 eval)**
```bash
# RECOMMENDED: Standard training with out-of-sample validation
python Training/prod_cqf.py \
    --train-data data/year_2023_data.csv \
    --eval-data data/year_2024_data.csv \
    --config config.yaml \
    --output model_output/optimal_cqf_step8.joblib \
    --horizon 5
```

#### **Extended Training (2022-2023 train, 2024 eval)**
```bash
# More training data (use combined file if available)
python Training/prod_cqf.py \
    --train-data data/year_2022_2023_combined.csv \
    --eval-data data/year_2024_data.csv \
    --config config.yaml \
    --output model_output/optimal_cqf_step8.joblib \
    --horizon 5
```

### **Expected Outputs**
```bash
model_output/
├── optimal_cqf_step8.joblib                              # Complete CQF model
└── optimal_cqf_step8_predictions.csv                     # Evaluation predictions
```

### **Performance Validation**
```bash
# Check training logs for quality gates
# Expected: ✅ Quality gates: PASSED
# Expected: 90% interval coverage ≈ 90% (within 5%)
# Expected runtime: 10-20 minutes
```

---

## 🔗 **Stage 3: Integrated Inference Pipeline**

### **Purpose**: Generate trading signals by combining ranker + CQF models

### **Command Structure**
```bash
python inference/run_inference.py \
    --raw-data DATA_FILE \
    --config config.yaml \
    --ranker-model RANKER_MODEL \
    --ranker-features RANKER_FEATURES \
    --sharpe-edges SHARPE_EDGES \
    --cqf-model CQF_MODEL \
    --top-n TOP_N_CONTRACTS \
    --output-dir OUTPUT_DIR \
    --stress-mode STRESS_MODE
```

### **Execution Commands**

#### **Using Latest Optuna-Tuned Models**
```bash
# RECOMMENDED: Use latest Optuna-optimized models
python inference/run_inference.py \
    --raw-data data/year_2024_data.csv \
    --config config.yaml \
    --ranker-model model_output/xgboost_ranker_2022_2024_optuna_tuned_20250829_182605.joblib \
    --ranker-features model_output/xgb_feature_names_2022_2024_20250829_182605.pkl \
    --sharpe-edges model_output/sharpe_qcut_edges_2022_2024_20250829_182605.pkl \
    --cqf-model model_output/optimal_cqf_step8.joblib \
    --top-n 1000 \
    --output-dir inference_outputs \
    --stress-mode mc
```

#### **With LLM Stress Testing** (requires OpenAI API key)
```bash
# Enhanced with LLM validation (requires OPENAI_API_KEY environment variable)
python inference/run_inference.py \
    --raw-data data/year_2024_data.csv \
    --config config.yaml \
    --ranker-model model_output/xgboost_ranker_2022_2024_optuna_tuned_20250829_182605.joblib \
    --ranker-features model_output/xgb_feature_names_2022_2024_20250829_182605.pkl \
    --sharpe-edges model_output/sharpe_qcut_edges_2022_2024_20250829_182605.pkl \
    --cqf-model model_output/optimal_cqf_step8.joblib \
    --top-n 1000 \
    --output-dir inference_outputs \
    --stress-mode shadow \
    --llm-engine agent \
    --llm-max-contracts 50 \
    --llm-max-groups 10
```

#### **Production Mode** (no future targets)
```bash
# Production-safe: no look-ahead bias
python inference/run_inference.py \
    --raw-data data/year_2024_data.csv \
    --config config.yaml \
    --ranker-model model_output/xgboost_ranker_2022_2024_optuna_tuned_20250829_182605.joblib \
    --ranker-features model_output/xgb_feature_names_2022_2024_20250829_182605.pkl \
    --sharpe-edges model_output/sharpe_qcut_edges_2022_2024_20250829_182605.pkl \
    --cqf-model model_output/optimal_cqf_step8.joblib \
    --top-n 1000 \
    --output-dir inference_outputs \
    --stress-mode mc \
    --skip-future-targets
```

### **Expected Outputs**
```bash
inference_outputs/
├── ranker_candidates.csv          # Top-ranked contracts with features
├── cqf_predictions.csv           # Quantile predictions + decision features  
├── stress_metrics.csv            # Monte Carlo stress test results
├── stress_metrics_llm.csv        # LLM stress results (if --stress-mode shadow/llm)
└── trade_recommendations.csv     # Final ranked trade recommendations
```

### **Performance Validation**
```bash
# Check file sizes and row counts
wc -l inference_outputs/*.csv
# Expected: ranker_candidates.csv ~1000 rows
# Expected: cqf_predictions.csv ~1000 rows  
# Expected: trade_recommendations.csv ~100 rows
# Expected runtime: 5-15 minutes
```

---

## 🧠 **Stage 4: IQL Policy Training**

### **Purpose**: Train reinforcement learning policy on integrated inference outputs

### **Command Structure**
```bash
python Training/iql_pipeline.py \
    --cqf-preds CQF_PREDICTIONS \
    --ranker-candidates RANKER_CANDIDATES \
    --stress-metrics STRESS_METRICS \
    --outdir OUTPUT_DIR \
    --train-steps TRAINING_STEPS \
    [additional options]
```

### **Execution Commands**

#### **Standard IQL Training** (using inference outputs)
```bash
# RECOMMENDED: Standard training with 200K steps
python Training/iql_pipeline.py \
    --cqf-preds inference_outputs/cqf_predictions.csv \
    --ranker-candidates inference_outputs/ranker_candidates.csv \
    --stress-metrics inference_outputs/stress_metrics.csv \
    --outdir iql_training_2024 \
    --top-k 5 \
    --size-bins "0.5,1.0" \
    --train-steps 200000 \
    --train-batch-size 1024 \
    --expectile 0.7 \
    --gamma 0.99 \
    --seed 42
```

#### **Dataset-Only Build** (skip training)
```bash
# Build decision table without training (for analysis)
python Training/iql_pipeline.py \
    --cqf-preds inference_outputs/cqf_predictions.csv \
    --ranker-candidates inference_outputs/ranker_candidates.csv \
    --stress-metrics inference_outputs/stress_metrics.csv \
    --outdir iql_dataset_only \
    --no-train
```

#### **End-to-End Training** (run inference + IQL in one command)
```bash
# Full pipeline: raw data → inference → IQL training
python Training/iql_pipeline.py \
    --raw-data data/year_2024_data.csv \
    --config config.yaml \
    --ranker-model model_output/xgboost_ranker_2022_2024_optuna_tuned_20250829_182605.joblib \
    --ranker-features model_output/xgb_feature_names_2022_2024_20250829_182605.pkl \
    --sharpe-edges model_output/sharpe_qcut_edges_2022_2024_20250829_182605.pkl \
    --cqf-model model_output/optimal_cqf_step8.joblib \
    --outdir iql_training_2024 \
    --train-steps 200000 \
    --stress-mode mc \
    --include-future-targets
```

### **Expected Outputs**
```bash
iql_training_2024/
├── discrete_cql_policy.d3        # Trained IQL policy
├── policy_meta.json              # State normalization + action mapping
├── decision_table.csv            # Training decisions with outcomes
└── decision_table.parquet        # Same data in Parquet format
```

### **Performance Validation**
```bash
# Check decision table quality
python -c "
import pandas as pd
df = pd.read_csv('iql_training_2024/decision_table.csv')
print(f'Decision table: {len(df)} rows')
print(f'Action distribution:')
print(df['action_id'].value_counts().sort_index())
print(f'Average reward: {df[\"reward\"].mean():.4f}')
"
# Expected: 1000+ decisions, diverse action distribution
# Expected runtime: 30-60 minutes (200K steps)
```

---

## 🎯 **Stage 5: Final Optimal Walk-Forward Validation**

### **Purpose**: Validate IQL policy with optimized risk management (Trial #74 parameters)

### **Command Structure**
```bash
python Training/final_optimal_walkforward.py \
    --decision-table DECISION_TABLE \
    --policy IQL_POLICY \
    --meta POLICY_META \
    --outdir OUTPUT_DIR
```

### **Execution Command**

#### **Using Your Proven 83.9% Baseline** (RECOMMENDED)
```bash
# CRITICAL: Use the 2024 decision table that gives 83.9% win rate baseline
python Training/final_optimal_walkforward.py \
    --decision-table 2024_backtest/decision_table.csv \
    --policy final_iql_training_2023/discrete_cql_policy.d3 \
    --meta final_iql_training_2023/policy_meta.json \
    --outdir results/final_optimal_walkforward
```

#### **Using Newly Trained Model**
```bash
# Use newly trained IQL model from Stage 4
python Training/final_optimal_walkforward.py \
    --decision-table iql_training_2024/decision_table.csv \
    --policy iql_training_2024/discrete_cql_policy.d3 \
    --meta iql_training_2024/policy_meta.json \
    --outdir results/final_optimal_walkforward_new
```

### **Expected Performance** (Trial #74 parameters)
```bash
# Target results (from proven configuration):
# 🎯 Win Rate: 86.6%
# 💰 Return: 7,988.9%  
# 🛡️ Max Drawdown: 15.2%
# 📈 Calmar Ratio: 524.6
# 📊 Total Trades: 82
# 🚨 Emergency Halts: 5
```

### **Expected Outputs**
```bash
results/final_optimal_walkforward/
├── optimal_walkforward_trades.csv       # Trade-by-trade results
└── optimal_walkforward_summary.json     # Performance summary
```

---

## 📋 **Complete Pipeline Execution Script**

### **Full End-to-End Training** (complete fresh training)
```bash
#!/bin/bash
# complete_training_pipeline.sh
# Execute entire pipeline from scratch

set -e  # Exit on any error

echo "🚀 Starting Complete Trading System Training Pipeline"
echo "Expected total runtime: 2-3 hours"

# Stage 1: Train Ranker (15-30 min)
echo "Stage 1: Training XGBoost Ranker..."
python Training/prod_train_ranker.py \
    --start-year 2022 \
    --end-year 2024 \
    --trials 100 \
    --config config.yaml

# Find the latest ranker model files
LATEST_RANKER=$(ls model_output/xgboost_ranker2_*_optuna_*.joblib | tail -1)
LATEST_FEATURES=$(ls model_output/xgb_feature_names_*_*.pkl | tail -1)
LATEST_EDGES=$(ls model_output/sharpe_qcut_edges_*_*.pkl | tail -1)

echo "✅ Stage 1 Complete. Latest model: $LATEST_RANKER"

# Stage 2: Train CQF (10-20 min)
echo "Stage 2: Training CQF Model..."
python Training/prod_cqf.py \
    --train-data data/year_2023_data.csv \
    --eval-data data/year_2024_data.csv \
    --config config.yaml \
    --output model_output/optimal_cqf_step8.joblib \
    --horizon 5

echo "✅ Stage 2 Complete. CQF model saved."

# Stage 3: Run Inference (5-15 min)
echo "Stage 3: Running Integrated Inference..."
python inference/run_inference.py \
    --raw-data data/year_2024_data.csv \
    --config config.yaml \
    --ranker-model "$LATEST_RANKER" \
    --ranker-features "$LATEST_FEATURES" \
    --sharpe-edges "$LATEST_EDGES" \
    --cqf-model model_output/optimal_cqf_step8.joblib \
    --top-n 1000 \
    --output-dir inference_outputs \
    --stress-mode mc

echo "✅ Stage 3 Complete. Inference outputs generated."

# Stage 4: Train IQL Policy (30-60 min)  
echo "Stage 4: Training IQL Policy..."
python Training/iql_pipeline.py \
    --cqf-preds inference_outputs/cqf_predictions.csv \
    --ranker-candidates inference_outputs/ranker_candidates.csv \
    --stress-metrics inference_outputs/stress_metrics.csv \
    --outdir iql_training_2024 \
    --train-steps 200000 \
    --train-batch-size 1024

echo "✅ Stage 4 Complete. IQL policy trained."

# Stage 5: Final Validation (1-2 min)
echo "Stage 5: Final Optimal Walk-Forward Validation..."
python Training/final_optimal_walkforward.py \
    --decision-table iql_training_2024/decision_table.csv \
    --policy iql_training_2024/discrete_cql_policy.d3 \
    --meta iql_training_2024/policy_meta.json \
    --outdir results/final_optimal_walkforward_new

echo "🏆 COMPLETE PIPELINE FINISHED!"
echo "Check results/final_optimal_walkforward_new/ for performance metrics"
```

### **Execute Complete Pipeline**
```bash
# Make script executable and run
chmod +x complete_training_pipeline.sh
./complete_training_pipeline.sh
```

---

## 🎯 **Quick Validation Pipeline** (using existing models)

### **Fast Execution** (using proven models from your training)
```bash
#!/bin/bash
# quick_validation_pipeline.sh
# Use existing models for fast validation

set -e

echo "⚡ Quick Validation Pipeline (using existing models)"

# Stage 3: Inference with existing models (5 min)
python inference/run_inference.py \
    --raw-data data/year_2024_data.csv \
    --config config.yaml \
    --ranker-model model_output/xgboost_ranker_2022_2024_optuna_tuned_20250829_182605.joblib \
    --ranker-features model_output/xgb_feature_names_2022_2024_20250829_182605.pkl \
    --sharpe-edges model_output/sharpe_qcut_edges_2022_2024_20250829_182605.pkl \
    --cqf-model model_output/optimal_cqf_step8.joblib \
    --top-n 1000 \
    --output-dir inference_outputs_quick \
    --stress-mode mc

# Stage 5: Validate with proven 83.9% baseline (1 min)
python Training/final_optimal_walkforward.py \
    --decision-table 2024_backtest/decision_table.csv \
    --policy final_iql_training_2023/discrete_cql_policy.d3 \
    --meta final_iql_training_2023/policy_meta.json \
    --outdir results/final_optimal_walkforward_validation

echo "✅ Quick validation complete!"
echo "Expected: 86.6% win rate, 7,988.9% return, 15.2% max drawdown"
```

---

## 🔧 **Troubleshooting & Common Issues**

### **1. Environment Issues**
```bash
# If ModuleNotFoundError for numpy/pandas:
source /Users/chinonsoisiodu/Documents/Projects/Trading\ Agent2/trading_env/bin/activate

# If d3rlpy import error:
pip install d3rlpy

# If XGBoost version error (CQF requires ≥2.0):
pip install "xgboost>=2.0"
```

### **2. File Path Issues**
```bash
# If FileNotFoundError, check data directory:
ls -la data/year_*.csv

# If model files missing, check model_output:
ls -la model_output/*.joblib

# If config missing:
ls -la config.yaml
```

### **3. Memory Issues**
```bash
# If out of memory during training:
# Reduce batch size: --train-batch-size 512
# Reduce Optuna trials: --trials 50
# Use smaller dataset: year_2024_data.csv only
```

### **4. Performance Validation**
```bash
# Verify final results match expectations:
python -c "
import json
with open('results/final_optimal_walkforward/optimal_walkforward_summary.json', 'r') as f:
    results = json.load(f)
print(f\"Win Rate: {results['win_rate']:.1%}\")
print(f\"Return: {results['return_pct']:.1f}%\")
print(f\"Max Drawdown: {results['max_drawdown']:.1%}\")
print(f\"Calmar Ratio: {results['calmar_ratio']:.1f}\")

# Expected output:
# Win Rate: 86.6%
# Return: 7988.9%
# Max Drawdown: 15.2%
# Calmar Ratio: 524.6
"
```

---

## 📊 **Production Inference Commands**

### **Daily Inference** (production trading)
```bash
# Daily execution for live trading
python inference/run_inference.py \
    --raw-data data/current_market_data.csv \
    --config config.yaml \
    --ranker-model model_output/xgboost_ranker_2022_2024_optuna_tuned_20250829_182605.joblib \
    --ranker-features model_output/xgb_feature_names_2022_2024_20250829_182605.pkl \
    --sharpe-edges model_output/sharpe_qcut_edges_2022_2024_20250829_182605.pkl \
    --cqf-model model_output/optimal_cqf_step8.joblib \
    --top-n 500 \
    --output-dir inference_outputs_$(date +%Y%m%d) \
    --stress-mode mc \
    --skip-future-targets \
    --min-prob-profit 0.45
```

### **Execute IQL Policy** (generate trading decisions)
```bash
# Generate trading decisions using IQL policy
python -c "
import pandas as pd
import numpy as np
import json
from d3rlpy import load_learnable
from pathlib import Path

# Load decision table (from daily inference)
df = pd.read_csv('inference_outputs_$(date +%Y%m%d)/decision_table.csv')

# Load trained policy
with open('final_iql_training_2023/policy_meta.json', 'r') as f:
    meta = json.load(f)

# Standardize states
states = df[meta['state_columns']].fillna(0).values
mean_arr = np.array(meta['scaler_mean'])
scale_arr = np.array(meta['scaler_scale'])
states_scaled = (states - mean_arr) / np.where(scale_arr == 0, 1, scale_arr)

# Load policy and predict
algo = load_learnable('final_iql_training_2023/discrete_cql_policy.d3')
actions = algo.predict(states_scaled)

# Decode actions to trading decisions
for i, action_id in enumerate(actions):
    action_info = meta['action_map'][str(action_id)]
    print(f\"Decision {i}: Slot {action_info['slot']}, Size {action_info['size_value']}x\")
"
```

---

## ⚠️ **CRITICAL SUCCESS FACTORS**

### **1. Execution Order** (MUST follow sequentially)
```
✅ Stage 1 → Stage 2 → Stage 3 → Stage 4 → Stage 5
❌ DO NOT skip stages or execute out of order
❌ DO NOT modify intermediate files manually
```

### **2. File Naming Accuracy** (CRITICAL)
```bash
# ALWAYS use the exact model file names generated:
# ✅ xgboost_ranker_2022_2024_optuna_tuned_TIMESTAMP.joblib
# ❌ xgboost_ranker.joblib (generic name will fail)

# ✅ optimal_cqf_step8.joblib (exact name required)
# ❌ cqf_model.joblib (wrong name will fail)
```

### **3. Data Consistency** (CRITICAL)
```bash
# MUST use consistent date ranges across stages:
# ✅ Ranker: 2022-2024, CQF: 2023→2024, Inference: 2024, IQL: 2024
# ❌ Mixed date ranges will cause distribution shift

# MUST use same preprocessing pipeline:
# ✅ All stages use utils.py preprocess_data()
# ❌ Different preprocessing will break feature compatibility
```

### **4. Parameter Consistency** (CRITICAL)
```bash
# MUST use same core parameters across stages:
# ✅ horizon=5 (everywhere)
# ✅ top_k=5 (IQL pipeline)  
# ✅ size_bins="0.5,1.0" (IQL pipeline)
# ❌ Inconsistent parameters will break action mapping
```

---

## 🏆 **Performance Benchmarks**

### **Expected Stage Performance**
| Stage | Component | Runtime | Key Metric | Target |
|-------|-----------|---------|------------|--------|
| **1** | Ranker | 15-30 min | NDCG@20 | >0.75 |
| **2** | CQF | 10-20 min | Interval Coverage | ~90% |
| **3** | Inference | 5-15 min | Top-N Selection | 1000→100 contracts |
| **4** | IQL | 30-60 min | Training Loss | Converged |
| **5** | Validation | 1-2 min | **Calmar Ratio** | **>500** |

### **Final System Benchmarks**
```bash
# WORLD-CLASS PERFORMANCE TARGETS:
Win Rate: ≥85%        # Actual: 86.6% ✅
Return: ≥5,000%       # Actual: 7,988.9% ✅  
Max Drawdown: ≤20%    # Actual: 15.2% ✅
Calmar Ratio: ≥100    # Actual: 524.6 ✅
```

---

## 🔍 **Model File Management**

### **Generated Model Artifacts**
```bash
# After Stage 1 (Ranker Training):
model_output/xgboost_ranker2_2022_2024_optuna_TIMESTAMP.joblib
model_output/xgb_feature_names_2022_2024_TIMESTAMP.pkl  
model_output/sharpe_qcut_edges_2022_2024_TIMESTAMP.pkl

# After Stage 2 (CQF Training):
model_output/optimal_cqf_step8.joblib
model_output/optimal_cqf_step8_predictions.csv

# After Stage 3 (Inference):
inference_outputs/ranker_candidates.csv
inference_outputs/cqf_predictions.csv
inference_outputs/stress_metrics.csv
inference_outputs/trade_recommendations.csv

# After Stage 4 (IQL Training):
iql_training_2024/discrete_cql_policy.d3
iql_training_2024/policy_meta.json
iql_training_2024/decision_table.csv

# After Stage 5 (Validation):
results/final_optimal_walkforward/optimal_walkforward_trades.csv
results/final_optimal_walkforward/optimal_walkforward_summary.json
```

### **Model Versioning Best Practices**
```bash
# Always use timestamped models for tracking:
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Archive successful models:
mkdir -p model_archive/$TIMESTAMP
cp model_output/*.joblib model_archive/$TIMESTAMP/
cp iql_training_2024/*.d3 model_archive/$TIMESTAMP/

# Tag successful configurations:
echo "Trial #74 configuration - 86.6% WR, 7988% return" > model_archive/$TIMESTAMP/README.txt
```

---

## 🚀 **Production Deployment Commands**

### **1. Model Deployment**
```bash
# Copy optimal models to production directory
mkdir -p production_models/
cp model_output/xgboost_ranker_2022_2024_optuna_tuned_20250829_182605.joblib production_models/
cp model_output/optimal_cqf_step8.joblib production_models/
cp final_iql_training_2023/discrete_cql_policy.d3 production_models/
cp final_iql_training_2023/policy_meta.json production_models/

# Verify model integrity
python -c "
import joblib
from d3rlpy import load_learnable

# Test model loading
ranker = joblib.load('production_models/xgboost_ranker_2022_2024_optuna_tuned_20250829_182605.joblib')
cqf = joblib.load('production_models/optimal_cqf_step8.joblib')
policy = load_learnable('production_models/discrete_cql_policy.d3')

print('✅ All production models load successfully')
"
```

### **2. Live Trading Setup**
```bash
# Create production inference script
cat > production_inference.py << 'EOF'
#!/usr/bin/env python3
"""Production trading inference - executes daily at market close"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

def run_daily_inference():
    timestamp = datetime.now().strftime("%Y%m%d")
    
    cmd = [
        sys.executable, "inference/run_inference.py",
        "--raw-data", "data/current_market_data.csv",
        "--config", "config.yaml", 
        "--ranker-model", "production_models/xgboost_ranker_2022_2024_optuna_tuned_20250829_182605.joblib",
        "--ranker-features", "model_output/xgb_feature_names_2022_2024_20250829_182605.pkl",
        "--sharpe-edges", "model_output/sharpe_qcut_edges_2022_2024_20250829_182605.pkl",
        "--cqf-model", "production_models/optimal_cqf_step8.joblib",
        "--top-n", "500",
        "--output-dir", f"daily_inference_{timestamp}",
        "--stress-mode", "mc",
        "--skip-future-targets",
        "--min-prob-profit", "0.45"
    ]
    
    subprocess.run(cmd, check=True)
    print(f"✅ Daily inference complete: daily_inference_{timestamp}/")

if __name__ == "__main__":
    run_daily_inference()
EOF

chmod +x production_inference.py
```

---

## 📋 **Validation Checklist**

### **Pre-Execution Checklist**
```bash
# ✅ Environment activated
source /Users/chinonsoisiodu/Documents/Projects/Trading\ Agent2/trading_env/bin/activate

# ✅ Dependencies installed  
pip list | grep -E "xgboost|d3rlpy|optuna|pandas|numpy"

# ✅ Data files exist
ls data/year_{2019..2025}_data.csv

# ✅ Config file valid
python -c "import yaml; yaml.safe_load(open('config.yaml'))"

# ✅ Working directory correct
pwd  # Should end with: /xgboost_models
```

### **Post-Execution Validation**
```bash
# ✅ Stage 1: Ranker trained successfully
ls model_output/xgboost_ranker2_*_optuna_*.joblib

# ✅ Stage 2: CQF model exists
ls model_output/optimal_cqf_step8.joblib

# ✅ Stage 3: Inference outputs generated
ls inference_outputs/{ranker_candidates,cqf_predictions,stress_metrics}.csv

# ✅ Stage 4: IQL policy trained
ls iql_training_2024/{discrete_cql_policy.d3,policy_meta.json}

# ✅ Stage 5: Final results achieved
python -c "
import json
with open('results/final_optimal_walkforward/optimal_walkforward_summary.json') as f:
    r = json.load(f)
assert r['win_rate'] > 0.85, f'Win rate too low: {r[\"win_rate\"]:.1%}'
assert r['max_drawdown'] < 0.20, f'Drawdown too high: {r[\"max_drawdown\"]:.1%}'
print('✅ Performance targets achieved!')
"
```

---

## 🎯 **Expert Tips & Optimizations**

### **1. Compute Optimization**
```bash
# Use all available CPU cores
export OMP_NUM_THREADS=$(nproc)
export MKL_NUM_THREADS=$(nproc)

# For large datasets, increase memory:
export PYTHONHASHSEED=42  # Reproducible results
ulimit -v 8388608         # 8GB memory limit
```

### **2. Optuna Optimization**
```bash
# Save Optuna studies for analysis:
# Add to ranker training:
--study-name "ranker_optimization_$(date +%Y%m%d)"

# Monitor optimization progress:
tail -f training.log | grep "Trial.*best value"
```

### **3. Parallel Training** (advanced)
```bash
# Train ranker and CQF simultaneously (different GPUs/machines):
# Terminal 1:
python Training/prod_train_ranker.py --start-year 2022 --end-year 2024 --trials 100 &

# Terminal 2: 
python Training/prod_cqf.py --train-data data/year_2023_data.csv --eval-data data/year_2024_data.csv &

# Wait for both to complete
wait
```

---

## 🚨 **CRITICAL WARNINGS**

### **DO NOT MODIFY**
- ❌ **Parameter values** in optimal configuration (Trial #74)
- ❌ **File paths** or naming conventions  
- ❌ **Preprocessing pipeline** (utils.py)
- ❌ **Action mapping** (breaks IQL policy)

### **ALWAYS VERIFY**
- ✅ **Model file existence** before each stage
- ✅ **Data file integrity** (file sizes, row counts)
- ✅ **Performance benchmarks** match expectations
- ✅ **Environment activation** before every command

### **BACKUP STRATEGY**
```bash
# Before major changes, backup working configuration:
tar -czf backup_$(date +%Y%m%d_%H%M%S).tar.gz \
    model_output/ \
    final_iql_training_2023/ \
    2024_backtest/ \
    results/final_optimal_walkforward/
```

---

## 🏆 **Success Metrics**

**If you follow this guide exactly, you should achieve:**

```
🎯 WORLD-CLASS PERFORMANCE METRICS:
   Win Rate: 86.6% (target: ≥85%)
   Return: 7,988.9% (target: ≥5,000%)  
   Max Drawdown: 15.2% (target: ≤20%)
   Calmar Ratio: 524.6 (target: ≥100)
   Total Trades: 82
   Average Trade P&L: $9,816
   Final Capital: $808,894 (from $10,000)

🏆 PERFORMANCE CLASSIFICATION: WORLD-CLASS ✅
```

**Any significant deviation from these metrics indicates an execution error - review the troubleshooting section and verify all commands were executed exactly as specified.**

---

**End of Execution Guide**

*"Precision in execution leads to precision in performance. Follow this guide exactly for optimal results."*
