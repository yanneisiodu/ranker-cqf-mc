# IQL Training Monitor

**Started**: September 29, 2025  
**Dataset**: 2023 SPY Options (1.9M rows)  
**Models**: 2022 XGBoost Ranker + CQF  
**Output**: `iql_out/2023_training_2022models/`

---

## 📊 Training Progress

### Current Status
Check the log file for real-time progress:
```bash
tail -f iql_training_2023.log
```

### Expected Milestones

#### Stage 1: Inference Pipeline (30-60 min)
```
Look for in logs:
  ✅ "Running inference pipeline: ..."
  ✅ "Loading and preparing data..."
  ✅ "Making predictions using loaded model..."
  ✅ "Generating quantile predictions..."
  ✅ "Predictions saved to: inference_outputs/..."
```

#### Stage 2: Decision Table (2-5 min)
```
Look for in logs:
  ✅ "Loaded merged dataset with X rows"
  ✅ "Built decision table with Y decisions"
  ✅ "Action distribution:"
```

#### Stage 3: IQL Training (15-30 min)
```
Look for in logs:
  ✅ "Training IQL model..."
  ✅ "epoch=1/20, step=10000, ..."
  ✅ "Saving checkpoint..."
  ✅ "Training complete"
```

#### Stage 4: Export (< 1 min)
```
Look for in logs:
  ✅ "Artifacts saved to iql_out/2023_training_2022models"
  ✅ "Training complete"
```

---

## 🔍 Monitoring Commands

### Check Progress
```bash
# Watch log in real-time
tail -f iql_training_2023.log

# Check last 50 lines
tail -50 iql_training_2023.log

# Check for errors
grep -i error iql_training_2023.log

# Check current stage
grep -E "stage|epoch|step" iql_training_2023.log | tail -5
```

### Check Output Files
```bash
# See what's been created
ls -lh iql_out/2023_training_2022models/

# Check inference outputs (if running)
ls -lh iql_out/2023_training_2022models/inference_outputs/
```

### Monitor System Resources
```bash
# CPU/Memory usage
top -pid $(pgrep -f iql_pipeline.py)

# Or simpler
ps aux | grep iql_pipeline
```

---

## 📁 Expected Output Structure

```
iql_out/2023_training_2022models/
├── inference_outputs/
│   ├── cqf_predictions.csv           # CQF quantiles per contract
│   └── ranker_candidates.csv         # Ranked contracts
├── decision_table.csv                # MDP decision states
├── decision_table.parquet            # Same, Parquet format
├── discrete_cql_policy.d3            # Trained IQL policy
├── policy_meta.json                  # Scaler + action map
└── action_map.json                   # Action encoding (if --no-train)
```

---

## ⚡ Quick Verification After Completion

### 1. Check files were created
```bash
ls -lh iql_out/2023_training_2022models/*.{csv,d3,json}
```

### 2. Verify decision table
```bash
python3 -c "
import pandas as pd
dt = pd.read_csv('iql_out/2023_training_2022models/decision_table.csv', nrows=5)
print(f'Decision table shape: {dt.shape}')
print(f'Columns: {list(dt.columns[:10])}')
print(f'Sample action_ids: {dt[\"action_id\"].value_counts().head()}')
"
```

### 3. Verify policy can be loaded
```bash
python3 -c "
from d3rlpy import load_learnable
policy = load_learnable('iql_out/2023_training_2022models/discrete_cql_policy.d3')
print(f'✅ Policy loaded: {type(policy)}')
"
```

---

## 🚨 Common Issues & Solutions

### Issue: "ModuleNotFoundError: regime_tools"
**Cause**: CQF model pickle references Training/regime_tools.py  
**Solution**: Ensure Training/ is in Python path or run from project root

### Issue: "Inference outputs missing"
**Cause**: Inference pipeline failed  
**Solution**: Check log for errors in ranker/CQF prediction stage

### Issue: "No valid state columns"
**Cause**: All columns filtered as potential leaks  
**Solution**: Check meta['state_columns'] has non-leaked features

### Issue: Training very slow
**Cause**: Running on CPU instead of MPS  
**Solution**: Verify device="mps" in iql_pipeline.py line 491

---

## 📊 Expected Performance

### Decision Table Stats
```
Typical for 2023 data:
  - Total decisions: 150,000 - 300,000
  - Unique dates: ~250 trading days
  - Candidates per decision: 5 (top-k)
  - Action space: 11 actions (5 slots × 2 sizes + no-trade)
```

### Training Metrics
```
Watch for in logs:
  - Loss decreasing over epochs
  - TD error stabilizing
  - Q-values converging
  
Good signs:
  - Smooth loss curves
  - Stable critic values
  - Action distribution not degenerate (uses multiple actions)
```

---

## ⏱️ Time Estimates by Hardware

### Apple Silicon (M1/M2/M3) with MPS
- Inference: 20-40 min
- Decision table: 2-3 min
- IQL training: 10-20 min
- **Total**: 35-65 min

### CPU Only
- Inference: 40-80 min
- Decision table: 3-5 min
- IQL training: 30-60 min
- **Total**: 75-145 min

---

## ✅ Success Checklist

After training completes:

- [ ] No errors in log file
- [ ] Decision table created (~150K-300K rows)
- [ ] Policy file exists (discrete_cql_policy.d3)
- [ ] Policy metadata exists (policy_meta.json)
- [ ] Action distribution shows diversity (not all action 0)
- [ ] Can load policy with d3rlpy.load_learnable()

---

**Monitor**: `tail -f iql_training_2023.log`  
**ETA**: Check back in 45-90 minutes
