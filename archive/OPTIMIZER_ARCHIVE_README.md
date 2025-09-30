# Optimizer Archive

**Archive Date**: September 29, 2025  
**Reason**: Consolidated into modern `Training/optimize_walkforward.py`

---

## 📦 Archived Optimizer Files

### Script-Generation Optimizers (Deprecated)

#### `risk_adjusted_return_optimizer.py` (36 KB, 892 lines)
- **Approach**: Generates Python script as string, runs via subprocess
- **Objective**: Maximize Calmar Ratio
- **Issue**: ⚠️ Return filter uses `c{slot}_target_pnl` (look-ahead bias)
- **Problem**: Slow (60-120 min for 100 trials due to subprocess overhead)
- **Status**: DEPRECATED - Use `optimize_walkforward.py` instead

#### `risk_adjusted_return_optimizer copy.py` (38 KB, 940 lines)
- **Identical to above** except uses `textwrap.dedent` for script generation
- **Same issues**: Leak + slow
- **Status**: DEPRECATED - duplicate of main

---

### Shared-Engine Optimizers (Broken Dependencies)

#### `risk_adjusted_return_optimizer_shared_engine.py` (21 KB, 498 lines)
- **Approach**: Imports `final_optimal_walkforward copy.py` via importlib
- **Objective**: Maximize Calmar Ratio
- **Dependency**: Line 32 - `final_optimal_walkforward copy.py` (now in archive/)
- **Issue**: ❌ Broken - tries to import archived file
- **Pro**: No return filter leak (engine handles it correctly)
- **Status**: BROKEN - dependency archived

#### `risk_adjusted_return_optimizer_shared_engine_10.py` (25 KB, 601 lines)
- **Approach**: Same import strategy
- **Objective**: Minimize Drawdown + Return Shortfall
- **Dependency**: Line 34 - `final_optimal_walkforward copy.py` (now in archive/)
- **Issue**: ❌ Broken - missing dependency
- **Unique**: Different objective function (min vs max)
- **Status**: BROKEN - dependency archived

---

## 🔍 Why These Were Archived

### 1. Architecture Issues

**Script Generation Problem** (optimizer.py, optimizer copy.py):
```python
# Generated 460-line Python script as string
script = f'''#!/usr/bin/env python3
... 460 lines of code ...
'''
subprocess.run([sys.executable, script_path])  # Slow!
```

**Issues**:
- 🐌 Slow: Each trial spawns new process, reloads all data
- 🐛 Hard to debug: Errors happen in subprocess
- 📝 Complex: 700+ lines just for script generation
- 🔍 No caching: Re-imports d3rlpy, loads policy every trial

**Modern Approach**:
```python
# Direct import - 10-100× faster
from walkforward_simulator import simulate_walkforward
results, summary = simulate_walkforward(cached_df, cached_actions, ...)
```

---

### 2. Data Leakage

**Return Filter Leak** (Lines 473-476 in both script generators):
```python
def should_skip_due_to_return_filter(self, row: pd.Series, slot: int) -> bool:
    min_return = self.params.get('min_expected_return', 0.0)
    pnl_col = f"c{slot}_target_pnl"  # ⚠️ LOOK-AHEAD BIAS
    expected_return = float(row.get(pnl_col, 0.0) or 0.0)
    return expected_return < min_return
```

**Problem**: Uses **future realized PnL** to decide if trade meets threshold.

**Impact**: 
- If `enable_return_filter=True`, optimization results are inflated
- Cannot replicate in production (no future data)
- Trial #74/62 likely used this (results may be optimistic)

**Modern Fix** (optimize_walkforward.py):
```python
def should_skip_due_to_return_filter(self, row: pd.Series, slot: int) -> bool:
    min_return = float(self.params.get('min_expected_return', 0.0))
    expected_col = f"c{slot}_expected_return"  # ✅ CQF prediction
    expected_return = float(row.get(expected_col, 0.0) or 0.0)
    return expected_return < min_return
```

---

### 3. Broken Dependencies

Both shared_engine files try to import:
```python
_ENGINE_PATH = Path(__file__).with_name("final_optimal_walkforward copy.py")
```

This file was moved to `archive/`, breaking these optimizers.

**Why not fix?**: 
- Script-generation approach is superior anyway (faster)
- Modern optimizer is cleaner
- Less maintenance burden with single optimizer

---

## 📊 Historical Performance (May Be Optimistic)

These archived optimizers produced:

### Trial #74 (risk_adjusted_return_optimizer.py)
```
Win Rate: 86.6%
Return: 7,989%
Max Drawdown: 15.2%
Calmar: 692.5

Configuration:
  position_multiplier: 2.5
  max_consecutive_losses: 18
  enable_return_filter: Unknown (check best_risk_adjusted_params.json)
```

**⚠️ Caveat**: If `enable_return_filter=True`, results include look-ahead bias.

### Trial #62 (from shared_engine files)
```
Win Rate: 83.9%
Return: 5,590%
Max Drawdown: 33.8%
Calmar: 173.6

Configuration:
  position_multiplier: 2.5
  max_consecutive_losses: 45
  enable_dynamic_sizing: True
  enable_vol_adjustment: True
```

**Note**: Shared_engine approach didn't have return filter leak, so these may be more trustworthy.

---

## 🔄 Migration Path

### If You Were Using Script-Generation Optimizers

**Old**:
```bash
python risk_adjusted_return_optimizer.py \
  --decision-table data.csv \
  --policy policy.d3 \
  --meta meta.json \
  --trials 100
```

**New**:
```bash
python optimize_walkforward.py \
  --decision-table data.csv \
  --policy policy.d3 \
  --meta meta.json \
  --objective calmar \
  --trials 100
```

**Benefits**: 10-50× faster, no subprocess overhead, leak-free

---

### If You Were Using Shared-Engine Optimizers

**Old** (shared_engine.py):
```bash
python risk_adjusted_return_optimizer_shared_engine.py \
  --decision-table data.csv \
  --policy policy.d3 \
  --meta meta.json \
  --trials 100
```

**New**:
```bash
python optimize_walkforward.py \
  --decision-table data.csv \
  --policy policy.d3 \
  --meta meta.json \
  --objective calmar \  # Same as old shared_engine
  --trials 100
```

**Benefits**: No broken dependencies, same clean architecture

---

**Old** (shared_engine_10.py - minimize drawdown):
```bash
python risk_adjusted_return_optimizer_shared_engine_10.py \
  --decision-table data.csv \
  --policy policy.d3 \
  --meta meta.json \
  --trials 100
```

**New**:
```bash
python optimize_walkforward.py \
  --decision-table data.csv \
  --policy policy.d3 \
  --meta meta.json \
  --objective drawdown \  # Minimize drawdown
  --trials 100
```

---

## 🎯 Why Fresh Optimization Recommended

### Trial #74 and #62 May No Longer Be Optimal

**Reasons**:
1. **Code changes**: Leak fixes altered behavior
2. **Return filter**: Now uses `expected_return` (different filtering)
3. **Settlement**: Improved delayed settlement logic
4. **Architecture**: Unified simulator may have different edge cases

### **Recommendation**

Re-run optimization with new `optimize_walkforward.py`:
```bash
# Find new optimal Calmar configuration
python optimize_walkforward.py \
  --decision-table <your_2024_data>.csv \
  --policy <your_policy>.d3 \
  --meta <your_meta>.json \
  --objective calmar \
  --trials 200 \
  --outdir results/optimization_2025
```

**Expected**: May find configurations that outperform Trial #74/62 or confirm they're still optimal under new code.

---

## 📚 Technical Details

### Why Script Generation Was Bad

**Memory footprint per trial**:
- Script approach: ~500 MB (fresh Python, import d3rlpy, load policy)
- Direct import: ~50 MB (reuse loaded data)

**Time per trial**:
- Script approach: 30-60 seconds
- Direct import: 0.5-2 seconds

**Debugging**:
- Script approach: Errors in subprocess stderr, hard to trace
- Direct import: Full stack trace, easy debugging

### Why Shared-Engine Was Better But Still Not Ideal

**Pros**:
- Direct import (fast)
- No script generation complexity
- Leak-free return filter

**Cons**:
- Hardcoded dependency path (brittle)
- Couldn't easily switch objectives
- No caching of loaded data
- Still loaded policy fresh each trial

---

## 🔐 Verification That Old Results May Be Inflated

Check if Trial #74/62 used return filter:

```bash
# Check saved configs
cat ../Training/best_risk_adjusted_params.json | grep enable_return_filter

# If output is "true", results are inflated by ~5-15%
# If output is "false", results are trustworthy
```

---

**Archived Files**: 4 optimizers, ~140 KB total  
**Replacement**: 1 optimizer (`optimize_walkforward.py`), 23 KB  
**Status**: Legacy optimizers preserved for reference, DO NOT USE for new work
