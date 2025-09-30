# Walkforward Simulator Migration Guide

**Date**: September 29, 2025  
**Status**: ✅ Consolidated to single unified file

---

## 📋 Summary

Consolidated 4 separate walkforward implementations into a single unified simulator:

**Old Files** (moved to `../archive/`):
- `final_optimal_walkforward.py` → Simple version (no delayed settlement)
- `final_optimal_walkforward copy.py` → Full backtest version  
- `final_optimal_walkforward_10.py` → Research version (v10 features)
- `final_optimal_walkforward_11.py` → Leak-free version

**New File**:
- `walkforward_simulator.py` → Unified implementation with mode flags

---

## 🚀 Quick Start

### Backtest Mode (Historical Analysis)
```bash
/Users/chinonsoisiodu/Documents/Projects/Trading\ Agent2/trading_env/bin/python3 \
  walkforward_simulator.py \
  --decision-table ../iql_out/decision_table.csv \
  --policy ../iql_out/discrete_cql_policy.d3 \
  --meta ../iql_out/policy_meta.json \
  --mode backtest \
  --config trial74 \
  --outdir ../results/backtest_trial74
```

### Leak-Free Mode (Production Validation)
```bash
/Users/chinonsoisiodu/Documents/Projects/Trading\ Agent2/trading_env/bin/python3 \
  walkforward_simulator.py \
  --decision-table ../iql_out/decision_table.csv \
  --policy ../iql_out/discrete_cql_policy.d3 \
  --meta ../iql_out/policy_meta.json \
  --mode leakfree \
  --config trial74 \
  --outdir ../results/leakfree_trial74
```

---

## 🎯 Configuration Presets

### Trial #74 (Default) - Maximum Returns
```
Win Rate: 86.6%
Returns: 7,989%
Max Drawdown: 15.2%
Calmar Ratio: 692.5

Features:
  - 2.5× position multiplier
  - Emergency-only controls
  - 18 consecutive loss limit
  - No dynamic sizing (trusts model)
```

### Trial #62 - Dynamic Risk Management
```
Win Rate: 83.9%
Returns: 5,590%
Max Drawdown: 33.8%
Calmar Ratio: 173.6

Features:
  - 2.5× position multiplier
  - 65% portfolio stop-loss
  - 45 consecutive loss limit
  - Dynamic sizing enabled
  - Volatility adjustment (25-day lookback)
```

---

## 📊 Mode Comparison

| Feature | Backtest Mode | Leak-Free Mode |
|---------|--------------|----------------|
| **PnL Source** | Actual `c{slot}_target_pnl` | Simulated from CQF predictions |
| **Price Source** | `c{slot}_future_option_price` | `c{slot}_last` or fallback |
| **Data Required** | Must have future targets | Only needs predictions |
| **Results** | Exact historical performance | Realistic with uncertainty |
| **Use Case** | Historical analysis | Production validation |
| **Reproducibility** | Deterministic | Has random component |

---

## 🔄 Migration from Old Files

### If You Were Using `final_optimal_walkforward.py`
```bash
# Old command
python final_optimal_walkforward.py --decision-table data.csv ...

# New command
python walkforward_simulator.py --decision-table data.csv --mode backtest --config trial74 ...
```

### If You Were Using `final_optimal_walkforward copy.py`
```bash
# Old command
python "final_optimal_walkforward copy.py" --decision-table data.csv ...

# New command  
python walkforward_simulator.py --decision-table data.csv --mode backtest --config trial62 ...
```

### If You Were Using `final_optimal_walkforward_11.py` (Leak-Free)
```bash
# Old command
python final_optimal_walkforward_11.py --decision-table data.csv ...

# New command
python walkforward_simulator.py --decision-table data.csv --mode leakfree --config trial62 ...
```

---

## 🛠️ Advanced Usage

### Custom Parameters
```python
from walkforward_simulator import simulate_walkforward, TRIAL_74_CONFIG

# Modify config
custom_params = TRIAL_74_CONFIG.copy()
custom_params['position_multiplier'] = 2.0  # More conservative
custom_params['max_consecutive_losses'] = 10  # Stricter

# Run programmatically
results, summary = simulate_walkforward(
    decision_df, predicted_actions, action_map,
    custom_params, initial_capital=10_000, mode="backtest"
)
```

### Import as Module
```python
from walkforward_simulator import (
    simulate_walkforward,
    WalkforwardEngine,
    load_decision_table,
    TRIAL_74_CONFIG,
    TRIAL_62_CONFIG
)
```

---

## ⚠️ Breaking Changes

### Removed Features (from v10)
- `halt_until` / trailing stop cooldown mechanism
- Daily notional tracking (`daily_notional_used`)
- Volatility targeting surface
- Per-trade and daily notional caps

**Rationale**: These added 100+ lines of complexity with minimal performance improvement. Trial #74 (simpler) outperforms v10.

### Fixed Issues
1. **Return filter leak**: Now uses `c{slot}_expected_return` (CQF prediction) instead of `c{slot}_target_pnl` (realized outcome)
2. **Unified architecture**: Single codebase for both backtest and leak-free testing
3. **Proper mode separation**: Can't accidentally mix future targets with leak-free mode

---

## 📈 Performance Expectations

### Backtest Mode with Trial #74
```
Expected Results (2024 data):
  Win Rate: ~86-87%
  Returns: 7,000-8,000%
  Max Drawdown: 15-20%
  Trades: 80-90
  Halts: 5-10
```

### Leak-Free Mode with Trial #74
```
Expected Results (will vary due to randomness):
  Win Rate: ~50-60% (depends on CQF accuracy)
  Returns: Varies widely (-50% to +200%)
  Max Drawdown: Higher than backtest
  
Note: Results depend on CQF prediction quality.
Run multiple times to assess stability.
```

---

## 🔍 Verification Checklist

After migration, verify:

- [ ] `walkforward_simulator.py` imports successfully
- [ ] Both modes execute without errors
- [ ] Backtest mode reproduces historical results
- [ ] Leak-free mode runs on data without future columns
- [ ] Old files safely archived (not deleted)
- [ ] Update any scripts that called old files

---

## 📞 Rollback

If needed, old files are in `../archive/`:
```bash
cp ../archive/final_optimal_walkforward_copy.py .
# or
cp ../archive/final_optimal_walkforward_11.py .
```

---

**Migration Complete**: All walkforward logic consolidated into `walkforward_simulator.py`
