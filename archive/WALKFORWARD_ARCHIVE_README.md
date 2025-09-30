# Walkforward Simulation Archive

**Archive Date**: September 29, 2025  
**Reason**: Consolidated into unified `Training/walkforward_simulator.py`

---

## 📦 Archived Files

### Historical Walkforward Implementations

All these files have been **superseded by** `Training/walkforward_simulator.py`, which provides:
- Unified codebase with mode flags (`--mode backtest|leakfree`)
- Built-in Trial #74 and Trial #62 configurations
- Better leak prevention
- Cleaner architecture

---

### File Descriptions

#### `final_optimal_walkforward.py` (18 KB, 449 lines)
- **Purpose**: Simplest walkforward implementation
- **Features**: Basic simulation, immediate settlement
- **Trial**: #74 (2.5× leverage, emergency-only controls)
- **Issues**: No delayed settlement (unrealistic)
- **Replaced By**: `walkforward_simulator.py --mode backtest --config trial74`

#### `final_optimal_walkforward_copy.py` (29 KB, 696 lines)
- **Purpose**: Full-featured backtest with delayed settlement
- **Features**: Delayed settlement, dynamic sizing, vol adjustment
- **Trial**: #62 (2.5× leverage + dynamic controls)
- **Issues**: Return filter had look-ahead bias (used `target_pnl` directly)
- **Replaced By**: `walkforward_simulator.py --mode backtest --config trial62`

#### `final_optimal_walkforward_10.py` (33 KB, 808 lines)
- **Purpose**: Research version with advanced risk features
- **Features**: Trailing stop cooldown, daily notional tracking, vol targeting
- **Trial**: V10 Trial #6 (1.6× leverage, very conservative)
- **Issues**: Overly complex (100+ extra lines), worse performance than Trial #74
- **Replaced By**: Advanced features deemed unnecessary; Trial #74/62 perform better

#### `final_optimal_walkforward_11.py` (30 KB, 722 lines)
- **Purpose**: Leak-free validation mode
- **Features**: Simulated PnL from CQF predictions, no future data dependency
- **Trial**: #62 configuration adapted for leak-free testing
- **Issues**: None - this was already leak-free
- **Replaced By**: `walkforward_simulator.py --mode leakfree --config trial62`

#### `walkforward_simulation.py` (13 KB, 336 lines)
- **Purpose**: Original basic walkforward from IQL pipeline
- **Features**: Simple risk-based position sizing with bypassed controls
- **Issues**: 
  - Bypassed risk constraints (commented out for testing)
  - Bypassed liquidity caps
  - Fixed notional ($1000/contract) instead of actual pricing
  - No delayed settlement
  - No advanced risk management
- **Replaced By**: `walkforward_simulator.py` (proper implementation of all features)

---

## 🔍 Key Differences vs. New Unified File

### What Was Improved

1. **Leak Prevention**:
   - Old: `copy` and `v10` had return filter using `c{slot}_target_pnl` (future leak)
   - New: Uses `c{slot}_expected_return` from CQF predictions (no leak)

2. **Mode Separation**:
   - Old: Separate files for backtest vs leak-free
   - New: Single file with `--mode` flag

3. **Configuration**:
   - Old: Hardcoded params in each file, different trial configs
   - New: Named presets (`TRIAL_74_CONFIG`, `TRIAL_62_CONFIG`)

4. **Architecture**:
   - Old: Code duplication across 4 files (90% identical)
   - New: Single implementation, ~500 lines total

5. **Testing**:
   - Old: `walkforward_simulation.py` had bypassed risk controls
   - New: Proper risk controls with optional flags

---

## 📊 Performance Comparison

All archived versions claimed similar performance:
- Win Rate: 83.9-86.6%
- Returns: 5,590-7,989%
- Max Drawdown: 15.2-33.8%

**Note**: Results depend heavily on:
1. Whether `enable_return_filter` was used (creates look-ahead bias if True)
2. Position multiplier setting (1.6× to 2.5×)
3. Dynamic sizing and vol adjustment settings

The unified file uses **verified configurations** from optimization studies.

---

## 🔄 Migration Path

### If You Need Original Functionality

Each archived file can still be run if needed:

```bash
# Run archived version
/Users/chinonsoisiodu/Documents/Projects/Trading\ Agent2/trading_env/bin/python3 \
  archive/final_optimal_walkforward_copy.py \
  --decision-table data.csv \
  --policy policy.d3 \
  --meta meta.json \
  --outdir results/
```

### Recommended: Use New Unified File

```bash
# Equivalent to old "copy" version
/Users/chinonsoisiodu/Documents/Projects/Trading\ Agent2/trading_env/bin/python3 \
  Training/walkforward_simulator.py \
  --decision-table data.csv \
  --policy policy.d3 \
  --meta meta.json \
  --mode backtest \
  --config trial62 \
  --outdir results/
```

---

## ⚠️ Known Issues in Archived Files

### `final_optimal_walkforward.py`
- No delayed settlement → unrealistic (P&L settles immediately)
- Not exportable as module

### `final_optimal_walkforward_copy.py` & `_10.py`
- Return filter uses `c{slot}_target_pnl` → look-ahead bias
- Should disable `enable_return_filter` or results are invalid

### `walkforward_simulation.py`
- All risk constraints bypassed with hardcoded values
- Fixed $1000 notional regardless of actual option prices
- Commented-out risk logic suggests this was for debugging only
- **DO NOT USE for production analysis**

### `final_optimal_walkforward_11.py`
- Actually the most correct implementation (leak-free)
- Random component in simulated PnL makes results non-deterministic
- Good for validation, not for reporting exact performance

---

## 🎯 Recommendation

**For all new work**: Use `Training/walkforward_simulator.py`

**Recovery scenarios**:
- Need original Trial #62 exact code → `final_optimal_walkforward_copy.py`
- Need v10 trailing stop logic → `final_optimal_walkforward_10.py`
- Need leak-free reference → `final_optimal_walkforward_11.py`

**Historical reference**: All files preserved in archive for reproducibility.

---

**Consolidation Status**: ✅ Complete  
**Files Preserved**: ✅ All archived safely  
**New Implementation**: ✅ Tested and validated
