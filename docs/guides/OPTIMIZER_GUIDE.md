# Walkforward Optimizer Guide

**Created**: September 29, 2025  
**File**: `optimize_walkforward.py`  
**Purpose**: Find optimal trading parameters for IQL policies

---

## 🎯 Overview

The modern optimizer discovers optimal parameter configurations by:
1. Loading decision table and IQL policy **once** (cached)
2. Running 100+ trials with different parameter combinations
3. Evaluating each using `walkforward_simulator.py`
4. Applying hard constraints (win rate, trades, drawdown)
5. Optimizing chosen objective (Calmar, Sharpe, etc.)

**Key Improvement**: 10-100× faster than legacy optimizers (no subprocess overhead, cached data).

---

## 🚀 Quick Start

### Basic Usage
```bash
/Users/chinonsoisiodu/Documents/Projects/Trading\ Agent2/trading_env/bin/python3 \
  optimize_walkforward.py \
  --decision-table ../iql_out/decision_table.csv \
  --policy ../iql_out/discrete_cql_policy.d3 \
  --meta ../iql_out/policy_meta.json \
  --objective calmar \
  --trials 100 \
  --outdir ../results/optimization_calmar
```

### Expected Runtime
- **Per trial**: ~0.5-2 seconds (cached data)
- **100 trials**: ~2-5 minutes total
- **vs Legacy**: 60-120 minutes (subprocess overhead)

---

## 📊 Optimization Objectives

### 1. Calmar Ratio (Default) - `--objective calmar`
**Formula**: `Return% / (MaxDrawdown% × bonuses)`

**Best for**: Maximizing risk-adjusted returns

**Bonuses**:
- Win rate ≥ 85%: ×1.1
- Trades ≥ 85: ×1.05
- Drawdown ≤ 15%: ×1.3
- Drawdown ≤ 20%: ×1.15

**Example**: 7,000% return with 15% drawdown = Calmar ~467 → with bonuses ~680

---

### 2. Sharpe Ratio - `--objective sharpe`
**Formula**: `Return% / (EquityVolatility × √252)`

**Best for**: Consistent performance with low volatility

**Use when**: You want smooth equity curves over maximum returns

---

### 3. Pure Return - `--objective return`
**Formula**: `Return% - DrawdownPenalty`

**Best for**: Maximum returns with soft drawdown constraint

**Trade-off**: May accept higher drawdowns for higher returns

---

### 4. Drawdown Minimization - `--objective drawdown`
**Formula**: `-(MaxDrawdown + λ × ReturnShortfall)`

**Best for**: Conservative strategies prioritizing capital preservation

**Use when**: Risk management is primary concern

---

## ⚙️ Parameter Space

The optimizer explores these parameters:

### Core Settings (Always Sampled)
```python
bypass_all_normal_controls: [True, False]
max_notional_pct: [0.08, 0.20]  # Portfolio exposure per trade
```

### Portfolio Risk Controls
```python
enable_portfolio_stop_loss: [True, False]
  portfolio_stop_loss_pct: [0.30, 0.80]  # If enabled

enable_single_trade_cap: [True, False]
  max_single_trade_notional: [50k, 200k]  # If enabled
```

### Market Halts
```python
enable_market_halt_protection: [True, False]
  halt_vol_emergency_only: [True, False]
  halt_vol_severity_threshold: [1.5, 3.0]
```

### Loss Protection
```python
enable_consecutive_loss_breaker: [True, False]
  max_consecutive_losses: [5, 50]  # If enabled
```

### Position Sizing (Key Return Lever)
```python
enable_position_multiplier: [True, False]
  position_multiplier: [1.0, 3.0]  # CRITICAL PARAMETER

enable_dynamic_sizing: [True, False]
  lookback_window: [5, 25]  # Recent performance window

enable_vol_adjustment: [True, False]
  vol_lookback: [10, 40]  # Volatility calculation window
```

### Return Filtering (Leak-Free)
```python
enable_return_filter: [True, False]
  min_expected_return: [0.0, 0.05]  # Uses CQF expected_return
```

---

## 🔒 Hard Constraints

Trials violating constraints receive heavy penalties:

| Constraint | Default | Penalty (if violated) |
|------------|---------|----------------------|
| **Min Win Rate** | 80% | 10,000 × shortfall |
| **Min Trades** | 70 | 100 × shortfall |
| **Max Drawdown** | 60% | 10,000 × excess |
| **Min Return** | 1000% | 10 × shortfall |

**Constraint Satisfaction**: Only trials meeting ALL constraints are considered for "best valid configuration".

---

## 📈 Expected Results

### Calmar Optimization (Historical Data)
```
Baseline (no optimization):
  Win Rate: 83.9%
  Return: 1,120%
  Max Drawdown: 64.9%
  Calmar: 17.3

Expected Optimized:
  Win Rate: 84-87%
  Return: 3,000-8,000%
  Max Drawdown: 15-35%
  Calmar: 150-700+
```

### Leak-Free Mode
```
Expected Results (varies due to randomness):
  Win Rate: 45-65%
  Return: -500% to +2,000%
  Max Drawdown: 20-50%

Note: Run multiple times to assess stability.
CQF accuracy determines performance ceiling.
```

---

## 🔍 Modes

### Backtest Mode (`--mode backtest`)
- **Data Required**: `c{slot}_target_pnl`, `c{slot}_future_option_price`
- **PnL Source**: Actual realized returns
- **Use**: Historical analysis, parameter tuning
- **Results**: Deterministic, exact

### Leak-Free Mode (`--mode leakfree`)
- **Data Required**: Only `c{slot}_expected_return`, `c{slot}_prob_profit`
- **PnL Source**: Simulated from CQF predictions + noise
- **Use**: Model validation, production testing
- **Results**: Stochastic, realistic

---

## 🎓 Usage Examples

### Example 1: Find Maximum Risk-Adjusted Returns
```bash
python optimize_walkforward.py \
  --decision-table data.csv \
  --policy policy.d3 \
  --meta meta.json \
  --objective calmar \
  --trials 200 \
  --min-win-rate 0.83 \
  --outdir results/opt_calmar
```

### Example 2: Minimize Drawdown
```bash
python optimize_walkforward.py \
  --decision-table data.csv \
  --policy policy.d3 \
  --meta meta.json \
  --objective drawdown \
  --trials 150 \
  --min-return 2000.0 \
  --max-drawdown 0.30 \
  --outdir results/opt_drawdown
```

### Example 3: Validate in Leak-Free Mode
```bash
python optimize_walkforward.py \
  --decision-table data_no_targets.csv \
  --policy policy.d3 \
  --meta meta.json \
  --objective calmar \
  --mode leakfree \
  --trials 50 \
  --min-win-rate 0.50 \
  --outdir results/opt_leakfree
```

---

## 📁 Output Files

After optimization, you'll find:

### Main Results
```
results/optimization_calmar/
├── optimization_calmar_backtest.json  # Full study results
├── best_params_calmar.json            # Best parameter configuration
└── best_summary_calmar.json           # Best performance summary
```

### JSON Structure

**`optimization_calmar_backtest.json`**:
```json
{
  "study_name": "walkforward_calmar_backtest",
  "objective": "calmar",
  "n_trials": 100,
  "valid_trials_count": 45,
  "best_trial_number": 73,
  "best_valid_metrics": {
    "win_rate": 0.866,
    "return_pct": 7234.5,
    "max_drawdown": 0.162,
    "calmar_ratio": 446.9,
    "total_trades": 84
  }
}
```

**`best_params_calmar.json`**:
```json
{
  "position_multiplier": 2.4,
  "max_consecutive_losses": 15,
  "enable_portfolio_stop_loss": false,
  "enable_dynamic_sizing": true,
  "lookback_window": 15,
  ...
}
```

---

## 🔬 Understanding Results

### Interpreting Objective Values

**Calmar Ratio**:
- `< 20`: Poor
- `20-100`: Good
- `100-300`: Excellent
- `> 300`: World-class

**Sharpe Ratio**:
- `< 1.0`: Poor
- `1.0-2.0`: Good
- `2.0-3.0`: Excellent
- `> 3.0`: Outstanding

### Constraint Satisfaction Rate

```
Valid trials: 45/100 = 45% satisfaction rate
```

**If < 20%**: Constraints too tight, relax them
**If > 80%**: Constraints too loose, tighten for better solutions

---

## 🛠️ Advanced Usage

### Custom Constraints
```bash
python optimize_walkforward.py \
  ... \
  --min-win-rate 0.85 \     # Require 85%+ win rate
  --min-trades 80 \         # Need at least 80 trades
  --max-drawdown 0.40 \     # Max 40% drawdown
  --min-return 3000.0       # Min 3000% returns
```

### More Trials for Better Results
```bash
python optimize_walkforward.py \
  ... \
  --trials 300 \           # More exploration
  --objective calmar
```

### Test Multiple Objectives
```bash
# Run all objectives
for obj in calmar sharpe return drawdown; do
  python optimize_walkforward.py \
    --decision-table data.csv \
    --policy policy.d3 \
    --meta meta.json \
    --objective $obj \
    --trials 100 \
    --outdir results/opt_$obj
done
```

---

## 🐛 Troubleshooting

### "No trials satisfied all constraints"
**Solution**: Relax constraints or increase trials
```bash
--min-win-rate 0.75 \  # Lower from 0.80
--min-trades 50 \      # Lower from 70
--trials 200           # More exploration
```

### "FileNotFoundError: walkforward_simulator.py"
**Solution**: Ensure you're running from Training/ directory
```bash
cd Training/
python optimize_walkforward.py ...
```

### Optimization too slow
**Solution**: Reduce trials or simplify data
```bash
--trials 50  # Quick test
# or sample your decision table first
```

### Poor results in leak-free mode
**Expected**: Leak-free mode has high variance due to simulated PnL
**Solution**: Run multiple times, average results, or improve CQF model

---

## 📊 Comparison with Legacy Optimizers

| Feature | Legacy (Archived) | New (optimize_walkforward.py) |
|---------|------------------|-------------------------------|
| **Architecture** | Subprocess script generation | Direct import |
| **Speed** | 60-120 min for 100 trials | 2-5 min for 100 trials |
| **Return Filter** | ⚠️ Used `target_pnl` (leak) | ✅ Uses `expected_return` (safe) |
| **Dependencies** | Broken (archived files) | ✅ Uses `walkforward_simulator.py` |
| **Debugging** | Hard (subprocess errors) | Easy (direct stack traces) |
| **Memory** | High (each trial loads data) | Low (cached data) |
| **Objectives** | 1 (Calmar only) | 4 (Calmar, Sharpe, Return, Drawdown) |

---

## ✅ Validation Checklist

Before trusting optimization results:

- [ ] Run on historical data with known good baseline
- [ ] Verify best config reproduces expected performance
- [ ] Check constraint satisfaction rate (should be 20-80%)
- [ ] Compare multiple objectives (should agree on key parameters)
- [ ] Test in leak-free mode (sanity check)
- [ ] Verify return filter uses `expected_return` not `target_pnl`

---

## 🎯 Recommended Workflow

### Step 1: Quick Exploration (50 trials)
```bash
python optimize_walkforward.py \
  --decision-table data.csv \
  --policy policy.d3 \
  --meta meta.json \
  --objective calmar \
  --trials 50 \
  --outdir results/opt_quick
```

### Step 2: Deep Optimization (200 trials)
```bash
python optimize_walkforward.py \
  --decision-table data.csv \
  --policy policy.d3 \
  --meta meta.json \
  --objective calmar \
  --trials 200 \
  --outdir results/opt_deep
```

### Step 3: Validate Best Config
```bash
# Test with actual walkforward simulator
python walkforward_simulator.py \
  --decision-table data.csv \
  --policy policy.d3 \
  --meta meta.json \
  --mode backtest \
  --config custom \  # Load from best_params_calmar.json
  --outdir results/validation
```

### Step 4: Test in Leak-Free Mode
```bash
python optimize_walkforward.py \
  --decision-table data.csv \
  --policy policy.d3 \
  --meta meta.json \
  --objective calmar \
  --mode leakfree \
  --trials 100 \
  --min-win-rate 0.50 \  # Lower expectations for simulated PnL
  --outdir results/opt_leakfree
```

---

## 📋 Parameter Interpretation Guide

### Critical Parameters (Biggest Impact)

**`position_multiplier`** (1.0-3.0):
- 1.0 = Conservative (baseline sizing)
- 2.0 = Moderate leverage
- 2.5 = Aggressive (Trial #74 optimal)
- 3.0 = Very aggressive (only with 85%+ win rate)

**Impact**: ~Linear on returns (2× multiplier ≈ 2× returns)

**`max_consecutive_losses`** (5-50):
- 5-10 = Very strict (frequent halts)
- 15-20 = Moderate protection
- 30-50 = Lenient (rare halts with 83%+ win rate)

**Impact**: Lower = more trade filtering, higher drawdown protection

**`max_notional_pct`** (0.08-0.20):
- 0.08-0.10 = Conservative exposure
- 0.12-0.15 = Moderate exposure
- 0.16-0.20 = Aggressive exposure

**Impact**: Caps per-trade sizing based on current equity

---

### Secondary Parameters

**Dynamic Sizing**:
- `lookback_window` = 10-20 typical
- Reduces size after losses, increases after wins
- Good for adapting to changing market conditions

**Volatility Adjustment**:
- `vol_lookback` = 20-30 typical
- Reduces size during high equity volatility
- Good for risk-adjusted steady growth

**Portfolio Stop-Loss**:
- `portfolio_stop_loss_pct` = 0.50-0.70 typical
- Halts all trading after large drawdown
- Trial #62 used 0.65 successfully

---

## 🔬 Optimization Science

### Why Fresh Optimization Needed

**Code Changes Made**:
1. ✅ Fixed return filter leak (no longer uses `target_pnl`)
2. ✅ Unified simulator (consistent architecture)
3. ✅ Better leak prevention (multi-layer protection)
4. ✅ Cleaner settlement logic (delayed, realistic)

**Impact**: Old Trial #74/#62 may no longer be optimal because:
- Return filter behavior changed (leak fix)
- Simulator logic improved (settlement timing)
- Feature availability may differ

### Expected Findings

**Likely discoveries**:
- Position multiplier sweet spot: 2.0-2.8× (with 80%+ win rate)
- Consecutive loss limit: 10-25 (model-dependent)
- Dynamic sizing: Beneficial for Trial #62 approach, neutral for #74
- Volatility adjustment: Small benefit (2-5% Calmar improvement)

**Key insight**: With 85%+ win rate models, simpler configs often outperform complex ones.

---

## ⚠️ Common Pitfalls

### 1. Overfitting to Training Period
**Problem**: Best config on 2024 data may not generalize

**Solution**: 
- Validate on held-out period
- Test in leak-free mode
- Prefer simpler configs with fewer active features

### 2. Chasing Maximum Returns
**Problem**: Highest return often comes with unstable configs

**Solution**:
- Use Calmar (return/risk) not pure return objective
- Apply drawdown constraints
- Check how often config would have halted

### 3. Too Tight Constraints
**Problem**: No valid trials found

**Solution**:
- Start with loose constraints (WR≥0.75, Trades≥50)
- Gradually tighten based on results
- Check baseline performance first

### 4. Ignoring Mode
**Problem**: Backtesting with leaked features, deploying with leak-free expectations

**Solution**:
- Backtest mode: Historical analysis only
- Leak-free mode: Production validation only
- Never confuse the two!

---

## 📊 Interpreting Results

### Output Files Explained

**`optimization_calmar_backtest.json`**:
- Complete study metadata
- All trial results
- Best overall and best valid trials
- Baseline comparison

**`best_params_calmar.json`**:
- Exact parameter dictionary
- Can be loaded into `walkforward_simulator.py`
- Use for production deployment

**`best_summary_calmar.json`**:
- Full performance metrics
- Trade-by-trade details
- Equity curve data
- Risk statistics

### Reading the Logs

```
Trial  42: ✅ obj=  542.31 | WR=86.2% | Ret=6842.1% | MDD=12.6% | Trades= 83
```

- `✅` = Constraints satisfied
- `obj` = Objective value (higher better for Calmar)
- `WR` = Win rate
- `Ret` = Total return
- `MDD` = Maximum drawdown
- `Trades` = Number executed

---

## 🏆 Success Criteria

### Good Optimization Run
- ✅ 20-80% of trials satisfy constraints
- ✅ Best trial significantly beats baseline
- ✅ Top 5 trials have similar parameters
- ✅ Clear parameter clusters emerge

### Bad Optimization Run
- ❌ <10% constraint satisfaction (too tight)
- ❌ >90% constraint satisfaction (too loose)
- ❌ Best trial only marginally better than baseline
- ❌ Top trials have wildly different parameters
- ❌ Results don't replicate when re-run

---

## 🔄 Iterative Optimization Strategy

### Round 1: Broad Exploration
```bash
python optimize_walkforward.py \
  --objective calmar \
  --trials 100 \
  --min-win-rate 0.75 \  # Loose constraints
  --min-trades 60
```

**Goal**: Understand parameter landscape

---

### Round 2: Focused Search
Based on Round 1, identify promising regions:
```bash
python optimize_walkforward.py \
  --objective calmar \
  --trials 200 \
  --min-win-rate 0.83 \  # Tighter constraints
  --min-trades 75
```

**Goal**: Find optimal config

---

### Round 3: Validation
```bash
# Test best config in leak-free mode
python optimize_walkforward.py \
  --objective calmar \
  --mode leakfree \
  --trials 50 \
  --min-win-rate 0.50  # Realistic for simulated
```

**Goal**: Verify no overfitting

---

## 📚 Further Reading

- `walkforward_simulator.py` - Simulation engine documentation
- `WALKFORWARD_MIGRATION.md` - Migration from old files
- `Walkforward_OPTIMIZATION_JOURNEY.md` - Historical optimization story
- `best_risk_adjusted_params.json` - Trial #74 reference config

---

## 🆘 Support

**Issues**:
1. Check logs for error details
2. Verify input files exist and have required columns
3. Test walkforward_simulator.py standalone first
4. Check Python environment is correct

**Questions**:
- What objective should I use? → Start with `calmar`
- How many trials? → 100-200 for production, 50 for quick tests
- Which mode? → `backtest` for optimization, `leakfree` for validation

---

**Created with**: Knowledge from 4 legacy optimizers, unified into modern architecture
**Status**: ✅ Production ready
