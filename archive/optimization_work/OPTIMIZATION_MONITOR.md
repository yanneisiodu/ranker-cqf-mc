# Leak-Free Optimization Monitor

**Started**: September 29, 2025  
**Objective**: Calmar Ratio (return/drawdown)  
**Mode**: Leak-free (simulated PnL)  
**Trials**: 100  
**Dataset**: 113 decisions from 2023

---

## 📊 What's Being Optimized

### Parameter Search Space

**Position Sizing** (Key Return Lever):
- `position_multiplier`: 1.0 to 3.0
- `enable_dynamic_sizing`: True/False
- `enable_vol_adjustment`: True/False

**Risk Controls**:
- `max_consecutive_losses`: 5 to 50
- `portfolio_stop_loss_pct`: 0.30 to 0.80
- `max_notional_pct`: 0.08 to 0.20

**Market Protection**:
- `enable_market_halt_protection`: True/False
- `halt_vol_emergency_only`: True/False

**Trade Filtering**:
- `enable_return_filter`: True/False (uses `expected_return` - leak-free!)
- `min_expected_return`: 0.0 to 0.05

---

## 🎯 Optimization Objective

**Calmar Ratio**: `Return% / (MaxDrawdown% × Bonuses)`

**Bonuses Applied**:
- Win rate ≥ 85%: ×1.1
- Trades ≥ 85: ×1.05
- Drawdown ≤ 15%: ×1.3
- Drawdown ≤ 20%: ×1.15

**Hard Constraints** (must satisfy ALL):
- Win Rate ≥ 40%
- Trades ≥ 5
- Max Drawdown ≤ 60%
- Return ≥ -100%

Trials violating constraints get heavy penalties.

---

## ⏱️ Expected Timeline

With cached data and direct imports:
- **Per trial**: 1-2 seconds
- **100 trials**: 2-5 minutes total
- **vs Legacy**: Would be 60-120 minutes!

---

## 📋 Monitoring Commands

### Watch Progress
```bash
tail -f optimization_leakfree.log
```

### Check Current Best
```bash
grep "✅" optimization_leakfree.log | tail -10
```

### Count Completed Trials
```bash
grep "Trial" optimization_leakfree.log | wc -l
```

---

## 📁 Expected Outputs

```
results/optimization_leakfree_calmar/
├── optimization_calmar_leakfree.json   # Full study results
├── best_params_calmar.json             # Optimal parameters
└── best_summary_calmar.json            # Performance details
```

---

## 🎯 What to Expect

### Typical Leak-Free Results
With simulated PnL and small dataset (113 decisions):

**Good Trial**:
- Win Rate: 45-60%
- Return: -20% to +50%
- Drawdown: 15-30%
- Calmar: -5 to +5

**Excellent Trial**:
- Win Rate: 55-65%
- Return: +10% to +100%
- Drawdown: 10-20%
- Calmar: +5 to +15

**Note**: High variance due to random component in simulated PnL.

---

## ✅ Success Indicators

- ✅ 20-50% of trials satisfy constraints
- ✅ Clear parameter patterns emerge (e.g., multiplier ~2.0-2.5)
- ✅ Top trials have similar configurations
- ✅ Best trial significantly better than baseline

---

## 🔍 After Completion

### Validate Best Configuration
```bash
# Test best config
python Training/walkforward_simulator.py \
  --decision-table iql_out/2023_training_2022models/decision_table.csv \
  --policy iql_out/2023_training_2022models/discrete_cql_policy.d3 \
  --meta iql_out/2023_training_2022models/policy_meta.json \
  --mode leakfree \
  --config custom  # Load from best_params_calmar.json
```

### Run Multiple Times
Since leak-free has random component, run 5-10 times to assess stability:
```bash
for i in {1..5}; do
  python Training/walkforward_simulator.py ... --outdir results/run_$i
done
```

---

**Monitor**: `tail -f optimization_leakfree.log`  
**ETA**: Check back in 2-5 minutes
