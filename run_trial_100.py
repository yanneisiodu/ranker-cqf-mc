#!/usr/bin/env python3
"""Run walkforward with Trial #100 optimal configuration."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'Training'))

from walkforward_simulator import simulate_walkforward, _load_meta, _load_policy_robust, _standardise_states
import pandas as pd
import json

# Load Trial #100 config
with open('results/trial_100_config.json') as f:
    params = json.load(f)

# Load data
meta = _load_meta(Path('iql_out/2023_training_2022models/policy_meta.json'))
df = pd.read_csv('iql_out/2023_with_targets/decision_table.csv')
df['date'] = pd.to_datetime(df['date'])

# Filter state columns (remove future info)
raw_state_cols = meta['state_columns']
filtered_state_cols = [
    col for col in raw_state_cols
    if not (col.endswith('target_pnl') or col.endswith('future_option_price') 
            or col == 's_target_pnl' or col.endswith('contractID'))
]

if len(filtered_state_cols) != len(raw_state_cols):
    keep_indices = [raw_state_cols.index(col) for col in filtered_state_cols]
    scaler_mean = [meta['scaler_mean'][idx] for idx in keep_indices]
    scaler_scale = [meta['scaler_scale'][idx] for idx in keep_indices]
else:
    scaler_mean = meta['scaler_mean']
    scaler_scale = meta['scaler_scale']

states = _standardise_states(df, filtered_state_cols, scaler_mean, scaler_scale)
algo = _load_policy_robust(Path('iql_out/2023_training_2022models/discrete_cql_policy.d3'), meta)
predicted_actions = algo.predict(states)

# Run walkforward
print('🚀 Running Trial #100 walkforward...\n')
results_df, summary = simulate_walkforward(
    df, predicted_actions, meta['action_map'], 
    params, initial_capital=10000, mode="backtest"
)

# Save results
Path('results/trial_100_walkforward').mkdir(parents=True, exist_ok=True)
results_df.to_csv('results/trial_100_walkforward/trades.csv', index=False)
with open('results/trial_100_walkforward/summary.json', 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print('=' * 100)
print('🎊 TRIAL #100 WALKFORWARD RESULTS')
print('=' * 100)

print(f'\n📊 Performance:')
print(f'  Win Rate: {summary["win_rate"]:.1%}')
print(f'  Total Return: {summary["return_pct"]:.1f}%')
print(f'  Max Drawdown: {summary["max_drawdown"]:.1%}')
print(f'  Calmar Ratio: {summary["calmar_ratio"]:.1f}')

print(f'\n💰 Trading:')
print(f'  Total Trades: {summary["total_trades"]}')
print(f'  Winning: {summary["winning_trades"]}')
print(f'  Losing: {summary["losing_trades"]}')
print(f'  Halted: {summary["halted_trades"]}')
print(f'  Skipped: {summary["skipped_trades"]}')

print(f'\n💵 Financial:')
print(f'  Initial: ${summary["initial_capital"]:,.2f}')
print(f'  Final: ${summary["final_capital"]:,.2f}')
print(f'  Total P&L: ${summary["total_pnl"]:,.2f}')
if summary.get("avg_trade_pnl"):
    print(f'  Avg Trade: ${summary["avg_trade_pnl"]:,.0f}')
    print(f'  Largest Win: ${summary.get("largest_win", 0):,.0f}')
    print(f'  Largest Loss: ${summary.get("largest_loss", 0):,.0f}')
    print(f'  Profit Factor: {summary.get("profit_factor", 0):.2f}')

# Compare to targets
print(f'\n🎯 vs TARGETS:')
wr_pass = "✅" if summary["win_rate"] >= 0.86 else "❌"
dd_pass = "✅" if summary["max_drawdown"] < 0.15 else "❌"
print(f'  Win Rate: {summary["win_rate"]:.1%} (target: ≥86%) {wr_pass}')
print(f'  Drawdown: {summary["max_drawdown"]:.1%} (target: <15%) {dd_pass}')

if summary["win_rate"] >= 0.86 and summary["max_drawdown"] < 0.15:
    print(f'\n🏆 BOTH TARGETS ACHIEVED!')
elif summary["win_rate"] >= 0.86:
    print(f'\n✅ WIN RATE TARGET ACHIEVED!')
    print(f'❌ Drawdown: {summary["max_drawdown"]:.1%} (need {(summary["max_drawdown"] - 0.15)*100:.1f}pp reduction)')
else:
    print(f'\n❌ Targets not met')

print(f'\n💾 Results saved to: results/trial_100_walkforward/')
