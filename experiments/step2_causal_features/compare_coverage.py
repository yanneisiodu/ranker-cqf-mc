#!/usr/bin/env python3
"""
Quick comparison of coverage statistics across different 2022→2023 CQF runs
"""

import pandas as pd
import numpy as np

def calculate_coverage_stats(pred_file, model_name):
    """Calculate coverage statistics for a prediction file"""
    print(f"\n🔍 Analyzing: {model_name}")
    
    df = pd.read_csv(pred_file)
    n = len(df)
    
    # Calculate coverage rates
    q05_coverage = ((df['target_actual'] >= df['q0.05']).sum() / n) * 100
    q95_coverage = ((df['target_actual'] <= df['q0.95']).sum() / n) * 100
    interval_coverage = (
        ((df['target_actual'] >= df['q0.05']) & 
         (df['target_actual'] <= df['q0.95'])).sum() / n
    ) * 100
    
    # Calculate interval characteristics
    interval_width = (df['q0.95'] - df['q0.05']).mean()
    q50_bias = (df['q0.50'] - df['target_actual']).mean()
    
    # Basic stats
    mean_target = df['target_actual'].mean()
    std_target = df['target_actual'].std()
    
    print(f"📊 Dataset: {n:,} predictions")
    print(f"📊 Target: μ={mean_target:.4f}, σ={std_target:.4f}")
    print(f"📊 Coverage:")
    print(f"   q0.05: {q05_coverage:.1f}% (target: 95.0%)")
    print(f"   q0.95: {q95_coverage:.1f}% (target: 95.0%)")
    print(f"   90% interval: {interval_coverage:.1f}% (target: 90.0%)")
    print(f"📊 Interval width: {interval_width:.2f}")
    print(f"📊 q0.50 bias: {q50_bias:.4f}")
    
    # Coverage errors
    q05_error = abs(q05_coverage - 95.0)
    q95_error = abs(q95_coverage - 95.0)
    interval_error = abs(interval_coverage - 90.0)
    max_error = max(q05_error, q95_error, interval_error)
    
    print(f"📊 Coverage errors:")
    print(f"   q0.05 error: {q05_error:.1f}%")
    print(f"   q0.95 error: {q95_error:.1f}%")
    print(f"   interval error: {interval_error:.1f}%")
    print(f"   max error: {max_error:.1f}%")
    
    # Quality gate (max error <= 7.5%)
    quality_gate = "✅ PASS" if max_error <= 7.5 else "❌ FAIL"
    print(f"🚨 Quality Gate: {quality_gate}")
    
    return {
        'model': model_name,
        'n': n,
        'q05_coverage': q05_coverage,
        'q95_coverage': q95_coverage,
        'interval_coverage': interval_coverage,
        'interval_width': interval_width,
        'q50_bias': q50_bias,
        'max_error': max_error,
        'quality_gate': max_error <= 7.5
    }

# Models to compare
models = [
    ('Archive/model_output/optimal_cqf_2022_to_2023_predictions.csv', 'Optimal CQF'),
    ('Archive/model_output/fixed_cqf_2022_to_2023_predictions.csv', 'Fixed CQF'), 
    ('Archive/model_output/leak_free_cqf_2022_to_2023_predictions.csv', 'Leak-Free CQF'),
    ('model_output/cqf_regime_test_predictions.csv', 'Regime-Adaptive CQF'),
    ('model_output/cqf_regime_tuned_predictions.csv', 'VIX-Adaptive CQF'),
    ('model_output/cqf_stable_period_predictions.csv', 'Stable-Period CQF'),
    ('model_output/cqf_simple_baseline_predictions.csv', 'Simple CQF')
]

results = []
for pred_file, model_name in models:
    try:
        result = calculate_coverage_stats(pred_file, model_name)
        results.append(result)
    except Exception as e:
        print(f"❌ Error analyzing {model_name}: {e}")

# Summary comparison
if results:
    print(f"\n📋 COMPARISON SUMMARY (2022 Train → 2023 Eval)")
    print("="*80)
    print(f"{'Model':<20} {'90% Coverage':<12} {'Max Error':<10} {'Width':<8} {'Gate':<6}")
    print("-"*80)
    
    for r in results:
        gate_symbol = "✅" if r['quality_gate'] else "❌"
        print(f"{r['model']:<20} {r['interval_coverage']:>8.1f}%    {r['max_error']:>6.1f}%   {r['interval_width']:>6.2f}  {gate_symbol}")
    
    print("\n🏆 Best performers:")
    best_coverage = max(results, key=lambda x: abs(x['interval_coverage'] - 90))
    best_error = min(results, key=lambda x: x['max_error'])
    best_width = min(results, key=lambda x: x['interval_width'])
    
    print(f"   Coverage: {best_coverage['model']} ({best_coverage['interval_coverage']:.1f}%)")
    print(f"   Min Error: {best_error['model']} ({best_error['max_error']:.1f}%)")
    print(f"   Tightest: {best_width['model']} ({best_width['interval_width']:.2f})")
