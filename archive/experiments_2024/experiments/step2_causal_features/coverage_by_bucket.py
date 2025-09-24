#!/usr/bin/env python3
"""
Coverage-by-bucket analysis for regime validation.

Generates detailed coverage heatmaps by VIX×DTE buckets to validate
that adaptive conformal calibration is working across all market regimes.

Usage:
    python coverage_by_bucket.py predictions.csv --target target_pnl --q05 q0.05 --q95 q0.95
"""

import pandas as pd
import numpy as np
import argparse
from regime_tools import _make_bins_vix, _make_bins_dte

def analyze_coverage_by_bucket(df: pd.DataFrame, 
                              target_col: str,
                              q05_col: str, 
                              q95_col: str,
                              vix_col: str = 'vix_d_close',
                              dte_col: str = 'days_to_exp') -> pd.DataFrame:
    """
    Generate coverage heatmap by VIX×DTE buckets.
    
    Returns DataFrame with coverage statistics for each regime bucket.
    """
    
    # Extract required columns
    y_true = df[target_col].values
    q05_pred = df[q05_col].values
    q95_pred = df[q95_col].values
    
    # Create regime bins
    vix_bins = _make_bins_vix(df[vix_col]) if vix_col in df.columns else pd.Series([0] * len(df))
    dte_bins = _make_bins_dte(df[dte_col]) if dte_col in df.columns else pd.Series([0] * len(df))
    
    # Labels for reporting
    vix_labels = ['VIX<15', 'VIX 15-20', 'VIX 20-30', 'VIX>30']
    dte_labels = ['DTE≤7', 'DTE 8-21', 'DTE 22-45', 'DTE>45']
    
    # Coverage analysis
    results = []
    overall_coverage = np.mean((y_true >= q05_pred) & (y_true <= q95_pred))
    
    print(f"📊 Overall 90% Interval Coverage: {overall_coverage:.1%}\n")
    
    # Create coverage heatmap
    coverage_matrix = np.zeros((4, 4))
    sample_matrix = np.zeros((4, 4))
    
    for i in range(4):  # VIX bins
        for j in range(4):  # DTE bins
            mask = (vix_bins == i) & (dte_bins == j)
            n_samples = mask.sum()
            
            if n_samples > 20:  # Minimum for estimate
                coverage = np.mean((y_true[mask] >= q05_pred[mask]) & (y_true[mask] <= q95_pred[mask]))
                coverage_matrix[i, j] = coverage
                sample_matrix[i, j] = n_samples
                
                # Individual bucket reporting
                vix_label = vix_labels[i] if i < len(vix_labels) else f"VIX_bin_{i}"
                dte_label = dte_labels[j] if j < len(dte_labels) else f"DTE_bin_{j}"
                
                status = "✅" if 0.85 <= coverage <= 0.95 else "⚠️" if coverage >= 0.80 else "❌"
                
                results.append({
                    'vix_bin': i,
                    'dte_bin': j, 
                    'vix_label': vix_label,
                    'dte_label': dte_label,
                    'coverage': coverage,
                    'n_samples': n_samples,
                    'status': status
                })
    
    # Print coverage heatmap
    print("📊 Coverage Heatmap (VIX × DTE):")
    print("     " + "".join(f"{dte_labels[j]:>12}" for j in range(4)))
    
    for i in range(4):
        row_label = f"{vix_labels[i]:>8}"
        row_values = ""
        for j in range(4):
            if sample_matrix[i, j] > 20:
                coverage = coverage_matrix[i, j]
                status = "✅" if 0.85 <= coverage <= 0.95 else "⚠️" if coverage >= 0.80 else "❌"
                row_values += f"{status}{coverage:>7.1%}   "
            else:
                row_values += "    --     "
        print(row_label + " " + row_values)
    
    print("\n📊 Sample Count Heatmap (VIX × DTE):")
    print("     " + "".join(f"{dte_labels[j]:>12}" for j in range(4)))
    
    for i in range(4):
        row_label = f"{vix_labels[i]:>8}"
        row_values = ""
        for j in range(4):
            n = int(sample_matrix[i, j])
            if n > 0:
                row_values += f"{n:>10,} "
            else:
                row_values += "       -- "
        print(row_label + " " + row_values)
    
    # Summary statistics
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        critical_failures = len(results_df[results_df['coverage'] < 0.80])
        warning_buckets = len(results_df[(results_df['coverage'] >= 0.80) & (results_df['coverage'] < 0.85)])
        good_buckets = len(results_df[results_df['coverage'] >= 0.85])
        
        print(f"\n📈 Bucket Summary:")
        print(f"  ✅ Good coverage (85-95%): {good_buckets}")
        print(f"  ⚠️  Warning coverage (80-85%): {warning_buckets}")  
        print(f"  ❌ Critical failures (<80%): {critical_failures}")
        
        if critical_failures > 0:
            print(f"\n❌ Critical Under-Coverage Buckets:")
            critical = results_df[results_df['coverage'] < 0.80]
            for _, row in critical.iterrows():
                print(f"  {row['vix_label']} × {row['dte_label']}: {row['coverage']:.1%} (n={row['n_samples']:,})")
    
    return results_df

def main():
    parser = argparse.ArgumentParser(description="Analyze coverage by VIX×DTE regime buckets")
    parser.add_argument("predictions_file", help="CSV file with predictions and actuals")
    parser.add_argument("--target", default="target_pnl", help="True target column name")
    parser.add_argument("--q05", default="q0.05", help="5th percentile prediction column")
    parser.add_argument("--q95", default="q0.95", help="95th percentile prediction column")
    parser.add_argument("--vix", default="vix_d_close", help="VIX column name")
    parser.add_argument("--dte", default="days_to_exp", help="Days to expiry column name")
    parser.add_argument("--output", help="Optional CSV output path for results")
    
    args = parser.parse_args()
    
    # Load predictions
    print(f"📂 Loading predictions from: {args.predictions_file}")
    df = pd.read_csv(args.predictions_file)
    
    # Check required columns
    required_cols = [args.target, args.q05, args.q95]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        return 1
    
    print(f"✅ Loaded {len(df):,} predictions")
    
    # Run analysis
    results_df = analyze_coverage_by_bucket(df, args.target, args.q05, args.q95, args.vix, args.dte)
    
    # Save results if requested
    if args.output and not results_df.empty:
        results_df.to_csv(args.output, index=False)
        print(f"💾 Results saved to: {args.output}")
    
    return 0

if __name__ == "__main__":
    exit(main())
