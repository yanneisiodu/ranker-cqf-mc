# Let's create a robust evaluation utility as a standalone Python module.
# It computes NDCG/Precision/MAP at K per group (e.g., per day), supports
# linear vs exponential gains, and includes alignment/sanity checks.
# It can be imported or run as a script on a CSV that contains columns:
#   group_col (e.g., 'date'), label_col (e.g., 'rank_label' or float percentile),
#   score_col (predicted scores), and optionally continuous_col (e.g., forward return)
# Usage example (programmatic):
#   from correct_evaluation import evaluate_grouped_ranking
#   metrics = evaluate_grouped_ranking(df, group_col='date', label_col='rank_label',
#                                      score_col='score', k_list=[1,5,10,20],
#                                      gain_scheme='exp2', positive_label_min=4)
#
# CLI example:
#   python /mnt/data/correct_evaluation.py --csv predictions.csv \
#     --group-col date --label-col rank_label --score-col score \
#     --k-list 1 5 10 20 --gain-scheme exp2 --positive-label-min 4

from typing import Dict, List, Optional, Tuple
import argparse
import numpy as np
import pandas as pd
from math import log2
from scipy.stats import spearmanr

def _clean_and_align(
    df: pd.DataFrame,
    group_col: str,
    label_col: str,
    score_col: str,
    continuous_col: Optional[str] = None,
) -> pd.DataFrame:
    """Drop rows with missing critical fields and ensure types are sane."""
    cols = [group_col, label_col, score_col]
    if continuous_col:
        cols.append(continuous_col)
    df2 = df.loc[:, cols].copy()
    # Drop NA rows
    df2 = df2.dropna(subset=[group_col, label_col, score_col])
    # Cast group to string to avoid subtle tz/dtype grouping issues
    df2[group_col] = df2[group_col].astype(str)
    # Ensure score is float
    df2[score_col] = df2[score_col].astype(float)
    # Labels can be int or float; keep as-is
    return df2

def _gains_from_labels(labels: np.ndarray, scheme: str = "exp2") -> np.ndarray:
    """Map labels to DCG gains."""
    if scheme == "exp2":
        # works best when labels are ordinal ints 0..L
        return (2.0 ** labels) - 1.0
    elif scheme == "linear":
        return labels.astype(float)
    elif scheme == "unit":
        # treat any nonzero label as 1
        return (labels.astype(float) > 0).astype(float)
    else:
        raise ValueError(f"Unknown gain scheme: {scheme}")

def _dcg_at_k(gains_sorted_by_pred: np.ndarray, k: Optional[int]) -> float:
    n = len(gains_sorted_by_pred)
    if n == 0:
        return 0.0
    kk = n if (k is None) else min(k, n)
    denom = np.log2(np.arange(2, 2 + kk))  # log2(2..k+1)
    return float(np.sum(gains_sorted_by_pred[:kk] / denom))

def _ndcg_at_k(labels: np.ndarray, scores: np.ndarray, k: Optional[int], gain_scheme: str) -> float:
    if labels.size == 0:
        return 0.0
    gains = _gains_from_labels(labels, scheme=gain_scheme)
    order = np.argsort(-scores, kind="mergesort")
    gains_pred = gains[order]
    dcg = _dcg_at_k(gains_pred, k)

    gains_ideal = np.sort(gains)[::-1]
    idcg = _dcg_at_k(gains_ideal, k)
    return 0.0 if idcg == 0 else float(dcg / idcg)

def _precision_at_k(labels: np.ndarray, scores: np.ndarray, k: int, positive_label_min: Optional[float]) -> float:
    """Binary precision@k: positives are labels >= positive_label_min.
       If positive_label_min is None, use top label value as threshold."""
    if labels.size == 0:
        return 0.0
    if positive_label_min is None:
        positive_label_min = float(np.max(labels))
    order = np.argsort(-scores, kind="mergesort")[: min(k, labels.size)]
    binary_relevance = (labels[order].astype(float) >= positive_label_min).astype(float)
    return float(np.mean(binary_relevance)) if binary_relevance.size > 0 else 0.0

def _average_precision_at_k(labels: np.ndarray, scores: np.ndarray, k: int, positive_label_min: Optional[float]) -> float:
    """AP@k requires binary relevance (labels >= threshold)."""
    if labels.size == 0:
        return 0.0
    if positive_label_min is None:
        positive_label_min = float(np.max(labels))
    order = np.argsort(-scores, kind="mergesort")[: min(k, labels.size)]
    rel = (labels[order].astype(float) >= positive_label_min).astype(float)
    if rel.sum() == 0:
        return 0.0
    precisions = []
    hits = 0
    for i, r in enumerate(rel, start=1):
        if r > 0:
            hits += 1
            precisions.append(hits / i)
    return float(np.mean(precisions)) if precisions else 0.0

def evaluate_grouped_ranking(
    df: pd.DataFrame,
    group_col: str,
    label_col: str,
    score_col: str,
    k_list: List[int] = [1, 5, 10, 20],
    gain_scheme: str = "exp2",
    positive_label_min: Optional[float] = None,
    continuous_col: Optional[str] = None,
) -> Dict[str, float]:
    """Compute per-group metrics and aggregate by simple mean across groups.
       Returns a dict of metrics + sanity stats."""
    df2 = _clean_and_align(df, group_col, label_col, score_col, continuous_col)
    # Basic alignment counts
    total_rows = len(df2)
    # Drop groups of size 0 (shouldn't happen) or 1 for rank metrics if you wish; we keep them but they have little effect
    grp = df2.groupby(group_col, sort=False)
    group_sizes = grp.size().to_numpy()
    results = {}

    # Per-group metrics
    ndcg_sums = {k: 0.0 for k in k_list}
    prec_sums = {k: 0.0 for k in k_list}
    map_sums = {k: 0.0 for k in k_list}
    n_groups = 0

    # Spearman collection
    spearman_vals = []
    spearman_cont_vals = []

    for gname, gdf in grp:
        labels = gdf[label_col].to_numpy()
        scores = gdf[score_col].to_numpy()

        # Skip all-NaN/empty groups
        if labels.size == 0:
            continue

        # NDCG/Precision/MAP per K
        for k in k_list:
            ndcg_sums[k] += _ndcg_at_k(labels, scores, k, gain_scheme)
            prec_sums[k] += _precision_at_k(labels, scores, k, positive_label_min)
            map_sums[k]  += _average_precision_at_k(labels, scores, k, positive_label_min)

        # Spearman on labels (ordinal/float). If constant, spearmanr returns nan; treat as 0.
        try:
            s = spearmanr(scores, labels, nan_policy="omit").correlation
        except Exception:
            s = np.nan
        if np.isnan(s):
            s = 0.0
        spearman_vals.append(float(s))

        # Optional: correlation vs continuous target (e.g., realized forward return)
        if continuous_col and continuous_col in gdf.columns:
            cont = gdf[continuous_col].to_numpy()
            try:
                sc = spearmanr(scores, cont, nan_policy="omit").correlation
            except Exception:
                sc = np.nan
            if np.isnan(sc):
                sc = 0.0
            spearman_cont_vals.append(float(sc))

        n_groups += 1

    if n_groups == 0:
        raise ValueError("No non-empty groups after cleaning; check your group_col and inputs.")

    # Aggregate means across groups
    for k in k_list:
        results[f"NDCG@{k}"] = ndcg_sums[k] / n_groups
        results[f"Precision@{k}"] = prec_sums[k] / n_groups
        results[f"MAP@{k}"] = map_sums[k] / n_groups

    # Spearman summaries
    results["Spearman(label)_median_per_group"] = float(np.median(spearman_vals)) if spearman_vals else 0.0
    results["Spearman(label)_mean_per_group"]   = float(np.mean(spearman_vals)) if spearman_vals else 0.0
    if spearman_cont_vals:
        results["Spearman(continuous)_median_per_group"] = float(np.median(spearman_cont_vals))
        results["Spearman(continuous)_mean_per_group"]   = float(np.mean(spearman_cont_vals))

    # Sanity stats
    results["num_groups"] = n_groups
    results["rows_used"] = int(total_rows)
    results["group_size_min"] = int(group_sizes.min()) if len(group_sizes) else 0
    results["group_size_median"] = float(np.median(group_sizes)) if len(group_sizes) else 0.0
    results["group_size_max"] = int(group_sizes.max()) if len(group_sizes) else 0

    return results

def _load_csv_safe(path: str) -> pd.DataFrame:
    # Robust CSV read with common options
    return pd.read_csv(path)

def _main():
    parser = argparse.ArgumentParser(description="Correct, leakage-safe grouped ranking evaluation.")
    parser.add_argument("--csv", type=str, required=True, help="CSV with predictions and labels.")
    parser.add_argument("--group-col", type=str, required=True, help="Grouping column (e.g., date).")
    parser.add_argument("--label-col", type=str, required=True, help="Label column (ordinal int or float).")
    parser.add_argument("--score-col", type=str, required=True, help="Predicted score column.")
    parser.add_argument("--continuous-col", type=str, default=None, help="Optional continuous target column (e.g., forward return).")
    parser.add_argument("--k-list", type=int, nargs="+", default=[1,5,10,20], help="List of K for @K metrics.")
    parser.add_argument("--gain-scheme", type=str, default="exp2", choices=["exp2","linear","unit"], help="DCG gain mapping.")
    parser.add_argument("--positive-label-min", type=float, default=None, help="Threshold for Precision/MAP positives (label >= threshold). Defaults to max label per group.")
    args = parser.parse_args()

    df = _load_csv_safe(args.csv)
    metrics = evaluate_grouped_ranking(
        df,
        group_col=args.group_col,
        label_col=args.label_col,
        score_col=args.score_col,
        k_list=args.k_list,
        gain_scheme=args.gain_scheme,
        positive_label_min=args.positive_label_min,
        continuous_col=args.continuous_col,
    )
    # Pretty print
    print("=== Evaluation (grouped) ===")
    for k in sorted([k for k in metrics.keys() if k.startswith("NDCG@")], key=lambda x: int(x.split("@")[1])):
        print(f"{k}: {metrics[k]:.4f}")
    for k in sorted([k for k in metrics.keys() if k.startswith("Precision@")], key=lambda x: int(x.split("@")[1])):
        print(f"{k}: {metrics[k]:.4f}")
    for k in sorted([k for k in metrics.keys() if k.startswith("MAP@")], key=lambda x: int(x.split("@")[1])):
        print(f"{k}: {metrics[k]:.4f}")
    if "Spearman(continuous)_median_per_group" in metrics:
        print(f"Spearman(score, continuous)_median_per_group: {metrics['Spearman(continuous)_median_per_group']:.4f}")
        print(f"Spearman(score, continuous)_mean_per_group:   {metrics['Spearman(continuous)_mean_per_group']:.4f}")
    print(f"Spearman(score, label)_median_per_group: {metrics['Spearman(label)_median_per_group']:.4f}")
    print(f"Spearman(score, label)_mean_per_group:   {metrics['Spearman(label)_mean_per_group']:.4f}")
    print(f"Groups: {metrics['num_groups']}, Rows used: {metrics['rows_used']}")
    print(f"Group size (min/median/max): {metrics['group_size_min']}/{metrics['group_size_median']}/{metrics['group_size_max']}")

if __name__ == "__main__":
    _main()
