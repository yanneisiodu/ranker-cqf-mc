import pandas as pd
import numpy as np
import joblib
import logging
import os
import time
import argparse
from sklearn.metrics import ndcg_score
import sys
from scipy.stats import spearmanr
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_config, preprocess_data, get_group_info
from logger import setup_logger

# --- Configuration & Constants ---
logger = setup_logger(__name__, level=logging.INFO)

TARGET_LOOKAHEAD_DAYS = 5 # Must match the training configuration
K_VALUES = [1, 5, 10, 20] # K values for metric calculation

# --- Helper Functions for Metrics ---
def calculate_precision_at_k(y_true_group, y_pred_group_scores, k):
    """Calculates Precision@k for a single group."""
    if k == 0:
        return 0.0
    y_true_arr = np.asarray(y_true_group)
    top_k_indices = np.argsort(y_pred_group_scores)[::-1][:k]
    top_k_true_relevance = y_true_arr[top_k_indices]
    num_relevant_in_top_k = np.sum(top_k_true_relevance > 0) # Assuming relevance > 0 means relevant
    return num_relevant_in_top_k / k

def calculate_average_precision_at_k(y_true_group, y_pred_group_scores, k):
    """Calculates Average Precision@k (AP@k) for a single group."""
    if k == 0:
        return 0.0
    
    y_true_arr = np.asarray(y_true_group)
    sorted_indices = np.argsort(y_pred_group_scores)[::-1]
    
    ap_sum = 0.0
    relevant_hits = 0
    num_items_to_consider = min(k, len(y_true_arr))

    for i in range(num_items_to_consider):
        idx = sorted_indices[i]
        if y_true_arr[idx] > 0: # Item at rank i+1 is relevant
            relevant_hits += 1
            precision_at_i_plus_1 = relevant_hits / (i + 1.0)
            ap_sum += precision_at_i_plus_1
            
    if relevant_hits == 0:
        return 0.0
        
    return ap_sum / relevant_hits

def dcg_at_k(gains_sorted_by_pred, k):
    return sum(g / np.log2(i + 2) for i, g in enumerate(gains_sorted_by_pred[:k]))

def ndcg_at_k_manual(y_true_labels, y_pred_scores, k):
    # Exponential gains to emphasize top ranks
    gains = (2 ** y_true_labels) - 1
    order_pred = np.argsort(y_pred_scores)[::-1]
    gains_pred = gains[order_pred]
    dcg = dcg_at_k(gains_pred, k)
    ideal_gains = np.sort(gains)[::-1]
    idcg = dcg_at_k(ideal_gains, k)
    return dcg / max(idcg, 1e-12)

# --- Helper Functions from Training Script (Adapted) ---
def load_raw_data_eval(file_path):
    t_start = time.time()
    logger.info(f"Loading raw evaluation data from: {file_path}")
    try:
        df = pd.read_csv(file_path, low_memory=False)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        logger.info(f"Loaded {len(df)} raw rows. Shape: {df.shape}. Date range: {df['date'].min()} to {df['date'].max()}")
        if 'contractID' not in df.columns:
             if 'contract_id' in df.columns: df = df.rename(columns={'contract_id': 'contractID'})
             elif 'option_symbol' in df.columns: df = df.rename(columns={'option_symbol': 'contractID'})
             else: raise ValueError("Missing 'contractID' or equivalent column for grouping.")
        df['contractID'] = df['contractID'].astype(str)
        df = df.sort_values(by=['date', 'contractID']).reset_index(drop=True)
        t_end = time.time()
        logger.info(f"Raw data loading finished in {t_end - t_start:.2f} seconds.")
        return df
    except FileNotFoundError:
        logger.error(f"Error: Input data file not found at {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading raw data from {file_path}: {e}", exc_info=True)
        raise

def calculate_target_eval(df, lookahead_days=TARGET_LOOKAHEAD_DAYS):
    t_start = time.time()
    logger.info(f"Calculating target variable (Sharpe Ratio over {lookahead_days} days) on dataframe with shape {df.shape}...")
    # Prefer raw price if available (Critical Fix 1)
    price_col = 'last_raw' if 'last_raw' in df.columns else 'last'
    if 'contractID' not in df.columns or price_col not in df.columns:
        logger.error("Cannot calculate target: 'contractID' or price column missing after preprocessing.")
        return df.assign(target_5d_sharpe=np.nan)
    df = df.sort_values(by=['contractID', 'date'])
    df_grouped = df.groupby('contractID')
    future_prices_dict = {}
    for i in range(1, lookahead_days + 1):
        future_prices_dict[f'price_d{i}'] = df_grouped[price_col].shift(-i)
    df_future = pd.DataFrame(future_prices_dict, index=df.index)
    daily_returns = pd.DataFrame(index=df.index)
    last_day_price = df[price_col]
    for i in range(1, lookahead_days + 1):
        current_day_price = df_future[f'price_d{i}']
        daily_returns[f'ret_d{i}'] = np.where(
            (last_day_price > 0) & (~last_day_price.isna()) & (~current_day_price.isna()),
            (current_day_price - last_day_price) / last_day_price,
            np.nan
        )
        last_day_price = current_day_price
    mean_daily_ret = daily_returns.mean(axis=1, skipna=True)
    std_daily_ret = daily_returns.std(axis=1, skipna=True, ddof=1)
    epsilon = 1e-8
    df['target_5d_sharpe'] = np.where(
        (~mean_daily_ret.isna()) & (~std_daily_ret.isna()) & (std_daily_ret > epsilon),
        mean_daily_ret / (std_daily_ret + epsilon),
        np.nan
    )
    t_end = time.time()
    logger.info(f"Target variable (Sharpe) calculation finished in {t_end - t_start:.2f} seconds.")
    return df.sort_values(by='date')

# --- Main Evaluation Function ---
def evaluate_model(args):
    logger.info(f"--- Starting Model Evaluation ---")
    logger.info(f"Model pipeline: {args.model_file}")
    logger.info(f"Evaluation data: {args.eval_data_file}")
    logger.info(f"Config file: {args.config_file}")
    logger.info(f"Sharpe edges file: {args.sharpe_edges_file}")
    logger.info(f"Feature list file: {args.feature_list_file}")

    try:
        model_pipeline = joblib.load(args.model_file)
        logger.info(f"Loaded model pipeline successfully.")
        sharpe_edges = joblib.load(args.sharpe_edges_file)
        logger.info(f"Loaded Sharpe quantile edges successfully: {sharpe_edges}")
        final_features_to_use = joblib.load(args.feature_list_file)
        logger.info(f"Loaded feature list with {len(final_features_to_use)} features.")
    except Exception as e:
        logger.error(f"Error loading model or artifacts: {e}", exc_info=True)
        return

    config = load_config(args.config_file)
    if not config: 
        logger.error("Failed to load configuration. Exiting.")
        return

    logger.info("--- Loading and Preparing Evaluation Data ---")
    df_eval_raw = load_raw_data_eval(args.eval_data_file)
    if df_eval_raw.empty: return

    df_eval_processed_utils, _ = preprocess_data(df_eval_raw, config, scaler=None)
    if df_eval_processed_utils.empty: return
    logger.info(f"Shape after utils.preprocess_data: {df_eval_processed_utils.shape}")

    # No re-computation of price_change_1d/iv_change_1d here; rely on preprocess_data (Step 1 fix)
    df_eval_with_target = calculate_target_eval(df_eval_processed_utils, TARGET_LOOKAHEAD_DAYS)
    if df_eval_with_target.empty: return
    logger.info(f"Shape after target calculation: {df_eval_with_target.shape}")

    logger.info("Converting true Sharpe Ratio to Integer Relevance Levels...")
    target_col = 'target_5d_sharpe'
    q1, q2, q3 = sharpe_edges[1], sharpe_edges[2], sharpe_edges[3]
    conditions = [
        df_eval_with_target[target_col] <= q1,
        (df_eval_with_target[target_col] > q1) & (df_eval_with_target[target_col] <= q2),
        (df_eval_with_target[target_col] > q2) & (df_eval_with_target[target_col] <= q3),
        df_eval_with_target[target_col] > q3
    ]
    choices = [0, 1, 2, 3]
    df_eval_with_target['target_relevance_int'] = np.select(conditions, choices, default=np.nan)
    df_eval_with_target.dropna(subset=['target_relevance_int'], inplace=True)
    df_eval_with_target['target_relevance_int'] = df_eval_with_target['target_relevance_int'].astype(int)
    if df_eval_with_target.empty: return
    logger.info(f"Shape after creating true integer relevance: {df_eval_with_target.shape}")
    logger.info(f"True relevance value counts:\n{df_eval_with_target['target_relevance_int'].value_counts().sort_index().to_string()}")

    X_eval_data = df_eval_with_target[final_features_to_use]
    y_eval_true = df_eval_with_target['target_relevance_int']
    df_eval_for_groups = df_eval_with_target.sort_values('date')
    X_eval_data = X_eval_data.loc[df_eval_for_groups.index]
    y_eval_true = y_eval_true.loc[df_eval_for_groups.index]

    if X_eval_data.empty: return

    logger.info(f"--- Making Predictions using loaded model pipeline (on {len(X_eval_data)} samples) ---")
    try:
        y_pred_scores = model_pipeline.predict(X_eval_data)
        logger.info(f"Predictions made successfully. Shape: {y_pred_scores.shape}")
    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        if X_eval_data.isnull().values.any():
            logger.error("X_eval_data contains NaNs before pipeline.predict:")
            logger.error(X_eval_data.isnull().sum()[X_eval_data.isnull().sum() > 0].to_string())
        if np.isinf(X_eval_data.select_dtypes(include=np.number)).values.any():
            logger.error("X_eval_data contains Infs before pipeline.predict")
        return

    logger.info(f"--- Calculating Metrics ---")
    eval_group_info = get_group_info(df_eval_for_groups)
    # Alignment assertion to prevent inflated metrics due to misaligned arrays
    total_rows = sum(eval_group_info)
    assert total_rows == len(y_pred_scores) == len(y_eval_true), (
        f"Alignment error: groups sum={total_rows}, preds={len(y_pred_scores)}, labels={len(y_eval_true)}"
    )

    ndcg_scores_by_k = {k: [] for k in K_VALUES}
    ndcg_exp_scores_by_k = {k: [] for k in K_VALUES}  # exponential-gain NDCG (sklearn)
    ndcg_manual_scores_by_k = {k: [] for k in K_VALUES}  # exponential-gain NDCG (manual)
    precision_scores_by_k = {k: [] for k in K_VALUES}
    ap_scores_by_k = {k: [] for k in K_VALUES}

    current_pos = 0
    top1_hits = 0
    group_counter = 0
    for group_size in eval_group_info:
        if group_size == 0: continue
        y_true_group = y_eval_true.iloc[current_pos : current_pos + group_size].values
        y_pred_group_scores = y_pred_scores[current_pos : current_pos + group_size]
        current_pos += group_size
        if len(y_true_group) < 1: continue
        group_counter += 1

        for k_val in K_VALUES:
            actual_k = min(k_val, len(y_true_group))
            if actual_k <= 0: continue
            
            try:
                ndcg_sc = ndcg_score(np.asarray([y_true_group]), np.asarray([y_pred_group_scores]), k=actual_k)
                ndcg_scores_by_k[k_val].append(ndcg_sc)
            except ValueError as ve:
                logger.warning(f"NDCG@{actual_k} error for group: {ve}")
                ndcg_scores_by_k[k_val].append(np.nan)

            # Exponential-gain NDCG: gains = 2^label - 1
            try:
                y_true_gain = (2 ** y_true_group) - 1
                ndcg_exp_sc = ndcg_score(np.asarray([y_true_gain]), np.asarray([y_pred_group_scores]), k=actual_k)
                ndcg_exp_scores_by_k[k_val].append(ndcg_exp_sc)
            except ValueError as ve:
                logger.warning(f"NDCG(exp gains)@{actual_k} error for group: {ve}")
                ndcg_exp_scores_by_k[k_val].append(np.nan)

            # Manual NDCG(exp gains) cross-check to prevent evaluation bugs
            try:
                ndcg_manual_sc = ndcg_at_k_manual(y_true_group, y_pred_group_scores, actual_k)
                ndcg_manual_scores_by_k[k_val].append(ndcg_manual_sc)
            except Exception as e:
                logger.warning(f"Manual NDCG@{actual_k} error for group: {e}")
                ndcg_manual_scores_by_k[k_val].append(np.nan)

            prec_sc = calculate_precision_at_k(y_true_group, y_pred_group_scores, actual_k)
            precision_scores_by_k[k_val].append(prec_sc)

            ap_sc = calculate_average_precision_at_k(y_true_group, y_pred_group_scores, actual_k)
            ap_scores_by_k[k_val].append(ap_sc)

        # Top-1 reality check
        try:
            gains = (2 ** y_true_group) - 1
            idx_true = int(np.argmax(gains))
            idx_pred = int(np.argmax(y_pred_group_scores))
            if idx_true == idx_pred:
                top1_hits += 1
        except Exception:
            pass
    
    logger.info(f"=== Evaluation Complete ===")
    logger.info(f"Number of groups (days) evaluated: {len(eval_group_info)}")

    # Group size stats
    if eval_group_info:
        import statistics as stats
        logger.info(f"Group size stats — min/median/max: {min(eval_group_info)}/{int(stats.median(eval_group_info))}/{max(eval_group_info)}")

    # Global Spearman between scores and continuous Sharpe target (sanity check)
    try:
        y_true_continuous = df_eval_for_groups['target_5d_sharpe'].loc[X_eval_data.index].values
        spearman_corr, _ = spearmanr(y_true_continuous, y_pred_scores)
        logger.info(f"Spearman(score, continuous Sharpe): {spearman_corr:.4f}")
    except Exception as e:
        logger.warning(f"Could not compute Spearman with continuous target: {e}")

    if group_counter > 0:
        logger.info(f"Top-1 reality check (fraction of days correct): {top1_hits / group_counter:.4f}")

    for k_val in K_VALUES:
        mean_ndcg = np.nanmean(ndcg_scores_by_k[k_val]) if ndcg_scores_by_k[k_val] else np.nan
        valid_ndcg_count = len([s for s in ndcg_scores_by_k[k_val] if not np.isnan(s)])
        logger.info(f"Mean NDCG@{k_val}: {mean_ndcg:.4f} (from {valid_ndcg_count} groups)")

        mean_ndcg_exp = np.nanmean(ndcg_exp_scores_by_k[k_val]) if ndcg_exp_scores_by_k[k_val] else np.nan
        logger.info(f"Mean NDCG(exp gains)@{k_val}: {mean_ndcg_exp:.4f}")

        mean_ndcg_manual = np.nanmean(ndcg_manual_scores_by_k[k_val]) if ndcg_manual_scores_by_k[k_val] else np.nan
        logger.info(f"Mean NDCG(manual exp gains)@{k_val}: {mean_ndcg_manual:.4f}")

        mean_precision = np.nanmean(precision_scores_by_k[k_val]) if precision_scores_by_k[k_val] else np.nan
        logger.info(f"Mean Precision@{k_val}: {mean_precision:.4f}")

        mean_ap = np.nanmean(ap_scores_by_k[k_val]) if ap_scores_by_k[k_val] else np.nan
        logger.info(f"Mean Average Precision (MAP)@{k_val}: {mean_ap:.4f}")
        logger.info("---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained XGBoost Ranking Model.")
    parser.add_argument("--model-file", type=str, required=True, help="Path to the trained .joblib model pipeline file.")
    parser.add_argument("--eval-data-file", type=str, required=True, help="Path to the CSV data file for evaluation.")
    parser.add_argument("--config-file", type=str, required=True, help="Path to the preprocessing config.yaml file.")
    parser.add_argument("--sharpe-edges-file", type=str, required=True, help="Path to the .pkl file containing Sharpe quantile edges.")
    parser.add_argument("--feature-list-file", type=str, required=True, help="Path to the .pkl file containing the list of feature names.")
    # Removed --ndcg-k as we use K_VALUES now

    args = parser.parse_args()
    evaluate_model(args) 