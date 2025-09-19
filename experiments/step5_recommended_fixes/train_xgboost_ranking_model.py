import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import ndcg_score
import joblib
import logging
import os
import time
import optuna
from functools import partial
from sklearn.impute import SimpleImputer
import sys
from datetime import datetime
import argparse
import json

# Import necessary functions from utils
# utils.py is expected to be in the same directory as this script
from utils import load_config, preprocess_data

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Main config file (used by preprocess_data)
CONFIG_FILE = "./config.yaml"

def set_determinism_and_resource_control(config):
    """Set environment variables for determinism and resource control."""
    seeds = config.get('seeds', {'python': 42, 'numpy': 42, 'xgboost': 42})
    threads = config.get('threads', {'OMP_NUM_THREADS': 1, 'MKL_NUM_THREADS': 1})
    
    # Set environment variables for determinism
    os.environ['PYTHONHASHSEED'] = str(seeds.get('python', 42))
    os.environ['OMP_NUM_THREADS'] = str(threads.get('OMP_NUM_THREADS', 1))
    os.environ['MKL_NUM_THREADS'] = str(threads.get('MKL_NUM_THREADS', 1))
    
    # Set numpy seed
    np.random.seed(seeds.get('numpy', 42))
    
    logging.info(f"Set determinism: PYTHONHASHSEED={os.environ['PYTHONHASHSEED']}, "
                 f"OMP_NUM_THREADS={os.environ['OMP_NUM_THREADS']}, "
                 f"MKL_NUM_THREADS={os.environ['MKL_NUM_THREADS']}, "
                 f"numpy_seed={seeds.get('numpy', 42)}")
    
    return seeds

def get_versions_metadata():
    """Get package versions for reproducibility."""
    import numpy, pandas, sklearn, xgboost
    
    versions = {
        'numpy': numpy.__version__,
        'pandas': pandas.__version__,
        'scikit_learn': sklearn.__version__,
        'xgboost': xgboost.__version__
    }
    return versions

def assert_group_contiguity(df, group_info):
    """Assert that group information is contiguous and matches data."""
    # Sort by date to ensure order
    df_sorted = df.sort_values('date', kind='mergesort')
    
    # Check that group sizes sum to total rows
    total_samples = sum(group_info)
    actual_samples = len(df_sorted)
    
    if total_samples != actual_samples:
        raise ValueError(f"Group size mismatch: sum(group_info)={total_samples} != len(df)={actual_samples}")
    
    # Check that dates are contiguous within the sorted order
    unique_dates = df_sorted['date'].unique()
    calculated_group_sizes = df_sorted.groupby('date').size().values.tolist()
    
    if calculated_group_sizes != group_info:
        raise ValueError(f"Group contiguity error: calculated sizes {calculated_group_sizes[:5]}... "
                        f"!= provided group_info {group_info[:5]}...")
    
    logging.info(f"Group contiguity verified: {len(unique_dates)} days, {total_samples} samples")

def drop_weak_days(df, config):
    """Drop days with insufficient data quality."""
    min_rows_per_day = config.get('min_rows_per_day', 200)
    min_target_coverage = config.get('min_target_coverage', 0.80)
    
    initial_rows = len(df)
    initial_days = df['date'].nunique()
    
    # Calculate per-day statistics
    daily_stats = df.groupby('date').agg({
        'target_5d_sharpe': ['count', 'size'],  # count non-NaN, size total
    }).round(4)
    daily_stats.columns = ['target_count', 'total_rows']
    daily_stats['coverage'] = daily_stats['target_count'] / daily_stats['total_rows']
    
    # Apply filters
    good_days_mask = (
        (daily_stats['total_rows'] >= min_rows_per_day) & 
        (daily_stats['coverage'] >= min_target_coverage)
    )
    
    good_days = daily_stats[good_days_mask].index
    bad_days = daily_stats[~good_days_mask].index
    
    if len(bad_days) > 0:
        logging.info(f"Dropping {len(bad_days)} weak days (min_rows={min_rows_per_day}, "
                    f"min_coverage={min_target_coverage}): {list(bad_days)[:5]}...")
        for day in bad_days[:3]:  # Log details for first few bad days
            stats = daily_stats.loc[day]
            logging.info(f"  {day}: {stats['total_rows']} rows, {stats['coverage']:.3f} coverage")
    
    # Filter to good days
    df_filtered = df[df['date'].isin(good_days)].copy()
    
    final_rows = len(df_filtered)
    final_days = df_filtered['date'].nunique()
    
    logging.info(f"Health check results: kept {final_days}/{initial_days} days, "
                f"{final_rows}/{initial_rows} rows "
                f"({(final_rows/initial_rows)*100:.1f}% retention)")
    
    return df_filtered

# Data Paths
INPUT_DATA_FILES = [
    "./year_2019_data.csv",
    "./year_2020_data.csv",
    "./year_2021_data.csv",
    "./year_2022_data.csv",
    "./year_2023_data.csv",
    "./year_2024_data.csv",
    "./year_2025_data.csv",
]
MODEL_OUTPUT_PATH = "./model_output/" # Output will be in xgboost_models/model_output/
# Model name indicating training data and Optuna tuning (example, actual name is dynamic)
MODEL_FILE = os.path.join(MODEL_OUTPUT_PATH, "xgboost_ranker_2019_2025_optuna_tuned.joblib")

# Model & Training Settings
TARGET_LOOKAHEAD_DAYS = 5
N_CV_SPLITS = 5
OPTUNA_N_TRIALS = 100
OPTUNA_TIMEOUT = None
NDCG_K = 20

NUMERICAL_FEATURES = [
    'days_to_exp', 'strike', 'last', 'bid', 'ask', 'volume', 'open_interest',
    'implied_volatility', 'delta', 'gamma', 'theta', 'vega', 'rho',
    'spy_d_close', 'spy_d_SMA_50', 'spy_d_RSI', 'spy_d_MACD_Hist',
    'vix_d_close',
    'moneyness', 'relative_spread', 'bid_ask_spread',
    'ofi',
    'price_change_1d', 'iv_change_1d',
    'zero_day_premium',
    'option_volume_oi_ratio',
    'mispricing_ratio',
    'risk_adjusted_signal',
    'iv_vix_ratio',
    'spy_momentum',
    'price__mean', 'price__standard_deviation',
]
CATEGORICAL_FEATURES = ['type']

FIXED_PARAMS = {
    'learning_rate'   : 0.06326509803163745,
    'max_depth'       : 3,
    'subsample'       : 0.7822444357631866,
    'colsample_bytree': 0.7333407813790271,
    'gamma'           : 0.14988133139247198,
    'reg_alpha'       : 1.0693728241301906e-06,
    'reg_lambda'      : 1.0903140623130713e-07
}

def load_raw_data(file_path):
    t_start = time.time()
    logging.info(f"Loading raw data from: {file_path}")
    try:
        df = pd.read_csv(file_path, low_memory=False)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        logging.info(f"Loaded {len(df)} raw rows. Shape: {df.shape}. Date range: {df['date'].min()} to {df['date'].max()}")
        if 'contractID' not in df.columns:
             if 'contract_id' in df.columns: df = df.rename(columns={'contract_id': 'contractID'})
             elif 'option_symbol' in df.columns: df = df.rename(columns={'option_symbol': 'contractID'})
             else: raise ValueError("Missing 'contractID' or equivalent column for grouping.")
        df['contractID'] = df['contractID'].astype(str)
        df = df.sort_values(by=['date', 'contractID']).reset_index(drop=True)
        t_end = time.time()
        logging.info(f"Raw data loading finished in {t_end - t_start:.2f} seconds.")
        return df
    except FileNotFoundError:
        logging.error(f"Error: Input data file not found at {file_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading raw data from {file_path}: {e}", exc_info=True)
        raise

def calculate_target(df, lookahead_days=5):
    t_start = time.time()
    logging.info(f"Calculating target variable (Sharpe Ratio over {lookahead_days} days) on dataframe with shape {df.shape}...")
    
    # Use raw column if available, otherwise fallback to scaled column
    price_col = 'last_raw' if 'last_raw' in df.columns else 'last'
    
    if 'contractID' not in df.columns or price_col not in df.columns:
        logging.error(f"Cannot calculate target: 'contractID' or '{price_col}' column missing after preprocessing.")
        return df.assign(target_5d_sharpe=np.nan)
    
    logging.info(f"Using column '{price_col}' for target calculation")
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
    initial_rows = len(df)
    df = df.dropna(subset=['target_5d_sharpe'])
    rows_dropped = initial_rows - len(df)
    if rows_dropped > 0:
        logging.info(f"Dropped {rows_dropped} rows due to NaN target (Sharpe) after calculation. Final shape: {df.shape}")
    else:
        logging.info(f"No rows dropped during target (Sharpe) calculation. Final shape: {df.shape}")
    t_end = time.time()
    logging.info(f"Target variable (Sharpe) calculation finished in {t_end - t_start:.2f} seconds.")
    return df.sort_values(by='date')

def create_preprocessor(numerical_features, categorical_features, available_columns):
    logging.debug(f"Creating preprocessor with Imputer -> Scaler...")
    transformers = []
    num_features_exist = [f for f in numerical_features if f in available_columns]
    cat_features_exist = [f for f in categorical_features if f in available_columns]
    if num_features_exist:
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        transformers.append(('num', numerical_pipeline, num_features_exist))
        logging.debug(f"Preprocessor applying Imputer(mean) -> StandardScaler to {len(num_features_exist)} numerical features.")
    else:
        logging.warning("No numerical features found for Imputer/StandardScaler.")
    if cat_features_exist:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features_exist))
        logging.debug(f"Preprocessor applying OneHotEncoder to {len(cat_features_exist)} categorical features.")
    else:
        logging.warning("No categorical features found for OneHotEncoder.")
    if not transformers:
        raise ValueError("No numerical or categorical features found to preprocess.")
    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
    logging.debug("Preprocessor created.")
    return preprocessor

def get_group_info(df):
    df_sorted = df.sort_values('date', kind='mergesort')
    group_info = df_sorted.groupby('date').size().tolist()
    
    # Assert group contiguity 
    assert_group_contiguity(df_sorted, group_info)
    
    logging.debug(f"Calculated group info for {len(group_info)} groups (days). Total samples: {sum(group_info)}.")
    return group_info

def split_train_val_by_recent_days(X, y, val_days=10):
    """Split data into train/val using recent days for validation."""
    X_sorted = X.sort_values('date', kind='mergesort')
    y_sorted = y.loc[X_sorted.index]
    
    unique_dates = sorted(X_sorted['date'].unique())
    if len(unique_dates) <= val_days:
        raise ValueError(f"Not enough unique dates ({len(unique_dates)}) for val_days={val_days}")
    
    # Use last val_days for validation
    val_dates = unique_dates[-val_days:]
    train_dates = unique_dates[:-val_days]
    
    train_mask = X_sorted['date'].isin(train_dates)
    val_mask = X_sorted['date'].isin(val_dates)
    
    X_train = X_sorted[train_mask]
    X_val = X_sorted[val_mask]
    y_train = y_sorted[train_mask]
    y_val = y_sorted[val_mask]
    
    # Verify no date overlap
    train_date_set = set(X_train['date'].unique())
    val_date_set = set(X_val['date'].unique())
    if train_date_set & val_date_set:
        raise ValueError(f"Date overlap between train and val: {train_date_set & val_date_set}")
    
    logging.info(f"Split: {len(train_dates)} train days ({len(X_train)} rows), "
                f"{len(val_dates)} val days ({len(X_val)} rows)")
    
    return X_train, X_val, y_train, y_val

def project_features_consistently(X, selected_features):
    """Project X to selected_features, adding missing cols with zeros, dropping extras."""
    missing_features = [f for f in selected_features if f not in X.columns and f != 'date']
    extra_features = [f for f in X.columns if f not in selected_features and f != 'date']
    
    if missing_features:
        logging.info(f"Adding {len(missing_features)} missing features with zeros: {missing_features[:5]}")
        for feat in missing_features:
            X[feat] = 0.0
    
    if extra_features:
        logging.info(f"Dropping {len(extra_features)} extra features: {extra_features[:5]}")
    
    # Reindex to selected features (keep date column if present)
    cols_to_keep = [f for f in selected_features if f in X.columns]
    if 'date' in X.columns:
        cols_to_keep = ['date'] + cols_to_keep
    
    return X[cols_to_keep]

def objective_rank(trial, X, y, n_splits, numerical_features, categorical_features, k_ndcg):
    params = {
        'objective': 'rank:ndcg',
        'eval_metric': f'ndcg@{k_ndcg}',
        'tree_method': 'hist',
        'seed': 42,
        'n_jobs': -1,
        'n_estimators': 1000,
        'early_stopping_rounds': 50,
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'gamma': trial.suggest_float('gamma', 1e-9, 1.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-9, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-9, 1.0, log=True),
    }
    tss = TimeSeriesSplit(n_splits=n_splits)
    fold_ndcg_scores = []
    logging.debug(f"Trial {trial.number}: Starting CV with params: {params}")
    fold_counter = 0
    available_features_in_X = [f for f in X.columns if f != 'date']
    numerical_features_in_fold = [f for f in numerical_features if f in available_features_in_X]
    categorical_features_in_fold = [f for f in categorical_features if f in available_features_in_X]
    features_to_use_in_fold = numerical_features_in_fold + categorical_features_in_fold
    for train_index, val_index in tss.split(X):
        fold_counter += 1
        X_train_fold = X.iloc[train_index][features_to_use_in_fold + ['date']]
        X_val_fold = X.iloc[val_index][features_to_use_in_fold + ['date']]
        y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]
        X_train_fold = X_train_fold.sort_values('date')
        y_train_fold = y_train_fold.loc[X_train_fold.index]
        X_val_fold = X_val_fold.sort_values('date')
        y_val_fold = y_val_fold.loc[X_val_fold.index]
        train_group_info = get_group_info(X_train_fold)
        val_group_info = get_group_info(X_val_fold)
        try:
            logging.debug(f"Trial {trial.number}, Fold {fold_counter}: Creating preprocessor for {len(features_to_use_in_fold)} features.")
            preprocessor = create_preprocessor(numerical_features_in_fold, categorical_features_in_fold, features_to_use_in_fold)
            model = xgb.XGBRanker(**params)
            pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('ranker', model)])
            logging.debug(f"Trial {trial.number}, Fold {fold_counter}: Fitting preprocessor...")
            pipeline.named_steps['preprocessor'].fit(X_train_fold[features_to_use_in_fold])
            logging.debug(f"Trial {trial.number}, Fold {fold_counter}: Transforming data...")
            X_train_processed = pipeline.named_steps['preprocessor'].transform(X_train_fold[features_to_use_in_fold])
            X_val_processed = pipeline.named_steps['preprocessor'].transform(X_val_fold[features_to_use_in_fold])
            eval_set = [(X_val_processed, y_val_fold)]
            logging.debug(f"Trial {trial.number}, Fold {fold_counter}: Fitting XGBoost ranker...")
            pipeline.named_steps['ranker'].fit(
                X_train_processed,
                y_train_fold,
                group=train_group_info,
                eval_set=eval_set,
                eval_group=[val_group_info],
                verbose=False
            )
            results = pipeline.named_steps['ranker'].evals_result()
            ndcg_at_k = results['validation_0'][f'ndcg@{k_ndcg}'][-1]
            fold_ndcg_scores.append(ndcg_at_k)
            logging.debug(f"Trial {trial.number}, Fold {fold_counter}: NDCG@{k_ndcg} = {ndcg_at_k:.4f}")
        except Exception as e:
            logging.warning(f"Trial {trial.number}, Fold {fold_counter}: Failed - {e}", exc_info=False)
            fold_ndcg_scores.append(np.nan)
    avg_ndcg = np.nanmean(fold_ndcg_scores)
    logging.info(f"Trial {trial.number}: Avg NDCG@{k_ndcg} = {avg_ndcg:.4f}")
    return 1.0 - avg_ndcg if not np.isnan(avg_ndcg) else 10.0

def train_ranking_model_with_optuna(data_files, config_file, optuna_trials:int, start_year:int, end_year:int, timestamp:str):
    overall_start_time = time.time()
    logging.info("--- Step 1: Loading Configuration ---")
    config = load_config(config_file)
    if not config: return
    
    # Set determinism and resource control
    seeds = set_determinism_and_resource_control(config)
    
    # Get version metadata for reproducibility
    versions = get_versions_metadata()
    logging.info(f"Package versions: {versions}")
    logging.info("--- Step 2: Loading Raw Data ---")
    all_dfs = []
    for data_file in data_files:
        try:
            df_year = load_raw_data(data_file)
            if df_year is not None and not df_year.empty:
                all_dfs.append(df_year)
            else:
                logging.warning(f"Skipping empty or failed load for: {data_file}")
        except Exception as e:
            logging.error(f"Failed to load {data_file}: {e}")
            return
    if not all_dfs:
        logging.error("No data loaded successfully. Cannot proceed.")
        return
    df_raw = pd.concat(all_dfs, ignore_index=True)
    df_raw = df_raw.sort_values(by=['date', 'contractID']).reset_index(drop=True)
    if df_raw is None or df_raw.empty: return
    logging.info(f"--- Raw data loaded. Shape: {df_raw.shape} ---")
    logging.info("--- Step 3: Preprocessing Data (using utils.preprocess_data) ---")
    t_start_preprocess = time.time()
    df_processed, _ = preprocess_data(df_raw, config, scaler=None)
    t_end_preprocess = time.time()
    if df_processed is None or df_processed.empty: return
    logging.info(f"Utils preprocessing done. Shape: {df_processed.shape}. Time: {t_end_preprocess - t_start_preprocess:.2f}s.")
    logging.info("--- Step 3.5: Engineering Basic Features (Price/IV Changes) ---")
    t_start_eng1 = time.time()
    df_processed = df_processed.sort_values(by=['contractID', 'date'])
    
    # Use raw columns for feature engineering if available
    price_col = 'last_raw' if 'last_raw' in df_processed.columns else 'last'
    iv_col = 'implied_volatility_raw' if 'implied_volatility_raw' in df_processed.columns else 'implied_volatility'
    
    if price_col in df_processed.columns:
        df_processed['price_change_1d'] = df_processed.groupby('contractID')[price_col].pct_change(1)
        # Handle NaN and inf values from pct_change
        df_processed['price_change_1d'] = df_processed['price_change_1d'].replace([np.inf, -np.inf], np.nan).fillna(0)
        logging.info(f"Calculated price_change_1d (percent change) using column: {price_col}")
    else: 
        df_processed['price_change_1d'] = 0
        logging.warning(f"Price column '{price_col}' not found for basic feature engineering")
        
    if iv_col in df_processed.columns:
        df_processed['iv_change_1d'] = df_processed.groupby('contractID')[iv_col].diff(1)
        logging.info(f"Calculated iv_change_1d using column: {iv_col}")
    else: 
        df_processed['iv_change_1d'] = 0
        logging.warning(f"IV column '{iv_col}' not found for basic feature engineering")
    df_processed['price_change_1d'] = df_processed['price_change_1d'].fillna(0)
    df_processed['iv_change_1d'] = df_processed['iv_change_1d'].fillna(0)
    t_end_eng1 = time.time()
    logging.info(f"Basic engineering done. Shape: {df_processed.shape}. Time: {t_end_eng1 - t_start_eng1:.2f}s.")
    logging.info("--- Step 4: Calculating Target Relevance Score (Sharpe Ratio) on Full Dataset ---")
    df_target = calculate_target(df_processed, TARGET_LOOKAHEAD_DAYS)
    if df_target is None or df_target.empty: return
    
    logging.info("--- Step 4.3: Applying Health Checks to Drop Weak Days ---")
    df_target = drop_weak_days(df_target, config)
    if df_target is None or df_target.empty: 
        logging.error("No data remaining after health checks.")
        return
    
    logging.info("--- Step 4.5: Converting Sharpe Ratio to Integer Relevance Levels (Full Dataset) ---")
    try:
        target_col = 'target_5d_sharpe'
        quantiles = df_target[target_col].quantile([0.25, 0.5, 0.75])
        q1, q2, q3 = quantiles[0.25], quantiles[0.5], quantiles[0.75]
        logging.info(f"Sharpe Quantiles for Relevance Binning (Full Dataset): Q1={q1:.4f}, Q2={q2:.4f}, Q3={q3:.4f}")
        conditions = [
            df_target[target_col] <= q1,
            (df_target[target_col] > q1) & (df_target[target_col] <= q2),
            (df_target[target_col] > q2) & (df_target[target_col] <= q3),
            df_target[target_col] > q3
        ]
        choices = [0, 1, 2, 3]
        df_target['target_relevance_int'] = np.select(conditions, choices, default=0)
        target_int_col = 'target_relevance_int'
        logging.info(f"Created integer relevance levels (Full Dataset). Value counts:\n{df_target[target_int_col].value_counts().sort_index().to_string()}")
        edges = [-np.inf, q1, q2, q3, np.inf]
        os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)
        edges_file = os.path.join(MODEL_OUTPUT_PATH, f"sharpe_qcut_edges_{start_year}_{end_year}_{timestamp}.pkl")
        joblib.dump(edges, edges_file)
        logging.info(f"Saved quantile edges for relevance binning to {edges_file}")
    except Exception as e:
        logging.error(f"Failed to convert Sharpe to integer relevance levels: {e}", exc_info=True)
        return
    logging.info("--- Step 5: Defining Features, Target (Integer Relevance), and Groups for Optuna (Full Dataset) ---")
    available_cols = df_target.columns
    numerical_features_in_data = [f for f in NUMERICAL_FEATURES if f in available_cols]
    categorical_features_in_data = [f for f in CATEGORICAL_FEATURES if f in available_cols]
    features_to_use = numerical_features_in_data + categorical_features_in_data
    target_int_col = 'target_relevance_int'
    if not features_to_use or target_int_col not in available_cols:
        logging.error("Could not define features or integer relevance target on filtered data.")
        return
    logging.info(f"Using {len(features_to_use)} features: {features_to_use}")
    logging.info(f"Using target: {target_int_col}")
    X = df_target[['date'] + features_to_use]
    y = df_target[target_int_col]
    if 'type' in features_to_use:
        X.loc[:, 'type'] = X['type'].astype(str)
    best_params = {}
    if optuna_trials > 0:
        logging.info(f"--- Step 6: Running Optuna Optimization ({optuna_trials} trials for rank:ndcg, Full {start_year}-{end_year} Dataset) --- ")
        study = optuna.create_study(direction='minimize')
        objective_with_data = partial(objective_rank, X=X, y=y,
                                       n_splits=N_CV_SPLITS,
                                       numerical_features=NUMERICAL_FEATURES,
                                       categorical_features=CATEGORICAL_FEATURES,
                                       k_ndcg=NDCG_K)
        try:
            study.optimize(objective_with_data, n_trials=optuna_trials, timeout=OPTUNA_TIMEOUT)
        except KeyboardInterrupt:
            logging.warning("Optuna interrupted by user. Using best params so far (if any).")
        except Exception as e:
            logging.error(f"Error during Optuna optimization: {e}", exc_info=True)
        if study.best_trial:
            best_params = study.best_trial.params
            logging.info(f"Best params from Optuna: {best_params}")
        else:
            logging.error("Optuna produced no best trial. Aborting training.")
            return
    else:
        logging.info("--- Step 6: Optuna disabled (trials=0). Using FIXED_PARAMS ---")
        best_params = FIXED_PARAMS.copy()
    logging.info(f"--- Step 7: Training Final Ranking Model (Full {start_year}-{end_year} Dataset) with Best Params ---")
    try:
        final_ltr_params = {
            'objective': 'rank:ndcg',
            'eval_metric': f'ndcg@{NDCG_K}',
            'tree_method': 'hist',
            'seed': seeds.get('xgboost', 42),
            'n_jobs': -1,
            'n_estimators': 1500,
            'early_stopping_rounds': 50,
            **best_params
        }
        
        final_features_to_use = [f for f in NUMERICAL_FEATURES + CATEGORICAL_FEATURES if f in X.columns]
        logging.info(f"Final model using {len(final_features_to_use)} features." )
        
        # Consistent feature projection
        X_projected = project_features_consistently(X, final_features_to_use)
        
        os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)
        features_file = os.path.join(MODEL_OUTPUT_PATH, f"xgb_feature_names_{start_year}_{end_year}_{timestamp}.pkl")
        joblib.dump(final_features_to_use, features_file)
        logging.info(f"Saved final feature list to {features_file}")
        
        # Split data for early stopping validation
        val_days = config.get('val_days', 10)
        X_train, X_val, y_train, y_val = split_train_val_by_recent_days(X_projected, y, val_days)
        
        # Get group info for train and val
        train_group_info = get_group_info(X_train)
        val_group_info = get_group_info(X_val)
        
        # Create and fit preprocessor on train data only
        final_preprocessor = create_preprocessor(NUMERICAL_FEATURES, CATEGORICAL_FEATURES, final_features_to_use)
        final_model = xgb.XGBRanker(**final_ltr_params)
        final_pipeline = Pipeline(steps=[('preprocessor', final_preprocessor), ('ranker', final_model)])
        
        # Fit preprocessor only on training data
        logging.info("Fitting preprocessor on training data only...")
        final_pipeline.named_steps['preprocessor'].fit(X_train[final_features_to_use])
        
        # Transform both train and val data
        X_train_processed = final_pipeline.named_steps['preprocessor'].transform(X_train[final_features_to_use])
        X_val_processed = final_pipeline.named_steps['preprocessor'].transform(X_val[final_features_to_use])
        
        logging.info(f"Fitting final ranking model with early stopping (val_days={val_days})...")
        logging.info(f"Training samples: {len(X_train_processed)}, Validation samples: {len(X_val_processed)}")
        
        # Train with early stopping
        final_pipeline.named_steps['ranker'].fit(
            X_train_processed, y_train,
            group=train_group_info,
            eval_set=[(X_val_processed, y_val)],
            eval_group=[val_group_info],
            verbose=False
        )
        
        # Log best iteration if available
        ranker = final_pipeline.named_steps['ranker']
        if hasattr(ranker, 'best_iteration') and ranker.best_iteration is not None:
            logging.info(f"Early stopping at iteration {ranker.best_iteration} (best_ntree_limit={getattr(ranker, 'best_ntree_limit', 'N/A')})")
        
        logging.info("Final ranking model training complete.")
        
        logging.info(f"--- Step 8: Saving Final Tuned Ranking Model ({start_year}-{end_year}) Pipeline --- ")
        os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)
        model_fname = f"xgboost_ranker_{start_year}_{end_year}_{'optuna_tuned' if optuna_trials>0 else 'fixed_params'}_{timestamp}.joblib"
        model_path = os.path.join(MODEL_OUTPUT_PATH, model_fname)
        joblib.dump(final_pipeline, model_path)
        logging.info(f"Final ranking pipeline saved successfully to {model_path}")
        
        # Save training metadata
        metadata = {
            'timestamp': timestamp,
            'training_years': f"{start_year}-{end_year}",
            'package_versions': versions,
            'seeds': seeds,
            'config_params': {
                'val_days': val_days,
                'min_rows_per_day': config.get('min_rows_per_day'),
                'min_target_coverage': config.get('min_target_coverage'),
                'spread_cap': config.get('spread_cap'),
            },
            'final_params': final_ltr_params,
            'features_count': len(final_features_to_use),
            'train_date_range': [str(X_train['date'].min()), str(X_train['date'].max())],
            'val_date_range': [str(X_val['date'].min()), str(X_val['date'].max())],
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'best_iteration': getattr(ranker, 'best_iteration', None),
            'optuna_trials': optuna_trials
        }
        
        metadata_file = os.path.join(MODEL_OUTPUT_PATH, f"training_metadata_{start_year}_{end_year}_{timestamp}.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        logging.info(f"Saved training metadata to {metadata_file}")
    except Exception as e:
        logging.error(f"Failed during final model training or saving: {e}", exc_info=True)
    overall_end_time = time.time()
    logging.info(f"--- Training (years {start_year}-{end_year}) finished in {overall_end_time - overall_start_time:.2f}s ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2019, help="Start year of training data (inclusive)")
    parser.add_argument("--end-year", type=int, default=2025, help="End year of training data (inclusive)")
    parser.add_argument("--trials", type=int, default=100, help="Number of Optuna trials (0 = disable)")
    # CONFIG_FILE is now updated to the local path
    parser.add_argument("--config", default=CONFIG_FILE, help="Path to preprocessing config YAML")
    args = parser.parse_args()

    # Build data files list dynamically with adjusted paths
    data_files = [f"./year_{y}_data.csv" for y in range(args.start_year, args.end_year + 1)]
    logging.info(f"Training on years {args.start_year}-{args.end_year}. Files: {data_files}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_ranking_model_with_optuna(data_files, args.config, args.trials, args.start_year, args.end_year, timestamp) 