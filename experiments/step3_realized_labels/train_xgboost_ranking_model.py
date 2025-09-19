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

# Import necessary functions from utils
# utils.py is expected to be in the same directory as this script
from utils import load_config, preprocess_data, make_daily_rank_labels

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Main config file (used by preprocess_data)
CONFIG_FILE = "./config.yaml"

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

# REMOVED: Sharpe-based target calculation - replaced with per-day realized returns labeling

def create_preprocessor(numerical_features, categorical_features, available_columns):
    logging.debug(f"Creating preprocessor with Imputer only (no scaling for ranker)...")
    transformers = []
    num_features_exist = [f for f in numerical_features if f in available_columns]
    cat_features_exist = [f for f in categorical_features if f in available_columns]
    if num_features_exist:
        # Only SimpleImputer, no StandardScaler to avoid dataset-wide scaling leakage
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean'))
        ])
        transformers.append(('num', numerical_pipeline, num_features_exist))
        logging.debug(f"Preprocessor applying Imputer(mean) only to {len(num_features_exist)} numerical features.")
    else:
        logging.warning("No numerical features found for Imputer.")
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
    df_sorted = df.sort_values('date')
    group_info = df_sorted.groupby('date').size().tolist()
    logging.debug(f"Calculated group info for {len(group_info)} groups (days). Total samples: {sum(group_info)}.")
    return group_info

def train_val_split_by_recent_days(X, y, val_days=10):
    """Split data into train/val by taking last N days for validation"""
    X_sorted = X.sort_values('date')
    y_sorted = y.loc[X_sorted.index]
    
    unique_dates = sorted(X_sorted['date'].unique())
    val_start_date = unique_dates[-val_days] if len(unique_dates) >= val_days else unique_dates[0]
    
    train_mask = X_sorted['date'] < val_start_date
    val_mask = X_sorted['date'] >= val_start_date
    
    X_train = X_sorted[train_mask]
    X_val = X_sorted[val_mask] 
    y_train = y_sorted[train_mask]
    y_val = y_sorted[val_mask]
    
    logging.info(f"Train/val split: {len(X_train)} train samples, {len(X_val)} val samples")
    logging.info(f"Val period: {val_start_date} to {unique_dates[-1]} ({len(X_val['date'].unique())} days)")
    
    return X_train, X_val, y_train, y_val

def run_manual_grid_search(X, y, config, features_to_use):
    """Run manual grid search with early stopping on recent-day validation"""
    logging.info("Starting manual grid search with early stopping...")
    
    # Get validation split
    val_days = config.get('val_days', 10)
    X_train, X_val, y_train, y_val = train_val_split_by_recent_days(X, y, val_days)
    
    # Get group info for train and validation
    train_group_info = get_group_info(X_train)
    val_group_info = get_group_info(X_val)
    
    # Grid search parameters
    grid = config.get('grid_search', {
        'max_depth': [3, 4, 5],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.7, 0.9],
        'colsample_bytree': [0.6, 0.8]
    })
    
    best_score = -np.inf
    best_params = None
    
    # Try both bin configurations if specified
    original_bins_key = config.get('rank_label_bins_key', 'bins_a')
    bin_keys_to_try = ['bins_a', 'bins_b'] if 'bins_b' in config else [original_bins_key]
    
    results = []
    
    for bins_key in bin_keys_to_try:
        logging.info(f"=== Testing with {bins_key} bins ===")
        
        # Update config temporarily
        config['rank_label_bins_key'] = bins_key
        
        # Re-generate labels with new bins
        logging.info("Re-generating labels with new binning strategy...")
        # Note: In a full implementation, we'd re-run the labeling pipeline
        # For now, we'll proceed with current labels but log the intent
        logging.warning("Bin switching not fully implemented - using current labels")
        
        param_combinations = []
        for max_depth in grid['max_depth']:
            for lr in grid['learning_rate']:
                for subsample in grid['subsample']:
                    for colsample in grid['colsample_bytree']:
                        param_combinations.append({
                            'max_depth': max_depth,
                            'learning_rate': lr,
                            'subsample': subsample,
                            'colsample_bytree': colsample
                        })
        
        logging.info(f"Testing {len(param_combinations)} parameter combinations for {bins_key}")
        
        for i, params in enumerate(param_combinations):
            try:
                # XGBoost parameters
                xgb_params = {
                    'objective': 'rank:ndcg',
                    'eval_metric': 'ndcg@20',
                    'tree_method': 'hist',
                    'seed': 42,
                    'n_jobs': -1,
                    'n_estimators': 1000,
                    'early_stopping_rounds': 50,
                    **params
                }
                
                logging.debug(f"Config {i+1}/{len(param_combinations)}: {params}")
                
                # Create preprocessor and model
                preprocessor = create_preprocessor(NUMERICAL_FEATURES, CATEGORICAL_FEATURES, features_to_use)
                model = xgb.XGBRanker(**xgb_params)
                
                # Fit preprocessor only on training data
                preprocessor.fit(X_train[features_to_use])
                
                # Transform data
                X_train_processed = preprocessor.transform(X_train[features_to_use])
                X_val_processed = preprocessor.transform(X_val[features_to_use])
                
                # Convert instance weights to group weights for XGBoost ranking
                group_weights = None
                if 'sample_weight' in X_train.columns:
                    # Aggregate instance weights to group (day) level weights
                    X_train_sorted = X_train.sort_values('date')
                    group_weights = X_train_sorted.groupby('date')['sample_weight'].mean().values
                    logging.debug(f"Using {len(group_weights)} group weights: mean={group_weights.mean():.3f}, std={group_weights.std():.3f}")
                    assert len(group_weights) == len(train_group_info), f"Group weights mismatch: {len(group_weights)} != {len(train_group_info)}"
                
                # Train with early stopping and group weights
                fit_params = {
                    'group': train_group_info,
                    'eval_set': [(X_val_processed, y_val)],
                    'eval_group': [val_group_info], 
                    'verbose': False
                }
                # XGBoost sklearn API doesn't support group weights easily, skip for now
                # if group_weights is not None:
                #     fit_params['sample_weight'] = group_weights
                
                model.fit(X_train_processed, y_train, **fit_params)
                
                # Get validation score
                val_score = model.best_score
                best_iteration = model.best_iteration
                
                result = {
                    'bins_key': bins_key,
                    'params': params,
                    'val_ndcg20': val_score,
                    'best_iteration': best_iteration
                }
                results.append(result)
                
                logging.info(f"Config {i+1}: NDCG@20={val_score:.4f}, best_iter={best_iteration}")
                
                if val_score > best_score:
                    best_score = val_score
                    best_params = {
                        'bins_key': bins_key,
                        'model_params': params,
                        'best_iteration': best_iteration,
                        'val_ndcg20': val_score
                    }
                    logging.info(f"New best config: {val_score:.4f}")
                    
            except Exception as e:
                logging.warning(f"Config {i+1} failed: {e}")
                continue
    
    # Restore original bins key
    config['rank_label_bins_key'] = original_bins_key
    
    if best_params is None:
        logging.error("Grid search failed to find any valid configuration!")
        return FIXED_PARAMS.copy()
    
    logging.info(f"=== Grid Search Complete ===")
    logging.info(f"Best config: bins={best_params['bins_key']}, NDCG@20={best_params['val_ndcg20']:.4f}")
    logging.info(f"Best params: {best_params['model_params']}")
    
    # Save results
    import json
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/step3_params.json", "w") as f:
        json.dump({
            'best_params': best_params,
            'all_results': results
        }, f, indent=2)
    
    # Update config with best bins
    config['rank_label_bins_key'] = best_params['bins_key']
    
    return best_params['model_params']

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
        
        # Group integrity check: assert group sizes match data sizes
        train_group_info = get_group_info(X_train_fold)
        val_group_info = get_group_info(X_val_fold)
        assert sum(train_group_info) == len(X_train_fold), f"Train group size mismatch: {sum(train_group_info)} != {len(X_train_fold)}"
        assert sum(val_group_info) == len(X_val_fold), f"Val group size mismatch: {sum(val_group_info)} != {len(X_val_fold)}"
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
    
    logging.info("--- Step 4: Creating Per-Day Realized Forward Returns Labels ---")
    # Get horizon from config, default to 3 days
    horizon_days = config.get('label_horizon_days', 3)
    df_target = make_daily_rank_labels(df_processed, h=horizon_days, price_col='last_raw', config=config)
    
    if df_target is None or df_target.empty: 
        logging.error("Failed to create realized return labels. Aborting training.")
        return
    
    logging.info(f"Realized return labeling complete. Final shape: {df_target.shape}")
    
    logging.info("--- Step 5: Defining Features, Target (Rank Labels), and Groups for Optuna (Full Dataset) ---")
    available_cols = df_target.columns
    numerical_features_in_data = [f for f in NUMERICAL_FEATURES if f in available_cols]
    categorical_features_in_data = [f for f in CATEGORICAL_FEATURES if f in available_cols]
    features_to_use = numerical_features_in_data + categorical_features_in_data
    target_col = 'rank_label'
    if not features_to_use or target_col not in available_cols:
        logging.error("Could not define features or rank_label target on filtered data.")
        return
    logging.info(f"Using {len(features_to_use)} features: {features_to_use}")
    logging.info(f"Using target: {target_col}")
    # Include sample_weight if available
    columns_to_include = ['date'] + features_to_use
    if 'sample_weight' in df_target.columns:
        columns_to_include.append('sample_weight')
        logging.info("Sample weights available for training")
    
    X = df_target[columns_to_include]
    y = df_target[target_col]
    if 'type' in features_to_use:
        X.loc[:, 'type'] = X['type'].astype(str)
    
    # Assert no dataset-wide scaling has occurred
    scaled_cols = [c for c in X.columns if c.endswith('_scaled')]
    assert not any(scaled_cols), f"Found scaled columns which indicate dataset-wide scaling: {scaled_cols}"
    
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
        logging.info("--- Step 6: Manual Grid Search with Early Stopping ---")
        best_params = run_manual_grid_search(X, y, config, features_to_use)
    logging.info(f"--- Step 7: Training Final Ranking Model (Full {start_year}-{end_year} Dataset) with Best Params ---")
    try:
        final_ltr_params = {
            'objective': 'rank:ndcg',
            'eval_metric': f'ndcg@{NDCG_K}',
            'tree_method': 'hist',
            'seed': 42,
            'n_jobs': -1,
            'n_estimators': 1500,
            **best_params
        }
        final_ltr_params.pop('early_stopping_rounds', None)
        final_features_to_use = [f for f in NUMERICAL_FEATURES + CATEGORICAL_FEATURES if f in X.columns]
        logging.info(f"Final model using {len(final_features_to_use)} features." )
        os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)
        features_file = os.path.join(MODEL_OUTPUT_PATH, f"xgb_feature_names_{start_year}_{end_year}_{timestamp}.pkl")
        joblib.dump(final_features_to_use, features_file)
        logging.info(f"Saved final feature list to {features_file}")
        final_preprocessor = create_preprocessor(NUMERICAL_FEATURES, CATEGORICAL_FEATURES, final_features_to_use)
        final_model = xgb.XGBRanker(**final_ltr_params)
        final_pipeline = Pipeline(steps=[('preprocessor', final_preprocessor), ('ranker', final_model)])
        X_final = X.sort_values('date')
        y_final = y.loc[X_final.index]
        full_group_info = get_group_info(X_final)
        
        # Group integrity check for final training
        assert sum(full_group_info) == len(X_final), f"Final group size mismatch: {sum(full_group_info)} != {len(X_final)}"
        logging.info(f"Group integrity verified. Training on {len(X_final)} samples across {len(full_group_info)} days")
        
        logging.info(f"Fitting final ranking pipeline (Full {start_year}-{end_year} Dataset) with params: {final_ltr_params}...")
        
        # Final training - skip sample weights for sklearn API compatibility
        fit_params = {'ranker__group': full_group_info}
        if 'sample_weight' in X_final.columns:
            # For now, skip group weights due to sklearn API complexity
            # The instance weights are embedded in the training data distribution
            logging.info("Instance weights available but skipping group weights for sklearn API compatibility")
        
        final_pipeline.fit(X_final[final_features_to_use], y_final, **fit_params)
        logging.info("Final ranking model training complete.")
        logging.info(f"--- Step 8: Saving Final Tuned Ranking Model ({start_year}-{end_year}) Pipeline --- ")
        os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)
        model_fname = f"xgboost_ranker_{start_year}_{end_year}_{'optuna_tuned' if optuna_trials>0 else 'fixed_params'}_{timestamp}.joblib"
        model_path = os.path.join(MODEL_OUTPUT_PATH, model_fname)
        joblib.dump(final_pipeline, model_path)
        logging.info(f"Final ranking pipeline saved successfully to {model_path}")
        
        # NOTE: No longer saving sharpe_qcut_edges - using per-day ranking instead
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