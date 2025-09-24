import os
import pandas as pd
import numpy as np
import yaml
import boto3
from datetime import datetime
# import awswrangler as wr
from sklearn.preprocessing import RobustScaler
# tsfresh removed - replaced with causal rolling features
# from tsfresh import extract_features
# from tsfresh.feature_extraction import MinimalFCParameters
import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from logger import setup_logger
import time
import gym

logger = setup_logger(__name__)

# tsfresh caching removed - no longer needed
# TS_FEATURES_CACHE = "ts_features_cache.parquet"

def load_config(config_file):
    logger.info("Loading configuration from %s", config_file)
    try:
        # Try direct path first
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
        # If that fails, try relative to script directory
        else:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            abs_config_path = os.path.join(script_dir, config_file)
            if os.path.exists(abs_config_path):
                with open(abs_config_path, 'r') as f:
                    config = yaml.safe_load(f)
            else:
                # One more try with just the filename
                base_config = os.path.basename(config_file)
                abs_config_path = os.path.join(script_dir, base_config)
                with open(abs_config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    
        logger.info("Configuration loaded successfully")
        
        # Ensure output directory is properly set
        if 'output_dir' in config and config['output_dir'].startswith("./"):
            # Convert relative path to be relative to script directory
            config['output_dir'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                             config['output_dir'][2:])
            
    except Exception as e:
        logger.error("Failed to load configuration: %s", e)
        raise e
    return config

def load_data(year="2010"):
    """
    Load data from CSV file.
    
    Args:
        year (str): Year of data to load ("2010" or "2011")
    
    Returns:
        pandas.DataFrame: Loaded data
    """
    filename = f"year_{year}_data.csv"
    # Original logic: Look for the data file in the same directory as this script
    local_csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    logger.info(f"Attempting to load data from: {local_csv_path}")
    df = pd.read_csv(local_csv_path)
    logger.info(f"Successfully loaded data with {len(df)} rows")
    return df

# def load_data(config_file_path="config.yaml"):  # Pass config file path as argument, default to "config.yaml"
#     """
#     Loads data from SageMaker's training channel if available,
#     otherwise from S3 using the path from config.yaml.
#     Logs the duration of the data loading step.
#     """
#     start_time = datetime.now()
#     config = None  # Initialize config to None

#     try:
#         with open(config_file_path, 'r') as f:
#             config = yaml.safe_load(f)
#         logger.info(f"Successfully loaded configuration from: {config_file_path}")
#     except FileNotFoundError:
#         logger.warning(f"Configuration file not found at: {config_file_path}. Falling back to environment variables and defaults.")
#     except yaml.YAMLError as e:
#         logger.error(f"Error parsing YAML config file: {e}")
#         config = None # Ensure config is None in case of YAML parsing error


#     try:
#         channel_path = os.environ.get("SM_CHANNEL_TRAIN")
#         logger.info("Checking SM_CHANNEL_TRAIN environment variable...")
#         logger.info("SM_CHANNEL_TRAIN value: %s", channel_path)
#         if channel_path:
#             logger.info("Loading data from SageMaker channel: %s", channel_path)
#             data_path_to_read = channel_path
#             logger.info(f"Attempting to read parquet dataset from channel path: {data_path_to_read}")
#             df = wr.s3.read_parquet(path=data_path_to_read, dataset=True)
#             logger.info(f"Successfully read parquet dataset from channel path.")

#         else:
#             s3_path_config = None
#             if config and 'data_source' in config and 's3_data_path' in config['data_source']:
#                 s3_path_config = config['data_source']['s3_data_path']
#                 logger.info("S3 data path found in config.yaml: %s", s3_path_config)
#             else:
#                 logger.warning("S3 data path not found in config.yaml or 'data_source.s3_data_path' not defined.")

#             if s3_path_config:
#                 s3_path = s3_path_config
#                 logger.info("Loading data directly from S3 using path from config.yaml: %s", s3_path)
#             else:
#                 # Fallback to hardcoded S3 path (or raise error if you prefer no hardcoding)
#                 s3_path = "s3://yel-spy-etled-4.0/processed_data/" # Default S3 path if config not found/path not in config
#                 logger.warning(f"Falling back to default hardcoded S3 path: {s3_path}. Consider setting 'data_source.s3_data_path' in config.yaml.")
#                 # Optionally, you could raise an error here instead of falling back to a hardcoded path:
#                 # raise ValueError("S3 data path not found in config.yaml and SM_CHANNEL_TRAIN not set.")


#             logger.info("Loading data directly from S3 using path: %s", s3_path)
#             data_path_to_read = s3_path
#             logger.info(f"Attempting to read parquet dataset from S3 path: {data_path_to_read}")
#             df = wr.s3.read_parquet(path=data_path_to_read, dataset=True)
#             logger.info(f"Successfully read parquet dataset from S3 path.")


#         duration = (datetime.now() - start_time).total_seconds()
#         logger.info("Data loaded: %d records with %d columns in %.2f seconds", len(df), len(df.columns), duration)
#         return df
#     except Exception as e:
#         logger.error("Data loading failed: %s", e)
#         logger.error(f"Exception details: {type(e)} - {e}")
#         raise


def validate_and_convert(df):
    """Ensure required columns are present and convert them to proper dtypes."""
    required_numeric_cols = [
        'strike', 'last', 'bid', 'ask', 'volume', 'open_interest',
        'implied_volatility', 'delta', 'gamma', 'theta', 'vega', 'rho',
        'days_to_exp', 'price_change', 'spy_d_close', 'iv_vix_ratio', 'spy_momentum',
        'fair_value', 'fred_dgs1', 'fred_dgs10', 'fred_dgs2', 'fred_dgs30',
        'fred_dgs3mo', 'fred_dgs5', 'fred_dgs6mo'
    ]
    for col in required_numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            logger.error("Missing required column: %s", col)
            raise ValueError(f"Missing required column: {col}")

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    if 'expiration' in df.columns:
        df['expiration'] = pd.to_datetime(df['expiration'], errors='coerce')
        
    logger.debug("Data conversion complete for required columns")
    return df

def _sanitize_spreads(df, config):
    """Sanitize bid/ask spreads: fix inverted quotes and cap spread_pct."""
    spread_cap = config.get('spread_cap', 0.6)
    initial_count = len(df)
    
    if 'bid' not in df.columns or 'ask' not in df.columns:
        logger.warning("bid/ask columns not found, skipping spread sanitization")
        return df
    
    # Fix inverted quotes (ask < bid)
    inverted_mask = df['ask'] < df['bid']
    if inverted_mask.sum() > 0:
        logger.info(f"Found {inverted_mask.sum()} inverted quotes (ask < bid), swapping...")
        df.loc[inverted_mask, ['bid', 'ask']] = df.loc[inverted_mask, ['ask', 'bid']].values
    
    # Recompute spread_pct from bid/ask mid
    mid = (df['bid'] + df['ask']) / 2
    spread_pct = np.where(mid > 0, (df['ask'] - df['bid']) / mid, np.nan)
    
    # Replace inf/neg values and cap
    spread_pct = np.clip(np.nan_to_num(spread_pct, nan=0, posinf=spread_cap, neginf=0), 
                        0, spread_cap)
    
    df['spread_pct'] = spread_pct
    
    # Log statistics
    capped_count = (df['spread_pct'] >= spread_cap).sum()
    if capped_count > 0:
        logger.info(f"Capped {capped_count} spread_pct values to {spread_cap}")
    
    logger.info(f"Spread sanitization complete. spread_pct stats: "
               f"mean={df['spread_pct'].mean():.4f}, "
               f"max={df['spread_pct'].max():.4f}, "
               f"capped_count={capped_count}")
    
    return df

def _convert_and_filter(df, config):
    """Helper to convert types and filter problematic data points."""
    logger.debug("Converting types and filtering data...")
    initial_count = len(df)

    # Convert columns to numeric
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
    numeric_columns = ["bid_size", "ask_size", "last", "fair_value", "delta", "theta", "days_to_exp", "open_interest"]
    for col in numeric_columns:
        if col in df.columns:
            df.loc[:, col] = pd.to_numeric(df[col], errors='coerce')

    # Apply spread sanitization before filtering
    df = _sanitize_spreads(df, config)

    # Filter 1: Remove options with zero or near-zero prices
    min_price_threshold = config.get('min_price_threshold', 0.05)
    df = df[df['last'] >= min_price_threshold]

    # Filter 2: Remove options with too large bid-ask spread relative to price  
    if 'bid' in df.columns and 'ask' in df.columns:
        df['relative_spread'] = (df['ask'] - df['bid']) / df['last']
        df = df[df['relative_spread'] <= 1.0]

    # Filter 3: Ensure minimum liquidity
    if 'volume' in df.columns and 'open_interest' in df.columns:
        df = df[(df['volume'] > 0) | (df['open_interest'] > 100)]

    logger.info("Filtered out %d problematic records (%.2f%%)",
                initial_count - len(df),
                (initial_count - len(df)) / initial_count * 100 if initial_count > 0 else 0)
    return df

def _engineer_features(df):
    """Helper to perform feature engineering."""
    logger.debug("Engineering features...")
    logger.info("Pre-feature engineering last stats: min=%f, max=%f, mean=%f, std=%f",
                df['last'].min(), df['last'].max(), df['last'].mean(), df['last'].std())

    # Ensure data is sorted by contractID and date for diff operations
    # This is crucial for .diff() to calculate changes against the PREVIOUS day for the SAME contract.
    if 'date' in df.columns and 'contractID' in df.columns:
        df = df.sort_values(by=['contractID', 'date'])
    else:
        logger.warning("'contractID' or 'date' column missing, cannot sort for lagged features. Lagged features might be incorrect.")

    # Calculate 1-day price change (percent change) using raw values when available
    if 'contractID' in df.columns:
        # Use raw column if available, otherwise fallback to scaled column
        price_col = 'last_raw' if 'last_raw' in df.columns else 'last'
        if price_col in df.columns:
            df['price_change_1d'] = df.groupby('contractID', group_keys=False)[price_col].pct_change(1)
            # Handle NaN and inf values from pct_change
            df['price_change_1d'] = df['price_change_1d'].replace([np.inf, -np.inf], np.nan).fillna(0)
            logger.debug(f"Calculated price_change_1d (percent change) using column: {price_col}")
        else:
            df['price_change_1d'] = np.nan
            logger.warning("Neither 'last_raw' nor 'last' column found, cannot calculate 'price_change_1d'.")
    else:
        df['price_change_1d'] = np.nan
        logger.warning("'contractID' column missing, cannot calculate 'price_change_1d'.")

    # Calculate 1-day IV change using raw values when available
    if 'contractID' in df.columns:
        # Use raw column if available, otherwise fallback to scaled column
        iv_col = 'implied_volatility_raw' if 'implied_volatility_raw' in df.columns else 'implied_volatility'
        if iv_col in df.columns:
            df['iv_change_1d'] = df.groupby('contractID', group_keys=False)[iv_col].diff(1)
            logger.debug(f"Calculated iv_change_1d using column: {iv_col}")
        else:
            df['iv_change_1d'] = np.nan
            logger.warning("Neither 'implied_volatility_raw' nor 'implied_volatility' column found, cannot calculate 'iv_change_1d'.")
    else:
        df['iv_change_1d'] = np.nan
        logger.warning("'contractID' column missing, cannot calculate 'iv_change_1d'.")

    # Order Flow Imbalance (OFI)
    if 'bid_size' in df.columns and 'ask_size' in df.columns and 'volume' in df.columns:
        denominator = df['bid_size'] + df['ask_size']
        df.loc[:, 'ofi'] = np.where(denominator == 0, 0,
                                   ((df['bid_size'] - df['ask_size']) / denominator) * df['volume'])
    else:
        df.loc[:, 'ofi'] = np.nan

    # Volatility Skew
    if {'iv_25delta_put', 'iv_25delta_call', 'spy_d_close'}.issubset(df.columns):
        df.loc[:, 'vol_skew'] = np.where(df['spy_d_close'] == 0, 0,
                                        (df['iv_25delta_put'] - df['iv_25delta_call']) / df['spy_d_close'])
    else:
        df.loc[:, 'vol_skew'] = np.nan

    # Zero-Day Premium Decay (use raw price for denominator)
    price_col_for_features = 'last_raw' if 'last_raw' in df.columns else 'last'
    if 'theta' in df.columns and price_col_for_features in df.columns and 'days_to_exp' in df.columns:
        denominator = df[price_col_for_features] * np.sqrt(df['days_to_exp'] + 1e-8)
        df.loc[:, 'zero_day_premium'] = np.where(denominator == 0, 0, df['theta'] / denominator)
        logger.debug(f"Calculated zero_day_premium using price column: {price_col_for_features}")
    else:
        df.loc[:, 'zero_day_premium'] = np.nan

    # Option Volume / OI Ratio
    if 'volume' in df.columns and 'open_interest' in df.columns:
        df.loc[:, 'option_volume_oi_ratio'] = np.where(df['open_interest'] == 0, 0,
                                                      df['volume'] / (df['open_interest'] + 1e-8))
    else:
        df.loc[:, 'option_volume_oi_ratio'] = np.nan

    # Mispricing Ratio (use raw price for numerator)
    if price_col_for_features in df.columns and 'fair_value' in df.columns:
        df.loc[:, 'mispricing_ratio'] = np.where(df['fair_value'] == 0, 0,
                                                (df[price_col_for_features] - df['fair_value']) / (df['fair_value'] + 1e-8))
        logger.debug(f"Calculated mispricing_ratio using price column: {price_col_for_features}")
    else:
        df.loc[:, 'mispricing_ratio'] = np.nan

    # Risk-Adjusted Signal
    epsilon = 1e-4
    if 'mispricing_ratio' in df.columns and 'delta' in df.columns and 'theta' in df.columns:
        df.loc[:, 'risk_adjusted_signal'] = df['mispricing_ratio'] * (df['delta'] / (df['theta'].abs() + epsilon))
    else:
        df.loc[:, 'risk_adjusted_signal'] = np.nan

    # Liquidity Adjustment to Risk Signal
    if 'bid_ask_spread' in df.columns and 'risk_adjusted_signal' in df.columns:
        df.loc[:, 'bid_ask_spread'] = pd.to_numeric(df['bid_ask_spread'], errors='coerce')
        df.loc[:, 'liquidity_adjustment'] = np.where(df['bid_ask_spread'] == 0, 0,
                                                    1 / (df['bid_ask_spread'] + 1e-3))
        # Ensure risk_adjusted_signal is numeric before multiplication
        df.loc[:, 'risk_adjusted_signal'] = pd.to_numeric(df['risk_adjusted_signal'], errors='coerce')
        df.loc[:, 'risk_adjusted_signal'] *= df['liquidity_adjustment']

    logger.debug("Feature engineering complete")
    logger.info("Post-feature engineering last stats: min=%f, max=%f, mean=%f, std=%f",
                df['last'].min(), df['last'].max(), df['last'].mean(), df['last'].std())
    return df

def _add_causal_rolling_features(df):
    """
    Add causal rolling features computed only from past data.
    No future information leaks into past rows.
    """
    logger.info("Adding causal rolling features...")
    
    # Ensure proper sorting for causal computation
    df = df.sort_values(['contractID', 'date']).reset_index(drop=True)
    
    # Define windows for rolling features
    windows = [5, 20]
    
    # Rolling features for price (using last_raw if available, otherwise last)
    price_col = 'last_raw' if 'last_raw' in df.columns else 'last'
    if price_col in df.columns:
        for w in windows:
            # Price rolling statistics
            df[f'price_roll_mean_{w}'] = df.groupby('contractID')[price_col].transform(
                lambda s: s.rolling(window=w, min_periods=max(1, w//4)).mean()
            )
            df[f'price_roll_std_{w}'] = df.groupby('contractID')[price_col].transform(
                lambda s: s.rolling(window=w, min_periods=max(1, w//4)).std()
            )
            df[f'price_roll_min_{w}'] = df.groupby('contractID')[price_col].transform(
                lambda s: s.rolling(window=w, min_periods=max(1, w//4)).min()
            )
            df[f'price_roll_max_{w}'] = df.groupby('contractID')[price_col].transform(
                lambda s: s.rolling(window=w, min_periods=max(1, w//4)).max()
            )
            
            # Z-score: (current - mean) / (std + epsilon)
            mean_col = f'price_roll_mean_{w}'
            std_col = f'price_roll_std_{w}'
            df[f'price_roll_zscore_{w}'] = (df[price_col] - df[mean_col]) / (df[std_col] + 1e-6)
    
    # Rolling features for implied volatility (using iv_raw if available)
    iv_col = 'implied_volatility_raw' if 'implied_volatility_raw' in df.columns else 'implied_volatility'
    if iv_col in df.columns:
        for w in windows:
            df[f'iv_roll_mean_{w}'] = df.groupby('contractID')[iv_col].transform(
                lambda s: s.rolling(window=w, min_periods=max(1, w//4)).mean()
            )
            df[f'iv_roll_std_{w}'] = df.groupby('contractID')[iv_col].transform(
                lambda s: s.rolling(window=w, min_periods=max(1, w//4)).std()
            )
    
    # Rolling features for volume and open interest
    if 'volume' in df.columns:
        for w in windows:
            df[f'vol_roll_mean_{w}'] = df.groupby('contractID')['volume'].transform(
                lambda s: s.rolling(window=w, min_periods=max(1, w//4)).mean()
            )
    
    if 'open_interest' in df.columns:
        for w in windows:
            df[f'oi_roll_mean_{w}'] = df.groupby('contractID')['open_interest'].transform(
                lambda s: s.rolling(window=w, min_periods=max(1, w//4)).mean()
            )
    
    # Volume/OI ratio (causal)
    if 'volume' in df.columns and 'open_interest' in df.columns:
        df['vol_oi_ratio'] = df['volume'] / (df['open_interest'] + 1e-6)
    
    # Fill NaN values in rolling features with 0 for early rows where windows aren't full
    rolling_cols = [col for col in df.columns if any(pattern in col for pattern in 
                   ['_roll_mean_', '_roll_std_', '_roll_min_', '_roll_max_', '_roll_zscore_'])]
    
    for col in rolling_cols:
        df[col] = df[col].fillna(0)
    
    if 'vol_oi_ratio' in df.columns:
        df['vol_oi_ratio'] = df['vol_oi_ratio'].fillna(0)
    
    logger.info(f"Added {len(rolling_cols) + ('vol_oi_ratio' in df.columns)} causal rolling features")
    return df

def _scale_data(df, scaler=None, config=None):
    """Helper to apply RobustScaler to numerical columns specified in config."""
    logger.debug("Applying scaling...")
    
    # Raw columns already created earlier in pipeline (before rolling features)
    # This prevents duplication and ensures rolling features use raw values
    
    if config is None:
        logger.error("Config not provided to _scale_data. Cannot determine columns to scale.")
        # Fallback or raise error - for now, try original dynamic selection with a warning
        # This path should ideally not be taken if config is always passed.
        logger.warning("Falling back to dynamic numerical column selection for scaling due to missing config.")
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_scaling = ['contractID', 'days_to_exp', 'last', 'date', 'expiration'] 
        numerical_cols = [col for col in numerical_cols if col not in exclude_scaling and col in df.columns]
    else:
        numerical_cols_from_config = config.get('numerical_cols_to_scale', [])
        # Ensure these columns actually exist in the DataFrame and are numeric
        numerical_cols = [
            col for col in numerical_cols_from_config 
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
        ]
        if not numerical_cols_from_config:
            logger.warning("'numerical_cols_to_scale' not found in config or is empty.")
        elif not numerical_cols: # If config had names but none were valid/present
            logger.warning("No valid/numeric columns found in DataFrame matching 'numerical_cols_to_scale' from config.")

    if not numerical_cols:
         logger.warning("No numerical columns identified for scaling. Skipping scaling.")
         return df, scaler

    # Replace inf with NaN before scaling
    df[numerical_cols] = df[numerical_cols].replace([np.inf, -np.inf], np.nan)

    # Log stats for numerical_cols to identify problematic columns before scaling
    # for col in numerical_cols:
    #     logger.debug("Pre-scaling %s stats: min=%f, max=%f, mean=%f, std=%f",
    #                  col, df[col].min(), df[col].max(), df[col].mean(), df[col].std())

    if scaler is None:
        scaler = RobustScaler()
        df.loc[:, numerical_cols] = scaler.fit_transform(df[numerical_cols]).astype(np.float32)
        logger.debug("Fit and applied RobustScaler.")
    else:
        df.loc[:, numerical_cols] = scaler.transform(df[numerical_cols]).astype(np.float32)
        logger.debug("Applied existing RobustScaler.")

    logger.info("Post-scaling last stats: min=%f, max=%f, mean=%f, std=%f",
                df['last'].min(), df['last'].max(), df['last'].mean(), df['last'].std())
    return df, scaler

def _final_cleanup(df):
    """Helper to drop rows with NaNs in critical columns."""
    critical_cols = ['contractID', 'date', 'last', 'fair_value', 'delta', 'theta']
    # Ensure critical columns exist before attempting to dropna
    cols_to_check = [col for col in critical_cols if col in df.columns]
    initial_count = len(df)
    df = df.dropna(subset=cols_to_check)
    dropped_count = initial_count - len(df)
    if dropped_count > 0:
        logger.info("Dropped %d rows missing critical values in columns: %s", dropped_count, cols_to_check)
    logger.debug("Final cleanup complete.")
    return df

def preprocess_data(df, config, scaler=None):
    """
    Preprocesses the input DataFrame by applying filtering, feature engineering,
    tsfresh feature extraction, scaling, and cleaning.

    Args:
        df (pd.DataFrame): Input DataFrame.
        config (dict): Configuration dictionary.
        scaler (RobustScaler, optional): Pre-fitted scaler. Defaults to None.

    Returns:
        tuple: Processed DataFrame and the fitted scaler.
    """
    logger.info("Starting preprocessing on data with %d records", len(df))
    if df.empty:
        logger.warning("Input DataFrame is empty. Skipping preprocessing.")
        return df, scaler

    df = df.copy() # Ensure we work on a copy

    # 1. Convert Types and Filter
    df = _convert_and_filter(df, config)
    if df.empty: logger.warning("DataFrame empty after filtering. Returning."); return df, scaler

    # 2. Engineer Features
    df = _engineer_features(df)
    if df.empty: logger.warning("DataFrame empty after feature engineering. Returning."); return df, scaler

    # 2.5. Create raw columns for causal feature computation (before scaling)
    raw_cols_to_preserve = ['last', 'implied_volatility']
    for col in raw_cols_to_preserve:
        if col in df.columns:
            df[f'{col}_raw'] = df[col].copy()
            logger.debug(f"Preserved raw column: {col}_raw")

    # 3. Add Causal Rolling Features (BEFORE scaling - uses raw values)
    if config.get('use_causal_rolling', True): # Optional causal rolling features via config
        df = _add_causal_rolling_features(df)
        if df.empty: logger.warning("DataFrame empty after causal rolling features. Returning."); return df, scaler
    else:
        logger.info("Skipping causal rolling feature extraction based on config.")

    # 4. Scale Data (rolling features will be scaled along with other numerics)
    df, scaler = _scale_data(df, scaler, config)
    if df.empty: logger.warning("DataFrame empty after scaling. Returning."); return df, scaler

    # 5. Final Cleanup (Drop NaNs)
    df = _final_cleanup(df)

    logger.info("Preprocessing complete. Final data has %d records", len(df))
    return df, scaler

def train_test_split_by_date(df, config):
    # Convert 'date' column to datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Sort by contractID and date to ensure sequential steps within each contract
    df = df.sort_values(by=['contractID', 'date']).reset_index(drop=True)
    
    # Define date ranges from config
    train_start = pd.to_datetime(config['train_start_date'])
    train_end = pd.to_datetime(config['train_end_date'])
    val_start = pd.to_datetime(config['validation_start_date'])
    val_end = pd.to_datetime(config['validation_end_date'])
    
    # Split into training and validation sets
    train_df = df[(df['date'] >= train_start) & (df['date'] <= train_end)].reset_index(drop=True)
    val_df = df[(df['date'] >= val_start) & (df['date'] <= val_end)].reset_index(drop=True)
    
    logger.info("Split data into %d training and %d validation records", len(train_df), len(val_df))
    logger.debug("Sample of train_df: %s", train_df[['contractID', 'date', 'last']].head().to_string())
    logger.debug("Sample of val_df: %s", val_df[['contractID', 'date', 'last']].head().to_string())
    
    return train_df, val_df

def create_env(data, config, max_holding_days_override=None, flatten_observation=False):
    """Creates an instance of the EnhancedCapitalConstrainedEnv.
    
    Args:
        data: DataFrame containing the environment data
        config: Configuration dictionary
        max_holding_days_override: Override the max_holding_days from config
        flatten_observation: Whether to flatten Dict observation space to Box
    """
    from enhanced_capital_constrained_env_new import EnhancedCapitalConstrainedEnv # Local import
    try:
        # Use the max_holding_days from config if not overridden
        if max_holding_days_override is None:
            max_holding_days_override = config.get('max_holding_days', 10)
            logger.info(f"Using max_holding_days from config: {max_holding_days_override}")
        
        env_kwargs = {
            'data': data,
            'config': config,
        }
        if max_holding_days_override is not None:
            env_kwargs['max_holding_days_override'] = max_holding_days_override
            logger.info(f"Creating env with max_holding_days: {max_holding_days_override}")
        env = EnhancedCapitalConstrainedEnv(**env_kwargs)
        
        # Apply FlattenDictObservationWrapper if requested
        if flatten_observation:
            from action_wrapper import FlattenDictObservationWrapper
            if isinstance(env.observation_space, gym.spaces.Dict):
                logger.info("Applying FlattenDictObservationWrapper to environment")
                env = FlattenDictObservationWrapper(env)
                logger.info(f"Flattened observation space to shape {env.observation_space.shape}")
            else:
                logger.warning("Cannot apply FlattenDictObservationWrapper - observation space is not Dict")
        
        return env
    except Exception as e:
        logger.error(f"Error creating environment: {e}", exc_info=True)
        return None

def create_scaler(data, config):
    """Creates and fits a scaler based on the provided data and config."""
    numerical_cols = config.get('numerical_features', [])
    # Ensure only existing columns are scaled
    numerical_cols = [col for col in numerical_cols if col in data.columns]
    
    if not numerical_cols:
        logger.warning("No numerical columns found or specified for scaling.")
        return None
        
    scaler = RobustScaler()
    try:
        scaler.fit(data[numerical_cols])
        logger.info("Scaler fitted successfully.")
        return scaler
    except Exception as e:
        logger.error(f"Error fitting scaler: {e}", exc_info=True)
        return None

# --- Evaluation Function (Moved from evaluate.py) ---
def fast_evaluate_model(system, test_env, n_eval_episodes=3, max_steps_per_episode=100, early_stop_threshold=None):
    """
    Efficiently evaluate the trained model on the test environment.
    
    Args:
        system: The AdvancedRLSystem containing the model to evaluate
        test_env: The wrapped test environment (or the system will wrap it)
        n_eval_episodes: Maximum number of episodes to evaluate (default: 3)
        max_steps_per_episode: Maximum steps per episode (default: 100)
        early_stop_threshold: If mean reward crosses this threshold, stop early (default: None)
        
    Returns:
        Dictionary of evaluation metrics
    """
    metrics = {
        'episode_rewards': [],
        'episode_lengths': [],
        'portfolio_values': [], # Aggregate portfolio values across steps
        'execution_times': [],
        'positions_taken': 0,
        'successful_trades': 0,
        'contracts_evaluated': set()  # Track unique contracts evaluated
    }
    
    # Use logger defined within utils.py or pass one in
    # Assuming logger is available globally or configured in utils.py
    logger = logging.getLogger(__name__) # Get logger instance

    total_start_time = time.time()
    logger.info(f"Starting fast evaluation: {n_eval_episodes} episodes, max {max_steps_per_episode} steps each")
    
    # IMPORTANT: When called from Optuna, test_env is the specifically created eval env.
    # When called from evaluate.py, test_env might be system.env. Needs careful handling.
    # Let's assume the function receives the correct environment instance to use.
    eval_env = test_env 
    if eval_env is None:
        logger.error("Evaluation environment is None. Cannot evaluate.")
        return None

    # Check observation space compatibility
    try:
        if hasattr(system, 'model') and hasattr(system.model, 'observation_space'):
            model_obs_shape = system.model.observation_space.shape
            eval_obs_shape = eval_env.observation_space.shape
            
            logger.info(f"Model expects observation shape {model_obs_shape}, " 
                      f"eval env provides {eval_obs_shape}")
            
            if model_obs_shape != eval_obs_shape:
                logger.error(f"Observation shape mismatch: model expects {model_obs_shape}, "
                           f"environment provides {eval_obs_shape}")
                logger.error("This will cause errors during prediction. Aborting evaluation.")
                return {
                    'error': 'observation_shape_mismatch',
                    'model_shape': model_obs_shape,
                    'env_shape': eval_obs_shape,
                    'mean_reward': -np.inf
                }
    except Exception as e:
        logger.warning(f"Could not verify observation space compatibility: {e}")

    # Check the data size in the evaluation environment if possible
    try:
        if hasattr(eval_env, 'env') and hasattr(eval_env.env, 'data_df'):
            data_size = len(eval_env.env.data_df)
            unique_contracts = eval_env.env.data_df['contractID'].nunique() if 'contractID' in eval_env.env.data_df.columns else 0
            logger.info(f"Evaluation environment data: {data_size} rows, {unique_contracts} unique contracts")
    except Exception as e:
        logger.warning(f"Could not determine evaluation data size: {e}")

    for i in range(n_eval_episodes):
        logger.info(f"Evaluating episode {i+1}/{n_eval_episodes}...")
        episode_reward = 0
        episode_start_time = time.time()
        step_count = 0
        terminated = False
        truncated = False
        episode_portfolio_values = [] # Track for this episode
        episode_seed = i + 42  # Generate a unique seed for each episode

        # Reset environment - use the provided test_env reset with explicit seed parameter
        logger.debug(f"Resetting environment with seed={episode_seed}")
        reset_output = eval_env.reset(seed=episode_seed)
        if isinstance(reset_output, tuple) and len(reset_output) == 2:
            obs, info = reset_output # Gym API: obs, info
        else:
            obs = reset_output # SB3 VecEnv API: obs
            info = {}
            
        # Try to identify which contract is being evaluated in this episode
        contract_id = None
        try:
            if hasattr(eval_env, 'env') and hasattr(eval_env.env, '_get_contract_id'):
                contract_id = eval_env.env._get_contract_id(eval_env.env.current_step)
                logger.info(f"  Episode {i+1} evaluating contract: {contract_id} (step: {eval_env.env.current_step})")
                metrics['contracts_evaluated'].add(contract_id)
        except Exception as e:
            logger.warning(f"Could not determine contract ID: {e}")
            
        # Episode loop with step limit
        while step_count < max_steps_per_episode:
            # Make prediction using the system's predict method
            try:
                action, _ = system.predict(obs, deterministic=True)
            except Exception as e:
                logger.error(f"Error during prediction: {e}")
                logger.error(f"Observation shape: {obs.shape if hasattr(obs, 'shape') else 'N/A'}")
                logger.error(f"Observation type: {type(obs)}")
                if isinstance(obs, dict):
                    for k, v in obs.items():
                        logger.error(f"  {k}: shape={v.shape if hasattr(v, 'shape') else 'N/A'}, type={type(v)}")
                # End episode on prediction error
                break
            
            # Execute action in the evaluation environment
            try:
                new_obs, reward, terminated, truncated, info = eval_env.step(action)
            except Exception as e:
                logger.error(f"Error during environment step: {e}")
                logger.error(f"Action shape: {action.shape if hasattr(action, 'shape') else 'N/A'}")
                logger.error(f"Action type: {type(action)}")
                # End episode on step error
                break
            
            # Update observation
            obs = new_obs

            # Track metrics
            episode_reward += reward
            step_count += 1
            
            if 'portfolio_value' in info:
                episode_portfolio_values.append(info['portfolio_value'])
            if info.get('position_opened', False):
                metrics['positions_taken'] += 1
            if info.get('trade_successful', False):
                metrics['successful_trades'] += 1
                
            if terminated or truncated:
                # Log termination reason if available
                logger.debug(f"  Episode ended at step {step_count}: terminated={terminated}, truncated={truncated}")
                break # Exit episode loop
                
        # Record episode results
        episode_end_time = time.time()
        metrics['episode_rewards'].append(episode_reward)
        metrics['episode_lengths'].append(step_count)
        metrics['execution_times'].append(episode_end_time - episode_start_time)
        metrics['portfolio_values'].extend(episode_portfolio_values) 
        
        logger.info(f"  Episode {i+1} reward: {episode_reward:.4f} ({step_count} steps, {episode_end_time - episode_start_time:.2f}s)")
        
        if early_stop_threshold is not None:
            if len(metrics['episode_rewards']) >= 2:
                current_mean_reward = np.mean(metrics['episode_rewards'])
                if current_mean_reward > early_stop_threshold:
                    logger.info(f"Early stopping: mean reward {current_mean_reward:.4f} exceeds threshold {early_stop_threshold}")
                    break
    
    total_end_time = time.time()
    
    # Log number of unique contracts evaluated
    metrics['num_unique_contracts'] = len(metrics['contracts_evaluated'])
    metrics['contracts_evaluated'] = list(metrics['contracts_evaluated'])
    logger.info(f"Evaluated {metrics['num_unique_contracts']} unique contracts: {', '.join(metrics['contracts_evaluated'][:5])}" + 
                (f"... and {len(metrics['contracts_evaluated']) - 5} more" if len(metrics['contracts_evaluated']) > 5 else ""))
    
    if metrics['episode_rewards']:
        metrics['mean_reward'] = np.mean(metrics['episode_rewards'])
        metrics['std_reward'] = np.std(metrics['episode_rewards'])
        metrics['mean_episode_length'] = np.mean(metrics['episode_lengths'])
        metrics['mean_execution_time_per_episode'] = np.mean(metrics['execution_times'])
        metrics['total_evaluation_time'] = total_end_time - total_start_time
        
        if metrics['positions_taken'] > 0:
            metrics['trade_success_rate'] = metrics['successful_trades'] / metrics['positions_taken']
        else:
            metrics['trade_success_rate'] = 0.0
            
        logger.info(f"Fast evaluation complete: mean reward {metrics['mean_reward']:.4f} ± {metrics['std_reward']:.4f}")
        logger.info(f"Total evaluation time: {metrics['total_evaluation_time']:.2f} seconds")
    else:
        logger.warning("No episodes were completed during evaluation.")
        metrics['mean_reward'] = np.nan
        metrics['std_reward'] = np.nan
        metrics['mean_episode_length'] = np.nan
        metrics['mean_execution_time_per_episode'] = np.nan
        metrics['total_evaluation_time'] = total_end_time - total_start_time
        metrics['trade_success_rate'] = np.nan
        
    if 'execution_times' in metrics:
        del metrics['execution_times']
        
    return metrics

def get_group_info(df):
    """Calculates group sizes based on the 'date' column."""
    # Ensure sorted by date for correct grouping
    df_sorted = df.sort_values('date')
    group_info = df_sorted.groupby('date').size().tolist()
    logging.debug(f"Calculated group info for {len(group_info)} groups (days). Total samples: {sum(group_info)}.")
    return group_info
