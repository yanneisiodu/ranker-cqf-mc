
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
from datetime import datetime
import argparse

from utils import load_config, preprocess_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CONFIG_FILE = "./config.yaml"
MODEL_OUTPUT_PATH = "./model_output/"
TARGET_LOOKAHEAD_DAYS = 5
N_CV_SPLITS = 5
OPTUNA_N_TRIALS = 100
OPTUNA_TIMEOUT = None
NDCG_K = 20
PURGE_DAYS = TARGET_LOOKAHEAD_DAYS

NUMERICAL_FEATURES = [
    'days_to_exp', 'strike', 'last', 'bid', 'ask', 'volume', 'open_interest',
    'implied_volatility', 'delta', 'gamma', 'theta', 'vega', 'rho',
    'spy_d_close', 'spy_d_SMA_50', 'spy_d_RSI', 'spy_d_MACD_Hist',
    'vix_d_close', 'moneyness', 'relative_spread', 'bid_ask_spread',
    'ofi', 'price_change_1d', 'iv_change_1d', 'zero_day_premium',
    'option_volume_oi_ratio', 'mispricing_ratio', 'risk_adjusted_signal',
    'iv_vix_ratio', 'spy_momentum', 'price__mean', 'price__standard_deviation',
]
CATEGORICAL_FEATURES = ['type']

FIXED_PARAMS = {
    'learning_rate': 0.06326509803163745,
    'max_depth': 3,
    'subsample': 0.7822444357631866,
    'colsample_bytree': 0.7333407813790271,
    'gamma': 0.14988133139247198,
    'reg_alpha': 1.0693728241301906e-06,
    'reg_lambda': 1.0903140623130713e-07,
}

class PurgedTimeSeriesSplit:
    def __init__(self, n_splits=5, purge_days=1):
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        self.n_splits = n_splits
        self.purge_days = purge_days

    def split(self, dates):
        dates = pd.Series(dates).reset_index(drop=True)
        unique_dates = dates.sort_values().unique()
        n_dates = len(unique_dates)
        fold_size = n_dates // (self.n_splits + 1)
        test_size = fold_size
        if test_size == 0:
            raise ValueError("Not enough unique dates to create the requested splits")
        for fold in range(self.n_splits):
            train_end_idx = (fold + 1) * test_size
            test_start_idx = train_end_idx
            test_end_idx = test_start_idx + test_size
            if test_end_idx > n_dates:
                test_end_idx = n_dates
            train_dates = unique_dates[:train_end_idx]
            test_dates = unique_dates[test_start_idx:test_end_idx]
            if self.purge_days > 0:
                purge_cutoff = test_dates[0]
                purge_window = pd.Timedelta(days=self.purge_days)
                purge_mask = dates < (pd.Timestamp(purge_cutoff) - purge_window)
                train_idx = dates.index.intersection(dates.index[purge_mask])
            else:
                train_idx = dates[dates.isin(train_dates)].index
            val_idx = dates[dates.isin(test_dates)].index
            yield train_idx, val_idx


def load_raw_data(file_path):
    df = pd.read_csv(file_path, low_memory=False)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    if 'contractID' not in df.columns:
        if 'contract_id' in df.columns:
            df = df.rename(columns={'contract_id': 'contractID'})
        elif 'option_symbol' in df.columns:
            df = df.rename(columns={'option_symbol': 'contractID'})
        else:
            raise ValueError("Missing contract identifier column")
    df['contractID'] = df['contractID'].astype(str)
    return df.sort_values(['date', 'contractID']).reset_index(drop=True)


def calculate_target(df, lookahead_days=5):
    price_col = 'last_raw' if 'last_raw' in df.columns else 'last'
    if price_col not in df.columns:
        raise ValueError("Price column missing for target calculation")

    df = df.sort_values(['contractID', 'date']).copy()
    df_grouped = df.groupby('contractID')

    future_prices = {
        f'price_d{i}': df_grouped[price_col].shift(-i)
        for i in range(1, lookahead_days + 1)
    }
    df_future = pd.DataFrame(future_prices, index=df.index)

    daily_returns = pd.DataFrame(index=df.index)
    last_day_price = df[price_col]
    for i in range(1, lookahead_days + 1):
        current_day_price = df_future[f'price_d{i}']
        daily_returns[f'ret_d{i}'] = np.where(
            (last_day_price > 0) & (~last_day_price.isna()) & (~current_day_price.isna()),
            (current_day_price - last_day_price) / last_day_price,
            np.nan,
        )
        last_day_price = current_day_price

    mean_ret = daily_returns.mean(axis=1, skipna=True)
    std_ret = daily_returns.std(axis=1, skipna=True, ddof=1)
    epsilon = 1e-8
    sharpe = np.where(std_ret > epsilon, mean_ret / (std_ret + epsilon), np.nan)
    df['target_5d_sharpe'] = sharpe
    df = df.dropna(subset=['target_5d_sharpe']).sort_values('date')
    return df


def create_preprocessor(numerical_features, categorical_features, available_columns):
    transformers = []
    num_cols = [f for f in numerical_features if f in available_columns]
    cat_cols = [f for f in categorical_features if f in available_columns]
    if num_cols:
        transformers.append(('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
        ]), num_cols))
    if cat_cols:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols))
    if not transformers:
        raise ValueError("No features available for preprocessing")
    return ColumnTransformer(transformers, remainder='drop')


def get_group_info(df):
    return df.sort_values('date').groupby('date').size().tolist()


def objective_rank(trial, X, y, dates, numerical_features, categorical_features, k_ndcg, n_splits, purge_days):
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
    splitter = PurgedTimeSeriesSplit(n_splits=n_splits, purge_days=purge_days)
    scores = []
    feature_cols = [col for col in X.columns if col != 'date']
    num_cols = [f for f in numerical_features if f in feature_cols]
    cat_cols = [f for f in categorical_features if f in feature_cols]

    for fold, (train_idx, val_idx) in enumerate(splitter.split(dates), start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        preprocessor = create_preprocessor(num_cols, cat_cols, feature_cols)
        model = xgb.XGBRanker(**params)
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('ranker', model)
        ])

        X_train_sorted = X_train.sort_values('date')
        y_train_sorted = y_train.loc[X_train_sorted.index]
        X_val_sorted = X_val.sort_values('date')
        y_val_sorted = y_val.loc[X_val_sorted.index]

        train_group = get_group_info(X_train_sorted)
        val_group = get_group_info(X_val_sorted)

        pipeline.named_steps['preprocessor'].fit(X_train_sorted[feature_cols])
        X_train_proc = pipeline.named_steps['preprocessor'].transform(X_train_sorted[feature_cols])
        X_val_proc = pipeline.named_steps['preprocessor'].transform(X_val_sorted[feature_cols])

        pipeline.named_steps['ranker'].fit(
            X_train_proc,
            y_train_sorted,
            group=train_group,
            eval_set=[(X_val_proc, y_val_sorted)],
            eval_group=[val_group],
            verbose=False,
        )
        ndcg = pipeline.named_steps['ranker'].evals_result()['validation_0'][f'ndcg@{k_ndcg}'][-1]
        scores.append(ndcg)
        logging.debug(f"Trial {trial.number} Fold {fold} NDCG@{k_ndcg}: {ndcg:.4f}")

    score = np.nanmean(scores)
    logging.info(f"Trial {trial.number} average NDCG@{k_ndcg}: {score:.4f}")
    return 1.0 - score if not np.isnan(score) else 10.0


def bin_relevance(df, target_col, output_dir, start_year, end_year, timestamp):
    quantiles = df[target_col].quantile([0.25, 0.5, 0.75])
    q1, q2, q3 = quantiles[0.25], quantiles[0.5], quantiles[0.75]
    conditions = [
        df[target_col] <= q1,
        (df[target_col] > q1) & (df[target_col] <= q2),
        (df[target_col] > q2) & (df[target_col] <= q3),
        df[target_col] > q3,
    ]
    df['target_relevance_int'] = np.select(conditions, [0, 1, 2, 3], default=0)
    edges = [-np.inf, q1, q2, q3, np.inf]
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(edges, os.path.join(output_dir, f"sharpe_qcut_edges_{start_year}_{end_year}_{timestamp}.pkl"))
    return df


def train_ranker(data_files, config_file, trials, start_year, end_year, timestamp):
    config = load_config(config_file)
    frames = [load_raw_data(path) for path in data_files]
    df_raw = pd.concat(frames, ignore_index=True).sort_values(['date', 'contractID']).reset_index(drop=True)

    df_processed, _ = preprocess_data(df_raw, config, scaler=None)
    df_processed = df_processed.sort_values(['contractID', 'date'])

    price_col = 'last_raw' if 'last_raw' in df_processed.columns else 'last'
    iv_col = 'implied_volatility_raw' if 'implied_volatility_raw' in df_processed.columns else 'implied_volatility'

    if price_col in df_processed.columns:
        df_processed['price_change_1d'] = df_processed.groupby('contractID')[price_col].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
    else:
        df_processed['price_change_1d'] = 0

    if iv_col in df_processed.columns:
        df_processed['iv_change_1d'] = df_processed.groupby('contractID')[iv_col].diff().fillna(0)
    else:
        df_processed['iv_change_1d'] = 0

    df_target = calculate_target(df_processed, TARGET_LOOKAHEAD_DAYS)
    df_target = bin_relevance(df_target, 'target_5d_sharpe', MODEL_OUTPUT_PATH, start_year, end_year, timestamp)

    feature_cols = [f for f in NUMERICAL_FEATURES + CATEGORICAL_FEATURES if f in df_target.columns]
    X = df_target[['date'] + feature_cols].copy()
    y = df_target['target_relevance_int']

    if 'type' in X.columns:
        X['type'] = X['type'].astype(str)

    dates = X['date']

    if trials > 0:
        study = optuna.create_study(direction='minimize')
        objective = partial(
            objective_rank,
            X=X,
            y=y,
            dates=dates,
            numerical_features=NUMERICAL_FEATURES,
            categorical_features=CATEGORICAL_FEATURES,
            k_ndcg=NDCG_K,
            n_splits=N_CV_SPLITS,
            purge_days=PURGE_DAYS,
        )
        study.optimize(objective, n_trials=trials, timeout=OPTUNA_TIMEOUT)
        best_params = study.best_trial.params if study.best_trial else FIXED_PARAMS.copy()
    else:
        best_params = FIXED_PARAMS.copy()

    final_params = {
        'objective': 'rank:ndcg',
        'eval_metric': f'ndcg@{NDCG_K}',
        'tree_method': 'hist',
        'seed': 42,
        'n_jobs': -1,
        'n_estimators': 1500,
        **best_params,
    }
    final_params.pop('early_stopping_rounds', None)

    preprocessor = create_preprocessor(NUMERICAL_FEATURES, CATEGORICAL_FEATURES, feature_cols)
    model = xgb.XGBRanker(**final_params)
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('ranker', model)
    ])

    X_sorted = X.sort_values('date')
    y_sorted = y.loc[X_sorted.index]
    group_info = get_group_info(X_sorted)

    pipeline.fit(X_sorted[feature_cols], y_sorted, ranker__group=group_info)

    os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)
    features_path = os.path.join(
        MODEL_OUTPUT_PATH,
        f"xgb_feature_names_{start_year}_{end_year}_{'optuna' if trials > 0 else 'fixed'}_{timestamp}.pkl"
    )
    joblib.dump(feature_cols, features_path)
    logging.info(f"Saved feature list to {features_path}")

    model_path = os.path.join(
        MODEL_OUTPUT_PATH,
        f"xgboost_ranker2_{start_year}_{end_year}_{'optuna' if trials > 0 else 'fixed'}_{timestamp}.joblib"
    )
    joblib.dump(pipeline, model_path)
    logging.info(f"Saved ranker pipeline to {model_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2019)
    parser.add_argument("--end-year", type=int, default=2023)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--config", default=CONFIG_FILE)
    args = parser.parse_args()

    if args.start_year == 2019 and args.end_year == 2023:
        data_files = ["../Data/train_2019_2023.csv"]  # Combined data if available
    else:
        # Updated to point to /Data directory (capital D)
        data_files = [f"../Data/year_{year}_data.csv" for year in range(args.start_year, args.end_year + 1)]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_ranker(data_files, args.config, args.trials, args.start_year, args.end_year, timestamp)


if __name__ == "__main__":
    main()
