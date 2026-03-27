from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from logger import setup_logger

logger = setup_logger(__name__)

EPS = 1e-12
TARGET_COLUMNS = {
    "entry_price",
    "exit_price",
    "exit_date",
    "exit_bid",
    "exit_mid_price",
    "exit_spy_d_close",
    "target_raw_return",
    "target_delta_hedged_return",
    "target_return",
    "target_profitable",
    "target_signed_log_return",
    "target_relevance",
}

DEFAULT_CONFIG: Dict[str, object] = {
    "data": {
        "horizon_days": 5,
        "target_mode": "net_long_return",
        "profit_threshold": 0.00,
        "commission_bps": 0.0,
        "min_price_threshold": 0.05,
        "max_relative_spread": 1.0,
        "min_volume": 0,
        "min_open_interest": 0,
        "drop_zero_dte": False,
    },
    "features": {
        "rolling_windows": [5, 20],
        "categorical": ["type"],
        "auto_include_prefixes": ["surface_", "regime_", "news_", "filing_", "embed_", "signal_", "aux_"],
        "numerical": [
            "days_to_exp",
            "strike",
            "last",
            "bid",
            "ask",
            "mid_price",
            "volume",
            "open_interest",
            "implied_volatility",
            "delta",
            "gamma",
            "theta",
            "vega",
            "rho",
            "spy_d_close",
            "spy_d_sma_50",
            "spy_d_rsi",
            "spy_d_macd_hist",
            "vix_d_close",
            "moneyness",
            "relative_spread",
            "bid_ask_spread",
            "liquidity_score",
            "iv_vix_ratio",
            "spy_momentum",
            "price_change_1d",
            "iv_change_1d",
            "mid_roll_mean_5",
            "mid_roll_std_5",
            "mid_roll_mean_20",
            "mid_roll_std_20",
            "iv_roll_mean_5",
            "iv_roll_std_5",
            "volume_roll_mean_5",
            "volume_roll_mean_20",
            "volume_oi_ratio",
            "mispricing_ratio",
            "theta_carry",
            "vega_per_dollar",
            "gamma_per_dollar",
            "time_value_ratio",
            "delta_abs",
            "dist_to_atm",
            "quote_efficiency",
            "mid_price_pct_rank",
            "volume_pct_rank",
            "open_interest_pct_rank",
            "iv_pct_rank",
            "relative_spread_pct_rank",
            "moneyness_pct_rank",
        ],
    },
    "ranker": {
        "n_splits": 4,
        "purge_days": 5,
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "top_k_eval": 20,
        "relevance_bins": 5,
        "min_fold_train_days": 10,
    },
    "meta_labeler": {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "calibration_days": 5,
        "n_splits": 4,
        "purge_days": 5,
    },
    "return_model": {
        "n_estimators": 120,
        "learning_rate": 0.05,
        "max_depth": 3,
        "subsample": 0.8,
        "calibration_days": 5,
        "n_splits": 4,
        "purge_days": 5,
    },
    "portfolio": {
        "starting_capital": 100000.0,
        "kelly_fraction": 0.25,
        "min_prob_to_trade": 0.55,
        "min_expected_return": 0.02,
        "max_position_pct": 0.05,
        "max_gross_pct": 0.50,
        "max_positions_per_day": 5,
        "max_expiration_pct": 0.15,
        "max_abs_delta": 0.60,
        "max_abs_gamma": 0.20,
        "max_abs_vega": 0.60,
        "min_trade_pct": 0.0025,
        "max_relative_spread": 0.25,
    },
    "paths": {
        "output_dir": "./model_output",
    },
    "platform": {
        "target_cloud": "gcp",
        "notes": "Designed to be wrapped by Vertex AI / Cloud Run services."
    }
}


@dataclass(frozen=True)
class SplitWindow:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


class PurgedWalkForwardSplit:
    """Expanding-window walk-forward split with date-level purging."""

    def __init__(self, n_splits: int = 4, purge_days: int = 5, min_train_days: int = 10):
        if n_splits < 1:
            raise ValueError("n_splits must be at least 1")
        if purge_days < 0:
            raise ValueError("purge_days must be non-negative")
        self.n_splits = n_splits
        self.purge_days = purge_days
        self.min_train_days = min_train_days

    def split(self, dates: Sequence[pd.Timestamp]):
        date_series = pd.Series(pd.to_datetime(dates)).reset_index(drop=True)
        unique_dates = np.sort(date_series.dropna().unique())
        if len(unique_dates) <= self.min_train_days + 1:
            raise ValueError("Not enough unique dates for a purged walk-forward split")

        remaining = len(unique_dates) - self.min_train_days
        test_block = max(1, remaining // self.n_splits)

        for split_idx in range(self.n_splits):
            test_start_pos = self.min_train_days + split_idx * test_block
            if test_start_pos >= len(unique_dates):
                break
            test_end_pos = min(len(unique_dates), test_start_pos + test_block)
            test_dates = unique_dates[test_start_pos:test_end_pos]
            if len(test_dates) == 0:
                continue
            purge_cutoff = pd.Timestamp(test_dates[0]) - pd.Timedelta(days=self.purge_days)
            train_mask = date_series < purge_cutoff
            test_mask = date_series.isin(test_dates)
            train_idx = date_series.index[train_mask].to_numpy()
            test_idx = date_series.index[test_mask].to_numpy()
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            yield train_idx, test_idx

    def describe(self, dates: Sequence[pd.Timestamp]) -> List[SplitWindow]:
        date_series = pd.Series(pd.to_datetime(dates)).reset_index(drop=True)
        windows: List[SplitWindow] = []
        for train_idx, test_idx in self.split(date_series):
            train_dates = date_series.iloc[train_idx]
            test_dates = date_series.iloc[test_idx]
            windows.append(
                SplitWindow(
                    train_start=pd.Timestamp(train_dates.min()),
                    train_end=pd.Timestamp(train_dates.max()),
                    test_start=pd.Timestamp(test_dates.min()),
                    test_end=pd.Timestamp(test_dates.max()),
                )
            )
        return windows


def deep_update(base: MutableMapping[str, object], override: Mapping[str, object]) -> MutableMapping[str, object]:
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            base[key] = deep_update(dict(base[key]), value)  # type: ignore[index]
        else:
            base[key] = value
    return base


def load_config(config_file: Optional[Union[str, os.PathLike]] = None) -> Dict[str, object]:
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    if config_file and Path(config_file).exists():
        with open(config_file, "r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise ValueError("Config file must deserialize to a mapping")
        config = deep_update(config, loaded)
    return config


def save_json(data: Mapping[str, object], path: Union[str, os.PathLike]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=str)


def safe_divide(numerator: Union[pd.Series, np.ndarray, float], denominator: Union[pd.Series, np.ndarray, float], fill_value: float = 0.0):
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    out = np.full_like(num, fill_value, dtype=float)
    mask = np.isfinite(num) & np.isfinite(den) & (np.abs(den) > EPS)
    out[mask] = num[mask] / den[mask]
    return out


def signed_log1p(values: Union[pd.Series, np.ndarray, float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return np.sign(arr) * np.log1p(np.abs(arr))


def inverse_signed_log1p(values: Union[pd.Series, np.ndarray, float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return np.sign(arr) * (np.expm1(np.abs(arr)))


_COLUMN_ALIASES = {
    "contract_id": "contractid",
    "option_symbol": "contractid",
    "option_type": "type",
    "mid": "mid_price",
    "mark": "mid_price",
}


_NUMERIC_COERCE_COLUMNS = [
    "strike",
    "last",
    "bid",
    "bid_size",
    "ask",
    "ask_size",
    "mid_price",
    "volume",
    "open_interest",
    "implied_volatility",
    "delta",
    "gamma",
    "theta",
    "vega",
    "rho",
    "days_to_exp",
    "price_change",
    "spy_d_open",
    "spy_d_close",
    "spy_d_high",
    "spy_d_low",
    "spy_d_volume",
    "spy_d_sma_50",
    "spy_d_rsi",
    "spy_d_macd_hist",
    "spy_d_bollinger_upper",
    "vix_d_close",
    "moneyness",
    "iv_vix_ratio",
    "bid_ask_spread",
    "liquidity_score",
    "spy_momentum",
    "fair_value",
    "vanna",
    "charm",
    "vomma",
    "speed",
    "zomma",
    "color",
    "ultima",
    "lambda_greek",
    "underlying_price",
    "realized_vol_5d",
    "realized_vol_20d",
    "realized_vol_60d",
    "vrp_20d",
    "fred_dgs1",
    "fred_dgs10",
    "fred_dgs2",
    "fred_dgs30",
    "fred_dgs3mo",
    "fred_dgs5",
    "fred_dgs6mo",
]


def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.copy()
    renamed.columns = [str(col).strip().lower() for col in renamed.columns]
    renamed = renamed.rename(columns={key: value for key, value in _COLUMN_ALIASES.items() if key in renamed.columns})
    if "date" not in renamed.columns:
        raise ValueError("Input data must contain a 'date' column")
    renamed["date"] = pd.to_datetime(renamed["date"], errors="coerce")
    renamed = renamed.dropna(subset=["date"])
    if "contractid" not in renamed.columns:
        raise ValueError("Input data must contain a contract identifier column")
    renamed["contractid"] = renamed["contractid"].astype(str)
    if "type" in renamed.columns:
        renamed["type"] = renamed["type"].astype(str).str.lower().str.strip()
    for column in _NUMERIC_COERCE_COLUMNS:
        if column in renamed.columns:
            renamed[column] = pd.to_numeric(renamed[column], errors="coerce")
    if "expiration" in renamed.columns:
        renamed["expiration"] = pd.to_datetime(renamed["expiration"], errors="coerce")
    return renamed.sort_values(["date", "contractid"]).reset_index(drop=True)


def load_market_data(file_paths: Union[str, os.PathLike, Sequence[Union[str, os.PathLike]]], nrows: Optional[int] = None) -> pd.DataFrame:
    if isinstance(file_paths, (str, os.PathLike)):
        paths: List[Union[str, os.PathLike]] = [file_paths]
    else:
        paths = list(file_paths)
    frames = []
    for path in paths:
        frame = pd.read_csv(path, low_memory=False, nrows=nrows)
        frames.append(canonicalize_columns(frame))
    if not frames:
        raise ValueError("No input files were provided")
    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values(["date", "contractid"]).reset_index(drop=True)


def _compute_intrinsic_value(df: pd.DataFrame) -> np.ndarray:
    if "type" not in df.columns or "spy_d_close" not in df.columns or "strike" not in df.columns:
        return np.zeros(len(df), dtype=float)
    option_type = df["type"].fillna("")
    spot = df["spy_d_close"].to_numpy(dtype=float)
    strike = df["strike"].to_numpy(dtype=float)
    intrinsic = np.where(option_type == "call", np.maximum(spot - strike, 0.0), np.maximum(strike - spot, 0.0))
    return intrinsic


def _add_cross_sectional_ranks(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    for column in columns:
        if column not in df.columns:
            continue
        df[f"{column}_pct_rank"] = df.groupby("date")[column].rank(pct=True, method="average")
    return df


def engineer_features(df: pd.DataFrame, config: Mapping[str, object]) -> pd.DataFrame:
    frame = canonicalize_columns(df)
    cfg_data = config.get("data", {})
    rolling_windows = [int(window) for window in config.get("features", {}).get("rolling_windows", [5, 20])]

    required = ["last", "bid", "ask", "volume", "open_interest", "days_to_exp"]
    for column in required:
        if column not in frame.columns:
            frame[column] = np.nan

    frame["mid_price"] = np.where(
        frame[["bid", "ask"]].notna().all(axis=1) & (frame["bid"] > 0) & (frame["ask"] > 0),
        (frame["bid"] + frame["ask"]) / 2.0,
        frame["last"],
    )
    frame["bid_ask_spread"] = np.where(
        frame[["bid", "ask"]].notna().all(axis=1),
        (frame["ask"] - frame["bid"]).clip(lower=0.0),
        frame.get("bid_ask_spread", np.nan),
    )
    frame["relative_spread"] = safe_divide(frame["bid_ask_spread"], frame["mid_price"], fill_value=np.nan)
    frame["delta_abs"] = frame.get("delta", pd.Series(np.nan, index=frame.index)).abs()
    frame["dist_to_atm"] = np.abs(np.log(np.clip(frame.get("strike", pd.Series(np.nan, index=frame.index)), EPS, None) /
                                           np.clip(frame.get("spy_d_close", pd.Series(np.nan, index=frame.index)), EPS, None)))
    frame["volume_oi_ratio"] = safe_divide(frame["volume"], frame["open_interest"].replace(0, np.nan), fill_value=0.0)
    frame["quote_efficiency"] = safe_divide(frame["volume"] + np.log1p(frame["open_interest"]), 1.0 + frame["relative_spread"].abs(), fill_value=0.0)

    if "fair_value" not in frame.columns:
        frame["fair_value"] = np.nan
    frame["mispricing_ratio"] = safe_divide(frame["mid_price"] - frame["fair_value"], frame["fair_value"], fill_value=0.0)
    frame["theta_carry"] = safe_divide(frame.get("theta", pd.Series(np.nan, index=frame.index)), frame["mid_price"], fill_value=0.0)
    frame["vega_per_dollar"] = safe_divide(frame.get("vega", pd.Series(np.nan, index=frame.index)), frame["mid_price"], fill_value=0.0)
    frame["gamma_per_dollar"] = safe_divide(frame.get("gamma", pd.Series(np.nan, index=frame.index)) * frame.get("spy_d_close", pd.Series(np.nan, index=frame.index)), frame["mid_price"], fill_value=0.0)

    intrinsic = _compute_intrinsic_value(frame)
    frame["time_value_ratio"] = safe_divide(frame["mid_price"] - intrinsic, frame["mid_price"], fill_value=0.0)

    if "liquidity_score" not in frame.columns or frame["liquidity_score"].isna().all():
        frame["liquidity_score"] = (
            np.log1p(frame["volume"].clip(lower=0.0))
            + 0.5 * np.log1p(frame["open_interest"].clip(lower=0.0))
            - frame["relative_spread"].fillna(frame["relative_spread"].median())
        )

    frame = frame.sort_values(["contractid", "date"]).reset_index(drop=True)
    grouped = frame.groupby("contractid", sort=False)
    prev_mid = grouped["mid_price"].shift(1)
    frame["price_change_1d"] = safe_divide(frame["mid_price"] - prev_mid, prev_mid, fill_value=0.0)
    if "implied_volatility" not in frame.columns:
        frame["implied_volatility"] = np.nan
    prev_iv = grouped["implied_volatility"].shift(1)
    frame["iv_change_1d"] = (frame["implied_volatility"] - prev_iv).fillna(0.0)

    for window in rolling_windows:
        min_periods = 1 if window < 4 else max(2, window // 3)
        frame[f"mid_roll_mean_{window}"] = grouped["mid_price"].transform(lambda s: s.shift(1).rolling(window, min_periods=min_periods).mean())
        frame[f"mid_roll_std_{window}"] = grouped["mid_price"].transform(lambda s: s.shift(1).rolling(window, min_periods=min_periods).std())
        frame[f"iv_roll_mean_{window}"] = grouped["implied_volatility"].transform(lambda s: s.shift(1).rolling(window, min_periods=min_periods).mean())
        frame[f"iv_roll_std_{window}"] = grouped["implied_volatility"].transform(lambda s: s.shift(1).rolling(window, min_periods=min_periods).std())
        frame[f"volume_roll_mean_{window}"] = grouped["volume"].transform(lambda s: s.shift(1).rolling(window, min_periods=min_periods).mean())

    frame = frame.sort_values(["date", "contractid"]).reset_index(drop=True)
    frame = _add_cross_sectional_ranks(
        frame,
        ["mid_price", "volume", "open_interest", "implied_volatility", "relative_spread", "moneyness"],
    )

    min_price = float(cfg_data.get("min_price_threshold", 0.05))
    max_rel_spread = float(cfg_data.get("max_relative_spread", 1.0))
    min_volume = float(cfg_data.get("min_volume", 0))
    min_oi = float(cfg_data.get("min_open_interest", 0))

    mask = frame["mid_price"].fillna(0.0) >= min_price
    mask &= frame["relative_spread"].fillna(np.inf) <= max_rel_spread
    mask &= (frame["volume"].fillna(0.0) >= min_volume) | (frame["open_interest"].fillna(0.0) >= min_oi)
    if bool(cfg_data.get("drop_zero_dte", False)):
        mask &= frame["days_to_exp"].fillna(0.0) > 0

    frame = frame.loc[mask].copy()
    frame = frame.sort_values(["date", "contractid"]).reset_index(drop=True)
    return frame


def build_targets(df: pd.DataFrame, config: Mapping[str, object]) -> pd.DataFrame:
    frame = df.sort_values(["contractid", "date"]).copy()
    cfg = config.get("data", {})
    horizon = int(cfg.get("horizon_days", 5))
    target_mode = str(cfg.get("target_mode", "net_long_return"))
    profit_threshold = float(cfg.get("profit_threshold", 0.0))
    commission_bps = float(cfg.get("commission_bps", 0.0)) / 10000.0

    grouped = frame.groupby("contractid", sort=False)
    frame["entry_price"] = np.where(frame["ask"].fillna(0.0) > 0.0, frame["ask"], frame["mid_price"])
    frame["exit_bid"] = grouped["bid"].shift(-horizon)
    frame["exit_mid_price"] = grouped["mid_price"].shift(-horizon)
    frame["exit_price"] = np.where(frame["exit_bid"].fillna(0.0) > 0.0, frame["exit_bid"], frame["exit_mid_price"])
    frame["exit_date"] = grouped["date"].shift(-horizon)
    frame["exit_spy_d_close"] = grouped["spy_d_close"].shift(-horizon) if "spy_d_close" in frame.columns else np.nan

    raw_option_return = safe_divide(frame["exit_price"] - frame["entry_price"], frame["entry_price"], fill_value=np.nan)
    if "delta" in frame.columns and "spy_d_close" in frame.columns:
        delta_hedged_pnl = (frame["exit_price"] - frame["entry_price"]) - frame["delta"] * (frame["exit_spy_d_close"] - frame["spy_d_close"])
        delta_hedged_return = safe_divide(delta_hedged_pnl, frame["entry_price"], fill_value=np.nan)
    else:
        delta_hedged_return = np.full(len(frame), np.nan)

    frame["target_raw_return"] = raw_option_return - commission_bps
    frame["target_delta_hedged_return"] = delta_hedged_return - commission_bps

    if target_mode == "net_delta_hedged_return":
        frame["target_return"] = frame["target_delta_hedged_return"]
    else:
        frame["target_return"] = frame["target_raw_return"]

    frame = frame[np.isfinite(frame["entry_price"]) & (frame["entry_price"] > 0)].copy()
    frame = frame[np.isfinite(frame["exit_price"]) & (frame["exit_price"] >= 0)].copy()
    frame = frame.dropna(subset=["exit_date", "target_return"])
    bounded = frame["target_return"].clip(lower=-0.999999)
    frame["target_signed_log_return"] = signed_log1p(bounded)
    frame["target_profitable"] = (frame["target_return"] > profit_threshold).astype(int)
    frame = frame.sort_values(["date", "contractid"]).reset_index(drop=True)
    return frame


def prepare_model_frame(
    data: Union[pd.DataFrame, str, os.PathLike, Sequence[Union[str, os.PathLike]]],
    config: Mapping[str, object],
    include_targets: bool = True,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        base = canonicalize_columns(data)
    else:
        base = load_market_data(data, nrows=nrows)
    features = engineer_features(base, config)
    if include_targets:
        return build_targets(features, config)
    return features.sort_values(["date", "contractid"]).reset_index(drop=True)


def preprocess_data(df: pd.DataFrame, config: Mapping[str, object], scaler=None):
    """Compatibility wrapper for older code paths.

    This function intentionally does *not* fit or apply any global scaler, because that would
    leak validation information. Preprocessing is limited to point-in-time-safe feature creation.
    """
    return prepare_model_frame(df, config, include_targets=False), scaler


def select_feature_columns(df: pd.DataFrame, config: Mapping[str, object]) -> Tuple[List[str], List[str], List[str]]:
    feature_cfg = config.get("features", {})
    configured_numeric = [
        col
        for col in feature_cfg.get("numerical", [])
        if col in df.columns and col not in TARGET_COLUMNS and not df[col].isna().all()
    ]
    auto_prefixes = tuple(str(prefix) for prefix in feature_cfg.get("auto_include_prefixes", []))
    auto_numeric = [
        col
        for col in df.columns
        if auto_prefixes
        and col.startswith(auto_prefixes)
        and col not in TARGET_COLUMNS
        and col not in configured_numeric
        and pd.api.types.is_numeric_dtype(df[col])
        and not df[col].isna().all()
    ]
    numeric = configured_numeric + sorted(auto_numeric)
    categorical = [
        col
        for col in feature_cfg.get("categorical", [])
        if col in df.columns and col not in TARGET_COLUMNS and not df[col].isna().all()
    ]
    feature_columns = numeric + categorical
    if not feature_columns:
        raise ValueError("No configured feature columns are present in the dataframe")
    return feature_columns, numeric, categorical


def build_preprocessor(numerical_features: Sequence[str], categorical_features: Sequence[str]) -> ColumnTransformer:
    transformers = []
    if numerical_features:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                list(numerical_features),
            )
        )
    if categorical_features:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                list(categorical_features),
            )
        )
    if not transformers:
        raise ValueError("At least one numerical or categorical feature is required")
    return ColumnTransformer(transformers=transformers, remainder="drop")


def group_sizes_by_date(df: pd.DataFrame) -> List[int]:
    return df.sort_values(["date", "contractid"]).groupby("date").size().tolist()


def compute_relevance_bins(target: pd.Series, n_bins: int = 5) -> np.ndarray:
    values = target.dropna().to_numpy(dtype=float)
    if len(values) == 0:
        raise ValueError("Cannot compute relevance bins from empty target series")
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    edges = np.quantile(values, quantiles)
    return np.asarray(edges, dtype=float)


def apply_relevance_bins(target: Union[pd.Series, np.ndarray], edges: Sequence[float]) -> np.ndarray:
    arr = np.asarray(target, dtype=float)
    return np.digitize(arr, bins=np.asarray(edges, dtype=float), right=True)


def split_train_calibration(dates: Sequence[pd.Timestamp], calibration_days: int, purge_days: int) -> Tuple[np.ndarray, np.ndarray]:
    date_series = pd.Series(pd.to_datetime(dates)).reset_index(drop=True)
    unique_dates = np.sort(date_series.unique())
    if len(unique_dates) < 3:
        raise ValueError("At least 3 unique dates are required for train/calibration splitting")

    effective_calibration_days = int(max(1, min(calibration_days, max(1, len(unique_dates) // 3))))
    effective_purge_days = int(max(0, min(purge_days, max(0, len(unique_dates) - effective_calibration_days - 2))))
    while len(unique_dates) <= effective_calibration_days + effective_purge_days + 1 and effective_purge_days > 0:
        effective_purge_days -= 1
    while len(unique_dates) <= effective_calibration_days + effective_purge_days + 1 and effective_calibration_days > 1:
        effective_calibration_days -= 1
    if len(unique_dates) <= effective_calibration_days + effective_purge_days + 1:
        effective_purge_days = 0
        effective_calibration_days = 1

    calibration_dates = unique_dates[-effective_calibration_days:]
    calibration_start = pd.Timestamp(calibration_dates[0])
    train_cutoff = calibration_start - pd.Timedelta(days=effective_purge_days)
    train_idx = date_series.index[date_series < train_cutoff].to_numpy()
    calibration_idx = date_series.index[date_series.isin(calibration_dates)].to_numpy()
    if len(train_idx) == 0 or len(calibration_idx) == 0:
        raise ValueError("Train/calibration split produced an empty partition")
    return train_idx, calibration_idx


def validate_purged_split(train_dates: Sequence[pd.Timestamp], test_dates: Sequence[pd.Timestamp], purge_days: int) -> None:
    train_dates = pd.to_datetime(pd.Series(train_dates))
    test_dates = pd.to_datetime(pd.Series(test_dates))
    if len(train_dates) == 0 or len(test_dates) == 0:
        raise ValueError("Both train and test dates must be non-empty")
    max_train = pd.Timestamp(train_dates.max())
    min_test = pd.Timestamp(test_dates.min())
    if not max_train < (min_test - pd.Timedelta(days=purge_days)):
        raise AssertionError(
            f"Purged split violated: max train date {max_train} is not earlier than test start {min_test} - {purge_days}d"
        )


def get_output_dir(config: Mapping[str, object], output_dir: Optional[Union[str, os.PathLike]] = None) -> Path:
    root = Path(output_dir or config.get("paths", {}).get("output_dir", "./model_output"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def summarize_frame(df: pd.DataFrame) -> Dict[str, object]:
    return {
        "rows": int(len(df)),
        "unique_dates": int(df["date"].nunique()) if "date" in df.columns else None,
        "date_min": str(df["date"].min()) if "date" in df.columns and len(df) else None,
        "date_max": str(df["date"].max()) if "date" in df.columns and len(df) else None,
        "unique_contracts": int(df["contractid"].nunique()) if "contractid" in df.columns else None,
    }


def daily_top_k_mean_return(df: pd.DataFrame, score_col: str, return_col: str, k: int = 5) -> float:
    samples = []
    for _, group in df.groupby("date"):
        subset = group.sort_values(score_col, ascending=False).head(k)
        if len(subset) == 0:
            continue
        samples.append(float(subset[return_col].mean()))
    return float(np.mean(samples)) if samples else float("nan")


def make_sample_by_dates(df: pd.DataFrame, n_dates: int) -> pd.DataFrame:
    unique_dates = np.sort(df["date"].unique())
    selected = unique_dates[:n_dates]
    return df[df["date"].isin(selected)].copy().reset_index(drop=True)
