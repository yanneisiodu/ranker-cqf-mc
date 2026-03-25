from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prod_hybrid_kelly import run_hybrid_pipeline
from prod_log_return_predictor import train_log_return_predictor
from prod_meta_labeler import train_meta_labeler
from prod_train_ranker import train_ranker
from utils import PurgedWalkForwardSplit, engineer_features, load_config, validate_purged_split


SAMPLE_DATA = ROOT / "tests" / "sample_option_data.csv"


def test_purged_walk_forward_split_has_temporal_gap() -> None:
    dates = pd.date_range("2025-01-01", periods=20, freq="B")
    splitter = PurgedWalkForwardSplit(n_splits=2, purge_days=2, min_train_days=5)
    found = 0
    for train_idx, test_idx in splitter.split(dates):
        validate_purged_split(pd.Series(dates).iloc[train_idx], pd.Series(dates).iloc[test_idx], purge_days=2)
        found += 1
    assert found >= 1


def test_feature_engineering_is_causal() -> None:
    df = pd.DataFrame(
        {
            "contractID": ["A", "A", "A", "B", "B", "B"],
            "date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"] * 2),
            "type": ["call", "call", "call", "put", "put", "put"],
            "strike": [100, 100, 100, 100, 100, 100],
            "last": [1.0, 2.0, 3.0, 1.0, 1.5, 2.0],
            "bid": [0.9, 1.9, 2.9, 0.9, 1.4, 1.9],
            "ask": [1.1, 2.1, 3.1, 1.1, 1.6, 2.1],
            "volume": [10, 20, 30, 10, 20, 30],
            "open_interest": [100, 100, 100, 100, 100, 100],
            "implied_volatility": [0.2, 0.3, 0.4, 0.2, 0.25, 0.3],
            "delta": [0.5, 0.5, 0.5, -0.5, -0.5, -0.5],
            "gamma": [0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
            "theta": [-0.02, -0.02, -0.02, -0.02, -0.02, -0.02],
            "vega": [0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
            "rho": [0.01, 0.01, 0.01, -0.01, -0.01, -0.01],
            "days_to_exp": [10, 9, 8, 10, 9, 8],
            "spy_d_close": [100, 101, 102, 100, 101, 102],
            "spy_d_sma_50": [99, 99.5, 100, 99, 99.5, 100],
            "spy_d_rsi": [50, 55, 60, 50, 55, 60],
            "spy_d_macd_hist": [0.1, 0.2, 0.3, 0.1, 0.2, 0.3],
            "vix_d_close": [20, 21, 22, 20, 21, 22],
            "moneyness": [1.0, 1.01, 1.02, 1.0, 0.99, 0.98],
            "iv_vix_ratio": [0.01, 0.014, 0.018, 0.01, 0.012, 0.014],
            "spy_momentum": [0.0, 0.01, 0.02, 0.0, 0.01, 0.02],
            "fair_value": [1.0, 2.0, 3.0, 1.0, 1.5, 2.0],
        }
    )
    config = load_config(None)
    engineered_1 = engineer_features(df, config)

    df_mutated = df.copy()
    df_mutated.loc[(df_mutated["contractID"] == "A") & (df_mutated["date"] == pd.Timestamp("2025-01-03")), "last"] = 999.0
    engineered_2 = engineer_features(df_mutated, config)

    # Past row should be unchanged when only a future observation changes.
    row_1 = engineered_1[(engineered_1["contractid"] == "A") & (engineered_1["date"] == pd.Timestamp("2025-01-02"))].iloc[0]
    row_2 = engineered_2[(engineered_2["contractid"] == "A") & (engineered_2["date"] == pd.Timestamp("2025-01-02"))].iloc[0]
    assert np.isclose(row_1["price_change_1d"], row_2["price_change_1d"])
    assert (pd.isna(row_1["mid_roll_mean_5"]) and pd.isna(row_2["mid_roll_mean_5"])) or np.isclose(row_1["mid_roll_mean_5"], row_2["mid_roll_mean_5"])


def test_end_to_end_smoke(tmp_path: Path) -> None:
    assert SAMPLE_DATA.exists(), "sample_option_data.csv is required for smoke tests"

    config = load_config(ROOT / "config.yaml")
    config["ranker"].update({"n_splits": 2, "min_fold_train_days": 4, "purge_days": 2, "n_estimators": 20, "top_k_eval": 5})
    config["meta_labeler"].update({"n_splits": 2, "purge_days": 2, "calibration_days": 2, "n_estimators": 20})
    config["return_model"].update({"n_splits": 2, "purge_days": 2, "calibration_days": 2, "n_estimators": 20})
    config["portfolio"].update({"max_positions_per_day": 3, "max_gross_pct": 0.20, "max_position_pct": 0.05})
    config["features"]["numerical"] = [
        feature for feature in config["features"]["numerical"]
        if not feature.startswith("mid_roll_") and not feature.startswith("iv_roll_") and not feature.startswith("volume_roll_")
    ]
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    ranker_result = train_ranker([str(SAMPLE_DATA)], config_file=str(config_path), output_dir=str(tmp_path / "ranker"))
    assert Path(ranker_result["artifact_path"]).exists()

    meta_result = train_meta_labeler(
        [str(SAMPLE_DATA)],
        config_file=str(config_path),
        output_dir=str(tmp_path / "meta"),
        ranker_artifact_path=ranker_result["artifact_path"],
    )
    assert Path(meta_result["artifact_path"]).exists()

    return_result = train_log_return_predictor(
        [str(SAMPLE_DATA)],
        config_file=str(config_path),
        output_dir=str(tmp_path / "return"),
        ranker_artifact_path=ranker_result["artifact_path"],
    )
    assert Path(return_result["artifact_path"]).exists()

    hybrid_result = run_hybrid_pipeline(
        data_files=[str(SAMPLE_DATA)],
        ranker_artifact_path=ranker_result["artifact_path"],
        meta_artifact_path=meta_result["artifact_path"],
        return_artifact_path=return_result["artifact_path"],
        config_file=str(config_path),
        output_dir=str(tmp_path / "hybrid"),
        include_targets=True,
    )
    assert Path(hybrid_result["predictions_path"]).exists()
    assert Path(hybrid_result["metrics_path"]).exists()
    assert "metrics" in hybrid_result
    assert np.isfinite(hybrid_result["metrics"]["ending_capital"])
