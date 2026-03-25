from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from advanced_utils import DateSurfaceEncoder, SurfaceEncoderConfig
from prod_advanced_stack import AdvancedTradeEngine, run_advanced_stack, train_advanced_stack
from prod_agent_watchdog import AgentDecisionService
from utils import load_config, prepare_model_frame

SAMPLE_DATA = ROOT / "tests" / "sample_option_data.csv"


def _tiny_advanced_config(tmp_path: Path) -> Path:
    config = load_config(ROOT / "config.yaml")
    config["ranker"].update({"n_splits": 2, "min_fold_train_days": 4, "purge_days": 2, "n_estimators": 15, "top_k_eval": 5})
    config["meta_labeler"].update({"n_estimators": 15, "calibration_days": 2, "purge_days": 2})
    config["return_model"].update({"n_estimators": 15, "calibration_days": 2, "purge_days": 2})
    config["portfolio"].update({"max_positions_per_day": 3, "max_gross_pct": 0.20, "max_position_pct": 0.05})
    config["advanced"]["training"].update({"calibration_days": 2, "purge_days": 2})
    config["advanced"]["surface_encoder"].update({"n_components": 2})
    config["advanced"]["regime_router"].update({"n_clusters": 2, "min_regime_rows": 40, "expert_blend_weight": 0.50})
    config["features"]["numerical"] = [
        feature for feature in config["features"]["numerical"]
        if not feature.startswith("mid_roll_") and not feature.startswith("iv_roll_") and not feature.startswith("volume_roll_")
    ]
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return config_path


def test_surface_encoder_is_date_local_when_encoder_is_fixed() -> None:
    frame = prepare_model_frame(SAMPLE_DATA, load_config(ROOT / "config.yaml"), include_targets=False)
    subset = frame[frame["date"] <= pd.Timestamp(sorted(frame["date"].unique())[7])].copy().reset_index(drop=True)
    encoder = DateSurfaceEncoder(SurfaceEncoderConfig(n_components=2)).fit(subset)

    transformed_1 = encoder.transform(subset)
    mutated = subset.copy()
    last_date = mutated["date"].max()
    mutated.loc[mutated["date"] == last_date, "implied_volatility"] = 9.99
    transformed_2 = encoder.transform(mutated)

    probe_date = sorted(subset["date"].unique())[2]
    row_1 = transformed_1[transformed_1["date"] == probe_date].iloc[0]
    row_2 = transformed_2[transformed_2["date"] == probe_date].iloc[0]
    assert np.isclose(row_1["surface_iv_mean"], row_2["surface_iv_mean"])
    assert np.isclose(row_1["surface_term_slope"], row_2["surface_term_slope"])
    if "surface_pc_1" in transformed_1.columns:
        assert np.isclose(row_1["surface_pc_1"], row_2["surface_pc_1"])


def test_advanced_stack_end_to_end_and_agent_packet(tmp_path: Path) -> None:
    config_path = _tiny_advanced_config(tmp_path)
    train_result = train_advanced_stack([str(SAMPLE_DATA)], config_file=str(config_path), output_dir=str(tmp_path / "advanced"))
    assert Path(train_result["artifact_path"]).exists()
    assert Path(train_result["metrics_path"]).exists()
    assert "holdout_backtest" in train_result["metrics"]
    assert "holdout_health" in train_result["metrics"]

    run_result = run_advanced_stack(
        data_files=[str(SAMPLE_DATA)],
        advanced_artifact_path=train_result["artifact_path"],
        config_file=str(config_path),
        output_dir=str(tmp_path / "run"),
        include_targets=True,
    )
    assert Path(run_result["predictions_path"]).exists()
    assert Path(run_result["metrics_path"]).exists()
    assert run_result["health"]["action"] in {"proceed", "shadow_only", "halt"}

    service = AgentDecisionService(train_result["artifact_path"], config_file=str(config_path))
    packet = service.decision_packet_from_files([str(SAMPLE_DATA)], include_targets=False)
    assert "execution_mode" in packet
    assert "health" in packet
    assert packet["execution_mode"] in {"proceed", "shadow_only", "halt"}


def test_watchdog_flags_extreme_drift(tmp_path: Path) -> None:
    config_path = _tiny_advanced_config(tmp_path)
    train_result = train_advanced_stack([str(SAMPLE_DATA)], config_file=str(config_path), output_dir=str(tmp_path / "advanced"))
    engine = AdvancedTradeEngine.from_path(train_result["artifact_path"], config_file=str(config_path))
    frame = prepare_model_frame([str(SAMPLE_DATA)], load_config(str(config_path)), include_targets=False)
    drifted = frame.copy()
    drifted["vix_d_close"] = 150.0
    drifted["relative_spread"] = 2.0
    drifted["spy_momentum"] = -5.0
    scored = engine.score_frame(drifted)
    health = engine.health_check(scored)
    assert health["action"] in {"shadow_only", "halt"}
