from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import IsotonicRegression
import xgboost as xgb
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, mean_absolute_error, mean_pinball_loss, mean_squared_error, precision_score, recall_score, roc_auc_score

from advanced_utils import DateSurfaceEncoder, DriftWatchdog, DriftWatchdogConfig, RegimeRouter, RegimeRouterConfig, SurfaceEncoderConfig
from logger import setup_logger
from prod_hybrid_kelly import HybridKellyConfig, HybridKellySizer, evaluate_hybrid_kelly
from prod_train_ranker import RankerConfig, _daily_ndcg, fit_ranker_artifact, predict_ranker_scores
from utils import (
    PurgedWalkForwardSplit,
    apply_relevance_bins,
    build_preprocessor,
    daily_top_k_mean_return,
    get_output_dir,
    inverse_signed_log1p,
    load_config,
    prepare_model_frame,
    save_json,
    select_feature_columns,
    split_train_calibration,
    summarize_frame,
    validate_purged_split,
)

logger = setup_logger(__name__)


@dataclass(frozen=True)
class AdvancedStackConfig:
    calibration_days: int = 5
    purge_days: int = 5

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> "AdvancedStackConfig":
        adv = config.get("advanced", {}).get("training", {})
        fallback_purge = int(config.get("data", {}).get("horizon_days", 5))
        return cls(
            calibration_days=int(adv.get("calibration_days", config.get("meta_labeler", {}).get("calibration_days", 5))),
            purge_days=int(adv.get("purge_days", config.get("meta_labeler", {}).get("purge_days", fallback_purge))),
        )


@dataclass(frozen=True)
class AdvancedMetaConfig:
    n_estimators: int = 300
    learning_rate: float = 0.05
    max_depth: int = 4
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    random_state: int = 42

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> "AdvancedMetaConfig":
        cfg = config.get("meta_labeler", {})
        return cls(
            n_estimators=int(cfg.get("n_estimators", 300)),
            learning_rate=float(cfg.get("learning_rate", 0.05)),
            max_depth=int(cfg.get("max_depth", 4)),
            subsample=float(cfg.get("subsample", 0.8)),
            colsample_bytree=float(cfg.get("colsample_bytree", 0.8)),
            reg_alpha=float(cfg.get("reg_alpha", 0.1)),
            reg_lambda=float(cfg.get("reg_lambda", 1.0)),
        )


@dataclass(frozen=True)
class AdvancedReturnConfig:
    n_estimators: int = 120
    learning_rate: float = 0.05
    max_depth: int = 3
    subsample: float = 0.8
    conformal_alpha: float = 0.10
    random_state: int = 42

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> "AdvancedReturnConfig":
        cfg = config.get("return_model", {})
        adv = config.get("advanced", {}).get("conformal", {})
        return cls(
            n_estimators=int(cfg.get("n_estimators", 120)),
            learning_rate=float(cfg.get("learning_rate", 0.05)),
            max_depth=int(cfg.get("max_depth", 3)),
            subsample=float(cfg.get("subsample", 0.8)),
            conformal_alpha=float(adv.get("alpha", 0.10)),
        )


def _advanced_numeric_features(frame: pd.DataFrame, base_feature_columns: Sequence[str], categorical_features: Sequence[str]) -> List[str]:
    base = list(base_feature_columns)
    categorical_set = set(categorical_features)
    extra = [
        col
        for col in frame.columns
        if (col.startswith("surface_") or col.startswith("regime_"))
        and col not in categorical_set
        and col != "regime_name"
        and pd.api.types.is_numeric_dtype(frame[col])
    ]
    return base + sorted([col for col in extra if col not in base])


def _classifier(cfg: AdvancedMetaConfig, scale_pos_weight: float) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=cfg.random_state,
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        max_depth=cfg.max_depth,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        reg_alpha=cfg.reg_alpha,
        reg_lambda=cfg.reg_lambda,
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
    )


def _fit_calibrator(raw_prob: np.ndarray, y_true: np.ndarray) -> Optional[IsotonicRegression]:
    if len(np.unique(y_true)) < 2:
        return None
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_prob, y_true)
    return calibrator


def _apply_calibrator(calibrator: Optional[IsotonicRegression], raw_prob: np.ndarray) -> np.ndarray:
    if calibrator is None:
        return np.clip(raw_prob, 1e-6, 1 - 1e-6)
    return np.clip(calibrator.predict(raw_prob), 1e-6, 1 - 1e-6)


def _classification_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    pred = (prob >= threshold).astype(int)
    metrics: Dict[str, float] = {
        "brier": float(brier_score_loss(y_true, prob)),
        "log_loss": float(log_loss(y_true, prob, labels=[0, 1])),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "average_precision": float(average_precision_score(y_true, prob)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, prob))
    except ValueError:
        metrics["roc_auc"] = float("nan")
    return metrics


class RegimeAwareMetaModel:
    def __init__(
        self,
        config: AdvancedMetaConfig,
        blend_weight: float,
        min_regime_rows: int,
        feature_columns: Sequence[str],
        categorical_features: Sequence[str],
    ):
        self.config = config
        self.blend_weight = blend_weight
        self.min_regime_rows = min_regime_rows
        self.feature_columns = list(feature_columns)
        self.categorical_features = list(categorical_features)
        self.numerical_features = [col for col in self.feature_columns if col not in set(self.categorical_features)]
        self.preprocessor = build_preprocessor(self.numerical_features, self.categorical_features)
        self.global_model: Optional[xgb.XGBClassifier] = None
        self.expert_models: Dict[int, xgb.XGBClassifier] = {}
        self.calibrator: Optional[IsotonicRegression] = None
        self.metrics_: Dict[str, object] = {}

    def fit(self, train_df: pd.DataFrame, calibration_df: pd.DataFrame) -> "RegimeAwareMetaModel":
        positives = float(train_df["target_profitable"].sum())
        negatives = float(len(train_df) - positives)
        scale_pos_weight = max(1.0, negatives / max(positives, 1.0))

        X_train = self.preprocessor.fit_transform(train_df[self.feature_columns])
        self.global_model = _classifier(self.config, scale_pos_weight)
        self.global_model.fit(X_train, train_df["target_profitable"].to_numpy(dtype=int), verbose=False)

        self.expert_models = {}
        for regime_id, subset in train_df.groupby("regime_id"):
            if len(subset) < self.min_regime_rows or subset["target_profitable"].nunique() < 2:
                continue
            local_pos = float(subset["target_profitable"].sum())
            local_neg = float(len(subset) - local_pos)
            local_spw = max(1.0, local_neg / max(local_pos, 1.0))
            expert = _classifier(self.config, local_spw)
            X_subset = self.preprocessor.transform(subset[self.feature_columns])
            expert.fit(X_subset, subset["target_profitable"].to_numpy(dtype=int), verbose=False)
            self.expert_models[int(regime_id)] = expert

        raw_cal = self.predict_raw(calibration_df)
        self.calibrator = _fit_calibrator(raw_cal, calibration_df["target_profitable"].to_numpy(dtype=int))
        prob = _apply_calibrator(self.calibrator, raw_cal)
        self.metrics_ = _classification_metrics(calibration_df["target_profitable"].to_numpy(dtype=int), prob)
        self.metrics_.update(
            {
                "train_rows": int(len(train_df)),
                "calibration_rows": int(len(calibration_df)),
                "experts_trained": int(len(self.expert_models)),
                "expert_regimes": sorted(int(key) for key in self.expert_models.keys()),
            }
        )
        return self

    def predict_raw(self, frame: pd.DataFrame) -> np.ndarray:
        if self.global_model is None:
            raise RuntimeError("Meta model must be fit before prediction")
        X = self.preprocessor.transform(frame[self.feature_columns])
        global_prob = self.global_model.predict_proba(X)[:, 1]
        final_prob = global_prob.copy()
        if not self.expert_models:
            return final_prob
        regime_conf = frame.get("regime_confidence", pd.Series(1.0, index=frame.index)).to_numpy(dtype=float)
        for regime_id, expert in self.expert_models.items():
            mask = frame["regime_id"].to_numpy(dtype=int) == int(regime_id)
            if not mask.any():
                continue
            expert_prob = expert.predict_proba(X[mask])[:, 1]
            weight = np.clip(self.blend_weight * regime_conf[mask], 0.0, 1.0)
            final_prob[mask] = (1.0 - weight) * global_prob[mask] + weight * expert_prob
        return np.clip(final_prob, 1e-6, 1 - 1e-6)

    def predict_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        scored = frame.copy()
        raw = self.predict_raw(scored)
        scored["prob_profit_raw"] = raw
        scored["prob_profit"] = _apply_calibrator(self.calibrator, raw)
        return scored


class RegimeAwareReturnModel:
    def __init__(
        self,
        config: AdvancedReturnConfig,
        blend_weight: float,
        min_regime_rows: int,
        feature_columns: Sequence[str],
        categorical_features: Sequence[str],
    ):
        self.config = config
        self.blend_weight = blend_weight
        self.min_regime_rows = min_regime_rows
        self.feature_columns = list(feature_columns)
        self.categorical_features = list(categorical_features)
        self.numerical_features = [col for col in self.feature_columns if col not in set(self.categorical_features)]
        self.preprocessor = build_preprocessor(self.numerical_features, self.categorical_features)
        self.models: Dict[str, xgb.XGBRegressor] = {}
        self.expert_models: Dict[int, Dict[str, xgb.XGBRegressor]] = {}
        self.global_qhat_: float = 0.0
        self.regime_qhat_: Dict[int, float] = {}
        self.metrics_: Dict[str, object] = {}

    def _make_regressor(self, objective: str, alpha: Optional[float] = None) -> xgb.XGBRegressor:
        params = {
            "objective": objective,
            "n_estimators": self.config.n_estimators,
            "learning_rate": self.config.learning_rate,
            "max_depth": self.config.max_depth,
            "subsample": self.config.subsample,
            "random_state": self.config.random_state,
            "tree_method": "hist",
            "n_jobs": -1,
        }
        if alpha is not None:
            params["quantile_alpha"] = alpha
        return xgb.XGBRegressor(**params)

    def fit(self, train_df: pd.DataFrame, calibration_df: pd.DataFrame) -> "RegimeAwareReturnModel":
        X_train = self.preprocessor.fit_transform(train_df[self.feature_columns])
        target = train_df["target_signed_log_return"].to_numpy(dtype=float)

        self.models = {
            "q10": self._make_regressor("reg:quantileerror", alpha=0.10),
            "mean": self._make_regressor("reg:squarederror"),
            "q90": self._make_regressor("reg:quantileerror", alpha=0.90),
        }
        for model in self.models.values():
            model.fit(X_train, target)

        self.expert_models = {}
        for regime_id, subset in train_df.groupby("regime_id"):
            if len(subset) < self.min_regime_rows:
                continue
            X_subset = self.preprocessor.transform(subset[self.feature_columns])
            y_subset = subset["target_signed_log_return"].to_numpy(dtype=float)
            expert_group = {
                "q10": self._make_regressor("reg:quantileerror", alpha=0.10),
                "mean": self._make_regressor("reg:squarederror"),
                "q90": self._make_regressor("reg:quantileerror", alpha=0.90),
            }
            for model in expert_group.values():
                model.fit(X_subset, y_subset)
            self.expert_models[int(regime_id)] = expert_group

        cal_preds = self.predict_signed_log(calibration_df, apply_conformal=False)
        cal_true = calibration_df["target_signed_log_return"].to_numpy(dtype=float)
        scores = np.maximum.reduce([
            cal_preds["low"] - cal_true,
            cal_true - cal_preds["high"],
            np.zeros(len(cal_true), dtype=float),
        ])
        self.global_qhat_ = self._conformal_quantile(scores)
        self.regime_qhat_ = {}
        for regime_id, subset in calibration_df.groupby("regime_id"):
            idx = subset.index.to_numpy(dtype=int)
            regime_scores = scores[idx]
            if len(regime_scores) >= max(10, self.min_regime_rows // 4):
                self.regime_qhat_[int(regime_id)] = self._conformal_quantile(regime_scores)

        conformed = self.predict_signed_log(calibration_df, apply_conformal=True)
        low = conformed["low"]
        mean = conformed["mean"]
        high = conformed["high"]
        self.metrics_ = {
            "mae_signed_log": float(mean_absolute_error(cal_true, mean)),
            "rmse_signed_log": float(np.sqrt(mean_squared_error(cal_true, mean))),
            "pinball_q10": float(mean_pinball_loss(cal_true, low, alpha=0.10)),
            "pinball_q90": float(mean_pinball_loss(cal_true, high, alpha=0.90)),
            "interval_80_coverage": float(np.mean((cal_true >= low) & (cal_true <= high))),
            "global_qhat": float(self.global_qhat_),
            "experts_trained": int(len(self.expert_models)),
            "expert_regimes": sorted(int(key) for key in self.expert_models.keys()),
            "train_rows": int(len(train_df)),
            "calibration_rows": int(len(calibration_df)),
            "sign_accuracy": float(np.mean((np.sign(mean) == np.sign(cal_true)).astype(float))),
        }
        return self

    def _conformal_quantile(self, scores: np.ndarray) -> float:
        arr = np.sort(np.asarray(scores, dtype=float))
        if len(arr) == 0:
            return 0.0
        alpha = np.clip(self.config.conformal_alpha, 1e-6, 0.5)
        q_index = int(np.ceil((len(arr) + 1) * (1.0 - alpha))) - 1
        q_index = int(np.clip(q_index, 0, len(arr) - 1))
        return float(arr[q_index])

    def predict_signed_log(self, frame: pd.DataFrame, apply_conformal: bool = True) -> Dict[str, np.ndarray]:
        if not self.models:
            raise RuntimeError("Return model must be fit before prediction")
        X = self.preprocessor.transform(frame[self.feature_columns])
        global_low = self.models["q10"].predict(X)
        global_mean = self.models["mean"].predict(X)
        global_high = self.models["q90"].predict(X)
        base = np.sort(np.vstack([global_low, global_mean, global_high]), axis=0)
        low = base[0].copy()
        mean = base[1].copy()
        high = base[2].copy()

        if self.expert_models:
            regime_conf = frame.get("regime_confidence", pd.Series(1.0, index=frame.index)).to_numpy(dtype=float)
            for regime_id, expert_group in self.expert_models.items():
                mask = frame["regime_id"].to_numpy(dtype=int) == int(regime_id)
                if not mask.any():
                    continue
                local_low = expert_group["q10"].predict(X[mask])
                local_mean = expert_group["mean"].predict(X[mask])
                local_high = expert_group["q90"].predict(X[mask])
                local_low, local_mean, local_high = np.sort(np.vstack([local_low, local_mean, local_high]), axis=0)
                weight = np.clip(self.blend_weight * regime_conf[mask], 0.0, 1.0)
                low[mask] = (1.0 - weight) * low[mask] + weight * local_low
                mean[mask] = (1.0 - weight) * mean[mask] + weight * local_mean
                high[mask] = (1.0 - weight) * high[mask] + weight * local_high

        if apply_conformal:
            qhat = np.full(len(frame), self.global_qhat_, dtype=float)
            if self.regime_qhat_:
                regime_conf = frame.get("regime_confidence", pd.Series(1.0, index=frame.index)).to_numpy(dtype=float)
                for regime_id, regime_qhat in self.regime_qhat_.items():
                    mask = frame["regime_id"].to_numpy(dtype=int) == int(regime_id)
                    if not mask.any():
                        continue
                    weight = np.clip(self.blend_weight * regime_conf[mask], 0.0, 1.0)
                    qhat[mask] = (1.0 - weight) * qhat[mask] + weight * regime_qhat
            low = low - qhat
            high = high + qhat

        low, mean, high = np.sort(np.vstack([low, mean, high]), axis=0)
        return {"low": low, "mean": mean, "high": high}

    def predict_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        scored = frame.copy()
        signed = self.predict_signed_log(scored, apply_conformal=True)
        scored["pred_signed_log_q10"] = signed["low"]
        scored["pred_signed_log_mean"] = signed["mean"]
        scored["pred_signed_log_q90"] = signed["high"]
        scored["pred_return_q10"] = inverse_signed_log1p(signed["low"])
        scored["expected_return"] = inverse_signed_log1p(signed["mean"])
        scored["pred_return_q90"] = inverse_signed_log1p(signed["high"])
        scored["uncertainty"] = scored["pred_return_q90"] - scored["pred_return_q10"]
        return scored


def _generate_advanced_oof_ranker_features(
    base_train_frame: pd.DataFrame,
    config: Mapping[str, object],
    base_feature_columns: Sequence[str],
    categorical_features: Sequence[str],
) -> Tuple[pd.DataFrame, List[Dict[str, object]]]:
    ranker_cfg = RankerConfig.from_config(config)
    surface_cfg = SurfaceEncoderConfig.from_config(config)
    regime_cfg = RegimeRouterConfig.from_config(config)

    working = base_train_frame.copy().sort_values(["date", "contractid"]).reset_index(drop=True)
    oof = pd.DataFrame(index=working.index, data={
        "ranker_score": np.nan,
        "ranker_rank": np.nan,
        "ranker_percentile": np.nan,
    })
    fold_metrics: List[Dict[str, object]] = []
    unique_dates = working["date"].nunique()
    effective_min_train_days = min(ranker_cfg.min_fold_train_days, max(2, unique_dates // 2 - 1))
    effective_n_splits = min(ranker_cfg.n_splits, max(1, unique_dates - effective_min_train_days - 1))
    splitter = PurgedWalkForwardSplit(
        n_splits=effective_n_splits,
        purge_days=ranker_cfg.purge_days,
        min_train_days=effective_min_train_days,
    )

    for fold_number, (train_idx, test_idx) in enumerate(splitter.split(working["date"]), start=1):
        fold_train_base = working.iloc[train_idx].copy().reset_index(drop=True)
        fold_test_base = working.iloc[test_idx].copy().reset_index(drop=True)
        validate_purged_split(fold_train_base["date"], fold_test_base["date"], ranker_cfg.purge_days)

        surface = DateSurfaceEncoder(surface_cfg).fit(fold_train_base)
        fold_train = surface.transform(fold_train_base)
        fold_test = surface.transform(fold_test_base)
        router = RegimeRouter(regime_cfg).fit(fold_train)
        fold_train = router.transform(fold_train)
        fold_test = router.transform(fold_test)

        ranker_feature_columns = _advanced_numeric_features(fold_train, base_feature_columns, categorical_features)
        fold_ranker_artifact = fit_ranker_artifact(fold_train, ranker_cfg, ranker_feature_columns)
        fold_test_scored = predict_ranker_scores(fold_ranker_artifact, fold_test)
        fold_test_scored["target_relevance"] = apply_relevance_bins(fold_test_scored["target_return"], fold_ranker_artifact["relevance_edges"])

        oof.loc[test_idx, ["ranker_score", "ranker_rank", "ranker_percentile"]] = fold_test_scored[
            ["ranker_score", "ranker_rank", "ranker_percentile"]
        ].to_numpy()
        fold_metrics.append(
            {
                "fold": fold_number,
                "train_rows": int(len(fold_train)),
                "test_rows": int(len(fold_test)),
                "train_end": str(fold_train["date"].max()),
                "test_start": str(fold_test["date"].min()),
                "test_end": str(fold_test["date"].max()),
                "ndcg_at_k": _daily_ndcg(fold_test_scored, ranker_cfg.top_k_eval),
                "top_k_mean_return": daily_top_k_mean_return(fold_test_scored, "ranker_score", "target_return", k=min(5, ranker_cfg.top_k_eval)),
            }
        )
    return oof, fold_metrics


class AdvancedTradeEngine:
    def __init__(self, artifact: Dict[str, object], config: Mapping[str, object]):
        self.artifact = artifact
        self.config = dict(config)
        self.hybrid_config = HybridKellyConfig.from_config(config)
        self.sizer = HybridKellySizer(self.hybrid_config)
        self.surface_encoder: DateSurfaceEncoder = artifact["surface_encoder"]
        self.regime_router: RegimeRouter = artifact["regime_router"]
        self.ranker_artifact: Dict[str, object] = artifact["ranker_artifact"]
        self.meta_model: RegimeAwareMetaModel = artifact["meta_model"]
        self.return_model: RegimeAwareReturnModel = artifact["return_model"]
        self.watchdog: DriftWatchdog = artifact["watchdog"]

    @classmethod
    def from_path(cls, artifact_path: str, config_file: Optional[str] = None) -> "AdvancedTradeEngine":
        config = load_config(config_file)
        artifact = joblib.load(artifact_path)
        return cls(artifact=artifact, config=config)

    def _augment(self, frame: pd.DataFrame) -> pd.DataFrame:
        augmented = self.surface_encoder.transform(frame)
        augmented = self.regime_router.transform(augmented)
        return augmented

    def score_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        if len(frame) == 0:
            return frame.copy()
        augmented = self._augment(frame)
        ranked = predict_ranker_scores(self.ranker_artifact, augmented)
        scored = self.meta_model.predict_frame(ranked)
        scored = self.return_model.predict_frame(scored)
        return self.sizer.size_candidates(scored)

    def health_check(self, frame_or_scored: pd.DataFrame) -> Dict[str, object]:
        return self.watchdog.evaluate(frame_or_scored)

    def build_trade_plan(
        self,
        frame: pd.DataFrame,
        as_of_date: Optional[str] = None,
        enforce_watchdog: bool = True,
    ) -> Dict[str, object]:
        scored = self.score_frame(frame)
        health = self.health_check(scored)
        if len(scored) == 0:
            return {
                "execution_mode": health["action"],
                "execute_live": False,
                "health": health,
                "trade_plan": [],
            }
        target_date = pd.Timestamp(as_of_date) if as_of_date else pd.Timestamp(scored["date"].max())
        day_slice = scored[scored["date"] == target_date].copy()
        allocated = self.sizer.allocate_for_date(
            day_slice,
            current_equity=self.hybrid_config.starting_capital,
            available_cash=self.hybrid_config.starting_capital,
            existing_exposures={"delta": 0.0, "gamma": 0.0, "vega": 0.0},
            open_contracts=[],
        )
        if enforce_watchdog and health["action"] == "halt":
            allocated = allocated.iloc[0:0].copy()
        cols = [
            "date",
            "contractid",
            "type",
            "strike",
            "expiration",
            "regime_name",
            "regime_confidence",
            "prob_profit",
            "expected_return",
            "pred_return_q10",
            "pred_return_q90",
            "assigned_weight",
            "selection_score",
            "ranker_score",
            "relative_spread",
            "volume",
            "open_interest",
        ]
        plan_cols = [col for col in cols if col in allocated.columns]
        return {
            "execution_mode": health["action"],
            "execute_live": health["action"] == "proceed",
            "health": health,
            "trade_plan": allocated.sort_values("assigned_weight", ascending=False)[plan_cols].to_dict(orient="records") if len(allocated) else [],
        }


def train_advanced_stack(
    data_files: Sequence[str],
    config_file: Optional[str] = None,
    output_dir: Optional[str] = None,
    nrows: Optional[int] = None,
) -> Dict[str, object]:
    config = load_config(config_file)
    base_frame = prepare_model_frame(data_files, config, include_targets=True, nrows=nrows)
    stack_cfg = AdvancedStackConfig.from_config(config)
    surface_cfg = SurfaceEncoderConfig.from_config(config)
    regime_cfg = RegimeRouterConfig.from_config(config)
    meta_cfg = AdvancedMetaConfig.from_config(config)
    return_cfg = AdvancedReturnConfig.from_config(config)
    ranker_cfg = RankerConfig.from_config(config)

    train_idx, calibration_idx = split_train_calibration(base_frame["date"], stack_cfg.calibration_days, stack_cfg.purge_days)
    train_base = base_frame.iloc[train_idx].copy().reset_index(drop=True)
    calibration_base = base_frame.iloc[calibration_idx].copy().reset_index(drop=True)
    calibration_start = pd.Timestamp(calibration_base["date"].min())
    logger.info("Training advanced stack on %s", summarize_frame(train_base))

    base_feature_columns, _, categorical_features = select_feature_columns(base_frame, config)

    oof_ranker, ranker_fold_metrics = _generate_advanced_oof_ranker_features(train_base, config, base_feature_columns, categorical_features)

    surface_encoder = DateSurfaceEncoder(surface_cfg).fit(train_base)
    train_aug = surface_encoder.transform(train_base)
    calibration_aug = surface_encoder.transform(calibration_base)
    regime_router = RegimeRouter(regime_cfg).fit(train_aug)
    train_aug = regime_router.transform(train_aug)
    calibration_aug = regime_router.transform(calibration_aug)

    ranker_feature_columns = _advanced_numeric_features(train_aug, base_feature_columns, categorical_features)
    train_ranker_artifact = fit_ranker_artifact(train_aug, ranker_cfg, ranker_feature_columns)
    calibration_ranked = predict_ranker_scores(train_ranker_artifact, calibration_aug)
    train_model_df = pd.concat([train_aug, oof_ranker], axis=1).dropna(subset=["ranker_score"]).reset_index(drop=True)
    calibration_model_df = calibration_ranked.copy().reset_index(drop=True)

    model_feature_columns = ranker_feature_columns + ["ranker_score", "ranker_rank", "ranker_percentile"]
    meta_model = RegimeAwareMetaModel(
        config=meta_cfg,
        blend_weight=regime_cfg.expert_blend_weight,
        min_regime_rows=regime_cfg.min_regime_rows,
        feature_columns=model_feature_columns,
        categorical_features=categorical_features,
    ).fit(train_model_df, calibration_model_df)
    return_model = RegimeAwareReturnModel(
        config=return_cfg,
        blend_weight=regime_cfg.expert_blend_weight,
        min_regime_rows=regime_cfg.min_regime_rows,
        feature_columns=model_feature_columns,
        categorical_features=categorical_features,
    ).fit(train_model_df, calibration_model_df)

    watchdog = DriftWatchdog(DriftWatchdogConfig.from_config(config)).fit(train_aug, ranker_feature_columns)

    artifact = {
        "artifact_type": "advanced_trade_stack",
        "config": {
            "stack": asdict(stack_cfg),
            "surface": asdict(surface_cfg),
            "regime": asdict(regime_cfg),
            "meta": asdict(meta_cfg),
            "return": asdict(return_cfg),
        },
        "base_feature_columns": list(base_feature_columns),
        "categorical_features": list(categorical_features),
        "ranker_feature_columns": list(ranker_feature_columns),
        "model_feature_columns": list(model_feature_columns),
        "train_cutoff": str(train_base["date"].max()),
        "calibration_start": str(calibration_start),
        "surface_encoder": surface_encoder,
        "regime_router": regime_router,
        "ranker_artifact": train_ranker_artifact,
        "meta_model": meta_model,
        "return_model": return_model,
        "watchdog": watchdog,
        "train_summary": summarize_frame(train_base),
        "calibration_summary": summarize_frame(calibration_base),
        "metrics": {
            "ranker_cv": ranker_fold_metrics,
            "meta": meta_model.metrics_,
            "return": return_model.metrics_,
        },
    }

    engine = AdvancedTradeEngine(artifact, config)
    eval_frame = base_frame[base_frame["date"] >= calibration_start].copy().reset_index(drop=True)
    holdout_predictions = engine.score_frame(eval_frame)
    holdout_health = engine.health_check(holdout_predictions)
    holdout_backtest = evaluate_hybrid_kelly(holdout_predictions, config)
    holdout_metrics = {key: value for key, value in holdout_backtest.items() if key not in {"equity_curve", "trade_log"}}
    artifact["metrics"]["holdout_backtest"] = holdout_metrics
    artifact["metrics"]["holdout_health"] = holdout_health

    root = get_output_dir(config, output_dir)
    artifact_path = root / "advanced_trade_stack_artifact.joblib"
    metrics_path = root / "advanced_trade_stack_metrics.json"
    holdout_predictions_path = root / "advanced_holdout_predictions.csv"
    holdout_equity_path = root / "advanced_holdout_equity_curve.csv"
    holdout_trades_path = root / "advanced_holdout_trade_log.csv"

    joblib.dump(artifact, artifact_path)
    save_json(artifact["metrics"], metrics_path)
    holdout_predictions.to_csv(holdout_predictions_path, index=False)
    holdout_backtest["equity_curve"].to_csv(holdout_equity_path, index=False)
    holdout_backtest["trade_log"].to_csv(holdout_trades_path, index=False)

    logger.info("Saved advanced trade stack artifact to %s", artifact_path)
    return {
        "artifact_path": str(artifact_path),
        "metrics_path": str(metrics_path),
        "holdout_predictions_path": str(holdout_predictions_path),
        "holdout_equity_curve_path": str(holdout_equity_path),
        "holdout_trade_log_path": str(holdout_trades_path),
        "metrics": artifact["metrics"],
    }


def run_advanced_stack(
    data_files: Sequence[str],
    advanced_artifact_path: str,
    config_file: Optional[str] = None,
    output_dir: Optional[str] = None,
    include_targets: bool = True,
    nrows: Optional[int] = None,
) -> Dict[str, object]:
    config = load_config(config_file)
    frame = prepare_model_frame(data_files, config, include_targets=include_targets, nrows=nrows)
    engine = AdvancedTradeEngine.from_path(advanced_artifact_path, config_file=config_file)
    predictions = engine.score_frame(frame)
    health = engine.health_check(predictions)

    root = get_output_dir(config, output_dir)
    predictions_path = root / "advanced_predictions.csv"
    predictions.to_csv(predictions_path, index=False)
    result: Dict[str, object] = {
        "predictions_path": str(predictions_path),
        "health": health,
    }
    if include_targets:
        backtest = evaluate_hybrid_kelly(predictions, config)
        metrics = {key: value for key, value in backtest.items() if key not in {"equity_curve", "trade_log"}}
        metrics_path = root / "advanced_backtest_metrics.json"
        equity_path = root / "advanced_equity_curve.csv"
        trades_path = root / "advanced_trade_log.csv"
        save_json({"health": health, "metrics": metrics}, metrics_path)
        backtest["equity_curve"].to_csv(equity_path, index=False)
        backtest["trade_log"].to_csv(trades_path, index=False)
        result.update(
            {
                "metrics_path": str(metrics_path),
                "equity_curve_path": str(equity_path),
                "trade_log_path": str(trades_path),
                "metrics": metrics,
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or run the P2/P3 advanced options trade stack")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the advanced stack artifact")
    train_parser.add_argument("--data", nargs="+", required=True, help="CSV file(s) with historical option snapshots")
    train_parser.add_argument("--config", default="./config.yaml", help="Path to YAML config")
    train_parser.add_argument("--output-dir", default=None, help="Directory for output artifacts")
    train_parser.add_argument("--nrows", type=int, default=None, help="Optional row cap for smoke runs")

    run_parser = subparsers.add_parser("run", help="Run the advanced stack on new data")
    run_parser.add_argument("--data", nargs="+", required=True, help="CSV file(s) with option snapshots")
    run_parser.add_argument("--advanced-artifact", required=True, help="Path to advanced trade stack artifact")
    run_parser.add_argument("--config", default="./config.yaml", help="Path to YAML config")
    run_parser.add_argument("--output-dir", default=None, help="Directory for predictions/backtests")
    run_parser.add_argument("--predict-only", action="store_true", help="Skip backtest metrics")
    run_parser.add_argument("--nrows", type=int, default=None, help="Optional row cap for smoke runs")

    args = parser.parse_args()
    if args.command == "train":
        result = train_advanced_stack(args.data, config_file=args.config, output_dir=args.output_dir, nrows=args.nrows)
    else:
        result = run_advanced_stack(
            data_files=args.data,
            advanced_artifact_path=args.advanced_artifact,
            config_file=args.config,
            output_dir=args.output_dir,
            include_targets=not args.predict_only,
            nrows=args.nrows,
        )
    logger.info("Done: %s", result)


if __name__ == "__main__":
    main()
