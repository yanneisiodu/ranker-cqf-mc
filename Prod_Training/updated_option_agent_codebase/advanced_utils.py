from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from logger import setup_logger

logger = setup_logger(__name__)
EPS = 1e-12


def _safe_numeric(frame: pd.DataFrame, column: str, fill: float = 0.0) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce").fillna(fill)
    return pd.Series(fill, index=frame.index, dtype=float)


@dataclass(frozen=True)
class SurfaceEncoderConfig:
    n_components: int = 3
    near_dte: int = 7
    mid_dte: int = 21
    long_dte: int = 60
    atm_threshold: float = 0.015
    near_threshold: float = 0.05

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> "SurfaceEncoderConfig":
        cfg = config.get("advanced", {}).get("surface_encoder", {})
        return cls(
            n_components=int(cfg.get("n_components", 3)),
            near_dte=int(cfg.get("near_dte", 7)),
            mid_dte=int(cfg.get("mid_dte", 21)),
            long_dte=int(cfg.get("long_dte", 60)),
            atm_threshold=float(cfg.get("atm_threshold", 0.015)),
            near_threshold=float(cfg.get("near_threshold", 0.05)),
        )


class DateSurfaceEncoder:
    """Builds date-local option-chain summary features plus latent surface factors.

    The latent factors are fit on the training partition only. Per-date summary features remain
    point-in-time safe because they only use same-date chain snapshots.
    """

    def __init__(self, config: SurfaceEncoderConfig):
        self.config = config
        self.imputer_: Optional[SimpleImputer] = None
        self.scaler_: Optional[StandardScaler] = None
        self.pca_: Optional[PCA] = None
        self.latent_columns_: List[str] = []
        self.summary_columns_: List[str] = []
        self.n_components_: int = 0

    def _bucketize(self, frame: pd.DataFrame) -> pd.DataFrame:
        work = frame.copy()
        dist = _safe_numeric(work, "dist_to_atm")
        if "dist_to_atm" not in work.columns:
            strike = _safe_numeric(work, "strike", fill=np.nan)
            spot = _safe_numeric(work, "spy_d_close", fill=np.nan)
            valid = (strike > 0) & (spot > 0)
            dist = pd.Series(np.where(valid, np.abs(np.log(strike / np.clip(spot, EPS, None))), np.nan), index=work.index)
        work["surface_dist_bucket"] = np.select(
            [dist <= self.config.atm_threshold, dist <= self.config.near_threshold],
            ["atm", "near"],
            default="wing",
        )
        dte = _safe_numeric(work, "days_to_exp")
        work["surface_tenor_bucket"] = pd.cut(
            dte,
            bins=[-np.inf, self.config.near_dte, self.config.mid_dte, self.config.long_dte, np.inf],
            labels=["near", "short", "mid", "long"],
            include_lowest=True,
        ).astype(str)
        work["surface_type"] = work.get("type", pd.Series("unknown", index=work.index)).astype(str).str.lower().fillna("unknown")
        work["delta_abs"] = _safe_numeric(work, "delta_abs")
        return work

    def _summary_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        work = self._bucketize(frame)
        grouped = work.groupby("date", sort=False)
        summary = grouped.agg(
            surface_iv_mean=("implied_volatility", "mean"),
            surface_iv_std=("implied_volatility", "std"),
            surface_spread_mean=("relative_spread", "mean"),
            surface_spread_std=("relative_spread", "std"),
            surface_liquidity_mean=("liquidity_score", "mean"),
            surface_delta_abs_mean=("delta_abs", "mean"),
            surface_contract_count=("contractid", "nunique"),
            surface_volume_total=("volume", "sum"),
            surface_oi_total=("open_interest", "sum"),
        )

        call_iv = work[work["surface_type"] == "call"].groupby("date", sort=False)["implied_volatility"].mean()
        put_iv = work[work["surface_type"] == "put"].groupby("date", sort=False)["implied_volatility"].mean()
        near_iv = work[work["surface_tenor_bucket"] == "near"].groupby("date", sort=False)["implied_volatility"].mean()
        long_iv = work[work["surface_tenor_bucket"] == "long"].groupby("date", sort=False)["implied_volatility"].mean()
        atm_iv = work[work["surface_dist_bucket"] == "atm"].groupby("date", sort=False)["implied_volatility"].mean()
        wing_iv = work[work["surface_dist_bucket"] == "wing"].groupby("date", sort=False)["implied_volatility"].mean()

        summary["surface_call_put_iv_spread"] = call_iv.reindex(summary.index) - put_iv.reindex(summary.index)
        summary["surface_term_slope"] = long_iv.reindex(summary.index) - near_iv.reindex(summary.index)
        summary["surface_smile_curvature"] = wing_iv.reindex(summary.index) - atm_iv.reindex(summary.index)
        summary["surface_volume_per_contract"] = summary["surface_volume_total"] / np.clip(summary["surface_contract_count"], 1, None)
        summary = summary.fillna(0.0)
        return summary

    def _latent_panel(self, frame: pd.DataFrame) -> pd.DataFrame:
        work = self._bucketize(frame)
        panel = (
            work.groupby(["date", "surface_type", "surface_tenor_bucket", "surface_dist_bucket"], sort=False)
            .agg(
                iv_mean=("implied_volatility", "mean"),
                spread_mean=("relative_spread", "mean"),
                volume_sum=("volume", "sum"),
                oi_sum=("open_interest", "sum"),
                delta_abs_mean=("delta_abs", "mean"),
                count=("contractid", "count"),
            )
            .unstack(["surface_type", "surface_tenor_bucket", "surface_dist_bucket"])
        )
        if isinstance(panel.columns, pd.MultiIndex):
            panel.columns = [
                "__".join([str(part) for part in col if str(part) != ""]).lower().replace(" ", "_")
                for col in panel.columns.to_flat_index()
            ]
        panel = panel.sort_index(axis=1).fillna(0.0)
        return panel

    def fit(self, frame: pd.DataFrame) -> "DateSurfaceEncoder":
        summary = self._summary_features(frame)
        latent_panel = self._latent_panel(frame)
        self.summary_columns_ = list(summary.columns)
        self.latent_columns_ = list(latent_panel.columns)
        if len(self.latent_columns_) == 0 or len(latent_panel) == 0:
            self.imputer_ = None
            self.scaler_ = None
            self.pca_ = None
            self.n_components_ = 0
            return self

        self.imputer_ = SimpleImputer(strategy="constant", fill_value=0.0)
        latent_values = self.imputer_.fit_transform(latent_panel)
        self.scaler_ = StandardScaler()
        scaled = self.scaler_.fit_transform(latent_values)
        max_components = min(self.config.n_components, scaled.shape[1], max(1, scaled.shape[0] - 1))
        if max_components <= 0:
            self.pca_ = None
            self.n_components_ = 0
            return self
        self.pca_ = PCA(n_components=max_components, random_state=42)
        self.pca_.fit(scaled)
        self.n_components_ = max_components
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        transformed = frame.copy().sort_values(["date", "contractid"]).reset_index(drop=True)
        summary = self._summary_features(transformed)
        summary = summary.reindex(columns=self.summary_columns_ or list(summary.columns), fill_value=0.0)

        latent_features = pd.DataFrame(index=summary.index)
        if self.pca_ is not None and self.imputer_ is not None and self.scaler_ is not None:
            latent_panel = self._latent_panel(transformed)
            latent_panel = latent_panel.reindex(columns=self.latent_columns_, fill_value=0.0)
            latent_values = self.imputer_.transform(latent_panel)
            scaled = self.scaler_.transform(latent_values)
            pcs = self.pca_.transform(scaled)
            latent_features = pd.DataFrame(
                pcs,
                index=summary.index,
                columns=[f"surface_pc_{i + 1}" for i in range(self.n_components_)],
            )

        date_features = pd.concat([summary, latent_features], axis=1).reset_index()
        merged = transformed.merge(date_features, on="date", how="left")
        for col in date_features.columns:
            if col != "date" and col in merged.columns:
                merged[col] = merged[col].fillna(0.0)
        return merged

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.fit(frame).transform(frame)


@dataclass(frozen=True)
class RegimeRouterConfig:
    n_clusters: int = 3
    random_state: int = 42
    expert_blend_weight: float = 0.65
    min_regime_rows: int = 150

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> "RegimeRouterConfig":
        cfg = config.get("advanced", {}).get("regime_router", {})
        return cls(
            n_clusters=int(cfg.get("n_clusters", 3)),
            random_state=int(cfg.get("random_state", 42)),
            expert_blend_weight=float(cfg.get("expert_blend_weight", 0.65)),
            min_regime_rows=int(cfg.get("min_regime_rows", 150)),
        )


class RegimeRouter:
    def __init__(self, config: RegimeRouterConfig):
        self.config = config
        self.imputer_: Optional[SimpleImputer] = None
        self.scaler_: Optional[StandardScaler] = None
        self.kmeans_: Optional[KMeans] = None
        self.state_columns_: List[str] = []
        self.regime_names_: Dict[int, str] = {0: "default"}
        self.n_clusters_: int = 1

    def _candidate_state_columns(self, frame: pd.DataFrame) -> List[str]:
        candidates = [
            "spy_momentum",
            "spy_d_rsi",
            "vix_d_close",
            "iv_vix_ratio",
            "relative_spread",
            "liquidity_score",
            "surface_iv_mean",
            "surface_iv_std",
            "surface_spread_mean",
            "surface_call_put_iv_spread",
            "surface_term_slope",
            "surface_smile_curvature",
            "surface_liquidity_mean",
        ]
        candidates.extend([col for col in frame.columns if col.startswith("surface_pc_")])
        return [col for col in candidates if col in frame.columns and pd.api.types.is_numeric_dtype(frame[col])]

    def _date_state(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.state_columns_:
            self.state_columns_ = self._candidate_state_columns(frame)
        if not self.state_columns_:
            state = pd.DataFrame(index=pd.Index(sorted(pd.to_datetime(frame["date"].unique())), name="date"))
            state["fallback_state"] = 0.0
            self.state_columns_ = ["fallback_state"]
            return state
        state = frame.groupby("date", sort=False)[self.state_columns_].mean().fillna(0.0)
        return state

    def _name_regimes(self, state: pd.DataFrame, labels: np.ndarray) -> Dict[int, str]:
        names: Dict[int, str] = {}
        state = state.copy()
        state["_label"] = labels
        median_momentum = state["spy_momentum"].median() if "spy_momentum" in state.columns else 0.0
        median_vix = state["vix_d_close"].median() if "vix_d_close" in state.columns else 0.0
        for label in sorted(np.unique(labels)):
            subset = state[state["_label"] == label]
            mom = subset["spy_momentum"].mean() if "spy_momentum" in subset.columns else 0.0
            vix = subset["vix_d_close"].mean() if "vix_d_close" in subset.columns else 0.0
            trend_tag = "risk_on" if mom >= median_momentum else "risk_off"
            vol_tag = "high_vol" if vix >= median_vix else "low_vol"
            names[int(label)] = f"{trend_tag}_{vol_tag}"
        return names

    def fit(self, frame: pd.DataFrame) -> "RegimeRouter":
        state = self._date_state(frame)
        self.imputer_ = SimpleImputer(strategy="constant", fill_value=0.0)
        values = self.imputer_.fit_transform(state)
        self.scaler_ = StandardScaler()
        scaled = self.scaler_.fit_transform(values)
        self.n_clusters_ = int(max(1, min(self.config.n_clusters, len(state))))
        if self.n_clusters_ <= 1:
            self.kmeans_ = None
            self.regime_names_ = {0: "default"}
            return self
        self.kmeans_ = KMeans(n_clusters=self.n_clusters_, random_state=self.config.random_state, n_init=10)
        labels = self.kmeans_.fit_predict(scaled)
        self.regime_names_ = self._name_regimes(state, labels)
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        transformed = frame.copy().sort_values(["date", "contractid"]).reset_index(drop=True)
        state = self._date_state(transformed)
        if self.imputer_ is None or self.scaler_ is None:
            raise RuntimeError("RegimeRouter must be fit before calling transform")
        values = self.imputer_.transform(state)
        scaled = self.scaler_.transform(values)

        if self.kmeans_ is None:
            regime_df = pd.DataFrame(index=state.index)
            regime_df["regime_id"] = 0
            regime_df["regime_confidence"] = 1.0
            regime_df["regime_name"] = "default"
            regime_df["regime_prob_0"] = 1.0
        else:
            distances = self.kmeans_.transform(scaled)
            labels = distances.argmin(axis=1)
            shifted = distances - distances.min(axis=1, keepdims=True)
            weights = np.exp(-shifted)
            probs = weights / np.clip(weights.sum(axis=1, keepdims=True), EPS, None)
            confidence = probs.max(axis=1)
            regime_df = pd.DataFrame(index=state.index)
            regime_df["regime_id"] = labels.astype(int)
            regime_df["regime_confidence"] = confidence.astype(float)
            regime_df["regime_name"] = regime_df["regime_id"].map(self.regime_names_).fillna("unknown")
            for regime in range(self.n_clusters_):
                regime_df[f"regime_prob_{regime}"] = probs[:, regime]

        regime_df = regime_df.reset_index()
        merged = transformed.merge(regime_df, on="date", how="left")
        if "regime_confidence" in merged.columns:
            merged["regime_confidence"] = merged["regime_confidence"].fillna(0.0)
        if "regime_id" in merged.columns:
            merged["regime_id"] = merged["regime_id"].fillna(0).astype(int)
        for col in [c for c in merged.columns if c.startswith("regime_prob_")]:
            merged[col] = merged[col].fillna(0.0)
        merged["regime_name"] = merged.get("regime_name", pd.Series("unknown", index=merged.index)).fillna("unknown")
        return merged

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.fit(frame).transform(frame)


@dataclass(frozen=True)
class DriftWatchdogConfig:
    psi_bins: int = 10
    yellow_psi: float = 0.15
    red_psi: float = 0.30
    max_missing_share: float = 0.25
    regime_confidence_floor: float = 0.45
    yellow_action: str = "shadow_only"
    red_action: str = "halt"
    feature_limit: int = 24

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> "DriftWatchdogConfig":
        cfg = config.get("advanced", {}).get("watchdog", {})
        return cls(
            psi_bins=int(cfg.get("psi_bins", 10)),
            yellow_psi=float(cfg.get("yellow_psi", 0.15)),
            red_psi=float(cfg.get("red_psi", 0.30)),
            max_missing_share=float(cfg.get("max_missing_share", 0.25)),
            regime_confidence_floor=float(cfg.get("regime_confidence_floor", 0.45)),
            yellow_action=str(cfg.get("yellow_action", "shadow_only")),
            red_action=str(cfg.get("red_action", "halt")),
            feature_limit=int(cfg.get("feature_limit", 24)),
        )


def _psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    ref = reference[np.isfinite(reference)]
    cur = current[np.isfinite(current)]
    if len(ref) < 5 or len(cur) < 5:
        return 0.0
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.quantile(ref, quantiles)
    edges[0] = -np.inf
    edges[-1] = np.inf
    edges = np.unique(edges)
    if len(edges) <= 2:
        return 0.0
    ref_hist, _ = np.histogram(ref, bins=edges)
    cur_hist, _ = np.histogram(cur, bins=edges)
    ref_prop = np.clip(ref_hist / max(ref_hist.sum(), 1), EPS, None)
    cur_prop = np.clip(cur_hist / max(cur_hist.sum(), 1), EPS, None)
    return float(np.sum((cur_prop - ref_prop) * np.log(cur_prop / ref_prop)))


class DriftWatchdog:
    def __init__(self, config: DriftWatchdogConfig):
        self.config = config
        self.reference_: Dict[str, Dict[str, float]] = {}
        self.feature_columns_: List[str] = []
        self.reference_regime_distribution_: Dict[str, float] = {}

    def _select_features(self, frame: pd.DataFrame, feature_columns: Sequence[str]) -> List[str]:
        numeric = [
            col
            for col in feature_columns
            if col in frame.columns and pd.api.types.is_numeric_dtype(frame[col]) and not frame[col].isna().all()
        ]
        preferred = [
            col
            for col in numeric
            if col.startswith("surface_") or col.startswith("regime_") or col in {"relative_spread", "liquidity_score", "iv_vix_ratio", "spy_momentum", "vix_d_close"}
        ]
        remaining = [col for col in numeric if col not in preferred]
        selected = preferred + remaining
        return selected[: self.config.feature_limit]

    def fit(self, frame: pd.DataFrame, feature_columns: Sequence[str]) -> "DriftWatchdog":
        self.feature_columns_ = self._select_features(frame, feature_columns)
        self.reference_ = {}
        for col in self.feature_columns_:
            series = pd.to_numeric(frame[col], errors="coerce")
            sample = series.dropna().to_numpy(dtype=float)
            if len(sample) > 500:
                sample = sample[np.linspace(0, len(sample) - 1, 500, dtype=int)]
            self.reference_[col] = {
                "median": float(series.median(skipna=True)),
                "missing_share": float(series.isna().mean()),
                "p05": float(series.quantile(0.05)),
                "p95": float(series.quantile(0.95)),
                "sample": sample.tolist(),
            }
        if "regime_name" in frame.columns:
            dist = frame["regime_name"].value_counts(normalize=True)
            self.reference_regime_distribution_ = {str(k): float(v) for k, v in dist.items()}
        return self

    def evaluate(self, frame: pd.DataFrame) -> Dict[str, object]:
        if len(frame) == 0:
            return {
                "alert_level": "red",
                "action": self.config.red_action,
                "retrain_recommended": True,
                "mean_psi": 0.0,
                "max_psi": 0.0,
                "worst_features": [],
                "max_missing_share": 1.0,
                "median_regime_confidence": 0.0,
                "median_uncertainty": 0.0,
                "reasons": ["empty_frame"],
            }
        psi_by_feature: Dict[str, float] = {}
        missing_delta: Dict[str, float] = {}
        out_of_band: Dict[str, float] = {}
        for col in self.feature_columns_:
            if col not in frame.columns or col not in self.reference_:
                continue
            series = pd.to_numeric(frame[col], errors="coerce")
            ref_stats = self.reference_[col]
            psi_by_feature[col] = _psi(
                np.asarray(ref_stats.get("sample", []), dtype=float),
                series.to_numpy(dtype=float),
                bins=self.config.psi_bins,
            )
            missing_delta[col] = float(series.isna().mean() - ref_stats["missing_share"])
            out_of_band[col] = float(((series < ref_stats["p05"]) | (series > ref_stats["p95"])) .mean())

        mean_psi = float(np.mean(list(psi_by_feature.values()))) if psi_by_feature else 0.0
        max_psi = float(np.max(list(psi_by_feature.values()))) if psi_by_feature else 0.0
        worst_features = sorted(psi_by_feature.items(), key=lambda kv: kv[1], reverse=True)[:5]
        max_missing_share = float(max((frame[col].isna().mean() for col in self.feature_columns_ if col in frame.columns), default=0.0))
        median_regime_confidence = float(frame["regime_confidence"].median()) if "regime_confidence" in frame.columns else 1.0

        alert_level = "green"
        action = "proceed"
        if (
            max_psi >= self.config.red_psi
            or max_missing_share >= self.config.max_missing_share
            or median_regime_confidence < self.config.regime_confidence_floor / 2.0
        ):
            alert_level = "red"
            action = self.config.red_action
        elif max_psi >= self.config.yellow_psi or mean_psi >= self.config.yellow_psi / 2.0 or median_regime_confidence < self.config.regime_confidence_floor:
            alert_level = "yellow"
            action = self.config.yellow_action

        reasons: List[str] = []
        if max_psi >= self.config.yellow_psi:
            reasons.append(f"feature_drift:max_psi={max_psi:.3f}")
        if max_missing_share >= self.config.max_missing_share:
            reasons.append(f"missingness:max_missing_share={max_missing_share:.3f}")
        if median_regime_confidence < self.config.regime_confidence_floor:
            reasons.append(f"regime_confidence:median={median_regime_confidence:.3f}")
        if "uncertainty" in frame.columns:
            median_uncertainty = float(pd.to_numeric(frame["uncertainty"], errors="coerce").median())
        else:
            median_uncertainty = 0.0
        if median_uncertainty > 0.75:
            alert_level = "red"
            action = self.config.red_action
            reasons.append(f"prediction_uncertainty:median={median_uncertainty:.3f}")

        return {
            "alert_level": alert_level,
            "action": action,
            "retrain_recommended": alert_level in {"yellow", "red"},
            "mean_psi": mean_psi,
            "max_psi": max_psi,
            "worst_features": [{"feature": feat, "psi": float(val)} for feat, val in worst_features],
            "max_missing_share": max_missing_share,
            "median_regime_confidence": median_regime_confidence,
            "median_uncertainty": median_uncertainty,
            "reasons": reasons,
        }
