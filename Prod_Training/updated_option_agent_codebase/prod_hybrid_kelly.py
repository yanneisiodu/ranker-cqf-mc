from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

from logger import setup_logger
from utils import get_output_dir, load_config, prepare_model_frame, save_json, summarize_frame

logger = setup_logger(__name__)


@dataclass(frozen=True)
class HybridKellyConfig:
    starting_capital: float = 100000.0
    kelly_fraction: float = 0.25
    min_prob_to_trade: float = 0.55
    min_expected_return: float = 0.02
    max_position_pct: float = 0.05
    max_gross_pct: float = 0.50
    max_positions_per_day: int = 5
    max_expiration_pct: float = 0.15
    max_abs_delta: float = 0.60
    max_abs_gamma: float = 0.20
    max_abs_vega: float = 0.60
    min_trade_pct: float = 0.0025
    max_relative_spread: float = 0.25
    random_state: int = 42

    @classmethod
    def from_config(cls, config: Dict[str, object]) -> "HybridKellyConfig":
        cfg = config.get("portfolio", {})
        return cls(
            starting_capital=float(cfg.get("starting_capital", 100000.0)),
            kelly_fraction=float(cfg.get("kelly_fraction", 0.25)),
            min_prob_to_trade=float(cfg.get("min_prob_to_trade", 0.55)),
            min_expected_return=float(cfg.get("min_expected_return", 0.02)),
            max_position_pct=float(cfg.get("max_position_pct", 0.05)),
            max_gross_pct=float(cfg.get("max_gross_pct", 0.50)),
            max_positions_per_day=int(cfg.get("max_positions_per_day", 5)),
            max_expiration_pct=float(cfg.get("max_expiration_pct", 0.15)),
            max_abs_delta=float(cfg.get("max_abs_delta", 0.60)),
            max_abs_gamma=float(cfg.get("max_abs_gamma", 0.20)),
            max_abs_vega=float(cfg.get("max_abs_vega", 0.60)),
            min_trade_pct=float(cfg.get("min_trade_pct", 0.0025)),
            max_relative_spread=float(cfg.get("max_relative_spread", 0.25)),
        )


@dataclass
class Position:
    contractid: str
    expiration: Optional[pd.Timestamp]
    quantity: float
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    last_mark: float
    assigned_weight: float
    delta: float
    gamma: float
    vega: float


@dataclass
class ClosedTrade:
    contractid: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    return_pct: float
    assigned_weight: float


def kelly_fraction(prob_win: float, upside: float, downside: float, fraction: float) -> float:
    if prob_win <= 0.0 or prob_win >= 1.0:
        return 0.0
    if upside <= 0.0 or downside <= 0.0:
        return 0.0
    b = upside / downside
    edge = (prob_win * b - (1.0 - prob_win)) / b
    return max(0.0, edge * fraction)


def _clamp_linear_exposure(current: float, coeff: float, limit: float, desired_weight: float) -> float:
    if desired_weight <= 0.0 or abs(coeff) < 1e-12:
        return max(0.0, desired_weight)
    if abs(current + desired_weight * coeff) <= limit:
        return desired_weight
    if coeff > 0:
        max_weight = (limit - current) / coeff
    else:
        max_weight = (-limit - current) / coeff
    return max(0.0, min(desired_weight, max_weight))


class HybridKellySizer:
    def __init__(self, config: HybridKellyConfig):
        self.config = config

    def size_candidates(self, frame: pd.DataFrame) -> pd.DataFrame:
        scored = frame.copy()
        scored["downside_estimate"] = np.maximum(scored["pred_return_q10"].abs(), 0.02)
        scored["upside_estimate"] = np.maximum.reduce([
            scored["pred_return_q90"].to_numpy(dtype=float),
            scored["expected_return"].clip(lower=0.0).to_numpy(dtype=float),
            np.full(len(scored), 0.02),
        ])
        scored["edge"] = scored["prob_profit"] * scored["upside_estimate"] - (1.0 - scored["prob_profit"]) * scored["downside_estimate"]
        scored["base_kelly"] = [
            kelly_fraction(p, u, d, self.config.kelly_fraction)
            for p, u, d in zip(scored["prob_profit"], scored["upside_estimate"], scored["downside_estimate"])
        ]
        spread_penalty = np.clip(1.0 - scored["relative_spread"].fillna(self.config.max_relative_spread) / self.config.max_relative_spread, 0.0, 1.0)
        liquidity_scale = np.clip(
            0.50 * scored.get("volume_pct_rank", 0.5).fillna(0.5)
            + 0.25 * scored.get("open_interest_pct_rank", 0.5).fillna(0.5)
            + 0.25 * scored.get("mid_price_pct_rank", 0.5).fillna(0.5),
            0.25,
            1.0,
        )
        confidence = np.clip((scored["prob_profit"] - self.config.min_prob_to_trade) / max(1e-6, 1 - self.config.min_prob_to_trade), 0.0, 1.0)

        scored["suggested_weight"] = np.minimum(
            self.config.max_position_pct,
            scored["base_kelly"] * spread_penalty * liquidity_scale * np.maximum(confidence, 0.2),
        )
        scored["suggested_weight"] = scored["suggested_weight"].clip(lower=0.0)
        scored["eligible"] = (
            (scored["prob_profit"] >= self.config.min_prob_to_trade)
            & (scored["expected_return"] >= self.config.min_expected_return)
            & (scored["edge"] > 0.0)
            & (scored["relative_spread"].fillna(np.inf) <= self.config.max_relative_spread)
            & (scored["suggested_weight"] >= self.config.min_trade_pct)
        )
        scored["selection_score"] = (
            scored["edge"] * (0.5 + scored["ranker_percentile"].fillna(0.5)) * (0.5 + scored["prob_profit"]) * spread_penalty
        )
        return scored

    def allocate_for_date(
        self,
        date_frame: pd.DataFrame,
        current_equity: float,
        available_cash: float,
        existing_exposures: Dict[str, float],
        open_contracts: Sequence[str],
    ) -> pd.DataFrame:
        candidates = date_frame.copy()
        if len(candidates) == 0:
            return candidates.iloc[0:0].copy()
        candidates = candidates[candidates["eligible"]].copy()
        if len(candidates) == 0:
            return candidates
        candidates = candidates[~candidates["contractid"].isin(open_contracts)].copy()
        if len(candidates) == 0:
            return candidates

        candidates = candidates.sort_values(["selection_score", "ranker_score"], ascending=False).reset_index(drop=True)
        remaining_gross = self.config.max_gross_pct
        expiration_alloc: Dict[object, float] = {}
        exposures = dict(existing_exposures)
        allocations: List[float] = []

        for _, row in candidates.iterrows():
            if len([w for w in allocations if w > 0]) >= self.config.max_positions_per_day:
                allocations.append(0.0)
                continue
            desired = float(min(row["suggested_weight"], remaining_gross, available_cash / max(current_equity, 1e-9)))
            if desired <= 0.0:
                allocations.append(0.0)
                continue

            exp_key = row.get("expiration")
            exp_used = expiration_alloc.get(exp_key, 0.0)
            desired = min(desired, self.config.max_expiration_pct - exp_used)
            desired = _clamp_linear_exposure(exposures.get("delta", 0.0), float(row.get("delta", 0.0)), self.config.max_abs_delta, desired)
            desired = _clamp_linear_exposure(exposures.get("gamma", 0.0), float(row.get("gamma", 0.0)), self.config.max_abs_gamma, desired)
            desired = _clamp_linear_exposure(exposures.get("vega", 0.0), float(row.get("vega", 0.0)), self.config.max_abs_vega, desired)

            if desired < self.config.min_trade_pct:
                allocations.append(0.0)
                continue

            allocations.append(desired)
            remaining_gross -= desired
            available_cash -= desired * current_equity
            expiration_alloc[exp_key] = expiration_alloc.get(exp_key, 0.0) + desired
            exposures["delta"] = exposures.get("delta", 0.0) + desired * float(row.get("delta", 0.0))
            exposures["gamma"] = exposures.get("gamma", 0.0) + desired * float(row.get("gamma", 0.0))
            exposures["vega"] = exposures.get("vega", 0.0) + desired * float(row.get("vega", 0.0))

        allocated = candidates.copy()
        allocated["assigned_weight"] = allocations
        return allocated[allocated["assigned_weight"] > 0].copy().reset_index(drop=True)


class HybridTradeEngine:
    def __init__(
        self,
        ranker_artifact: Dict[str, object],
        meta_artifact: Dict[str, object],
        return_artifact: Dict[str, object],
        config: Dict[str, object],
    ):
        self.ranker_artifact = ranker_artifact
        self.meta_artifact = meta_artifact
        self.return_artifact = return_artifact
        self.config = config
        self.hybrid_config = HybridKellyConfig.from_config(config)
        self.sizer = HybridKellySizer(self.hybrid_config)

    @classmethod
    def from_paths(
        cls,
        ranker_artifact_path: str,
        meta_artifact_path: str,
        return_artifact_path: str,
        config_file: Optional[str] = None,
    ) -> "HybridTradeEngine":
        config = load_config(config_file)
        return cls(
            ranker_artifact=joblib.load(ranker_artifact_path),
            meta_artifact=joblib.load(meta_artifact_path),
            return_artifact=joblib.load(return_artifact_path),
            config=config,
        )

    def score_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        scored = frame.copy().sort_values(["date", "contractid"]).reset_index(drop=True)

        # Ranker predictions
        ranker_X = self.ranker_artifact["preprocessor"].transform(scored[self.ranker_artifact["feature_columns"]])
        scored["ranker_score"] = self.ranker_artifact["model"].predict(ranker_X)
        scored["ranker_rank"] = scored.groupby("date")["ranker_score"].rank(ascending=False, method="first")
        counts = scored.groupby("date")["ranker_score"].transform("count")
        scored["ranker_percentile"] = 1.0 - (scored["ranker_rank"] - 1.0) / counts.clip(lower=1)

        # Meta-labeler predictions
        meta_X = self.meta_artifact["preprocessor"].transform(scored[self.meta_artifact["feature_columns"]])
        raw_prob = self.meta_artifact["model"].predict_proba(meta_X)[:, 1]
        calibrator = self.meta_artifact.get("calibrator")
        if calibrator is None:
            scored["prob_profit"] = np.clip(raw_prob, 1e-6, 1 - 1e-6)
        else:
            scored["prob_profit"] = np.clip(calibrator.predict(raw_prob), 1e-6, 1 - 1e-6)
        scored["prob_profit_raw"] = raw_prob

        # Return distribution predictions
        return_X = self.return_artifact["preprocessor"].transform(scored[self.return_artifact["feature_columns"]])
        low = self.return_artifact["models"]["q10"].predict(return_X)
        mean = self.return_artifact["models"]["mean"].predict(return_X)
        high = self.return_artifact["models"]["q90"].predict(return_X)
        low, mean, high = np.sort(np.vstack([low, mean, high]), axis=0)
        scored["pred_signed_log_q10"] = low
        scored["pred_signed_log_mean"] = mean
        scored["pred_signed_log_q90"] = high
        scored["pred_return_q10"] = np.sign(low) * (np.expm1(np.abs(low)))
        scored["expected_return"] = np.sign(mean) * (np.expm1(np.abs(mean)))
        scored["pred_return_q90"] = np.sign(high) * (np.expm1(np.abs(high)))
        scored["uncertainty"] = scored["pred_return_q90"] - scored["pred_return_q10"]

        return self.sizer.size_candidates(scored)

    def build_trade_plan(self, frame: pd.DataFrame, as_of_date: Optional[str] = None) -> List[Dict[str, object]]:
        scored = self.score_frame(frame)
        target_date = pd.Timestamp(as_of_date) if as_of_date else pd.Timestamp(scored["date"].max())
        day_slice = scored[scored["date"] == target_date].copy()
        if len(day_slice) == 0:
            return []
        allocated = self.sizer.allocate_for_date(
            day_slice,
            current_equity=self.hybrid_config.starting_capital,
            available_cash=self.hybrid_config.starting_capital,
            existing_exposures={"delta": 0.0, "gamma": 0.0, "vega": 0.0},
            open_contracts=[],
        )
        if len(allocated) == 0:
            return []
        cols = [
            "date",
            "contractid",
            "type",
            "strike",
            "expiration",
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
        present = [col for col in cols if col in allocated.columns]
        plan = allocated.sort_values("assigned_weight", ascending=False)[present].to_dict(orient="records")
        return plan


def _position_mark(position: Position, quotes: pd.DataFrame) -> float:
    if len(quotes) == 0:
        return position.last_mark
    row = quotes.iloc[-1]
    if pd.notna(row.get("bid")) and float(row.get("bid")) > 0.0:
        return float(row.get("bid"))
    if pd.notna(row.get("mid_price")):
        return float(row.get("mid_price"))
    return position.last_mark


def evaluate_hybrid_kelly(
    predictions: pd.DataFrame,
    config: Dict[str, object],
) -> Dict[str, object]:
    cfg = HybridKellyConfig.from_config(config)
    quotes = predictions.sort_values(["date", "contractid"]).copy()
    grouped_quotes = {key: group.copy() for key, group in quotes.groupby(["date", "contractid"], sort=False)}
    dates = sorted(pd.to_datetime(quotes["date"].dropna().unique()))

    cash = cfg.starting_capital
    open_positions: Dict[str, Position] = {}
    closed_trades: List[ClosedTrade] = []
    daily_rows: List[Dict[str, object]] = []
    sizer = HybridKellySizer(cfg)

    for current_date in dates:
        # Mark positions to current date.
        for contractid, position in list(open_positions.items()):
            key = (current_date, contractid)
            quote = grouped_quotes.get(key)
            if quote is not None:
                position.last_mark = _position_mark(position, quote)

        # Process exits.
        for contractid, position in list(open_positions.items()):
            if current_date < position.exit_date:
                continue
            key = (current_date, contractid)
            quote = grouped_quotes.get(key)
            exit_price = _position_mark(position, quote if quote is not None else pd.DataFrame())
            cash += position.quantity * exit_price
            pnl = position.quantity * (exit_price - position.entry_price)
            closed_trades.append(
                ClosedTrade(
                    contractid=contractid,
                    entry_date=position.entry_date,
                    exit_date=current_date,
                    entry_price=position.entry_price,
                    exit_price=exit_price,
                    quantity=position.quantity,
                    pnl=pnl,
                    return_pct=(exit_price - position.entry_price) / max(position.entry_price, 1e-9),
                    assigned_weight=position.assigned_weight,
                )
            )
            del open_positions[contractid]

        marked_value = sum(position.quantity * position.last_mark for position in open_positions.values())
        equity_before_new = cash + marked_value
        exposures = {
            "delta": sum(position.assigned_weight * position.delta for position in open_positions.values()),
            "gamma": sum(position.assigned_weight * position.gamma for position in open_positions.values()),
            "vega": sum(position.assigned_weight * position.vega for position in open_positions.values()),
        }

        today = quotes[quotes["date"] == current_date].copy()
        allocated = sizer.allocate_for_date(
            today,
            current_equity=equity_before_new,
            available_cash=cash,
            existing_exposures=exposures,
            open_contracts=list(open_positions.keys()),
        )

        for _, row in allocated.iterrows():
            entry_price = float(row.get("entry_price", row.get("ask", row.get("mid_price", np.nan))))
            if not np.isfinite(entry_price) or entry_price <= 0.0:
                continue
            budget = float(row["assigned_weight"]) * equity_before_new
            budget = min(budget, cash)
            if budget <= 0.0:
                continue
            quantity = budget / entry_price
            if quantity <= 0.0:
                continue
            cash -= quantity * entry_price
            mark = float(row.get("bid", row.get("mid_price", entry_price)))
            if not np.isfinite(mark) or mark <= 0.0:
                mark = entry_price
            open_positions[row["contractid"]] = Position(
                contractid=str(row["contractid"]),
                expiration=pd.Timestamp(row["expiration"]) if "expiration" in row and pd.notna(row["expiration"]) else None,
                quantity=quantity,
                entry_date=pd.Timestamp(current_date),
                exit_date=pd.Timestamp(row["exit_date"]),
                entry_price=entry_price,
                last_mark=mark,
                assigned_weight=float(row["assigned_weight"]),
                delta=float(row.get("delta", 0.0) or 0.0),
                gamma=float(row.get("gamma", 0.0) or 0.0),
                vega=float(row.get("vega", 0.0) or 0.0),
            )

        marked_value = sum(position.quantity * position.last_mark for position in open_positions.values())
        equity = cash + marked_value
        daily_rows.append(
            {
                "date": current_date,
                "cash": cash,
                "marked_value": marked_value,
                "equity": equity,
                "open_positions": len(open_positions),
                "closed_trades": len(closed_trades),
                "delta_exposure": sum(position.assigned_weight * position.delta for position in open_positions.values()),
                "gamma_exposure": sum(position.assigned_weight * position.gamma for position in open_positions.values()),
                "vega_exposure": sum(position.assigned_weight * position.vega for position in open_positions.values()),
            }
        )

    # Liquidate any remaining positions at the final observed mark.
    final_date = dates[-1] if dates else None
    if final_date is not None:
        for contractid, position in list(open_positions.items()):
            cash += position.quantity * position.last_mark
            pnl = position.quantity * (position.last_mark - position.entry_price)
            closed_trades.append(
                ClosedTrade(
                    contractid=contractid,
                    entry_date=position.entry_date,
                    exit_date=final_date,
                    entry_price=position.entry_price,
                    exit_price=position.last_mark,
                    quantity=position.quantity,
                    pnl=pnl,
                    return_pct=(position.last_mark - position.entry_price) / max(position.entry_price, 1e-9),
                    assigned_weight=position.assigned_weight,
                )
            )
            del open_positions[contractid]
        if daily_rows:
            daily_rows[-1]["cash"] = cash
            daily_rows[-1]["marked_value"] = 0.0
            daily_rows[-1]["equity"] = cash
            daily_rows[-1]["open_positions"] = 0

    equity_curve = pd.DataFrame(daily_rows)
    trade_df = pd.DataFrame([asdict(trade) for trade in closed_trades])
    if len(equity_curve) > 1:
        equity_curve["daily_return"] = equity_curve["equity"].pct_change().fillna(0.0)
        running_peak = equity_curve["equity"].cummax()
        drawdown = equity_curve["equity"] / running_peak - 1.0
        sharpe_like = np.sqrt(252.0) * equity_curve["daily_return"].mean() / max(equity_curve["daily_return"].std(ddof=1), 1e-9)
        max_drawdown = float(drawdown.min())
    else:
        sharpe_like = float("nan")
        max_drawdown = float("nan")

    results = {
        "starting_capital": cfg.starting_capital,
        "ending_capital": float(equity_curve["equity"].iloc[-1]) if len(equity_curve) else cfg.starting_capital,
        "total_return": float((equity_curve["equity"].iloc[-1] / cfg.starting_capital) - 1.0) if len(equity_curve) else 0.0,
        "max_drawdown": max_drawdown,
        "sharpe_like": float(sharpe_like),
        "trade_count": int(len(trade_df)),
        "win_rate": float((trade_df["pnl"] > 0).mean()) if len(trade_df) else float("nan"),
        "average_trade_return": float(trade_df["return_pct"].mean()) if len(trade_df) else float("nan"),
        "median_trade_return": float(trade_df["return_pct"].median()) if len(trade_df) else float("nan"),
        "equity_curve": equity_curve,
        "trade_log": trade_df,
    }
    return results


def run_hybrid_pipeline(
    data_files: Sequence[str],
    ranker_artifact_path: str,
    meta_artifact_path: str,
    return_artifact_path: str,
    config_file: Optional[str] = None,
    output_dir: Optional[str] = None,
    include_targets: bool = True,
    nrows: Optional[int] = None,
) -> Dict[str, object]:
    config = load_config(config_file)
    engine = HybridTradeEngine.from_paths(
        ranker_artifact_path=ranker_artifact_path,
        meta_artifact_path=meta_artifact_path,
        return_artifact_path=return_artifact_path,
        config_file=config_file,
    )
    frame = prepare_model_frame(data_files, config, include_targets=include_targets, nrows=nrows)
    logger.info("Scoring frame %s", summarize_frame(frame))
    predictions = engine.score_frame(frame)

    root = get_output_dir(config, output_dir)
    predictions_path = root / "hybrid_predictions.csv"
    predictions.to_csv(predictions_path, index=False)

    result: Dict[str, object] = {"predictions_path": str(predictions_path)}
    if include_targets:
        backtest = evaluate_hybrid_kelly(predictions, config)
        metrics = {key: value for key, value in backtest.items() if key not in {"equity_curve", "trade_log"}}
        metrics_path = root / "hybrid_backtest_metrics.json"
        equity_path = root / "hybrid_equity_curve.csv"
        trades_path = root / "hybrid_trade_log.csv"
        save_json(metrics, metrics_path)
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
    parser = argparse.ArgumentParser(description="Score options candidates and run a stateful hybrid-Kelly backtest")
    parser.add_argument("--data", nargs="+", required=True, help="CSV file(s) with option market snapshots")
    parser.add_argument("--ranker-artifact", required=True, help="Path to ranker artifact")
    parser.add_argument("--meta-artifact", required=True, help="Path to meta-labeler artifact")
    parser.add_argument("--return-artifact", required=True, help="Path to return distribution artifact")
    parser.add_argument("--config", default="./config.yaml", help="Path to YAML config")
    parser.add_argument("--output-dir", default=None, help="Directory for scored outputs")
    parser.add_argument("--predict-only", action="store_true", help="Skip backtest metrics and only emit predictions")
    parser.add_argument("--nrows", type=int, default=None, help="Optional row cap for quick smoke runs")
    args = parser.parse_args()

    result = run_hybrid_pipeline(
        data_files=args.data,
        ranker_artifact_path=args.ranker_artifact,
        meta_artifact_path=args.meta_artifact,
        return_artifact_path=args.return_artifact,
        config_file=args.config,
        output_dir=args.output_dir,
        include_targets=not args.predict_only,
        nrows=args.nrows,
    )
    logger.info("Hybrid pipeline complete: %s", result)


if __name__ == "__main__":
    main()
