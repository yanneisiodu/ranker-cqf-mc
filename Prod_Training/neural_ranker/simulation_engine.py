"""Event-driven simulation engine with causal settlement.

Core rules:
  1. Positions are opened on entry_date, capital is reserved immediately
  2. Every day, open positions are marked-to-market against current prices
  3. Positions exit on: take-profit, stop-loss, trailing stop, or max hold expiry
  4. P&L is realized ONLY when the position exits
  5. Capital cannot be reinvested until the position is settled
  6. Efficacy/history features only use matured (settled) outcomes

This module is used by all backtests to ensure causal correctness.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class ExitStrategy:
    """Configurable exit rules."""
    take_profit_pct: float = 0.50       # exit if up X% from entry
    stop_loss_pct: float = 0.20         # exit if down X% from entry
    trailing_stop_pct: float = 0.15     # exit if drops X% from peak
    max_hold_days: int = 5              # force exit after N trading days

    @classmethod
    def from_config(cls, config: Dict) -> "ExitStrategy":
        cfg = config.get("exit_strategy", {})
        return cls(**{k: cfg[k] for k in cfg if k in cls.__dataclass_fields__})


@dataclass
class ExecutionConfig:
    """Tradability filters — applied on RAW (pre-normalization) values."""
    min_price: float = 0.10
    max_relative_spread: float = 0.30
    min_volume: int = 10
    min_open_interest: int = 50
    max_volume_participation: float = 0.10

    @classmethod
    def from_config(cls, config: Dict) -> "ExecutionConfig":
        cfg = config.get("execution", {})
        return cls(**{k: cfg[k] for k in cfg if k in cls.__dataclass_fields__})


@dataclass
class RiskConfig:
    """Position sizing and risk limits."""
    starting_capital: float = 10000.0
    max_position_pct: float = 0.05
    max_gross_pct: float = 0.50
    max_positions: int = 10
    max_same_direction: int = 7
    max_same_expiry: int = 3
    drawdown_reduce_threshold: float = 0.20
    drawdown_reduce_factor: float = 0.50

    @classmethod
    def from_config(cls, config: Dict) -> "RiskConfig":
        cfg = config.get("risk", {})
        return cls(**{k: cfg[k] for k in cfg if k in cls.__dataclass_fields__})


@dataclass
class OpenPosition:
    """A position that is currently held."""
    entry_date: pd.Timestamp
    contractid: str
    option_type: str
    strike: float
    days_to_exp: float
    entry_price: float
    n_contracts: int
    cost: float                        # capital reserved
    score: float
    rank: int
    peak_price: float = 0.0           # for trailing stop
    hold_days: int = 0
    exit_reason: str = ""
    exit_override: Optional["ExitStrategy"] = None  # per-position bucketed exit
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.peak_price = self.entry_price


@dataclass
class SettledTrade:
    """A position that has been closed."""
    entry_date: str
    exit_date: str
    contractid: str
    option_type: str
    strike: float
    days_to_exp: float
    entry_price: float
    exit_price: float
    n_contracts: int
    cost: float
    exit_value: float
    pnl: float
    return_pct: float
    hold_days: int
    exit_reason: str
    score: float
    rank: int
    meta: Dict[str, Any] = field(default_factory=dict)


class MaturedHistoryQueue:
    """Tracks outcomes that have matured (position settled).

    Only settled positions contribute to efficacy features.
    This prevents future-leakage in rolling hit rates.
    """

    def __init__(self):
        self.settled: List[SettledTrade] = []

    def add(self, trade: SettledTrade):
        self.settled.append(trade)

    def recent_trades(self, n: int = 25) -> List[SettledTrade]:
        return self.settled[-n:]

    def call_hit_rate(self, n: int = 25) -> float:
        recent = [t for t in self.recent_trades(n) if t.option_type == "call"]
        if len(recent) < 3:
            return 0.5
        return sum(1 for t in recent if t.pnl > 0) / len(recent)

    def put_hit_rate(self, n: int = 25) -> float:
        recent = [t for t in self.recent_trades(n) if t.option_type == "put"]
        if len(recent) < 3:
            return 0.5
        return sum(1 for t in recent if t.pnl > 0) / len(recent)

    def basket_hit_rate(self, n: int = 5) -> float:
        """Hit rate of recent settled baskets (by entry date)."""
        if len(self.settled) < 3:
            return 0.5
        recent = self.settled[-n * 10:]  # grab enough to cover n entry dates
        by_date = {}
        for t in recent:
            by_date.setdefault(t.entry_date, []).append(t.pnl)
        dates = sorted(by_date.keys())[-n:]
        if not dates:
            return 0.5
        wins = sum(1 for d in dates if sum(by_date[d]) > 0)
        return wins / len(dates)

    def consecutive_losses(self) -> int:
        """Count consecutive losing entry-date baskets from most recent."""
        by_date = {}
        for t in self.settled:
            by_date.setdefault(t.entry_date, []).append(t.pnl)
        dates = sorted(by_date.keys())
        count = 0
        for d in reversed(dates):
            if sum(by_date[d]) < 0:
                count += 1
            else:
                break
        return count

    def current_drawdown(self) -> float:
        """Current drawdown from peak equity based on settled trades."""
        if not self.settled:
            return 0.0
        cumulative_pnl = np.cumsum([t.pnl for t in self.settled])
        peak = np.maximum.accumulate(cumulative_pnl)
        if peak[-1] <= 0:
            return 0.0
        return float((cumulative_pnl[-1] - peak[-1]) / peak[-1])


class SimulationEngine:
    """Event-driven simulation with causal settlement and exit strategy."""

    def __init__(
        self,
        exit_strategy: ExitStrategy,
        risk_config: RiskConfig,
    ):
        self.exit_strategy = exit_strategy
        self.risk_config = risk_config

        self.cash = risk_config.starting_capital
        self.open_positions: List[OpenPosition] = []
        self.history = MaturedHistoryQueue()
        self.equity_curve: List[Dict] = []
        self.peak_equity = risk_config.starting_capital

    @property
    def reserved_capital(self) -> float:
        return sum(p.cost for p in self.open_positions)

    @property
    def available_capital(self) -> float:
        return self.cash

    @property
    def equity(self) -> float:
        return self.cash + self.reserved_capital

    def step(self, current_date: pd.Timestamp, price_lookup: Dict[str, float]) -> List[SettledTrade]:
        """Process one trading day.

        1. Mark-to-market all open positions
        2. Check exit conditions and settle positions that trigger
        3. Returns list of settled trades for this day

        Args:
            current_date: today's date
            price_lookup: contractid -> current bid price (for exit valuation)
        """
        settled_today = []
        remaining = []

        for pos in self.open_positions:
            pos.hold_days += 1
            current_price = price_lookup.get(pos.contractid, 0)

            # Update peak for trailing stop
            if current_price > pos.peak_price:
                pos.peak_price = current_price

            # Check exit conditions
            exit_reason = self._check_exit(pos, current_price)

            if exit_reason:
                # Settle the position
                exit_value = pos.n_contracts * current_price * 100
                pnl = exit_value - pos.cost

                trade = SettledTrade(
                    entry_date=str(pos.entry_date),
                    exit_date=str(current_date),
                    contractid=pos.contractid,
                    option_type=pos.option_type,
                    strike=pos.strike,
                    days_to_exp=pos.days_to_exp,
                    entry_price=pos.entry_price,
                    exit_price=current_price,
                    n_contracts=pos.n_contracts,
                    cost=pos.cost,
                    exit_value=exit_value,
                    pnl=pnl,
                    return_pct=pnl / pos.cost if pos.cost > 0 else 0,
                    hold_days=pos.hold_days,
                    exit_reason=exit_reason,
                    score=pos.score,
                    rank=pos.rank,
                    meta=pos.meta,
                )

                # Return capital
                self.cash += exit_value
                settled_today.append(trade)
                self.history.add(trade)
            else:
                remaining.append(pos)

        self.open_positions = remaining

        # Update peak equity
        self.peak_equity = max(self.peak_equity, self.equity)

        return settled_today

    def _check_exit(self, pos: OpenPosition, current_price: float) -> str:
        """Check if a position should exit. Returns reason or empty string."""
        if current_price <= 0:
            return "worthless"

        # Use per-position exit if set, otherwise global
        es = pos.exit_override if pos.exit_override is not None else self.exit_strategy

        ret = (current_price - pos.entry_price) / pos.entry_price

        # Take profit
        if ret >= es.take_profit_pct:
            return "take_profit"

        # Stop loss
        if ret <= -es.stop_loss_pct:
            return "stop_loss"

        # Trailing stop
        if pos.peak_price > pos.entry_price:
            drop_from_peak = (pos.peak_price - current_price) / pos.peak_price
            if drop_from_peak >= es.trailing_stop_pct:
                return "trailing_stop"

        # Max hold
        if pos.hold_days >= es.max_hold_days:
            return "max_hold"

        return ""

    def open_position(self, pos: OpenPosition) -> bool:
        """Open a new position if capital allows.

        Returns True if position was opened, False if rejected.
        """
        if pos.cost > self.available_capital:
            return False
        if pos.cost <= 0:
            return False

        # Reserve capital
        self.cash -= pos.cost
        self.open_positions.append(pos)
        return True

    def get_capacity(self) -> Dict[str, Any]:
        """Return current capacity for new positions."""
        dd = 1 - self.equity / self.peak_equity if self.peak_equity > 0 else 0
        max_gross = self.equity * self.risk_config.max_gross_pct

        # Drawdown reduction
        if dd > self.risk_config.drawdown_reduce_threshold:
            max_gross *= self.risk_config.drawdown_reduce_factor

        return {
            "available_cash": self.available_capital,
            "max_gross": max_gross,
            "current_reserved": self.reserved_capital,
            "remaining_gross": max(0, max_gross - self.reserved_capital),
            "max_per_position": self.equity * self.risk_config.max_position_pct,
            "n_open": len(self.open_positions),
            "max_new_positions": max(0, self.risk_config.max_positions - len(self.open_positions)),
            "current_drawdown": dd,
            "call_count": sum(1 for p in self.open_positions if p.option_type == "call"),
            "put_count": sum(1 for p in self.open_positions if p.option_type == "put"),
        }

    def record_equity(self, date: pd.Timestamp, n_new_trades: int, settled: List[SettledTrade]):
        """Record equity curve entry for this day."""
        settled_pnl = sum(t.pnl for t in settled)
        self.equity_curve.append({
            "date": date,
            "cash": self.cash,
            "reserved": self.reserved_capital,
            "equity": self.equity,
            "n_open": len(self.open_positions),
            "n_settled": len(settled),
            "settled_pnl": settled_pnl,
            "n_new": n_new_trades,
            "peak_equity": self.peak_equity,
            "drawdown": 1 - self.equity / self.peak_equity if self.peak_equity > 0 else 0,
        })

    def get_results(self) -> Dict[str, Any]:
        """Compute final simulation metrics."""
        eq = pd.DataFrame(self.equity_curve) if self.equity_curve else pd.DataFrame()
        trades = [t.__dict__ for t in self.history.settled]
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

        m = {
            "starting_capital": self.risk_config.starting_capital,
            "ending_equity": self.equity,
            "ending_cash": self.cash,
            "total_return_pct": (self.equity / self.risk_config.starting_capital - 1) * 100,
            "total_trades": len(self.history.settled),
            "open_positions_remaining": len(self.open_positions),
        }

        if len(trades_df) > 0:
            m["win_rate"] = (trades_df["pnl"] > 0).mean() * 100
            m["avg_trade_return"] = trades_df["return_pct"].mean() * 100
            m["median_trade_return"] = trades_df["return_pct"].median() * 100
            m["best_trade"] = trades_df["return_pct"].max() * 100
            m["worst_trade"] = trades_df["return_pct"].min() * 100
            m["avg_hold_days"] = trades_df["hold_days"].mean()

            # Exit reason breakdown
            for reason in ["take_profit", "stop_loss", "trailing_stop", "max_hold", "worthless"]:
                count = (trades_df["exit_reason"] == reason).sum()
                if count > 0:
                    m[f"exit_{reason}_count"] = int(count)
                    m[f"exit_{reason}_pct"] = count / len(trades_df) * 100

            # Side breakdown
            for side in ["call", "put"]:
                st = trades_df[trades_df["option_type"] == side]
                if len(st) > 0:
                    m[f"{side}_trades"] = len(st)
                    m[f"{side}_win_rate"] = (st["pnl"] > 0).mean() * 100
                    m[f"{side}_avg_return"] = st["return_pct"].mean() * 100

        if len(eq) > 1:
            m["max_drawdown_pct"] = eq["drawdown"].max() * 100
            daily_equity = eq["equity"].values
            daily_returns = np.diff(daily_equity) / daily_equity[:-1]
            m["sharpe"] = float(daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0

        return {"metrics": m, "equity_curve": eq, "trades": trades_df}


def filter_tradeable_raw(
    day: pd.DataFrame,
    exec_config: ExecutionConfig,
) -> pd.DataFrame:
    """Filter to tradeable options using RAW (pre-normalization) columns only."""
    mask = pd.Series(True, index=day.index)

    if "ask" in day.columns:
        mask &= day["ask"] >= exec_config.min_price
    if "relative_spread" in day.columns:
        mask &= day["relative_spread"] <= exec_config.max_relative_spread
    if "volume" in day.columns:
        mask &= day["volume"] >= exec_config.min_volume
    if "open_interest" in day.columns:
        mask &= day["open_interest"] >= exec_config.min_open_interest

    return day[mask].copy()
