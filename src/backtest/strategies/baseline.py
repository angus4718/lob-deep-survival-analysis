"""Baseline backtest strategies."""

from __future__ import annotations

import numpy as np

from src.backtest.strategies.base import BaseStrategy
from src.backtest.types import DecisionAction, MarketSnapshot, TradingDecision


class AlwaysPlaceLimitOrderStrategy(BaseStrategy):
    """Place every candidate limit order and leave it untouched."""

    def decide(self, snapshot: MarketSnapshot) -> TradingDecision:
        action = DecisionAction.HOLD if snapshot.position_open else DecisionAction.SUBMIT
        return TradingDecision(
            action=action,
            limit_price=_as_float(snapshot.row.get("price")),
            reason="baseline_hold" if snapshot.position_open else "baseline_submit",
            diagnostics={
                "update_idx": int(snapshot.update_idx),
                "end_idx": int(snapshot.end_idx) if snapshot.end_idx is not None else None,
                "is_initial": bool(snapshot.is_initial),
                "position_open": bool(snapshot.position_open),
            },
        )


def _as_float(value) -> float | None:
    try:
        if value is None or not np.isfinite(float(value)):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
