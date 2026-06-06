"""Shared backtest data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd


class DecisionAction(str, Enum):
    """Actions a strategy can take for one candidate order."""

    SUBMIT = "submit"
    SKIP = "skip"
    HOLD = "hold"
    CANCEL = "cancel"


@dataclass(frozen=True)
class MarketSnapshot:
    """One labeled candidate order plus model-ready features."""

    row_index: int
    row: pd.Series
    features: np.ndarray
    update_idx: int = 0
    end_idx: int | None = None
    is_initial: bool = True
    position_open: bool = False

    @property
    def order_id(self) -> Any:
        return self.row.get("order_id")

    @property
    def side(self) -> str:
        return str(self.row.get("side", "")).upper()


@dataclass(frozen=True)
class TradingDecision:
    """Strategy output consumed by the backtest engine."""

    action: DecisionAction | str
    limit_price: float | None = None
    size: float = 1.0
    reason: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def should_submit(self) -> bool:
        return DecisionAction(self.action) == DecisionAction.SUBMIT

    @property
    def should_cancel(self) -> bool:
        return DecisionAction(self.action) == DecisionAction.CANCEL


@dataclass(frozen=True)
class BacktestResult:
    """Per-order backtest result after applying metrics."""

    row_index: int
    order_id: Any
    decision: TradingDecision
    metrics: dict[str, Any]
    diagnostics: dict[str, Any] = field(default_factory=dict)
