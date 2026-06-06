"""Strategy interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.backtest.types import MarketSnapshot, TradingDecision


class DecisionLogic(ABC):
    """Converts model output into a trading decision."""

    @abstractmethod
    def decide(self, prediction: dict[str, Any], snapshot: MarketSnapshot) -> TradingDecision:
        """Return a decision for one market snapshot."""


class BaseStrategy(ABC):
    """Base strategy contract consumed by BacktestEngine."""

    @abstractmethod
    def decide(self, snapshot: MarketSnapshot) -> TradingDecision:
        """Return a trading decision for one candidate order."""
