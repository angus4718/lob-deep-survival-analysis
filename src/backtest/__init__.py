"""Backtesting utilities for labeled LOB survival datasets."""

from .data import BacktestDataset, BacktestFeatureBuilder
from .engine import BacktestEngine
from .latency import MeasuredLatencyProvider, StaticLatencyProvider
from .metrics import ImplementationShortfallMetric, select_toxic_cost_window
from .reports import BacktestReport
from .types import BacktestResult, MarketSnapshot, TradingDecision

try:
    from .raw_engine import RawDatabentoBacktestEngine
except ModuleNotFoundError as exc:  # pragma: no cover - optional raw replay deps

    class RawDatabentoBacktestEngine:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "RawDatabentoBacktestEngine requires optional raw replay dependencies."
            ) from exc

__all__ = [
    "BacktestDataset",
    "BacktestEngine",
    "BacktestFeatureBuilder",
    "BacktestReport",
    "BacktestResult",
    "ImplementationShortfallMetric",
    "MarketSnapshot",
    "MeasuredLatencyProvider",
    "RawDatabentoBacktestEngine",
    "StaticLatencyProvider",
    "TradingDecision",
    "select_toxic_cost_window",
]
