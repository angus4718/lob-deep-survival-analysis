"""Domain package exports.

Keep torch-dependent datatypes lazily imported so lightweight modules
that only need enums (e.g., labeling / dataset scripts) do not trigger
heavy torch imports at package import time.
"""

from .enums import EventType

__all__ = [
    "EventType",
    "SurvivalSample",
    "CompetingRisksOutput",
    "ExecutionResult",
    "BacktestReport",
]


def __getattr__(name: str):
    if name in {
        "SurvivalSample",
        "CompetingRisksOutput",
        "ExecutionResult",
        "BacktestReport",
    }:
        from .datatypes import (
            BacktestReport,
            CompetingRisksOutput,
            ExecutionResult,
            SurvivalSample,
        )

        return {
            "SurvivalSample": SurvivalSample,
            "CompetingRisksOutput": CompetingRisksOutput,
            "ExecutionResult": ExecutionResult,
            "BacktestReport": BacktestReport,
        }[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
