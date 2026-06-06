"""Strategy implementations for the backtest engine."""

from .base import BaseStrategy, DecisionLogic
from .baseline import AlwaysPlaceLimitOrderStrategy

try:
    from .deephit import (
        DeepHitPredictionCache,
        DeepHitStrategy,
        DeepHitThresholdDecisionLogic,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - optional torch path
    if exc.name != "torch":
        raise

    class DeepHitPredictionCache:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("DeepHitPredictionCache requires torch to be installed.") from exc

    class DeepHitStrategy:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("DeepHitStrategy requires torch to be installed.") from exc

    class DeepHitThresholdDecisionLogic:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("DeepHitThresholdDecisionLogic requires torch to be installed.") from exc

__all__ = [
    "BaseStrategy",
    "DecisionLogic",
    "AlwaysPlaceLimitOrderStrategy",
    "DeepHitPredictionCache",
    "DeepHitStrategy",
    "DeepHitThresholdDecisionLogic",
]
