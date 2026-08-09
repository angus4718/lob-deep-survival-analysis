"""Strategy implementations for the backtest engine."""

from .base import BaseStrategy, DecisionLogic
from .baseline import AlwaysPlaceLimitOrderStrategy

try:
    from .deephit import (
        DeepHitPredictionCache,
        DeepHitStrategy,
        DeepHitToxicCIFDecisionLogic,
        DeepHitThresholdDecisionLogic,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - optional torch path
    if exc.name != "torch":
        raise
    _TORCH_IMPORT_ERROR = exc

    class DeepHitPredictionCache:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("DeepHitPredictionCache requires torch to be installed.") from _TORCH_IMPORT_ERROR

    class DeepHitStrategy:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("DeepHitStrategy requires torch to be installed.") from _TORCH_IMPORT_ERROR

    class DeepHitThresholdDecisionLogic:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("DeepHitThresholdDecisionLogic requires torch to be installed.") from _TORCH_IMPORT_ERROR

    class DeepHitToxicCIFDecisionLogic:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("DeepHitToxicCIFDecisionLogic requires torch to be installed.") from _TORCH_IMPORT_ERROR

__all__ = [
    "BaseStrategy",
    "DecisionLogic",
    "AlwaysPlaceLimitOrderStrategy",
    "DeepHitPredictionCache",
    "DeepHitStrategy",
    "DeepHitThresholdDecisionLogic",
    "DeepHitToxicCIFDecisionLogic",
]
