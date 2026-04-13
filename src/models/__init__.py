"""Standardized model backbones for DeepHit competing risks."""

from .base import BaseDeepHitCompetingModel
from .gru import DeepHitRNNCompeting
from .gru_transformer import DeepHitRNNTransformerCompeting
from .transformer import DeepHitTransformerCompeting

try:
    from .mamba import DeepHitMambaCompeting
except Exception as exc:  # pragma: no cover - optional dependency path
    class DeepHitMambaCompeting:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "DeepHitMambaCompeting requires optional dependencies (mamba_ssm / causal-conv1d). "
                "Install them in your environment before using MODEL_NAME='mamba'."
            ) from exc

__all__ = [
    "BaseDeepHitCompetingModel",
    "DeepHitRNNCompeting",
    "DeepHitRNNTransformerCompeting",
    "DeepHitTransformerCompeting",
    "DeepHitMambaCompeting",
]
