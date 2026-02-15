"""Dataclasses that define contracts between datasets, models, and backtests.

These types standardize the tensors and metadata passed across the training
pipeline so components can interoperate without tight coupling.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Union

import torch


@dataclass(frozen=True)
class SurvivalSample:
    x_seq: torch.Tensor
    event_type: int
    time_bin: Optional[int]
    meta: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class CompetingRisksOutput:
    pmf: torch.Tensor
    attention_weights: Optional[torch.Tensor] = None
    aux_predictions: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None


@dataclass(frozen=True)
class ExecutionResult:
    exec_price: Optional[float]
    exec_time: Optional[float]
    event_type: int
    meta: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class BacktestReport:
    metrics: Dict[str, float]
    results: Sequence[ExecutionResult]
    meta: Optional[Dict[str, Any]] = None


__all__ = [
    "SurvivalSample",
    "CompetingRisksOutput",
    "ExecutionResult",
    "BacktestReport",
]
