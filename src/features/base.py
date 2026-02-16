"""Base feature transform interface for LOB snapshots and sequences."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

import torch


class BaseLOBTransform(ABC):
    """Transform LOB states into fixed-length tensors.

    Subclasses should implement `transform_snapshot` for a single LOB state.
    The default `transform_sequence` stacks snapshot features over time.
    """

    @abstractmethod
    def transform_snapshot(self, lob_state: Any) -> torch.Tensor:
        """Convert one LOB state into a 1D feature tensor."""

    def transform_sequence(self, lob_states: Sequence[Any]) -> torch.Tensor:
        """Convert a sequence of LOB states into a (T, D) tensor."""
        features = [self.transform_snapshot(state) for state in lob_states]
        if not features:
            return torch.empty((0, 0))
        return torch.stack(features, dim=0)
