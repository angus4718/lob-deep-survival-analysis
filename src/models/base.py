"""Shared abstractions and utilities for DeepHit competing-risk backbones."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseDeepHitCompetingModel(nn.Module, ABC):
    """Abstract interface for DeepHit-compatible competing-risk models.

    Implementations must return logits with shape:
    ``(batch_size, num_events, num_time_steps)``.
    """

    def __init__(self, num_events: int, num_time_steps: int) -> None:
        super().__init__()
        if num_events <= 0:
            raise ValueError("num_events must be positive.")
        if num_time_steps <= 0:
            raise ValueError("num_time_steps must be positive.")
        self.num_events = num_events
        self.num_time_steps = num_time_steps

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits of shape ``(batch_size, num_events, num_time_steps)``."""

    def _build_cause_specific_heads(
        self,
        input_size: int,
        hidden_size: int,
        dropout: float,
        activation: str = "relu",
        use_batch_norm: bool = True,
    ) -> nn.ModuleList:
        layers: list[nn.Sequential] = []
        for _ in range(self.num_events):
            modules: list[nn.Module] = [nn.Linear(input_size, hidden_size)]
            if activation == "gelu":
                modules.append(nn.GELU())
            else:
                modules.append(nn.ReLU(inplace=True))
            if use_batch_norm:
                modules.append(nn.BatchNorm1d(hidden_size))
            modules.append(nn.Dropout(dropout))
            modules.append(nn.Linear(hidden_size, self.num_time_steps))
            layers.append(nn.Sequential(*modules))
        return nn.ModuleList(layers)


def masked_attention_pooling(
    sequence: torch.Tensor,
    query: torch.Tensor,
    key_projection: nn.Linear,
    scale: float,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute masked attention pooling over sequence states.

    Args:
        sequence: Tensor of shape ``(batch, seq_len, hidden_size)``.
        query: Tensor of shape ``(batch, hidden_size)``.
        key_projection: Linear layer used to project sequence into keys.
        scale: Attention scale, typically ``1 / sqrt(hidden_size)``.
        mask: Tensor of shape ``(batch, seq_len)`` where values ``> 0.5`` are valid.
    """

    keys = key_projection(sequence)  # (batch, seq_len, hidden)
    scores = torch.bmm(query.unsqueeze(1), keys.transpose(1, 2)) * scale
    scores = scores.squeeze(1)  # (batch, seq_len)
    scores = scores.masked_fill(~(mask > 0.5), -1e9)
    weights = torch.softmax(scores, dim=1)  # (batch, seq_len)
    return torch.sum(sequence * weights.unsqueeze(-1), dim=1)
