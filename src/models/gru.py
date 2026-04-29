"""GRU backbone for DeepHit competing risks."""

from __future__ import annotations

import torch
from torch import nn

from .base import BaseDeepHitCompetingModel


class DeepHitRNNCompeting(BaseDeepHitCompetingModel):
    """GRU encoder with attention pooling and residual latest-step skip connection."""

    def __init__(
        self,
        num_features: int,
        num_events: int,
        num_time_steps: int,
        hidden_size: int = 160,
        num_layers: int = 2,
        rnn_dropout: float = 0.2,
        fc_hidden: int = 256,
        fc_dropout: float = 0.2,
    ) -> None:
        super().__init__(num_features, num_events, num_time_steps, hidden_size, fc_hidden, fc_dropout)
        self.pre_encoder_norm = nn.LayerNorm(hidden_size)
        self.rnn = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=rnn_dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )

    def encode(
        self, x_proj: torch.Tensor, _mask: torch.Tensor, _lengths: torch.Tensor
    ) -> torch.Tensor:
        h = self.pre_encoder_norm(x_proj)
        rnn_out, _ = self.rnn(h)
        return rnn_out
