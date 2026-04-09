"""GRU backbone for DeepHit competing risks."""

from __future__ import annotations

import torch
from torch import nn

from .base import BaseDeepHitCompetingModel, masked_attention_pooling


class DeepHitRNNCompeting(BaseDeepHitCompetingModel):
    """GRU model with attention pooling and residual latest-step skip connection."""

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
        super().__init__(num_events=num_events, num_time_steps=num_time_steps)
        self.hidden_size = hidden_size

        self.rnn = nn.GRU(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=rnn_dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )

        self.attn_query_proj = nn.Linear(hidden_size, hidden_size)
        self.attn_key_proj = nn.Linear(hidden_size, hidden_size)
        self.attn_scale = 1.0 / (hidden_size ** 0.5)

        self.input_proj_residual = nn.Linear(num_features, hidden_size)
        self.cause_specific_heads = self._build_cause_specific_heads(
            input_size=hidden_size,
            hidden_size=fc_hidden,
            dropout=fc_dropout,
            activation="relu",
            use_batch_norm=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Mask is expected as the final channel: valid timestep if value > 0.5.
        mask = x[:, :, -1]  # (batch, seq_len)
        rnn_out, _ = self.rnn(x)  # (batch, seq_len, hidden)

        last_input = x[:, -1, :]  # (batch, num_features)
        input_proj = self.input_proj_residual(last_input)  # (batch, hidden)
        query = self.attn_query_proj(input_proj)

        seq_repr = masked_attention_pooling(
            sequence=rnn_out,
            query=query,
            key_projection=self.attn_key_proj,
            scale=self.attn_scale,
            mask=mask,
        )

        combined_repr = seq_repr + input_proj
        logits = torch.stack([head(combined_repr) for head in self.cause_specific_heads], dim=1)
        return logits
