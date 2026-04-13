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
        self.output_norm = nn.LayerNorm(hidden_size)

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

        self.pred_dim = num_features - 1
        self.aux_head = nn.Linear(hidden_size, self.pred_dim)
        self._cache: dict[str, torch.Tensor] = {}

    def _lengths_from_mask(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = x[:, :, -1]
        lengths = mask.sum(dim=1).long().clamp(min=1, max=x.size(1))
        return mask, lengths

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Mask is expected as the final channel: valid timestep if value > 0.5.
        mask, lengths = self._lengths_from_mask(x)
        rnn_out, _ = self.rnn(x)  # (batch, seq_len, hidden)
        rnn_out = self.output_norm(rnn_out)

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

        self._cache = {
            "state_out": rnn_out,
            "mask": mask,
            "lengths": lengths,
        }

        combined_repr = seq_repr + input_proj
        logits = torch.stack([head(combined_repr) for head in self.cause_specific_heads], dim=1)
        return logits

    def aux_next_step_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Auxiliary MSE loss to predict next-step features."""
        if not self._cache:
            _ = self.forward(x)

        state_out = self._cache["state_out"]
        mask = self._cache["mask"]

        if x.size(1) <= 1:
            return torch.tensor(0.0, dtype=torch.float32, device=x.device)

        pred_next = self.aux_head(state_out[:, :-1, :])
        target_next = x[:, 1:, : self.pred_dim]

        pair_valid = (mask[:, 1:] > 0.5) & (mask[:, :-1] > 0.5)
        pair_valid_f = pair_valid.float().unsqueeze(-1)

        denom = pair_valid_f.sum() * self.pred_dim + 1e-8
        mse = ((pred_next - target_next) ** 2 * pair_valid_f).sum() / denom
        return mse
