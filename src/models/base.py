"""Shared abstractions and utilities for DeepHit competing-risk backbones."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseDeepHitCompetingModel(nn.Module, ABC):
    """Abstract interface for DeepHit-compatible competing-risk models.

    Subclasses implement only the encoder via ``encode()``. All shared layers
    (input projection, attention pooling, cause-specific heads, auxiliary head)
    and the full forward pass live here.
    """

    def __init__(
        self,
        num_features: int,
        num_events: int,
        num_time_steps: int,
        hidden_size: int,
        fc_hidden: int,
        fc_dropout: float,
    ) -> None:
        super().__init__()
        if num_events <= 0:
            raise ValueError("num_events must be positive.")
        if num_time_steps <= 0:
            raise ValueError("num_time_steps must be positive.")
        self.num_events = num_events
        self.num_time_steps = num_time_steps

        self.input_projection = nn.Linear(num_features, hidden_size)
        self.input_proj_residual = nn.Linear(num_features, hidden_size)
        self.attn_query_proj = nn.Linear(hidden_size, hidden_size)
        self.attn_key_proj = nn.Linear(hidden_size, hidden_size)
        self.attn_scale = 1.0 / (hidden_size ** 0.5)

        self.cause_specific_heads = self._build_cause_specific_heads(
            input_size=hidden_size,
            hidden_size=fc_hidden,
            dropout=fc_dropout,
            activation="gelu",
            use_batch_norm=True,
        )

        self.pred_dim = num_features - 1
        self.aux_head = nn.Linear(hidden_size, self.pred_dim)
        self._cache: dict[str, torch.Tensor] = {}

    @abstractmethod
    def encode(
        self, x_proj: torch.Tensor, mask: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """Encode projected input into sequence representations.

        Args:
            x_proj: Shape ``(batch, seq_len, hidden_size)`` — output of ``input_projection``.
            mask: Valid-timestep mask of shape ``(batch, seq_len)``.
            lengths: Valid sequence lengths of shape ``(batch,)``.

        Returns:
            Shape ``(batch, seq_len, hidden_size)``.
        """

    def _lengths_from_mask(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = x[:, :, -1]
        lengths = mask.sum(dim=1).long().clamp(min=1, max=x.size(1))
        return mask, lengths

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits of shape ``(batch_size, num_events, num_time_steps)``."""
        mask, lengths = self._lengths_from_mask(x)

        x_proj = self.input_projection(x)
        x_res = self.input_proj_residual(x[:, -1, :])

        state_out = self.encode(x_proj, mask, lengths)

        self._cache = {"state_out": state_out, "mask": mask, "lengths": lengths}

        query = self.attn_query_proj(x_res)
        seq_repr = masked_attention_pooling(
            sequence=state_out,
            query=query,
            key_projection=self.attn_key_proj,
            scale=self.attn_scale,
            mask=mask,
        )

        combined = seq_repr + x_res
        return torch.stack([head(combined) for head in self.cause_specific_heads], dim=1)

    def aux_next_step_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Auxiliary MSE loss predicting next-step features."""
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
        return ((pred_next - target_next) ** 2 * pair_valid_f).sum() / denom

    def _build_cause_specific_heads(
        self,
        input_size: int,
        hidden_size: int,
        dropout: float,
        activation: str = "relu",
        use_batch_norm: bool = True,
    ) -> nn.ModuleList:
        hidden_size = int(hidden_size)
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
    fill_value = torch.finfo(scores.dtype).min
    scores = scores.masked_fill(~(mask > 0.5), fill_value)
    weights = torch.softmax(scores, dim=1)  # (batch, seq_len)
    return torch.sum(sequence * weights.unsqueeze(-1), dim=1)
