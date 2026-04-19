"""Mamba backbone for DeepHit competing risks."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .base import BaseDeepHitCompetingModel, masked_attention_pooling

try:
    from mamba_ssm import Mamba as MambaBlock
except ImportError:  # pragma: no cover - optional dependency
    MambaBlock = None


class ResidualMambaBlock(nn.Module):
    """Pre-norm residual wrapper around a Mamba block."""

    def __init__(
        self,
        hidden_size: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if MambaBlock is None:
            raise ImportError(
                "mamba_ssm is required for DeepHitMambaCompeting. "
                "Install it with `pip install mamba-ssm` in your environment."
            )
        self.norm = nn.LayerNorm(hidden_size)
        self.block = MambaBlock(
            d_model=hidden_size,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.block(self.norm(x)))


class DeepHitMambaCompeting(BaseDeepHitCompetingModel):
    """Lean Mamba DeepHit model with attention pooling and residual skip."""

    def __init__(
        self,
        num_features: int,
        num_events: int,
        num_time_steps: int,
        hidden_size: int = 144,
        num_mamba_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        mamba_dropout: float = 0.15,
        fc_hidden: int = 192,
        fc_dropout: float = 0.2,
    ) -> None:
        super().__init__(num_events=num_events, num_time_steps=num_time_steps)

        self.num_features = num_features
        self.hidden_size = hidden_size

        self.input_proj = nn.Linear(num_features, hidden_size)
        self.input_proj_residual = nn.Linear(num_features, hidden_size)

        self.ssm_layers = nn.ModuleList(
            [
                ResidualMambaBlock(
                    hidden_size=hidden_size,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=mamba_dropout,
                )
                for _ in range(num_mamba_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_size)

        self.attn_query_proj = nn.Linear(hidden_size, hidden_size)
        self.attn_key_proj = nn.Linear(hidden_size, hidden_size)
        self.attn_scale = 1.0 / (hidden_size ** 0.5)

        self.cause_heads = self._build_cause_specific_heads(
            input_size=hidden_size,
            hidden_size=fc_hidden,
            dropout=fc_dropout,
            activation="gelu",
            use_batch_norm=False,
        )

        self.pred_dim = num_features - 1
        self.aux_head = nn.Linear(hidden_size, self.pred_dim)
        self._cache: dict[str, torch.Tensor] = {}

    def _lengths_from_mask(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = x[:, :, -1]
        lengths = mask.sum(dim=1).long().clamp(min=1, max=x.size(1))
        return mask, lengths

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask, lengths = self._lengths_from_mask(x)
        h = self.input_proj(x)
        last_input_proj = self.input_proj_residual(x[:, -1, :])

        for layer in self.ssm_layers:
            if self.training and h.requires_grad:
                h = checkpoint(layer, h, use_reentrant=False)
            else:
                h = layer(h)
        h = self.output_norm(h)

        query = self.attn_query_proj(last_input_proj)
        seq_repr = masked_attention_pooling(
            sequence=h,
            query=query,
            key_projection=self.attn_key_proj,
            scale=self.attn_scale,
            mask=mask,
        )
        seq_repr = seq_repr + last_input_proj

        self._cache = {
            "state_out": h,
            "mask": mask,
            "lengths": lengths,
        }

        logits = torch.stack([head(seq_repr) for head in self.cause_heads], dim=1)
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
