"""Mamba backbone for DeepHit competing risks."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .base import BaseDeepHitCompetingModel

try:
    from mamba_ssm import Mamba as MambaBlock
except ImportError:
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
                "Install it with `pip install mamba-ssm` and ensure any optional "
                "dependencies required by your environment are available."
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
        super().__init__(num_features, num_events, num_time_steps, hidden_size, fc_hidden, fc_dropout)
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

    def encode(
        self, x_proj: torch.Tensor, _mask: torch.Tensor, _lengths: torch.Tensor
    ) -> torch.Tensor:
        h = x_proj
        for layer in self.ssm_layers:
            if self.training:
                h = checkpoint(layer, h, use_reentrant=False)
            else:
                h = layer(h)
        return h
