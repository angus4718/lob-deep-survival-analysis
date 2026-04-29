"""Transformer-only backbone for DeepHit competing risks."""

from __future__ import annotations

import torch
import torch.utils.checkpoint
from torch import nn

from .base import BaseDeepHitCompetingModel


class DeepHitTransformerCompeting(BaseDeepHitCompetingModel):
    """Transformer encoder with learnable positional embeddings."""

    def __init__(
        self,
        num_features: int,
        num_events: int,
        num_time_steps: int,
        hidden_size: int = 96,
        num_layers: int = 2,
        num_heads: int = 4,
        transformer_ff_dim: int = 320,
        transformer_dropout: float = 0.1,
        max_seq_len: int = 20,
        fc_hidden: int = 112,
        fc_dropout: float = 0.2,
    ) -> None:
        super().__init__(num_features, num_events, num_time_steps, hidden_size, fc_hidden, fc_dropout)
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads.")

        self.max_seq_len = max_seq_len

        self.positional_embedding = nn.Parameter(torch.zeros(1, max_seq_len, hidden_size))
        nn.init.normal_(self.positional_embedding, mean=0.0, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=transformer_ff_dim,
            dropout=transformer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.post_transformer_norm = nn.LayerNorm(hidden_size)
        self.latest_transformer_output: torch.Tensor | None = None

    def encode(
        self, x_proj: torch.Tensor, _mask: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        seq_len = x_proj.size(1)
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}.")

        x_with_pos = x_proj + self.positional_embedding[:, :seq_len, :]
        tr_in = x_with_pos

        positions = torch.arange(seq_len, device=x_proj.device).unsqueeze(0)
        pad_lens = seq_len - lengths.unsqueeze(1)
        padding_mask = positions < pad_lens

        if self.training:
            tr_out = torch.utils.checkpoint.checkpoint(
                self.transformer,
                tr_in,
                use_reentrant=False,
                src_key_padding_mask=padding_mask,
            )
        else:
            tr_out = self.transformer(tr_in, src_key_padding_mask=padding_mask)

        tr_out = self.post_transformer_norm(tr_out)
        self.latest_transformer_output = tr_out
        return tr_out
