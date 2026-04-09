"""GRU + Transformer backbone for DeepHit competing risks."""

from __future__ import annotations

import torch
from torch import nn

from .base import BaseDeepHitCompetingModel, masked_attention_pooling


class DeepHitRNNTransformerCompeting(BaseDeepHitCompetingModel):
    """GRU encoder followed by Transformer encoder with attention pooling."""

    def __init__(
        self,
        num_features: int,
        num_events: int,
        num_time_steps: int,
        hidden_size: int = 96,
        num_layers: int = 2,
        rnn_dropout: float = 0.2,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        transformer_ff_dim: int = 192,
        transformer_dropout: float = 0.1,
        max_seq_len: int = 20,
        fc_hidden: int = 128,
        fc_dropout: float = 0.2,
    ) -> None:
        super().__init__(num_events=num_events, num_time_steps=num_time_steps)
        if hidden_size % transformer_heads != 0:
            raise ValueError("hidden_size must be divisible by transformer_heads.")

        self.max_seq_len = max_seq_len

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

        self.positional_embedding = nn.Parameter(torch.zeros(1, max_seq_len, hidden_size))
        nn.init.normal_(self.positional_embedding, mean=0.0, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=transformer_heads,
            dim_feedforward=transformer_ff_dim,
            dropout=transformer_dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.pre_transformer_norm = nn.LayerNorm(hidden_size)

        self.cause_specific_heads = self._build_cause_specific_heads(
            input_size=hidden_size,
            hidden_size=fc_hidden,
            dropout=fc_dropout,
            activation="relu",
            use_batch_norm=True,
        )

        self.latest_transformer_output: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = x[:, :, -1]  # (batch, seq_len)
        seq_len = x.size(1)
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}.")

        last_input = x[:, -1, :]
        input_proj = self.input_proj_residual(last_input)

        rnn_out, _ = self.rnn(x)
        tr_in = self.pre_transformer_norm(rnn_out + self.positional_embedding[:, :seq_len, :])

        seq_lens = mask.sum(dim=1).long().clamp(min=1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        pad_lens = seq_len - seq_lens.unsqueeze(1)
        padding_mask = positions < pad_lens

        tr_out = self.transformer(tr_in, src_key_padding_mask=padding_mask)
        self.latest_transformer_output = tr_out

        query = self.attn_query_proj(input_proj)
        seq_repr = masked_attention_pooling(
            sequence=tr_out,
            query=query,
            key_projection=self.attn_key_proj,
            scale=self.attn_scale,
            mask=mask,
        )

        combined_repr = seq_repr + input_proj
        logits = torch.stack([head(combined_repr) for head in self.cause_specific_heads], dim=1)
        return logits
