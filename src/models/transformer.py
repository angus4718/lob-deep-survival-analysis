"""Transformer-only backbone for DeepHit competing risks."""

from __future__ import annotations

import torch
from torch import nn

from .base import BaseDeepHitCompetingModel, masked_attention_pooling


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
        super().__init__(num_events=num_events, num_time_steps=num_time_steps)
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads.")

        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size

        self.input_projection = nn.Linear(num_features, hidden_size)
        self.input_proj_residual = nn.Linear(num_features, hidden_size)

        self.positional_embedding = nn.Parameter(torch.zeros(1, max_seq_len, hidden_size))
        nn.init.normal_(self.positional_embedding, mean=0.0, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=transformer_ff_dim,
            dropout=transformer_dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pre_encoder_norm = nn.LayerNorm(hidden_size)

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
        self.latest_transformer_output: torch.Tensor | None = None

    def _lengths_from_mask(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = x[:, :, -1]
        lengths = mask.sum(dim=1).long().clamp(min=1, max=x.size(1))
        return mask, lengths

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask, lengths = self._lengths_from_mask(x)
        seq_len = x.size(1)
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}.")

        x_proj = self.input_projection(x)
        last_input = x[:, -1, :]
        input_proj_residual = self.input_proj_residual(last_input)

        x_with_pos = x_proj + self.positional_embedding[:, :seq_len, :]
        tr_in = self.pre_encoder_norm(x_with_pos)

        seq_lens = mask.sum(dim=1).long().clamp(min=1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        pad_lens = seq_len - seq_lens.unsqueeze(1)
        padding_mask = positions < pad_lens

        tr_out = self.transformer(tr_in, src_key_padding_mask=padding_mask)
        self.latest_transformer_output = tr_out

        self._cache = {
            "state_out": tr_out,
            "mask": mask,
            "lengths": lengths,
        }

        query = self.attn_query_proj(input_proj_residual)
        seq_repr = masked_attention_pooling(
            sequence=tr_out,
            query=query,
            key_projection=self.attn_key_proj,
            scale=self.attn_scale,
            mask=mask,
        )

        combined_repr = seq_repr + input_proj_residual
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
