from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

try:
    from mamba_ssm import Mamba as MambaBlock
except ImportError:  # pragma: no cover - optional dependency
    MambaBlock = None


def masked_attention_pooling(
    sequence: torch.Tensor,
    query: torch.Tensor,
    key_projection: nn.Linear,
    scale: float,
    mask: torch.Tensor,
) -> torch.Tensor:
    keys = key_projection(sequence)
    scores = torch.bmm(query.unsqueeze(1), keys.transpose(1, 2)) * scale
    scores = scores.squeeze(1)
    scores = scores.masked_fill(~(mask > 0.5), -1e9)
    weights = torch.softmax(scores, dim=1)
    return torch.sum(sequence * weights.unsqueeze(-1), dim=1)


class DynamicDeepHitRNNTransformerCompeting(nn.Module):
    """Dynamic DeepHit backbone: GRU encoder + Transformer encoder + aux head."""

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
        max_seq_len: int = 500,
        fc_hidden: int = 128,
        fc_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if hidden_size % transformer_heads != 0:
            raise ValueError("hidden_size must be divisible by transformer_heads.")

        self.num_events = num_events
        self.num_time_steps = num_time_steps
        self.num_features = num_features
        self.max_seq_len = max_seq_len
        self.pred_dim = num_features - 1

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

        self.cause_specific_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, fc_hidden),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm1d(fc_hidden),
                    nn.Dropout(fc_dropout),
                    nn.Linear(fc_hidden, num_time_steps),
                )
                for _ in range(num_events)
            ]
        )

        self.aux_head = nn.Linear(hidden_size, self.pred_dim)
        self._cache: dict[str, torch.Tensor] = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = x[:, :, -1]
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
        self._cache = {
            "state_out": tr_out,
            "mask": mask,
            "lengths": seq_lens,
        }

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

    def aux_next_step_loss(self, x: torch.Tensor) -> torch.Tensor:
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


class ResidualMambaBlock(nn.Module):
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
                "mamba_ssm is required for DynamicDeepHitMambaCompeting. "
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


class DynamicDeepHitMambaCompeting(nn.Module):
    """Dynamic DeepHit backbone: Mamba + attention pooling + aux head."""

    def __init__(
        self,
        num_features: int,
        num_events: int,
        num_time_steps: int,
        hidden_size: int = 128,
        num_mamba_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        mamba_dropout: float = 0.15,
        fc_hidden: int = 160,
        fc_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.num_events = num_events
        self.num_time_steps = num_time_steps
        self.pred_dim = num_features - 1

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

        self.cause_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, fc_hidden),
                    nn.GELU(),
                    nn.Dropout(fc_dropout),
                    nn.Linear(fc_hidden, num_time_steps),
                )
                for _ in range(num_events)
            ]
        )

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
