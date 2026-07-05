"""Live feature materialization for raw Databento backtests."""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.features.compose import ToxicityFeatures
from src.features.representation import RepresentationTransform

from .data import build_dynamic_feature_window
from .book_utils import BookSnapshot, book_to_snapshot


@dataclass
class LiveFeatureBuilder:
    """Build DeepHit features from a raw replay snapshot buffer."""

    lookback_steps: int = 500
    lob_dim: int = 20
    tox_dim: int = 12
    feat_mean: np.ndarray | None = None
    feat_std: np.ndarray | None = None
    max_buffer_len: int | None = None
    lob_transform: RepresentationTransform = field(
        default_factory=lambda: RepresentationTransform(representation="raw_top5")
    )
    toxicity_features: ToxicityFeatures = field(default_factory=ToxicityFeatures)

    def __post_init__(self) -> None:
        maxlen = int(self.max_buffer_len or self.lookback_steps)
        self.snapshot_buffer: collections.deque[BookSnapshot] = collections.deque(
            maxlen=max(1, maxlen)
        )

    @property
    def feature_dim(self) -> int:
        return int(self.lob_dim + self.tox_dim + 2)

    def append_book(self, book: Any, ts_event: int) -> bool:
        snapshot = book_to_snapshot(book, ts_event)
        if snapshot is None:
            return False
        self.snapshot_buffer.append(snapshot)
        return True

    def build_initial_sequences(
        self,
        *,
        current_vahead: int,
    ) -> tuple[list[list[float]], list[list[float]]]:
        snapshots = list(self.snapshot_buffer)
        if not snapshots:
            return [], []
        lob_tensor = self.lob_transform.transform_sequence_from_dicts(
            snapshots,
            self.lookback_steps,
            pad_to_length=False,
        )
        tox_tensor = self.toxicity_features.transform_sequence_from_dicts(
            snapshots,
            self.lookback_steps,
            pad_to_length=False,
        )
        lob_seq = lob_tensor.tolist()
        tox_seq = self.toxicity_features.augment_rows_with_queue_position(
            tox_tensor.tolist(),
            current_vahead,
        )
        return lob_seq, tox_seq

    def build_step(
        self,
        snapshot: BookSnapshot,
        *,
        current_vahead: int,
    ) -> tuple[list[float] | None, list[float] | None]:
        lob_step, tox_base = self.build_shared_step(snapshot)
        if lob_step is None or tox_base is None:
            return None, None
        return lob_step, self.augment_toxicity_step(tox_base, current_vahead)

    def build_shared_step(
        self,
        snapshot: BookSnapshot,
    ) -> tuple[list[float] | None, list[float] | None]:
        """Build a reusable raw LOB step and queue-position-free toxicity step."""
        lob_tensor = self.lob_transform.transform_sequence_from_dicts(
            [snapshot],
            n_lookback=1,
            pad_to_length=False,
        )
        tox_tensor = self.toxicity_features.transform_sequence_from_dicts(
            [snapshot],
            n_lookback=1,
            pad_to_length=False,
        )
        if lob_tensor.shape[0] == 0 or tox_tensor.shape[0] == 0:
            return None, None
        return lob_tensor[-1].tolist(), tox_tensor[-1].tolist()

    def augment_toxicity_step(
        self,
        tox_base_step: list[float],
        current_vahead: int,
    ) -> list[float]:
        """Add order-specific queue position to a shared toxicity base step."""
        return self.toxicity_features.augment_row_with_queue_position(
            list(tox_base_step),
            current_vahead,
        )

    def build_model_features(
        self,
        *,
        lob_sequence: list[list[float]],
        toxicity_sequence: list[list[float]],
        side: str,
        end_idx: int | None = None,
    ) -> np.ndarray:
        lob_arr = np.asarray(lob_sequence, dtype=np.float32)
        tox_arr = np.asarray(toxicity_sequence, dtype=np.float32)
        if lob_arr.ndim != 2 or tox_arr.ndim != 2:
            raise ValueError("LOB and toxicity sequences must be 2D arrays.")
        seq_len = min(lob_arr.shape[0], tox_arr.shape[0])
        if seq_len <= 0:
            raise ValueError("Cannot build features from empty live sequences.")
        selected_end_idx = seq_len - 1 if end_idx is None else int(end_idx)
        return build_dynamic_feature_window(
            lob_seq=lob_arr[:seq_len],
            tox_seq=tox_arr[:seq_len],
            side=side,
            end_idx=selected_end_idx,
            lookback_steps=self.lookback_steps,
            lob_dim=self.lob_dim,
            tox_dim=self.tox_dim,
            feat_mean=self.feat_mean,
            feat_std=self.feat_std,
        )
