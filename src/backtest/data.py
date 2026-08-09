"""Dataset loading and feature materialization for backtests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.labeling.utils import ms_to_suffix


def _configured_post_trade_columns() -> list[str]:
    columns: list[str] = []
    for window_ms in CONFIG.labeling.tox_post_trade_move_windows_ms:
        suffix = ms_to_suffix(int(window_ms))
        columns.extend(
            [
                f"post_trade_best_bid_{suffix}",
                f"post_trade_best_ask_{suffix}",
            ]
        )
    return columns


DEFAULT_REQUIRED_COLUMNS = [
    "order_id",
    "entry_time",
    "duration_s",
    "event_type",
    "side",
    "price",
    "status_reason",
    "best_bid_at_entry",
    "best_ask_at_entry",
    "best_bid_at_execution",
    "best_ask_at_execution",
    "best_bid_at_post_trade",
    "best_ask_at_post_trade",
    "sequence_length",
    "lob_sequence_raw_top5",
    "lob_sequence",
    "toxicity_sequence",
] + _configured_post_trade_columns()

WINDOW_SELECTION_COLUMNS = [
    "event",
    "status_reason",
    "side",
    "price",
    "best_bid_at_entry",
    "best_ask_at_entry",
] + _configured_post_trade_columns()


@dataclass(frozen=True)
class BacktestFeatureBuilder:
    """Materialize dynamic DeepHit input for one labeled order row."""

    lookback_steps: int = 500
    lob_dim: int = 20
    tox_dim: int = 12
    snapshot_policy: str = "entry"
    feat_mean: np.ndarray | None = None
    feat_std: np.ndarray | None = None
    lob_sequence_col: str = "lob_sequence_raw_top5"
    fallback_lob_sequence_col: str = "lob_sequence"
    tox_sequence_col: str = "toxicity_sequence"
    sequence_length_col: str = "sequence_length"

    @property
    def feature_dim(self) -> int:
        return int(self.lob_dim + self.tox_dim + 2)

    def build(self, row: pd.Series, end_idx: int | None = None) -> np.ndarray:
        lob_seq, tox_seq, seq_len = self.materialize_sequences(row)
        selected_end_idx = self._select_end_idx(seq_len) if end_idx is None else int(end_idx)
        return self.build_from_sequences(
            lob_seq=lob_seq,
            tox_seq=tox_seq,
            side=row.get("side"),
            end_idx=selected_end_idx,
            seq_len=seq_len,
        )

    def materialize_sequences(self, row: pd.Series) -> tuple[np.ndarray, np.ndarray, int]:
        lob_rep = row.get(self.lob_sequence_col)
        if lob_rep is None and self.fallback_lob_sequence_col:
            lob_rep = row.get(self.fallback_lob_sequence_col)
        tox_rep = row.get(self.tox_sequence_col)

        lob_seq = safe_stack_sequence(lob_rep, self.lob_dim)
        tox_seq = safe_stack_sequence(tox_rep, self.tox_dim)
        seq_len = min(lob_seq.shape[0], tox_seq.shape[0])
        seq_len = self._clip_sequence_len(seq_len, row.get(self.sequence_length_col))
        if seq_len <= 0:
            raise ValueError(f"row {getattr(row, 'name', '<unknown>')} has no sequence data")
        return lob_seq[:seq_len], tox_seq[:seq_len], int(seq_len)

    def build_from_sequences(
        self,
        *,
        lob_seq: np.ndarray,
        tox_seq: np.ndarray,
        side,
        end_idx: int,
        seq_len: int | None = None,
    ) -> np.ndarray:
        if seq_len is None:
            seq_len = min(np.asarray(lob_seq).shape[0], np.asarray(tox_seq).shape[0])
        selected_end_idx = int(np.clip(int(end_idx), 0, int(seq_len) - 1))
        return build_dynamic_feature_window(
            lob_seq=np.asarray(lob_seq)[: int(seq_len)],
            tox_seq=np.asarray(tox_seq)[: int(seq_len)],
            side=side,
            end_idx=selected_end_idx,
            lookback_steps=self.lookback_steps,
            lob_dim=self.lob_dim,
            tox_dim=self.tox_dim,
            feat_mean=self.feat_mean,
            feat_std=self.feat_std,
        )

    def _clip_sequence_len(self, observed_len: int, raw_seq_len) -> int:
        try:
            seq_len = int(raw_seq_len)
        except (TypeError, ValueError):
            return int(observed_len)
        if seq_len <= 0:
            return int(observed_len)
        return int(min(observed_len, seq_len))

    def _select_end_idx(self, seq_len: int) -> int:
        policy = self.snapshot_policy.lower()
        if policy == "entry":
            return self.initial_end_idx(seq_len)
        if policy in {"latest", "final"}:
            return seq_len - 1
        if policy.startswith("index:"):
            return int(np.clip(int(policy.split(":", 1)[1]), 0, seq_len - 1))
        raise ValueError(
            "snapshot_policy must be one of 'entry', 'latest', 'final', or 'index:<n>'"
        )

    def sequence_length(self, row: pd.Series) -> int:
        return self.materialize_sequences(row)[2]

    def initial_end_idx(self, seq_len: int) -> int:
        return int(min(self.lookback_steps - 1, max(seq_len - 1, 0)))


class BacktestDataset:
    """Parquet-backed labeled dataset iterator."""

    def __init__(
        self,
        path: str | Path,
        feature_builder: BacktestFeatureBuilder,
        *,
        columns: Iterable[str] | None = None,
        calibration_path: str | Path | None = None,
        sample_fraction: float = 1.0,
        sample_seed: int = CONFIG.random_seed,
        row_limit: int | None = None,
        row_offset: int = 0,
    ) -> None:
        if not (0.0 < float(sample_fraction) <= 1.0):
            raise ValueError("sample_fraction must be in the interval (0, 1].")
        if row_limit is not None and int(row_limit) < 1:
            raise ValueError("row_limit must be >= 1 or None.")
        if int(row_offset) < 0:
            raise ValueError("row_offset must be >= 0.")
        self.path = Path(path)
        self.feature_builder = feature_builder
        self.columns = list(columns) if columns is not None else list(DEFAULT_REQUIRED_COLUMNS)
        self.sample_fraction = float(sample_fraction)
        self.sample_seed = int(sample_seed)
        self.row_limit = int(row_limit) if row_limit is not None else None
        self.row_offset = int(row_offset)
        self.calibration_path = (
            Path(calibration_path)
            if calibration_path is not None
            else self._infer_train_calibration_path(self.path)
        )
        self._frame_cache: pd.DataFrame | None = None
        self._calibration_frame_cache: pd.DataFrame | None = None
        self._sequence_cache: dict[tuple, tuple[np.ndarray, np.ndarray, int]] = {}

    def load_frame(self) -> pd.DataFrame:
        if self._frame_cache is not None:
            return self._frame_cache
        if not self.path.exists():
            raise FileNotFoundError(f"Backtest dataset not found: {self.path}")
        df = pd.read_parquet(self.path, columns=self._available_columns(self.path, self.columns))
        if self.row_offset:
            df = df.iloc[self.row_offset :]
        if self.row_limit is not None:
            df = df.head(self.row_limit)
        if self.sample_fraction < 1.0:
            df = (
                df.sample(frac=self.sample_fraction, random_state=self.sample_seed)
                .sort_index()
            )
        self._frame_cache = df
        return df

    def load_calibration_frame(self) -> pd.DataFrame | None:
        if self._calibration_frame_cache is not None:
            return self._calibration_frame_cache
        if self.calibration_path is None or not self.calibration_path.exists():
            return None
        columns = self._available_columns(self.calibration_path, WINDOW_SELECTION_COLUMNS)
        if not columns:
            return None
        self._calibration_frame_cache = pd.read_parquet(self.calibration_path, columns=columns)
        return self._calibration_frame_cache

    def _available_columns(self, path: Path, columns: Iterable[str]) -> list[str]:
        try:
            import pyarrow.parquet as pq

            available = set(pq.ParquetFile(path).schema_arrow.names)
        except Exception:
            return list(columns)
        return [column for column in columns if column in available]

    def _infer_train_calibration_path(self, path: Path) -> Path | None:
        name = path.name
        if name.endswith("_test.parquet"):
            candidate = path.with_name(name.removesuffix("_test.parquet") + "_train.parquet")
            if candidate.exists():
                return candidate
        return None

    def iter_snapshots(self) -> "BacktestSnapshotIterable":
        return BacktestSnapshotIterable(self)

    def sequence_length_for_row(self, row_index: int, row: pd.Series) -> int:
        return self._materialized_sequences(row_index, row)[2]

    def build_features_for_row(
        self,
        row_index: int,
        row: pd.Series,
        *,
        end_idx: int,
    ) -> np.ndarray:
        lob_seq, tox_seq, seq_len = self._materialized_sequences(row_index, row)
        return self.feature_builder.build_from_sequences(
            lob_seq=lob_seq,
            tox_seq=tox_seq,
            side=row.get("side"),
            end_idx=end_idx,
            seq_len=seq_len,
        )

    def _materialized_sequences(
        self,
        row_index: int,
        row: pd.Series,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        key = self._row_cache_key(row_index, row)
        cached = self._sequence_cache.get(key)
        if cached is not None:
            return cached
        materialized = self.feature_builder.materialize_sequences(row)
        self._sequence_cache[key] = materialized
        return materialized

    def _row_cache_key(self, row_index: int, row: pd.Series) -> tuple:
        return (
            _cache_scalar(row.get("order_id")),
            _cache_scalar(row.get("entry_time")),
            int(row_index),
        )

    @classmethod
    def from_runtime_artifacts(
        cls,
        path: str | Path,
        *,
        runtime_npz_path: str | Path,
        lookback_steps: int = 500,
        lob_dim: int = 20,
        tox_dim: int = 12,
        snapshot_policy: str = "entry",
        sample_fraction: float = 1.0,
        sample_seed: int = CONFIG.random_seed,
        row_limit: int | None = None,
        row_offset: int = 0,
    ) -> "BacktestDataset":
        with np.load(runtime_npz_path, allow_pickle=False) as data:
            feat_mean = data["feat_mean"]
            feat_std = data["feat_std"]
        builder = BacktestFeatureBuilder(
            lookback_steps=lookback_steps,
            lob_dim=lob_dim,
            tox_dim=tox_dim,
            snapshot_policy=snapshot_policy,
            feat_mean=feat_mean,
            feat_std=feat_std,
        )
        return cls(
            path,
            builder,
            sample_fraction=sample_fraction,
            sample_seed=sample_seed,
            row_limit=row_limit,
            row_offset=row_offset,
        )


@dataclass(frozen=True)
class BacktestSnapshotIterable:
    """Iterable snapshots plus optional train-frame calibration context."""

    dataset: BacktestDataset

    def __iter__(self) -> Iterator:
        from .types import MarketSnapshot

        df = self.dataset.load_frame()
        for row_index, row in df.iterrows():
            seq_len = self.dataset.sequence_length_for_row(int(row_index), row)
            end_idx = self.dataset.feature_builder.initial_end_idx(seq_len)
            yield MarketSnapshot(
                row_index=int(row_index),
                row=row,
                features=self.dataset.build_features_for_row(
                    int(row_index),
                    row,
                    end_idx=end_idx,
                ),
                update_idx=0,
                end_idx=end_idx,
                is_initial=True,
                position_open=False,
            )

    def calibration_frame(self) -> pd.DataFrame | None:
        return self.dataset.load_calibration_frame()

    def lifecycle_snapshots(
        self,
        snapshot: "MarketSnapshot",
        *,
        stride: int = 1,
        max_evaluations: int | None = None,
    ) -> Iterator:
        from .types import MarketSnapshot

        if int(stride) < 1:
            raise ValueError("stride must be >= 1")
        if max_evaluations is not None and int(max_evaluations) < 1:
            raise ValueError("max_evaluations must be >= 1 or None")

        seq_len = self.dataset.sequence_length_for_row(snapshot.row_index, snapshot.row)
        initial_end_idx = self.dataset.feature_builder.initial_end_idx(seq_len)
        candidates = np.arange(initial_end_idx + 1, seq_len, int(stride), dtype=np.int64)
        if max_evaluations is not None and candidates.size > int(max_evaluations):
            positions = np.rint(
                np.linspace(0, candidates.size - 1, num=int(max_evaluations), dtype=np.float64)
            ).astype(np.int64)
            positions = np.unique(np.clip(positions, 0, candidates.size - 1))
            if positions.size < int(max_evaluations):
                all_pos = np.arange(candidates.size, dtype=np.int64)
                missing = all_pos[~np.isin(all_pos, positions)]
                positions = np.concatenate(
                    [positions, missing[: int(max_evaluations) - positions.size]]
                )
            candidates = candidates[np.sort(positions[: int(max_evaluations)])]

        for end_idx in candidates.tolist():
            yield MarketSnapshot(
                row_index=snapshot.row_index,
                row=snapshot.row,
                features=self.dataset.build_features_for_row(
                    snapshot.row_index,
                    snapshot.row,
                    end_idx=end_idx,
                ),
                update_idx=int(end_idx - initial_end_idx),
                end_idx=int(end_idx),
                is_initial=False,
                position_open=True,
            )


def safe_stack_sequence(rep, feat_dim: int) -> np.ndarray:
    if rep is None:
        return np.empty((0, feat_dim), dtype=np.float32)
    try:
        if pd.isna(rep):
            return np.empty((0, feat_dim), dtype=np.float32)
    except (ValueError, TypeError):
        pass
    rows = [np.asarray(row, dtype=np.float32).reshape(-1) for row in rep]
    if not rows:
        return np.empty((0, feat_dim), dtype=np.float32)
    arr = np.stack(rows, axis=0)
    if arr.ndim != 2 or arr.shape[1] != feat_dim:
        raise ValueError(f"Unexpected shape {arr.shape}; expected (*, {feat_dim})")
    return arr


def side_to_float(side_val) -> float:
    side_str = str(side_val).upper()
    if side_str == "B":
        return 1.0
    if side_str == "A":
        return 0.0
    return 0.5


def window_with_left_padding(
    arr: np.ndarray,
    *,
    end_idx: int,
    lookback_steps: int,
    feat_dim: int,
) -> tuple[np.ndarray, int]:
    start_idx = max(0, end_idx - lookback_steps + 1)
    chunk = arr[start_idx : end_idx + 1]
    valid_len = int(chunk.shape[0])
    if valid_len == lookback_steps:
        return chunk.astype(np.float32, copy=False), valid_len

    out = np.zeros((lookback_steps, feat_dim), dtype=np.float32)
    if valid_len > 0:
        out[-valid_len:, :] = chunk
    return out, valid_len


def build_dynamic_feature_window(
    *,
    lob_seq: np.ndarray,
    tox_seq: np.ndarray,
    side,
    end_idx: int,
    lookback_steps: int,
    lob_dim: int,
    tox_dim: int,
    feat_mean: np.ndarray | None = None,
    feat_std: np.ndarray | None = None,
) -> np.ndarray:
    """Build one model-ready dynamic feature window from raw LOB/tox sequences."""
    lob_arr = np.asarray(lob_seq, dtype=np.float32)
    tox_arr = np.asarray(tox_seq, dtype=np.float32)
    if lob_arr.ndim != 2 or lob_arr.shape[1] != int(lob_dim):
        raise ValueError(f"Unexpected LOB shape {lob_arr.shape}; expected (*, {lob_dim})")
    if tox_arr.ndim != 2 or tox_arr.shape[1] != int(tox_dim):
        raise ValueError(
            f"Unexpected toxicity shape {tox_arr.shape}; expected (*, {tox_dim})"
        )
    seq_len = min(lob_arr.shape[0], tox_arr.shape[0])
    if seq_len <= 0:
        raise ValueError("Cannot build a dynamic feature window from empty sequences")

    selected_end_idx = int(np.clip(int(end_idx), 0, seq_len - 1))
    lob_win, valid_len = window_with_left_padding(
        lob_arr[:seq_len],
        end_idx=selected_end_idx,
        lookback_steps=int(lookback_steps),
        feat_dim=int(lob_dim),
    )
    tox_win, _ = window_with_left_padding(
        tox_arr[:seq_len],
        end_idx=selected_end_idx,
        lookback_steps=int(lookback_steps),
        feat_dim=int(tox_dim),
    )

    feature_dim = int(lob_dim) + int(tox_dim) + 2
    x = np.zeros((int(lookback_steps), feature_dim), dtype=np.float32)
    x[:, : int(lob_dim)] = lob_win
    x[:, int(lob_dim) : int(lob_dim) + int(tox_dim)] = tox_win

    mask = np.zeros(int(lookback_steps), dtype=np.float32)
    if valid_len > 0:
        mask[-valid_len:] = 1.0
    x[:, -2] = side_to_float(side) * mask
    x[:, -1] = mask

    if feat_mean is not None and feat_std is not None:
        x = apply_dynamic_normalizer(x[None, :, :], feat_mean, feat_std)[0]
    return x.astype(np.float32, copy=False)


def apply_dynamic_normalizer(
    x: np.ndarray,
    feat_mean: np.ndarray,
    feat_std: np.ndarray,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    feat_mean = np.asarray(feat_mean, dtype=np.float32)
    feat_std = np.asarray(feat_std, dtype=np.float32)
    side_col_idx = x.shape[2] - 2
    mask_col_idx = x.shape[2] - 1

    x_norm = ((x - feat_mean) / feat_std).astype(np.float32)
    x_norm[..., side_col_idx] = x[..., side_col_idx]
    x_norm[..., mask_col_idx] = x[..., mask_col_idx]
    pad_rows = x[..., mask_col_idx] < 0.5
    x_norm[pad_rows] = 0.0
    return x_norm


def _cache_scalar(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, np.generic):
        return value.item()
    return value
