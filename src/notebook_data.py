"""Notebook data-preparation helpers for standardized and dynamic DeepHit notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from pycox.preprocessing.discretization import IdxDiscUnknownC, discretize
from pycox.preprocessing.label_transforms import LabTransDiscreteTime


def safe_stack_representation(rep, feat_dim: int) -> np.ndarray:
    # Handle None and pandas NA values
    if rep is None:
        return np.empty((0, feat_dim), dtype=np.float32)
    try:
        if pd.isna(rep):
            return np.empty((0, feat_dim), dtype=np.float32)
    except (ValueError, TypeError):
        pass  # rep is not a scalar, proceed with iteration
    rows = [np.asarray(row, dtype=np.float32).reshape(-1) for row in rep]
    if len(rows) == 0:
        return np.empty((0, feat_dim), dtype=np.float32)
    arr = np.stack(rows, axis=0)
    if arr.ndim != 2 or arr.shape[1] != feat_dim:
        raise ValueError(f"Unexpected shape {arr.shape}; expected (*, {feat_dim})")
    return arr


def safe_stack_sequence(rep, feat_dim: int) -> np.ndarray:
    """Stack a variable-length list representation into a 2D float array."""
    return safe_stack_representation(rep, feat_dim=feat_dim)


def side_to_float(side_val) -> float:
    """Convert side encoding to a numeric feature used by notebook models."""
    side_str = str(side_val).upper()
    if side_str == "B":
        return 1.0
    if side_str == "A":
        return 0.0
    return 0.5


def window_with_left_padding(
    arr: np.ndarray,
    end_idx: int,
    lookback_steps: int,
    feat_dim: int,
) -> tuple[np.ndarray, int]:
    """Create a trailing lookback window ending at `end_idx`, left-padded with zeros."""
    start_idx = max(0, end_idx - lookback_steps + 1)
    chunk = arr[start_idx : end_idx + 1]
    valid_len = chunk.shape[0]

    if valid_len == lookback_steps:
        return chunk, valid_len

    out = np.zeros((lookback_steps, feat_dim), dtype=np.float32)
    if valid_len > 0:
        out[-valid_len:, :] = chunk
    return out, valid_len


def left_pad_or_truncate(arr: np.ndarray, lookback_steps: int) -> tuple[np.ndarray, int]:
    if arr.shape[0] >= lookback_steps:
        return arr[-lookback_steps:], lookback_steps
    valid_len = arr.shape[0]
    pad_len = lookback_steps - valid_len
    pad = np.zeros((pad_len, arr.shape[1]), dtype=np.float32)
    return np.concatenate([pad, arr], axis=0), valid_len


def extract_lob_features(df: pd.DataFrame, lookback_steps: int, feat_dim: int = 20) -> np.ndarray:
    rows = []
    for rep in df["entry_representation"]:
        arr_raw = safe_stack_representation(rep, feat_dim=feat_dim)
        arr, _ = left_pad_or_truncate(arr_raw, lookback_steps)
        rows.append(arr)
    return np.stack(rows, axis=0)


def extract_toxicity_features(df: pd.DataFrame, lookback_steps: int, feat_dim: int = 12) -> np.ndarray:
    side_raw = df["side"]
    if pd.api.types.is_numeric_dtype(side_raw):
        side_vals = side_raw.astype(np.float32).to_numpy()
    else:
        side_vals = (
            side_raw.astype(str)
            .str.upper()
            .map({"B": 1.0, "A": 0.0})
            .fillna(0.5)
            .astype(np.float32)
            .to_numpy()
        )

    rows = []
    for rep, side_val in zip(df["toxicity_representation"], side_vals):
        arr_raw = safe_stack_representation(rep, feat_dim=feat_dim)
        arr, valid_len = left_pad_or_truncate(arr_raw, lookback_steps)

        mask_col = np.zeros((lookback_steps, 1), dtype=np.float32)
        if valid_len > 0:
            mask_col[-valid_len:, 0] = 1.0

        side_col = np.full((lookback_steps, 1), side_val, dtype=np.float32) * mask_col
        rows.append(np.concatenate([arr, side_col, mask_col], axis=1))

    return np.stack(rows, axis=0)


@dataclass(frozen=True)
class DynamicOrderStore:
    """Compact order-level storage for dynamic sample reconstruction."""

    lob_sequences: list[np.ndarray]
    tox_sequences: list[np.ndarray]
    side_values: np.ndarray
    order_ids: np.ndarray
    entry_times: np.ndarray
    source_row_idx: np.ndarray
    lookback_steps: int
    lob_dim: int
    tox_dim: int

    def __len__(self) -> int:
        return len(self.lob_sequences)


@dataclass(frozen=True)
class DynamicSampleManifest:
    """Sample-level pointers and labels for dynamic window materialization."""

    order_ptr: np.ndarray
    end_idx: np.ndarray
    y: np.ndarray
    d: np.ndarray
    order_ids: np.ndarray
    entry_times: np.ndarray
    source_row_idx: np.ndarray
    update_idx: np.ndarray

    def __len__(self) -> int:
        return int(self.order_ptr.shape[0])


def _infer_tox_time_delta_col(tox_seq: np.ndarray) -> int:
    """Infer which toxicity feature column stores log1p(time_delta_ms).

    Current dataset conventions:
    - 11 columns: last column is time-delta.
    - 12 columns: queue-position is appended last, so time-delta is second-last.
    """
    if tox_seq.ndim != 2 or tox_seq.shape[1] <= 0:
        raise ValueError("toxicity sequence must be a 2D array with at least one column")
    if tox_seq.shape[1] >= 12:
        return tox_seq.shape[1] - 2
    return tox_seq.shape[1] - 1


def _select_sample_indices(
    anchor_idx: int,
    seq_len: int,
    max_samples_per_order: int | None,
) -> list[int]:
    """Select update indices to cover an order lifecycle with optional cap.

    When capped, indices are approximately uniformly spaced from anchor to final
    update so late, information-rich snapshots are not systematically dropped.
    """
    candidates = np.arange(anchor_idx, seq_len, dtype=np.int64)
    if max_samples_per_order is None or candidates.size <= max_samples_per_order:
        return candidates.tolist()

    target = int(max_samples_per_order)
    positions = np.rint(
        np.linspace(0, candidates.size - 1, num=target, dtype=np.float64)
    ).astype(np.int64)
    positions = np.unique(np.clip(positions, 0, candidates.size - 1))

    if positions.size < target:
        all_pos = np.arange(candidates.size, dtype=np.int64)
        missing = all_pos[~np.isin(all_pos, positions)]
        positions = np.concatenate([positions, missing[: target - positions.size]])

    positions = np.sort(positions[:target])
    return candidates[positions].tolist()


def build_dynamic_samples_manifest(
    df: pd.DataFrame,
    lookback_steps: int,
    lob_dim: int,
    tox_dim: int,
    admin_censor_time: float | None = None,
    duration_col: str = "duration_s",
    event_col: str = "event_type_competing",
    side_col: str = "side",
    order_id_col: str = "order_id",
    entry_time_col: str = "entry_time",
    lob_col: str = "lob_sequence",
    tox_col: str = "toxicity_sequence",
    tox_time_delta_col: int | None = None,
    seq_len_col: str = "sequence_length",
    max_samples_per_order: int | None = None,
    validate_remaining_time: bool = True,
) -> tuple[DynamicOrderStore, DynamicSampleManifest]:
    """Build compact order storage and sample manifest for dynamic samples.

    This stores each order's raw sequences once and records per-sample pointers
    instead of materializing all rolling windows up front.
    """

    if admin_censor_time is not None and admin_censor_time <= 0:
        raise ValueError("admin_censor_time must be positive when provided")

    if max_samples_per_order is not None and max_samples_per_order < 1:
        raise ValueError("max_samples_per_order must be >= 1 or None")

    order_lob_sequences: list[np.ndarray] = []
    order_tox_sequences: list[np.ndarray] = []
    order_side_values: list[float] = []
    order_ids_list: list[int] = []
    entry_times_list: list[int] = []
    order_source_row_idx_list: list[int] = []

    sample_order_ptr: list[int] = []
    sample_end_idx: list[int] = []
    sample_y: list[float] = []
    sample_d: list[int] = []
    sample_order_ids: list[int] = []
    sample_entry_times: list[int] = []
    sample_source_row_idx: list[int] = []
    sample_update_idx: list[int] = []

    for row_idx, row in enumerate(df.itertuples(index=False)):
        lob_seq = safe_stack_sequence(getattr(row, lob_col), lob_dim)
        tox_seq = safe_stack_sequence(getattr(row, tox_col), tox_dim)

        seq_len = min(lob_seq.shape[0], tox_seq.shape[0])
        seq_len_raw = getattr(row, seq_len_col, np.nan)
        try:
            seq_len_val = float(seq_len_raw)
            if np.isfinite(seq_len_val) and seq_len_val > 0:
                seq_len = min(seq_len, int(seq_len_val))
        except (TypeError, ValueError):
            pass

        if seq_len <= 0:
            continue

        lob_seq = lob_seq[:seq_len]
        tox_seq = tox_seq[:seq_len]

        if tox_time_delta_col is None:
            time_delta_col = _infer_tox_time_delta_col(tox_seq)
        else:
            time_delta_col = int(tox_time_delta_col)
            if not (-tox_seq.shape[1] <= time_delta_col < tox_seq.shape[1]):
                raise ValueError(
                    f"tox_time_delta_col={time_delta_col} is out of bounds for "
                    f"toxicity sequence width {tox_seq.shape[1]}"
                )

        log_delta_ms = np.nan_to_num(
            tox_seq[:, time_delta_col].astype(np.float64),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        delta_ms = np.expm1(np.clip(log_delta_ms, a_min=0.0, a_max=50.0))
        cum_elapsed_s = np.cumsum(np.maximum(delta_ms, 0.0)) / 1e3

        anchor_idx = min(lookback_steps - 1, seq_len - 1)

        total_duration_s = float(getattr(row, duration_col))
        event_code = int(getattr(row, event_col))
        side_val = side_to_float(getattr(row, side_col))

        raw_order_id = getattr(row, order_id_col, None)
        if pd.isna(raw_order_id):
            order_id = row_idx
        else:
            order_id = int(raw_order_id)

        entry_time = int(getattr(row, entry_time_col))

        sample_indices = _select_sample_indices(
            anchor_idx=anchor_idx,
            seq_len=seq_len,
            max_samples_per_order=max_samples_per_order,
        )

        order_ptr = len(order_lob_sequences)
        order_lob_sequences.append(lob_seq)
        order_tox_sequences.append(tox_seq)
        order_side_values.append(side_val)
        order_ids_list.append(order_id)
        entry_times_list.append(entry_time)
        order_source_row_idx_list.append(row_idx)

        prev_remaining_s: float | None = None
        for end_idx in sample_indices:
            elapsed_s = max(float(cum_elapsed_s[end_idx]), 0.0)
            remaining_s_true = max(total_duration_s - elapsed_s, 0.0)

            if (
                validate_remaining_time
                and prev_remaining_s is not None
                and remaining_s_true > prev_remaining_s
            ):
                raise ValueError(
                    "remaining time increased within one order; "
                    "check toxicity time-delta column/units. "
                    f"order_id={order_id}, source_row={row_idx}"
                )
            prev_remaining_s = remaining_s_true

            if admin_censor_time is None:
                remaining_s = remaining_s_true
                sample_event_code = event_code
            else:
                if event_code > 0 and remaining_s_true <= admin_censor_time:
                    sample_event_code = event_code
                    remaining_s = remaining_s_true
                else:
                    sample_event_code = 0
                    remaining_s = min(remaining_s_true, admin_censor_time)
                remaining_s = max(remaining_s, 0.0)

            sample_order_ptr.append(order_ptr)
            sample_end_idx.append(end_idx)
            sample_y.append(remaining_s)
            sample_d.append(sample_event_code)
            sample_order_ids.append(order_id)
            sample_entry_times.append(entry_time)
            sample_source_row_idx.append(row_idx)
            sample_update_idx.append(end_idx - anchor_idx)

    order_store = DynamicOrderStore(
        lob_sequences=order_lob_sequences,
        tox_sequences=order_tox_sequences,
        side_values=np.asarray(order_side_values, dtype=np.float32),
        order_ids=np.asarray(order_ids_list, dtype=np.int64),
        entry_times=np.asarray(entry_times_list, dtype=np.int64),
        source_row_idx=np.asarray(order_source_row_idx_list, dtype=np.int64),
        lookback_steps=lookback_steps,
        lob_dim=lob_dim,
        tox_dim=tox_dim,
    )
    manifest = DynamicSampleManifest(
        order_ptr=np.asarray(sample_order_ptr, dtype=np.int32),
        end_idx=np.asarray(sample_end_idx, dtype=np.int32),
        y=np.asarray(sample_y, dtype=np.float32),
        d=np.asarray(sample_d, dtype=np.int64),
        order_ids=np.asarray(sample_order_ids, dtype=np.int64),
        entry_times=np.asarray(sample_entry_times, dtype=np.int64),
        source_row_idx=np.asarray(sample_source_row_idx, dtype=np.int64),
        update_idx=np.asarray(sample_update_idx, dtype=np.int32),
    )
    return order_store, manifest


def _materialize_dynamic_window(
    order_store: DynamicOrderStore,
    order_ptr: int,
    end_idx: int,
) -> np.ndarray:
    """Materialize one dynamic feature window from compact order storage."""
    lob_seq = order_store.lob_sequences[order_ptr]
    tox_seq = order_store.tox_sequences[order_ptr]
    side_val = float(order_store.side_values[order_ptr])

    lob_win, valid_len = window_with_left_padding(
        lob_seq,
        end_idx=end_idx,
        lookback_steps=order_store.lookback_steps,
        feat_dim=order_store.lob_dim,
    )
    tox_win, _ = window_with_left_padding(
        tox_seq,
        end_idx=end_idx,
        lookback_steps=order_store.lookback_steps,
        feat_dim=order_store.tox_dim,
    )

    feat_dim_total = order_store.lob_dim + order_store.tox_dim + 2
    x = np.zeros((order_store.lookback_steps, feat_dim_total), dtype=np.float32)
    x[:, : order_store.lob_dim] = lob_win
    x[:, order_store.lob_dim : order_store.lob_dim + order_store.tox_dim] = tox_win

    mask_col = np.zeros(order_store.lookback_steps, dtype=np.float32)
    if valid_len > 0:
        mask_col[-valid_len:] = 1.0
    x[:, -2] = side_val * mask_col
    x[:, -1] = mask_col
    return x


def materialize_dynamic_samples_from_manifest(
    order_store: DynamicOrderStore,
    manifest: DynamicSampleManifest,
    sample_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Materialize dynamic windows (and labels) for selected sample indices."""
    if sample_indices is None:
        sample_indices_arr = np.arange(len(manifest), dtype=np.int64)
    else:
        sample_indices_arr = np.asarray(sample_indices, dtype=np.int64).reshape(-1)

    feat_dim_total = order_store.lob_dim + order_store.tox_dim + 2
    n_samples = int(sample_indices_arr.shape[0])
    if n_samples == 0:
        x = np.empty((0, order_store.lookback_steps, feat_dim_total), dtype=np.float32)
        return (
            x,
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int32),
        )

    x = np.empty((n_samples, order_store.lookback_steps, feat_dim_total), dtype=np.float32)
    for out_idx, sample_idx in enumerate(sample_indices_arr.tolist()):
        order_ptr = int(manifest.order_ptr[sample_idx])
        end_idx = int(manifest.end_idx[sample_idx])
        x[out_idx] = _materialize_dynamic_window(order_store, order_ptr, end_idx)

    y = manifest.y[sample_indices_arr].astype(np.float32, copy=False)
    d = manifest.d[sample_indices_arr].astype(np.int64, copy=False)
    order_ids = manifest.order_ids[sample_indices_arr].astype(np.int64, copy=False)
    entry_times = manifest.entry_times[sample_indices_arr].astype(np.int64, copy=False)
    update_idxs = manifest.update_idx[sample_indices_arr].astype(np.int32, copy=False)
    return x, y, d, order_ids, entry_times, update_idxs


def select_manifest_indices_by_order_ids(
    manifest: DynamicSampleManifest,
    target_order_ids: np.ndarray,
) -> np.ndarray:
    """Return sample indices whose order ids belong to `target_order_ids`."""
    target_order_ids_arr = np.asarray(target_order_ids, dtype=np.int64).reshape(-1)
    if target_order_ids_arr.size == 0 or len(manifest) == 0:
        return np.empty((0,), dtype=np.int64)

    mask = np.isin(manifest.order_ids, target_order_ids_arr)
    return np.flatnonzero(mask).astype(np.int64)


def _cap_manifest_indices_random_by_source_row(
    manifest: DynamicSampleManifest,
    sample_indices: np.ndarray,
    max_samples_per_source_row: int,
    *,
    seed: int | None,
) -> np.ndarray:
    """Randomly cap selected sample indices per source-row group."""
    idx = np.asarray(sample_indices, dtype=np.int64).reshape(-1)
    if idx.size == 0:
        return np.empty((0,), dtype=np.int64)

    cap = int(max_samples_per_source_row)
    if cap < 1:
        raise ValueError("max_samples_per_source_row must be >= 1 or None")

    source_rows = manifest.source_row_idx[idx].astype(np.int64, copy=False)
    rng = np.random.default_rng(seed)

    selected_chunks: list[np.ndarray] = []
    for source_row in np.unique(source_rows):
        group_idx = idx[source_rows == source_row]
        if group_idx.size > cap:
            group_idx = np.sort(
                rng.choice(group_idx, size=cap, replace=False).astype(np.int64)
            )
        else:
            group_idx = np.sort(group_idx.astype(np.int64, copy=False))
        selected_chunks.append(group_idx)

    return np.sort(np.concatenate(selected_chunks).astype(np.int64, copy=False))


def select_manifest_indices_by_source_rows(
    manifest: DynamicSampleManifest,
    target_source_rows: np.ndarray,
    *,
    max_samples_per_source_row: int | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Return manifest indices for source rows, with optional random per-row cap.

    Args:
        manifest: Dynamic sample manifest.
        target_source_rows: Source dataframe row indices to select.
        max_samples_per_source_row: Optional random cap per source row.
        seed: Random seed used when cap is applied.
    """
    target_source_rows_arr = np.asarray(target_source_rows, dtype=np.int64).reshape(-1)
    if target_source_rows_arr.size == 0 or len(manifest) == 0:
        return np.empty((0,), dtype=np.int64)

    mask = np.isin(manifest.source_row_idx, target_source_rows_arr)
    selected = np.flatnonzero(mask).astype(np.int64)
    if max_samples_per_source_row is None:
        return selected
    return _cap_manifest_indices_random_by_source_row(
        manifest,
        selected,
        max_samples_per_source_row,
        seed=seed,
    )


class DynamicSampleDataset(torch.utils.data.Dataset):
    """Lazy dataset that materializes dynamic windows per sample on demand."""

    def __init__(
        self,
        order_store: DynamicOrderStore,
        manifest: DynamicSampleManifest,
        sample_indices: np.ndarray | None = None,
    ):
        self.order_store = order_store
        self.manifest = manifest
        if sample_indices is None:
            self.sample_indices = np.arange(len(manifest), dtype=np.int64)
        else:
            self.sample_indices = np.asarray(sample_indices, dtype=np.int64).reshape(-1)

    def __len__(self) -> int:
        return int(self.sample_indices.shape[0])

    def __getitem__(self, idx: int):
        sample_idx = int(self.sample_indices[idx])
        order_ptr = int(self.manifest.order_ptr[sample_idx])
        end_idx = int(self.manifest.end_idx[sample_idx])
        x = _materialize_dynamic_window(self.order_store, order_ptr, end_idx)

        return (
            torch.from_numpy(x),
            torch.tensor(float(self.manifest.y[sample_idx]), dtype=torch.float32),
            torch.tensor(int(self.manifest.d[sample_idx]), dtype=torch.int64),
            torch.tensor(int(self.manifest.order_ids[sample_idx]), dtype=torch.int64),
            torch.tensor(int(self.manifest.entry_times[sample_idx]), dtype=torch.int64),
            torch.tensor(int(self.manifest.update_idx[sample_idx]), dtype=torch.int64),
        )


def build_dynamic_samples(
    df: pd.DataFrame,
    lookback_steps: int,
    lob_dim: int,
    tox_dim: int,
    admin_censor_time: float | None = None,
    duration_col: str = "duration_s",
    event_col: str = "event_type_competing",
    side_col: str = "side",
    order_id_col: str = "order_id",
    entry_time_col: str = "entry_time",
    lob_col: str = "lob_sequence",
    tox_col: str = "toxicity_sequence",
    tox_time_delta_col: int | None = None,
    seq_len_col: str = "sequence_length",
    max_samples_per_order: int | None = None,
    validate_remaining_time: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Expand each order into per-update dynamic samples with remaining-time targets.

    This eager API is preserved for backwards compatibility and now uses the
    manifest-based path internally.
    """
    order_store, manifest = build_dynamic_samples_manifest(
        df,
        lookback_steps=lookback_steps,
        lob_dim=lob_dim,
        tox_dim=tox_dim,
        admin_censor_time=admin_censor_time,
        duration_col=duration_col,
        event_col=event_col,
        side_col=side_col,
        order_id_col=order_id_col,
        entry_time_col=entry_time_col,
        lob_col=lob_col,
        tox_col=tox_col,
        tox_time_delta_col=tox_time_delta_col,
        seq_len_col=seq_len_col,
        max_samples_per_order=max_samples_per_order,
        validate_remaining_time=validate_remaining_time,
    )
    return materialize_dynamic_samples_from_manifest(order_store, manifest)


def normalize_dynamic_sequences(
    x_train: np.ndarray,
    x_other_list: Sequence[np.ndarray],
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
    """Normalize dynamic sequences and preserve side/mask channels.

    Assumes the final two channels are side and mask respectively.
    """
    feat_mean = x_train.mean(axis=(0, 1), keepdims=True)
    feat_std = x_train.std(axis=(0, 1), keepdims=True) + 1e-8

    side_col_idx = x_train.shape[2] - 2
    mask_col_idx = x_train.shape[2] - 1
    feat_mean[..., side_col_idx] = 0.0
    feat_std[..., side_col_idx] = 1.0
    feat_mean[..., mask_col_idx] = 0.0
    feat_std[..., mask_col_idx] = 1.0

    x_train_norm = apply_dynamic_normalizer(x_train, feat_mean, feat_std)
    x_other_norm = [apply_dynamic_normalizer(x, feat_mean, feat_std) for x in x_other_list]
    return x_train_norm, x_other_norm, feat_mean, feat_std


def apply_dynamic_normalizer(
    x: np.ndarray,
    feat_mean: np.ndarray,
    feat_std: np.ndarray,
) -> np.ndarray:
    """Apply dynamic-sequence normalization while preserving side/mask channels."""
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


def fit_dynamic_normalizer_from_manifest(
    order_store: DynamicOrderStore,
    manifest: DynamicSampleManifest,
    sample_indices: np.ndarray,
    *,
    chunk_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute normalization statistics from selected manifest samples in chunks."""
    sample_indices_arr = np.asarray(sample_indices, dtype=np.int64).reshape(-1)
    if sample_indices_arr.size == 0:
        raise ValueError("sample_indices must not be empty")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    feat_dim_total = order_store.lob_dim + order_store.tox_dim + 2
    sum_feat = np.zeros((feat_dim_total,), dtype=np.float64)
    sumsq_feat = np.zeros((feat_dim_total,), dtype=np.float64)
    n_rows = 0

    for start in range(0, sample_indices_arr.size, chunk_size):
        chunk_idx = sample_indices_arr[start : start + chunk_size]
        x_chunk, _, _, _, _, _ = materialize_dynamic_samples_from_manifest(
            order_store,
            manifest,
            chunk_idx,
        )
        if x_chunk.size == 0:
            continue
        x_chunk64 = x_chunk.astype(np.float64, copy=False)
        sum_feat += x_chunk64.sum(axis=(0, 1))
        sumsq_feat += np.square(x_chunk64).sum(axis=(0, 1))
        n_rows += int(x_chunk64.shape[0] * x_chunk64.shape[1])

    if n_rows <= 0:
        raise ValueError("no rows found while computing normalization statistics")

    feat_mean_1d = sum_feat / float(n_rows)
    feat_var_1d = np.maximum((sumsq_feat / float(n_rows)) - np.square(feat_mean_1d), 0.0)
    feat_std_1d = np.sqrt(feat_var_1d) + 1e-8

    side_col_idx = feat_dim_total - 2
    mask_col_idx = feat_dim_total - 1
    feat_mean_1d[side_col_idx] = 0.0
    feat_std_1d[side_col_idx] = 1.0
    feat_mean_1d[mask_col_idx] = 0.0
    feat_std_1d[mask_col_idx] = 1.0

    feat_mean = feat_mean_1d.reshape(1, 1, -1).astype(np.float32)
    feat_std = feat_std_1d.reshape(1, 1, -1).astype(np.float32)
    return feat_mean, feat_std


def per_order_sample_weights(order_ids) -> np.ndarray:
    """Return per-sample weights so each order has equal total weight."""
    if len(order_ids) == 0:
        return np.array([], dtype=np.float32)

    counts = pd.Series(order_ids).value_counts()
    weights = np.asarray([1.0 / counts[oid] for oid in order_ids], dtype=np.float64)
    weights *= len(weights) / weights.sum()
    return weights.astype(np.float32)


def group_indices_by_order(order_ids: np.ndarray) -> dict[int, np.ndarray]:
    """Map each order id to the sorted sample indices belonging to that order."""
    order_ids = np.asarray(order_ids, dtype=np.int64)
    if order_ids.ndim != 1:
        raise ValueError("order_ids must be a 1D array")

    groups: dict[int, list[int]] = {}
    for idx, oid in enumerate(order_ids.tolist()):
        groups.setdefault(int(oid), []).append(int(idx))

    return {
        oid: np.asarray(indices, dtype=np.int64)
        for oid, indices in groups.items()
    }


def build_order_batch_indices(
    order_ids: np.ndarray,
    orders_per_batch: int,
    *,
    shuffle: bool = True,
    seed: int | None = None,
) -> list[np.ndarray]:
    """Build sample-index batches where each batch contains complete orders.

    Each returned array contains all sample indices for a subset of order ids.
    No order is split across batches.
    """
    if orders_per_batch <= 0:
        raise ValueError("orders_per_batch must be positive")

    order_to_indices = group_indices_by_order(np.asarray(order_ids, dtype=np.int64))
    if len(order_to_indices) == 0:
        return []

    unique_orders = np.asarray(sorted(order_to_indices.keys()), dtype=np.int64)
    if shuffle:
        rng = np.random.default_rng(seed)
        unique_orders = unique_orders[rng.permutation(len(unique_orders))]

    batches: list[np.ndarray] = []
    for start in range(0, len(unique_orders), orders_per_batch):
        batch_orders = unique_orders[start : start + orders_per_batch]
        batch_indices = np.concatenate(
            [order_to_indices[int(oid)] for oid in batch_orders], axis=0
        )
        batches.append(batch_indices.astype(np.int64, copy=False))

    return batches


def choose_time_horizon_from_train_fills(
    durations: np.ndarray,
    events: np.ndarray,
    train_mask: np.ndarray,
    *,
    fill_event_codes: Sequence[int] = (1, 2),
    quantile: float = 75.0,
) -> float:
    """Compute a horizon from training-set fill durations only.

    Args:
        durations: Per-sample durations in seconds.
        events: Per-sample event codes (0 means censored).
        train_mask: Boolean mask selecting the training split.
        fill_event_codes: Event codes considered as fills.
        quantile: Quantile in [0, 100] used to define the horizon.
    """
    durations_arr = np.asarray(durations, dtype=np.float64).reshape(-1)
    events_arr = np.asarray(events, dtype=np.int64).reshape(-1)
    train_mask_arr = np.asarray(train_mask, dtype=bool).reshape(-1)

    if not (durations_arr.size == events_arr.size == train_mask_arr.size):
        raise ValueError("durations, events, and train_mask must have the same length")
    if not (0.0 <= float(quantile) <= 100.0):
        raise ValueError("quantile must be between 0 and 100")

    is_train_fill = train_mask_arr & np.isin(events_arr, np.asarray(fill_event_codes, dtype=np.int64))
    if not np.any(is_train_fill):
        raise ValueError("No fill events found in the training split to compute T_max.")

    return float(np.percentile(durations_arr[is_train_fill], quantile))


def recensor_after_horizon(
    durations: np.ndarray,
    events: np.ndarray,
    horizon: float,
    *,
    censored_code: int = 0,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Recensor uncensored samples beyond a time horizon.

    Returns copied arrays `(durations_h, events_h, n_recensored)`.
    """
    durations_arr = np.asarray(durations, dtype=np.float32).reshape(-1)
    events_arr = np.asarray(events, dtype=np.int64).reshape(-1)

    if durations_arr.size != events_arr.size:
        raise ValueError("durations and events must have the same length")
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    durations_h = durations_arr.copy()
    events_h = events_arr.copy()

    late_uncensored_mask = (events_h > censored_code) & (durations_h > float(horizon))
    n_recensored = int(late_uncensored_mask.sum())
    if n_recensored > 0:
        events_h[late_uncensored_mask] = int(censored_code)
        durations_h[late_uncensored_mask] = float(horizon)

    return durations_h, events_h, n_recensored


def best_day_cut(target_row: int, day_end_idx: Sequence[int]) -> int:
    return min(range(len(day_end_idx)), key=lambda i: abs(day_end_idx[i] - target_row))


class LabTransform(LabTransDiscreteTime):
    @staticmethod
    def _fit_inputs_for_quantiles(
        durations: np.ndarray,
        events: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float | None]:
        """Drop censored-at-max rows from quantile cut estimation only."""
        durations_arr = np.asarray(durations)
        events_arr = np.asarray(events)
        if durations_arr.size == 0:
            return durations_arr, events_arr, None

        full_max = float(np.max(durations_arr))
        censored_at_max = (events_arr <= 0) & np.isclose(
            durations_arr,
            full_max,
            rtol=0.0,
            atol=1e-12,
        )
        if np.any(censored_at_max) and np.any(~censored_at_max):
            return durations_arr[~censored_at_max], events_arr[~censored_at_max], full_max
        return durations_arr, events_arr, full_max

    def _ensure_right_boundary(self, full_max: float | None) -> None:
        """Keep the last cut aligned with the full (pre-filter) duration range."""
        if full_max is None or self.cuts is None or len(self.cuts) == 0:
            return
        if full_max > float(self.cuts[-1]):
            self.cuts = self.cuts.copy()
            self.cuts[-1] = np.asarray(full_max, dtype=self.cuts.dtype)
            self.idu = IdxDiscUnknownC(self.cuts)

    def _fit_duration_quantile_cuts(
        self,
        fit_durations: np.ndarray,
        full_max: float | None,
    ) -> np.ndarray:
        """Build robust quantile cuts from durations only, avoiding KM edge cases."""
        durations_arr = np.asarray(fit_durations, dtype=np.float64)
        durations_arr = durations_arr[np.isfinite(durations_arr)]
        if durations_arr.size == 0:
            raise ValueError("Cannot fit quantile cuts on empty durations.")

        requested_cuts = int(self._cuts)
        if requested_cuts < 2:
            requested_cuts = 2

        min_cut = float(self._min) if self._min is not None else float(np.min(durations_arr))
        max_cut = full_max if full_max is not None else float(np.max(durations_arr))
        if max_cut <= min_cut:
            max_cut = min_cut + float(np.finfo(np.float64).eps)

        probs = np.linspace(0.0, 1.0, requested_cuts, dtype=np.float64)
        cuts = np.quantile(durations_arr, probs, method="linear")
        cuts = np.asarray(cuts, dtype=np.float64)
        cuts[0] = min_cut
        cuts[-1] = max_cut

        # Ties can collapse quantiles. Use deterministic rank-jitter as a
        # tie-break so we can still honor the requested cut count.
        if np.unique(cuts).size < requested_cuts:
            sorted_durations = np.sort(durations_arr, kind="mergesort")
            span = max(max_cut - min_cut, 1.0)
            jitter = np.linspace(0.0, span * 1e-9, sorted_durations.size, dtype=np.float64)
            cuts = np.quantile(sorted_durations + jitter, probs, method="linear")
            cuts = np.asarray(cuts, dtype=np.float64)
            cuts[0] = min_cut
            cuts[-1] = max_cut

        # Enforce non-decreasing order after boundary anchoring.
        cuts = np.maximum.accumulate(cuts)
        unique_cuts = np.unique(cuts)

        # If quantiles still collapse, fall back to a stable fixed-size grid.
        if unique_cuts.size < requested_cuts:
            unique_cuts = np.linspace(min_cut, max_cut, num=requested_cuts, dtype=np.float64)
        return unique_cuts.astype(self._dtype, copy=False)

    def fit(self, durations, events):
        fit_durations = durations
        fit_events = events
        full_max: float | None = None
        if getattr(self, "_scheme", None) == "quantiles":
            fit_durations, fit_events, full_max = self._fit_inputs_for_quantiles(
                durations,
                events,
            )

            if self._predefined_cuts:
                super().fit(fit_durations, fit_events)
                self._ensure_right_boundary(full_max)
                return self

            current_dtype = getattr(self, "_dtype", None)
            if current_dtype is None:
                if isinstance(fit_durations, np.ndarray):
                    self._dtype = fit_durations.dtype
                else:
                    self._dtype = type(fit_durations[0])
            if self._dtype is float:
                self._dtype = "float64"

            try:
                self.cuts = self._fit_duration_quantile_cuts(fit_durations, full_max)
                self.idu = IdxDiscUnknownC(self.cuts)
                self._ensure_right_boundary(full_max)
                return self
            except Exception:
                # If robust quantile fitting fails, keep notebook execution stable.
                original_scheme = self._scheme
                self._scheme = "equidistant"
                try:
                    super().fit(fit_durations, fit_events)
                    self._ensure_right_boundary(full_max)
                    return self
                finally:
                    self._scheme = original_scheme

        super().fit(fit_durations, fit_events)
        self._ensure_right_boundary(full_max)
        return self

    def transform(self, durations, events):
        event_mask = np.asarray(events) > 0
        durations_arr = np.asarray(durations)

        if event_mask.size == 0 or not np.any(event_mask):
            durations_c = durations_arr.copy()
            if durations_c.size > 0:
                max_cut = float(np.max(self.cuts))
                durations_c[durations_c > max_cut] = max_cut
                time_disc = discretize(
                    durations_c,
                    self.cuts,
                    side=self.idu.duc.censor_side,
                    error_on_larger=True,
                )
                idx_durations = self.idu.di.transform(time_disc)
            else:
                idx_durations = np.empty(0, dtype=np.int64)
            is_event = np.zeros_like(event_mask, dtype=np.int64)
        else:
            idx_durations, is_event = super().transform(durations_arr, event_mask)

        events_out = np.asarray(events).astype("int64")
        events_out[np.asarray(is_event) == 0] = 0
        return idx_durations, events_out


def make_tensors(X_np, Y_disc_np, D_disc_np):
    X = torch.tensor(X_np, dtype=torch.float32, device="cpu")
    Y = torch.tensor(Y_disc_np, dtype=torch.int64, device="cpu")
    D = torch.tensor(D_disc_np, dtype=torch.int64, device="cpu")
    return X, Y, D