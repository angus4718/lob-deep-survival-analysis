from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

try:
    from .common import EVENT_CENSORED, detect_post_trade_windows_ms, ns_to_session_date, safe_rate
except ImportError:
    from common import EVENT_CENSORED, detect_post_trade_windows_ms, ns_to_session_date, safe_rate


LOB_FEATURE_DIM = 20
TOXICITY_RAW_DIM = 12


@dataclass
class SplitInfo:
    train_mask: np.ndarray
    val_mask: np.ndarray
    test_mask: np.ndarray
    train_end: int
    val_end: int
    n_days: int
    train_days: list[str]
    val_days: list[str]
    test_days: list[str]


@dataclass
class PreprocessingStats:
    feat_mean: np.ndarray
    feat_std: np.ndarray
    mask_col_idx: int


@dataclass
class DatasetBundle:
    dataset_path: Path
    full_frame: pd.DataFrame
    eval_frame: pd.DataFrame
    x_eval: np.ndarray
    y_eval: np.ndarray
    d_eval: np.ndarray
    split_name: str
    split_info: SplitInfo
    preprocessing: PreprocessingStats
    training_priors: dict[str, float]
    available_post_trade_windows_ms: list[int]

    @property
    def feature_dim(self) -> int:
        return int(self.x_eval.shape[2])


def load_standardized_static_dataset(
    dataset_path: Path,
    lookback_steps: int,
    max_time_s: float,
    split_name: str = "test",
) -> DatasetBundle:
    dataset_path = dataset_path.resolve()
    df_raw = _load_dataset_frame(dataset_path)
    df = _prepare_frame(df_raw, max_time_s=max_time_s)

    x_all = _build_feature_tensor(df, lookback_steps=lookback_steps)
    y_all = df["duration_s"].to_numpy(dtype=np.float32)
    d_all = df["event_type_competing"].to_numpy(dtype=np.int64)

    split_info = compute_temporal_split(df)
    preprocessing = fit_standardization(x_all[split_info.train_mask])
    x_all_std = apply_standardization(x_all, preprocessing)

    train_events = d_all[split_info.train_mask]
    training_priors = {
        "p_favorable_fill": safe_rate(float((train_events == 1).sum()), float(len(train_events))),
        "p_toxic_fill": safe_rate(float((train_events == 2).sum()), float(len(train_events))),
        "p_any_fill": safe_rate(float(np.isin(train_events, [1, 2]).sum()), float(len(train_events))),
    }

    eval_mask = _select_split_mask(split_info, split_name)
    eval_frame = df.loc[eval_mask].reset_index(drop=True)
    x_eval = x_all_std[eval_mask]
    y_eval = y_all[eval_mask]
    d_eval = d_all[eval_mask]

    available_windows = sorted(detect_post_trade_windows_ms(list(df.columns)).keys())

    return DatasetBundle(
        dataset_path=dataset_path,
        full_frame=df,
        eval_frame=eval_frame,
        x_eval=x_eval,
        y_eval=y_eval,
        d_eval=d_eval,
        split_name=split_name,
        split_info=split_info,
        preprocessing=preprocessing,
        training_priors=training_priors,
        available_post_trade_windows_ms=available_windows,
    )


def compute_temporal_split(df: pd.DataFrame) -> SplitInfo:
    entry_ns = df["entry_time"].to_numpy()
    dates = ns_to_session_date(entry_ns)
    unique_days = sorted(dates.unique())
    n_days = len(unique_days)
    n = len(df)
    if n_days < 3:
        raise ValueError("Need at least 3 trading days to build train/val/test splits.")

    target_train_end = int(n * 0.70)
    target_val_end = int(n * 0.85)
    day_end_idx = [(dates <= d).sum() - 1 for d in unique_days]

    def best_day_cut(target_row: int) -> int:
        return min(range(len(day_end_idx)), key=lambda i: abs(day_end_idx[i] - target_row))

    train_day_idx = best_day_cut(target_train_end)
    val_day_idx = best_day_cut(target_val_end)
    train_day_idx = min(train_day_idx, n_days - 3)
    val_day_idx = max(train_day_idx + 1, min(val_day_idx, n_days - 2))

    train_end = day_end_idx[train_day_idx] + 1
    val_end = day_end_idx[val_day_idx] + 1

    idx = np.arange(n)
    train_mask = idx < train_end
    val_mask = (idx >= train_end) & (idx < val_end)
    test_mask = idx >= val_end

    return SplitInfo(
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        train_end=train_end,
        val_end=val_end,
        n_days=n_days,
        train_days=[str(day.date()) for day in unique_days[: train_day_idx + 1]],
        val_days=[str(day.date()) for day in unique_days[train_day_idx + 1 : val_day_idx + 1]],
        test_days=[str(day.date()) for day in unique_days[val_day_idx + 1 :]],
    )


def fit_standardization(x_train: np.ndarray) -> PreprocessingStats:
    feat_mean = x_train.mean(axis=(0, 1), keepdims=True)
    feat_std = x_train.std(axis=(0, 1), keepdims=True) + 1e-8
    mask_col_idx = x_train.shape[2] - 1
    feat_mean[..., mask_col_idx] = 0.0
    feat_std[..., mask_col_idx] = 1.0
    return PreprocessingStats(
        feat_mean=feat_mean.astype(np.float32),
        feat_std=feat_std.astype(np.float32),
        mask_col_idx=int(mask_col_idx),
    )


def apply_standardization(x_np: np.ndarray, preprocessing: PreprocessingStats) -> np.ndarray:
    x_std = ((x_np - preprocessing.feat_mean) / preprocessing.feat_std).astype(np.float32)
    mask_idx = preprocessing.mask_col_idx
    x_std[..., mask_idx] = x_np[..., mask_idx]
    return x_std


def _select_split_mask(split_info: SplitInfo, split_name: str) -> np.ndarray:
    split_name = split_name.lower()
    if split_name == "train":
        return split_info.train_mask
    if split_name == "val":
        return split_info.val_mask
    if split_name == "test":
        return split_info.test_mask
    if split_name == "all":
        return np.ones_like(split_info.train_mask, dtype=bool)
    raise ValueError(f"Unsupported split_name {split_name!r}. Use train/val/test/all.")


def _load_dataset_frame(dataset_path: Path) -> pd.DataFrame:
    query = f"SELECT * FROM read_parquet('{dataset_path.as_posix()}')"
    df = duckdb.query(query).df()
    if "entry_time" not in df.columns:
        raise ValueError(f"{dataset_path} does not look like a labeled dataset: missing entry_time.")
    return df.sort_values("entry_time").reset_index(drop=True)


def _prepare_frame(df_raw: pd.DataFrame, max_time_s: float) -> pd.DataFrame:
    df = df_raw.copy()
    if "entry_representation_raw_top5" not in df.columns:
        raise ValueError("Dataset is missing entry_representation_raw_top5.")
    if "toxicity_representation" not in df.columns:
        raise ValueError("Dataset is missing toxicity_representation.")
    if "event_type" not in df.columns:
        raise ValueError("Dataset is missing event_type.")
    if "duration_s" not in df.columns:
        raise ValueError("Dataset is missing duration_s.")

    df["entry_representation"] = df["entry_representation_raw_top5"]
    df["event_type_competing"] = df["event_type"].astype("int64")

    late_uncensored_mask = df["duration_s"] > max_time_s
    if late_uncensored_mask.any():
        df.loc[late_uncensored_mask, "event_type_competing"] = EVENT_CENSORED
        df.loc[late_uncensored_mask, "duration_s"] = max_time_s

    df["entry_date"] = ns_to_session_date(df["entry_time"])
    return df


def _build_feature_tensor(df: pd.DataFrame, lookback_steps: int) -> np.ndarray:
    x_lob = _extract_lob_features(df, lookback_steps=lookback_steps)
    x_tox = _extract_toxicity_features(df, lookback_steps=lookback_steps)
    return np.concatenate([x_lob, x_tox], axis=2)


def _extract_lob_features(df: pd.DataFrame, lookback_steps: int) -> np.ndarray:
    rows = []
    for rep in df["entry_representation"]:
        arr_raw = _safe_stack_representation(rep, feat_dim=LOB_FEATURE_DIM)
        arr, _ = _left_pad_or_truncate(arr_raw, lookback_steps)
        rows.append(arr)
    return np.stack(rows, axis=0)


def _extract_toxicity_features(df: pd.DataFrame, lookback_steps: int) -> np.ndarray:
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
        arr_raw = _safe_stack_representation(rep, feat_dim=TOXICITY_RAW_DIM)
        arr, valid_len = _left_pad_or_truncate(arr_raw, lookback_steps)

        mask_col = np.zeros((lookback_steps, 1), dtype=np.float32)
        if valid_len > 0:
            mask_col[-valid_len:, 0] = 1.0

        side_col = np.full((lookback_steps, 1), side_val, dtype=np.float32) * mask_col
        rows.append(np.concatenate([arr, side_col, mask_col], axis=1))

    return np.stack(rows, axis=0)


def _safe_stack_representation(rep: Any, feat_dim: int) -> np.ndarray:
    if rep is None:
        return np.empty((0, feat_dim), dtype=np.float32)
    rows = [np.asarray(row, dtype=np.float32).reshape(-1) for row in rep]
    if len(rows) == 0:
        return np.empty((0, feat_dim), dtype=np.float32)
    arr = np.stack(rows, axis=0)
    if arr.ndim != 2 or arr.shape[1] != feat_dim:
        raise ValueError(f"Unexpected representation shape {arr.shape}; expected (*, {feat_dim}).")
    return arr


def _left_pad_or_truncate(arr: np.ndarray, lookback_steps: int) -> tuple[np.ndarray, int]:
    if arr.shape[0] >= lookback_steps:
        return arr[-lookback_steps:], lookback_steps
    valid_len = arr.shape[0]
    pad_len = lookback_steps - valid_len
    pad = np.zeros((pad_len, arr.shape[1]), dtype=np.float32)
    return np.concatenate([pad, arr], axis=0), valid_len
