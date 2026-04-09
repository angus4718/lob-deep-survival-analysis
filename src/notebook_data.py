"""Notebook data-preparation helpers for standardized DeepHit notebooks."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import torch
from pycox.preprocessing.label_transforms import LabTransDiscreteTime


def safe_stack_representation(rep, feat_dim: int) -> np.ndarray:
    if rep is None:
        return np.empty((0, feat_dim), dtype=np.float32)
    rows = [np.asarray(row, dtype=np.float32).reshape(-1) for row in rep]
    if len(rows) == 0:
        return np.empty((0, feat_dim), dtype=np.float32)
    arr = np.stack(rows, axis=0)
    if arr.ndim != 2 or arr.shape[1] != feat_dim:
        raise ValueError(f"Unexpected shape {arr.shape}; expected (*, {feat_dim})")
    return arr


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


def best_day_cut(target_row: int, day_end_idx: Sequence[int]) -> int:
    return min(range(len(day_end_idx)), key=lambda i: abs(day_end_idx[i] - target_row))


class LabTransform(LabTransDiscreteTime):
    def transform(self, durations, events):
        durations, is_event = super().transform(durations, events > 0)
        events = events.astype("int64")
        events[is_event == 0] = 0
        return durations, events


def make_tensors(X_np, Y_disc_np, D_disc_np):
    X = torch.tensor(X_np, dtype=torch.float32, device="cpu")
    Y = torch.tensor(Y_disc_np, dtype=torch.int64, device="cpu")
    D = torch.tensor(D_disc_np, dtype=torch.int64, device="cpu")
    return X, Y, D