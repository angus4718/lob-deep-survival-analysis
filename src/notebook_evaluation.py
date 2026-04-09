"""Notebook evaluation helpers for standardized DeepHit notebooks."""

from __future__ import annotations

import numpy as np


def standard_brier_score(durations, events, cif_k, event_code, time_grid):
    """Compute standard (non-IPCW) Brier score for one cause over time."""
    T = len(time_grid)
    bs = np.zeros(T, dtype=np.float64)
    for j, t in enumerate(time_grid):
        label = ((durations <= t) & (events == event_code)).astype(float)
        residual = (label - cif_k[j, :]) ** 2
        bs[j] = np.mean(residual)
    return bs


def uninformed_brier_score(durations, events, predicted_prob, event_code, time_grid):
    """Brier score for a constant-probability baseline model."""
    T = len(time_grid)
    bs_uninf = np.zeros(T, dtype=np.float64)
    for j, t in enumerate(time_grid):
        label = ((durations <= t) & (events == event_code)).astype(float)
        residual = (label - predicted_prob) ** 2
        bs_uninf[j] = np.mean(residual)
    return bs_uninf