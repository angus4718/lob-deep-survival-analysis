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


def uninformed_cif_curve_from_train(durations_train, events_train, event_code, time_grid):
    """Time-varying uninformed CIF estimate from the training split.

    For each t in time_grid, returns P(T <= t, event=event_code) estimated as
    the empirical training proportion.
    """
    durations_train = np.asarray(durations_train)
    events_train = np.asarray(events_train)
    if durations_train.shape[0] != events_train.shape[0]:
        raise ValueError("durations_train and events_train must have the same length")

    probs = np.zeros(len(time_grid), dtype=np.float64)
    for j, t in enumerate(time_grid):
        probs[j] = np.mean((durations_train <= t) & (events_train == event_code))
    return probs


def uninformed_brier_score(durations, events, predicted_prob, event_code, time_grid):
    """Brier score for a constant or time-varying uninformed baseline model.

    predicted_prob can be either:
    - scalar: constant baseline probability for all times
    - array-like of shape (len(time_grid),): time-varying baseline probability
    """
    T = len(time_grid)
    prob_arr = np.asarray(predicted_prob, dtype=np.float64)
    if prob_arr.ndim == 0:
        prob_arr = np.full(T, float(prob_arr), dtype=np.float64)
    else:
        prob_arr = prob_arr.reshape(-1)
        if prob_arr.shape[0] != T:
            raise ValueError("predicted_prob must be scalar or have length len(time_grid)")

    bs_uninf = np.zeros(T, dtype=np.float64)
    for j, t in enumerate(time_grid):
        label = ((durations <= t) & (events == event_code)).astype(float)
        residual = (label - prob_arr[j]) ** 2
        bs_uninf[j] = np.mean(residual)
    return bs_uninf