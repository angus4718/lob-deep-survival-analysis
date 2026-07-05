"""Small console logging and optional tqdm helpers for backtests."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TypeVar

T = TypeVar("T")


def log(enabled: bool, message: str) -> None:
    """Print a backtest log line when verbose logging is enabled."""
    if enabled:
        print(f"[backtest] {message}", flush=True)


def maybe_tqdm(iterable: Iterable[T], *, enabled: bool, **kwargs) -> Iterable[T]:
    """Wrap an iterable in tqdm when requested and available."""
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm
    except Exception:
        return iterable
    return tqdm(iterable, **kwargs)


def set_progress_postfix(progress_iter: Iterable[T], **values) -> None:
    """Update tqdm postfix when the iterable is a tqdm object."""
    setter = getattr(progress_iter, "set_postfix", None)
    if not callable(setter):
        return
    try:
        setter(values, refresh=False)
    except TypeError:
        setter(**values)
