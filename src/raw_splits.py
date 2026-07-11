"""Shared raw DBN split helpers for dataset builds and raw backtests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class RawChunk:
    """One timestamp-bounded raw replay chunk."""

    index: int
    start_ns: int | None
    end_ns: int | None
    message_total: int | None = None

    def overlaps(self, start_ns: int, end_ns: int) -> bool:
        chunk_start = self.start_ns if self.start_ns is not None else -float("inf")
        chunk_end = self.end_ns if self.end_ns is not None else float("inf")
        return chunk_start < int(end_ns) and int(start_ns) < chunk_end


def load_or_build_split_cache(
    cache_path: str | Path,
    dbn_file: str | Path,
    *,
    verbose: bool = True,
    analyzer: Callable[[str, bool], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Load cached split metadata or scan the raw file and cache it."""
    cache_path = Path(cache_path)
    dbn_file = Path(dbn_file)
    if cache_path.exists():
        with cache_path.open() as f:
            cached = json.load(f)
        if isinstance(cached, dict) and {
            "split_points",
            "messages_between_splits",
            "total_messages",
        }.issubset(cached.keys()):
            if verbose:
                print(
                    "[cache] Loaded split metadata: "
                    f"{len(cached['split_points'])} split points, "
                    f"{cached['total_messages']:,} total messages"
                )
            return cached
        if verbose:
            print("[cache] Cache format not recognized. Rebuilding split metadata...")

    if verbose:
        print(f"[cache] No valid cache found - scanning {dbn_file} for split metadata...")
    if analyzer is None:
        from src.order_tracking import analyze_empty_market_splits

        analyzer = analyze_empty_market_splits

    analyzed = analyzer(str(dbn_file), verbose)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w") as f:
        json.dump(analyzed, f)
    if verbose:
        print(
            "[cache] Saved split metadata: "
            f"{len(analyzed['split_points'])} split points, "
            f"{analyzed['total_messages']:,} total messages"
        )
    return analyzed


def build_raw_chunks(
    *,
    empty_points: list[int],
    n_workers: int,
    messages_between_splits: list[int] | None = None,
    total_messages: int | None = None,
) -> list[RawChunk]:
    """Select balanced split points and return replay chunk boundaries."""
    split_ts = select_balanced_split_points(
        empty_points=empty_points,
        n_workers=n_workers,
        messages_between_splits=messages_between_splits,
    )
    boundaries: list[int | None] = [None] + split_ts + [None]
    message_totals = _selected_chunk_message_totals(
        empty_points=empty_points,
        split_ts=split_ts,
        messages_between_splits=messages_between_splits,
        total_messages=total_messages,
    )
    return [
        RawChunk(
            index=i,
            start_ns=boundaries[i],
            end_ns=boundaries[i + 1],
            message_total=message_totals[i] if message_totals is not None else None,
        )
        for i in range(len(boundaries) - 1)
    ]


def select_balanced_split_points(
    *,
    empty_points: list[int],
    n_workers: int,
    messages_between_splits: list[int] | None = None,
) -> list[int]:
    """Choose split timestamps, preferring message-balanced chunks."""
    if int(n_workers) <= 1 or not empty_points:
        return []
    if (
        messages_between_splits is not None
        and len(messages_between_splits) == len(empty_points) + 1
    ):
        return _select_split_points_by_message_count(
            empty_points,
            messages_between_splits,
            int(n_workers),
        )
    return _select_split_points(empty_points, int(n_workers))


def filter_chunks_for_order_range(
    chunks: list[RawChunk],
    *,
    order_start_ns: int,
    order_end_ns: int,
) -> list[RawChunk]:
    """Drop raw chunks whose timestamp ranges cannot contain assigned orders."""
    return [
        chunk
        for chunk in chunks
        if chunk.overlaps(int(order_start_ns), int(order_end_ns))
    ]


def _selected_chunk_message_totals(
    *,
    empty_points: list[int],
    split_ts: list[int],
    messages_between_splits: list[int] | None,
    total_messages: int | None,
) -> list[int] | None:
    if (
        messages_between_splits is None
        or len(messages_between_splits) != len(empty_points) + 1
    ):
        return None
    running = 0
    total_before_point: dict[int, int] = {}
    for split_point, seg_count in zip(empty_points, messages_between_splits[:-1]):
        running += int(seg_count)
        total_before_point[int(split_point)] = running

    resolved_total = (
        int(total_messages)
        if total_messages is not None
        else running + int(messages_between_splits[-1])
    )
    per_split = [total_before_point.get(int(ts)) for ts in split_ts]
    if not all(value is not None for value in per_split):
        return None
    cumulative = [0] + [int(value) for value in per_split] + [resolved_total]
    return [
        max(0, int(cumulative[i + 1] - cumulative[i]))
        for i in range(len(cumulative) - 1)
    ]


def _select_split_points(empty_points: list[int], n: int) -> list[int]:
    if n <= 1 or not empty_points:
        return []
    lo, hi = int(empty_points[0]), int(empty_points[-1])
    if lo == hi:
        return []
    targets = [lo + i * (hi - lo) // n for i in range(1, n)]
    chosen: set[int] = set()
    for target in targets:
        chosen.add(min(empty_points, key=lambda t: abs(int(t) - target)))
    return sorted(int(item) for item in chosen)


def _select_split_points_by_message_count(
    empty_points: list[int],
    messages_between_splits: list[int],
    n: int,
) -> list[int]:
    if n <= 1 or not empty_points:
        return []
    if len(messages_between_splits) != len(empty_points) + 1:
        return _select_split_points(empty_points, n)

    cumulative = [0]
    for i in range(len(empty_points)):
        cumulative.append(cumulative[-1] + int(messages_between_splits[i]))
    total_messages = cumulative[-1] + int(messages_between_splits[-1])
    target_cumulative = [i * total_messages // n for i in range(1, n)]

    chosen: set[int] = set()
    for target in target_cumulative:
        best_idx = min(
            range(len(empty_points)),
            key=lambda i: abs(cumulative[i + 1] - target),
        )
        chosen.add(int(empty_points[best_idx]))
    return sorted(chosen)
