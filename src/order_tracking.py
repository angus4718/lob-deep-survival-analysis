"""
Order tracking and sampling utilities for LOB survival analysis.

This module implements sampling and tracking of virtual (simulated)
orders from a message-by-message exchange data stream. It provides
low-level helpers and high-level `OrderTracker.process_stream` to iterate
over a DBN file and produce Parquet output with tracked order lifetimes.

The main classes are:
- `OrderTracker`: orchestrates sampling, spawning, and updating tracked
    virtual orders while streaming market-by-order (MBO) messages.
- `VirtualOrder`: representation of a simulated tracked order with
    update logic for queue-depletion and censoring.
"""

import collections
import hashlib
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from uuid import uuid4
import pandas as pd
import databento as db
from dataclasses import dataclass, field
from typing import Callable, Any, List, Dict, Optional, Tuple
import random

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from .lob_implementation import Market, Book

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
from .config import CONFIG
from .labeling.base import BaseLabeler
from .labeling.competing_risks import ExecutionCompetingRisksLabeler
from .labeling.utils import ms_to_suffix
from .features.base import BaseLOBTransform
from .features.representation import RepresentationTransform
from .features.compose import ToxicityFeatures, ComposeTransforms


def _prepare_output_path(path: str) -> str:
    """Return a normalized absolute output path and ensure its parent exists."""
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return str(output_path.resolve())


def _market_is_empty(market: Market) -> bool:
    """Return True iff every book in the market has no resting orders."""
    for pub_books in market.books.values():
        for book in pub_books.values():
            if book.bids or book.offers or book.orders_by_id:
                return False
    return True


def find_empty_market_points(file_path: str, verbose: bool = True) -> list[int]:
    """
    Scan a DBN file in a single pass and return a sorted list of
    ``ts_event`` values (nanoseconds) at which the aggregated order book
    transitions from *non-empty* to completely *empty*.

    These timestamps are safe split-points for parallel processing because
    each chunk can start with a clean (empty) book state.

    Args:
        file_path: Path to the DBN or DBN.ZST file.
        verbose: Print progress dots to stdout.

    Returns:
        Sorted list of nanosecond timestamps.
    """
    data = db.DBNStore.from_file(file_path)
    market = Market()
    had_orders = False
    empty_timestamps: List[int] = []
    count = 0

    for mbo in data:
        if not isinstance(mbo, db.MBOMsg):
            continue
        count += 1
        if verbose and count % 1_000_000 == 0:
            print(
                f"  [scan] {count:,} messages scanned, "
                f"{len(empty_timestamps)} split points found so far"
            )
        try:
            market.apply(mbo)
        except (KeyError, AssertionError):
            continue
        is_empty = _market_is_empty(market)
        if not had_orders and not is_empty:
            had_orders = True
        if had_orders and is_empty:
            empty_timestamps.append(mbo.ts_event)
            had_orders = False  # reset: next non-empty to empty is a new event

    if verbose:
        print(
            f"  [scan] Finished. {count:,} messages, "
            f"{len(empty_timestamps)} empty-market transitions found."
        )
    return empty_timestamps


def count_messages_between_split_points(
    file_path: str,
    split_points: list[int],
    verbose: bool = True,
) -> tuple[list[int], int]:
    """
    Count messages in each segment induced by ``split_points``.

    Segment definition matches chunk guards used by ``process_stream``:
    - Segment 0: ``ts_event < split_points[0]``
    - Segment i: ``split_points[i-1] <= ts_event < split_points[i]``
    - Last segment: ``ts_event >= split_points[-1]``

    Args:
        file_path: Path to DBN or DBN.ZST file.
        split_points: Sorted split timestamps (ns).
        verbose: Print progress while counting.

    Returns:
        Tuple of ``(messages_between_splits, total_messages)`` where
        ``messages_between_splits`` has length ``len(split_points) + 1``.
    """
    data = db.DBNStore.from_file(file_path)
    counts = [0] * (len(split_points) + 1)
    split_idx = 0
    total_messages = 0

    for record in data:
        total_messages += 1
        ts_event = getattr(record, "ts_event", None)
        if ts_event is None:
            counts[-1] += 1
            continue

        while split_idx < len(split_points) and ts_event >= split_points[split_idx]:
            split_idx += 1
        counts[split_idx] += 1

        if verbose and total_messages % 1_000_000 == 0:
            print(f"  [count] {total_messages:,} messages counted")

    if verbose:
        print(
            f"  [count] Finished. {total_messages:,} messages across "
            f"{len(counts)} segment(s)."
        )

    return counts, total_messages


def analyze_empty_market_splits(file_path: str, verbose: bool = True) -> dict:
    """
    Compute split metadata for progress accounting and cache it.

    Returns a dict containing:
    - ``split_points``: empty-market split timestamps.
    - ``messages_between_splits``: exact message counts per split segment.
    - ``total_messages``: total message count in the file.
    """
    split_points = find_empty_market_points(file_path=file_path, verbose=verbose)
    segment_counts, total_messages = count_messages_between_split_points(
        file_path=file_path,
        split_points=split_points,
        verbose=verbose,
    )
    return {
        "split_points": split_points,
        "messages_between_splits": segment_counts,
        "total_messages": total_messages,
    }


def _select_split_points(empty_points: list[int], n: int) -> list[int]:
    """
    Choose ``n - 1`` timestamps from *empty_points* that divide the
    timeline ``[empty_points[0], empty_points[-1]]`` into *n* roughly
    equal segments.

    Args:
        empty_points: Sorted list of empty-market timestamps (ns).
        n: Desired number of chunks.

    Returns:
        Sorted list of ``n - 1`` split timestamps (may be fewer if there
        are not enough distinct empty points).
    """
    if n <= 1 or not empty_points:
        return []
    lo, hi = empty_points[0], empty_points[-1]
    if lo == hi:
        return []
    targets = [lo + i * (hi - lo) // n for i in range(1, n)]
    chosen: set = set()
    for target in targets:
        best = min(empty_points, key=lambda t: abs(t - target))
        chosen.add(best)
    return sorted(chosen)


def _select_split_points_by_message_count(
    empty_points: list[int],
    messages_between_splits: list[int],
    n: int,
) -> list[int]:
    """
    Choose ``n - 1`` timestamps from *empty_points* that divide the
    data into *n* chunks with roughly equal message counts, while ensuring
    each chunk starts at an empty-market point.

    Args:
        empty_points: Sorted list of empty-market timestamps (ns).
        messages_between_splits: Message counts for each segment between
            consecutive empty points. Must have length len(empty_points) + 1.
            Segment i contains messages between empty_points[i-1] and
            empty_points[i] (or file boundaries).
        n: Desired number of chunks.

    Returns:
        Sorted list of ``n - 1`` split timestamps that start empty-market
        transitions and balance message counts across chunks.
    """
    if n <= 1 or not empty_points:
        return []
    if len(messages_between_splits) != len(empty_points) + 1:
        # Fallback to time-based split if message counts don't match
        return _select_split_points(empty_points, n)

    # Build cumulative message counts at each empty point
    # cumulative[i] = total messages from file start up to (but not including) empty_points[i]
    cumulative = [0]
    for i in range(len(empty_points)):
        cumulative.append(cumulative[-1] + messages_between_splits[i])
    total_messages = cumulative[-1] + messages_between_splits[-1]

    # Target cumulative message count for each chunk boundary
    target_cumulative = [i * total_messages // n for i in range(1, n)]

    # For each target, find the closest empty point
    chosen: set = set()
    for target in target_cumulative:
        # Find the empty point with cumulative closest to target
        best_idx = min(
            range(len(empty_points)),
            key=lambda i: abs(cumulative[i + 1] - target),
        )
        chosen.add(empty_points[best_idx])

    return sorted(chosen)


def _chunk_worker(kwargs: dict) -> dict:
    """
    Module-level worker function executed by ``ProcessPoolExecutor``.

    Instantiates a fresh :class:`OrderTracker`, processes one time-slice
    of the DBN file, and writes results to a temporary Parquet file.

    Args:
        kwargs: Dict produced by :meth:`OrderTracker.process_stream_parallel`.

    Returns:
        Dict with keys ``output_parquet``, ``scheduled_virtual``,
        ``spawned_virtual``.
    """
    import sys

    project_root = kwargs.pop("_project_root", None)
    if project_root and project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Re-import after path fix (needed in spawned worker processes)
    from src.order_tracking import OrderTracker

    tracker_kwargs = kwargs.pop("tracker_kwargs")
    file_path = kwargs.pop("file_path")
    output_parquet = _prepare_output_path(kwargs.pop("output_parquet"))
    chunk_ts_start: Optional[int] = kwargs.pop("chunk_ts_start", None)
    chunk_ts_end: Optional[int] = kwargs.pop("chunk_ts_end", None)
    chunk_idx: int = kwargs.pop("chunk_idx", 0)

    # Remaining keys forwarded to process_stream
    process_kwargs = kwargs

    tracker = OrderTracker(**tracker_kwargs)
    tracker.process_stream(
        file_path=file_path,
        output_parquet=output_parquet,
        chunk_ts_start=chunk_ts_start,
        chunk_ts_end=chunk_ts_end,
        **process_kwargs,
    )
    return {
        "chunk_idx": chunk_idx,
        "output_parquet": output_parquet,
        "scheduled_virtual": tracker.inst.scheduled_virtual,
        "spawned_virtual": tracker.inst.spawned_virtual,
    }


@dataclass(slots=True)
class Instrumentation:
    """
    Counters for sampling instrumentation.

    Counters are incremented during schedule creation and when samples are
    spawned to help with run diagnostics.
    """

    scheduled_virtual: int = 0
    spawned_virtual: int = 0


@dataclass(slots=True)
class TrackedOrder:
    """
    Base class for tracked orders.

    Attributes:
        internal_id: Internal integer id assigned by the tracker.
        entry_time: Entry timestamp (ns) for the tracked order.
        price: Limit price for the tracked order.
        side: Side ('B' or 'A').
        status: Current lifecycle status string.
        end_time: Timestamp (ns) when order ended (filled/censored).
        order_type: Human-readable type label.
    """

    internal_id: int
    entry_time: int
    price: int
    side: str
    status: str = "ACTIVE"
    end_time: int = 0
    order_type: str = "UNKNOWN"

    def is_active(self) -> bool:
        """Return True if order is still active (not filled/censored)."""
        return self.status == "ACTIVE"

    def on_fill(self, mbo: db.MBOMsg):
        """Mark the order as filled using the given MBO message timestamp."""
        self.status = "FILLED"
        self.end_time = mbo.ts_event

    def on_censor(self, reason: str, mbo: db.MBOMsg | None = None):
        """Mark the order as censored for the given reason."""
        self.status = reason
        self.end_time = mbo.ts_event if mbo is not None else self.end_time

    def update(self, mbo: db.MBOMsg, book: Book, market: Market):
        """Update order state in response to an MBO message.

        Subclasses must implement this method.
        """
        raise NotImplementedError()

    def to_dict(self) -> dict:
        """Return a serializable dict representing the tracked order record."""
        duration_ns = self.end_time - self.entry_time
        event = 1 if self.status == "FILLED" else 0
        side_val = getattr(self.side, "value", self.side)
        if isinstance(side_val, bytes):
            try:
                side_val = side_val.decode()
            except Exception:
                side_val = str(side_val)

        vol = getattr(self, "initial_size", None)
        if vol is None:
            vol = getattr(self, "remaining_size", None)
        if vol is None:
            try:
                if isinstance(self, VirtualOrder):
                    vol = 1
                else:
                    vol = 0
            except Exception:
                vol = 0

        base = {
            "order_id": self.internal_id,
            "entry_time": self.entry_time,
            "duration_s": duration_ns / 1e9,
            "event": event,
            "status_reason": self.status,
            "price": self.price,
            "side": side_val,
            "volume": vol,
            "order_type": self.order_type,
        }

        if isinstance(self, VirtualOrder):
            base["best_bid_at_entry"] = getattr(self, "best_bid_at_entry", None)
            base["best_ask_at_entry"] = getattr(self, "best_ask_at_entry", None)
            base["best_bid_at_post_trade"] = getattr(
                self, "best_bid_at_post_trade", None
            )
            base["best_ask_at_post_trade"] = getattr(
                self, "best_ask_at_post_trade", None
            )

            # Raw data mode fields - post-trade BBO at each window
            post_trade_windows = getattr(self, "post_trade_windows_ms", [])
            raw_post_bbo = getattr(self, "raw_post_trade_bbo_list", None)

            if post_trade_windows and raw_post_bbo:
                for window_ms, (bid, ask) in zip(post_trade_windows, raw_post_bbo):
                    suffix = ms_to_suffix(window_ms)
                    base[f"post_trade_best_bid_{suffix}"] = bid
                    base[f"post_trade_best_ask_{suffix}"] = ask

            # Four LOB representation modes
            base["entry_representation_market_depth"] = getattr(
                self, "entry_representation_market_depth", None
            )
            base["entry_representation_moving_window"] = getattr(
                self, "entry_representation_moving_window", None
            )
            base["entry_representation_raw_top5"] = getattr(
                self, "entry_representation_raw_top5", None
            )
            base["entry_representation_diff_top5"] = getattr(
                self, "entry_representation_diff_top5", None
            )

            # Toxicity representation (shared across all modes)
            base["toxicity_representation"] = getattr(
                self, "toxicity_representation", None
            )

            # LOB sequences for all four modes
            base["lob_sequence_market_depth"] = (
                getattr(self, "lob_sequence_market_depth", None) or None
            )
            base["lob_sequence_moving_window"] = (
                getattr(self, "lob_sequence_moving_window", None) or None
            )
            base["lob_sequence_raw_top5"] = (
                getattr(self, "lob_sequence_raw_top5", None) or None
            )
            base["lob_sequence_diff_top5"] = (
                getattr(self, "lob_sequence_diff_top5", None) or None
            )

            # Toxicity sequence
            base["toxicity_sequence"] = getattr(self, "toxicity_sequence", None) or None

            # Legacy fields (kept for backward compatibility)
            base["entry_representation"] = getattr(
                self, "entry_representation_moving_window", None
            )
            base["lob_sequence"] = (
                getattr(self, "lob_sequence_moving_window", None) or None
            )
            base["sequence_length"] = len(
                getattr(self, "lob_sequence_moving_window", []) or []
            )

        return base


@dataclass(slots=True)
class VirtualOrder(TrackedOrder):
    """
    Virtual (synthetic) order used for survival analysis sampling.

    Virtual orders approximate the lifecycle of a limit order placed at the
    top-of-book at a given timestamp. They track volume ahead and ids for
    basic queue-depletion logic.
    """

    current_vahead: int = 0
    ids_ahead: dict[int, int] = field(default_factory=dict)
    best_bid_at_entry: int | None = None
    best_ask_at_entry: int | None = None
    best_bid_at_post_trade: int | None = None
    best_ask_at_post_trade: int | None = None
    post_trade_deadline_ts: int | None = None

    # Multiple post-trade capture windows (ms → ns conversion happens in OrderTracker)
    post_trade_windows_ms: List[int] = field(default_factory=list)  # List of ms windows
    post_trade_bbo_dict: Dict[int, tuple] = field(
        default_factory=dict
    )  # {ms: (bid, ask)}

    # Market-depth representation (41 dims)
    entry_representation_market_depth: list | None = None  # (T_hist, 41)
    lob_sequence_market_depth: list[list[float]] = field(
        default_factory=list
    )  # (T_var, 41)

    # Moving-window representation (41 dims)
    entry_representation_moving_window: list | None = None  # (T_hist, 41)
    lob_sequence_moving_window: list[list[float]] = field(
        default_factory=list
    )  # (T_var, 41)

    # Raw top-5 representation (20 dims: 5 bid levels * 2 + 5 ask levels * 2)
    entry_representation_raw_top5: list | None = None  # (T_hist, 20)
    lob_sequence_raw_top5: list[list[float]] = field(
        default_factory=list
    )  # (T_var, 20)

    # Proportional difference top-5 representation (20 dims: 5 bid levels * 2 + 5 ask levels * 2)
    entry_representation_diff_top5: list | None = None  # (T_hist, 20)
    lob_sequence_diff_top5: list[list[float]] = field(
        default_factory=list
    )  # (T_var, 20)

    # Toxicity representation (consistent across all modes)
    toxicity_representation: list | None = None  # (T_hist, toxicity_dim), no padding
    lob_sequence: list[list[float]] = field(
        default_factory=list
    )  # (T_var, 2W+1) - deprecated, kept for backward compat
    toxicity_sequence: list[list[float]] = field(
        default_factory=list
    )  # (T_var, toxicity_dim)

    raw_post_trade_bbo_list: List[tuple] = field(
        default_factory=list
    )  # List of (bid, ask) for each window

    def __post_init__(self):
        self.order_type = "VIRTUAL"

    def update(self, mbo: db.MBOMsg, book: Book, market: Market):
        if not self.is_active():
            return

        # Fill when no volume is ahead
        if self.current_vahead <= 0:
            self.on_fill(mbo)
            # Initialize post_trade_windows_ms and setup tracking for multiple time windows
            self.post_trade_windows_ms = list(
                CONFIG.labeling.tox_post_trade_move_windows_ms
            )
            self.post_trade_deadline_ts = mbo.ts_event + int(
                max(self.post_trade_windows_ms) * 1e6
            )
            return

        # Adjust volume ahead in response to cancels/fills/modifies
        if mbo.order_id in self.ids_ahead:
            if mbo.action == "C":
                loss = min(mbo.size, self.ids_ahead[mbo.order_id])
                self.current_vahead -= loss
                self.ids_ahead[mbo.order_id] -= loss
                if self.ids_ahead[mbo.order_id] <= 0:
                    del self.ids_ahead[mbo.order_id]

            elif mbo.action == "F":
                loss = min(mbo.size, self.ids_ahead[mbo.order_id])
                self.current_vahead -= loss
                self.ids_ahead[mbo.order_id] -= loss
                if self.ids_ahead[mbo.order_id] <= 0:
                    del self.ids_ahead[mbo.order_id]

            elif mbo.action == "M":
                old_size = self.ids_ahead[mbo.order_id]
                if mbo.size > old_size:
                    self.current_vahead -= old_size
                    del self.ids_ahead[mbo.order_id]
                elif mbo.size < old_size:
                    diff = old_size - mbo.size
                    self.current_vahead -= diff
                    self.ids_ahead[mbo.order_id] = mbo.size

    def record_post_trade_context(self, book: Book, elapsed_ms: Optional[int] = None):
        """Record post-trade BBO at the current time.

        Args:
            book: The order book to extract BBO from.
            elapsed_ms: Optional elapsed milliseconds since fill. If provided,
                stores BBO in post_trade_bbo_dict for that window.
        """
        b, a = book.bbo()
        bid_price = b.price if b else None
        ask_price = a.price if a else None

        # update the legacy single post-trade fields (for backward compat)
        if (
            elapsed_ms is None
            or len(self.post_trade_windows_ms) <= 4
            or elapsed_ms < self.post_trade_windows_ms[4]
        ):
            self.best_bid_at_post_trade = bid_price
            self.best_ask_at_post_trade = ask_price

        # Store in the dict for this specific window if elapsed_ms is provided
        if elapsed_ms is not None:
            self.post_trade_bbo_dict[elapsed_ms] = (bid_price, ask_price)
            self.raw_post_trade_bbo_list.append((bid_price, ask_price))


@dataclass
class OrderTracker:
    """
    Orchestrates sampling and tracking of virtual (simulated) orders.

    The tracker maintains in-memory lists of active virtual orders,
    sampling schedules per day, and writes finished records to a Parquet
    file via a streaming buffer.

    Args:
        samples_per_day: Target number of samples per trading day.
        time_censor_s: Optional maximum lifetime in seconds before an active
            order is censored as CENSORED_TIME. If None, no time censor is
            applied.
        lookback_period: Number of historical snapshots to maintain in buffer.
        snapshot_bin_messages: Number of messages between snapshots for LOB
            representation sampling. Default is 10 messages per snapshot
            (was 100 ms time-based in prior versions).
        representation_modes: List of LOB representation modes to include.
            Valid options: "market_depth", "moving_window", "raw_top5".
            If None, defaults to all three modes for backward compatibility.
    """

    def __init__(
        self,
        samples_per_day: int = 100,
        time_censor_s: float | None = None,
        random_seed: int | None = CONFIG.random_seed,
        labeler: BaseLabeler | None = None,
        representation_transform: BaseLOBTransform | None = None,
        include_representation: bool = True,
        lookback_period: int = 10,
        snapshot_bin_messages: int = 10,
        representation_modes: list[str] | None = None,
        raw_data_mode: bool = False,
    ):
        self.market = Market()
        self.active_virtual: list[VirtualOrder] = []
        self.post_trade_virtual: list[VirtualOrder] = []
        self.completed: list[TrackedOrder] = []

        self.last_sample_time = 0
        self.virtual_oid_counter = 0

        self.time_censor_ns: int | None = (
            int(time_censor_s * 1e9) if time_censor_s is not None else None
        )
        self.random_seed = random_seed

        self.samples_per_day = samples_per_day
        self.virtual_sample_schedule: dict[int, list[int]] = {}
        self.virtual_sample_index: dict[int, int] = {}
        self.pending_virtual: dict[int, list[int]] = {}

        self.raw_data_mode = raw_data_mode
        self.labeler = (
            None if raw_data_mode else (labeler or ExecutionCompetingRisksLabeler())
        )

        # Set default representation modes if not specified (backward compatible)
        if representation_modes is None:
            representation_modes = ["market_depth", "moving_window", "raw_top5"]
        self.representation_modes = set(representation_modes)

        # Create three separate representation transforms for all modes
        self.representation_transform = (
            representation_transform or RepresentationTransform()
        )
        self.representation_transform_market_depth = (
            RepresentationTransform(representation="market_depth")
            if "market_depth" in self.representation_modes
            else None
        )
        self.representation_transform_moving_window = (
            RepresentationTransform(representation="moving_window")
            if "moving_window" in self.representation_modes
            else None
        )
        self.representation_transform_raw_top5 = (
            RepresentationTransform(representation="raw_top5")
            if "raw_top5" in self.representation_modes
            else None
        )
        self.representation_transform_diff_top5 = (
            RepresentationTransform(representation="diff_top5")
            if "diff_top5" in self.representation_modes
            else None
        )

        self.toxicity_features = ToxicityFeatures()
        self.include_representation = include_representation

        self.lookback_period = lookback_period
        self.snapshot_bin_messages = snapshot_bin_messages
        self._message_count_since_snapshot: int = 0
        self._lob_snapshot_buffer: collections.deque = collections.deque(
            maxlen=lookback_period
        )
        self._last_snapshot_ts: int = 0

        self.inst = Instrumentation()

        # Store primitive constructor args so workers can recreate a tracker
        self._init_kwargs = dict(
            samples_per_day=samples_per_day,
            time_censor_s=time_censor_s,
            random_seed=random_seed,
            include_representation=include_representation,
            lookback_period=lookback_period,
            snapshot_bin_messages=snapshot_bin_messages,
            representation_modes=representation_modes,
            raw_data_mode=raw_data_mode,
        )

    def _apply_labeling(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Apply configured labeler to a completed order record."""
        if self.labeler is None:
            return record
        try:
            result = self.labeler.label(record)
        except Exception:
            return record

        event_type = result.get("event_type")
        if event_type is not None:
            try:
                record["event_type"] = int(event_type)
            except Exception:
                record["event_type"] = event_type
        record["event_time_bin"] = result.get("event_time_bin")

        extras = result.get("extras", {}) or {}
        if isinstance(extras, dict):
            record.update(extras)
        return record

    def _append_completed_order(
        self, out_buffer, parquet_writer, output_parquet, parquet_batch_size, order
    ):
        """Serialize, label, append, and maybe flush one completed order."""
        self._enforce_time_censor_horizon(order)
        record = order.to_dict()
        record = self._apply_labeling(record)
        out_buffer.append(record)
        out_buffer, parquet_writer = self._maybe_flush(
            out_buffer, parquet_writer, output_parquet, parquet_batch_size
        )
        return out_buffer, parquet_writer

    def _enforce_time_censor_horizon(self, order: TrackedOrder) -> bool:
        """
        Clamp completed order lifetime to the configured time-censor horizon.

        Returns:
            True if the order was clipped to the horizon, otherwise False.
        """
        end_time = getattr(order, "end_time", 0)
        if end_time <= 0:
            return False
        if self.time_censor_ns is None:
            return False

        censor_deadline = order.entry_time + self.time_censor_ns
        if end_time <= censor_deadline:
            return False

        order.end_time = censor_deadline
        order.status = "CENSORED_TIME"
        return True

    def _filter_record_by_representation_modes(
        self, record: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Filter a record to exclude representation columns for unselected modes.

        Args:
            record: Dictionary record from TrackedOrder.to_dict()

        Returns:
            Filtered dictionary with only selected representation columns.
        """
        rep_columns = {
            "market_depth": [
                "entry_representation_market_depth",
                "lob_sequence_market_depth",
            ],
            "moving_window": [
                "entry_representation_moving_window",
                "lob_sequence_moving_window",
            ],
            "raw_top5": ["entry_representation_raw_top5", "lob_sequence_raw_top5"],
            "diff_top5": ["entry_representation_diff_top5", "lob_sequence_diff_top5"],
        }

        for mode, columns in rep_columns.items():
            if mode not in self.representation_modes:
                for col in columns:
                    record.pop(col, None)

        return record

    def _maybe_flush(
        self, out_buffer, parquet_writer, output_parquet, parquet_batch_size
    ):
        """
        Flush buffered records to Parquet when buffer reaches batch size.

        Args:
            out_buffer: List of dict records waiting to be written.
            parquet_writer: Existing ParquetWriter or None.
            output_parquet: Path to output parquet file.
            parquet_batch_size: Threshold size to trigger a flush.

        Returns:
            Tuple of (out_buffer, parquet_writer) after potential flush.
        """
        if len(out_buffer) >= parquet_batch_size:
            output_parquet = _prepare_output_path(output_parquet)
            # Filter records to include only selected representation modes
            filtered_buffer = [
                self._filter_record_by_representation_modes(rec) for rec in out_buffer
            ]
            table = pa.Table.from_pylist(filtered_buffer)
            if parquet_writer is None:
                parquet_writer = pq.ParquetWriter(output_parquet, table.schema)
            parquet_writer.write_table(table)
            out_buffer = []
        return out_buffer, parquet_writer

    def _init_day_schedule(self, day, day_start_ns, day_end_ns, samples_per_day):
        """Create deterministic per-day schedules if not already present."""
        if day in self.virtual_sample_schedule:
            return
        self.samples_per_day = samples_per_day
        # Schedule virtual sampling times
        virtual_target = self.samples_per_day
        v_count = max(1, (virtual_target + 1) // 2)

        # Derive day-specific RNG from stable inputs so schedules are invariant
        # to worker count and chunk execution order.
        seed_material = (
            f"{self.random_seed}|{day}|{day_start_ns}|{day_end_ns}|{samples_per_day}"
        ).encode("ascii")
        day_seed = int.from_bytes(
            hashlib.sha256(seed_material).digest()[:8], "big", signed=False
        )
        day_rng = random.Random(day_seed)
        v_times = sorted(
            int(day_rng.uniform(day_start_ns, day_end_ns)) for _ in range(v_count)
        )

        # In chunked mode, keep only samples whose timestamps belong to this chunk.
        chunk_start = getattr(self, "_chunk_ts_start", None)
        chunk_end = getattr(self, "_chunk_ts_end", None)
        if chunk_start is not None:
            v_times = [t for t in v_times if t >= chunk_start]
        if chunk_end is not None:
            v_times = [t for t in v_times if t < chunk_end]

        self.virtual_sample_schedule[day] = v_times
        self.virtual_sample_index[day] = 0
        self.inst.scheduled_virtual += len(v_times) * 2
        self.pending_virtual[day] = []

    def _move_scheduled_virtual_to_pending(self, day, mbo_ts):
        """Move scheduled virtual timestamps that have passed into pending."""
        v_schedule = self.virtual_sample_schedule.get(day, [])
        v_idx = self.virtual_sample_index.get(day, 0)
        pending_v = self.pending_virtual.get(day, [])
        while v_idx < len(v_schedule) and mbo_ts >= v_schedule[v_idx]:
            pending_v.append(v_schedule[v_idx])
            v_idx += 1
        self.virtual_sample_index[day] = v_idx
        self.pending_virtual[day] = pending_v

    def _update_active_virtual(
        self, mbo, book, out_buffer, parquet_writer, parquet_batch_size, output_parquet
    ):
        """Update all active virtual orders with the incoming MBO message.
        Orders that finish by fill are moved to post_trade_virtual for post-trade context.
        Others are appened to `out_buffer` immediately for eventual write.
        """
        next_active_virtual = []
        for v in self.active_virtual:
            v.update(mbo, book, self.market)
            if (
                self.time_censor_ns is not None
                and v.is_active()
                and (mbo.ts_event - v.entry_time > self.time_censor_ns)
            ):
                v.on_censor("CENSORED_TIME")
                v.end_time = v.entry_time + self.time_censor_ns
            if v.is_active():
                next_active_virtual.append(v)
            elif v.status == "FILLED":
                clipped = self._enforce_time_censor_horizon(v)
                if clipped:
                    out_buffer, parquet_writer = self._append_completed_order(
                        out_buffer,
                        parquet_writer,
                        output_parquet,
                        parquet_batch_size,
                        v,
                    )
                else:
                    self.post_trade_virtual.append(v)
            else:
                out_buffer, parquet_writer = self._append_completed_order(
                    out_buffer,
                    parquet_writer,
                    output_parquet,
                    parquet_batch_size,
                    v,
                )
        self.active_virtual = next_active_virtual
        return out_buffer, parquet_writer

    def _close_open_orders_for_day_end(
        self,
        day_end_ts: int,
        day: int,
        book,
        out_buffer,
        parquet_writer,
        parquet_batch_size,
        output_parquet,
    ):
        """Finalize all open orders at regular-session end for the trading day."""
        if self.active_virtual:
            for v in self.active_virtual:
                if v.is_active():
                    v.on_censor("CENSORED_END")
                    v.end_time = day_end_ts
                out_buffer, parquet_writer = self._append_completed_order(
                    out_buffer,
                    parquet_writer,
                    output_parquet,
                    parquet_batch_size,
                    v,
                )
            self.active_virtual = []

        if self.post_trade_virtual:
            for v in self.post_trade_virtual:
                if book is not None:
                    v.record_post_trade_context(book)
                if getattr(v, "end_time", 0) == 0:
                    v.end_time = day_end_ts
                out_buffer, parquet_writer = self._append_completed_order(
                    out_buffer,
                    parquet_writer,
                    output_parquet,
                    parquet_batch_size,
                    v,
                )
            self.post_trade_virtual = []

        # Process any pending virtual orders that were scheduled but never spawned
        # (e.g., scheduled after the last MBO message of the day).
        # Spawn them at day_end_ts and immediately censor them.
        pending_v = self.pending_virtual.get(day, [])
        if pending_v and book is not None:
            best_bid, best_ask = book.bbo()
            if best_bid and best_ask:
                # Use the same spawn logic to create orders with valid entry features
                self._spawn_virtuals_for_pending(day, book, best_bid, best_ask)
                # Now censor all newly-spawned actives at day end
                for v in self.active_virtual:
                    if v.is_active():
                        v.on_censor("CENSORED_END")
                        v.end_time = day_end_ts
                    out_buffer, parquet_writer = self._append_completed_order(
                        out_buffer,
                        parquet_writer,
                        output_parquet,
                        parquet_batch_size,
                        v,
                    )
                self.active_virtual = []
        elif pending_v:
            # If book state is unavailable at day end, mark pending orders as failed to spawn
            self.pending_virtual[day] = []

        return out_buffer, parquet_writer

    def _update_post_trade_virtual(
        self, mbo, book, out_buffer, parquet_writer, parquet_batch_size, output_parquet
    ):
        """Update post-trade virtuals and capture BBO at each configured time window.

        For each post-trade virtual order, checks if any of the configured windows
        have been reached (e.g., 1ms, 10ms, 50ms, etc.). Records the market BBO at
        each reached window, and finalizes the order once all windows are captured.
        """
        next_post_trade_virtual = []
        for v in self.post_trade_virtual:
            fill_time = v.end_time
            elapsed_ns = mbo.ts_event - fill_time
            elapsed_ms = elapsed_ns / 1e6

            # Check if we have any unrecorded windows that have now passed
            windows_to_record = []
            for window_ms in v.post_trade_windows_ms:
                if window_ms not in v.post_trade_bbo_dict and elapsed_ms >= window_ms:
                    windows_to_record.append(window_ms)

            # Record BBO for each newly-passed window
            for window_ms in windows_to_record:
                v.record_post_trade_context(book, elapsed_ms=int(window_ms))

            # Check if all windows have been recorded or if we've exceeded the max deadline
            is_complete = len(v.post_trade_bbo_dict) == len(
                v.post_trade_windows_ms
            ) or (
                v.post_trade_deadline_ts is not None
                and mbo.ts_event >= v.post_trade_deadline_ts
            )

            if is_complete:
                # Ensure we have recorded at least the first window's BBO
                if not v.post_trade_bbo_dict and book is not None:
                    v.record_post_trade_context(
                        book,
                        elapsed_ms=(
                            int(v.post_trade_windows_ms[0])
                            if v.post_trade_windows_ms
                            else 1
                        ),
                    )
                out_buffer, parquet_writer = self._append_completed_order(
                    out_buffer,
                    parquet_writer,
                    output_parquet,
                    parquet_batch_size,
                    v,
                )
            else:
                next_post_trade_virtual.append(v)
        self.post_trade_virtual = next_post_trade_virtual
        return out_buffer, parquet_writer

    def _get_entry_feature_sequences(
        self,
    ) -> Tuple[
        List[List[float]],
        List[List[float]],
        List[List[float]],
        List[List[float]],
        List[List[float]],
        List[List[float]],
    ]:
        """Build variable-length historical sequences from the snapshot buffer for all modes.

        Returns:
            Tuple of (entry_market_depth, entry_moving_window, entry_raw_top5,
                     entry_diff_top5, toxicity_representation, unused_legacy)
        """
        if not self.include_representation:
            return [], [], [], [], [], []

        buf = list(self._lob_snapshot_buffer)
        if not buf:
            return [], [], [], [], [], []

        entry_market_depth: List[List[float]] = []
        entry_moving_window: List[List[float]] = []
        entry_raw_top5: List[List[float]] = []
        entry_diff_top5: List[List[float]] = []
        toxicity_representation: List[List[float]] = []

        try:
            # Market depth representation
            if (
                "market_depth" in self.representation_modes
                and self.representation_transform_market_depth is not None
            ):
                md_tensor = self.representation_transform_market_depth.transform_sequence_from_dicts(
                    buf,
                    self.lookback_period,
                    pad_to_length=False,
                )
                entry_market_depth = md_tensor.tolist()

            # Moving window representation
            if (
                "moving_window" in self.representation_modes
                and self.representation_transform_moving_window is not None
            ):
                mw_tensor = self.representation_transform_moving_window.transform_sequence_from_dicts(
                    buf,
                    self.lookback_period,
                    pad_to_length=False,
                )
                entry_moving_window = mw_tensor.tolist()

            # Raw top-5 representation
            if (
                "raw_top5" in self.representation_modes
                and self.representation_transform_raw_top5 is not None
            ):
                rt_tensor = self.representation_transform_raw_top5.transform_sequence_from_dicts(
                    buf,
                    self.lookback_period,
                    pad_to_length=False,
                )
                entry_raw_top5 = rt_tensor.tolist()

            # Proportional difference top-5 representation
            if (
                "diff_top5" in self.representation_modes
                and self.representation_transform_diff_top5 is not None
            ):
                pt_tensor = self.representation_transform_diff_top5.transform_sequence_from_dicts(
                    buf,
                    self.lookback_period,
                    pad_to_length=False,
                )
                entry_diff_top5 = pt_tensor.tolist()

            # Toxicity representation (shared across all modes)
            if self.toxicity_features is not None:
                tox_tensor = self.toxicity_features.transform_sequence_from_dicts(
                    buf,
                    self.lookback_period,
                    pad_to_length=False,
                )
                toxicity_representation = tox_tensor.tolist()

        except Exception:
            pass

        # Convert empty lists to None for consistent schema (avoid PyArrow type conflicts)
        entry_market_depth = entry_market_depth if entry_market_depth else None
        entry_moving_window = entry_moving_window if entry_moving_window else None
        entry_raw_top5 = entry_raw_top5 if entry_raw_top5 else None
        entry_diff_top5 = entry_diff_top5 if entry_diff_top5 else None
        toxicity_representation = (
            toxicity_representation if toxicity_representation else None
        )

        return (
            entry_market_depth,
            entry_moving_window,
            entry_raw_top5,
            entry_diff_top5,
            toxicity_representation,
            [],
        )

    def _append_snapshot_to_active_virtuals(
        self, bids_snap: Dict[int, int], asks_snap: Dict[int, int], ts_event: int
    ) -> None:
        """Append one feature step to active orders for all representation modes.

        Args:
            bids_snap: Dict of bid prices to aggregate sizes.
            asks_snap: Dict of ask prices to aggregate sizes.
            ts_event: Timestamp in nanoseconds for time delta calculation.
        """
        if not self.active_virtual or not self.include_representation:
            return

        lob_md_step: Optional[List[float]] = None  # market depth
        lob_mw_step: Optional[List[float]] = None  # moving window
        lob_rt_step: Optional[List[float]] = None  # raw top5
        lob_dt_step: Optional[List[float]] = None  # diff top5
        tox_step: Optional[List[float]] = None
        # Include timestamp in snapshot for time delta computation
        snapshot = [(bids_snap, asks_snap, ts_event)]

        # Market depth
        try:
            if self.representation_transform_market_depth is not None:
                md_tensor = self.representation_transform_market_depth.transform_sequence_from_dicts(
                    snapshot,
                    n_lookback=1,
                    pad_to_length=False,
                )
                if md_tensor.shape[0] > 0:
                    lob_md_step = md_tensor[-1].tolist()
        except Exception:
            lob_md_step = None

        # Moving window
        try:
            if self.representation_transform_moving_window is not None:
                mw_tensor = self.representation_transform_moving_window.transform_sequence_from_dicts(
                    snapshot,
                    n_lookback=1,
                    pad_to_length=False,
                )
                if mw_tensor.shape[0] > 0:
                    lob_mw_step = mw_tensor[-1].tolist()
        except Exception:
            lob_mw_step = None

        # Raw top-5
        try:
            if self.representation_transform_raw_top5 is not None:
                rt_tensor = self.representation_transform_raw_top5.transform_sequence_from_dicts(
                    snapshot,
                    n_lookback=1,
                    pad_to_length=False,
                )
                if rt_tensor.shape[0] > 0:
                    lob_rt_step = rt_tensor[-1].tolist()
        except Exception:
            lob_rt_step = None

        # Proportional difference top-5
        try:
            if self.representation_transform_diff_top5 is not None:
                pt_tensor = self.representation_transform_diff_top5.transform_sequence_from_dicts(
                    snapshot,
                    n_lookback=1,
                    pad_to_length=False,
                )
                if pt_tensor.shape[0] > 0:
                    lob_dt_step = pt_tensor[-1].tolist()
        except Exception:
            lob_dt_step = None

        # Toxicity
        if self.toxicity_features is not None:
            try:
                tox_tensor = self.toxicity_features.transform_sequence_from_dicts(
                    snapshot,
                    n_lookback=1,
                    pad_to_length=False,
                )
                if tox_tensor.shape[0] > 0:
                    tox_step = tox_tensor[-1].tolist()
            except Exception:
                tox_step = None

        for v in self.active_virtual:
            if lob_md_step is not None:
                v.lob_sequence_market_depth.append(list(lob_md_step))
            if lob_mw_step is not None:
                v.lob_sequence_moving_window.append(list(lob_mw_step))
            if lob_rt_step is not None:
                v.lob_sequence_raw_top5.append(list(lob_rt_step))
            if lob_dt_step is not None:
                v.lob_sequence_diff_top5.append(list(lob_dt_step))
            if tox_step is not None:
                v.toxicity_sequence.append(
                    self.toxicity_features.augment_row_with_queue_position(
                        tox_step,
                        v.current_vahead,
                    )
                )

    def _spawn_virtuals_for_pending(self, day, book, best_bid, best_ask):
        """
        Spawn virtual orders for any pending scheduled timestamps.

        For each pending timestamp the function computes the queue ahead
        (with a relaxed fallback to the nearest populated level) and
        appends a new `VirtualOrder` to the active list.
        """
        pending_v = self.pending_virtual.get(day, [])
        while pending_v:
            scheduled_ts = pending_v.pop(0)
            (
                entry_market_depth,
                entry_moving_window,
                entry_raw_top5,
                entry_diff_top5,
                toxicity_representation,
                _,
            ) = self._get_entry_feature_sequences()

            bid_level_orders = book.bids.get(best_bid.price)
            if not bid_level_orders:
                best_p = None
                best_lvl = None
                for p, lvl in book.bids.items():
                    if lvl and getattr(lvl, "orders", None):
                        if best_p is None or p > best_p:
                            best_p = p
                            best_lvl = lvl
                bid_level_orders = best_lvl

            if bid_level_orders:
                ids_ahead = {o.order_id: o.size for o in bid_level_orders.orders}
                current_vahead = sum(
                    getattr(o, "size", 0) for o in bid_level_orders.orders
                )
            else:
                ids_ahead = {}
                current_vahead = getattr(best_bid, "size", 0) if best_bid else 0

            bid_toxicity_representation = (
                self.toxicity_features.augment_rows_with_queue_position(
                    toxicity_representation if toxicity_representation else [],
                    current_vahead,
                )
            )

            v = VirtualOrder(
                internal_id=self.virtual_oid_counter,
                entry_time=scheduled_ts,
                price=best_bid.price,
                side="B",
                current_vahead=current_vahead,
                ids_ahead=ids_ahead,
                best_bid_at_entry=best_bid.price if best_bid else None,
                best_ask_at_entry=best_ask.price if best_ask else None,
                entry_representation_market_depth=(
                    entry_market_depth
                    if "market_depth" in self.representation_modes
                    else None
                ),
                entry_representation_moving_window=(
                    entry_moving_window
                    if "moving_window" in self.representation_modes
                    else None
                ),
                entry_representation_raw_top5=(
                    entry_raw_top5 if "raw_top5" in self.representation_modes else None
                ),
                entry_representation_diff_top5=(
                    entry_diff_top5
                    if "diff_top5" in self.representation_modes
                    else None
                ),
                toxicity_representation=bid_toxicity_representation,
                lob_sequence_market_depth=(
                    [list(row) for row in entry_market_depth]
                    if "market_depth" in self.representation_modes
                    and entry_market_depth
                    else []
                ),
                lob_sequence_moving_window=(
                    [list(row) for row in entry_moving_window]
                    if "moving_window" in self.representation_modes
                    and entry_moving_window
                    else []
                ),
                lob_sequence_raw_top5=(
                    [list(row) for row in entry_raw_top5]
                    if "raw_top5" in self.representation_modes and entry_raw_top5
                    else []
                ),
                lob_sequence_diff_top5=(
                    [list(row) for row in entry_diff_top5]
                    if "diff_top5" in self.representation_modes and entry_diff_top5
                    else []
                ),
                lob_sequence=[],  # deprecated, kept for backward compat
                toxicity_sequence=[list(row) for row in bid_toxicity_representation],
            )
            self.active_virtual.append(v)
            self.virtual_oid_counter += 1
            self.inst.spawned_virtual += 1

            ask_level_orders = book.offers.get(best_ask.price)
            if not ask_level_orders:
                best_p = None
                best_lvl = None
                for p, lvl in book.offers.items():
                    if lvl and getattr(lvl, "orders", None):
                        if best_p is None or p < best_p:
                            best_p = p
                            best_lvl = lvl
                ask_level_orders = best_lvl

            if ask_level_orders:
                ids_ahead = {o.order_id: o.size for o in ask_level_orders.orders}
                current_vahead = sum(
                    getattr(o, "size", 0) for o in ask_level_orders.orders
                )
            else:
                ids_ahead = {}
                current_vahead = getattr(best_ask, "size", 0) if best_ask else 0

            ask_toxicity_representation = (
                self.toxicity_features.augment_rows_with_queue_position(
                    toxicity_representation if toxicity_representation else [],
                    current_vahead,
                )
            )

            v = VirtualOrder(
                internal_id=self.virtual_oid_counter,
                entry_time=scheduled_ts,
                price=best_ask.price,
                side="A",
                current_vahead=current_vahead,
                ids_ahead=ids_ahead,
                best_bid_at_entry=best_bid.price if best_bid else None,
                best_ask_at_entry=best_ask.price if best_ask else None,
                entry_representation_market_depth=(
                    entry_market_depth
                    if "market_depth" in self.representation_modes
                    else None
                ),
                entry_representation_moving_window=(
                    entry_moving_window
                    if "moving_window" in self.representation_modes
                    else None
                ),
                entry_representation_raw_top5=(
                    entry_raw_top5 if "raw_top5" in self.representation_modes else None
                ),
                entry_representation_diff_top5=(
                    entry_diff_top5
                    if "diff_top5" in self.representation_modes
                    else None
                ),
                toxicity_representation=ask_toxicity_representation,
                lob_sequence_market_depth=(
                    [list(row) for row in entry_market_depth]
                    if "market_depth" in self.representation_modes
                    and entry_market_depth
                    else []
                ),
                lob_sequence_moving_window=(
                    [list(row) for row in entry_moving_window]
                    if "moving_window" in self.representation_modes
                    and entry_moving_window
                    else []
                ),
                lob_sequence_raw_top5=(
                    [list(row) for row in entry_raw_top5]
                    if "raw_top5" in self.representation_modes and entry_raw_top5
                    else []
                ),
                lob_sequence_diff_top5=(
                    [list(row) for row in entry_diff_top5]
                    if "diff_top5" in self.representation_modes and entry_diff_top5
                    else []
                ),
                lob_sequence=[],  # deprecated, kept for backward compat
                toxicity_sequence=[list(row) for row in ask_toxicity_representation],
            )
            self.active_virtual.append(v)
            self.virtual_oid_counter += 1
            self.inst.spawned_virtual += 1

        self.pending_virtual[day] = pending_v

    def process_stream(
        self,
        file_path: str,
        output_parquet: str,
        limit: int = 10_000_000,
        progress_interval: int = 100000,
        progress_callback: Optional[Callable[[int, int, int], None]] = None,
        samples_per_day: int = 100,
        target_day: Optional[str] = None,
        chunk_ts_start: Optional[int] = None,
        chunk_ts_end: Optional[int] = None,
        tqdm_position: Optional[int] = None,
        tqdm_desc: Optional[str] = None,
        tqdm_total: Optional[int] = None,
    ):
        """
        Stream MBO messages from `file_path` and write tracked orders.

        Args:
            file_path: Path to DBN file to stream.
            output_parquet: Path to output Parquet file to write records.
            limit: Optional message limit to process.
            progress_interval: Interval (messages) to update the progress bar.
            progress_callback: Optional callable invoked as
                `(count, last_ts, active_virtual)`.
            samples_per_day: Target number of samples per trading day.
            target_day: Optional local date filter (`YYYY-MM-DD`, New York time).
                When provided, only messages from this trading day are sampled.
            chunk_ts_start: Optional nanosecond timestamp; messages strictly
                before this value are skipped without affecting book state.
                Pass an empty-market boundary so book initialises cleanly.
            chunk_ts_end: Optional nanosecond timestamp; iteration stops when
                a message at or beyond this value is encountered.
            tqdm_position: Row position of this bar in the terminal (for
                simultaneous multi-chunk display).  Pass ``None`` for
                single-stream use.
            tqdm_desc: Label shown to the left of the progress bar.
        """
        output_parquet = _prepare_output_path(output_parquet)
        self._chunk_ts_start = chunk_ts_start
        self._chunk_ts_end = chunk_ts_end
        data = db.DBNStore.from_file(file_path)
        count = 0
        last_ts = 0
        _desc = tqdm_desc or (
            f"Chunk {tqdm_position}" if tqdm_position is not None else "Processing"
        )

        # Use explicit total when provided (e.g., cached exact counts for chunk).
        # Otherwise, fall back to an estimate derived from record size.
        _total = tqdm_total
        if _total is None:
            try:
                _peek = db.DBNStore.from_file(file_path)
                _first_byte = _peek.reader.read(1)
                del _peek
                _rec_size = _first_byte[0] * 4 if _first_byte else 0
                _total = (data.nbytes // _rec_size) if _rec_size > 0 else None
            except Exception:
                _total = None
        if _total and limit:
            _total = min(_total, limit)

        pbar = tqdm(
            desc=_desc,
            total=_total,
            unit=" msg",
            position=tqdm_position,
            leave=True,
            dynamic_ncols=True,
            miniters=progress_interval,
        )
        parquet_writer = None
        parquet_batch_size = 100000
        out_buffer: List[dict] = []

        current_day = None
        current_local_date = None
        day_start_ns = None
        day_end_ns = None
        inst = self.inst
        skipped_apply_errors = 0

        target_day_date = None
        if target_day is not None:
            target_day_date = pd.Timestamp(target_day).date()
        target_day_seen = False
        chunk_count = 0  # Counter for messages within this chunk

        for mbo in data:
            count += 1
            last_ts = mbo.ts_event

            # Chunk boundary guards - check these first before updating progress
            if chunk_ts_start is not None and mbo.ts_event < chunk_ts_start:
                continue
            if chunk_ts_end is not None and mbo.ts_event >= chunk_ts_end:
                break
            if limit and count > limit:
                break

            # Message is within chunk boundaries, increment chunk counter
            chunk_count += 1

            # Update the progress bar only for messages within this chunk
            if progress_interval and chunk_count % progress_interval == 0:
                pbar.update(progress_interval)
                if progress_callback is not None:
                    try:
                        progress_callback(
                            chunk_count,
                            last_ts,
                            len(self.active_virtual),
                        )
                    except Exception:
                        pass

            # Determine which trading day this message belongs to (for trading hours check)
            # This must happen before market state updates to enable early exit for out-of-hours
            msg_day = int(mbo.ts_event // (86400 * 1e9))
            if msg_day != current_day:
                ts_dt = pd.to_datetime(mbo.ts_event, unit="ns", utc=True).tz_convert(
                    "America/New_York"
                )
                local_midnight = pd.Timestamp(ts_dt.date()).tz_localize(
                    "America/New_York"
                )
                day_start_dt = local_midnight + pd.Timedelta(hours=9, minutes=30)

                # Check if this is an early closure day (1:00 PM ET close)
                # These include Thanksgiving (Nov 28, 2025) and Christmas Eve (Dec 24, 2025)
                if ts_dt.date() in (
                    pd.Timestamp("2025-11-28").date(),
                    pd.Timestamp("2025-12-24").date(),
                ):
                    day_end_dt = local_midnight + pd.Timedelta(
                        hours=13, minutes=0
                    )  # 1:00 PM ET
                else:
                    day_end_dt = local_midnight + pd.Timedelta(
                        hours=16, minutes=0
                    )  # 4:00 PM ET

                day_start_ns = int(day_start_dt.tz_convert("UTC").value)
                day_end_ns = int(day_end_dt.tz_convert("UTC").value)
                local_date = ts_dt.date()

                if current_local_date is not None and local_date != current_local_date:
                    out_buffer, parquet_writer = self._close_open_orders_for_day_end(
                        day_end_ts=int(
                            (
                                pd.Timestamp(current_local_date)
                                .tz_localize("America/New_York")
                                .replace(hour=16, minute=0)
                                .tz_convert("UTC")
                                .value
                            )
                        ),
                        day=current_local_date,
                        book=None,
                        out_buffer=out_buffer,
                        parquet_writer=parquet_writer,
                        parquet_batch_size=parquet_batch_size,
                        output_parquet=output_parquet,
                    )

                current_local_date = local_date
                current_day = msg_day
                self._message_count_since_snapshot = 0  # Reset counter for new day

            # EARLY EXIT: Skip expensive operations for messages outside trading hours.
            # Still update market state for book continuity, but skip order tracking work.
            if last_ts < day_start_ns or last_ts > day_end_ns:
                # Update market to keep book state accurate for next in-hours message
                try:
                    self.market.apply(mbo)
                except (KeyError, AssertionError):
                    skipped_apply_errors += 1

                if last_ts > day_end_ns:
                    day_end_book = None
                    try:
                        day_end_book = self.market.get_book(
                            mbo.instrument_id, mbo.publisher_id
                        )
                    except KeyError:
                        pass
                    out_buffer, parquet_writer = self._close_open_orders_for_day_end(
                        day_end_ts=day_end_ns,
                        day=current_local_date,
                        book=day_end_book,
                        out_buffer=out_buffer,
                        parquet_writer=parquet_writer,
                        parquet_batch_size=parquet_batch_size,
                        output_parquet=output_parquet,
                    )
                continue

            if target_day_date is not None:
                if local_date < target_day_date:
                    continue
                if local_date > target_day_date:
                    if target_day_seen:
                        break
                    continue
                target_day_seen = True

            # Update market state for in-hours messages
            try:
                self.market.apply(mbo)
            except (KeyError, AssertionError):
                skipped_apply_errors += 1
                continue

            day = local_date

            self._init_day_schedule(day, day_start_ns, day_end_ns, samples_per_day)

            self._move_scheduled_virtual_to_pending(day, mbo.ts_event)

            # Increment message counter (only for in-hours messages)
            self._message_count_since_snapshot += 1
            snapshot_due = (
                self._message_count_since_snapshot >= self.snapshot_bin_messages
            )
            if (
                not self.active_virtual
                and not self.post_trade_virtual
                and not self.pending_virtual.get(day)
                and not snapshot_due
            ):
                continue

            try:
                book = self.market.get_book(mbo.instrument_id, mbo.publisher_id)
            except KeyError:
                continue

            best_bid, best_ask = book.bbo()
            if not best_bid or not best_ask:
                continue

            if snapshot_due:
                bids_snap = {
                    px: sum(o.size for o in lo.orders) for px, lo in book.bids.items()
                }
                asks_snap = {
                    px: sum(o.size for o in lo.orders) for px, lo in book.offers.items()
                }
                # Include timestamp in snapshot tuple for time delta calculation
                self._lob_snapshot_buffer.append((bids_snap, asks_snap, mbo.ts_event))
                self._last_snapshot_ts = mbo.ts_event
                self._message_count_since_snapshot = 0
                self._append_snapshot_to_active_virtuals(
                    bids_snap, asks_snap, mbo.ts_event
                )

            out_buffer, parquet_writer = self._update_active_virtual(
                mbo,
                book,
                out_buffer,
                parquet_writer,
                parquet_batch_size,
                output_parquet,
            )

            out_buffer, parquet_writer = self._update_post_trade_virtual(
                mbo,
                book,
                out_buffer,
                parquet_writer,
                parquet_batch_size,
                output_parquet,
            )

            self._spawn_virtuals_for_pending(day, book, best_bid, best_ask)

        pbar.close()

        for v in self.active_virtual:
            if v.is_active():
                v.on_censor("CENSORED_END")
                if getattr(v, "end_time", 0) == 0:
                    v.end_time = last_ts
            out_buffer, parquet_writer = self._append_completed_order(
                out_buffer,
                parquet_writer,
                output_parquet,
                parquet_batch_size,
                v,
            )

        for v in self.post_trade_virtual:
            if getattr(v, "end_time", 0) == 0:
                v.end_time = last_ts
            out_buffer, parquet_writer = self._append_completed_order(
                out_buffer,
                parquet_writer,
                output_parquet,
                parquet_batch_size,
                v,
            )

        if out_buffer:
            output_parquet = _prepare_output_path(output_parquet)
            # Filter records to include only selected representation modes
            filtered_buffer = [
                self._filter_record_by_representation_modes(rec) for rec in out_buffer
            ]
            table = pa.Table.from_pylist(filtered_buffer)
            if parquet_writer is None:
                parquet_writer = pq.ParquetWriter(output_parquet, table.schema)
            parquet_writer.write_table(table)
            out_buffer = []

        if parquet_writer is not None:
            parquet_writer.close()
        try:
            print(
                f"Sampling summary: scheduled_virtual={inst.scheduled_virtual}, spawned_virtual={inst.spawned_virtual}"
            )
            if skipped_apply_errors:
                print(f"Skipped out-of-sequence book updates: {skipped_apply_errors}")
        except Exception:
            pass

    def process_stream_parallel(
        self,
        file_path: str,
        output_parquet: str,
        n_workers: int = 4,
        empty_points: Optional[List[int]] = None,
        limit: int = None,
        progress_interval: int = 100_000,
        samples_per_day: int = 100,
        target_day: Optional[str] = None,
        empty_scan_verbose: bool = True,
        messages_between_splits: Optional[List[int]] = None,
        total_messages: Optional[int] = None,
    ) -> None:
        """
        Parallel variant of :meth:`process_stream`.

        Strategy
        --------
        1. **Scan pass** - stream the file once cheaply to find every
           timestamp where the market book transitions from non-empty to
           completely empty (:func:`find_empty_market_points`).  These are
           valid split points because each chunk can begin with a clean
           (empty) book state.
        2. **Split** - pick ``n_workers - 1`` empty points that best
           divide the timeline into *n_workers* equal-length intervals
           (:func:`_select_split_points`).
        3. **Parallel chunks** - each worker opens the file independently,
           skips messages before its ``chunk_ts_start``, processes up to
           its ``chunk_ts_end``, and writes to a private temp Parquet file.
           Workers run in separate *processes* (``ProcessPoolExecutor``) to
           bypass the GIL.
        4. **Merge** - concatenate all temp Parquet files into
           *output_parquet* in chunk order and remove the temp files.

        Args:
            file_path: Path to the DBN or DBN.ZST file.
            output_parquet: Path to the final merged Parquet output.
            n_workers: Number of parallel worker processes.
            empty_points: Pre-computed list of empty-market timestamps (ns).
                If *None* the scan pass is run automatically.
            limit: Per-chunk message limit forwarded to
                :meth:`process_stream`.
            progress_interval: Progress-print interval forwarded to each
                worker.
            samples_per_day: Samples per trading day forwarded to each
                worker.
            target_day: Optional day filter forwarded to each worker.
            empty_scan_verbose: Print progress during the scan pass.
        """

        # Discover split points
        if empty_points is None:
            print(f"[parallel] Scanning {file_path} for empty-market split points...")
            empty_points = find_empty_market_points(
                file_path, verbose=empty_scan_verbose
            )
            print(f"[parallel] Found {len(empty_points)} empty-market transitions.")

        # Use message-count-aware selection if available, otherwise fall back to time-based
        if (
            messages_between_splits is not None
            and len(messages_between_splits) == len(empty_points) + 1
        ):
            split_ts = _select_split_points_by_message_count(
                empty_points, messages_between_splits, n_workers
            )
        else:
            split_ts = _select_split_points(empty_points, n_workers)
        # Derive per-chunk progress totals.
        # Preferred path: exact cached counts projected onto selected split_ts.
        cumulative_chunk_totals: Optional[List[int]] = None

        def _per_chunk_total(_: Optional[int]) -> Optional[int]:
            return None

        if (
            messages_between_splits is not None
            and empty_points is not None
            and len(messages_between_splits) == len(empty_points) + 1
        ):
            try:
                # Build exact cumulative totals for each selected split boundary:
                # total messages with ts_event < split_ts[i].
                running = 0
                total_before_point: Dict[int, int] = {}
                for i, split_point in enumerate(empty_points):
                    total_before_point[int(split_point)] = running
                    running += int(messages_between_splits[i])

                calculated_total = running + int(messages_between_splits[-1])
                resolved_total = (
                    int(total_messages)
                    if total_messages is not None
                    else int(calculated_total)
                )

                per_split = [total_before_point.get(int(ts)) for ts in split_ts]
                if all(v is not None for v in per_split):
                    cumulative_chunk_totals = [int(v) for v in per_split]
                    cumulative_chunk_totals.append(resolved_total)
                    # Prepend 0 so we can compute per-chunk totals as cum[i+1] - cum[i]
                    cumulative_chunk_totals = [0] + cumulative_chunk_totals
            except Exception:
                cumulative_chunk_totals = None

        # Fallback path: estimate from byte-size/time-span when exact counts
        # are unavailable or mismatched.
        if cumulative_chunk_totals is None:
            try:
                _peek = db.DBNStore.from_file(file_path)
                _first_byte = _peek.reader.read(1)
                _rec_size_p = _first_byte[0] * 4 if _first_byte else 0
                _full_records: Optional[int] = (
                    (_peek.nbytes // _rec_size_p) if _rec_size_p > 0 else None
                )
                _file_ts_start: Optional[int] = _peek.metadata.start
                _file_ts_end: Optional[int] = _peek.metadata.end
                del _peek
            except Exception:
                _full_records = None
                _file_ts_start = None
                _file_ts_end = None

            def _per_chunk_total(ts_end: Optional[int]) -> Optional[int]:
                """Records from file start up to ts_end (None means full file)."""
                if (
                    _full_records is None
                    or _file_ts_start is None
                    or _file_ts_end is None
                ):
                    return None
                if ts_end is None:
                    return _full_records
                ts_range = _file_ts_end - _file_ts_start
                if ts_range <= 0:
                    return _full_records
                return max(1, int(_full_records * (ts_end - _file_ts_start) / ts_range))

        # Build (ts_start, ts_end) pairs; None means "from beginning / to end"
        boundaries: List[Optional[int]] = [None] + split_ts + [None]
        chunks: List[tuple] = [
            (boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)
        ]
        actual_workers = len(chunks)
        if (
            cumulative_chunk_totals is not None
            and len(cumulative_chunk_totals) != actual_workers + 1
        ):
            cumulative_chunk_totals = None
        print(
            f"[parallel] {actual_workers} chunk(s). "
            f"Split timestamps (UTC ns): {split_ts}"
        )

        # Build per-worker kwargs
        output_parquet = _prepare_output_path(output_parquet)
        temp_root = Path(output_parquet).parent / ".lob_parallel_tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        temp_dir = tempfile.mkdtemp(
            prefix=f"lob_parallel_{uuid4().hex}_",
            dir=str(temp_root),
        )
        temp_dir = str(Path(temp_dir).resolve())
        temp_files: List[str] = []
        worker_args: List[dict] = []

        for i, (ts_start, ts_end) in enumerate(chunks):
            tmp = _prepare_output_path(os.path.join(temp_dir, f"chunk_{i:03d}.parquet"))
            temp_files.append(tmp)

            # Calculate per-chunk total (not cumulative)
            chunk_total = None
            if cumulative_chunk_totals is not None:
                # Convert cumulative to per-chunk: chunk_i total = cum[i+1] - cum[i]
                chunk_total = int(
                    cumulative_chunk_totals[i + 1] - cumulative_chunk_totals[i]
                )
            else:
                chunk_total = _per_chunk_total(ts_end)

            worker_tracker_kwargs = dict(self._init_kwargs)
            if self.random_seed is not None:
                worker_tracker_kwargs["random_seed"] = self.random_seed

            worker_args.append(
                dict(
                    _project_root=_PROJECT_ROOT,
                    tracker_kwargs=worker_tracker_kwargs,
                    file_path=file_path,
                    output_parquet=tmp,
                    chunk_ts_start=ts_start,
                    chunk_ts_end=ts_end,
                    chunk_idx=i,
                    limit=limit,
                    progress_interval=progress_interval,
                    samples_per_day=samples_per_day,
                    target_day=target_day,
                    tqdm_position=i,
                    tqdm_desc=f"Chunk {i:02d}",
                    tqdm_total=chunk_total,
                )
            )

        # Process chunks in parallel
        results: List[Optional[dict]] = [None] * actual_workers
        with ProcessPoolExecutor(max_workers=actual_workers) as executor:
            future_to_idx = {
                executor.submit(_chunk_worker, args): args["chunk_idx"]
                for args in worker_args
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    res = future.result()
                    results[idx] = res
                    self.inst.scheduled_virtual += res["scheduled_virtual"]
                    self.inst.spawned_virtual += res["spawned_virtual"]
                    print(
                        f"[parallel] Chunk {idx} done - "
                        f"scheduled={res['scheduled_virtual']}, "
                        f"spawned={res['spawned_virtual']}"
                    )
                except Exception as exc:
                    print(f"[parallel] Chunk {idx} raised an exception: {exc}")
                    raise

        # Merge temp Parquet files in chunk order
        tables: List[pa.Table] = []
        for i, tmp in enumerate(temp_files):
            if os.path.exists(tmp):
                tables.append(pq.read_table(tmp))
            else:
                print(f"[parallel] Warning: expected chunk file {tmp} not found.")

        if tables:
            merged = pa.concat_tables(tables, promote_options="default")
            pq.write_table(merged, output_parquet)
            print(
                f"[parallel] Merged {len(tables)} chunk(s): "
                f"{merged.num_rows:,} rows written to {output_parquet}"
            )
        else:
            print("[parallel] No output produced - all chunks were empty.")

        # Clean up temp files
        for tmp in temp_files:
            try:
                os.unlink(tmp)
            except OSError:
                pass
        try:
            os.rmdir(temp_dir)
            temp_root.rmdir()
        except OSError:
            pass

        print(
            f"[parallel] Total: scheduled_virtual={self.inst.scheduled_virtual}, "
            f"spawned_virtual={self.inst.spawned_virtual}"
        )
