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
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
import databento as db
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
import random

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from .lob_implementation import Market, Book

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
from .config import CONFIG
from .labeling.base import BaseLabeler
from .labeling.competing_risks import ExecutionCompetingRisksLabeler
from .features.base import BaseLOBTransform
from .features.representation import RepresentationTransform


def _market_is_empty(market: Market) -> bool:
    """Return True iff every book in the market has no resting orders."""
    for pub_books in market.books.values():
        for book in pub_books.values():
            if book.bids or book.offers or book.orders_by_id:
                return False
    return True


def find_empty_market_points(file_path: str, verbose: bool = True) -> List[int]:
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


def _select_split_points(empty_points: List[int], n: int) -> List[int]:
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
    output_parquet = kwargs.pop("output_parquet")
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


@dataclass
class Instrumentation:
    """
    Counters for sampling instrumentation.

    Counters are incremented during schedule creation and when samples are
    spawned to help with run diagnostics.
    """

    scheduled_virtual: int = 0
    spawned_virtual: int = 0


@dataclass
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

    def on_censor(self, reason: str, mbo: Optional[db.MBOMsg] = None):
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
            base["entry_representation"] = getattr(self, "entry_representation", None)

        return base


@dataclass
class VirtualOrder(TrackedOrder):
    """
    Virtual (synthetic) order used for survival analysis sampling.

    Virtual orders approximate the lifecycle of a limit order placed at the
    top-of-book at a given timestamp. They track volume ahead and ids for
    basic queue-depletion logic.
    """

    current_vahead: int = 0
    ids_ahead: Dict[int, int] = field(default_factory=dict)
    best_bid_at_entry: int = None
    best_ask_at_entry: int = None
    best_bid_at_post_trade: int = None
    best_ask_at_post_trade: int = None
    entry_representation: Optional[List] = None  # (lookback, 2W+1) 2-D list

    def __post_init__(self):
        self.order_type = "VIRTUAL"

    def update(self, mbo: db.MBOMsg, book: Book, market: Market):
        if not self.is_active():
            return

        b, a = book.bbo()
        best_bid_px = b.price if b else None
        best_ask_px = a.price if a else None

        # PRICE RUNAWAY DETECTION
        if self.side == "B" and best_bid_px is not None:
            price_move_bps = ((self.price - best_bid_px) / self.price) * 10000
            if price_move_bps > CONFIG.labeling.price_runaway_bps:
                self.on_censor("PRICE_RUNAWAY", mbo)
                return
        elif self.side == "A" and best_ask_px is not None:
            price_move_bps = ((best_ask_px - self.price) / self.price) * 10000
            if price_move_bps > CONFIG.labeling.price_runaway_bps:
                self.on_censor("PRICE_RUNAWAY", mbo)
                return

        # Fill when no volume is ahead
        if self.current_vahead <= 0:
            self.on_fill(mbo)
            # Start post-trade context tracking
            self.post_trade_counter = 0
            self.post_trade_window = CONFIG.labeling.tox_post_trade_move_window_events
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

    def record_post_trade_context(self, book: Book):
        b, a = book.bbo()
        self.best_bid_at_post_trade = b.price if b else None
        self.best_ask_at_post_trade = a.price if a else None


@dataclass
class OrderTracker:
    """
    Orchestrates sampling and tracking of virtual (simulated) orders.

    The tracker maintains in-memory lists of active virtual orders,
    sampling schedules per day, and writes finished records to a Parquet
    file via a streaming buffer.

    Args:
        samples_per_day: Target number of samples per trading day.
    """

    def __init__(
        self,
        samples_per_day: int = 100,
        time_censor_s: float = 300.0,
        labeler: Optional[BaseLabeler] = None,
        representation_transform: Optional[BaseLOBTransform] = None,
        include_representation: bool = True,
        lookback_period: int = 10,
        snapshot_bin_s: Optional[float] = None,
    ):
        self.market = Market()
        self.active_virtual: List[VirtualOrder] = []
        self.post_trade_virtual: List[VirtualOrder] = []
        self.completed: List[TrackedOrder] = []

        self.last_sample_time = 0
        self.virtual_oid_counter = 0

        # Time censor threshold (nanoseconds)
        self.time_censor_ns = int(time_censor_s * 1e9)

        self.samples_per_day = samples_per_day
        self.virtual_sample_schedule: Dict[int, List[int]] = {}
        self.virtual_sample_index: Dict[int, int] = {}
        self.pending_virtual: Dict[int, List[int]] = {}

        self.labeler = labeler or ExecutionCompetingRisksLabeler()
        self.representation_transform = (
            representation_transform or RepresentationTransform()
        )
        self.include_representation = include_representation

        self.lookback_period = lookback_period
        _bin_s = (
            snapshot_bin_s if snapshot_bin_s is not None else CONFIG.features.interval_s
        )
        self._snapshot_bin_ns: int = int(_bin_s * 1e9)
        self._lob_snapshot_buffer: collections.deque = collections.deque(
            maxlen=lookback_period
        )
        self._last_snapshot_ts: int = 0

        self.inst = Instrumentation()

        # Store primitive constructor args so workers can recreate a tracker
        self._init_kwargs = dict(
            samples_per_day=samples_per_day,
            time_censor_s=time_censor_s,
            include_representation=include_representation,
            lookback_period=lookback_period,
            snapshot_bin_s=_bin_s,
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
        record = order.to_dict()
        record = self._apply_labeling(record)
        out_buffer.append(record)
        out_buffer, parquet_writer = self._maybe_flush(
            out_buffer, parquet_writer, output_parquet, parquet_batch_size
        )
        return out_buffer, parquet_writer

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
            table = pa.Table.from_pylist(out_buffer)
            if parquet_writer is None:
                parquet_writer = pq.ParquetWriter(output_parquet, table.schema)
            parquet_writer.write_table(table)
            out_buffer = []
        return out_buffer, parquet_writer

    def _init_day_schedule(self, day, day_start_ns, day_end_ns, samples_per_day):
        """Create per-day random sampling schedules if not already present."""
        if day in self.virtual_sample_schedule:
            return
        self.samples_per_day = samples_per_day
        v_times: List[int] = []
        # Schedule virtual sampling times
        virtual_target = self.samples_per_day
        v_count = max(1, (virtual_target + 1) // 2)
        v_times = sorted(
            int(random.uniform(day_start_ns, day_end_ns)) for _ in range(v_count)
        )
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
            # Time censoring
            if v.is_active() and (mbo.ts_event - v.entry_time > self.time_censor_ns):
                v.on_censor("CENSORED_TIME", mbo)

            if v.is_active():
                next_active_virtual.append(v)
            elif v.status == "FILLED":
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

    def _update_post_trade_virtual(
        self, mbo, book, out_buffer, parquet_writer, parquet_batch_size, output_parquet
    ):
        """Update post-trade virtuals, increment counter, and capture BBO at window end."""
        next_post_trade_virtual = []
        for v in self.post_trade_virtual:
            v.post_trade_counter += 1
            if v.post_trade_counter >= v.post_trade_window:
                v.record_post_trade_context(book)
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
            entry_representation = None
            if (
                self.include_representation
                and self.representation_transform is not None
            ):
                try:
                    buf = list(self._lob_snapshot_buffer)
                    entry_representation = (
                        self.representation_transform.transform_sequence_from_dicts(
                            buf, self.lookback_period
                        ).tolist()
                    )
                except Exception:
                    entry_representation = None

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

            v = VirtualOrder(
                internal_id=self.virtual_oid_counter,
                entry_time=scheduled_ts,
                price=best_bid.price,
                side="B",
                current_vahead=current_vahead,
                ids_ahead=ids_ahead,
                best_bid_at_entry=best_bid.price if best_bid else None,
                best_ask_at_entry=best_ask.price if best_ask else None,
                entry_representation=entry_representation,
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

            v = VirtualOrder(
                internal_id=self.virtual_oid_counter,
                entry_time=scheduled_ts,
                price=best_ask.price,
                side="A",
                current_vahead=current_vahead,
                ids_ahead=ids_ahead,
                best_bid_at_entry=best_bid.price if best_bid else None,
                best_ask_at_entry=best_ask.price if best_ask else None,
                entry_representation=entry_representation,
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
        data = db.DBNStore.from_file(file_path)
        count = 0
        last_ts = 0
        _desc = tqdm_desc or (
            f"Chunk {tqdm_position}" if tqdm_position is not None else "Processing"
        )
        pbar = tqdm(
            desc=_desc,
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
        day_start_ns = None
        day_end_ns = None
        inst = self.inst
        skipped_apply_errors = 0

        target_day_date = None
        if target_day is not None:
            target_day_date = pd.Timestamp(target_day).date()
        target_day_seen = False

        for mbo in data:
            count += 1
            last_ts = mbo.ts_event

            # Chunk boundary guards
            if chunk_ts_start is not None and mbo.ts_event < chunk_ts_start:
                continue
            if chunk_ts_end is not None and mbo.ts_event >= chunk_ts_end:
                break

            if progress_interval and count % progress_interval == 0:
                pbar.update(progress_interval)
                if progress_callback is not None:
                    try:
                        progress_callback(
                            count,
                            last_ts,
                            len(self.active_virtual),
                        )
                    except Exception:
                        pass
            if limit and count > limit:
                break

            # Update market state first to preserve continuity for in-session cancels/modifies.
            try:
                self.market.apply(mbo)
            except (KeyError, AssertionError):
                skipped_apply_errors += 1
                continue

            msg_day = int(mbo.ts_event // (86400 * 1e9))
            if msg_day != current_day:
                ts_dt = pd.to_datetime(mbo.ts_event, unit="ns", utc=True).tz_convert(
                    "America/New_York"
                )
                local_midnight = pd.Timestamp(ts_dt.date()).tz_localize(
                    "America/New_York"
                )
                day_start_dt = local_midnight + pd.Timedelta(hours=9, minutes=30)
                day_end_dt = local_midnight + pd.Timedelta(hours=16, minutes=0)
                day_start_ns = int(day_start_dt.tz_convert("UTC").value)
                day_end_ns = int(day_end_dt.tz_convert("UTC").value)
                local_date = ts_dt.date()
                current_day = msg_day

            if target_day_date is not None:
                if local_date < target_day_date:
                    continue
                if local_date > target_day_date:
                    if target_day_seen:
                        break
                    continue
                target_day_seen = True

            if last_ts < day_start_ns or last_ts > day_end_ns:
                continue

            day = local_date

            self._init_day_schedule(day, day_start_ns, day_end_ns, samples_per_day)

            self._move_scheduled_virtual_to_pending(day, mbo.ts_event)

            snapshot_due = (
                mbo.ts_event - self._last_snapshot_ts >= self._snapshot_bin_ns
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
                self._lob_snapshot_buffer.append((bids_snap, asks_snap))
                self._last_snapshot_ts = mbo.ts_event

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
            table = pa.Table.from_pylist(out_buffer)
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

        split_ts = _select_split_points(empty_points, n_workers)

        # Build (ts_start, ts_end) pairs; None means "from beginning / to end"
        boundaries: List[Optional[int]] = [None] + split_ts + [None]
        chunks: List[tuple] = [
            (boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)
        ]
        actual_workers = len(chunks)
        print(
            f"[parallel] {actual_workers} chunk(s). "
            f"Split timestamps (UTC ns): {split_ts}"
        )

        # Build per-worker kwargs
        Path(output_parquet).parent.mkdir(parents=True, exist_ok=True)
        temp_dir = tempfile.mkdtemp(prefix="lob_parallel_")
        temp_files: List[str] = []
        worker_args: List[dict] = []

        for i, (ts_start, ts_end) in enumerate(chunks):
            tmp = os.path.join(temp_dir, f"chunk_{i:03d}.parquet")
            temp_files.append(tmp)
            worker_args.append(
                dict(
                    _project_root=_PROJECT_ROOT,
                    tracker_kwargs=dict(
                        self._init_kwargs
                    ),  # shallow copy of primitives
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
        except OSError:
            pass

        print(
            f"[parallel] Total: scheduled_virtual={self.inst.scheduled_virtual}, "
            f"spawned_virtual={self.inst.spawned_virtual}"
        )
