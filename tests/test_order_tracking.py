import os
import sys
import math
from types import SimpleNamespace
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

# Ensure project root is in sys.path for src imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.order_tracking import (
    OrderTracker,
    VirtualOrder,
    _prepare_output_path,
    _select_split_points,
    _select_split_points_by_message_count,
)
import src.order_tracking as order_tracking


class DummyLevel:
    def __init__(self, orders):
        self.orders = [SimpleNamespace(order_id=o[0], size=o[1]) for o in orders]

    @property
    def level(self):
        return SimpleNamespace(size=sum(getattr(o, "size", 0) for o in self.orders))


class DummyBook:
    def __init__(self, bids=None, offers=None):
        # maps price->level
        self.bids = bids or {}
        self.offers = offers or {}

    def bbo(self):
        # return best bid and ask as simple objects with price and size
        best_bid = None
        best_ask = None
        if self.bids:
            max_p = max(self.bids.keys())
            lvl = self.bids[max_p]
            size = sum(getattr(o, "size", 0) for o in lvl.orders) if lvl.orders else 0
            best_bid = SimpleNamespace(price=max_p, size=size)
        if self.offers:
            min_p = min(self.offers.keys())
            lvl = self.offers[min_p]
            size = sum(getattr(o, "size", 0) for o in lvl.orders) if lvl.orders else 0
            best_ask = SimpleNamespace(price=min_p, size=size)
        return best_bid, best_ask


class DummyMBO:
    def __init__(self, ts_event=0, order_id=0, action="A", size=0, price=0):
        self.ts_event = ts_event
        self.order_id = order_id
        self.action = action
        self.size = size
        self.price = price


def test_init_day_schedule_creates_schedule():
    tr = OrderTracker(samples_per_day=10)
    day = 1
    day_start_ns = 0
    day_end_ns = 1_000_000_000  # 1s span
    tr._init_day_schedule(day, day_start_ns, day_end_ns, samples_per_day=10)
    assert day in tr.virtual_sample_schedule
    v_times = tr.virtual_sample_schedule[day]
    assert isinstance(v_times, list)
    # v_count is computed as max(1, (target+1)//2)
    expected_count = max(1, (10 + 1) // 2)
    assert len(v_times) == expected_count
    assert tr.virtual_sample_index[day] == 0
    assert tr.pending_virtual[day] == []


def test_move_scheduled_virtual_to_pending():
    tr = OrderTracker(samples_per_day=4)
    day = 2
    # create two scheduled times
    t1 = 1000
    t2 = 2000
    tr.virtual_sample_schedule[day] = [t1, t2]
    tr.virtual_sample_index[day] = 0
    tr.pending_virtual[day] = []

    # move when mbo_ts >= t1
    tr._move_scheduled_virtual_to_pending(day, mbo_ts=1500)
    assert tr.virtual_sample_index[day] == 1
    assert tr.pending_virtual[day] == [t1]


def test_spawn_virtuals_for_pending_creates_virtuals():
    tr = OrderTracker(samples_per_day=2)
    day = 3
    tr.pending_virtual[day] = [123456]

    # prepare a book with an order queue at bid and ask
    bid_level = DummyLevel(orders=[(11, 5), (12, 3)])
    ask_level = DummyLevel(orders=[(21, 2), (22, 4)])
    book = DummyBook(bids={100: bid_level}, offers={101: ask_level})
    best_bid = SimpleNamespace(price=100, size=8)
    best_ask = SimpleNamespace(price=101, size=6)

    tr._spawn_virtuals_for_pending(day, book, best_bid, best_ask)
    # two virtual orders (bid & ask) spawned
    assert len(tr.active_virtual) >= 2
    # check current_vahead calculated
    v_bid = tr.active_virtual[-2]
    v_ask = tr.active_virtual[-1]
    assert v_bid.current_vahead == 8
    assert v_ask.current_vahead == 6


def test_update_active_virtual_fill_and_censor():
    tr = OrderTracker(samples_per_day=1, time_censor_s=1.0)
    # create a virtual that will be filled immediately (current_vahead=0)
    v = VirtualOrder(
        internal_id=1, entry_time=0, price=100, side="B", current_vahead=0, ids_ahead={}
    )
    tr.active_virtual = [v]
    mbo = DummyMBO(ts_event=1)
    out_buffer = []
    parquet_writer = None
    out_buffer, parquet_writer = tr._update_active_virtual(
        mbo, DummyBook(), out_buffer, parquet_writer, 10, "out.parquet"
    )
    # Filled orders now move to post-trade tracking first
    assert len(out_buffer) == 0
    assert len(tr.active_virtual) == 0
    assert len(tr.post_trade_virtual) == 1

    # now test time censoring: create v2 with entry_time far in past
    v2 = VirtualOrder(
        internal_id=2,
        entry_time=0,
        price=100,
        side="B",
        current_vahead=10,
        ids_ahead={},
    )
    tr.active_virtual = [v2]
    # create an mbo with ts_event beyond censor threshold (time_censor_s=1s -> 1e9 ns)
    long_mbo = DummyMBO(ts_event=int(2 * 1e9))
    out_buffer = []
    out_buffer, _ = tr._update_active_virtual(
        long_mbo, DummyBook(), out_buffer, parquet_writer, 10, "out.parquet"
    )
    # v2 should be censored and moved out
    assert len(out_buffer) == 1
    rec = out_buffer[0]
    assert rec["status_reason"].startswith("CENSORED")
    assert abs(rec["duration_s"] - 1.0) < 1e-12


def test_late_fill_is_converted_to_time_censor():
    tr = OrderTracker(samples_per_day=1, time_censor_s=1.0)
    # current_vahead == 0 makes this virtual fill immediately on update.
    # The update timestamp is beyond the censor horizon, so it must be censored.
    v = VirtualOrder(
        internal_id=30,
        entry_time=0,
        price=100,
        side="B",
        current_vahead=0,
        ids_ahead={},
    )
    tr.active_virtual = [v]
    mbo = DummyMBO(ts_event=int(2 * 1e9))
    out_buffer = []
    parquet_writer = None

    out_buffer, parquet_writer = tr._update_active_virtual(
        mbo, DummyBook(), out_buffer, parquet_writer, 10, "out.parquet"
    )

    assert len(tr.post_trade_virtual) == 0
    assert len(out_buffer) == 1
    rec = out_buffer[0]
    assert rec["status_reason"] == "CENSORED_TIME"
    assert rec["event"] == 0
    assert abs(rec["duration_s"] - 1.0) < 1e-12


def test_virtual_update_unknown_id_no_change():
    v = VirtualOrder(
        internal_id=1,
        entry_time=0,
        price=100,
        side="B",
        current_vahead=5,
        ids_ahead={1: 2},
    )
    mbo = DummyMBO(ts_event=1, order_id=999, action="F", size=1)
    # should not raise and current_vahead stays the same
    v.update(mbo, DummyBook(), None)
    assert v.current_vahead == 5


def test_virtual_update_modify_increase_behavior():
    # The implementation subtracts the old size and deletes the id when M increases
    v = VirtualOrder(
        internal_id=2,
        entry_time=0,
        price=100,
        side="B",
        current_vahead=5,
        ids_ahead={1: 2},
    )
    mbo = DummyMBO(ts_event=1, order_id=1, action="M", size=5)  # new size > old_size
    v.update(mbo, DummyBook(), None)
    # old_size (2) should have been subtracted and id removed
    assert v.current_vahead == 3
    assert 1 not in v.ids_ahead


def test_virtual_multiple_events_same_ts_cumulative():
    v = VirtualOrder(
        internal_id=3,
        entry_time=0,
        price=100,
        side="B",
        current_vahead=10,
        ids_ahead={1: 4, 2: 3},
    )
    mbo1 = DummyMBO(ts_event=1, order_id=1, action="C", size=2)
    mbo2 = DummyMBO(ts_event=1, order_id=2, action="F", size=1)
    mbo3 = DummyMBO(ts_event=1, order_id=1, action="M", size=1)
    v.update(mbo1, DummyBook(), None)
    v.update(mbo2, DummyBook(), None)
    v.update(mbo3, DummyBook(), None)
    # After C(2) on id1 -> vahead 8 (id1 now 2)
    # After F(1) on id2 -> vahead 7 (id2 now 2)
    # After M(1) on id1 where old=2 and new=1 -> diff=1 subtracted -> vahead 6
    assert v.current_vahead == 6


def test_time_censor_boundary_not_censored_at_equal():
    tr = OrderTracker(samples_per_day=1, time_censor_s=1.0)
    v = VirtualOrder(
        internal_id=4,
        entry_time=0,
        price=100,
        side="B",
        current_vahead=10,
        ids_ahead={},
    )
    tr.active_virtual = [v]
    mbo = DummyMBO(ts_event=int(1e9))  # exactly equal to censor threshold
    out_buffer = []
    parquet_writer = None
    out_buffer, parquet_writer = tr._update_active_virtual(
        mbo, DummyBook(), out_buffer, parquet_writer, 10, "out.parquet"
    )
    # not censored because implementation uses strict >
    assert len(tr.active_virtual) == 1


def test_init_day_schedule_with_zero_samples_creates_one():
    tr = OrderTracker(samples_per_day=0)
    day = 42
    tr._init_day_schedule(day, 0, 1_000_000_000, samples_per_day=0)
    assert day in tr.virtual_sample_schedule
    assert len(tr.virtual_sample_schedule[day]) >= 1


def test_prepare_output_path_creates_parent_directory(tmp_path):
    target = tmp_path / "nested" / "chunk_000.parquet"

    prepared = Path(_prepare_output_path(str(target)))

    assert prepared == target.resolve()
    assert prepared.parent.is_dir()


def test_spawn_fallback_to_best_size_when_level_empty():
    tr = OrderTracker(samples_per_day=1)
    day = 7
    tr.pending_virtual[day] = [123]

    # create levels but with empty orders list to force fallback
    class EmptyLevel:
        def __init__(self):
            self.orders = []

    bid_lvl = EmptyLevel()
    ask_lvl = EmptyLevel()
    book = DummyBook(bids={100: bid_lvl}, offers={101: ask_lvl})
    best_bid = SimpleNamespace(price=100, size=7)
    best_ask = SimpleNamespace(price=101, size=5)
    tr._spawn_virtuals_for_pending(day, book, best_bid, best_ask)
    # When an empty level exists, implementation uses its orders (empty) -> vahead 0
    assert any(v.current_vahead == 0 for v in tr.active_virtual)

    # Now test when level is missing entirely: fallback should use best_* size
    tr2 = OrderTracker(samples_per_day=1)
    day2 = 8
    tr2.pending_virtual[day2] = [456]
    book2 = DummyBook(bids={}, offers={})
    tr2._spawn_virtuals_for_pending(day2, book2, best_bid, best_ask)
    assert any(
        v.current_vahead == 7 or v.current_vahead == 5 for v in tr2.active_virtual
    )


def test_multiple_pending_same_timestamp_internal_ids_unique():
    tr = OrderTracker(samples_per_day=1)
    day = 9
    tr.pending_virtual[day] = [999, 999, 999]
    bid_level = DummyLevel(orders=[(11, 1)])
    ask_level = DummyLevel(orders=[(21, 1)])
    book = DummyBook(bids={100: bid_level}, offers={101: ask_level})
    best_bid = SimpleNamespace(price=100, size=1)
    best_ask = SimpleNamespace(price=101, size=1)
    tr._spawn_virtuals_for_pending(day, book, best_bid, best_ask)
    ids = [v.internal_id for v in tr.active_virtual[-6:]]
    assert len(ids) >= 6  # three pairs (bid+ask) created
    assert len(set(ids)) == len(ids)


def test_move_scheduled_virtual_skips_multiple():
    tr = OrderTracker(samples_per_day=5)
    day = 11
    tr.virtual_sample_schedule[day] = [100, 200, 300]
    tr.virtual_sample_index[day] = 0
    tr.pending_virtual[day] = []
    tr._move_scheduled_virtual_to_pending(day, mbo_ts=350)
    assert tr.virtual_sample_index[day] == 3
    assert tr.pending_virtual[day] == [100, 200, 300]


def test_update_active_virtual_multiple_finishes():
    tr = OrderTracker(samples_per_day=1)
    v1 = VirtualOrder(
        internal_id=20, entry_time=0, price=10, side="B", current_vahead=0, ids_ahead={}
    )
    v2 = VirtualOrder(
        internal_id=21, entry_time=0, price=11, side="A", current_vahead=0, ids_ahead={}
    )
    tr.active_virtual = [v1, v2]
    mbo = DummyMBO(ts_event=1)
    out_buffer = []
    parquet_writer = None
    out_buffer, parquet_writer = tr._update_active_virtual(
        mbo, DummyBook(), out_buffer, parquet_writer, 10, "out.parquet"
    )
    assert len(out_buffer) == 0
    assert len(tr.active_virtual) == 0
    assert len(tr.post_trade_virtual) == 2


def test_maybe_flush_multiple_batches_and_readback(tmp_path):
    tr = OrderTracker(samples_per_day=1)
    parquet_path = str(tmp_path / "multi.parquet")
    out_buffer = [{"x": 1}, {"x": 2}]
    parquet_writer = None
    out_buffer, parquet_writer = tr._maybe_flush(
        out_buffer, parquet_writer, parquet_path, parquet_batch_size=2
    )
    assert out_buffer == []
    assert parquet_writer is not None
    # write another batch reusing writer
    out_buffer = [{"x": 3}, {"x": 4}]
    out_buffer, parquet_writer = tr._maybe_flush(
        out_buffer, parquet_writer, parquet_path, parquet_batch_size=2
    )
    parquet_writer.close()
    # read back and check rows
    table = pq.read_table(parquet_path)
    arr = table.to_pydict()["x"]
    assert sorted(arr) == [1, 2, 3, 4]


def test_time_censor_ns_conversion():
    tr = OrderTracker(time_censor_s=2.5)
    assert tr.time_censor_ns == int(2.5 * 1e9)


def test_default_tracker_disables_time_censor():
    tr = OrderTracker()
    assert tr.time_censor_ns is None


def test_trackedorder_to_dict_fields():
    # Verify to_dict emits expected keys and volume heuristic
    v = VirtualOrder(
        internal_id=5, entry_time=0, price=50, side="A", current_vahead=0, ids_ahead={}
    )
    # mark filled
    mbo = DummyMBO(ts_event=100)
    v.on_fill(mbo)
    rec = v.to_dict()
    assert rec["order_id"] == 5
    assert rec["event"] == 1
    assert abs(rec["duration_s"] - 100 / 1e9) < 1e-12
    assert rec["volume"] == 1


def test_spawn_uses_sequence_entry_representation():
    tr = OrderTracker(samples_per_day=1, lookback_period=5)
    day = 12
    tr.pending_virtual[day] = [999]

    bid_level = DummyLevel(orders=[(11, 5), (12, 3)])
    ask_level = DummyLevel(orders=[(21, 2), (22, 4)])
    book = DummyBook(bids={100: bid_level}, offers={101: ask_level})
    best_bid = SimpleNamespace(price=100, size=8)
    best_ask = SimpleNamespace(price=101, size=6)

    # Manually populate snapshot buffer with 3 snapshots
    for _ in range(3):
        bids_snap = {100: 8}
        asks_snap = {101: 6}
        tr._lob_snapshot_buffer.append((bids_snap, asks_snap))

    tr._spawn_virtuals_for_pending(day, book, best_bid, best_ask)

    assert len(tr.active_virtual) >= 2
    sample_repr = tr.active_virtual[-1].entry_representation_moving_window
    assert sample_repr is not None
    assert isinstance(sample_repr, list)
    assert len(sample_repr) == 3
    assert isinstance(sample_repr[0], list)

    # Dynamic sequences are initialized from available history with no padding.
    sample_seq = tr.active_virtual[-1].lob_sequence_moving_window
    assert isinstance(sample_seq, list)
    assert len(sample_seq) == 3


def test_maybe_flush_writes_parquet(tmp_path):
    tr = OrderTracker(samples_per_day=1)
    out_buffer = [{"a": 1}, {"a": 2}]
    parquet_path = str(tmp_path / "out.parquet")
    parquet_writer = None
    out_buffer, parquet_writer = tr._maybe_flush(
        out_buffer, parquet_writer, parquet_path, parquet_batch_size=2
    )
    # after flush, buffer should be empty and writer created
    assert out_buffer == []
    assert parquet_writer is not None
    parquet_writer.close()
    assert os.path.exists(parquet_path)


def test_spawn_appends_queue_position_to_toxicity_features():
    tr = OrderTracker(samples_per_day=1, lookback_period=5)
    day = 13
    tr.pending_virtual[day] = [111]

    bid_level = DummyLevel(orders=[(11, 5), (12, 3)])
    ask_level = DummyLevel(orders=[(21, 2), (22, 4)])
    book = DummyBook(bids={100: bid_level}, offers={101: ask_level})
    best_bid = SimpleNamespace(price=100, size=8)
    best_ask = SimpleNamespace(price=101, size=6)

    for _ in range(3):
        tr._lob_snapshot_buffer.append(({100: 8}, {101: 6}))

    tr._spawn_virtuals_for_pending(day, book, best_bid, best_ask)

    v_bid = tr.active_virtual[-2]
    v_ask = tr.active_virtual[-1]

    assert v_bid.toxicity_representation
    assert v_ask.toxicity_representation

    expected_bid_qp = math.log1p(float(v_bid.current_vahead))
    expected_ask_qp = math.log1p(float(v_ask.current_vahead))

    assert all(row[-1] == expected_bid_qp for row in v_bid.toxicity_representation)
    assert all(row[-1] == expected_ask_qp for row in v_ask.toxicity_representation)

    assert all(row[-1] == expected_bid_qp for row in v_bid.toxicity_sequence)
    assert all(row[-1] == expected_ask_qp for row in v_ask.toxicity_sequence)


def test_append_snapshot_uses_each_order_queue_position_in_toxicity_sequence():
    tr = OrderTracker(samples_per_day=1)
    v_bid = VirtualOrder(
        internal_id=40,
        entry_time=0,
        price=100,
        side="B",
        current_vahead=7,
        ids_ahead={},
    )
    v_ask = VirtualOrder(
        internal_id=41,
        entry_time=0,
        price=101,
        side="A",
        current_vahead=2,
        ids_ahead={},
    )
    tr.active_virtual = [v_bid, v_ask]

    tr._append_snapshot_to_active_virtuals({100: 10}, {101: 9}, ts_event=1)

    assert len(v_bid.toxicity_sequence) == 1
    assert len(v_ask.toxicity_sequence) == 1
    assert v_bid.toxicity_sequence[0][-1] == math.log1p(7.0)
    assert v_ask.toxicity_sequence[0][-1] == math.log1p(2.0)


def test_select_split_points_by_message_count_prefers_balanced_boundaries():
    # 4 candidate empty points produce 5 message segments.
    # For 3 workers, ideal cumulative boundaries are around 40 and 80.
    empty_points = [10, 20, 30, 40]
    messages_between_splits = [5, 35, 35, 20, 5]

    split_ts = _select_split_points_by_message_count(
        empty_points=empty_points,
        messages_between_splits=messages_between_splits,
        n=3,
    )

    # cumulative at empty points: [5, 40, 75, 95], so closest picks are 20 and 30
    assert split_ts == [20, 30]


def test_select_split_points_by_message_count_falls_back_on_mismatch():
    empty_points = [100, 200, 300, 400]
    # Invalid length should trigger fallback to time-based selector.
    invalid_counts = [1, 2]

    from_fallback = _select_split_points(empty_points, 3)
    from_message_count = _select_split_points_by_message_count(
        empty_points=empty_points,
        messages_between_splits=invalid_counts,
        n=3,
    )

    assert from_message_count == from_fallback


def test_filter_record_by_representation_modes_removes_unselected_columns():
    tr = OrderTracker(representation_modes=["moving_window", "raw_top5"])
    record = {
        "entry_representation_market_depth": [[1.0]],
        "lob_sequence_market_depth": [[1.0]],
        "entry_representation_moving_window": [[2.0]],
        "lob_sequence_moving_window": [[2.0]],
        "entry_representation_raw_top5": [[3.0]],
        "lob_sequence_raw_top5": [[3.0]],
        "entry_representation_diff_top5": [[4.0]],
        "lob_sequence_diff_top5": [[4.0]],
    }

    filtered = tr._filter_record_by_representation_modes(record)

    assert "entry_representation_market_depth" not in filtered
    assert "lob_sequence_market_depth" not in filtered
    assert "entry_representation_diff_top5" not in filtered
    assert "lob_sequence_diff_top5" not in filtered
    assert "entry_representation_moving_window" in filtered
    assert "lob_sequence_moving_window" in filtered
    assert "entry_representation_raw_top5" in filtered
    assert "lob_sequence_raw_top5" in filtered


def test_enforce_time_censor_horizon_clips_completed_order():
    tr = OrderTracker(time_censor_s=1.0)
    v = VirtualOrder(
        internal_id=55,
        entry_time=0,
        price=100,
        side="B",
        current_vahead=0,
        ids_ahead={},
    )
    v.status = "FILLED"
    v.end_time = int(2e9)

    clipped = tr._enforce_time_censor_horizon(v)

    assert clipped is True
    assert v.status == "CENSORED_TIME"
    assert v.end_time == int(1e9)


class _ImmediateFuture:
    def __init__(self, fn, args):
        self._result = fn(args)

    def result(self):
        return self._result


class _ImmediateExecutor:
    def __init__(self, max_workers=None):
        self.max_workers = max_workers

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def submit(self, fn, args):
        return _ImmediateFuture(fn, args)


def _write_synthetic_chunk(output_parquet, ts_start, ts_end, seed, rows):
    selected = []
    for row in rows:
        ts_event = row["ts_event"]
        if ts_start is not None and ts_event < ts_start:
            continue
        if ts_end is not None and ts_event >= ts_end:
            continue
        selected.append(
            {
                "row_id": int(row["row_id"]),
                "ts_event": int(ts_event),
                "seed": int(seed),
            }
        )

    schema = pa.schema(
        [
            ("row_id", pa.int64()),
            ("ts_event", pa.int64()),
            ("seed", pa.int64()),
        ]
    )
    table = pa.Table.from_pylist(selected, schema=schema)
    pq.write_table(table, output_parquet)


def test_parallel_output_matches_single_process_under_fixed_seed(tmp_path, monkeypatch):
    synthetic_rows = [
        {"row_id": 1, "ts_event": 10},
        {"row_id": 2, "ts_event": 20},
        {"row_id": 3, "ts_event": 30},
        {"row_id": 4, "ts_event": 40},
        {"row_id": 5, "ts_event": 50},
        {"row_id": 6, "ts_event": 60},
        {"row_id": 7, "ts_event": 70},
    ]

    def _fake_chunk_worker(kwargs):
        output_parquet = kwargs["output_parquet"]
        chunk_ts_start = kwargs.get("chunk_ts_start")
        chunk_ts_end = kwargs.get("chunk_ts_end")
        seed = kwargs.get("tracker_kwargs", {}).get("random_seed", -1)
        _write_synthetic_chunk(
            output_parquet=output_parquet,
            ts_start=chunk_ts_start,
            ts_end=chunk_ts_end,
            seed=seed,
            rows=synthetic_rows,
        )
        return {
            "chunk_idx": kwargs["chunk_idx"],
            "output_parquet": output_parquet,
            "scheduled_virtual": 0,
            "spawned_virtual": 0,
        }

    monkeypatch.setattr(order_tracking, "_chunk_worker", _fake_chunk_worker)
    monkeypatch.setattr(order_tracking, "ProcessPoolExecutor", _ImmediateExecutor)
    monkeypatch.setattr(order_tracking, "as_completed", lambda futures: futures)

    fixed_seed = 2026
    single_out = tmp_path / "single.parquet"
    parallel_out = tmp_path / "parallel.parquet"

    _write_synthetic_chunk(
        output_parquet=str(single_out),
        ts_start=None,
        ts_end=None,
        seed=fixed_seed,
        rows=synthetic_rows,
    )

    tracker = OrderTracker(random_seed=fixed_seed)
    tracker.process_stream_parallel(
        file_path="unused.dbn.zst",
        output_parquet=str(parallel_out),
        n_workers=3,
        empty_points=[30, 60],
        empty_scan_verbose=False,
    )

    single_rows = pq.read_table(single_out).to_pylist()
    parallel_rows = pq.read_table(parallel_out).to_pylist()
    assert parallel_rows == single_rows


def test_parallel_output_is_repeatable_with_fixed_seed(tmp_path, monkeypatch):
    synthetic_rows = [
        {"row_id": 11, "ts_event": 10},
        {"row_id": 12, "ts_event": 20},
        {"row_id": 13, "ts_event": 30},
        {"row_id": 14, "ts_event": 40},
        {"row_id": 15, "ts_event": 50},
    ]

    def _fake_chunk_worker(kwargs):
        output_parquet = kwargs["output_parquet"]
        chunk_ts_start = kwargs.get("chunk_ts_start")
        chunk_ts_end = kwargs.get("chunk_ts_end")
        seed = kwargs.get("tracker_kwargs", {}).get("random_seed", -1)
        _write_synthetic_chunk(
            output_parquet=output_parquet,
            ts_start=chunk_ts_start,
            ts_end=chunk_ts_end,
            seed=seed,
            rows=synthetic_rows,
        )
        return {
            "chunk_idx": kwargs["chunk_idx"],
            "output_parquet": output_parquet,
            "scheduled_virtual": 0,
            "spawned_virtual": 0,
        }

    monkeypatch.setattr(order_tracking, "_chunk_worker", _fake_chunk_worker)
    monkeypatch.setattr(order_tracking, "ProcessPoolExecutor", _ImmediateExecutor)
    monkeypatch.setattr(order_tracking, "as_completed", lambda futures: futures)

    fixed_seed = 12345
    first_out = tmp_path / "parallel_first.parquet"
    second_out = tmp_path / "parallel_second.parquet"

    tracker_a = OrderTracker(random_seed=fixed_seed)
    tracker_b = OrderTracker(random_seed=fixed_seed)

    tracker_a.process_stream_parallel(
        file_path="unused.dbn.zst",
        output_parquet=str(first_out),
        n_workers=2,
        empty_points=[30],
        empty_scan_verbose=False,
    )
    tracker_b.process_stream_parallel(
        file_path="unused.dbn.zst",
        output_parquet=str(second_out),
        n_workers=2,
        empty_points=[30],
        empty_scan_verbose=False,
    )

    first_rows = pq.read_table(first_out).to_pylist()
    second_rows = pq.read_table(second_out).to_pylist()
    assert first_rows == second_rows


def test_parallel_worker_count_invariance_schema_and_row_order(tmp_path, monkeypatch):
    synthetic_rows = [
        {"row_id": 101, "ts_event": 10},
        {"row_id": 102, "ts_event": 20},
        {"row_id": 103, "ts_event": 30},
        {"row_id": 104, "ts_event": 40},
        {"row_id": 105, "ts_event": 50},
        {"row_id": 106, "ts_event": 60},
        {"row_id": 107, "ts_event": 70},
        {"row_id": 108, "ts_event": 80},
    ]

    def _fake_chunk_worker(kwargs):
        output_parquet = kwargs["output_parquet"]
        chunk_ts_start = kwargs.get("chunk_ts_start")
        chunk_ts_end = kwargs.get("chunk_ts_end")
        seed = kwargs.get("tracker_kwargs", {}).get("random_seed", -1)
        _write_synthetic_chunk(
            output_parquet=output_parquet,
            ts_start=chunk_ts_start,
            ts_end=chunk_ts_end,
            seed=seed,
            rows=synthetic_rows,
        )
        return {
            "chunk_idx": kwargs["chunk_idx"],
            "output_parquet": output_parquet,
            "scheduled_virtual": 0,
            "spawned_virtual": 0,
        }

    monkeypatch.setattr(order_tracking, "_chunk_worker", _fake_chunk_worker)
    monkeypatch.setattr(order_tracking, "ProcessPoolExecutor", _ImmediateExecutor)
    monkeypatch.setattr(order_tracking, "as_completed", lambda futures: futures)

    fixed_seed = 999
    out_2_workers = tmp_path / "parallel_2_workers.parquet"
    out_4_workers = tmp_path / "parallel_4_workers.parquet"

    tracker_2 = OrderTracker(random_seed=fixed_seed)
    tracker_4 = OrderTracker(random_seed=fixed_seed)

    # Empty-market points define legal split boundaries.
    # Running with 2 vs 4 workers exercises different chunking layouts.
    empty_points = [30, 50, 70]

    tracker_2.process_stream_parallel(
        file_path="unused.dbn.zst",
        output_parquet=str(out_2_workers),
        n_workers=2,
        empty_points=empty_points,
        empty_scan_verbose=False,
    )
    tracker_4.process_stream_parallel(
        file_path="unused.dbn.zst",
        output_parquet=str(out_4_workers),
        n_workers=4,
        empty_points=empty_points,
        empty_scan_verbose=False,
    )

    table_2 = pq.read_table(out_2_workers)
    table_4 = pq.read_table(out_4_workers)

    assert table_2.schema == table_4.schema

    rows_2 = table_2.to_pylist()
    rows_4 = table_4.to_pylist()
    assert rows_2 == rows_4

    # Explicit row-order check for readability in failures.
    assert [r["row_id"] for r in rows_2] == [r["row_id"] for r in rows_4]
