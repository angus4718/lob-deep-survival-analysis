import os
import sys
from types import SimpleNamespace
from pathlib import Path
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from order_tracking import OrderTracker, VirtualOrder


class DummyLevel:
    def __init__(self, orders):
        self.orders = [SimpleNamespace(order_id=o[0], size=o[1]) for o in orders]


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
    # v should be moved to out_buffer and active_virtual empty
    assert len(out_buffer) == 1
    assert len(tr.active_virtual) == 0

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
    assert len(out_buffer) == 2
    assert len(tr.active_virtual) == 0


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
