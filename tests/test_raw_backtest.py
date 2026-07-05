from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest import (
    BacktestFeatureBuilder,
    RawDatabentoBacktestEngine,
    StaticLatencyProvider,
)
from src.backtest.metrics import ImplementationShortfallMetric
from src.backtest.strategies.base import BaseStrategy
from src.backtest.types import DecisionAction, TradingDecision


class SubmitStrategy(BaseStrategy):
    def decide(self, snapshot):
        return TradingDecision(
            action=DecisionAction.SUBMIT,
            limit_price=float(snapshot.row["price"]),
            reason="submit",
        )


class SkipStrategy(BaseStrategy):
    def decide(self, snapshot):
        return TradingDecision(
            action=DecisionAction.SKIP,
            limit_price=float(snapshot.row["price"]),
            reason="skip",
        )


class LifecycleCancelStrategy(BaseStrategy):
    def decide(self, snapshot):
        if snapshot.position_open:
            return TradingDecision(
                action=DecisionAction.CANCEL,
                limit_price=float(snapshot.row["price"]),
                reason="cancel",
            )
        return TradingDecision(
            action=DecisionAction.SUBMIT,
            limit_price=float(snapshot.row["price"]),
            reason="submit",
        )


class HoldLifecycleStrategy(BaseStrategy):
    def decide(self, snapshot):
        return TradingDecision(
            action=DecisionAction.HOLD if snapshot.position_open else DecisionAction.SUBMIT,
            limit_price=float(snapshot.row["price"]),
            reason="hold" if snapshot.position_open else "submit",
        )


class DummyLevel:
    def __init__(self, orders):
        self.orders = [SimpleNamespace(order_id=oid, size=size) for oid, size in orders]

    @property
    def level(self):
        return SimpleNamespace(size=sum(order.size for order in self.orders))


class DummyBook:
    def __init__(self):
        self.bids = {}
        self.offers = {}

    def set_bbo(self, bid_price=100, bid_size=5, ask_price=101, ask_size=5):
        self.bids = {bid_price: DummyLevel([(1, bid_size)])}
        self.offers = {ask_price: DummyLevel([(2, ask_size)])}

    def bbo(self):
        bid = None
        ask = None
        if self.bids:
            px = max(self.bids)
            bid = SimpleNamespace(price=px, size=self.bids[px].level.size)
        if self.offers:
            px = min(self.offers)
            ask = SimpleNamespace(price=px, size=self.offers[px].level.size)
        return bid, ask


class DummyMarket:
    def __init__(self):
        self.book = DummyBook()
        self.book.set_bbo()

    def apply(self, mbo):
        if getattr(mbo, "bid_price", None) is not None:
            self.book.set_bbo(
                bid_price=mbo.bid_price,
                bid_size=getattr(mbo, "bid_size", 5),
                ask_price=mbo.ask_price,
                ask_size=getattr(mbo, "ask_size", 5),
            )

    def get_book(self, instrument_id, publisher_id):
        return self.book


def mbo(ts_event, **kwargs):
    base = {
        "ts_event": ts_event,
        "instrument_id": 1,
        "publisher_id": 1,
        "order_id": 0,
        "action": "N",
        "size": 0,
    }
    base.update(kwargs)
    return SimpleNamespace(**base)


def labeled_orders(side="B", price=100, duration_s=0.000001):
    return pd.DataFrame(
        [
            {
                "order_id": 10,
                "entry_time": 100,
                "duration_s": duration_s,
                "event_type": 0,
                "side": side,
                "price": price,
                "status_reason": "CENSORED_TIME",
                "best_bid_at_entry": 100,
                "best_ask_at_entry": 101,
            }
        ]
    )


def labeled_orders_with_features():
    lob_a = [[float(i)] * 20 for i in range(1, 3)]
    tox_a = [[float(i)] * 12 for i in range(1, 3)]
    lob_b = [[10.0] * 20, [20.0] * 20]
    tox_b = [[10.0] * 12, [20.0] * 12]
    return pd.DataFrame(
        [
            {
                "order_id": 10,
                "entry_time": 100,
                "duration_s": 0.0001,
                "event_type": 0,
                "side": "B",
                "price": 100,
                "status_reason": "CENSORED_TIME",
                "best_bid_at_entry": 100,
                "best_ask_at_entry": 101,
                "lob_sequence_raw_top5": lob_a,
                "toxicity_sequence": tox_a,
                "sequence_length": 2,
            },
            {
                "order_id": 11,
                "entry_time": 130,
                "duration_s": 0.0001,
                "event_type": 0,
                "side": "B",
                "price": 100,
                "status_reason": "CENSORED_TIME",
                "best_bid_at_entry": 100,
                "best_ask_at_entry": 101,
                "lob_sequence_raw_top5": lob_b,
                "toxicity_sequence": tox_b,
                "sequence_length": 2,
            },
        ]
    )


def feature_builder():
    return BacktestFeatureBuilder(
        lookback_steps=2,
        lob_dim=20,
        tox_dim=12,
        snapshot_policy="entry",
    )


def metric():
    return ImplementationShortfallMetric(
        selected_window_ms=1000,
        calibrate_toxic_window=False,
    )


def test_raw_engine_applies_static_latency_before_submit():
    records = [
        mbo(100),
        mbo(110),
        mbo(120, order_id=1, action="F", size=5),
    ]
    report = RawDatabentoBacktestEngine(
        SubmitStrategy(),
        records=records,
        market=DummyMarket(),
        feature_builder=feature_builder(),
        metrics=[metric()],
        latency_provider=StaticLatencyProvider(latency_ns=10),
        snapshot_bin_messages=1,
    ).run(labeled_orders())

    raw = report.raw_frame()
    assert len(raw) == 1
    assert raw.loc[0, "decision_model_latency_ns"] == 10
    assert raw.loc[0, "submitted"]
    assert raw.loc[0, "filled"]
    assert raw.loc[0, "decision_submit_time"] == 110


def test_raw_report_summary_includes_average_latency_ms():
    records = [
        mbo(100),
        mbo(500_100),
    ]
    report = RawDatabentoBacktestEngine(
        SkipStrategy(),
        records=records,
        market=DummyMarket(),
        feature_builder=feature_builder(),
        metrics=[metric()],
        latency_provider=StaticLatencyProvider.from_microseconds(500),
        snapshot_bin_messages=1,
    ).run(labeled_orders())

    summary = report.summary_frame()
    assert summary.loc[0, "average_latency_ms"] == 0.5


def test_raw_engine_skip_uses_same_side_quote_after_latency():
    records = [
        mbo(100, bid_price=100, ask_price=101),
        mbo(120, bid_price=100, ask_price=105),
    ]
    report = RawDatabentoBacktestEngine(
        SkipStrategy(),
        records=records,
        market=DummyMarket(),
        feature_builder=feature_builder(),
        metrics=[metric()],
        latency_provider=StaticLatencyProvider(latency_ns=20),
        snapshot_bin_messages=1,
    ).run(labeled_orders())

    raw = report.raw_frame()
    assert len(raw) == 1
    assert not raw.loc[0, "submitted"]
    assert raw.loc[0, "cost_type"] == "opportunity_cost"
    assert raw.loc[0, "opportunity_quote_side"] == "ask"
    assert raw.loc[0, "implementation_shortfall_raw"] == 4.0


def test_raw_engine_lifecycle_cancel_uses_cancel_latency():
    records = [
        mbo(100),
        mbo(110),
        mbo(115),
        mbo(120),
    ]
    report = RawDatabentoBacktestEngine(
        LifecycleCancelStrategy(),
        records=records,
        market=DummyMarket(),
        feature_builder=feature_builder(),
        metrics=[metric()],
        latency_provider=StaticLatencyProvider(latency_ns=5),
        snapshot_bin_messages=1,
        lifecycle_aware=True,
        lifecycle_stride=1,
    ).run(labeled_orders(duration_s=0.0001))

    raw = report.raw_frame()
    assert len(raw) == 1
    assert raw.loc[0, "decision_action"] == "cancel"
    assert raw.loc[0, "canceled"]
    assert raw.loc[0, "decision_end_time"] == 120


def test_raw_engine_rejects_orders_before_raw_start_by_default():
    records = [
        mbo(200),
    ]
    try:
        RawDatabentoBacktestEngine(
            SkipStrategy(),
            records=records,
            market=DummyMarket(),
            feature_builder=feature_builder(),
            metrics=[metric()],
            snapshot_bin_messages=1,
        ).run(labeled_orders())
    except ValueError as exc:
        assert "Raw replay starts after the first labeled order entry_time" in str(exc)
    else:
        raise AssertionError("Expected raw replay coverage validation to fail.")


def test_raw_engine_merges_rebuilt_tail_for_overlapping_initial_order():
    records = [
        mbo(100),
        mbo(110),
        mbo(120),
        mbo(130),
        mbo(140),
    ]
    report = RawDatabentoBacktestEngine(
        HoldLifecycleStrategy(),
        records=records,
        market=DummyMarket(),
        feature_builder=feature_builder(),
        metrics=[metric()],
        latency_provider=StaticLatencyProvider(latency_ns=0),
        snapshot_bin_messages=1,
        lifecycle_aware=True,
        lifecycle_stride=10,
    ).run(labeled_orders_with_features())

    raw = report.raw_frame()
    assert len(raw) == 2
    second = raw.loc[raw["order_id"] == 11].iloc[0]
    assert second["decision_raw_mode"]
    assert second["decision_initial_feature_source"] == "label_rebuilt_tail"
    assert second["decision_end_idx"] == 1


def test_raw_engine_does_not_reuse_stale_rebuilt_tail_after_active_order_finishes():
    records = [
        mbo(100),
        mbo(110),
        mbo(120, order_id=1, action="F", size=5),
        mbo(130),
        mbo(140),
    ]
    report = RawDatabentoBacktestEngine(
        HoldLifecycleStrategy(),
        records=records,
        market=DummyMarket(),
        feature_builder=feature_builder(),
        metrics=[metric()],
        latency_provider=StaticLatencyProvider(latency_ns=0),
        snapshot_bin_messages=1,
        lifecycle_aware=True,
        lifecycle_stride=10,
    ).run(labeled_orders_with_features())

    raw = report.raw_frame()
    assert len(raw) == 2
    second = raw.loc[raw["order_id"] == 11].iloc[0]
    assert second["decision_initial_feature_source"] == "label"


def test_raw_engine_default_does_not_censor_at_labeled_duration():
    records = [
        mbo(100),
        mbo(200),
    ]
    report = RawDatabentoBacktestEngine(
        SubmitStrategy(),
        records=records,
        market=DummyMarket(),
        feature_builder=feature_builder(),
        metrics=[metric()],
        latency_provider=StaticLatencyProvider(latency_ns=0),
        snapshot_bin_messages=1,
    ).run(labeled_orders(duration_s=0.000000001))

    raw = report.raw_frame()
    assert len(raw) == 1
    assert raw.loc[0, "decision_raw_status"] == "CENSORED_END"
    assert raw.loc[0, "decision_has_deadline"]
    assert raw.loc[0, "decision_deadline_time"] > raw.loc[0, "decision_observation_time"]


def test_raw_engine_default_deadline_is_regular_session_end():
    observation_time = int(
        pd.Timestamp("2025-10-01 15:59:59.999999990", tz="America/New_York")
        .tz_convert("UTC")
        .value
    )
    day_end_time = int(
        pd.Timestamp("2025-10-01 16:00:00", tz="America/New_York")
        .tz_convert("UTC")
        .value
    )
    records = [
        mbo(observation_time),
        mbo(day_end_time + 1),
    ]
    rows = labeled_orders(duration_s=999.0)
    rows.loc[0, "entry_time"] = observation_time

    report = RawDatabentoBacktestEngine(
        SubmitStrategy(),
        records=records,
        market=DummyMarket(),
        feature_builder=feature_builder(),
        metrics=[metric()],
        latency_provider=StaticLatencyProvider(latency_ns=0),
        snapshot_bin_messages=1,
    ).run(rows)

    raw = report.raw_frame()
    assert len(raw) == 1
    assert raw.loc[0, "decision_raw_status"] == "CENSORED_TIME"
    assert raw.loc[0, "decision_deadline_time"] == day_end_time


def test_raw_engine_labeled_duration_deadline_starts_at_observation_time():
    records = [
        mbo(200),
        mbo(205),
        mbo(211),
    ]
    report = RawDatabentoBacktestEngine(
        SubmitStrategy(),
        records=records,
        market=DummyMarket(),
        feature_builder=feature_builder(),
        metrics=[metric()],
        latency_provider=StaticLatencyProvider(latency_ns=0),
        snapshot_bin_messages=1,
        strict_time_coverage=False,
        censor_at_labeled_duration=True,
    ).run(labeled_orders(duration_s=0.00000001))

    raw = report.raw_frame()
    assert len(raw) == 1
    assert raw.loc[0, "decision_raw_status"] == "CENSORED_TIME"
    assert raw.loc[0, "decision_has_deadline"]
    assert raw.loc[0, "decision_deadline_time"] == 210
