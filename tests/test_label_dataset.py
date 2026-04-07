import importlib
import sys
from pathlib import Path

import pandas as pd


# Ensure project root is in sys.path for scripts imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


label_dataset = importlib.import_module("scripts.label_dataset")


def _to_utc_ns(local_dt: str) -> int:
    return int(pd.Timestamp(local_dt, tz="America/New_York").tz_convert("UTC").value)


def test_label_dataset_returns_when_raw_missing(tmp_path):
    raw_path = tmp_path / "missing.parquet"
    out_path = tmp_path / "labeled.parquet"

    label_dataset.label_dataset_adaptive_toxic_window(raw_path, out_path)

    assert not out_path.exists()


def test_label_dataset_returns_when_no_filled_orders(tmp_path):
    raw_path = tmp_path / "raw.parquet"
    out_path = tmp_path / "labeled.parquet"

    df = pd.DataFrame(
        [
            {
                "entry_time": _to_utc_ns("2026-01-05 10:00:00"),
                "event": 0,
                "status_reason": "CENSORED_END",
                "duration_s": 0.5,
                "price": 100,
                "side": "B",
                "best_bid_at_entry": 99,
                "best_ask_at_entry": 101,
            },
            {
                "entry_time": _to_utc_ns("2026-01-06 10:00:00"),
                "event": 0,
                "status_reason": "CENSORED_END",
                "duration_s": 0.4,
                "price": 101,
                "side": "A",
                "best_bid_at_entry": 100,
                "best_ask_at_entry": 102,
            },
            {
                "entry_time": _to_utc_ns("2026-01-07 10:00:00"),
                "event": 0,
                "status_reason": "CENSORED_END",
                "duration_s": 0.3,
                "price": 102,
                "side": "B",
                "best_bid_at_entry": 101,
                "best_ask_at_entry": 103,
            },
        ]
    )
    df.to_parquet(raw_path)

    label_dataset.label_dataset_adaptive_toxic_window(raw_path, out_path)

    assert not out_path.exists()


def test_day_boundary_splits_follow_exact_rule(tmp_path):
    rows = []
    for day_offset, day in enumerate(["2026-01-05", "2026-01-06", "2026-01-07"]):
        for minute in range(3):
            rows.append(
                {
                    "entry_time": _to_utc_ns(f"{day} 10:00:0{minute}"),
                    "event": 1 if minute == 0 else 0,
                    "status_reason": "FILLED" if minute == 0 else "CENSORED_END",
                    "duration_s": 0.2 + day_offset,
                    "price": 100 + day_offset,
                    "side": "B",
                    "best_bid_at_entry": 99 + day_offset,
                    "best_ask_at_entry": 101 + day_offset,
                }
            )

    df = pd.DataFrame(rows)
    train_mask, val_mask, test_mask, train_days, val_days, test_days = (
        label_dataset._compute_day_boundary_splits(df)
    )

    assert len(train_days) == 1
    assert len(val_days) == 1
    assert len(test_days) == 1
    assert train_mask.sum() == 3
    assert val_mask.sum() == 3
    assert test_mask.sum() == 3
    assert (train_mask | val_mask | test_mask).all()
    assert not (train_mask & val_mask).any()
    assert not (train_mask & test_mask).any()
    assert not (val_mask & test_mask).any()


def test_label_dataset_relabels_and_writes_output(tmp_path, monkeypatch):
    raw_path = tmp_path / "raw.parquet"
    out_path = tmp_path / "labeled.parquet"

    # 9 records across 3 days; day-boundary split should allocate 1 day each to
    # train, val, and test, while all 9 rows are labeled in the output.
    rows = [
        {
            "entry_time": _to_utc_ns("2026-01-05 10:00:00"),
            "event": 1,
            "status_reason": "FILLED",
            "duration_s": 0.2,
            "price": 100,
            "side": "B",
            "best_bid_at_entry": 99,
            "best_ask_at_entry": 101,
            "best_bid_at_post_trade": 100,
            "best_ask_at_post_trade": 101,
            "post_trade_best_bid_1ms": 100,
            "post_trade_best_ask_1ms": 101,
        },
        {
            "entry_time": _to_utc_ns("2026-01-05 10:00:01"),
            "event": 0,
            "status_reason": "CENSORED_END",
            "duration_s": 0.5,
            "price": 101,
            "side": "A",
            "best_bid_at_entry": 100,
            "best_ask_at_entry": 102,
            "best_bid_at_post_trade": 100,
            "best_ask_at_post_trade": 102,
            "post_trade_best_bid_1ms": 100,
            "post_trade_best_ask_1ms": 102,
        },
        {
            "entry_time": _to_utc_ns("2026-01-05 10:00:02"),
            "event": 1,
            "status_reason": "FILLED",
            "duration_s": 0.3,
            "price": 102,
            "side": "B",
            "best_bid_at_entry": 101,
            "best_ask_at_entry": 103,
            "best_bid_at_post_trade": 102,
            "best_ask_at_post_trade": 103,
            "post_trade_best_bid_1ms": 102,
            "post_trade_best_ask_1ms": 103,
        },
        {
            "entry_time": _to_utc_ns("2026-01-05 10:00:03"),
            "event": 0,
            "status_reason": "CENSORED_TIME",
            "duration_s": 1.0,
            "price": 103,
            "side": "A",
            "best_bid_at_entry": 102,
            "best_ask_at_entry": 104,
            "best_bid_at_post_trade": 102,
            "best_ask_at_post_trade": 104,
            "post_trade_best_bid_1ms": 102,
            "post_trade_best_ask_1ms": 104,
        },
        {
            "entry_time": _to_utc_ns("2026-01-06 10:00:00"),
            "event": 1,
            "status_reason": "FILLED",
            "duration_s": 0.2,
            "price": 110,
            "side": "B",
            "best_bid_at_entry": 109,
            "best_ask_at_entry": 111,
        },
        {
            "entry_time": _to_utc_ns("2026-01-06 10:00:01"),
            "event": 0,
            "status_reason": "CENSORED_END",
            "duration_s": 0.6,
            "price": 111,
            "side": "A",
            "best_bid_at_entry": 110,
            "best_ask_at_entry": 112,
        },
        {
            "entry_time": _to_utc_ns("2026-01-06 10:00:02"),
            "event": 1,
            "status_reason": "FILLED",
            "duration_s": 0.25,
            "price": 112,
            "side": "B",
            "best_bid_at_entry": 111,
            "best_ask_at_entry": 113,
        },
        {
            "entry_time": _to_utc_ns("2026-01-06 10:00:03"),
            "event": 0,
            "status_reason": "CENSORED_END",
            "duration_s": 0.7,
            "price": 113,
            "side": "A",
            "best_bid_at_entry": 112,
            "best_ask_at_entry": 114,
        },
        {
            "entry_time": _to_utc_ns("2026-01-07 10:00:00"),
            "event": 1,
            "status_reason": "FILLED",
            "duration_s": 0.15,
            "price": 120,
            "side": "B",
            "best_bid_at_entry": 119,
            "best_ask_at_entry": 121,
            "best_bid_at_post_trade": 120,
            "best_ask_at_post_trade": 121,
            "post_trade_best_bid_1ms": 120,
            "post_trade_best_ask_1ms": 121,
        },
        {
            "entry_time": _to_utc_ns("2026-01-07 10:00:01"),
            "event": 0,
            "status_reason": "CENSORED_END",
            "duration_s": 0.35,
            "price": 121,
            "side": "A",
            "best_bid_at_entry": 120,
            "best_ask_at_entry": 122,
            "best_bid_at_post_trade": 120,
            "best_ask_at_post_trade": 122,
            "post_trade_best_bid_1ms": 120,
            "post_trade_best_ask_1ms": 122,
        },
        {
            "entry_time": _to_utc_ns("2026-01-07 10:00:02"),
            "event": 1,
            "status_reason": "FILLED",
            "duration_s": 0.4,
            "price": 122,
            "side": "B",
            "best_bid_at_entry": 121,
            "best_ask_at_entry": 123,
            "best_bid_at_post_trade": 122,
            "best_ask_at_post_trade": 123,
            "post_trade_best_bid_1ms": 122,
            "post_trade_best_ask_1ms": 123,
        },
    ]
    pd.DataFrame(rows).to_parquet(raw_path)

    seen = {}

    class FakeWindowSelector:
        pass

    class FakeAnalyzer:
        def __init__(self, window_selector, winsorize):
            assert isinstance(window_selector, FakeWindowSelector)
            assert winsorize is True

        def analyze(self, df_filled):
            seen["filled_rows"] = len(df_filled)
            return {"selected_window": {"found": True, "chosen_window_ms": 25}}

    class FakeLabeler:
        def __init__(self, selected_window):
            seen["selected_window"] = selected_window

        def label(self, record):
            base_event = 1 if record.get("status_reason") == "FILLED" else 0
            return {
                "event_type": base_event,
                "extras": {"label_selected_window_ms": seen["selected_window"]},
            }

    monkeypatch.setattr(
        label_dataset, "StabilizationWindowSelector", FakeWindowSelector
    )
    monkeypatch.setattr(label_dataset, "MarkoutAnalyzer", FakeAnalyzer)
    monkeypatch.setattr(label_dataset, "ExecutionCompetingRisksLabeler", FakeLabeler)

    label_dataset.label_dataset_adaptive_toxic_window(raw_path, out_path)

    assert out_path.exists()
    labeled = pd.read_parquet(out_path)

    # Window calibration uses the first day only, while all 11 records are labeled.
    assert len(labeled) == 11
    assert seen["filled_rows"] == 2
    assert seen["selected_window"] == 25

    assert "event_type" in labeled.columns
    assert "label_selected_window_ms" in labeled.columns
    assert set(labeled["event_type"].tolist()) == {0, 1}


def test_label_dataset_main_uses_configured_paths(monkeypatch):
    seen = {}

    def fake_label(raw_path, out_path):
        seen["raw_path"] = raw_path
        seen["out_path"] = out_path

    monkeypatch.setattr(
        label_dataset, "label_dataset_adaptive_toxic_window", fake_label
    )

    label_dataset.main()

    assert seen["raw_path"] == label_dataset.RAW_DATASET_PATH
    assert seen["out_path"] == label_dataset.OUTPUT_DATASET_PATH
