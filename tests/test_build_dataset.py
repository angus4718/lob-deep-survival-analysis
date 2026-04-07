import importlib
import json
import sys
from pathlib import Path

import pandas as pd


# Ensure project root is in sys.path for scripts imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


build_dataset = importlib.import_module("scripts.build_dataset")


def test_load_or_build_split_cache_reads_valid_cache(tmp_path, monkeypatch):
    cache_path = tmp_path / "split_cache.json"
    cached = {
        "split_points": [100, 200],
        "messages_between_splits": [10, 20, 30],
        "total_messages": 60,
    }
    cache_path.write_text(json.dumps(cached), encoding="utf-8")

    def _should_not_run(*args, **kwargs):
        raise AssertionError("analyze_empty_market_splits should not be called")

    monkeypatch.setattr(build_dataset, "analyze_empty_market_splits", _should_not_run)

    out = build_dataset._load_or_build_split_cache(
        cache_path, tmp_path / "file.dbn.zst"
    )

    assert out == cached


def test_load_or_build_split_cache_rebuilds_invalid_cache(tmp_path, monkeypatch):
    cache_path = tmp_path / "split_cache.json"
    cache_path.write_text("{}", encoding="utf-8")

    analyzed = {
        "split_points": [111],
        "messages_between_splits": [4, 5],
        "total_messages": 9,
    }
    monkeypatch.setattr(
        build_dataset,
        "analyze_empty_market_splits",
        lambda file_path, verbose: analyzed.copy(),
    )

    out = build_dataset._load_or_build_split_cache(
        cache_path, tmp_path / "file.dbn.zst"
    )

    assert out == analyzed

    on_disk = json.loads(cache_path.read_text(encoding="utf-8"))
    assert on_disk == analyzed


def test_main_parallel_mode_passes_split_metadata(tmp_path, monkeypatch):
    dbn_file = tmp_path / "input.dbn.zst"
    dbn_file.write_bytes(b"x")

    out_file = tmp_path / "out.parquet"
    split_cache_file = tmp_path / "split_cache.json"

    calls = {"parallel": 0, "single": 0}

    class FakeTracker:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def process_stream_parallel(self, **kwargs):
            calls["parallel"] += 1
            calls["parallel_kwargs"] = kwargs
            pd.DataFrame({"event_type": [1, 2]}).to_parquet(kwargs["output_parquet"])

        def process_stream(self, **kwargs):
            calls["single"] += 1
            pd.DataFrame({"event_type": [1]}).to_parquet(kwargs["output_parquet"])

    split_meta = {
        "split_points": [1000, 2000],
        "messages_between_splits": [7, 8, 9],
        "total_messages": 24,
    }

    monkeypatch.setattr(build_dataset, "OrderTracker", FakeTracker)
    monkeypatch.setattr(
        build_dataset,
        "_load_or_build_split_cache",
        lambda cache_path, dbn: split_meta,
    )
    monkeypatch.setattr(build_dataset, "dbn_path", dbn_file)
    monkeypatch.setattr(build_dataset, "output_path", out_file)
    monkeypatch.setattr(build_dataset, "split_cache_path", split_cache_file)
    monkeypatch.setattr(build_dataset, "SKIP_FIRST_SEGMENT_IF_RAW_DATA_EXISTS", False)
    monkeypatch.setattr(build_dataset, "N_WORKERS", 2)

    build_dataset.main()

    assert calls["parallel"] == 1
    assert calls["single"] == 0
    assert calls["parallel_kwargs"]["empty_points"] == split_meta["split_points"]
    assert (
        calls["parallel_kwargs"]["messages_between_splits"]
        == split_meta["messages_between_splits"]
    )
    assert calls["parallel_kwargs"]["total_messages"] == split_meta["total_messages"]


def test_main_returns_when_dbn_missing(tmp_path, monkeypatch):
    missing_dbn = tmp_path / "does_not_exist.dbn.zst"

    monkeypatch.setattr(build_dataset, "dbn_path", missing_dbn)

    # Should return early without raising.
    build_dataset.main()


def test_main_single_process_mode_calls_process_stream(tmp_path, monkeypatch):
    dbn_file = tmp_path / "input.dbn.zst"
    dbn_file.write_bytes(b"x")

    out_file = tmp_path / "out.parquet"

    calls = {"parallel": 0, "single": 0}

    class FakeTracker:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def process_stream_parallel(self, **kwargs):
            calls["parallel"] += 1
            pd.DataFrame({"event": [1]}).to_parquet(kwargs["output_parquet"])

        def process_stream(self, **kwargs):
            calls["single"] += 1
            calls["single_kwargs"] = kwargs
            pd.DataFrame({"event": [1]}).to_parquet(kwargs["output_parquet"])

    monkeypatch.setattr(build_dataset, "OrderTracker", FakeTracker)
    monkeypatch.setattr(build_dataset, "dbn_path", dbn_file)
    monkeypatch.setattr(build_dataset, "output_path", out_file)
    monkeypatch.setattr(build_dataset, "SKIP_FIRST_SEGMENT_IF_RAW_DATA_EXISTS", False)
    monkeypatch.setattr(build_dataset, "N_WORKERS", 1)

    build_dataset.main()

    assert calls["parallel"] == 0
    assert calls["single"] == 1
    assert calls["single_kwargs"]["target_day"] == build_dataset.TARGET_DAY


def test_main_skips_first_segment_when_output_exists(tmp_path, monkeypatch):
    dbn_file = tmp_path / "input.dbn.zst"
    dbn_file.write_bytes(b"x")

    out_file = tmp_path / "out.parquet"
    pd.DataFrame({"event": [0]}).to_parquet(out_file)

    class ShouldNotConstructTracker:
        def __init__(self, **kwargs):
            raise AssertionError("OrderTracker should not be instantiated")

    monkeypatch.setattr(build_dataset, "OrderTracker", ShouldNotConstructTracker)
    monkeypatch.setattr(build_dataset, "dbn_path", dbn_file)
    monkeypatch.setattr(build_dataset, "output_path", out_file)
    monkeypatch.setattr(build_dataset, "SKIP_FIRST_SEGMENT_IF_RAW_DATA_EXISTS", True)
    monkeypatch.setattr(build_dataset, "N_WORKERS", 2)

    # Should not raise: existing output should be reused.
    build_dataset.main()


def test_main_auto_selects_workers_when_none(tmp_path, monkeypatch):
    dbn_file = tmp_path / "input.dbn.zst"
    dbn_file.write_bytes(b"x")
    out_file = tmp_path / "out.parquet"

    calls = {"parallel": 0}

    class FakeTracker:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def process_stream_parallel(self, **kwargs):
            calls["parallel"] += 1
            calls["n_workers"] = kwargs["n_workers"]
            pd.DataFrame({"event": [1]}).to_parquet(kwargs["output_parquet"])

        def process_stream(self, **kwargs):
            pd.DataFrame({"event": [1]}).to_parquet(kwargs["output_parquet"])

    monkeypatch.setattr(build_dataset, "OrderTracker", FakeTracker)
    monkeypatch.setattr(build_dataset, "dbn_path", dbn_file)
    monkeypatch.setattr(build_dataset, "output_path", out_file)
    monkeypatch.setattr(build_dataset, "SKIP_FIRST_SEGMENT_IF_RAW_DATA_EXISTS", False)
    monkeypatch.setattr(build_dataset, "N_WORKERS", None)
    monkeypatch.setattr(build_dataset.multiprocessing, "cpu_count", lambda: 6)

    build_dataset.main()

    assert calls["parallel"] == 1
    assert calls["n_workers"] == 6
