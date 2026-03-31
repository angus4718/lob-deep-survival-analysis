from __future__ import annotations

import json
import os
import sys
import multiprocessing
from pathlib import Path
import pyarrow.compute as pc
import pyarrow.parquet as pq

# Ensure reproducible results across parallel runs
# Must be set BEFORE importing numpy or spawning processes
if "PYTHONHASHSEED" not in os.environ:
    os.environ["PYTHONHASHSEED"] = "0"

import numpy as np
import random

# Ensure project root is in sys.path for src imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.order_tracking import (
    OrderTracker,
    analyze_empty_market_splits,
)
from src.config import CONFIG

SYMBOL = "AAPL"
START_DATE = "2025-10-01"
END_DATE = "2025-11-01"

SAMPLES_PER_DAY = 1000
# Keep long-lived orders for dynamic models; disable fixed time censoring.
TIME_CENSOR_S = None
LOOKBACK_PERIOD = 200
RANDOM_SEED = CONFIG.random_seed
PROGRESS_INTERVAL = 100_000

# LOB representation modes to include in dataset.
# Valid options: "market_depth", "moving_window", "raw_top5"
# Default: all three modes.
REPRESENTATION_MODES = ["raw_top5"]

# Set to a "YYYY-MM-DD" string to restrict processing to a single trading day,
# or None to process the entire file.
TARGET_DAY = None  # e.g. "2025-12-01"

# Set the number of parallel worker processes,
# or None to auto-select based on available CPU cores (leaving 2 cores free).
N_WORKERS = None

FILE_NAME = (
    f"XNAS_ITCH_{SYMBOL}_mbo_{START_DATE.replace('-', '')}_{END_DATE.replace('-', '')}"
)
dbn_path = Path("data") / "raw" / f"{FILE_NAME}.dbn.zst"
output_path = Path("data") / "datasets" / f"dataset_{FILE_NAME}.parquet"
split_cache_path = Path("data") / "datasets" / f"{FILE_NAME}_split_points.json"


def _load_or_build_split_cache(cache_path: Path, dbn_file: Path) -> dict:
    """Load cached split metadata or build it on first run."""
    if cache_path.exists():
        with open(cache_path) as f:
            cached = json.load(f)

        if isinstance(cached, dict) and {
            "split_points",
            "messages_between_splits",
            "total_messages",
        }.issubset(cached.keys()):
            print(
                "[cache] Loaded split metadata: "
                f"{len(cached['split_points'])} split points, "
                f"{cached['total_messages']:,} total messages"
            )
            return cached

        print("[cache] Cache format not recognized. Rebuilding split metadata...")

    print(f"[cache] No valid cache found - scanning {dbn_file} for split metadata...")
    analyzed = analyze_empty_market_splits(file_path=str(dbn_file), verbose=True)
    analyzed["version"] = 2
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(analyzed, f)
    print(
        "[cache] Saved split metadata: "
        f"{len(analyzed['split_points'])} split points, "
        f"{analyzed['total_messages']:,} total messages"
    )
    return analyzed


def main() -> None:
    # Ensure reproducibility by seeding all random sources
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

    global N_WORKERS
    if N_WORKERS is None:
        total_cores = multiprocessing.cpu_count()
        N_WORKERS = max(1, total_cores - 2)
        print(
            f"Auto-selected {N_WORKERS} workers based on {total_cores} available cores."
        )

    if not os.path.exists(dbn_path):
        print(f"File {dbn_path} not found.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    tracker = OrderTracker(
        samples_per_day=SAMPLES_PER_DAY,
        time_censor_s=TIME_CENSOR_S,
        lookback_period=LOOKBACK_PERIOD,
        random_seed=RANDOM_SEED,
        representation_modes=REPRESENTATION_MODES,
    )

    if N_WORKERS > 1:
        print(f"Running in parallel mode with {N_WORKERS} workers.")

        split_cache = _load_or_build_split_cache(split_cache_path, dbn_path)
        empty_points = split_cache.get("split_points", [])
        messages_between_splits = split_cache.get("messages_between_splits", [])
        total_messages = split_cache.get("total_messages")

        tracker.process_stream_parallel(
            file_path=str(dbn_path),
            output_parquet=str(output_path),
            n_workers=N_WORKERS,
            empty_points=empty_points,
            messages_between_splits=messages_between_splits,
            total_messages=total_messages,
            progress_interval=PROGRESS_INTERVAL,
            samples_per_day=SAMPLES_PER_DAY,
            target_day=TARGET_DAY,
        )
    else:
        print("Running in single-process mode.")
        tracker.process_stream(
            file_path=str(dbn_path),
            output_parquet=str(output_path),
            limit=None,
            progress_interval=PROGRESS_INTERVAL,
            samples_per_day=SAMPLES_PER_DAY,
            target_day=TARGET_DAY,
        )

    parquet_file = pq.ParquetFile(output_path)
    columns = parquet_file.schema.names
    shape = (parquet_file.metadata.num_rows, len(columns))

    print("shape:", shape)
    print("columns:", columns)

    print("\nEvent type distribution:")
    if "event_type" not in columns:
        print("event_type column not found")
        return

    event_col = pq.read_table(output_path, columns=["event_type"])["event_type"]
    counts = pc.value_counts(event_col).to_pylist()
    counts_sorted = sorted(
        counts,
        key=lambda x: (x["values"] is None, x["values"]),
    )
    for row in counts_sorted:
        print(f"{row['values']}: {row['counts']}")


if __name__ == "__main__":
    main()
