from __future__ import annotations

import json
import os
import sys
import multiprocessing
from pathlib import Path
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pandas as pd

# Ensure project root is in sys.path for src imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.order_tracking import (
    OrderTracker,
    analyze_empty_market_splits,
)
from src.config import CONFIG
from src.labeling.window_selecting import MarkoutAnalyzer, StabilizationWindowSelector
from src.labeling.competing_risks import ExecutionCompetingRisksLabeler

ADAPTIVE_TOXIC_WINDOW = True
# Skip the first segment if raw data already exists (saves time when iterating on labeling)
SKIP_FIRST_SEGMENT_IF_RAW_DATA_EXISTS = True

SYMBOL = "AAPL"
START_DATE = "2025-11-01"
END_DATE = "2025-12-01"

SAMPLES_PER_DAY = 1000
# Keep long-lived orders for dynamic models; disable fixed time censoring.
TIME_CENSOR_S = None
LOOKBACK_PERIOD = 20
RANDOM_SEED = CONFIG.random_seed
PROGRESS_INTERVAL = 100_000

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
if ADAPTIVE_TOXIC_WINDOW:
    final_output_path = output_path
    output_path = Path("data") / "datasets" / f"raw_dataset_{FILE_NAME}.parquet"
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


def _process_adaptive_toxic_window(raw_output_path: Path, final_output_path: Path) -> None:
    """
    Second segment: Use MarkoutAnalyzer to determine optimal window, then label filled orders.

    This runs only if ADAPTIVE_TOXIC_WINDOW is True and the raw data file exists.

    Args:
        raw_output_path: Path to the raw dataset created in the first segment
        final_output_path: Path to write the final labeled dataset
    """
    if not raw_output_path.exists():
        print(f"[adaptive_window] Raw data file not found at {raw_output_path}")
        return

    print(f"[adaptive_window] Loading raw dataset from {raw_output_path}...")
    df_raw = pd.read_parquet(raw_output_path)

    # Filter for filled orders to compute markouts
    filled_mask = df_raw["event"] == 1
    df_filled = df_raw[filled_mask].copy()

    if len(df_filled) == 0:
        print("[adaptive_window] No filled orders found in raw dataset")
        return

    print(f"[adaptive_window] Found {len(df_filled)} filled orders")

    # Initialize MarkoutAnalyzer with StabilizationWindowSelector
    window_selector = StabilizationWindowSelector()
    analyzer = MarkoutAnalyzer(window_selector=window_selector, winsorize=True)

    # Analyze markouts and select optimal window
    print("[adaptive_window] Analyzing markouts and selecting optimal window...")
    result = analyzer.analyze(df_filled)

    selected_window_result = result["selected_window"]

    # Print selection results
    print(f"[adaptive_window] Window selection result: {selected_window_result}")

    if selected_window_result.get("found"):
        selected_window_ms = selected_window_result["chosen_window_ms"]
        print(f"[adaptive_window] Selected window: {selected_window_ms} ms")
    else:
        print(f"[adaptive_window] Window selection failed: {selected_window_result.get('reason')}")
        selected_window_ms = None

    # Create a new labeler with the selected window
    print("[adaptive_window] Re-labeling filled orders with selected window...")
    labeler = ExecutionCompetingRisksLabeler(selected_window=selected_window_ms)

    # Apply labeling to all records in the raw dataset
    records = []
    for idx, row in df_raw.iterrows():
        record = {
            "status_reason": row.get("status_reason", "UNKNOWN"),
            "duration_s": row.get("duration_s", 0.0),
            "price": row.get("price"),
            "side": row.get("side"),
            "best_bid_at_entry": row.get("best_bid_at_entry"),
            "best_ask_at_entry": row.get("best_ask_at_entry"),
        }

        # Add post-trade BBO fields (legacy single window)
        record["best_bid_at_post_trade"] = row.get("best_bid_at_post_trade")
        record["best_ask_at_post_trade"] = row.get("best_ask_at_post_trade")

        # Add multi-window post-trade fields if they exist
        for col in df_raw.columns:
            if col.startswith("post_trade_best_"):
                record[col] = row.get(col)

        try:
            label_result = labeler.label(record)
            row_dict = row.to_dict()
            row_dict["event_type"] = label_result.get("event_type")
            row_dict["event_time_bin"] = label_result.get("event_time_bin")

            # Merge extras into the record
            extras = label_result.get("extras", {})
            if isinstance(extras, dict):
                row_dict.update(extras)

            records.append(row_dict)
        except Exception as e:
            print(f"[adaptive_window] Error labeling record {idx}: {e}")
            records.append(row.to_dict())

    df_labeled = pd.DataFrame(records)

    print(f"[adaptive_window] Writing {len(df_labeled)} records to {final_output_path}...")
    df_labeled.to_parquet(final_output_path)

    # Print final statistics
    parquet_file = pq.ParquetFile(final_output_path)
    columns = parquet_file.schema.names
    shape = (parquet_file.metadata.num_rows, len(columns))

    print(f"[adaptive_window] Final dataset shape: {shape}")
    print("[adaptive_window] Event type distribution:")

    if "event_type" in columns:
        event_col = pq.read_table(final_output_path, columns=["event_type"])["event_type"]
        counts = pc.value_counts(event_col).to_pylist()
        counts_sorted = sorted(
            counts,
            key=lambda x: (x["values"] is None, x["values"]),
        )
        for row in counts_sorted:
            print(f"  {row['values']}: {row['counts']}")

    print("[adaptive_window] Second segment complete")



def main() -> None:
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

    # Check if raw data already exists and we should skip the first segment
    if SKIP_FIRST_SEGMENT_IF_RAW_DATA_EXISTS and output_path.exists():
        print(f"[main] Raw data already exists at {output_path}")
        print(f"[main] Skipping first segment (SKIP_FIRST_SEGMENT_IF_RAW_DATA_EXISTS=True)")
    else:
        # First segment: process raw data from DBN file
        print("[main] Starting first segment: processing raw data from DBN file...")

        tracker = OrderTracker(
            samples_per_day=SAMPLES_PER_DAY,
            time_censor_s=TIME_CENSOR_S,
            lookback_period=LOOKBACK_PERIOD,
            random_seed=RANDOM_SEED,
            raw_data_mode=ADAPTIVE_TOXIC_WINDOW
        )

        if N_WORKERS > 1:
            print(f"Running in parallel mode with {N_WORKERS} workers.")

            split_cache = _load_or_build_split_cache(split_cache_path, dbn_path)
            empty_points = split_cache.get("split_points", [])
            messages_between_splits = split_cache.get("messages_between_splits")
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

        if not ADAPTIVE_TOXIC_WINDOW:
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

    # Second segment: adaptive toxic window labeling
    if ADAPTIVE_TOXIC_WINDOW:
        if not output_path.exists():
            print(f"\n[main] Skipping adaptive window segment: raw data file not found at {output_path}")
        else:
            print("\n[main] Starting second segment: adaptive toxic window labeling...")
            _process_adaptive_toxic_window(output_path, final_output_path)


if __name__ == "__main__":
    main()
