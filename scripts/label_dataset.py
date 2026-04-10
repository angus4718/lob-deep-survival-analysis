"""Apply adaptive toxic window labeling to a raw dataset."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pandas as pd

# Ensure project root is in sys.path for src imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.labeling.window_selecting import MarkoutAnalyzer, StabilizationWindowSelector
from src.labeling.competing_risks import ExecutionCompetingRisksLabeler


DEFAULT_TICKER = "AAPL"
DEFAULT_START_DATE = "2025-10-01"
DEFAULT_END_DATE = "2026-01-01"
DEFAULT_DATASETS_DIR = Path("/ocean/projects/cis260122p/shared/data/datasets")


def _dataset_paths(
    ticker: str, start_date: str, end_date: str, datasets_dir: Path
) -> tuple[Path, Path]:
    """Build input/output dataset paths from ticker and date bounds."""
    start_ymd = start_date.replace("-", "")
    end_ymd = end_date.replace("-", "")
    raw_path = (
        datasets_dir
        / f"raw_dataset_XNAS_ITCH_{ticker}_mbo_{start_ymd}_{end_ymd}.parquet"
    )
    labeled_path = (
        datasets_dir
        / f"labeled_dataset_XNAS_ITCH_{ticker}_mbo_{start_ymd}_{end_ymd}.parquet"
    )
    return raw_path, labeled_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply adaptive toxic window labeling to a raw dataset."
    )
    parser.add_argument(
        "--ticker",
        default=os.getenv("SYMBOL", DEFAULT_TICKER),
        help="Ticker symbol used in dataset filename (default: env SYMBOL or AAPL).",
    )
    parser.add_argument(
        "--start-date",
        default=os.getenv("START_DATE", DEFAULT_START_DATE),
        help="Start date in YYYY-MM-DD format (default: env START_DATE or 2025-10-01).",
    )
    parser.add_argument(
        "--end-date",
        default=os.getenv("END_DATE", DEFAULT_END_DATE),
        help="End date in YYYY-MM-DD format (default: env END_DATE or 2026-01-01).",
    )
    parser.add_argument(
        "--datasets-dir",
        default=os.getenv("DATASETS_DIR", str(DEFAULT_DATASETS_DIR)),
        help=(
            "Directory containing raw/labeled dataset parquet files "
            "(default: env DATASETS_DIR or shared datasets path)."
        ),
    )
    return parser.parse_args()


def _compute_day_boundary_splits(df_raw: pd.DataFrame):
    """Compute train/val/test masks using day-boundary cuts."""
    entry_ns = df_raw["entry_time"].values
    dates = (
        pd.to_datetime(entry_ns, unit="ns", utc=True)
        .tz_convert("America/New_York")
        .normalize()
    )

    unique_days = sorted(dates.unique())
    n_days = len(unique_days)
    if n_days < 3:
        raise ValueError(
            "Day-boundary split requires at least 3 trading days to produce train, val, and test partitions."
        )

    n = len(df_raw)
    target_train_end = int(n * 0.70)
    target_val_end = int(n * 0.85)

    day_end_idx = [(dates <= d).sum() - 1 for d in unique_days]

    def _best_day_cut(target_row: int) -> int:
        """Return the index of the day whose END row is closest to target_row."""
        return min(range(n_days), key=lambda i: abs(day_end_idx[i] - target_row))

    train_day_idx = _best_day_cut(target_train_end)
    val_day_idx = _best_day_cut(target_val_end)
    train_day_idx = min(train_day_idx, n_days - 3)
    val_day_idx = max(train_day_idx + 1, min(val_day_idx, n_days - 2))

    train_end = day_end_idx[train_day_idx] + 1
    val_end = day_end_idx[val_day_idx] + 1

    idx = np.arange(n)
    train_mask = idx < train_end
    val_mask = (idx >= train_end) & (idx < val_end)
    test_mask = idx >= val_end

    train_days = unique_days[: train_day_idx + 1]
    val_days = unique_days[train_day_idx + 1 : val_day_idx + 1]
    test_days = unique_days[val_day_idx + 1 :]

    return train_mask, val_mask, test_mask, train_days, val_days, test_days


def label_dataset_adaptive_toxic_window(
    raw_output_path: Path, final_output_path: Path
) -> None:
    """
    Apply adaptive toxic window labeling to a raw dataset.

    Uses MarkoutAnalyzer to determine optimal window from filled orders (first 70% by day boundary),
    then labels ALL orders (100%) in the dataset with that calibrated window.

    Args:
        raw_output_path: Path to the raw dataset (from dataset generation)
        final_output_path: Path to write the final labeled dataset
    """
    if not raw_output_path.exists():
        print(f"[label] Raw data file not found at {raw_output_path}")
        return

    print(f"[label] Loading raw dataset from {raw_output_path}...")
    df_raw = pd.read_parquet(raw_output_path)

    print("[label] Applying full temporal split (70/85% by day boundary)...")
    train_mask, val_mask, test_mask, train_days, val_days, test_days = (
        _compute_day_boundary_splits(df_raw)
    )

    df_train = df_raw[train_mask].copy()
    df_val = df_raw[val_mask].copy()
    df_test = df_raw[test_mask].copy()

    print("[label] Using full temporal splits without train/val subsampling:")
    print(f"  Train samples: {len(df_train):,} over {len(train_days)} day(s)")
    print(f"  Val samples  : {len(df_val):,} over {len(val_days)} day(s)")
    print(f"  Test samples : {len(df_test):,} over {len(test_days)} day(s)")
    print(f"[label] Full dataset for labeling: {len(df_raw):,} records")

    # Filter for filled orders in training set to compute markouts
    filled_mask = df_train["event"] == 1
    df_filled = df_train[filled_mask].copy()

    if len(df_filled) == 0:
        print("[label] No filled orders found in raw dataset")
        return

    print(f"[label] Found {len(df_filled)} filled orders")

    # Initialize MarkoutAnalyzer with StabilizationWindowSelector
    window_selector = StabilizationWindowSelector()
    analyzer = MarkoutAnalyzer(window_selector=window_selector, winsorize=True)

    # Analyze markouts and select optimal window
    print("[label] Analyzing markouts and selecting optimal window...")
    result = analyzer.analyze(df_filled)

    selected_window_result = result["selected_window"]

    # Print selection results
    print(f"[label] Window selection result: {selected_window_result}")

    if selected_window_result.get("found"):
        selected_window_ms = selected_window_result["chosen_window_ms"]
        print(f"[label] Selected window: {selected_window_ms} ms")
    else:
        print(
            f"[label] Window selection failed: {selected_window_result.get('reason')}"
        )
        selected_window_ms = None

    # Create a new labeler with the selected window
    print("[label] Labeling all data points with calibrated window...")
    labeler = ExecutionCompetingRisksLabeler(selected_window=selected_window_ms)

    # Apply labeling to ALL records in the raw dataset (100%, not just training 70%)
    records = []
    for idx, row in df_raw.iterrows():
        record = {
            "status_reason": row.get("status_reason", "UNKNOWN"),
            "duration_s": row.get("duration_s", 0.0),
            "price": row.get("price"),
            "side": row.get("side"),
            "best_bid_at_entry": row.get("best_bid_at_entry"),
            "best_ask_at_entry": row.get("best_ask_at_entry"),
            "best_bid_at_execution": row.get("best_bid_at_execution"),
            "best_ask_at_execution": row.get("best_ask_at_execution"),
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

            # Merge extras into the record
            extras = label_result.get("extras", {})
            if isinstance(extras, dict):
                row_dict.update(extras)

            records.append(row_dict)
        except Exception as e:
            print(f"[label] Error labeling record {idx}: {e}")
            records.append(row.to_dict())

    df_labeled = pd.DataFrame(records)

    print(f"[label] Writing {len(df_labeled)} records to {final_output_path}...")
    df_labeled.to_parquet(final_output_path)

    # Print final statistics
    parquet_file = pq.ParquetFile(final_output_path)
    columns = parquet_file.schema.names
    shape = (parquet_file.metadata.num_rows, len(columns))

    print(f"[label] Final dataset shape: {shape}")
    print("[label] Event type distribution:")

    if "event_type" in columns:
        event_col = pq.read_table(final_output_path, columns=["event_type"])[
            "event_type"
        ]
        counts = pc.value_counts(event_col).to_pylist()
        counts_sorted = sorted(
            counts,
            key=lambda x: (x["values"] is None, x["values"]),
        )
        for row in counts_sorted:
            print(f"  {row['values']}: {row['counts']}")

    print("[label] Labeling complete")


def main() -> None:
    """Apply adaptive toxic window labeling with optional CLI overrides."""
    args = _parse_args()
    datasets_dir = Path(args.datasets_dir)
    raw_path, output_path = _dataset_paths(
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        datasets_dir=datasets_dir,
    )
    label_dataset_adaptive_toxic_window(raw_path, output_path)


if __name__ == "__main__":
    main()
