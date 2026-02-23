from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

# Ensure project root is in sys.path for src imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.order_tracking import OrderTracker

SYMBOL = "AAPL"
START_DATE = "2025-12-01"
END_DATE = "2026-01-01"

SAMPLES_PER_DAY = 100
TIME_CENSOR_S = 300.0
LOOKBACK_PERIOD = 20
PROGRESS_INTERVAL = 100_000

# Set to a "YYYY-MM-DD" string to restrict processing to a single trading day,
# or None to process the entire file.
TARGET_DAY = None  # e.g. "2025-12-01"

# Number of parallel worker processes.
N_WORKERS = 20

FILE_NAME = (
    f"XNAS_ITCH_{SYMBOL}_mbo_{START_DATE.replace('-', '')}_{END_DATE.replace('-', '')}"
)
dbn_path = Path("data") / "raw" / f"{FILE_NAME}.dbn.zst"
output_path = Path("data") / "datasets" / f"dataset_{FILE_NAME}.parquet"


def main() -> None:
    if not os.path.exists(dbn_path):
        print(f"File {dbn_path} not found.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    tracker = OrderTracker(
        samples_per_day=SAMPLES_PER_DAY,
        time_censor_s=TIME_CENSOR_S,
        lookback_period=LOOKBACK_PERIOD,
    )

    if N_WORKERS > 1:
        print(f"Running in parallel mode with {N_WORKERS} workers.")
        tracker.process_stream_parallel(
            file_path=str(dbn_path),
            output_parquet=str(output_path),
            n_workers=N_WORKERS,
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

    df = pd.read_parquet(output_path)
    print("shape:", df.shape)
    print("columns:", list(df.columns))

    print("\nEvent type distribution:")
    print(df["event_type"].value_counts(dropna=False).sort_index())


if __name__ == "__main__":
    main()
"""
[parallel] Found 28 empty-market transitions.
[parallel] 17 chunk(s). Split timestamps (UTC ns): [1764749102426337186, 1764835502538123401, 1764921902657087068, 1765181105923313585, 1765267501489234482, 1765353897623140716, 1765526701745496793, 1765785892856718339, 1765872295424774199, 1766045092238363115, 1766134800131914751, 1766390703422316446, 1766563491896459827, 1766613600229985855, 1766995503018619276, 1767081888103273843]
[parallel] Merged 17 chunk(s): 2,200 rows written to data\datasets\dataset_XNAS_ITCH_AAPL_mbo_20251201_20260101.parquet
[parallel] Total: scheduled_virtual=2200, spawned_virtual=2200
shape: (2200, 19)
columns: ['order_id', 'entry_time', 'duration_s', 'event', 'status_reason', 'price', 'side', 'volume', 'order_type', 'best_bid_at_entry', 'best_ask_at_entry', 'best_bid_at_post_trade', 'best_ask_at_post_trade', 'entry_representation', 'event_type', 'event_time_bin', 'post_trade_adverse_move_bps', 'post_trade_spread_bps', 'post_trade_recorded']

Event type distribution:
event_type
0      49
1    2001
2     133
3      17
Name: count, dtype: int64
"""
