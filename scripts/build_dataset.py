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
FILE_NAME = (
    f"XNAS_ITCH_{SYMBOL}_mbo_{START_DATE.replace('-', '')}_{END_DATE.replace('-', '')}"
)
SAMPLES_PER_DAY = 100
TIME_CENSOR_S = 300.0
PROGRESS_INTERVAL = 1_000_000

dbn_path = Path("data") / "raw" / f"{FILE_NAME}.dbn.zst"
output_path = Path("data") / "datasets" / f"dataset_{FILE_NAME}.parquet"

if not os.path.exists(dbn_path):
    print(f"File {dbn_path} not found.")
else:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tracker = OrderTracker(
        samples_per_day=SAMPLES_PER_DAY,
        time_censor_s=TIME_CENSOR_S,
    )

    tracker.process_stream(
        file_path=str(dbn_path),
        output_parquet=str(output_path),
        limit=None,
        progress_interval=PROGRESS_INTERVAL,
        samples_per_day=SAMPLES_PER_DAY,
    )

    df = pd.read_parquet(output_path)
    print("shape:", df.shape)
    print("columns:", list(df.columns))

    print("\nEvent type distribution:")
    print(df["event_type"].value_counts(dropna=False).sort_index())
