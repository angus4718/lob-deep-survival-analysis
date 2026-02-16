from __future__ import annotations
import os
import sys
from pathlib import Path

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

data_path = str(Path("data") / "raw" / f"{FILE_NAME}.dbn.zst")
output_parquet = f"data/simulated_orders/simulated_orders_{FILE_NAME}.parquet"

if not os.path.exists(data_path):
    print(f"File {data_path} not found.")
else:
    tracker = OrderTracker(samples_per_day=SAMPLES_PER_DAY)
    tracker.process_stream(data_path, output_parquet, limit=None)
