"""
Script to download historical market data from Databento and convert to parquet.
If the data file already exists locally, it loads from the file instead of downloading again.

Create a .env file in the root of your project with the following content:
DATABENTO_API_KEY=<your_api_key_here>
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import databento as db

load_dotenv()

api_key = os.getenv("DATABENTO_API_KEY")
if not api_key:
    raise ValueError("DATABENTO_API_KEY not found in environment variables.")

SYMBOL = "AAPL"
START_DATE = "2025-12-01"
END_DATE = "2026-01-01"
FILE_NAME = (
    f"XNAS_ITCH_{SYMBOL}_mbo_{START_DATE.replace('-', '')}_{END_DATE.replace('-', '')}"
)
NO_DOWNLOAD_LOCK = True  # Safety flag to prevent accidental downloads

client = db.Historical(api_key)
data_path = str(Path("data") / "raw" / f"{FILE_NAME}.dbn.zst")
if Path(data_path).exists():
    print("Loading data from local file...")
    data = db.DBNStore.from_file(data_path)
elif not NO_DOWNLOAD_LOCK:
    print("Downloading data from Databento...")
    data = client.timeseries.get_range(
        dataset="XNAS.ITCH",
        start=START_DATE,
        end=END_DATE,
        symbols=SYMBOL,
        schema="mbo",
        path=data_path,
    )
else:
    raise RuntimeError(
        "Data file not found locally and NO_DOWNLOAD_LOCK is set. Aborting download."
    )
