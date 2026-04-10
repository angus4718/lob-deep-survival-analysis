"""
Merge multiple survival analysis datasets (Parquet files).

This script combines multiple labeled dataset Parquet files into a single merged
dataset, with optional statistics reporting.

Configuration:
    PARQUET_FILES: List of Parquet file paths to merge.
    OUTPUT_PATH: Output path for merged dataset (optional).
    VERBOSE: Print progress messages.

Usage:
    Modify the PARQUET_FILES variable, then run:
        python merge_datasets.py

Examples:
    # Merge two datasets
    PARQUET_FILES = [
        "data/datasets/dataset_XNAS_ITCH_AAPL_mbo_20241001_20241031.parquet",
        "data/datasets/dataset_XNAS_ITCH_AAPL_mbo_20241101_20241130.parquet",
    ]
    OUTPUT_PATH = "data/datasets/merged.parquet"

    # Merge without saving
    PARQUET_FILES = ["file1.parquet", "file2.parquet"]
    OUTPUT_PATH = None
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

# Ensure project root is in sys.path for src imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.domain.enums import EventType

# ============================================================================
# USER CONFIGURATION
# ============================================================================

TICKER = "GILD"  # Set the ticker symbol for the dataset being merged

# List of Parquet files to merge
PARQUET_FILES: list[str] = [
    f"/ocean/projects/cis260122p/shared/data/datasets/dataset_XNAS_ITCH_{TICKER}_mbo_20251001_20251101.parquet",
    f"/ocean/projects/cis260122p/shared/data/datasets/dataset_XNAS_ITCH_{TICKER}_mbo_20251101_20251201.parquet",
    f"/ocean/projects/cis260122p/shared/data/datasets/dataset_XNAS_ITCH_{TICKER}_mbo_20251201_20260101.parquet",
]

# Output path for merged dataset (optional)
OUTPUT_PATH: str | None = (
    f"/ocean/projects/cis260122p/shared/data/datasets/raw_dataset_XNAS_ITCH_{TICKER}_mbo_20251001_20260101.parquet"
)

# Print progress messages
VERBOSE: bool = True


def load_datasets(
    parquet_files: list[Path], verbose: bool = True
) -> list[pd.DataFrame]:
    """
    Load multiple Parquet files into memory.

    Args:
        parquet_files: List of paths to Parquet files.
        verbose: Print loading progress.

    Returns:
        List of DataFrames.
    """
    dfs = []
    for i, file in enumerate(parquet_files, 1):
        try:
            df = pd.read_parquet(file)
            if verbose:
                print(
                    f"[{i}/{len(parquet_files)}] Loaded {file.name}: {len(df):,} rows"
                )
            dfs.append(df)
        except Exception as e:
            print(f"Error loading {file.name}: {e}", file=sys.stderr)
            raise

    return dfs


def merge_datasets(dfs: list[pd.DataFrame], verbose: bool = True) -> pd.DataFrame:
    """
    Merge multiple DataFrames into a single DataFrame.

    Args:
        dfs: List of DataFrames to merge.
        verbose: Print merge progress.

    Returns:
        Merged DataFrame with index reset.
    """
    if not dfs:
        raise ValueError("No DataFrames to merge")

    if len(dfs) == 1:
        merged = dfs[0].reset_index(drop=True)
        if verbose:
            print("Single dataset - no merging needed")
    else:
        merged = pd.concat(dfs, ignore_index=True, sort=True)
        if verbose:
            print(f"Merged {len(dfs)} datasets: {len(merged):,} total rows")

    return merged


def get_event_statistics(df: pd.DataFrame) -> dict:
    """
    Get event type distribution statistics.

    Args:
        df: Labeled dataset DataFrame.

    Returns:
        Dictionary with event statistics.
    """
    stats = {}
    if "event_type" in df.columns:
        event_counts = df["event_type"].value_counts().sort_index()
        for event_code, count in event_counts.items():
            try:
                event_name = EventType(event_code).name
            except ValueError:
                event_name = f"UNKNOWN({event_code})"
            stats[event_name] = int(count)
    return stats


def print_merge_statistics(
    dfs_info: list[dict],
    merged_df: pd.DataFrame,
    prefix: str = "",
) -> None:
    """
    Print detailed statistics about merged datasets.

    Args:
        dfs_info: List of dicts with dataset names and row counts.
        merged_df: Merged DataFrame.
        prefix: String prefix for output lines.
    """
    print(f"\n{prefix}{'=' * 70}")
    print(f"{prefix}MERGE STATISTICS")
    print(f"{prefix}{'=' * 70}")

    # Input datasets
    print(f"\n{prefix}Input Datasets ({len(dfs_info)}):")
    total_input_rows = 0
    for info in dfs_info:
        print(f"{prefix}  - {info['name']}: {info['rows']:,} rows")
        total_input_rows += info["rows"]

    # Merged dataset
    print(f"\n{prefix}Merged Dataset:")
    print(f"{prefix}  - Total rows: {len(merged_df):,}")
    print(f"{prefix}  - Columns: {len(merged_df.columns)}")
    print(f"{prefix}  - Column names: {list(merged_df.columns)}")

    # Data types
    if len(set(str(dt) for dt in merged_df.dtypes.values)) < 10:
        print(f"\n{prefix}Data types:")
        for col, dtype in merged_df.dtypes.items():
            print(f"{prefix}  - {col}: {dtype}")

    # Event statistics
    if "event_type" in merged_df.columns:
        event_stats = get_event_statistics(merged_df)
        print(f"\n{prefix}Event Type Distribution:")
        for event_name, count in sorted(event_stats.items()):
            pct = 100 * count / len(merged_df)
            print(f"{prefix}  - {event_name}: {count:,} ({pct:.2f}%)")

    # Duration statistics
    if "duration_s" in merged_df.columns:
        print(f"\n{prefix}Duration Statistics (seconds):")
        duration_stats = merged_df["duration_s"].describe()
        for stat_name in ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]:
            value = duration_stats[stat_name]
            print(f"{prefix}  - {stat_name}: {value:,.3f}")

    print(f"\n{prefix}{'=' * 70}\n")


def merge_parquet_datasets(
    parquet_files: list[Path],
    output_path: Optional[Path] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Merge multiple Parquet files into a single dataset.

    Args:
        parquet_files: List of paths to Parquet files to merge.
        output_path: Path to save merged dataset. If None, files are not saved.
        verbose: Print progress messages.

    Returns:
        Merged DataFrame.

    Raises:
        FileNotFoundError: If any input file doesn't exist.
        ValueError: If the file list is empty.
    """
    if not parquet_files:
        raise ValueError("No Parquet files provided to merge.")

    # Validate all files exist
    for file in parquet_files:
        if not file.exists():
            raise FileNotFoundError(f"Parquet file not found: {file}")

    if verbose:
        print(f"Merging {len(parquet_files)} dataset(s)...")
        for f in parquet_files:
            print(f"  - {f.name}")

    # Load datasets
    dfs = load_datasets(parquet_files, verbose=verbose)

    # Track input info
    dfs_info = [{"name": f.name, "rows": len(df)} for f, df in zip(parquet_files, dfs)]

    # Merge
    merged_df = merge_datasets(dfs, verbose=verbose)

    # Print statistics
    if verbose:
        print_merge_statistics(dfs_info, merged_df)

    # Save
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        merged_df.to_parquet(output_path, index=False)
        if verbose:
            print(f"✓ Merged dataset saved to {output_path}")

    return merged_df


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Merge multiple survival analysis datasets (Parquet files)."
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default=TICKER,
        help=f"Ticker symbol for the dataset being merged (default: {TICKER})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=VERBOSE,
        help="Print verbose progress messages",
    )
    args = parser.parse_args()

    try:
        # Construct parquet files and output path using the ticker
        parquet_files = [
            f"/ocean/projects/cis260122p/shared/data/datasets/dataset_XNAS_ITCH_{args.ticker}_mbo_20251001_20251101.parquet",
            f"/ocean/projects/cis260122p/shared/data/datasets/dataset_XNAS_ITCH_{args.ticker}_mbo_20251101_20251201.parquet",
            f"/ocean/projects/cis260122p/shared/data/datasets/dataset_XNAS_ITCH_{args.ticker}_mbo_20251201_20260101.parquet",
        ]
        output_path = f"/ocean/projects/cis260122p/shared/data/datasets/raw_dataset_XNAS_ITCH_{args.ticker}_mbo_20251001_20260101.parquet"

        if not parquet_files:
            raise ValueError("PARQUET_FILES is empty. Set the files to merge.")

        parquet_paths = [Path(f) for f in parquet_files]
        output_path_obj = Path(output_path) if output_path else None

        merge_parquet_datasets(
            parquet_files=parquet_paths,
            output_path=output_path_obj,
            verbose=args.verbose,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
