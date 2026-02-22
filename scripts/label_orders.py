"""
Post-process raw order tracking output with event labeling.

Usage:
    python scripts/label_orders.py \
        --input data/simulated_orders/simulated_orders_*.parquet \
        --output data/labeled_orders/labeled_orders_*.parquet
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.labeling.dataset import SurvivalDataset
from src.labeling.competing_risks import ExecutionCompetingRisksLabeler


def main():
    parser = argparse.ArgumentParser(
        description="Label raw order tracking output with competing risk event types."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to raw Parquet file from OrderTracker.process_stream",
    )
    parser.add_argument(
        "--output",
        required=False,
        help="Optional path to save labeled Parquet file",
    )
    
    args = parser.parse_args()
    
    input_path = args.input
    output_path = args.output
    
    # Check if input exists
    if not Path(input_path).exists():
        print(f"Error: Input file {input_path} not found.")
        sys.exit(1)
    
    # Create dataset with default labeler
    labeler = ExecutionCompetingRisksLabeler()
    dataset = SurvivalDataset(labeler=labeler)
    
    # Load and label
    print(f"Loading orders from {input_path}...")
    df = dataset.load_and_label(input_path, output_parquet_path=output_path)
    
    # Print summary
    print(f"\nDataset shape: {df.shape}")
    print(f"\nEvent type distribution:")
    print(dataset.get_event_distribution(df))
    print(f"\nTime bin statistics:")
    print(dataset.get_time_bin_stats(df))
    
    if not output_path:
        print(f"\nTo save labeled output, use --output flag.")


if __name__ == "__main__":
    main()
