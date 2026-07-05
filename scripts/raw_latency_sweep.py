from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest import BacktestDataset, RawDatabentoBacktestEngine, StaticLatencyProvider
from src.backtest.strategies import DeepHitStrategy, DeepHitThresholdDecisionLogic


DATASET_PATH = "data/datasets/labeled_dataset_XNAS_ITCH_AAPL_mbo_20251001_20260101_test.parquet"
RUNTIME_NPZ_PATH = "data/datasets/labeled_dataset_XNAS_ITCH_AAPL_mbo_20251001_20260101_dynamic_preprocessed.npz"
MODEL_PATH = "checkpoints/dynamic_deephit_gru_transformer_AAPL_best_epoch_base_net_i1xfzhr5.pt"
RAW_PATH = "data/raw/XNAS_ITCH_AAPL_mbo_20251001_20251101.dbn.zst"
SPLIT_CACHE_PATH = "data/raw/XNAS_ITCH_AAPL_mbo_20251001_20251101.split_points.json"

SAMPLE_FRACTION = 0.1
LIFECYCLE_AWARE = True
LIFECYCLE_STRIDE = 50
SNAPSHOT_BIN_MESSAGES = 15
OUTPUT_SUMMARY_PATH = "reports/raw_latency_sweep_summary.csv"
OUTPUT_RAW_DIR = "reports/raw_latency_sweep"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a parallel raw backtest latency sweep."
    )
    parser.add_argument(
        "--lifecycle-aware",
        action=argparse.BooleanOptionalAction,
        default=LIFECYCLE_AWARE,
        help="Enable lifecycle-aware cancel decisions during raw replay.",
    )
    parser.add_argument(
        "--output-summary",
        default=OUTPUT_SUMMARY_PATH,
        help="CSV path for concatenated latency summaries.",
    )
    parser.add_argument(
        "--raw-output-dir",
        default=OUTPUT_RAW_DIR,
        help="Directory for per-latency raw backtest parquet outputs.",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help="Parallel worker count. Defaults to raw engine's split/core heuristic.",
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=SAMPLE_FRACTION,
        help="Random subsample fraction for the labeled test dataset.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable verbose/progress output from raw replay.",
    )
    return parser.parse_args()


def latency_grid_ms() -> list[float]:
    positive = np.geomspace(0.01, 50.0, 14)
    values = [0.0] + [round(float(v), 2) for v in positive]
    return sorted(dict.fromkeys(values))


def make_dataset(sample_fraction: float) -> BacktestDataset:
    return BacktestDataset.from_runtime_artifacts(
        DATASET_PATH,
        runtime_npz_path=RUNTIME_NPZ_PATH,
        snapshot_policy="entry",
        sample_fraction=sample_fraction,
    )


def make_strategy() -> DeepHitStrategy:
    return DeepHitStrategy.from_saved_model(
        MODEL_PATH,
        model_name="gru_transformer",
        decision_logic=DeepHitThresholdDecisionLogic(
            max_toxic_probability=0.28,
            min_fill_probability=0.6,
            horizon_index=9,
        ),
    )


def run_one_latency(
    *,
    dataset: BacktestDataset,
    strategy: DeepHitStrategy,
    latency_ms: float,
    lifecycle_aware: bool,
    n_workers: int | None,
    raw_output_dir: Path,
    verbose: bool,
) -> pd.DataFrame:
    latency_token = f"{latency_ms:.6f}".rstrip("0").rstrip(".").replace(".", "p")
    raw_path = raw_output_dir / f"raw_latency_{latency_token}ms_backtest.parquet"
    summary_path = raw_output_dir / f"raw_latency_{latency_token}ms_summary.csv"

    engine = RawDatabentoBacktestEngine(
        strategy,
        raw_path=RAW_PATH,
        latency_provider=StaticLatencyProvider.from_milliseconds(latency_ms),
        snapshot_bin_messages=SNAPSHOT_BIN_MESSAGES,
        lifecycle_aware=lifecycle_aware,
        lifecycle_stride=LIFECYCLE_STRIDE,
        verbose=verbose,
        progress=verbose,
    )
    report = engine.run_parallel(
        dataset,
        n_workers=n_workers,
        split_cache_path=SPLIT_CACHE_PATH,
    )
    report.write(raw_path, summary_path)

    summary = report.summary_frame()
    summary.insert(0, "latency_ms", float(latency_ms))
    summary.insert(1, "lifecycle_aware", bool(lifecycle_aware))
    summary.insert(2, "raw_backtest_path", str(raw_path))
    return summary


def main() -> None:
    args = parse_args()
    raw_output_dir = Path(args.raw_output_dir)
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    output_summary = Path(args.output_summary)
    output_summary.parent.mkdir(parents=True, exist_ok=True)

    dataset = make_dataset(args.sample_fraction)
    strategy = make_strategy()
    verbose = not args.quiet

    summaries = []
    for latency_ms in latency_grid_ms():
        if verbose:
            print(
                f"\n[latency-sweep] Running {latency_ms:.6g} ms "
                f"(lifecycle_aware={args.lifecycle_aware})"
            )
        summaries.append(
            run_one_latency(
                dataset=dataset,
                strategy=strategy,
                latency_ms=latency_ms,
                lifecycle_aware=args.lifecycle_aware,
                n_workers=args.n_workers,
                raw_output_dir=raw_output_dir,
                verbose=verbose,
            )
        )

    combined = pd.concat(summaries, ignore_index=True)
    combined.to_csv(output_summary, index=False)
    print(f"\n[latency-sweep] Wrote {len(combined)} rows to {output_summary}")


if __name__ == "__main__":
    main()
