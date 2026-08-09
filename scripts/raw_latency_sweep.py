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


DATA_START_DATE = "2025-10-01"
DATA_END_DATE = "2026-01-01"
RAW_END_DATE = "2025-11-01"

DATA_ROOT = "data"
ARTIFACT_ROOT = "checkpoints"
OUTPUT_ROOT = "reports/raw_latency_sweep"

SAMPLE_FRACTION = 0.1
LIFECYCLE_AWARE = True
LIFECYCLE_STRIDE = 50
SNAPSHOT_BIN_MESSAGES = 15
MAX_TOXIC_PROBABILITY = 0.7
MIN_FILL_PROBABILITY = 0.3
HORIZON_INDEX = 9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a raw Databento latency sweep for one ticker/model pair. "
            "By default this uses sequential chunked replay so GPU inference can "
            "remain in the main process while non-overlapping raw split chunks are skipped."
        )
    )
    parser.add_argument("--ticker", default="AAPL", help="Ticker symbol, e.g. AAPL.")
    parser.add_argument(
        "--model-type",
        default="mamba",
        help="DeepHit architecture/model type. Model file defaults to [model_type]_[ticker].pt.",
    )
    parser.add_argument(
        "--mode",
        choices=("sequential", "parallel"),
        default="sequential",
        help=(
            "sequential: chunked single-process raw replay, GPU-safe. "
            "parallel: process-pool raw replay, CPU-safe for CUDA strategies."
        ),
    )
    parser.add_argument(
        "--data-root",
        default=DATA_ROOT,
        help=(
            "Root containing datasets/ and raw/."
        ),
    )
    parser.add_argument(
        "--artifact-root",
        default=ARTIFACT_ROOT,
        help=(
            "Root containing model artifacts. The default model path is "
            "[artifact_root]/[ticker]/[model_type]_[ticker].pt."
        ),
    )
    parser.add_argument(
        "--output-root",
        default=OUTPUT_ROOT,
        help="Directory root for output parquet/CSV files.",
    )
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Optional explicit labeled test parquet path.",
    )
    parser.add_argument(
        "--runtime-npz-path",
        default=None,
        help="Optional explicit dynamic preprocessed runtime NPZ path.",
    )
    parser.add_argument("--raw-path", default=None, help="Optional explicit raw DBN path.")
    parser.add_argument(
        "--split-cache-path",
        default=None,
        help="Optional explicit raw split cache path.",
    )
    parser.add_argument("--model-path", default=None, help="Optional explicit model PT path.")
    parser.add_argument(
        "--data-start-date",
        default=DATA_START_DATE,
        help="Dataset/raw start date shared across tickers.",
    )
    parser.add_argument(
        "--data-end-date",
        default=DATA_END_DATE,
        help="Dataset end date shared across tickers.",
    )
    parser.add_argument(
        "--raw-end-date",
        default=RAW_END_DATE,
        help="Raw DBN end date shared across tickers.",
    )
    parser.add_argument(
        "--lifecycle-aware",
        action=argparse.BooleanOptionalAction,
        default=LIFECYCLE_AWARE,
        help="Enable lifecycle-aware cancel decisions during raw replay.",
    )
    parser.add_argument(
        "--lifecycle-stride",
        type=int,
        default=LIFECYCLE_STRIDE,
        help="Lifecycle decision stride in rebuilt raw snapshots.",
    )
    parser.add_argument(
        "--lifecycle-max-evaluations",
        type=int,
        default=None,
        help="Optional cap on lifecycle decisions per order.",
    )
    parser.add_argument(
        "--snapshot-bin-messages",
        type=int,
        default=SNAPSHOT_BIN_MESSAGES,
        help="Raw messages per rebuilt snapshot.",
    )
    parser.add_argument(
        "--max-toxic-probability",
        type=float,
        default=MAX_TOXIC_PROBABILITY,
        help="DeepHit decision threshold.",
    )
    parser.add_argument(
        "--min-fill-probability",
        type=float,
        default=MIN_FILL_PROBABILITY,
        help="DeepHit decision threshold.",
    )
    parser.add_argument(
        "--horizon-index",
        type=int,
        default=HORIZON_INDEX,
        help="DeepHit CIF horizon index used by decision logic.",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help="Parallel worker count. Used only when --mode parallel.",
    )
    parser.add_argument(
        "--mp-start-method",
        default=None,
        help="Optional multiprocessing start method for --mode parallel.",
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=SAMPLE_FRACTION,
        help="Random subsample fraction for the labeled test dataset.",
    )
    parser.add_argument(
        "--latencies-ms",
        default=None,
        help="Comma-separated latency values in ms. Defaults to the built-in log grid.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable verbose/progress output from raw replay.",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override tqdm progress. Defaults to verbose mode.",
    )
    return parser.parse_args()


def latency_grid_ms(raw: str | None = None) -> list[float]:
    if raw:
        values = [round(float(item.strip()), 6) for item in raw.split(",") if item.strip()]
        return sorted(dict.fromkeys(values))
    positive = np.geomspace(0.01, 50.0, 14)
    values = [0.0] + [round(float(v), 2) for v in positive]
    return sorted(dict.fromkeys(values))


def compact_date(date: str) -> str:
    return str(date).replace("-", "")


def default_paths(args: argparse.Namespace) -> dict[str, Path]:
    ticker = args.ticker.upper()
    data_root = Path(args.data_root)
    artifact_root = Path(args.artifact_root)
    start = compact_date(args.data_start_date)
    data_end = compact_date(args.data_end_date)
    raw_end = compact_date(args.raw_end_date)
    dataset_stem = f"labeled_dataset_XNAS_ITCH_{ticker}_mbo_{start}_{data_end}"
    raw_stem = f"XNAS_ITCH_{ticker}_mbo_{start}_{raw_end}"
    raw_path = data_root / "raw" / ticker / f"{raw_stem}.dbn.zst"
    return {
        "dataset_path": data_root / "datasets" / f"{dataset_stem}_test.parquet",
        "runtime_npz_path": data_root
        / "datasets"
        / f"{dataset_stem}_dynamic_preprocessed.npz",
        "raw_path": raw_path,
        "split_cache_path": data_root / "datasets" / f"{raw_stem}_split_points.json",
        "model_path": artifact_root / ticker / f"{args.model_type}_{ticker}.pt",
    }


def resolved_paths(args: argparse.Namespace) -> dict[str, Path]:
    paths = default_paths(args)
    overrides = {
        "dataset_path": args.dataset_path,
        "runtime_npz_path": args.runtime_npz_path,
        "raw_path": args.raw_path,
        "split_cache_path": args.split_cache_path,
        "model_path": args.model_path,
    }
    for key, value in overrides.items():
        if value:
            paths[key] = Path(value)
    return paths


def make_dataset(args: argparse.Namespace, paths: dict[str, Path]) -> BacktestDataset:
    return BacktestDataset.from_runtime_artifacts(
        paths["dataset_path"],
        runtime_npz_path=paths["runtime_npz_path"],
        snapshot_policy="entry",
        sample_fraction=args.sample_fraction,
    )


def make_strategy(args: argparse.Namespace, paths: dict[str, Path]) -> DeepHitStrategy:
    return DeepHitStrategy.from_saved_model(
        paths["model_path"],
        model_name=args.model_type,
        decision_logic=DeepHitThresholdDecisionLogic(
            max_toxic_probability=args.max_toxic_probability,
            min_fill_probability=args.min_fill_probability,
            horizon_index=args.horizon_index,
        ),
    )


def run_one_latency(
    *,
    args: argparse.Namespace,
    paths: dict[str, Path],
    dataset: BacktestDataset,
    strategy: DeepHitStrategy,
    latency_ms: float,
    raw_output_dir: Path,
    verbose: bool,
    progress: bool | None,
) -> pd.DataFrame:
    latency_token = f"{latency_ms:.6f}".rstrip("0").rstrip(".").replace(".", "p")
    raw_path = raw_output_dir / f"raw_latency_{latency_token}ms_backtest.parquet"
    summary_path = raw_output_dir / f"raw_latency_{latency_token}ms_summary.csv"

    engine = RawDatabentoBacktestEngine(
        strategy,
        raw_path=paths["raw_path"],
        latency_provider=StaticLatencyProvider.from_milliseconds(latency_ms),
        snapshot_bin_messages=args.snapshot_bin_messages,
        lifecycle_aware=args.lifecycle_aware,
        lifecycle_stride=args.lifecycle_stride,
        lifecycle_max_evaluations=args.lifecycle_max_evaluations,
        verbose=verbose,
        progress=progress,
    )
    if args.mode == "parallel":
        report = engine.run_parallel(
            dataset,
            n_workers=args.n_workers,
            split_cache_path=paths["split_cache_path"],
            mp_start_method=args.mp_start_method,
        )
    else:
        report = engine.run(
            dataset,
            split_cache_path=paths["split_cache_path"],
        )
    report.write(raw_path, summary_path)

    summary = report.summary_frame()
    summary.insert(0, "latency_ms", float(latency_ms))
    summary.insert(1, "ticker", args.ticker.upper())
    summary.insert(2, "model_type", args.model_type)
    summary.insert(3, "mode", args.mode)
    summary.insert(4, "lifecycle_aware", bool(args.lifecycle_aware))
    summary.insert(5, "raw_backtest_path", str(raw_path))
    return summary


def main() -> None:
    args = parse_args()
    paths = resolved_paths(args)

    ticker = args.ticker.upper()
    lifecycle_tag = "lifecycle" if args.lifecycle_aware else "no_lifecycle"
    output_root = Path(args.output_root)
    raw_output_dir = output_root / ticker / args.model_type / lifecycle_tag
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    output_summary = raw_output_dir / "raw_latency_sweep_summary.csv"

    verbose = not args.quiet
    progress = args.progress if args.progress is not None else verbose

    if verbose:
        print("[latency-sweep] Configuration:")
        print(f"  ticker={ticker}")
        print(f"  model_type={args.model_type}")
        print(f"  mode={args.mode}")
        print(f"  dataset={paths['dataset_path']}")
        print(f"  runtime_npz={paths['runtime_npz_path']}")
        print(f"  raw={paths['raw_path']}")
        print(f"  split_cache={paths['split_cache_path']}")
        print(f"  model={paths['model_path']}")
        print(f"  output={output_summary}")

    dataset = make_dataset(args, paths)
    strategy = make_strategy(args, paths)

    summaries = []
    for latency_ms in latency_grid_ms(args.latencies_ms):
        if verbose:
            print(
                f"\n[latency-sweep] Running {latency_ms:.6g} ms "
                f"(ticker={ticker}, model={args.model_type}, "
                f"lifecycle_aware={args.lifecycle_aware}, mode={args.mode})"
            )
        summaries.append(
            run_one_latency(
                args=args,
                paths=paths,
                dataset=dataset,
                strategy=strategy,
                latency_ms=latency_ms,
                raw_output_dir=raw_output_dir,
                verbose=verbose,
                progress=progress,
            )
        )

    combined = pd.concat(summaries, ignore_index=True)
    combined.to_csv(output_summary, index=False)
    print(f"\n[latency-sweep] Wrote {len(combined)} rows to {output_summary}")


if __name__ == "__main__":
    main()
