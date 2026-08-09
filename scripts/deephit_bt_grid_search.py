"""Grid search threshold tuning for DeepHit backtest strategies."""

from __future__ import annotations

import sys
import argparse
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest import BacktestDataset, BacktestEngine
from src.backtest.metrics import ImplementationShortfallMetric
from src.backtest.strategies import (
    DeepHitPredictionCache,
    DeepHitStrategy,
    DeepHitToxicCIFDecisionLogic,
)


DATA_START_DATE = "2025-10-01"
DATA_END_DATE = "2026-01-01"
DATA_ROOT = "data"
ARTIFACT_ROOT = "checkpoints"
OUTPUT_ROOT = "reports/deephit_bt_grid_search_toxic_cif"


def parse_float_grid(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("Grid must contain at least one float value.")
    return values


def parse_int_grid(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("Grid must contain at least one integer value.")
    return values


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune DeepHit decision thresholds on the first N test orders."
    )
    parser.add_argument("--ticker", default="AAPL", help="Ticker symbol, e.g. AAPL.")
    parser.add_argument(
        "--model-type",
        default=None,
        help=(
            "DeepHit architecture/model type. Model file defaults to "
            "[model_type]_[ticker].pt. If omitted, --model-name is used."
        ),
    )
    parser.add_argument(
        "--model-name",
        default="gru_transformer",
        help="Backward-compatible alias for --model-type.",
    )
    parser.add_argument(
        "--data-root",
        default=DATA_ROOT,
        help=(
            "Root containing datasets/. On PSC this is usually "
            "/ocean/projects/cis260122p/shared/data."
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
        help=(
            "Root for default grid-search CSV/cache outputs. Defaults to "
            "reports/deephit_bt_grid_search_toxic_cif/[ticker]/[model_type]/."
        ),
    )
    parser.add_argument(
        "--data-start-date",
        default=DATA_START_DATE,
        help="Dataset start date shared across tickers.",
    )
    parser.add_argument(
        "--data-end-date",
        default=DATA_END_DATE,
        help="Dataset end date shared across tickers.",
    )
    parser.add_argument(
        "--test-path",
        default=None,
        help="Optional explicit labeled test parquet path.",
    )
    parser.add_argument(
        "--runtime-npz-path",
        default=None,
        help="Optional explicit dynamic preprocessed runtime NPZ path.",
    )
    parser.add_argument("--model-path", default=None, help="Optional explicit model PT path.")
    parser.add_argument("--row-limit", type=int, default=4000)
    parser.add_argument("--snapshot-policy", default="entry")
    parser.add_argument("--lifecycle-aware", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lifecycle-stride", type=int, default=50)
    parser.add_argument("--lifecycle-max-evaluations", type=int, default=None)
    parser.add_argument(
        "--max-toxic-cif-grid",
        "--max-toxic-grid",
        dest="max_toxic_cif_grid",
        default="0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00",
        help="Comma-separated grid for maximum toxic CIF threshold.",
    )
    parser.add_argument(
        "--min-fill-grid",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--horizon-index-grid",
        default="4, 9, 14, 19, 24, 29",
        help="Comma-separated grid for DeepHit CIF horizon indices.",
    )
    parser.add_argument(
        "--cache-path",
        default=None,
        help="Optional explicit local prediction-cache pickle path.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional explicit grid-search result CSV path.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help=(
            "Write accumulated results every N completed new grid points. "
            "Default: 1."
        ),
    )
    parser.add_argument(
        "--cache-save-every",
        type=int,
        default=10,
        help=(
            "Persist the prediction cache every N completed new grid points. "
            "Use 1 for maximum fault tolerance or 0 to save only at the end. "
            "Default: 10."
        ),
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume by skipping parameter combinations already present in output CSV.",
    )
    return parser.parse_args(argv)


def compact_date(date: str) -> str:
    return str(date).replace("-", "")


def resolved_model_type(args: argparse.Namespace) -> str:
    return str(args.model_type or args.model_name)


def default_paths(args: argparse.Namespace) -> dict[str, Path]:
    ticker = args.ticker.upper()
    model_type = resolved_model_type(args)
    data_root = Path(args.data_root)
    artifact_root = Path(args.artifact_root)
    output_root = Path(args.output_root)
    start = compact_date(args.data_start_date)
    end = compact_date(args.data_end_date)
    dataset_stem = f"labeled_dataset_XNAS_ITCH_{ticker}_mbo_{start}_{end}"
    output_dir = output_root / ticker / model_type
    return {
        "test_path": data_root / "datasets" / f"{dataset_stem}_test.parquet",
        "runtime_npz_path": data_root / "datasets" / f"{dataset_stem}_dynamic_preprocessed.npz",
        "model_path": artifact_root / ticker / f"{model_type}_{ticker}.pt",
        "cache_path": output_dir / "deephit_toxic_cif_grid_predictions.pkl",
        "output_csv": output_dir / "deephit_toxic_cif_grid_search.csv",
    }


def resolved_paths(args: argparse.Namespace) -> dict[str, Path]:
    paths = default_paths(args)
    overrides = {
        "test_path": args.test_path,
        "runtime_npz_path": args.runtime_npz_path,
        "model_path": args.model_path,
        "cache_path": args.cache_path,
        "output_csv": args.output_csv,
    }
    for key, value in overrides.items():
        if value:
            paths[key] = Path(value)
    return paths


def load_cache(path: str | Path) -> DeepHitPredictionCache:
    cache_path = Path(path)
    if cache_path.exists():
        return DeepHitPredictionCache.from_file(cache_path)
    return DeepHitPredictionCache()


def build_metric(dataset: BacktestDataset) -> ImplementationShortfallMetric:
    calibration_frame = dataset.load_calibration_frame()
    if calibration_frame is None or calibration_frame.empty:
        calibration_frame = dataset.load_frame()
    return ImplementationShortfallMetric.from_labeled_orders(calibration_frame)


def combo_key(max_toxic_cif: float, horizon_index: int) -> tuple[str, int]:
    return (
        _float_key(max_toxic_cif),
        int(horizon_index),
    )


def completed_combo_keys(results: pd.DataFrame) -> set[tuple[str, int]]:
    required = {"max_toxic_cif", "horizon_index"}
    if results.empty or not required.issubset(results.columns):
        return set()
    keys: set[tuple[str, int]] = set()
    for row in results[list(required)].itertuples(index=False):
        try:
            keys.add(
                combo_key(
                    float(getattr(row, "max_toxic_cif")),
                    int(getattr(row, "horizon_index")),
                )
            )
        except (TypeError, ValueError):
            continue
    return keys


def load_existing_results(path: str | Path, *, resume: bool) -> pd.DataFrame:
    output_path = Path(path)
    if not resume or not output_path.exists():
        return pd.DataFrame()
    try:
        results = pd.read_csv(output_path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    print(f"Loaded {len(results):,} existing grid-search row(s) from {output_path}.")
    return results


def sort_results(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return results
    is_col = (
        "mean_is_bps"
        if "mean_is_bps" in results.columns
        else "mean_implementation_shortfall_bps"
    )
    skipped_col = "skipped" if "skipped" in results.columns else "skipped_orders"
    sort_cols = [col for col in [is_col, skipped_col] if col in results.columns]
    if sort_cols:
        return results.sort_values(sort_cols, ascending=[True] * len(sort_cols))
    return results


def write_results_checkpoint(results: pd.DataFrame, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")
    sort_results(results).to_csv(tmp_path, index=False)
    tmp_path.replace(output_path)


def _float_key(value: float) -> str:
    return f"{float(value):.12g}"


def main() -> None:
    args = parse_args()
    if int(args.checkpoint_every) < 1:
        raise ValueError("checkpoint_every must be >= 1")
    if int(args.cache_save_every) < 0:
        raise ValueError("cache_save_every must be >= 0")

    ticker = args.ticker.upper()
    model_type = resolved_model_type(args)
    paths = resolved_paths(args)
    max_toxic_cif_grid = parse_float_grid(args.max_toxic_cif_grid)
    horizon_index_grid = parse_int_grid(args.horizon_index_grid)
    output_path = paths["output_csv"]
    results = load_existing_results(output_path, resume=bool(args.resume))
    completed = completed_combo_keys(results)

    print("[deephit-grid] Configuration:")
    print(f"  ticker={ticker}")
    print(f"  model_type={model_type}")
    print(f"  test_path={paths['test_path']}")
    print(f"  runtime_npz={paths['runtime_npz_path']}")
    print(f"  model_path={paths['model_path']}")
    print(f"  cache_path={paths['cache_path']}")
    print(f"  output_csv={output_path}")
    print(f"  row_limit={args.row_limit}")
    print(f"  lifecycle_aware={args.lifecycle_aware}")
    print(f"  lifecycle_stride={args.lifecycle_stride}")
    print(f"  max_toxic_cif_grid={max_toxic_cif_grid}")
    print(f"  horizon_index_grid={horizon_index_grid}")
    print(f"  checkpoint_every={args.checkpoint_every}")
    print(f"  cache_save_every={args.cache_save_every}")

    dataset = BacktestDataset.from_runtime_artifacts(
        paths["test_path"],
        runtime_npz_path=paths["runtime_npz_path"],
        snapshot_policy=args.snapshot_policy,
        row_limit=args.row_limit,
    )
    metric = build_metric(dataset)
    cache = load_cache(paths["cache_path"])

    strategy = DeepHitStrategy.from_saved_model(
        paths["model_path"],
        model_name=model_type,
        decision_logic=DeepHitToxicCIFDecisionLogic(),
        prediction_cache=cache,
    )

    rows: list[dict[str, Any]] = results.to_dict("records") if not results.empty else []
    total = len(max_toxic_cif_grid) * len(horizon_index_grid)
    skipped_existing = 0
    completed_new = 0
    for run_idx, (max_toxic_cif, horizon_index) in enumerate(
        product(max_toxic_cif_grid, horizon_index_grid),
        start=1,
    ):
        key = combo_key(max_toxic_cif, horizon_index)
        if key in completed:
            skipped_existing += 1
            print(
                f"[{run_idx}/{total}] skipping completed "
                f"max_toxic_cif={max_toxic_cif:.4f}, "
                f"horizon_index={horizon_index}",
                flush=True,
            )
            continue

        print(
            f"[{run_idx}/{total}] max_toxic_cif={max_toxic_cif:.4f}, "
            f"horizon_index={horizon_index}",
            flush=True,
        )
        strategy.decision_logic = DeepHitToxicCIFDecisionLogic(
            max_toxic_cif=max_toxic_cif,
            horizon_index=horizon_index,
        )
        report = BacktestEngine(
            strategy,
            metrics=[metric],
            lifecycle_aware=bool(args.lifecycle_aware),
            lifecycle_stride=int(args.lifecycle_stride),
            lifecycle_max_evaluations=args.lifecycle_max_evaluations,
        ).run(dataset.iter_snapshots())
        summary = report.summary_frame().iloc[0].to_dict()
        summary.update(
            {
                "ticker": ticker,
                "model_type": model_type,
                "decision_logic": "toxic_cif",
                "max_toxic_cif": max_toxic_cif,
                "horizon_index": horizon_index,
                "cache_size": len(cache),
                "cache_hits": cache.hits,
                "cache_misses": cache.misses,
            }
        )
        rows.append(summary)
        completed.add(key)
        completed_new += 1
        if (
            int(args.cache_save_every) > 0
            and completed_new % int(args.cache_save_every) == 0
        ):
            cache.save(paths["cache_path"])
            print(f"Saved prediction cache to {paths['cache_path']}.", flush=True)
        if completed_new % int(args.checkpoint_every) == 0:
            checkpoint = pd.DataFrame(rows)
            write_results_checkpoint(checkpoint, output_path)
            print(
                f"Checkpointed {len(checkpoint):,} total row(s) to {output_path}.",
                flush=True,
            )

    results = pd.DataFrame(rows)
    results = sort_results(results)
    write_results_checkpoint(results, output_path)
    cache.save(paths["cache_path"])

    print(
        f"Wrote grid-search results to {output_path} "
        f"({completed_new} new, {skipped_existing} resumed/skipped)."
    )
    if not results.empty:
        is_col = (
            "mean_is_bps"
            if "mean_is_bps" in results.columns
            else "mean_implementation_shortfall_bps"
        )
        best = results.iloc[0]
        print(
            "Best row: "
            f"max_toxic_cif={best['max_toxic_cif']}, "
            f"horizon_index={best['horizon_index']}, "
            f"{is_col}={best.get(is_col)}"
        )


if __name__ == "__main__":
    main()
