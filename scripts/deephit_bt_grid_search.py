"""Grid search threshold tuning for DeepHit backtest strategies."""

from __future__ import annotations

import sys
import argparse
from itertools import product
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest import BacktestDataset, BacktestEngine
from src.backtest.metrics import ImplementationShortfallMetric
from src.backtest.strategies import (
    DeepHitPredictionCache,
    DeepHitStrategy,
    DeepHitThresholdDecisionLogic,
)


DEFAULT_TEST_PATH = (
    "data/datasets/labeled_dataset_XNAS_ITCH_AAPL_mbo_20251001_20260101_test.parquet"
)
DEFAULT_RUNTIME_NPZ_PATH = (
    "data/datasets/labeled_dataset_XNAS_ITCH_AAPL_mbo_20251001_20260101_dynamic_preprocessed.npz"
)
DEFAULT_MODEL_PATH = (
    "checkpoints/dynamic_deephit_gru_transformer_AAPL_best_epoch_base_net_i1xfzhr5.pt"
)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune DeepHit decision thresholds on the first N test orders."
    )
    parser.add_argument("--test-path", default=DEFAULT_TEST_PATH)
    parser.add_argument("--runtime-npz-path", default=DEFAULT_RUNTIME_NPZ_PATH)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model-name", default="gru_transformer")
    parser.add_argument("--row-limit", type=int, default=4000)
    parser.add_argument("--snapshot-policy", default="entry")
    parser.add_argument("--lifecycle-aware", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lifecycle-stride", type=int, default=50)
    parser.add_argument("--lifecycle-max-evaluations", type=int, default=None)
    parser.add_argument(
        "--max-toxic-grid",
        default="0.22,0.25,0.28,0.31,0.34,0.37,0.40",
        help="Comma-separated grid for max conditional toxic probability.",
    )
    parser.add_argument(
        "--min-fill-grid",
        default="0.50,0.60,0.70,0.80,0.90",
        help="Comma-separated grid for minimum fill CIF.",
    )
    parser.add_argument(
        "--horizon-index-grid",
        default="9",
        help="Comma-separated grid for DeepHit CIF horizon indices.",
    )
    parser.add_argument(
        "--cache-path",
        default="cache/deephit_grid_predictions.pkl",
        help="Local prediction-cache pickle path.",
    )
    parser.add_argument(
        "--output-csv",
        default="reports/deephit_threshold_grid_search.csv",
        help="Grid-search result CSV path.",
    )
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()
    max_toxic_grid = parse_float_grid(args.max_toxic_grid)
    min_fill_grid = parse_float_grid(args.min_fill_grid)
    horizon_index_grid = parse_int_grid(args.horizon_index_grid)

    dataset = BacktestDataset.from_runtime_artifacts(
        args.test_path,
        runtime_npz_path=args.runtime_npz_path,
        snapshot_policy=args.snapshot_policy,
        row_limit=args.row_limit,
    )
    metric = build_metric(dataset)
    cache = load_cache(args.cache_path)

    strategy = DeepHitStrategy.from_saved_model(
        args.model_path,
        model_name=args.model_name,
        decision_logic=DeepHitThresholdDecisionLogic(),
        prediction_cache=cache,
    )

    rows: list[dict] = []
    total = len(max_toxic_grid) * len(min_fill_grid) * len(horizon_index_grid)
    for run_idx, (max_toxic, min_fill, horizon_index) in enumerate(
        product(max_toxic_grid, min_fill_grid, horizon_index_grid),
        start=1,
    ):
        print(
            f"[{run_idx}/{total}] max_toxic_probability={max_toxic:.4f}, "
            f"min_fill_probability={min_fill:.4f}, "
            f"horizon_index={horizon_index}",
            flush=True,
        )
        strategy.decision_logic = DeepHitThresholdDecisionLogic(
            max_toxic_probability=max_toxic,
            min_fill_probability=min_fill,
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
                "max_toxic_probability": max_toxic,
                "min_fill_probability": min_fill,
                "horizon_index": horizon_index,
                "cache_size": len(cache),
                "cache_hits": cache.hits,
                "cache_misses": cache.misses,
            }
        )
        rows.append(summary)
        cache.save(args.cache_path)

    results = pd.DataFrame(rows)
    sort_cols = [
        col
        for col in ["mean_implementation_shortfall_bps", "skipped_orders"]
        if col in results.columns
    ]
    if sort_cols:
        results = results.sort_values(sort_cols, ascending=[True] * len(sort_cols))

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    cache.save(args.cache_path)

    print(f"Wrote grid-search results to {output_path}")
    if not results.empty:
        best = results.iloc[0]
        print(
            "Best row: "
            f"max_toxic_probability={best['max_toxic_probability']}, "
            f"min_fill_probability={best['min_fill_probability']}, "
            f"horizon_index={best['horizon_index']}, "
            f"mean_implementation_shortfall_bps={best.get('mean_implementation_shortfall_bps')}"
        )


if __name__ == "__main__":
    main()
