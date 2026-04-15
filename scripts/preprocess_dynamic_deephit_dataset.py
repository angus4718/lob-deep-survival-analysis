"""Preprocess and split a labeled dataset for the dynamic DeepHit notebook flow.

This script mirrors the order-level dataset preprocessing in
`notebooks/dynamic_models/standardized_dynamic_deephit.ipynb`:
1) Normalize raw-top5 column names.
2) Compute day-boundary train/val/test split with 70/85 row targets.
3) Write three parquet files for train/val/test.
4) Build an uncapped dynamic-sample manifest using train-derived horizon censoring.
5) Precompute runtime-ready arrays used by the loss-tuning notebook.

The dynamic-manifest step intentionally uses `max_samples_per_order=None`
(no cap) to match notebook cell 3a behavior.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.notebook_data import (  # noqa: E402
    LabTransform,
    build_dynamic_samples_manifest,
    choose_time_horizon_from_train_fills,
    fit_dynamic_normalizer_from_manifest,
    select_manifest_indices_by_source_rows,
)

DEFAULT_TICKER = "AAPL"
DEFAULT_START_DATE = "2025-10-01"
DEFAULT_END_DATE = "2026-01-01"
DEFAULT_EXCHANGE = "XNAS"
DEFAULT_FEED = "ITCH"
DEFAULT_SCHEMA = "mbo"
DEFAULT_DATASETS_DIR = Path("/ocean/projects/cis260122p/shared/data/datasets")
DEFAULT_LOOKBACK_STEPS = 500
DEFAULT_LOB_FEATURE_DIM = 20
DEFAULT_TOX_FEATURE_DIM = 12
DEFAULT_HORIZON_QUANTILE = 95.0
DEFAULT_TRAIN_ORDER_SUBSAMPLE_FRACTION = 0.3
DEFAULT_TRAIN_ORDER_SUBSAMPLE_SEED = 4718
DEFAULT_MAX_SAMPLES_PER_SOURCE_ROW = 100
DEFAULT_SPLIT_CAP_RANDOM_SEED = 4718
DEFAULT_NUM_TIME_STEPS = 30
DEFAULT_NORMALIZER_CHUNK_SIZE = 2048
PREPROCESSING_CONTRACT_VERSION = 1


def _best_day_cut(target_row: int, day_end_idx: Sequence[int]) -> int:
    return min(range(len(day_end_idx)), key=lambda i: abs(day_end_idx[i] - target_row))


def _dataset_path_from_parts(
    *,
    datasets_dir: Path,
    ticker: str,
    start_date: str,
    end_date: str,
    exchange: str,
    feed: str,
    schema: str,
) -> Path:
    start_ymd = start_date.replace("-", "")
    end_ymd = end_date.replace("-", "")
    name = f"labeled_dataset_{exchange}_{feed}_{ticker}_{schema}_{start_ymd}_{end_ymd}.parquet"
    return datasets_dir / name


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess and split dynamic DeepHit labeled dataset into train/val/test parquet files, "
            "then build an uncapped dynamic sample manifest."
        )
    )
    parser.add_argument(
        "--input-path",
        default=os.getenv("INPUT_PATH", "").strip() or None,
        help="Optional explicit input labeled parquet path.",
    )
    parser.add_argument(
        "--datasets-dir",
        default=os.getenv("DATASETS_DIR", str(DEFAULT_DATASETS_DIR)),
        help="Base datasets directory used to resolve default input/output paths.",
    )
    parser.add_argument(
        "--ticker",
        default=os.getenv("SYMBOL", DEFAULT_TICKER),
        help="Ticker symbol used when --input-path is not provided.",
    )
    parser.add_argument(
        "--start-date",
        default=os.getenv("START_DATE", DEFAULT_START_DATE),
        help="Dataset start date in YYYY-MM-DD used when --input-path is not provided.",
    )
    parser.add_argument(
        "--end-date",
        default=os.getenv("END_DATE", DEFAULT_END_DATE),
        help="Dataset end date in YYYY-MM-DD used when --input-path is not provided.",
    )
    parser.add_argument(
        "--exchange",
        default=os.getenv("EXCHANGE", DEFAULT_EXCHANGE),
        help="Exchange code used when --input-path is not provided.",
    )
    parser.add_argument(
        "--feed",
        default=os.getenv("FEED", DEFAULT_FEED),
        help="Feed/protocol code used when --input-path is not provided.",
    )
    parser.add_argument(
        "--schema",
        default=os.getenv("SCHEMA", DEFAULT_SCHEMA),
        help="Schema suffix used when --input-path is not provided.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("OUTPUT_DIR", "").strip() or None,
        help="Output directory for split parquets (default: --datasets-dir).",
    )
    parser.add_argument(
        "--output-prefix",
        default=os.getenv("OUTPUT_PREFIX", "").strip() or None,
        help="Prefix for output filenames. Defaults to input file stem.",
    )
    parser.add_argument(
        "--compression",
        default=os.getenv("PARQUET_COMPRESSION", "zstd"),
        help="Parquet compression codec passed to pandas.to_parquet.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output parquet files if present.",
    )
    parser.add_argument(
        "--skip-dynamic-manifest",
        action="store_true",
        help="Skip building dynamic manifest artifacts after writing split parquets.",
    )
    parser.add_argument(
        "--manifest-prefix",
        default=os.getenv("MANIFEST_PREFIX", "").strip() or None,
        help="Prefix for manifest outputs. Defaults to --output-prefix value.",
    )
    parser.add_argument(
        "--lookback-steps",
        type=int,
        default=int(os.getenv("LOOKBACK_STEPS", str(DEFAULT_LOOKBACK_STEPS))),
        help="Lookback window size used for dynamic manifest building.",
    )
    parser.add_argument(
        "--lob-feature-dim",
        type=int,
        default=int(os.getenv("LOB_FEATURE_DIM", str(DEFAULT_LOB_FEATURE_DIM))),
        help="LOB feature width used by dynamic manifest building.",
    )
    parser.add_argument(
        "--tox-feature-dim",
        type=int,
        default=int(os.getenv("TOX_FEATURE_DIM", str(DEFAULT_TOX_FEATURE_DIM))),
        help="Toxicity feature width used by dynamic manifest building.",
    )
    parser.add_argument(
        "--tox-time-delta-col",
        type=int,
        default=None,
        help=(
            "Explicit toxicity-sequence time-delta column index. "
            "Default: -2 when tox-feature-dim >= 12, else -1."
        ),
    )
    parser.add_argument(
        "--horizon-quantile",
        type=float,
        default=float(os.getenv("HORIZON_QUANTILE", str(DEFAULT_HORIZON_QUANTILE))),
        help="Train-fill quantile used to derive dynamic admin-censoring horizon T_MAX.",
    )
    parser.add_argument(
        "--train-order-subsample-fraction",
        type=float,
        default=float(
            os.getenv(
                "TRAIN_ORDER_SUBSAMPLE_FRACTION",
                str(DEFAULT_TRAIN_ORDER_SUBSAMPLE_FRACTION),
            )
        ),
        help="Fraction of training order instances kept for tuning-time preparation.",
    )
    parser.add_argument(
        "--train-order-subsample-seed",
        type=int,
        default=int(
            os.getenv(
                "TRAIN_ORDER_SUBSAMPLE_SEED",
                str(DEFAULT_TRAIN_ORDER_SUBSAMPLE_SEED),
            )
        ),
        help="Random seed used for optional train-order subsampling.",
    )
    parser.add_argument(
        "--max-samples-per-source-row",
        type=int,
        default=int(
            os.getenv(
                "MAX_SAMPLES_PER_SOURCE_ROW",
                str(DEFAULT_MAX_SAMPLES_PER_SOURCE_ROW),
            )
        ),
        help="Random cap applied per order instance for train/val dynamic samples.",
    )
    parser.add_argument(
        "--split-cap-random-seed",
        type=int,
        default=int(
            os.getenv(
                "SPLIT_CAP_RANDOM_SEED",
                str(DEFAULT_SPLIT_CAP_RANDOM_SEED),
            )
        ),
        help="Base seed used by random per-order-instance sample caps.",
    )
    parser.add_argument(
        "--num-time-steps",
        type=int,
        default=int(os.getenv("NUM_TIME_STEPS", str(DEFAULT_NUM_TIME_STEPS))),
        help="Number of discretized time bins used by LabTransform.",
    )
    parser.add_argument(
        "--normalizer-chunk-size",
        type=int,
        default=int(os.getenv("NORMALIZER_CHUNK_SIZE", str(DEFAULT_NORMALIZER_CHUNK_SIZE))),
        help="Chunk size used when fitting manifest-based normalization statistics.",
    )
    return parser.parse_args()


def _resolve_input_path(args: argparse.Namespace) -> Path:
    datasets_dir = Path(args.datasets_dir)
    if args.input_path:
        input_path = Path(args.input_path)
    else:
        input_path = _dataset_path_from_parts(
            datasets_dir=datasets_dir,
            ticker=str(args.ticker),
            start_date=str(args.start_date),
            end_date=str(args.end_date),
            exchange=str(args.exchange),
            feed=str(args.feed),
            schema=str(args.schema),
        )
    if not input_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {input_path}")
    return input_path


def _normalize_notebook_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["entry_representation"] = out["entry_representation_raw_top5"]
    out["lob_sequence"] = out["lob_sequence_raw_top5"]

    required_cols = [
        "order_id",
        "entry_time",
        "duration_s",
        "event_type",
        "side",
        "toxicity_representation",
        "lob_sequence",
        "toxicity_sequence",
    ]
    missing_cols = [c for c in required_cols if c not in out.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns after preprocessing: {missing_cols}")

    return out


def _compute_day_boundary_splits(df: pd.DataFrame):
    entry_ns = df["entry_time"].to_numpy()
    dates = (
        pd.to_datetime(entry_ns, unit="ns", utc=True)
        .tz_convert("America/New_York")
        .normalize()
    )

    unique_days = sorted(dates.unique())
    n_days = len(unique_days)
    if n_days < 3:
        raise ValueError(
            "Day-boundary split requires at least 3 trading days to produce train/val/test."
        )

    n_rows = len(df)
    target_train_end = int(n_rows * 0.70)
    target_val_end = int(n_rows * 0.85)

    day_end_idx = [(dates <= d).sum() - 1 for d in unique_days]

    train_day_idx = _best_day_cut(target_train_end, day_end_idx)
    val_day_idx = _best_day_cut(target_val_end, day_end_idx)
    train_day_idx = min(train_day_idx, n_days - 3)
    val_day_idx = max(train_day_idx + 1, min(val_day_idx, n_days - 2))

    train_end = day_end_idx[train_day_idx] + 1
    val_end = day_end_idx[val_day_idx] + 1

    idx = np.arange(n_rows)
    train_mask = idx < train_end
    val_mask = (idx >= train_end) & (idx < val_end)
    test_mask = idx >= val_end

    # Partition integrity checks.
    if np.any(train_mask & val_mask) or np.any(train_mask & test_mask) or np.any(val_mask & test_mask):
        raise RuntimeError("Split masks overlap; expected mutually exclusive partitions.")
    if not np.all(train_mask | val_mask | test_mask):
        raise RuntimeError("Split masks do not cover all rows.")

    train_days = unique_days[: train_day_idx + 1]
    val_days = unique_days[train_day_idx + 1 : val_day_idx + 1]
    test_days = unique_days[val_day_idx + 1 :]

    return train_mask, val_mask, test_mask, train_days, val_days, test_days


def _format_day_span(days) -> str:
    if len(days) == 0:
        return "(empty)"
    start = pd.Timestamp(days[0]).date().isoformat()
    end = pd.Timestamp(days[-1]).date().isoformat()
    return f"{start} -> {end} ({len(days)} day(s))"


def _resolve_output_paths(
    *,
    output_dir: Path,
    output_prefix: str,
) -> tuple[Path, Path, Path]:
    train_path = output_dir / f"{output_prefix}_train.parquet"
    val_path = output_dir / f"{output_prefix}_val.parquet"
    test_path = output_dir / f"{output_prefix}_test.parquet"
    return train_path, val_path, test_path


def _resolve_manifest_output_paths(
    *,
    output_dir: Path,
    manifest_prefix: str,
) -> tuple[Path, Path, Path, Path, Path]:
    manifest_table_path = output_dir / f"{manifest_prefix}_dynamic_manifest.parquet"
    order_store_pickle_path = output_dir / f"{manifest_prefix}_dynamic_order_store.pkl"
    sample_manifest_parquet_path = output_dir / f"{manifest_prefix}_dynamic_sample_manifest.parquet"
    manifest_meta_path = output_dir / f"{manifest_prefix}_dynamic_manifest_meta.json"
    preprocessed_npz_path = output_dir / f"{manifest_prefix}_dynamic_preprocessed.npz"
    return (
        manifest_table_path,
        order_store_pickle_path,
        sample_manifest_parquet_path,
        manifest_meta_path,
        preprocessed_npz_path,
    )


def _ensure_writable(paths: Sequence[Path], overwrite: bool) -> None:
    if overwrite:
        return
    existing = [p for p in paths if p.exists()]
    if existing:
        joined = "\n".join(str(p) for p in existing)
        raise FileExistsError(
            "Output file(s) already exist. Re-run with --overwrite to replace:\n" + joined
        )


def _build_dynamic_manifest_uncapped(
    *,
    df_split: pd.DataFrame,
    lookback_steps: int,
    lob_feature_dim: int,
    tox_feature_dim: int,
    horizon_quantile: float,
    tox_time_delta_col: int | None,
):
    if "dataset_split" not in df_split.columns:
        raise ValueError("Expected 'dataset_split' column for dynamic manifest build.")

    split_values = df_split["dataset_split"].astype(str).str.lower()
    train_mask = split_values == "train"
    if not train_mask.any():
        raise ValueError("No train rows found while building dynamic manifest.")

    durations = df_split["duration_s"].to_numpy(dtype=np.float32, copy=False)
    events = df_split["event_type"].to_numpy(dtype=np.int64, copy=False)
    t_max = choose_time_horizon_from_train_fills(
        durations=durations,
        events=events,
        train_mask=train_mask.to_numpy(),
        fill_event_codes=(1, 2),
        quantile=float(horizon_quantile),
    )

    effective_tox_time_delta_col = (
        int(tox_time_delta_col)
        if tox_time_delta_col is not None
        else (-2 if int(tox_feature_dim) >= 12 else -1)
    )

    print(
        f"Building uncapped dynamic sample manifest from {len(df_split):,} orders "
        f"(tox_time_delta_col={effective_tox_time_delta_col}, T_MAX={t_max:.4f}s)..."
    )
    order_store, sample_manifest = build_dynamic_samples_manifest(
        df_split,
        lookback_steps=int(lookback_steps),
        lob_dim=int(lob_feature_dim),
        tox_dim=int(tox_feature_dim),
        duration_col="duration_s",
        event_col="event_type",
        admin_censor_time=float(t_max),
        max_samples_per_order=None,
        tox_time_delta_col=effective_tox_time_delta_col,
        validate_remaining_time=True,
    )

    return order_store, sample_manifest, float(t_max), effective_tox_time_delta_col


def _slice_manifest_arrays(sample_manifest, sample_idx: np.ndarray):
    idx = np.asarray(sample_idx, dtype=np.int64)
    return (
        sample_manifest.y[idx].astype(np.float32, copy=False),
        sample_manifest.d[idx].astype(np.int64, copy=False),
        sample_manifest.source_row_idx[idx].astype(np.int64, copy=False),
        sample_manifest.update_idx[idx].astype(np.int64, copy=False),
    )


def _prepare_runtime_artifacts(
    *,
    df_split: pd.DataFrame,
    order_store,
    sample_manifest,
    t_max: float,
    train_order_subsample_fraction: float,
    train_order_subsample_seed: int,
    max_samples_per_source_row: int,
    split_cap_random_seed: int,
    num_time_steps: int,
    normalizer_chunk_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    if "dataset_split" not in df_split.columns:
        raise ValueError("Expected 'dataset_split' column while preparing runtime artifacts.")
    if not (0.0 < float(train_order_subsample_fraction) <= 1.0):
        raise ValueError("train_order_subsample_fraction must be in the interval (0, 1].")
    if int(max_samples_per_source_row) < 1:
        raise ValueError("max_samples_per_source_row must be >= 1.")
    if int(num_time_steps) < 2:
        raise ValueError("num_time_steps must be >= 2.")
    if int(normalizer_chunk_size) < 1:
        raise ValueError("normalizer_chunk_size must be >= 1.")

    split_values = df_split["dataset_split"].astype(str).str.lower().to_numpy()
    if not np.isin(split_values, ["train", "val", "test"]).all():
        raise ValueError("dataset_split must only contain train/val/test labels.")

    train_order_mask = split_values == "train"
    val_order_mask = split_values == "val"
    test_order_mask = split_values == "test"

    train_source_rows_full = np.flatnonzero(train_order_mask).astype(np.int64)
    val_source_rows = np.flatnonzero(val_order_mask).astype(np.int64)
    test_source_rows = np.flatnonzero(test_order_mask).astype(np.int64)

    if train_source_rows_full.size == 0 or val_source_rows.size == 0 or test_source_rows.size == 0:
        raise ValueError(
            "Split rows missing while preparing runtime artifacts: "
            f"train={train_source_rows_full.size}, val={val_source_rows.size}, test={test_source_rows.size}"
        )

    if float(train_order_subsample_fraction) < 1.0:
        rng = np.random.default_rng(int(train_order_subsample_seed))
        n_train_orders_full = int(train_source_rows_full.size)
        n_train_orders_keep = max(
            1,
            int(np.floor(n_train_orders_full * float(train_order_subsample_fraction))),
        )
        train_source_rows = np.sort(
            rng.choice(train_source_rows_full, size=n_train_orders_keep, replace=False).astype(np.int64)
        )
    else:
        train_source_rows = train_source_rows_full

    train_sample_idx_full = select_manifest_indices_by_source_rows(
        sample_manifest,
        train_source_rows,
    )
    val_sample_idx_full = select_manifest_indices_by_source_rows(
        sample_manifest,
        val_source_rows,
    )
    test_sample_idx_full = select_manifest_indices_by_source_rows(
        sample_manifest,
        test_source_rows,
    )

    train_sample_idx = select_manifest_indices_by_source_rows(
        sample_manifest,
        train_source_rows,
        max_samples_per_source_row=int(max_samples_per_source_row),
        seed=int(split_cap_random_seed),
    )
    val_sample_idx = select_manifest_indices_by_source_rows(
        sample_manifest,
        val_source_rows,
        max_samples_per_source_row=int(max_samples_per_source_row),
        seed=int(split_cap_random_seed) + 1,
    )
    test_sample_idx = test_sample_idx_full

    if train_sample_idx.size == 0 or val_sample_idx.size == 0 or test_sample_idx.size == 0:
        raise ValueError(
            "Runtime artifact sample indices are empty: "
            f"train={train_sample_idx.size}, val={val_sample_idx.size}, test={test_sample_idx.size}"
        )

    (
        y_train,
        d_train,
        order_keys_train,
        update_idx_train,
    ) = _slice_manifest_arrays(sample_manifest, train_sample_idx)
    (
        y_val,
        d_val,
        order_keys_val,
        update_idx_val,
    ) = _slice_manifest_arrays(sample_manifest, val_sample_idx)

    feat_mean, feat_std = fit_dynamic_normalizer_from_manifest(
        order_store,
        sample_manifest,
        train_sample_idx,
        chunk_size=int(normalizer_chunk_size),
    )

    order_durations_train = np.minimum(
        df_split.loc[train_source_rows, "duration_s"].astype(np.float64).to_numpy(),
        float(t_max),
    ).astype(np.float32)
    label_transform = LabTransform(int(num_time_steps), scheme="quantiles")
    _, _ = label_transform.fit_transform(
        order_durations_train.copy(),
        np.zeros(len(order_durations_train), dtype=np.int64),
    )

    y_train_disc, d_train_disc = label_transform.transform(y_train.copy(), d_train.copy())
    y_val_disc, d_val_disc = label_transform.transform(y_val.copy(), d_val.copy())
    time_grid = np.asarray(label_transform.cuts, dtype=np.float32)

    runtime_artifacts = {
        # Required by standardized_dynamic_deephit_loss_opt.ipynb.
        "train_sample_idx": train_sample_idx.astype(np.int64, copy=False),
        "val_sample_idx": val_sample_idx.astype(np.int64, copy=False),
        "Y_train_disc": np.asarray(y_train_disc, dtype=np.int64),
        "D_train_disc": np.asarray(d_train_disc, dtype=np.int64),
        "Y_val": y_val,
        "D_val": d_val,
        "Y_val_disc": np.asarray(y_val_disc, dtype=np.int64),
        "D_val_disc": np.asarray(d_val_disc, dtype=np.int64),
        "ORDER_KEYS_TRAIN": order_keys_train,
        "ORDER_KEYS_VAL": order_keys_val,
        "UPDATE_IDX_TRAIN": update_idx_train,
        "UPDATE_IDX_VAL": update_idx_val,
        "feat_mean": np.asarray(feat_mean, dtype=np.float32),
        "feat_std": np.asarray(feat_std, dtype=np.float32),
        "time_grid": time_grid,
    }

    runtime_counts = {
        "train_order_count_full": int(train_source_rows_full.size),
        "train_order_count": int(train_source_rows.size),
        "val_order_count": int(val_source_rows.size),
        "test_order_count": int(test_source_rows.size),
        "train_sample_count_full": int(train_sample_idx_full.shape[0]),
        "val_sample_count_full": int(val_sample_idx_full.shape[0]),
        "test_sample_count_full": int(test_sample_idx_full.shape[0]),
        "train_sample_count": int(train_sample_idx.shape[0]),
        "val_sample_count": int(val_sample_idx.shape[0]),
        "test_sample_count": int(test_sample_idx.shape[0]),
        "feature_dim_total": int(order_store.lob_dim + order_store.tox_dim + 2),
        "sequence_length": int(order_store.lookback_steps),
        "output_steps": int(time_grid.shape[0]),
    }

    return runtime_artifacts, runtime_counts


def _manifest_to_dataframe(sample_manifest) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_ptr": sample_manifest.order_ptr.astype(np.int32, copy=False),
            "end_idx": sample_manifest.end_idx.astype(np.int32, copy=False),
            "y": sample_manifest.y.astype(np.float32, copy=False),
            "d": sample_manifest.d.astype(np.int64, copy=False),
            "order_ids": sample_manifest.order_ids.astype(np.int64, copy=False),
            "entry_times": sample_manifest.entry_times.astype(np.int64, copy=False),
            "source_row_idx": sample_manifest.source_row_idx.astype(np.int64, copy=False),
            "update_idx": sample_manifest.update_idx.astype(np.int32, copy=False),
        }
    )


def main() -> None:
    args = _parse_args()

    input_path = _resolve_input_path(args)
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.datasets_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_prefix = args.output_prefix or input_path.stem
    manifest_prefix = args.manifest_prefix or output_prefix

    train_path, val_path, test_path = _resolve_output_paths(
        output_dir=output_dir,
        output_prefix=output_prefix,
    )
    (
        manifest_table_path,
        order_store_pickle_path,
        sample_manifest_parquet_path,
        manifest_meta_path,
        preprocessed_npz_path,
    ) = _resolve_manifest_output_paths(
        output_dir=output_dir,
        manifest_prefix=manifest_prefix,
    )

    outputs_to_check = [train_path, val_path, test_path]
    if not args.skip_dynamic_manifest:
        outputs_to_check.extend(
            [
                manifest_table_path,
                order_store_pickle_path,
                sample_manifest_parquet_path,
                manifest_meta_path,
                preprocessed_npz_path,
            ]
        )
    _ensure_writable(outputs_to_check, overwrite=bool(args.overwrite))

    print(f"Loading: {input_path}")
    df_raw = pd.read_parquet(input_path)
    print(f"Loaded rows: {len(df_raw):,}")

    df = _normalize_notebook_columns(df_raw)

    train_mask, val_mask, test_mask, train_days, val_days, test_days = _compute_day_boundary_splits(df)

    train_df = df.loc[train_mask].copy()
    val_df = df.loc[val_mask].copy()
    test_df = df.loc[test_mask].copy()

    train_df["dataset_split"] = "train"
    val_df["dataset_split"] = "val"
    test_df["dataset_split"] = "test"

    total_rows = len(df)
    split_rows = len(train_df) + len(val_df) + len(test_df)
    if split_rows != total_rows:
        raise RuntimeError(
            f"Row-count mismatch after split: total={total_rows}, split_sum={split_rows}"
        )

    print("Day-boundary split summary:")
    print(f"  Train: {len(train_df):,} rows | days { _format_day_span(train_days) }")
    print(f"  Val  : {len(val_df):,} rows | days { _format_day_span(val_days) }")
    print(f"  Test : {len(test_df):,} rows | days { _format_day_span(test_days) }")

    print("Writing output parquets...")
    train_df.to_parquet(train_path, compression=args.compression)
    val_df.to_parquet(val_path, compression=args.compression)
    test_df.to_parquet(test_path, compression=args.compression)

    if args.skip_dynamic_manifest:
        print("Skipping dynamic-manifest build (--skip-dynamic-manifest).")
        print("Done.")
        print(f"  Train parquet: {train_path}")
        print(f"  Val parquet  : {val_path}")
        print(f"  Test parquet : {test_path}")
        return

    df_for_dynamic = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True)
    (
        order_store,
        sample_manifest,
        t_max,
        effective_tox_time_delta_col,
    ) = _build_dynamic_manifest_uncapped(
        df_split=df_for_dynamic,
        lookback_steps=int(args.lookback_steps),
        lob_feature_dim=int(args.lob_feature_dim),
        tox_feature_dim=int(args.tox_feature_dim),
        horizon_quantile=float(args.horizon_quantile),
        tox_time_delta_col=args.tox_time_delta_col,
    )

    print(f"Dynamic samples in uncapped manifest: {len(sample_manifest):,}")
    manifest_event_counts = pd.Series(sample_manifest.d).value_counts().sort_index()
    print("Manifest event distribution (per-snapshot admin censoring):")
    for code in sorted(manifest_event_counts.index):
        print(f"  {int(code)}: {int(manifest_event_counts[code]):>8}")

    manifest_order_counts = pd.Series(sample_manifest.source_row_idx).value_counts().to_numpy()
    if manifest_order_counts.size > 0:
        print(
            "Manifest samples/order-instance stats:",
            {
                "min": int(manifest_order_counts.min()),
                "p50": int(np.percentile(manifest_order_counts, 50)),
                "p90": int(np.percentile(manifest_order_counts, 90)),
                "max": int(manifest_order_counts.max()),
            },
        )
    else:
        print("Manifest samples/order stats unavailable: no dynamic samples generated.")

    print("Writing dynamic manifest artifacts...")
    manifest_df = _manifest_to_dataframe(sample_manifest)
    manifest_df.to_parquet(manifest_table_path, compression=args.compression)
    manifest_df.to_parquet(sample_manifest_parquet_path, compression=args.compression)

    with open(order_store_pickle_path, "wb") as f:
        pickle.dump(order_store, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Preparing runtime preprocessing artifacts...")
    runtime_artifacts, runtime_counts = _prepare_runtime_artifacts(
        df_split=df_for_dynamic,
        order_store=order_store,
        sample_manifest=sample_manifest,
        t_max=float(t_max),
        train_order_subsample_fraction=float(args.train_order_subsample_fraction),
        train_order_subsample_seed=int(args.train_order_subsample_seed),
        max_samples_per_source_row=int(args.max_samples_per_source_row),
        split_cap_random_seed=int(args.split_cap_random_seed),
        num_time_steps=int(args.num_time_steps),
        normalizer_chunk_size=int(args.normalizer_chunk_size),
    )
    np.savez_compressed(preprocessed_npz_path, **runtime_artifacts)

    manifest_meta = {
        "input_path": str(input_path),
        "train_parquet_path": str(train_path),
        "val_parquet_path": str(val_path),
        "test_parquet_path": str(test_path),
        "lookback_steps": int(args.lookback_steps),
        "lob_feature_dim": int(args.lob_feature_dim),
        "tox_feature_dim": int(args.tox_feature_dim),
        "horizon_quantile": float(args.horizon_quantile),
        "t_max": float(t_max),
        "tox_time_delta_col": int(effective_tox_time_delta_col),
        "max_samples_per_order": None,
        "n_orders": int(len(df_for_dynamic)),
        "n_train_orders": int(len(train_df)),
        "n_val_orders": int(len(val_df)),
        "n_test_orders": int(len(test_df)),
        "n_manifest_samples": int(len(sample_manifest)),
        "manifest_table_path": str(manifest_table_path),
        "order_store_pickle_path": str(order_store_pickle_path),
        "sample_manifest_parquet_path": str(sample_manifest_parquet_path),
        "dynamic_preprocessed_path": str(preprocessed_npz_path),
        "preprocessing_contract_version": PREPROCESSING_CONTRACT_VERSION,
        "preprocessing_config": {
            "train_order_subsample_fraction": float(args.train_order_subsample_fraction),
            "train_order_subsample_seed": int(args.train_order_subsample_seed),
            "max_samples_per_source_row": int(args.max_samples_per_source_row),
            "split_cap_random_seed": int(args.split_cap_random_seed),
            "num_time_steps": int(args.num_time_steps),
            "normalizer_chunk_size": int(args.normalizer_chunk_size),
        },
        "runtime_counts": {
            "train_order_count_full": runtime_counts["train_order_count_full"],
            "train_order_count": runtime_counts["train_order_count"],
            "val_order_count": runtime_counts["val_order_count"],
            "test_order_count": runtime_counts["test_order_count"],
            "train_sample_count_full": runtime_counts["train_sample_count_full"],
            "val_sample_count_full": runtime_counts["val_sample_count_full"],
            "test_sample_count_full": runtime_counts["test_sample_count_full"],
            "train_sample_count": runtime_counts["train_sample_count"],
            "val_sample_count": runtime_counts["val_sample_count"],
            "test_sample_count": runtime_counts["test_sample_count"],
            "feature_dim_total": runtime_counts["feature_dim_total"],
            "sequence_length": runtime_counts["sequence_length"],
            "output_steps": runtime_counts["output_steps"],
        },
    }
    with open(manifest_meta_path, "w", encoding="utf-8") as f:
        json.dump(manifest_meta, f, indent=2)

    print("Done.")
    print(f"  Train parquet: {train_path}")
    print(f"  Val parquet  : {val_path}")
    print(f"  Test parquet : {test_path}")
    print(f"  Manifest parquet: {manifest_table_path}")
    print(f"  Sample manifest parquet: {sample_manifest_parquet_path}")
    print(f"  Order store pickle: {order_store_pickle_path}")
    print(f"  Runtime preprocessed NPZ: {preprocessed_npz_path}")
    print(f"  Manifest metadata: {manifest_meta_path}")


if __name__ == "__main__":
    main()
