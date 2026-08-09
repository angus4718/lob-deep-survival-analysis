"""Run final standard-engine backtests from selected DeepHit toxic-CIF settings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.data import BacktestDataset
from src.backtest.engine import BacktestEngine
from src.backtest.metrics import ImplementationShortfallMetric
from src.backtest.reports import (
    DEFAULT_TIME_WEIGHT_MIN_HORIZON_MS,
    _time_weighted_shortfall,
)
from src.backtest.strategies.baseline import AlwaysPlaceLimitOrderStrategy


DATA_START_DATE = "2025-10-01"
DATA_END_DATE = "2026-01-01"
DEFAULT_MODEL_TYPES = ("gru", "gru_transformer", "transformer", "mamba")
BOOTSTRAP_SUMMARY_METRICS = (
    "mean_is_bps",
    "median_is_bps",
    "time_weighted_mean_is_bps",
    "toxic_cost_mean_bps",
    "fill_adjusted_toxic_cost_mean_bps",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run final backtest evaluation on held-out test rows after the grid-search "
            "prefix. One invocation handles one ticker and loops through all model "
            "types plus the always-place baseline."
        )
    )
    parser.add_argument("--ticker", required=True, help="Ticker symbol, e.g. AAPL.")
    parser.add_argument(
        "--model-types",
        default=",".join(DEFAULT_MODEL_TYPES),
        help="Comma-separated model types to evaluate before the baseline.",
    )
    parser.add_argument(
        "--settings-csv",
        default="reports/final_backtest/optimal_toxic_cif_strategy_settings.csv",
        help="CSV exported by bt_grid_search_toxic_cif_analysis.ipynb.",
    )
    parser.add_argument(
        "--data-root",
        default="data",
        help="Root containing datasets/.",
    )
    parser.add_argument(
        "--artifact-root",
        default="checkpoints",
        help="Root containing [ticker]/[model_type]_[ticker].pt files.",
    )
    parser.add_argument(
        "--output-root",
        default="reports/final_backtest_toxic_cif",
        help="Root where final raw, summary, and bootstrap CSVs are written.",
    )
    parser.add_argument("--data-start-date", default=DATA_START_DATE)
    parser.add_argument("--data-end-date", default=DATA_END_DATE)
    parser.add_argument("--row-offset", type=int, default=4000)
    parser.add_argument("--row-limit", type=int, default=5000)
    parser.add_argument("--snapshot-policy", default="entry")
    parser.add_argument("--lifecycle-aware", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lifecycle-stride", type=int, default=50)
    parser.add_argument("--lifecycle-max-evaluations", type=int, default=None)
    parser.add_argument("--device", default=None, help="Optional torch device for model runs.")
    parser.add_argument("--bootstrap-trials", type=int, default=1000)
    parser.add_argument(
        "--bootstrap-models",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Bootstrap raw results for model strategies and the baseline. "
            "Use --no-bootstrap-models for quick evaluation."
        ),
    )
    parser.add_argument(
        "--bootstrap-sample-size",
        type=int,
        default=None,
        help="Rows sampled with replacement per bootstrap trial. Default: raw row count.",
    )
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument(
        "--time-weight-min-horizon-ms",
        type=float,
        default=DEFAULT_TIME_WEIGHT_MIN_HORIZON_MS,
    )
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip a run if its summary CSV already exists.",
    )
    return parser.parse_args()


def compact_date(date: str) -> str:
    return str(date).replace("-", "")


def parse_model_types(raw: str) -> list[str]:
    values = [part.strip() for part in raw.replace(" ", ",").split(",") if part.strip()]
    if not values:
        raise ValueError("--model-types must contain at least one model type.")
    return values


def dataset_paths(args: argparse.Namespace, ticker: str) -> tuple[Path, Path]:
    root = Path(args.data_root)
    start = compact_date(args.data_start_date)
    end = compact_date(args.data_end_date)
    stem = f"labeled_dataset_XNAS_ITCH_{ticker}_mbo_{start}_{end}"
    return (
        root / "datasets" / f"{stem}_test.parquet",
        root / "datasets" / f"{stem}_dynamic_preprocessed.npz",
    )


def model_path(args: argparse.Namespace, ticker: str, model_type: str) -> Path:
    return Path(args.artifact_root) / ticker / f"{model_type}_{ticker}.pt"


def load_settings(path: str | Path) -> pd.DataFrame:
    settings = pd.read_csv(path)
    required = {
        "ticker",
        "model_type",
        "max_toxic_cif",
        "horizon_index",
    }
    missing = required - set(settings.columns)
    if missing:
        raise ValueError(f"Settings CSV is missing required columns: {sorted(missing)}")
    settings["ticker"] = settings["ticker"].astype(str).str.upper()
    settings["model_type"] = settings["model_type"].astype(str)
    return settings


def setting_for(settings: pd.DataFrame, ticker: str, model_type: str) -> pd.Series:
    match = settings[
        settings["ticker"].eq(ticker.upper())
        & settings["model_type"].eq(str(model_type))
    ]
    if match.empty:
        raise ValueError(
            f"No selected setting found for ticker={ticker}, model_type={model_type}."
        )
    return match.iloc[0]


def build_metric(dataset: BacktestDataset) -> ImplementationShortfallMetric:
    calibration_frame = dataset.load_calibration_frame()
    if calibration_frame is None or calibration_frame.empty:
        calibration_frame = dataset.load_frame()
    return ImplementationShortfallMetric.from_labeled_orders(calibration_frame)


def run_one(
    *,
    args: argparse.Namespace,
    dataset: BacktestDataset,
    metric: ImplementationShortfallMetric,
    ticker: str,
    strategy_name: str,
    strategy,
    selected_setting: dict[str, Any] | None = None,
    bootstrap: bool,
) -> None:
    out_dir = Path(args.output_root) / ticker / strategy_name
    raw_path = out_dir / "backtest.parquet"
    summary_path = out_dir / "summary.csv"
    bootstrap_path = out_dir / "bootstrap_trials.csv"

    bootstrap_complete = (not bootstrap) or bootstrap_path.exists()
    if bool(args.skip_existing) and summary_path.exists() and bootstrap_complete:
        if bootstrap and not summary_has_bootstrap_columns(summary_path):
            summary = pd.read_csv(summary_path)
            trials = pd.read_csv(bootstrap_path)
            summary = append_bootstrap_summary(summary, trials)
            summary.to_csv(summary_path, index=False)
            print(f"[final-bt] Backfilled bootstrap mean/CI columns in {summary_path}")
        print(f"[final-bt] Skipping existing {ticker}/{strategy_name}: {summary_path}")
        return

    print(f"[final-bt] Running {ticker}/{strategy_name}")
    report = BacktestEngine(
        strategy,
        metrics=[metric],
        lifecycle_aware=bool(args.lifecycle_aware),
        lifecycle_stride=int(args.lifecycle_stride),
        lifecycle_max_evaluations=args.lifecycle_max_evaluations,
        verbose=bool(args.verbose),
        progress=bool(args.progress),
    ).run(dataset.iter_snapshots())
    report.write(raw_path, summary_path)

    summary = pd.read_csv(summary_path)
    for key, value in {
        "ticker": ticker,
        "strategy": strategy_name,
        "row_offset": int(args.row_offset),
        "row_limit": int(args.row_limit) if args.row_limit is not None else np.nan,
        **(selected_setting or {}),
    }.items():
        summary[key] = value
    summary.to_csv(summary_path, index=False)
    print(f"[final-bt] Wrote {summary_path}")

    if bootstrap:
        trials = bootstrap_raw_results(
            pd.read_parquet(raw_path),
            ticker=ticker,
            strategy=strategy_name,
            n_trials=int(args.bootstrap_trials),
            sample_size=args.bootstrap_sample_size,
            seed=int(args.bootstrap_seed),
            min_horizon_ms=float(args.time_weight_min_horizon_ms),
            selected_setting=selected_setting or {},
        )
        bootstrap_path.parent.mkdir(parents=True, exist_ok=True)
        trials.to_csv(bootstrap_path, index=False)
        print(f"[final-bt] Wrote {bootstrap_path}")
        summary = append_bootstrap_summary(summary, trials)
        summary.to_csv(summary_path, index=False)
        print(f"[final-bt] Updated {summary_path} with bootstrap mean/CI columns")


def bootstrap_raw_results(
    raw: pd.DataFrame,
    *,
    ticker: str,
    strategy: str,
    n_trials: int,
    sample_size: int | None,
    seed: int,
    min_horizon_ms: float,
    selected_setting: dict[str, Any],
) -> pd.DataFrame:
    if n_trials < 1:
        return pd.DataFrame()
    if raw.empty:
        raise ValueError("Cannot bootstrap an empty raw backtest result frame.")

    n_rows = len(raw)
    draw_size = int(sample_size) if sample_size is not None else n_rows
    if draw_size < 1:
        raise ValueError("--bootstrap-sample-size must be >= 1 or omitted.")

    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for trial in range(n_trials):
        idx = rng.integers(0, n_rows, size=draw_size)
        sample = raw.iloc[idx].reset_index(drop=True)
        rows.append(
            bootstrap_summary_row(
                sample,
                ticker=ticker,
                strategy=strategy,
                trial=trial,
                sample_size=draw_size,
                min_horizon_ms=min_horizon_ms,
                selected_setting=selected_setting,
            )
        )
    return pd.DataFrame(rows)


def bootstrap_summary_row(
    sample: pd.DataFrame,
    *,
    ticker: str,
    strategy: str,
    trial: int,
    sample_size: int,
    min_horizon_ms: float,
    selected_setting: dict[str, Any],
) -> dict[str, Any]:
    metric_ok = (
        sample["metric_status"].fillna("").eq("ok")
        if "metric_status" in sample.columns
        else pd.Series(True, index=sample.index)
    )
    submitted = _bool_col(sample, "submitted")
    filled = _bool_col(sample, "filled")
    canceled = _bool_col(sample, "canceled")
    valid = sample.loc[metric_ok].copy()
    if "implementation_shortfall_bps" in valid.columns:
        values = pd.to_numeric(valid["implementation_shortfall_bps"], errors="coerce")
    else:
        values = pd.Series(np.nan, index=valid.index, dtype=float)
    values = values[np.isfinite(values)]
    weighted = _time_weighted_shortfall(valid, min_horizon_ms=min_horizon_ms)
    toxic_values = _cost_type_values(valid, "toxic_cost")
    fill_adjusted_values = _numeric_values(sample, "fill_adjusted_toxic_cost_bps")

    row: dict[str, Any] = {
        "ticker": ticker,
        "strategy": strategy,
        "trial": int(trial),
        "sample_size": int(sample_size),
        "orders": int(len(sample)),
        "submitted": int(submitted.sum()),
        "skipped": int((~submitted).sum()),
        "filled": int((submitted & filled).sum()),
        "unfilled": int((submitted & ~filled).sum()),
        "canceled": int((submitted & canceled).sum()),
        "metric_ok": int(metric_ok.sum()),
        "metric_failed": int((~metric_ok).sum()),
        "fill_rate": float((submitted & filled).sum() / submitted.sum())
        if submitted.sum()
        else np.nan,
        "mean_is_bps": float(values.mean()) if len(values) else np.nan,
        "median_is_bps": float(values.median()) if len(values) else np.nan,
        "time_weighted_mean_is_bps": weighted["mean_bps"],
        "toxic_cost_mean_bps": (
            float(toxic_values.mean()) if len(toxic_values) else np.nan
        ),
        "fill_adjusted_toxic_cost_mean_bps": (
            float(fill_adjusted_values.mean()) if len(fill_adjusted_values) else np.nan
        ),
        "time_weight_sum": weighted["weight_sum"],
    }
    row.update(selected_setting)
    return row


def append_bootstrap_summary(
    summary: pd.DataFrame,
    trials: pd.DataFrame,
    *,
    metrics: tuple[str, ...] = BOOTSTRAP_SUMMARY_METRICS,
    ci: float = 0.95,
) -> pd.DataFrame:
    out = summary.copy()
    if trials.empty:
        return out
    alpha = (1.0 - float(ci)) / 2.0
    for metric in metrics:
        if metric not in trials.columns:
            continue
        vals = pd.to_numeric(trials[metric], errors="coerce")
        vals = vals[np.isfinite(vals)]
        out[f"bootstrap_{metric}_mean"] = float(vals.mean()) if len(vals) else np.nan
        out[f"bootstrap_{metric}_ci_low"] = (
            float(vals.quantile(alpha)) if len(vals) else np.nan
        )
        out[f"bootstrap_{metric}_ci_high"] = (
            float(vals.quantile(1.0 - alpha)) if len(vals) else np.nan
        )
    return out


def summary_has_bootstrap_columns(path: str | Path) -> bool:
    try:
        columns = set(pd.read_csv(path, nrows=0).columns)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return False
    expected = {
        f"bootstrap_{metric}_{suffix}"
        for metric in BOOTSTRAP_SUMMARY_METRICS
        for suffix in ("mean", "ci_low", "ci_high")
    }
    return expected.issubset(columns)


def _cost_type_values(df: pd.DataFrame, cost_type: str) -> pd.Series:
    if "cost_type" not in df.columns or "implementation_shortfall_bps" not in df.columns:
        return pd.Series(dtype=float)
    values = pd.to_numeric(
        df.loc[df["cost_type"].eq(cost_type), "implementation_shortfall_bps"],
        errors="coerce",
    )
    return values[np.isfinite(values)]


def _numeric_values(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(dtype=float)
    values = pd.to_numeric(df[column], errors="coerce")
    return values[np.isfinite(values)]


def _bool_col(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(False, index=df.index)
    return df[column].fillna(False).astype(bool)


def main() -> None:
    args = parse_args()
    from src.backtest.strategies.deephit import (
        DeepHitStrategy,
        DeepHitToxicCIFDecisionLogic,
    )

    ticker = args.ticker.upper()
    model_types = parse_model_types(args.model_types)
    settings = load_settings(args.settings_csv)
    test_path, runtime_npz_path = dataset_paths(args, ticker)

    print("[final-bt] Configuration:")
    print(f"  ticker={ticker}")
    print(f"  model_types={model_types}")
    print(f"  settings_csv={args.settings_csv}")
    print(f"  test_path={test_path}")
    print(f"  runtime_npz={runtime_npz_path}")
    print(f"  row_offset={args.row_offset}")
    print(f"  row_limit={args.row_limit}")

    dataset = BacktestDataset.from_runtime_artifacts(
        test_path,
        runtime_npz_path=runtime_npz_path,
        snapshot_policy=args.snapshot_policy,
        row_offset=args.row_offset,
        row_limit=args.row_limit,
    )
    metric = build_metric(dataset)

    for model_type in model_types:
        selected = setting_for(settings, ticker, model_type)
        selected_setting = {
            "model_type": model_type,
            "decision_logic": "toxic_cif",
            "max_toxic_cif": float(selected["max_toxic_cif"]),
            "horizon_index": int(selected["horizon_index"]),
            "selection_objective_metric": selected.get(
                "objective_metric",
                selected.get("objective_col"),
            ),
            "selection_objective_value": selected.get(
                "objective",
                selected.get("objective_value"),
            ),
            "selection_warning": selected.get("selection_warning"),
        }
        strategy = DeepHitStrategy.from_saved_model(
            model_path(args, ticker, model_type),
            model_name=model_type,
            decision_logic=DeepHitToxicCIFDecisionLogic(
                max_toxic_cif=selected_setting["max_toxic_cif"],
                horizon_index=selected_setting["horizon_index"],
            ),
            device=args.device,
        )
        run_one(
            args=args,
            dataset=dataset,
            metric=metric,
            ticker=ticker,
            strategy_name=model_type,
            strategy=strategy,
            selected_setting=selected_setting,
            bootstrap=bool(args.bootstrap_models),
        )

    run_one(
        args=args,
        dataset=dataset,
        metric=metric,
        ticker=ticker,
        strategy_name="baseline",
        strategy=AlwaysPlaceLimitOrderStrategy(),
        selected_setting={
            "model_type": "baseline",
            "decision_logic": "always_place_limit_order",
        },
        bootstrap=bool(args.bootstrap_models),
    )
    write_ticker_summary(Path(args.output_root), ticker)


def write_ticker_summary(output_root: Path, ticker: str) -> None:
    ticker_dir = output_root / ticker
    rows = []
    for path in sorted(ticker_dir.glob("*/summary.csv")):
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        rows.append(frame.iloc[0].to_dict())
    if not rows:
        return
    out = pd.DataFrame(rows).sort_values("strategy").reset_index(drop=True)
    out_path = ticker_dir / "summary_all_strategies.csv"
    out.to_csv(out_path, index=False)
    print(f"[final-bt] Wrote {out_path}")


if __name__ == "__main__":
    main()
