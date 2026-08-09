from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


LATENCY_RE = re.compile(
    r"^raw_latency_(?P<latency>[0-9]+(?:p[0-9]+)?)ms_(?:backtest|summary)"
)
DEFAULT_TIME_WEIGHT_MIN_HORIZON_MS = 0.001
NS_PER_MS = 1_000_000.0
DEFAULT_BOOTSTRAP_METRICS = (
    "mean_is_bps",
    "median_is_bps",
    "time_weighted_mean_is_bps",
    "toxic_cost_mean_bps",
    "fill_adjusted_toxic_cost_mean_bps",
)


def sweep_result_dir(
    output_root: str | Path,
    *,
    ticker: str,
    model_type: str,
    lifecycle_aware: bool,
) -> Path:
    lifecycle_dir = "lifecycle" if lifecycle_aware else "no_lifecycle"
    return Path(output_root) / ticker.upper() / model_type / lifecycle_dir


def load_sweep_summary(
    output_root: str | Path,
    *,
    ticker: str,
    model_type: str,
    lifecycle_aware: bool = True,
) -> pd.DataFrame:
    """Load a finished combined summary, or combine finished per-latency summaries."""
    result_dir = sweep_result_dir(
        output_root,
        ticker=ticker,
        model_type=model_type,
        lifecycle_aware=lifecycle_aware,
    )
    combined_path = result_dir / "raw_latency_sweep_summary.csv"
    if combined_path.exists():
        df = pd.read_csv(combined_path)
    else:
        paths = sorted(result_dir.glob("raw_latency_*ms_summary.csv"))
        if not paths:
            raise FileNotFoundError(f"No summary CSVs found under {result_dir}")
        frames = [_read_latency_summary(path) for path in paths]
        df = pd.concat(frames, ignore_index=True, sort=False)

    if "latency_ms" not in df.columns:
        df["latency_ms"] = df.get("summary_path", pd.Series(index=df.index)).map(
            _latency_from_path
        )
    return _sort_by_latency(df)


def list_raw_backtest_paths(
    output_root: str | Path,
    *,
    ticker: str,
    model_type: str,
    lifecycle_aware: bool = True,
) -> list[Path]:
    result_dir = sweep_result_dir(
        output_root,
        ticker=ticker,
        model_type=model_type,
        lifecycle_aware=lifecycle_aware,
    )
    paths = sorted(
        result_dir.glob("raw_latency_*ms_backtest.parquet"),
        key=_latency_from_path,
    )
    bad_paths = [path for path in paths if not np.isfinite(_latency_from_path(path))]
    if bad_paths:
        examples = ", ".join(str(path.name) for path in bad_paths[:5])
        raise ValueError(
            "Could not parse latency from raw parquet filename(s): "
            f"{examples}. Expected names like raw_latency_0p01ms_backtest.parquet."
        )
    return paths


def bootstrap_sweep_raw_parquets(
    output_root: str | Path,
    *,
    ticker: str,
    model_type: str,
    lifecycle_aware: bool = True,
    sample_size: int | None = None,
    n_trials: int = 1000,
    min_horizon_ms: float = DEFAULT_TIME_WEIGHT_MIN_HORIZON_MS,
    random_state: int | None = 42,
    progress: bool = True,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    rng = np.random.default_rng(random_state)
    paths = list_raw_backtest_paths(
        output_root,
        ticker=ticker,
        model_type=model_type,
        lifecycle_aware=lifecycle_aware,
    )
    if not paths:
        result_dir = sweep_result_dir(
            output_root,
            ticker=ticker,
            model_type=model_type,
            lifecycle_aware=lifecycle_aware,
        )
        raise FileNotFoundError(f"No raw backtest parquet files found under {result_dir}")

    path_iter = _maybe_tqdm(paths, enabled=progress, desc="Bootstrap parquets")
    for path in path_iter:
        df = _prepare_backtest_rows(pd.read_parquet(path))
        if df.empty:
            continue
        n = len(df) if sample_size is None else int(sample_size)
        n = max(n, 1)
        latency_ms = _latency_from_path(path)
        if not np.isfinite(latency_ms):
            raise ValueError(f"Could not parse latency from raw parquet filename: {path}")

        trial_iter = _maybe_tqdm(
            range(int(n_trials)),
            enabled=progress,
            desc=f"{path.stem}",
            leave=False,
        )
        for trial in trial_iter:
            sample_idx = rng.integers(0, len(df), size=n)
            sample = df.iloc[sample_idx]
            is_values = _valid_is_values(sample)
            rows.append(
                {
                    "latency_ms": latency_ms,
                    "trial": trial,
                    "sample_size": n,
                    "raw_backtest_path": str(path),
                    "mean_is_bps": float(is_values.mean()) if len(is_values) else np.nan,
                    "median_is_bps": float(is_values.median()) if len(is_values) else np.nan,
                    "time_weighted_mean_is_bps": _time_weighted_shortfall(
                        sample,
                        min_horizon_ms=float(min_horizon_ms),
                    ),
                    "toxic_cost_mean_bps": _mean_cost_type_shortfall(
                        sample,
                        cost_type="toxic_cost",
                    ),
                    "fill_adjusted_toxic_cost_mean_bps": _mean_numeric_column(
                        sample,
                        "fill_adjusted_toxic_cost_bps",
                    ),
                }
            )
    if not rows:
        raise FileNotFoundError("No usable raw backtest parquet rows found.")
    return _sort_by_latency(pd.DataFrame(rows))


def summarize_bootstrap(
    boot: pd.DataFrame,
    *,
    metrics: Iterable[str] = DEFAULT_BOOTSTRAP_METRICS,
    ci: float = 0.95,
) -> pd.DataFrame:
    if boot.empty:
        return pd.DataFrame(
            columns=[
                "latency_ms",
                *[
                    f"{metric}_{suffix}"
                    for metric in metrics
                    for suffix in ("mean", "ci_low", "ci_high")
                ],
            ]
        )
    if "latency_ms" not in boot.columns:
        raise ValueError("Bootstrap frame is missing latency_ms.")
    boot = boot.copy()
    boot["latency_ms"] = pd.to_numeric(boot["latency_ms"], errors="coerce")
    boot = boot[np.isfinite(boot["latency_ms"])]
    if boot.empty:
        raise ValueError(
            "Bootstrap frame has no finite latency_ms values. Check raw parquet filenames."
        )

    alpha = (1.0 - float(ci)) / 2.0
    rows: list[dict[str, float | str]] = []
    for latency_ms, group in boot.groupby("latency_ms", sort=True):
        row: dict[str, float | str] = {"latency_ms": float(latency_ms)}
        for metric in metrics:
            vals = pd.to_numeric(group[metric], errors="coerce")
            vals = vals[np.isfinite(vals)]
            row[f"{metric}_mean"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{metric}_ci_low"] = (
                float(vals.quantile(alpha)) if len(vals) else np.nan
            )
            row[f"{metric}_ci_high"] = (
                float(vals.quantile(1.0 - alpha)) if len(vals) else np.nan
            )
        rows.append(row)
    if not rows:
        return pd.DataFrame({"latency_ms": pd.Series(dtype=float)})
    return pd.DataFrame(rows).sort_values("latency_ms").reset_index(drop=True)


def bootstrap_latency_delta_summary(
    output_root: str | Path,
    *,
    tickers: Sequence[str],
    model_type: str,
    lifecycle_aware: bool = True,
    metric: str = "median_is_bps",
    metrics: Sequence[str] | None = None,
    baseline_latency_ms: float = 0.0,
    sample_size: int | None = None,
    n_trials: int = 1000,
    min_horizon_ms: float = DEFAULT_TIME_WEIGHT_MIN_HORIZON_MS,
    random_state: int | None = 42,
    progress: bool = True,
) -> pd.DataFrame:
    """Bootstrap each ticker and express selected metrics relative to baseline latency.

    The returned frame has one row per ticker/latency with the bootstrap mean,
    confidence interval, baseline value, and delta from the baseline. By default
    it computes mean IS, median IS, time-weighted mean IS, toxic-cost mean, and
    fill-adjusted toxic-cost mean together so plots can switch metrics without
    re-running the bootstrap.
    """
    metric_names = tuple(metrics or DEFAULT_BOOTSTRAP_METRICS)
    if metric not in metric_names:
        metric_names = (*metric_names, metric)

    rows: list[pd.DataFrame] = []
    ticker_iter = _maybe_tqdm(
        [str(t).upper() for t in tickers],
        enabled=progress,
        desc=f"{model_type} tickers",
    )
    for ticker in ticker_iter:
        boot = bootstrap_sweep_raw_parquets(
            output_root,
            ticker=ticker,
            model_type=model_type,
            lifecycle_aware=lifecycle_aware,
            sample_size=sample_size,
            n_trials=n_trials,
            min_horizon_ms=min_horizon_ms,
            random_state=random_state,
            progress=progress,
        )
        summary = summarize_bootstrap(boot, metrics=metric_names)
        summary.insert(0, "ticker", ticker)
        summary.insert(1, "model_type", model_type)
        rows.append(_add_metric_deltas(summary, metric_names, baseline_latency_ms))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True, sort=False)


def build_latency_delta_panel_data(
    panel_specs: Sequence[dict[str, Any]],
    *,
    output_root: str | Path,
    lifecycle_aware: bool = True,
    metric: str = "median_is_bps",
    metrics: Sequence[str] | None = None,
    baseline_latency_ms: float = 0.0,
    sample_size: int | None = None,
    n_trials: int = 1000,
    min_horizon_ms: float = DEFAULT_TIME_WEIGHT_MIN_HORIZON_MS,
    random_state: int | None = 42,
    progress: bool = True,
) -> pd.DataFrame:
    """Generate bootstrap latency-delta data for a multi-panel figure.

    By default, computes baseline and delta columns for mean IS, median IS,
    time-weighted mean IS, toxic-cost mean, and fill-adjusted toxic-cost mean
    in one pass.
    """
    if len(panel_specs) != 4:
        raise ValueError("panel_specs must contain exactly four panels for a 2x2 figure.")

    all_frames: list[pd.DataFrame] = []
    for panel_idx, spec in enumerate(panel_specs):
        model_type = str(spec["model_type"])
        tickers = [str(t).upper() for t in spec["tickers"]]
        title = str(spec.get("title") or f"{model_type}: {', '.join(tickers)}")

        panel_output_root = Path(output_root)
        if model_type == "mamba":
            panel_output_root = panel_output_root.with_name(panel_output_root.name + "_gpu")
        panel_df = bootstrap_latency_delta_summary(
            panel_output_root,
            tickers=tickers,
            model_type=model_type,
            lifecycle_aware=lifecycle_aware,
            metric=metric,
            metrics=metrics,
            baseline_latency_ms=baseline_latency_ms,
            sample_size=sample_size,
            n_trials=n_trials,
            min_horizon_ms=min_horizon_ms,
            random_state=random_state,
            progress=progress,
        )
        panel_df.insert(0, "panel_index", int(panel_idx))
        panel_df.insert(1, "panel_title", title)
        all_frames.append(panel_df)

    return pd.concat(all_frames, ignore_index=True, sort=False) if all_frames else pd.DataFrame()


def plot_latency_delta_panel_data(
    delta_df: pd.DataFrame,
    *,
    metric: str = "median_is_bps",
    figsize: tuple[float, float] = (11.0, 7.5),
    background_alpha: float = 0.25,
    average_color: str = "black",
    linthresh: float = 0.01,
    sharey: bool = True,
    common_y_limits: bool = True,
    y_limits: tuple[float, float] | None = None,
    legend_y: float = 0.94,
    layout_top: float = 0.88,
):
    """Plot a compact 2x2 latency-delta figure from precomputed bootstrap data."""
    import matplotlib.pyplot as plt

    if delta_df.empty:
        raise ValueError("delta_df is empty.")
    required = {"panel_index", "panel_title", "ticker", "latency_ms", f"{metric}_delta"}
    missing = required.difference(delta_df.columns)
    if missing:
        raise ValueError(f"delta_df is missing required columns: {sorted(missing)}")

    panel_ids = sorted(pd.unique(delta_df["panel_index"]))
    if len(panel_ids) != 4:
        raise ValueError(f"Expected exactly four panels, found {len(panel_ids)}.")

    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True, sharey=sharey)
    axes_flat = axes.reshape(-1)
    value_col = f"{metric}_delta"
    resolved_y_limits = (
        tuple(float(v) for v in y_limits)
        if y_limits is not None
        else _shared_y_limits(delta_df[value_col]) if common_y_limits else None
    )

    for ax, panel_id in zip(axes_flat, panel_ids):
        panel_df = delta_df[delta_df["panel_index"].eq(panel_id)].copy()
        panel_df["latency_ms"] = pd.to_numeric(panel_df["latency_ms"], errors="coerce")
        panel_df[value_col] = pd.to_numeric(panel_df[value_col], errors="coerce")
        panel_df = panel_df[np.isfinite(panel_df["latency_ms"]) & np.isfinite(panel_df[value_col])]
        title = str(panel_df["panel_title"].iloc[0])

        for ticker, group in panel_df.groupby("ticker", sort=True):
            group = group.sort_values("latency_ms")
            ax.plot(
                group["latency_ms"],
                group[value_col],
                marker="o",
                linewidth=1.0,
                alpha=background_alpha,
                label=ticker,
            )

        avg = (
            panel_df.groupby("latency_ms", as_index=False)[value_col]
            .mean()
            .sort_values("latency_ms")
        )
        ax.plot(
            avg["latency_ms"],
            avg[value_col],
            marker="o",
            linewidth=2.4,
            color=average_color,
            label="Average",
        )
        ax.axhline(0.0, color="0.5", linewidth=0.8, linestyle="--", alpha=0.7)
        ax.set_title(title)
        ax.set_xscale("symlog", linthresh=linthresh)
        if resolved_y_limits is not None:
            ax.set_ylim(*resolved_y_limits)
        ax.grid(True, alpha=0.25)

    metric_label = _metric_display_name(metric)
    for ax in axes[:, 0]:
        ax.set_ylabel(rf"$\Delta$ {metric_label} (bps)")
    for ax in axes[-1, :]:
        ax.set_xlabel("Injected latency (ms)")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, float(legend_y)),
        ncol=min(len(labels), 6),
        frameon=False,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, float(layout_top)))
    return fig, axes


def plot_latency_delta_panel_set(
    panel_specs: Sequence[dict[str, Any]],
    *,
    output_root: str | Path,
    lifecycle_aware: bool = True,
    metric: str = "median_is_bps",
    metrics: Sequence[str] | None = None,
    baseline_latency_ms: float = 0.0,
    sample_size: int | None = None,
    n_trials: int = 1000,
    min_horizon_ms: float = DEFAULT_TIME_WEIGHT_MIN_HORIZON_MS,
    random_state: int | None = 42,
    progress: bool = True,
    figsize: tuple[float, float] = (11.0, 7.5),
    background_alpha: float = 0.25,
    average_color: str = "black",
    linthresh: float = 0.01,
    sharey: bool = True,
    common_y_limits: bool = True,
    y_limits: tuple[float, float] | None = None,
    legend_y: float = 0.94,
    layout_top: float = 0.88,
):
    """Generate and plot a compact 2x2 latency-delta figure.

    ``panel_specs`` entries should contain ``model_type`` and ``tickers``; an
    optional ``title`` overrides the default title. Each panel draws one faint
    line per ticker and a solid cross-ticker average in front.
    """
    combined = build_latency_delta_panel_data(
        panel_specs,
        output_root=output_root,
        lifecycle_aware=lifecycle_aware,
        metric=metric,
        metrics=metrics,
        baseline_latency_ms=baseline_latency_ms,
        sample_size=sample_size,
        n_trials=n_trials,
        min_horizon_ms=min_horizon_ms,
        random_state=random_state,
        progress=progress,
    )
    fig, axes = plot_latency_delta_panel_data(
        combined,
        metric=metric,
        figsize=figsize,
        background_alpha=background_alpha,
        average_color=average_color,
        linthresh=linthresh,
        sharey=sharey,
        common_y_limits=common_y_limits,
        y_limits=y_limits,
        legend_y=legend_y,
        layout_top=layout_top,
    )
    return fig, axes, combined


def _read_latency_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.insert(0, "summary_path", str(path))
    if "latency_ms" not in df.columns:
        df.insert(0, "latency_ms", _latency_from_path(path))
    return df


def _add_metric_delta(
    summary: pd.DataFrame,
    metric: str,
    baseline_latency_ms: float,
) -> pd.DataFrame:
    mean_col = f"{metric}_mean"
    if mean_col not in summary.columns:
        raise ValueError(f"Bootstrap summary is missing {mean_col}.")
    out = summary.copy()
    out["latency_ms"] = pd.to_numeric(out["latency_ms"], errors="coerce")
    baseline_mask = np.isclose(
        out["latency_ms"].astype(float),
        float(baseline_latency_ms),
        rtol=0.0,
        atol=1e-9,
    )
    if not baseline_mask.any():
        available = ", ".join(f"{v:g}" for v in sorted(out["latency_ms"].dropna().unique()))
        raise ValueError(
            f"No baseline latency {baseline_latency_ms:g} ms found. "
            f"Available latencies: {available}"
        )
    baseline_value = float(out.loc[baseline_mask, mean_col].iloc[0])
    out[f"{metric}_baseline"] = baseline_value
    out[f"{metric}_delta"] = pd.to_numeric(out[mean_col], errors="coerce") - baseline_value
    return out


def _add_metric_deltas(
    summary: pd.DataFrame,
    metrics: Iterable[str],
    baseline_latency_ms: float,
) -> pd.DataFrame:
    out = summary.copy()
    for metric in metrics:
        out = _add_metric_delta(out, metric, baseline_latency_ms)
    return out


def _shared_y_limits(values: pd.Series, *, pad_fraction: float = 0.08) -> tuple[float, float] | None:
    vals = pd.to_numeric(values, errors="coerce")
    vals = vals[np.isfinite(vals)]
    if vals.empty:
        return None
    lo = float(vals.min())
    hi = float(vals.max())
    if np.isclose(lo, hi):
        pad = max(abs(lo) * pad_fraction, 1e-6)
    else:
        pad = (hi - lo) * pad_fraction
    return lo - pad, hi + pad


def _metric_display_name(metric: str) -> str:
    labels = {
        "median_is_bps": "Median IS",
        "mean_is_bps": "Mean IS",
        "time_weighted_mean_is_bps": "Time-weighted mean IS",
        "toxic_cost_mean_bps": "Mean toxic cost",
        "fill_adjusted_toxic_cost_mean_bps": "Fill-adjusted mean toxic cost",
    }
    return labels.get(metric, metric.replace("_", " "))


def _prepare_backtest_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "implementation_shortfall_bps" not in df.columns:
        raise ValueError("Raw backtest parquet is missing implementation_shortfall_bps")
    out = df.copy()
    out["implementation_shortfall_bps"] = pd.to_numeric(
        out["implementation_shortfall_bps"],
        errors="coerce",
    )
    if "fill_adjusted_toxic_cost_bps" in out.columns:
        out["fill_adjusted_toxic_cost_bps"] = pd.to_numeric(
            out["fill_adjusted_toxic_cost_bps"],
            errors="coerce",
        )
    return out.reset_index(drop=True)


def _valid_is_values(df: pd.DataFrame) -> pd.Series:
    if "implementation_shortfall_bps" not in df.columns:
        return pd.Series(dtype=float)
    valid = df
    if "metric_status" in valid.columns:
        valid = valid[valid["metric_status"].fillna("").eq("ok")]
    values = pd.to_numeric(valid["implementation_shortfall_bps"], errors="coerce")
    return values[np.isfinite(values)]


def _time_weighted_shortfall(df: pd.DataFrame, *, min_horizon_ms: float) -> float:
    if df.empty or "implementation_shortfall_bps" not in df or "cost_type" not in df:
        return np.nan
    valid = df.copy()
    if "metric_status" in valid.columns:
        valid = valid[valid["metric_status"].fillna("").eq("ok")]
    valid["implementation_shortfall_bps"] = pd.to_numeric(
        valid["implementation_shortfall_bps"],
        errors="coerce",
    )
    valid = valid[np.isfinite(valid["implementation_shortfall_bps"])]
    if valid.empty:
        return np.nan

    horizons_ms = _measurement_horizon_ms(valid)
    effective_horizons_ms = horizons_ms.clip(lower=float(min_horizon_ms))
    weights = 1.0 / effective_horizons_ms
    mask = np.isfinite(weights) & np.isfinite(valid["implementation_shortfall_bps"])
    if not mask.any():
        return np.nan
    weights = _normalize_weights(weights.loc[mask])
    return _weighted_mean(valid.loc[mask, "implementation_shortfall_bps"], weights)


def _mean_cost_type_shortfall(df: pd.DataFrame, *, cost_type: str) -> float:
    if df.empty or "implementation_shortfall_bps" not in df or "cost_type" not in df:
        return np.nan
    valid = df.copy()
    if "metric_status" in valid.columns:
        valid = valid[valid["metric_status"].fillna("").eq("ok")]
    valid = valid[valid["cost_type"].eq(cost_type)].copy()
    if valid.empty:
        return np.nan
    values = pd.to_numeric(valid["implementation_shortfall_bps"], errors="coerce")
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else np.nan


def _mean_numeric_column(df: pd.DataFrame, column: str) -> float:
    if df.empty or column not in df.columns:
        return np.nan
    values = pd.to_numeric(df[column], errors="coerce")
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else np.nan


def _measurement_horizon_ms(df: pd.DataFrame) -> pd.Series:
    horizons = pd.Series(np.nan, index=df.index, dtype=float)

    toxic_mask = df["cost_type"].eq("toxic_cost")
    if "selected_window_ms" in df.columns:
        horizons.loc[toxic_mask] = pd.to_numeric(
            df.loc[toxic_mask, "selected_window_ms"],
            errors="coerce",
        )

    opportunity_mask = df["cost_type"].eq("opportunity_cost")
    if opportunity_mask.any():
        start_ns = _opportunity_start_ns(df.loc[opportunity_mask])
        end_ns = _numeric_col(df.loc[opportunity_mask], "decision_end_time")
        horizons.loc[opportunity_mask] = (end_ns - start_ns) / NS_PER_MS

    return horizons


def _opportunity_start_ns(df: pd.DataFrame) -> pd.Series:
    submitted = _bool_col(df, "submitted")
    submit_time = _numeric_col(df, "decision_submit_time")
    effective_time = _numeric_col(df, "decision_decision_effective_time")
    observation_time = _numeric_col(df, "decision_observation_time")

    start = observation_time.copy()
    start = start.where(~submitted | submit_time.isna(), submit_time)
    start = start.where(submitted | effective_time.isna(), effective_time)
    return start


def _numeric_col(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def _bool_col(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(False, index=df.index)
    return df[column].fillna(False).astype(bool)


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce")
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not mask.any():
        return np.nan
    return float(np.average(values.loc[mask], weights=weights.loc[mask]))


def _normalize_weights(weights: pd.Series) -> pd.Series:
    total = float(weights.sum(skipna=True))
    if not np.isfinite(total) or total <= 0.0:
        return weights * np.nan
    return weights / total


def _latency_from_path(path: str | Path | float | int | None) -> float:
    if path is None or (isinstance(path, float) and np.isnan(path)):
        return np.nan
    name = Path(path).name if isinstance(path, (str, Path)) else str(path)
    match = LATENCY_RE.search(name)
    if not match:
        return np.nan
    token = match.group("latency").replace("p", ".")
    try:
        return float(token)
    except ValueError:
        return np.nan


def _sort_by_latency(df: pd.DataFrame) -> pd.DataFrame:
    if "latency_ms" in df.columns:
        return df.sort_values("latency_ms").reset_index(drop=True)
    return df.reset_index(drop=True)


def _maybe_tqdm(iterable, *, enabled: bool, desc: str, leave: bool = True):
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm
    except Exception:
        return iterable
    return tqdm(iterable, desc=desc, leave=leave)
