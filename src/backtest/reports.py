"""Raw and summarized backtest reports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import CONFIG

from .types import BacktestResult

NS_PER_MS = 1_000_000.0
DEFAULT_TIME_WEIGHT_MIN_HORIZON_MS = 0.001


@dataclass(frozen=True)
class BacktestReport:
    """Container for raw per-order results and aggregate summaries."""

    results: list[BacktestResult]

    def raw_frame(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for result in self.results:
            row = {
                "row_index": result.row_index,
                "order_id": result.order_id,
                "decision_action": str(result.decision.action.value if hasattr(result.decision.action, "value") else result.decision.action),
                "decision_reason": result.decision.reason,
                "decision_limit_price": _price_to_dollars(result.decision.limit_price),
                "decision_limit_price_raw": result.decision.limit_price,
                "decision_size": result.decision.size,
            }
            row.update(result.metrics)
            for key, value in result.diagnostics.items():
                row[f"decision_{key}"] = value
            rows.append(row)
        return pd.DataFrame(rows)

    def summary_frame(self) -> pd.DataFrame:
        raw = self.raw_frame()
        if raw.empty:
            return pd.DataFrame([{"orders": 0}])

        submitted = raw["submitted"].fillna(False).astype(bool) if "submitted" in raw else pd.Series(False, index=raw.index)
        filled = raw["filled"].fillna(False).astype(bool) if "filled" in raw else pd.Series(False, index=raw.index)
        canceled = raw["canceled"].fillna(False).astype(bool) if "canceled" in raw else pd.Series(False, index=raw.index)
        if "metric_status" in raw:
            metric_ok = raw["metric_status"].fillna("").eq("ok")
        else:
            metric_ok = pd.Series(True, index=raw.index)
        costs = raw.loc[metric_ok, "implementation_shortfall_bps"].astype(float)
        costs = costs[np.isfinite(costs)]
        weighted = _time_weighted_shortfall(
            raw.loc[metric_ok],
            min_horizon_ms=DEFAULT_TIME_WEIGHT_MIN_HORIZON_MS,
        )

        summary = {
            "orders": int(len(raw)),
            "submitted": int(submitted.sum()),
            "skipped": int((~submitted).sum()),
            "filled": int((submitted & filled).sum()),
            "unfilled": int((submitted & ~filled).sum()),
            "canceled": int((submitted & canceled).sum()),
            "metric_ok": int(metric_ok.sum()),
            "metric_failed": int((~metric_ok).sum()),
            "fill_rate": float((submitted & filled).sum() / submitted.sum()) if submitted.sum() else np.nan,
            "mean_is_bps": float(costs.mean()) if len(costs) else np.nan,
            "median_is_bps": float(costs.median()) if len(costs) else np.nan,
            "time_weighted_mean_is_bps": weighted["mean_bps"],
            "time_weight_min_horizon_ms": weighted["min_horizon_ms"],
            "time_weight_sum": weighted["weight_sum"],
            "time_weight_horizon_ms_mean": weighted["horizon_ms_mean"],
            "time_weight_horizon_ms_median": weighted["horizon_ms_median"],
            "time_weight_effective_horizon_ms_mean": weighted[
                "effective_horizon_ms_mean"
            ],
            "time_weight_effective_horizon_ms_median": weighted[
                "effective_horizon_ms_median"
            ],
            "total_is": float(
                raw.loc[metric_ok, "implementation_shortfall"].astype(float).sum(skipna=True)
            )
            if "implementation_shortfall" in raw
            else np.nan,
            "total_is_raw": float(
                raw.loc[metric_ok, "implementation_shortfall_raw"].astype(float).sum(skipna=True)
            )
            if "implementation_shortfall_raw" in raw
            else np.nan,
        }
        if "decision_model_latency_ns" in raw:
            latencies_ns = raw["decision_model_latency_ns"].astype(float)
            latencies_ns = latencies_ns[np.isfinite(latencies_ns)]
            summary["average_latency_ms"] = (
                float(latencies_ns.mean() / 1_000_000.0) if len(latencies_ns) else np.nan
            )
        if "missing_reason" in raw:
            for reason, group in raw.loc[~metric_ok].groupby("missing_reason", dropna=False):
                summary[f"failed_{reason}_count"] = int(len(group))
        for cost_type, group in raw.loc[metric_ok].groupby("cost_type", dropna=False):
            vals = group["implementation_shortfall_bps"].astype(float)
            vals = vals[np.isfinite(vals)]
            summary[f"{cost_type}_count"] = int(len(group))
            summary[f"{cost_type}_mean_bps"] = float(vals.mean()) if len(vals) else np.nan
            group_weighted = weighted["by_cost_type"].get(str(cost_type), {})
            summary[f"{cost_type}_time_weighted_mean_bps"] = group_weighted.get(
                "mean_bps",
                np.nan,
            )
            summary[f"{cost_type}_weight_sum"] = group_weighted.get("weight_sum", 0.0)
            summary[f"{cost_type}_horizon_ms_mean"] = group_weighted.get(
                "horizon_ms_mean",
                np.nan,
            )
            summary[f"{cost_type}_horizon_ms_median"] = group_weighted.get(
                "horizon_ms_median",
                np.nan,
            )
        return pd.DataFrame([summary])

    def write(self, raw_path: str | Path, summary_path: str | Path) -> None:
        raw_path = Path(raw_path)
        summary_path = Path(summary_path)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        self.raw_frame().to_parquet(raw_path, index=False)
        self.summary_frame().to_csv(summary_path, index=False)


def _price_to_dollars(value) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return float(out / CONFIG.data.price_unit)


def _time_weighted_shortfall(
    raw: pd.DataFrame,
    *,
    min_horizon_ms: float,
) -> dict[str, Any]:
    empty = {
        "mean_bps": np.nan,
        "min_horizon_ms": float(min_horizon_ms),
        "weight_sum": 0.0,
        "horizon_ms_mean": np.nan,
        "horizon_ms_median": np.nan,
        "effective_horizon_ms_mean": np.nan,
        "effective_horizon_ms_median": np.nan,
        "by_cost_type": {},
    }
    if raw.empty or "implementation_shortfall_bps" not in raw or "cost_type" not in raw:
        return empty

    valid = raw.copy()
    valid["implementation_shortfall_bps"] = pd.to_numeric(
        valid["implementation_shortfall_bps"],
        errors="coerce",
    )
    valid = valid[np.isfinite(valid["implementation_shortfall_bps"])]
    if valid.empty:
        return empty

    horizons_ms = _measurement_horizon_ms(valid)
    effective_horizons_ms = horizons_ms.clip(lower=float(min_horizon_ms))
    weights = 1.0 / effective_horizons_ms
    finite = np.isfinite(weights) & np.isfinite(valid["implementation_shortfall_bps"])
    valid = valid.loc[finite].copy()
    horizons_ms = horizons_ms.loc[finite]
    effective_horizons_ms = effective_horizons_ms.loc[finite]
    weights = _normalize_weights(weights.loc[finite])
    if valid.empty:
        return empty

    out = {
        "mean_bps": _weighted_mean(valid["implementation_shortfall_bps"], weights),
        "min_horizon_ms": float(min_horizon_ms),
        "weight_sum": float(weights.sum()) if len(weights) else 0.0,
        "horizon_ms_mean": _mean(horizons_ms),
        "horizon_ms_median": _median(horizons_ms),
        "effective_horizon_ms_mean": _mean(effective_horizons_ms),
        "effective_horizon_ms_median": _median(effective_horizons_ms),
        "by_cost_type": {},
    }

    for cost_type in valid["cost_type"].dropna().unique():
        mask = valid["cost_type"].eq(cost_type)
        group_weights = weights.loc[mask]
        out["by_cost_type"][str(cost_type)] = {
            "mean_bps": _weighted_mean(
                valid.loc[mask, "implementation_shortfall_bps"],
                group_weights,
            ),
            "weight_sum": float(group_weights.sum()) if len(group_weights) else 0.0,
            "horizon_ms_mean": _mean(horizons_ms.loc[mask]),
            "horizon_ms_median": _median(horizons_ms.loc[mask]),
        }
    return out


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


def _mean(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce")
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else np.nan


def _median(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce")
    values = values[np.isfinite(values)]
    return float(values.median()) if len(values) else np.nan


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
