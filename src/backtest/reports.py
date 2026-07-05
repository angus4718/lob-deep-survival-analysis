"""Raw and summarized backtest reports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import CONFIG

from .types import BacktestResult


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

        summary = {
            "orders": int(len(raw)),
            "submitted_orders": int(submitted.sum()),
            "skipped_orders": int((~submitted).sum()),
            "filled_orders": int((submitted & filled).sum()),
            "unfilled_orders": int((submitted & ~filled).sum()),
            "canceled_orders": int((submitted & canceled).sum()),
            "metric_ok_orders": int(metric_ok.sum()),
            "metric_failed_orders": int((~metric_ok).sum()),
            "fill_rate": float((submitted & filled).sum() / submitted.sum()) if submitted.sum() else np.nan,
            "mean_implementation_shortfall_bps": float(costs.mean()) if len(costs) else np.nan,
            "median_implementation_shortfall_bps": float(costs.median()) if len(costs) else np.nan,
            "total_implementation_shortfall": float(
                raw.loc[metric_ok, "implementation_shortfall"].astype(float).sum(skipna=True)
            )
            if "implementation_shortfall" in raw
            else np.nan,
            "total_implementation_shortfall_raw": float(
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
