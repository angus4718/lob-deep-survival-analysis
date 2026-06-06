"""Backtest orchestration."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

from .metrics import ImplementationShortfallMetric
from .reports import BacktestReport
from .strategies.base import BaseStrategy
from .types import BacktestResult, DecisionAction, MarketSnapshot, TradingDecision


class BacktestEngine:
    """Run a strategy over labeled market snapshots and collect metrics."""

    def __init__(
        self,
        strategy: BaseStrategy,
        *,
        metrics: Iterable[ImplementationShortfallMetric] | None = None,
        lifecycle_aware: bool = False,
        lifecycle_stride: int = 1,
        lifecycle_max_evaluations: int | None = None,
    ) -> None:
        if int(lifecycle_stride) < 1:
            raise ValueError("lifecycle_stride must be >= 1")
        if lifecycle_max_evaluations is not None and int(lifecycle_max_evaluations) < 1:
            raise ValueError("lifecycle_max_evaluations must be >= 1 or None")
        self.strategy = strategy
        self.metrics = list(metrics) if metrics is not None else [ImplementationShortfallMetric()]
        self.lifecycle_aware = bool(lifecycle_aware)
        self.lifecycle_stride = int(lifecycle_stride)
        self.lifecycle_max_evaluations = (
            int(lifecycle_max_evaluations)
            if lifecycle_max_evaluations is not None
            else None
        )

    def run(self, snapshots: Iterable[MarketSnapshot]) -> BacktestReport:
        calibration_rows = self._load_calibration_rows(snapshots)
        pending: list[tuple[MarketSnapshot, Any]] = []
        rows: list[dict[str, Any]] = []
        lifecycle_provider = getattr(snapshots, "lifecycle_snapshots", None)
        for snapshot in snapshots:
            decision = self.strategy.decide(snapshot)
            if self.lifecycle_aware:
                decision = self._run_lifecycle(
                    snapshot,
                    decision,
                    lifecycle_provider if callable(lifecycle_provider) else None,
                )
            else:
                decision = self._with_lifecycle_diagnostics(
                    decision,
                    lifecycle_evaluations=0,
                    final_update_idx=int(snapshot.update_idx),
                    final_end_idx=snapshot.end_idx,
                )
            pending.append((snapshot, decision))
            rows.append(snapshot.row.to_dict())

        replay_rows = pd.DataFrame(rows)
        metrics = self._prepare_metrics(
            calibration_rows if calibration_rows is not None else replay_rows
        )
        results: list[BacktestResult] = []
        for snapshot, decision in pending:
            metric_values = {}
            for metric in metrics:
                metric_values.update(metric.evaluate(snapshot.row, decision))
            results.append(
                BacktestResult(
                    row_index=snapshot.row_index,
                    order_id=snapshot.order_id,
                    decision=decision,
                    metrics=metric_values,
                    diagnostics=decision.diagnostics,
                )
            )
        return BacktestReport(results)

    def _run_lifecycle(
        self,
        snapshot: MarketSnapshot,
        initial_decision: TradingDecision,
        lifecycle_provider,
    ) -> TradingDecision:
        if lifecycle_provider is None or not initial_decision.should_submit:
            return self._with_lifecycle_diagnostics(
                initial_decision,
                lifecycle_evaluations=0,
                final_update_idx=int(snapshot.update_idx),
                final_end_idx=snapshot.end_idx,
            )

        lifecycle_evaluations = 0
        last_hold: TradingDecision | None = None
        for lifecycle_snapshot in lifecycle_provider(
            snapshot,
            stride=self.lifecycle_stride,
            max_evaluations=self.lifecycle_max_evaluations,
        ):
            lifecycle_evaluations += 1
            decision = self.strategy.decide(lifecycle_snapshot)
            if decision.should_cancel:
                return self._with_lifecycle_diagnostics(
                    decision,
                    lifecycle_evaluations=lifecycle_evaluations,
                    final_update_idx=int(lifecycle_snapshot.update_idx),
                    final_end_idx=lifecycle_snapshot.end_idx,
                    initial_action=DecisionAction.SUBMIT.value,
                )
            last_hold = decision

        final_update_idx = (
            int(last_hold.diagnostics.get("update_idx"))
            if last_hold is not None and last_hold.diagnostics.get("update_idx") is not None
            else int(snapshot.update_idx)
        )
        final_end_idx = (
            int(last_hold.diagnostics.get("end_idx"))
            if last_hold is not None and last_hold.diagnostics.get("end_idx") is not None
            else snapshot.end_idx
        )
        return self._with_lifecycle_diagnostics(
            initial_decision,
            lifecycle_evaluations=lifecycle_evaluations,
            final_update_idx=final_update_idx,
            final_end_idx=final_end_idx,
            initial_action=DecisionAction.SUBMIT.value,
        )

    def _with_lifecycle_diagnostics(
        self,
        decision: TradingDecision,
        *,
        lifecycle_evaluations: int,
        final_update_idx: int,
        final_end_idx: int | None,
        initial_action: str | None = None,
    ) -> TradingDecision:
        diagnostics = dict(decision.diagnostics)
        diagnostics["lifecycle_evaluations"] = int(lifecycle_evaluations)
        diagnostics["final_update_idx"] = int(final_update_idx)
        diagnostics["final_end_idx"] = int(final_end_idx) if final_end_idx is not None else None
        if decision.should_cancel or DecisionAction(decision.action) == DecisionAction.SKIP:
            diagnostics["reference_end_idx"] = (
                int(final_end_idx) if final_end_idx is not None else None
            )
        if initial_action is not None:
            diagnostics["initial_action"] = initial_action
        return TradingDecision(
            action=decision.action,
            limit_price=decision.limit_price,
            size=decision.size,
            reason=decision.reason,
            diagnostics=diagnostics,
        )

    def _load_calibration_rows(self, snapshots: Iterable[MarketSnapshot]) -> pd.DataFrame | None:
        loader = getattr(snapshots, "calibration_frame", None)
        if not callable(loader):
            return None
        frame = loader()
        if frame is None or frame.empty:
            return None
        return frame

    def _prepare_metrics(self, rows: pd.DataFrame) -> list[ImplementationShortfallMetric]:
        prepared: list[ImplementationShortfallMetric] = []
        for metric in self.metrics:
            if (
                isinstance(metric, ImplementationShortfallMetric)
                and metric.calibrate_toxic_window
                and metric.window_selection is None
                and not rows.empty
            ):
                prepared.append(
                    ImplementationShortfallMetric.from_labeled_orders(
                        rows,
                        unfilled_lob_sequence_col=metric.unfilled_lob_sequence_col,
                        price_unit=metric.price_unit,
                    )
                )
            else:
                prepared.append(metric)
        return prepared
