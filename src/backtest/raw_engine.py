"""Raw Databento replay backtest engine with inference latency modeling."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.raw_splits import (
    RawChunk,
    build_raw_chunks,
    filter_chunks_for_order_range,
    load_or_build_split_cache,
)

from .data import BacktestDataset, BacktestFeatureBuilder, safe_stack_sequence
from .execution import RawBacktestOrder, coerce_price
from .latency import LatencyProvider, StaticLatencyProvider
from .book_utils import book_to_snapshot, queue_ahead_at_price
from .metrics import ImplementationShortfallMetric
from .progress import log, maybe_tqdm, set_progress_postfix
from .reports import BacktestReport
from .strategies.base import BaseStrategy
from .types import BacktestResult, DecisionAction, MarketSnapshot, TradingDecision
from .live_features import LiveFeatureBuilder


_NY_TZ = "America/New_York"


@dataclass
class _PendingDecision:
    order: RawBacktestOrder
    decision: TradingDecision
    effective_time: int
    latency_ns: int
    kind: str


@dataclass(frozen=True)
class _RebuiltFeatureSnapshot:
    ts_event: int
    lob_step: list[float]
    tox_base_step: list[float]


class RawDatabentoBacktestEngine:
    """Run a strategy over labeled candidate orders while replaying raw MBO data."""

    def __init__(
        self,
        strategy: BaseStrategy,
        *,
        raw_path: str | Path | None = None,
        records: Iterable[Any] | None = None,
        market: Any | None = None,
        feature_builder: BacktestFeatureBuilder | None = None,
        metrics: Iterable[ImplementationShortfallMetric] | None = None,
        latency_provider: LatencyProvider | None = None,
        snapshot_bin_messages: int = 15,
        lifecycle_aware: bool = False,
        lifecycle_stride: int = 50,
        lifecycle_max_evaluations: int | None = None,
        censor_at_labeled_duration: bool = False,
        raw_order_max_lifetime_s: float | None = None,
        strict_time_coverage: bool = True,
        raw_replay_start_ns: int | None = None,
        raw_replay_end_ns: int | None = None,
        verbose: bool = False,
        progress: bool | None = None,
        progress_interval: int = 100_000,
    ) -> None:
        if raw_path is None and records is None:
            raise ValueError("Either raw_path or records must be provided.")
        if int(snapshot_bin_messages) < 1:
            raise ValueError("snapshot_bin_messages must be >= 1.")
        if int(lifecycle_stride) < 1:
            raise ValueError("lifecycle_stride must be >= 1.")
        if lifecycle_max_evaluations is not None and int(lifecycle_max_evaluations) < 1:
            raise ValueError("lifecycle_max_evaluations must be >= 1 or None.")
        if raw_order_max_lifetime_s is not None and float(raw_order_max_lifetime_s) <= 0:
            raise ValueError("raw_order_max_lifetime_s must be > 0 or None.")
        if int(progress_interval) < 1:
            raise ValueError("progress_interval must be >= 1.")

        self.strategy = strategy
        self.raw_path = Path(raw_path) if raw_path is not None else None
        self.records = records
        self.market = market if market is not None else self._default_market()
        self.feature_builder = feature_builder
        self.metrics = list(metrics) if metrics is not None else [ImplementationShortfallMetric()]
        self.latency_provider = latency_provider or StaticLatencyProvider(0)
        self.snapshot_bin_messages = int(snapshot_bin_messages)
        self.lifecycle_aware = bool(lifecycle_aware)
        self.lifecycle_stride = int(lifecycle_stride)
        self.lifecycle_max_evaluations = (
            int(lifecycle_max_evaluations)
            if lifecycle_max_evaluations is not None
            else None
        )
        self.censor_at_labeled_duration = bool(censor_at_labeled_duration)
        self.raw_order_max_lifetime_s = (
            float(raw_order_max_lifetime_s)
            if raw_order_max_lifetime_s is not None
            else None
        )
        self.strict_time_coverage = bool(strict_time_coverage)
        self.raw_replay_start_ns = (
            int(raw_replay_start_ns) if raw_replay_start_ns is not None else None
        )
        self.raw_replay_end_ns = (
            int(raw_replay_end_ns) if raw_replay_end_ns is not None else None
        )
        self.verbose = bool(verbose)
        self.progress = self.verbose if progress is None else bool(progress)
        self.progress_interval = int(progress_interval)
        self._stats: dict[str, int] = {}
        self._rebuilt_snapshots: deque[_RebuiltFeatureSnapshot] = deque()

    def run(self, orders: BacktestDataset | pd.DataFrame) -> BacktestReport:
        orders_df, calibration_frame, feature_builder = self._load_orders(orders)
        if orders_df.empty:
            log(self.verbose, "No labeled orders supplied; returning an empty report.")
            return BacktestReport([])

        self._stats = self._empty_stats()
        log(
            self.verbose,
            "Starting raw replay backtest "
            f"(lifecycle_aware={self.lifecycle_aware}, "
            f"latency_provider={type(self.latency_provider).__name__}, "
            f"censor_at_labeled_duration={self.censor_at_labeled_duration}, "
            f"raw_order_max_lifetime_s={self.raw_order_max_lifetime_s}).",
        )
        log(self.verbose, f"Loaded {len(orders_df):,} labeled orders.")
        live_features = self._make_live_feature_builder(feature_builder)
        self._rebuilt_snapshots = deque(maxlen=max(1, int(feature_builder.lookback_steps)))
        log(self.verbose, "Preparing metrics.")
        metrics = self._prepare_metrics(
            calibration_frame if calibration_frame is not None else orders_df
        )

        scheduled = self._scheduled_orders(orders_df)
        if scheduled:
            log(
                self.verbose,
                "Scheduled "
                f"{len(scheduled):,} orders from {_format_ns(int(scheduled[0][0]))} "
                f"to {_format_ns(int(scheduled[-1][0]))}.",
            )
        else:
            log(self.verbose, "No schedulable orders had usable entry_time/price.")
            return BacktestReport([])
        next_order_idx = 0
        pending: list[_PendingDecision] = []
        active: list[RawBacktestOrder] = []
        post_trade: list[RawBacktestOrder] = []
        completed: list[RawBacktestOrder] = []

        message_count_since_snapshot = 0
        last_book = None
        last_ts = 0
        first_raw_ts: int | None = None

        raw_iter = maybe_tqdm(
            self._iter_records(),
            enabled=self.progress,
            desc="Raw MBO replay",
            unit="msg",
            mininterval=1.0,
        )
        for mbo in raw_iter:
            self._stats["raw_messages"] += 1
            if not hasattr(mbo, "ts_event"):
                continue
            ts_event = int(mbo.ts_event)
            last_ts = ts_event
            if first_raw_ts is None:
                first_raw_ts = ts_event
                log(self.verbose, f"First raw message timestamp: {_format_ns(first_raw_ts)}.")
                self._validate_raw_start_covers_orders(
                    first_raw_ts=first_raw_ts,
                    first_order_ts=int(scheduled[0][0]),
                )

            if last_book is not None and pending:
                self._apply_due_initial_decisions(
                    active,
                    post_trade,
                    completed,
                    pending,
                    last_book,
                    ts_event,
                )
                self._apply_due_cancels(active, completed, pending, last_book, ts_event)
            if last_book is not None and active:
                self._censor_due_orders(active, completed, last_book, ts_event)

            try:
                self.market.apply(mbo)
            except (KeyError, AssertionError, ValueError):
                self._stats["market_apply_errors"] += 1
                continue
            self._stats["applied_messages"] += 1

            book = self._book_for_record(mbo)
            if book is None:
                self._stats["missing_books"] += 1
                continue
            last_book = book

            if self.lifecycle_aware and active:
                message_count_since_snapshot += 1
                if message_count_since_snapshot >= self.snapshot_bin_messages:
                    snapshot = book_to_snapshot(book, ts_event)
                    if snapshot is not None:
                        rebuilt = self._record_rebuilt_snapshot(
                            snapshot,
                            live_features,
                            ts_event,
                        )
                        if rebuilt is not None:
                            self._append_rebuilt_snapshot_to_active_orders(
                                active,
                                rebuilt,
                                live_features,
                            )
                        self._maybe_schedule_lifecycle_decisions(
                            active,
                            ts_event,
                            live_features,
                            pending,
                        )
                    message_count_since_snapshot = 0

            if active:
                self._update_active_orders(active, post_trade, mbo, book)
                self._apply_due_cancels(active, completed, pending, book, ts_event)
                self._censor_due_orders(active, completed, book, ts_event)
            if post_trade:
                self._update_post_trade_orders(post_trade, completed, book, ts_event)

            if self.lifecycle_aware and not active:
                message_count_since_snapshot = 0
                self._rebuilt_snapshots.clear()

            next_order_idx = self._observe_due_orders(
                scheduled,
                next_order_idx,
                ts_event,
                book,
                feature_builder,
                live_features,
                pending,
                completed,
            )
            if pending: 
                self._apply_due_initial_decisions(
                    active,
                    post_trade,
                    completed,
                    pending,
                    book,
                    ts_event,
                )

            if next_order_idx >= len(scheduled) and not active and not post_trade and not pending:
                # Keep this conservative: only break once all scheduled orders are complete.
                break
            if self.progress and self._stats["raw_messages"] % self.progress_interval == 0:
                set_progress_postfix(
                    raw_iter,
                    time=pd.to_datetime(ts_event, unit="ns", utc=True).tz_convert(
                        "America/New_York"
                    ),
                    observed=next_order_idx,
                    active=len(active),
                    pending=len(pending),
                    done=len(completed),
                )

        if last_book is not None:
            self._finalize_remaining(pending, active, post_trade, completed, last_book, last_ts)
        self._validate_raw_end_covers_orders(
            scheduled=scheduled,
            next_order_idx=next_order_idx,
            last_raw_ts=last_ts if first_raw_ts is not None else None,
        )

        results = self._build_results(completed, metrics)
        log(
            self.verbose,
            "Finished raw replay: "
            f"raw_messages={self._stats.get('raw_messages', 0):,}, "
            f"applied={self._stats.get('applied_messages', 0):,}, "
            f"observed_orders={self._stats.get('observed_orders', 0):,}, "
            f"label_features={self._stats.get('initial_features_from_label', 0):,}, "
            f"rebuilt_tail_features={self._stats.get('initial_features_with_rebuilt_tail', 0):,}, "
            f"live_fallback_features={self._stats.get('initial_features_from_live_fallback', 0):,}, "
            f"lifecycle_snapshots={self._stats.get('lifecycle_rebuilt_snapshots', 0):,}, "
            f"submitted={self._stats.get('submitted_orders', 0):,}, "
            f"skipped={self._stats.get('skipped_orders', 0):,}, "
            f"filled={self._stats.get('filled_orders', 0):,}, "
            f"canceled={self._stats.get('canceled_orders', 0):,}, "
            f"censored={self._stats.get('censored_orders', 0):,}, "
            f"results={len(results):,}.",
        )
        return BacktestReport(results)

    def run_parallel(
        self,
        orders: BacktestDataset | pd.DataFrame,
        *,
        n_workers: int | None = None,
        split_cache_path: str | Path | None = None,
        empty_points: list[int] | None = None,
        messages_between_splits: list[int] | None = None,
        total_messages: int | None = None,
        empty_scan_verbose: bool = True,
        mp_start_method: str | None = None,
    ) -> BacktestReport:
        """Run raw replay in multiple safe empty-book chunks."""
        if self.raw_path is None:
            raise ValueError("run_parallel requires raw_path; records cannot be chunked.")
        if self.records is not None:
            raise ValueError("run_parallel does not support in-memory records.")
        orders_df, calibration_frame, feature_builder = self._load_orders(orders)
        if orders_df.empty:
            log(self.verbose, "No labeled orders supplied; returning an empty report.")
            return BacktestReport([])

        scheduled = self._scheduled_orders(orders_df)
        if not scheduled:
            log(self.verbose, "No schedulable orders had usable entry_time/price.")
            return BacktestReport([])
        order_start_ns = int(scheduled[0][0])
        order_end_ns = int(scheduled[-1][0]) + 1

        if empty_points is None:
            if split_cache_path is None:
                split_cache_path = self.raw_path.with_suffix(
                    self.raw_path.suffix + ".split_points.json"
                )
            split_cache = load_or_build_split_cache(
                split_cache_path,
                self.raw_path,
                verbose=self.verbose or empty_scan_verbose,
            )
            empty_points = [int(item) for item in split_cache.get("split_points", [])]
            messages_between_splits = split_cache.get("messages_between_splits")
            total_messages = split_cache.get("total_messages")

        resolved_workers = self._resolve_parallel_worker_count(n_workers, empty_points or [])
        log(
            self.verbose,
            f"Resolved raw parallel workers: {resolved_workers} "
            f"(requested={n_workers}, split_points={len(empty_points or [])}).",
        )
        if resolved_workers <= 1:
            return self.run(orders)

        chunks = build_raw_chunks(
            empty_points=[int(item) for item in (empty_points or [])],
            n_workers=resolved_workers,
            messages_between_splits=messages_between_splits,
            total_messages=total_messages,
        )
        if not chunks:
            log(self.verbose, "No split chunks available; falling back to single-process raw run.")
            return self.run(orders)

        chunks = filter_chunks_for_order_range(
            chunks,
            order_start_ns=order_start_ns,
            order_end_ns=order_end_ns,
        )
        if not chunks:
            log(self.verbose, "No raw chunks overlap the labeled order range.")
            return BacktestReport([])

        prepared_metrics = self._prepare_metrics(
            calibration_frame if calibration_frame is not None else orders_df
        )
        worker_args = []
        for chunk in chunks:
            chunk_orders = self._orders_for_chunk(orders_df, chunk)
            if chunk_orders.empty:
                continue
            replay_end_ns = self._chunk_replay_end_ns(chunk, chunk_orders)
            worker_args.append(
                {
                    "chunk": chunk,
                    "strategy": self.strategy,
                    "raw_path": str(self.raw_path),
                    "orders_df": chunk_orders,
                    "feature_builder": feature_builder,
                    "metrics": prepared_metrics,
                    "latency_provider": self.latency_provider,
                    "snapshot_bin_messages": self.snapshot_bin_messages,
                    "lifecycle_aware": self.lifecycle_aware,
                    "lifecycle_stride": self.lifecycle_stride,
                    "lifecycle_max_evaluations": self.lifecycle_max_evaluations,
                    "censor_at_labeled_duration": self.censor_at_labeled_duration,
                    "raw_order_max_lifetime_s": self.raw_order_max_lifetime_s,
                    "strict_time_coverage": self.strict_time_coverage,
                    "raw_replay_start_ns": chunk.start_ns,
                    "raw_replay_end_ns": replay_end_ns,
                    "verbose": self.verbose,
                    "progress": self.progress,
                    "progress_interval": self.progress_interval,
                }
            )

        if not worker_args:
            log(self.verbose, "No raw chunks had assigned labeled orders.")
            return BacktestReport([])

        log(
            self.verbose,
            "Starting parallel raw replay "
            f"with {len(worker_args)} active chunk(s) out of {len(chunks)} overlapping chunk(s).",
        )
        mp_context = _multiprocessing_context(mp_start_method)
        results: list[BacktestResult] = []
        with ProcessPoolExecutor(
            max_workers=min(resolved_workers, len(worker_args)),
            mp_context=mp_context,
        ) as executor:
            future_to_chunk = {
                executor.submit(_raw_backtest_chunk_worker, args): args["chunk"]
                for args in worker_args
            }
            for future in as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                chunk_results = future.result()
                log(
                    self.verbose,
                    f"Raw chunk {chunk.index} finished with {len(chunk_results):,} result(s).",
                )
                results.extend(chunk_results)

        results.sort(key=lambda result: int(result.row_index))
        return BacktestReport(results)

    def _empty_stats(self) -> dict[str, int]:
        return {
            "raw_messages": 0,
            "applied_messages": 0,
            "market_apply_errors": 0,
            "missing_books": 0,
            "book_snapshots": 0,
            "observed_orders": 0,
            "missing_live_features": 0,
            "submitted_orders": 0,
            "skipped_orders": 0,
            "filled_orders": 0,
            "canceled_orders": 0,
            "censored_orders": 0,
            "lifecycle_cancel_requests": 0,
            "initial_features_from_label": 0,
            "initial_features_with_rebuilt_tail": 0,
            "initial_features_from_live_fallback": 0,
            "lifecycle_rebuilt_snapshots": 0,
        }

    def _load_orders(
        self,
        orders: BacktestDataset | pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None, BacktestFeatureBuilder]:
        if isinstance(orders, BacktestDataset):
            df = orders.load_frame()
            calibration = orders.load_calibration_frame()
            feature_builder = orders.feature_builder
        else:
            df = orders.copy()
            calibration = None
            if self.feature_builder is None:
                raise ValueError(
                    "feature_builder must be provided when running raw backtest "
                    "from a DataFrame."
                )
            feature_builder = self.feature_builder
        if self.feature_builder is not None:
            feature_builder = self.feature_builder
        return df, calibration, feature_builder

    def _make_live_feature_builder(
        self,
        builder: BacktestFeatureBuilder,
    ):
        from .live_features import LiveFeatureBuilder

        return LiveFeatureBuilder(
            lookback_steps=builder.lookback_steps,
            lob_dim=builder.lob_dim,
            tox_dim=builder.tox_dim,
            feat_mean=builder.feat_mean,
            feat_std=builder.feat_std,
            max_buffer_len=builder.lookback_steps,
        )

    def _scheduled_orders(self, df: pd.DataFrame) -> list[tuple[int, int, pd.Series]]:
        scheduled: list[tuple[int, int, pd.Series]] = []
        for row_index, row in df.iterrows():
            entry_time = _safe_int(row.get("entry_time"))
            price = coerce_price(row.get("price"))
            if entry_time is None or price is None:
                continue
            scheduled.append((entry_time, int(row_index), row))
        scheduled.sort(key=lambda item: item[0])
        return scheduled

    def _orders_for_chunk(self, df: pd.DataFrame, chunk: RawChunk) -> pd.DataFrame:
        entry_time = pd.to_numeric(df["entry_time"], errors="coerce")
        mask = pd.Series(True, index=df.index)
        if chunk.start_ns is not None:
            mask &= entry_time >= int(chunk.start_ns)
        if chunk.end_ns is not None:
            mask &= entry_time < int(chunk.end_ns)
        return df.loc[mask].copy()

    def _resolve_parallel_worker_count(
        self,
        n_workers: int | None,
        empty_points: list[int],
    ) -> int:
        if n_workers is not None:
            return max(1, int(n_workers))
        cpu_workers = max(1, int((os.cpu_count() or 1) * (2.0 / 3.0)))
        split_workers = max(1, len(empty_points))
        return max(1, min(split_workers, cpu_workers))

    def _chunk_replay_end_ns(self, chunk: RawChunk, orders_df: pd.DataFrame) -> int | None:
        replay_end = int(chunk.end_ns) if chunk.end_ns is not None else None
        post_trade_ns = self._max_post_trade_window_ns()
        for _, row in orders_df.iterrows():
            entry_time = _safe_int(row.get("entry_time"))
            if entry_time is None:
                continue
            order_end = self._regular_session_end_ns(entry_time)
            lifetime_s = self._raw_order_lifetime_s(row)
            if lifetime_s is not None:
                lifetime_deadline = int(entry_time) + max(0, int(float(lifetime_s) * 1e9))
                order_end = min(int(order_end), int(lifetime_deadline))
            needed_end = int(order_end) + int(post_trade_ns)
            replay_end = needed_end if replay_end is None else max(int(replay_end), needed_end)
        return replay_end

    def _max_post_trade_window_ns(self) -> int:
        max_ms = 0
        for metric in self.metrics:
            if isinstance(metric, ImplementationShortfallMetric):
                max_ms = max(max_ms, int(metric.selected_window_ms))
        try:
            configured = max(int(v) for v in CONFIG.labeling.tox_post_trade_move_windows_ms)
            max_ms = max(max_ms, configured)
        except ValueError:
            pass
        return int(max_ms * 1_000_000)

    def _iter_records(self) -> Iterable[Any]:
        if self.records is not None:
            return iter(self.records)
        if self.raw_path is None:
            return iter(())
        import databento as db

        return self._iter_bounded_records(db.DBNStore.from_file(str(self.raw_path)))

    def _iter_bounded_records(self, records: Iterable[Any]) -> Iterable[Any]:
        for record in records:
            ts_event = getattr(record, "ts_event", None)
            if ts_event is None:
                if self.raw_replay_start_ns is None:
                    yield record
                continue
            ts_event = int(ts_event)
            if self.raw_replay_start_ns is not None and ts_event < self.raw_replay_start_ns:
                continue
            if self.raw_replay_end_ns is not None and ts_event >= self.raw_replay_end_ns:
                break
            yield record

    def _default_market(self) -> Any:
        from src.lob_implementation import Market

        return Market()

    def _book_for_record(self, mbo: Any) -> Any | None:
        try:
            return self.market.get_book(mbo.instrument_id, mbo.publisher_id)
        except Exception:
            return None

    def _observe_due_orders(
        self,
        scheduled: list[tuple[int, int, pd.Series]],
        next_order_idx: int,
        ts_event: int,
        book: Any,
        feature_builder: BacktestFeatureBuilder,
        live_features: LiveFeatureBuilder,
        pending: list[_PendingDecision],
        completed: list[RawBacktestOrder],
    ) -> int:
        while next_order_idx < len(scheduled) and scheduled[next_order_idx][0] <= ts_event:
            _, row_index, row = scheduled[next_order_idx]
            next_order_idx += 1
            self._stats["observed_orders"] += 1
            order = self._make_order(row_index, row, ts_event)
            order.set_entry_bbo(book)

            try:
                features, end_idx = self._build_initial_features_from_label(
                    order,
                    row,
                    feature_builder,
                    book,
                    live_features,
                    ts_event,
                )
                self._stats["initial_features_from_label"] += 1
            except ValueError:
                try:
                    features, end_idx = self._build_initial_features_from_live(
                        order,
                        book,
                        ts_event,
                        live_features,
                    )
                    self._stats["initial_features_from_live_fallback"] += 1
                    order.initial_feature_source = "live_fallback"
                except ValueError:
                    self._stats["missing_live_features"] += 1
                    self._stats["skipped_orders"] += 1
                    decision = TradingDecision(
                        action=DecisionAction.SKIP,
                        limit_price=order.limit_price,
                        reason="missing_initial_features",
                        diagnostics={
                            "raw_mode": True,
                            "observation_time": int(ts_event),
                            "model_latency_ns": 0,
                        },
                    )
                    order.decision = decision
                    order.skip(book, ts_event)
                    completed.append(order)
                    continue

            snapshot = MarketSnapshot(
                row_index=order.row_index,
                row=self._snapshot_row(order),
                features=features,
                update_idx=0,
                end_idx=end_idx,
                is_initial=True,
                position_open=False,
            )
            latency_result = self.latency_provider.run(lambda: self.strategy.decide(snapshot))
            effective_time = int(ts_event) + int(latency_result.latency_ns)
            decision = self._with_raw_diagnostics(
                latency_result.value,
                order=order,
                model_latency_ns=int(latency_result.latency_ns),
                observation_time=int(ts_event),
                decision_effective_time=effective_time,
                lifecycle_evaluations=0,
                update_idx=0,
                end_idx=end_idx,
            )
            order.decision = decision
            order.model_latency_ns = int(latency_result.latency_ns)
            pending.append(
                _PendingDecision(
                    order=order,
                    decision=decision,
                    effective_time=effective_time,
                    latency_ns=int(latency_result.latency_ns),
                    kind="initial",
                )
            )
        return next_order_idx

    def _build_initial_features_from_label(
        self,
        order: RawBacktestOrder,
        row: pd.Series,
        feature_builder: BacktestFeatureBuilder,
        book: Any,
        live_features: LiveFeatureBuilder,
        ts_event: int,
    ) -> tuple[np.ndarray, int]:
        lob_rep = row.get(feature_builder.lob_sequence_col)
        if lob_rep is None and feature_builder.fallback_lob_sequence_col:
            lob_rep = row.get(feature_builder.fallback_lob_sequence_col)
        tox_rep = row.get(feature_builder.tox_sequence_col)

        lob_seq = safe_stack_sequence(lob_rep, feature_builder.lob_dim)
        tox_seq = safe_stack_sequence(tox_rep, feature_builder.tox_dim)
        seq_len = min(lob_seq.shape[0], tox_seq.shape[0])
        seq_len = feature_builder._clip_sequence_len(
            seq_len,
            row.get(feature_builder.sequence_length_col),
        )
        if seq_len <= 0:
            raise ValueError("labeled row has no sequence data")

        end_idx = feature_builder.initial_end_idx(seq_len)
        order.lob_sequence_raw_top5 = lob_seq[: end_idx + 1].tolist()
        order.toxicity_sequence = tox_seq[: end_idx + 1].tolist()
        order.initial_feature_source = "label"
        if self._merge_rebuilt_tail_for_initial_order(
            order,
            book,
            live_features,
            ts_event,
            max_sequence_len=end_idx + 1,
        ):
            order.initial_feature_source = "label_rebuilt_tail"
            self._stats["initial_features_with_rebuilt_tail"] += 1
            features = live_features.build_model_features(
                lob_sequence=order.lob_sequence_raw_top5,
                toxicity_sequence=order.toxicity_sequence,
                side=order.side,
                end_idx=len(order.lob_sequence_raw_top5) - 1,
            )
            return features, len(order.lob_sequence_raw_top5) - 1
        return feature_builder.build(row, end_idx=end_idx), int(end_idx)

    def _merge_rebuilt_tail_for_initial_order(
        self,
        order: RawBacktestOrder,
        book: Any,
        live_features: LiveFeatureBuilder,
        ts_event: int,
        *,
        max_sequence_len: int,
    ) -> bool:
        if not self.lifecycle_aware or not self._rebuilt_snapshots:
            return False
        tail = [
            snapshot
            for snapshot in self._rebuilt_snapshots
            if int(snapshot.ts_event) <= int(ts_event)
        ]
        if not tail:
            return False

        observed_queue, _ = queue_ahead_at_price(
            book,
            side=order.side,
            price=order.limit_price,
        )
        max_len = max(1, int(max_sequence_len))
        tail = tail[-max_len:]
        prefix_len = max(0, min(len(order.lob_sequence_raw_top5), max_len) - len(tail))
        lob_prefix = order.lob_sequence_raw_top5[:prefix_len]
        tox_prefix = order.toxicity_sequence[:prefix_len]
        lob_tail = [list(item.lob_step) for item in tail]
        tox_tail = [
            live_features.augment_toxicity_step(item.tox_base_step, observed_queue)
            for item in tail
        ]
        order.lob_sequence_raw_top5 = lob_prefix + lob_tail
        order.toxicity_sequence = tox_prefix + tox_tail
        return bool(lob_tail)

    def _build_initial_features_from_live(
        self,
        order: RawBacktestOrder,
        book: Any,
        ts_event: int,
        live_features: LiveFeatureBuilder,
    ) -> tuple[np.ndarray, int]:
        if not live_features.snapshot_buffer:
            live_features.append_book(book, ts_event)

        observed_queue, _ = queue_ahead_at_price(
            book,
            side=order.side,
            price=order.limit_price,
        )
        lob_seq, tox_seq = live_features.build_initial_sequences(
            current_vahead=observed_queue
        )
        order.lob_sequence_raw_top5 = [list(item) for item in lob_seq]
        order.toxicity_sequence = [list(item) for item in tox_seq]
        features = live_features.build_model_features(
            lob_sequence=order.lob_sequence_raw_top5,
            toxicity_sequence=order.toxicity_sequence,
            side=order.side,
        )
        return features, max(0, len(order.lob_sequence_raw_top5) - 1)

    def _make_order(self, row_index: int, row: pd.Series, observation_time: int) -> RawBacktestOrder:
        deadline = self._regular_session_end_ns(observation_time)
        lifetime_s = self._raw_order_lifetime_s(row)
        if lifetime_s is not None:
            lifetime_deadline = int(observation_time) + max(0, int(float(lifetime_s) * 1e9))
            deadline = min(int(deadline), int(lifetime_deadline))
        price = coerce_price(row.get("price"))
        if price is None:
            price = 0
        return RawBacktestOrder(
            row_index=int(row_index),
            row=row,
            observation_time=int(observation_time),
            deadline_time=int(deadline),
            side=str(row.get("side", "")).upper(),
            limit_price=int(price),
        )

    def _raw_order_lifetime_s(self, row: pd.Series) -> float | None:
        candidates: list[float] = []
        if self.censor_at_labeled_duration:
            duration_s = _safe_float(row.get("duration_s"))
            if duration_s is not None:
                candidates.append(max(0.0, float(duration_s)))
        if self.raw_order_max_lifetime_s is not None:
            candidates.append(float(self.raw_order_max_lifetime_s))
        return min(candidates) if candidates else None

    def _regular_session_end_ns(self, ts_event: int) -> int:
        ts_dt = pd.to_datetime(int(ts_event), unit="ns", utc=True).tz_convert(_NY_TZ)
        local_midnight = pd.Timestamp(ts_dt.date()).tz_localize(_NY_TZ)
        is_early_close = ts_dt.date() in (
            pd.Timestamp("2025-11-28").date(),
            pd.Timestamp("2025-12-24").date(),
        )
        if is_early_close:
            day_end_dt = local_midnight + pd.Timedelta(hours=13)
        else:
            day_end_dt = local_midnight + pd.Timedelta(hours=16)
        day_end_ns = int(day_end_dt.tz_convert("UTC").value)
        if day_end_ns <= int(ts_event):
            day_end_dt = day_end_dt + pd.Timedelta(days=1)
            day_end_ns = int(day_end_dt.tz_convert("UTC").value)
        return day_end_ns

    def _record_rebuilt_snapshot(
        self,
        snapshot,
        live_features: LiveFeatureBuilder,
        ts_event: int,
    ) -> _RebuiltFeatureSnapshot | None:
        lob_step, tox_base_step = live_features.build_shared_step(snapshot)
        if lob_step is None or tox_base_step is None:
            return None
        rebuilt = _RebuiltFeatureSnapshot(
            ts_event=int(ts_event),
            lob_step=list(lob_step),
            tox_base_step=list(tox_base_step),
        )
        self._rebuilt_snapshots.append(rebuilt)
        self._stats["book_snapshots"] += 1
        self._stats["lifecycle_rebuilt_snapshots"] += 1
        return rebuilt

    def _append_rebuilt_snapshot_to_active_orders(
        self,
        active: list[RawBacktestOrder],
        rebuilt: _RebuiltFeatureSnapshot,
        live_features: LiveFeatureBuilder,
    ) -> None:
        for order in active:
            if not order.is_active:
                continue
            tox_step = live_features.augment_toxicity_step(
                rebuilt.tox_base_step,
                order.current_vahead,
            )
            order.append_feature_step(rebuilt.lob_step, tox_step)

    def _maybe_schedule_lifecycle_decisions(
        self,
        active: list[RawBacktestOrder],
        ts_event: int,
        live_features: LiveFeatureBuilder,
        pending: list[_PendingDecision],
    ) -> None:
        if not self.lifecycle_aware:
            return
        for order in active:
            if order.status != "ACTIVE":
                continue
            if (
                self.lifecycle_max_evaluations is not None
                and order.lifecycle_evaluations >= self.lifecycle_max_evaluations
            ):
                continue
            sequence_count = len(order.lob_sequence_raw_top5)
            if sequence_count - order.last_lifecycle_snapshot_count < self.lifecycle_stride:
                continue
            order.last_lifecycle_snapshot_count = sequence_count
            order.lifecycle_evaluations += 1

            try:
                features = live_features.build_model_features(
                    lob_sequence=order.lob_sequence_raw_top5,
                    toxicity_sequence=order.toxicity_sequence,
                    side=order.side,
                )
            except ValueError:
                continue

            snapshot = MarketSnapshot(
                row_index=order.row_index,
                row=self._snapshot_row(order),
                features=features,
                update_idx=max(0, sequence_count - 1),
                end_idx=sequence_count - 1,
                is_initial=False,
                position_open=True,
            )
            latency_result = self.latency_provider.run(lambda: self.strategy.decide(snapshot))
            decision = self._with_raw_diagnostics(
                latency_result.value,
                order=order,
                model_latency_ns=int(latency_result.latency_ns),
                observation_time=int(ts_event),
                decision_effective_time=int(ts_event) + int(latency_result.latency_ns),
                lifecycle_evaluations=order.lifecycle_evaluations,
                update_idx=int(snapshot.update_idx),
                end_idx=snapshot.end_idx,
                initial_action=DecisionAction.SUBMIT.value,
            )
            if decision.should_cancel:
                self._stats["lifecycle_cancel_requests"] += 1
                order.request_cancel(
                    request_ts=int(ts_event),
                    effective_ts=int(ts_event) + int(latency_result.latency_ns),
                )
                pending.append(
                    _PendingDecision(
                        order=order,
                        decision=decision,
                        effective_time=int(ts_event) + int(latency_result.latency_ns),
                        latency_ns=int(latency_result.latency_ns),
                        kind="cancel",
                    )
                )

    def _update_active_orders(
        self,
        active: list[RawBacktestOrder],
        post_trade: list[RawBacktestOrder],
        mbo: Any,
        book: Any,
    ) -> None:
        remaining: list[RawBacktestOrder] = []
        for order in active:
            order.update_with_mbo(mbo, book)
            if order.status == "FILLED":
                self._stats["filled_orders"] += 1
                post_trade.append(order)
            elif order.is_active:
                remaining.append(order)
        active[:] = remaining

    def _apply_due_initial_decisions(
        self,
        active: list[RawBacktestOrder],
        post_trade: list[RawBacktestOrder],
        completed: list[RawBacktestOrder],
        pending: list[_PendingDecision],
        book: Any,
        ts_event: int,
    ) -> None:
        keep: list[_PendingDecision] = []
        for item in pending:
            if item.kind != "initial" or item.effective_time > ts_event:
                keep.append(item)
                continue
            order = item.order
            if order.is_terminal:
                continue
            action = DecisionAction(item.decision.action)
            if action == DecisionAction.SKIP:
                self._stats["skipped_orders"] += 1
                order.decision = item.decision
                order.skip(book, item.effective_time)
                completed.append(order)
            elif action == DecisionAction.SUBMIT:
                order.decision = item.decision
                if item.effective_time >= order.deadline_time:
                    self._stats["censored_orders"] += 1
                    order.censor(book, item.effective_time, reason="CENSORED_LATENCY")
                    completed.append(order)
                    continue
                order.submit(book, item.effective_time)
                self._stats["submitted_orders"] += 1
                order.last_lifecycle_snapshot_count = len(order.lob_sequence_raw_top5)
                if order.status == "FILLED":
                    self._stats["filled_orders"] += 1
                    post_trade.append(order)
                else:
                    active.append(order)
            else:
                self._stats["skipped_orders"] += 1
                order.decision = item.decision
                order.skip(book, item.effective_time)
                completed.append(order)
        pending[:] = keep

    def _apply_due_cancels(
        self,
        active: list[RawBacktestOrder],
        completed: list[RawBacktestOrder],
        pending: list[_PendingDecision],
        book: Any,
        ts_event: int,
    ) -> None:
        keep_pending: list[_PendingDecision] = []
        canceled_ids: set[int] = set()
        for item in pending:
            if item.kind != "cancel" or item.effective_time > ts_event:
                keep_pending.append(item)
                continue
            order = item.order
            if not order.is_active:
                continue
            if order.apply_cancel_if_due(book, item.effective_time):
                self._stats["canceled_orders"] += 1
                order.decision = item.decision
                completed.append(order)
                canceled_ids.add(id(order))
        pending[:] = keep_pending
        if canceled_ids:
            active[:] = [order for order in active if id(order) not in canceled_ids]

    def _censor_due_orders(
        self,
        active: list[RawBacktestOrder],
        completed: list[RawBacktestOrder],
        book: Any,
        ts_event: int,
    ) -> None:
        remaining: list[RawBacktestOrder] = []
        for order in active:
            if ts_event >= order.deadline_time:
                self._stats["censored_orders"] += 1
                order.censor(book, order.deadline_time)
                completed.append(order)
            else:
                remaining.append(order)
        active[:] = remaining

    def _update_post_trade_orders(
        self,
        post_trade: list[RawBacktestOrder],
        completed: list[RawBacktestOrder],
        book: Any,
        ts_event: int,
    ) -> None:
        remaining: list[RawBacktestOrder] = []
        for order in post_trade:
            order.record_post_trade(book, ts_event)
            if order.needs_post_trade:
                remaining.append(order)
            else:
                completed.append(order)
        post_trade[:] = remaining

    def _finalize_remaining(
        self,
        pending: list[_PendingDecision],
        active: list[RawBacktestOrder],
        post_trade: list[RawBacktestOrder],
        completed: list[RawBacktestOrder],
        book: Any,
        ts_event: int,
    ) -> None:
        for item in pending:
            order = item.order
            if order.is_terminal:
                continue
            if item.kind == "initial" and DecisionAction(item.decision.action) == DecisionAction.SKIP:
                self._stats["skipped_orders"] += 1
                order.decision = item.decision
                skip_time = item.effective_time if item.effective_time <= ts_event else ts_event
                order.skip(book, skip_time)
            else:
                self._stats["censored_orders"] += 1
                censor_time = min(int(ts_event), int(order.deadline_time))
                reason = "CENSORED_TIME" if order.deadline_time <= ts_event else "CENSORED_END"
                order.censor(book, censor_time, reason=reason)
            completed.append(order)
        pending.clear()

        for order in active:
            if not order.is_terminal:
                self._stats["censored_orders"] += 1
                censor_time = min(int(ts_event), int(order.deadline_time))
                reason = "CENSORED_TIME" if order.deadline_time <= ts_event else "CENSORED_END"
                order.censor(book, censor_time, reason=reason)
                completed.append(order)
        active.clear()

        for order in post_trade:
            order.force_complete_post_trade(book)
            completed.append(order)
        post_trade.clear()

    def _snapshot_row(self, order: RawBacktestOrder) -> pd.Series:
        row = dict(order.row.to_dict())
        row.update(
            {
                "side": order.side,
                "price": order.limit_price,
                "best_bid_at_entry": order.best_bid_at_entry,
                "best_ask_at_entry": order.best_ask_at_entry,
                "lob_sequence_raw_top5": order.lob_sequence_raw_top5,
                "toxicity_sequence": order.toxicity_sequence,
                "sequence_length": len(order.lob_sequence_raw_top5),
            }
        )
        return pd.Series(row)

    def _with_raw_diagnostics(
        self,
        decision: TradingDecision,
        *,
        order: RawBacktestOrder,
        model_latency_ns: int,
        observation_time: int,
        decision_effective_time: int,
        lifecycle_evaluations: int,
        update_idx: int,
        end_idx: int | None,
        initial_action: str | None = None,
    ) -> TradingDecision:
        diagnostics = dict(decision.diagnostics)
        diagnostics.update(
            {
                "raw_mode": True,
                "model_latency_ns": int(model_latency_ns),
                "observation_time": int(observation_time),
                "decision_effective_time": int(decision_effective_time),
                "lifecycle_evaluations": int(lifecycle_evaluations),
                "initial_feature_source": order.initial_feature_source,
                "update_idx": int(update_idx),
                "end_idx": int(end_idx) if end_idx is not None else None,
            }
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

    def _build_results(
        self,
        completed: list[RawBacktestOrder],
        metrics: list[ImplementationShortfallMetric],
    ) -> list[BacktestResult]:
        results: list[BacktestResult] = []
        completed_sorted = sorted(completed, key=lambda order: order.row_index)
        for order in completed_sorted:
            decision = self._final_decision(order)
            metric_row = order.to_metric_row()
            metric_values: dict[str, Any] = {}
            for metric in metrics:
                metric_values.update(metric.evaluate(metric_row, decision))
            results.append(
                BacktestResult(
                    row_index=order.row_index,
                    order_id=order.order_id,
                    decision=decision,
                    metrics=metric_values,
                    diagnostics=decision.diagnostics,
                )
            )
        return results

    def _final_decision(self, order: RawBacktestOrder) -> TradingDecision:
        decision = order.decision or TradingDecision(
            action=DecisionAction.SKIP,
            limit_price=order.limit_price,
            reason="missing_decision",
        )
        diagnostics = dict(decision.diagnostics)
        diagnostics.update(
            {
                "raw_status": order.status,
                "submit_time": order.submit_time,
                "end_time": order.end_time,
                "deadline_time": order.deadline_time,
                "has_deadline": True,
                "censor_at_labeled_duration": self.censor_at_labeled_duration,
                "raw_order_max_lifetime_s": self.raw_order_max_lifetime_s,
                "initial_queue_ahead": order.initial_queue_ahead,
                "initial_ids_ahead_count": order.initial_ids_ahead_count,
                "initial_tracking_queue": order.initial_tracking_queue,
                "initial_level_exists": order.initial_level_exists,
                "initial_level_size": order.initial_level_size,
                "submit_crossed": order.submit_crossed,
                "submit_marketable": order.submit_marketable,
                "marketable_after_submit_seen": order.marketable_after_submit_seen,
                "marketable_after_submit_time": order.marketable_after_submit_time,
                "marketable_after_submit_bid": order.marketable_after_submit_bid,
                "marketable_after_submit_ask": order.marketable_after_submit_ask,
                "last_queue_ahead": order.last_queue_ahead,
                "last_ids_ahead_count": order.last_ids_ahead_count,
                "terminal_queue_ahead": order.terminal_queue_ahead,
                "terminal_ids_ahead_count": order.terminal_ids_ahead_count,
                "terminal_level_exists": order.terminal_level_exists,
                "terminal_level_size": order.terminal_level_size,
                "terminal_marketable": order.terminal_marketable,
                "fill_trigger": order.fill_trigger,
                "fill_mbo_action": order.fill_mbo_action,
                "fill_mbo_order_id": order.fill_mbo_order_id,
                "final_update_idx": max(0, len(order.lob_sequence_raw_top5) - 1),
                "final_end_idx": max(0, len(order.lob_sequence_raw_top5) - 1),
                "lifecycle_evaluations": int(order.lifecycle_evaluations),
            }
        )
        if order.status != "FILLED":
            diagnostics["reference_end_idx"] = max(0, len(order.lob_sequence_raw_top5) - 1)
        return TradingDecision(
            action=decision.action,
            limit_price=decision.limit_price,
            size=decision.size,
            reason=decision.reason,
            diagnostics=diagnostics,
        )

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

    def _validate_raw_start_covers_orders(
        self,
        *,
        first_raw_ts: int,
        first_order_ts: int,
    ) -> None:
        if not self.strict_time_coverage:
            return
        if int(first_order_ts) >= int(first_raw_ts):
            return
        raise ValueError(
            "Raw replay starts after the first labeled order entry_time. "
            "The raw engine cannot reconstruct entry features/BBO for orders "
            "before raw data coverage starts. "
            f"first_order_entry_time={first_order_ts} ({_format_ns(first_order_ts)}), "
            f"first_raw_ts={first_raw_ts} ({_format_ns(first_raw_ts)}). "
            "Use a raw DBN file covering the labeled test rows, filter the labeled "
            "dataset to the raw file's date range, or set strict_time_coverage=False "
            "only if you intentionally want partial/late replay behavior."
        )

    def _validate_raw_end_covers_orders(
        self,
        *,
        scheduled: list[tuple[int, int, pd.Series]],
        next_order_idx: int,
        last_raw_ts: int | None,
    ) -> None:
        if not self.strict_time_coverage:
            return
        if next_order_idx >= len(scheduled):
            return
        first_missing_ts = int(scheduled[next_order_idx][0])
        raise ValueError(
            "Raw replay ended before all labeled orders were observed. "
            f"first_unobserved_order_entry_time={first_missing_ts} "
            f"({_format_ns(first_missing_ts)}), "
            f"last_raw_ts={last_raw_ts} ({_format_ns(last_raw_ts) if last_raw_ts else 'none'}). "
            "Use a raw DBN file covering the full labeled order window or filter "
            "the labeled dataset to the raw file's date range."
        )


def _raw_backtest_chunk_worker(kwargs: dict[str, Any]) -> list[BacktestResult]:
    """Run one raw backtest chunk in a worker process."""
    chunk: RawChunk = kwargs.pop("chunk")
    engine = RawDatabentoBacktestEngine(
        kwargs.pop("strategy"),
        raw_path=kwargs.pop("raw_path"),
        feature_builder=kwargs.pop("feature_builder"),
        metrics=kwargs.pop("metrics"),
        latency_provider=kwargs.pop("latency_provider"),
        snapshot_bin_messages=kwargs.pop("snapshot_bin_messages"),
        lifecycle_aware=kwargs.pop("lifecycle_aware"),
        lifecycle_stride=kwargs.pop("lifecycle_stride"),
        lifecycle_max_evaluations=kwargs.pop("lifecycle_max_evaluations"),
        censor_at_labeled_duration=kwargs.pop("censor_at_labeled_duration"),
        raw_order_max_lifetime_s=kwargs.pop("raw_order_max_lifetime_s"),
        strict_time_coverage=kwargs.pop("strict_time_coverage"),
        raw_replay_start_ns=kwargs.pop("raw_replay_start_ns"),
        raw_replay_end_ns=kwargs.pop("raw_replay_end_ns"),
        verbose=kwargs.pop("verbose"),
        progress=kwargs.pop("progress"),
        progress_interval=kwargs.pop("progress_interval"),
    )
    report = engine.run(kwargs.pop("orders_df"))
    results = list(report.results)
    for result in results:
        result.diagnostics["raw_chunk_index"] = int(chunk.index)
        result.diagnostics["raw_chunk_start_ns"] = chunk.start_ns
        result.diagnostics["raw_chunk_end_ns"] = chunk.end_ns
    return results


def _multiprocessing_context(start_method: str | None):
    if start_method is not None:
        return mp.get_context(start_method)
    if os.name == "posix":
        try:
            return mp.get_context("fork")
        except ValueError:
            return None
    return None


def _safe_int(value) -> int | None:
    try:
        if value is None or not np.isfinite(float(value)):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value) -> float | None:
    try:
        if value is None or not np.isfinite(float(value)):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_ns(value: int) -> str:
    try:
        return str(pd.to_datetime(int(value), unit="ns", utc=True))
    except Exception:
        return str(value)
