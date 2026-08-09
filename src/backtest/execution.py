"""Simulated order execution state for raw-feed backtests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.labeling.utils import ms_to_suffix

from .book_utils import best_bid_ask, book_to_snapshot, queue_ahead_at_price


@dataclass
class RawBacktestOrder:
    """State for one candidate order during raw MBO replay."""

    row_index: int
    row: pd.Series
    observation_time: int
    deadline_time: int
    side: str
    limit_price: int
    decision: Any | None = None
    status: str = "PENDING_DECISION"
    end_time: int | None = None
    submit_time: int | None = None
    cancel_request_time: int | None = None
    cancel_effective_time: int | None = None
    model_latency_ns: int = 0
    lifecycle_evaluations: int = 0
    last_lifecycle_snapshot_count: int = 0
    current_vahead: int = 0
    ids_ahead: dict[int, int] = field(default_factory=dict)
    tracking_queue: bool = False
    fill_price: int | None = None
    best_bid_at_entry: int | None = None
    best_ask_at_entry: int | None = None
    best_bid_at_submission: int | None = None
    best_ask_at_submission: int | None = None
    best_bid_at_execution: int | None = None
    best_ask_at_execution: int | None = None
    opportunity_reference_bid: int | None = None
    opportunity_reference_ask: int | None = None
    initial_feature_source: str | None = None
    initial_queue_ahead: int | None = None
    initial_ids_ahead_count: int | None = None
    initial_tracking_queue: bool | None = None
    initial_level_exists: bool | None = None
    initial_level_size: int | None = None
    submit_crossed: bool = False
    submit_marketable: bool = False
    marketable_after_submit_seen: bool = False
    marketable_after_submit_time: int | None = None
    marketable_after_submit_bid: int | None = None
    marketable_after_submit_ask: int | None = None
    last_queue_ahead: int | None = None
    last_ids_ahead_count: int | None = None
    terminal_queue_ahead: int | None = None
    terminal_ids_ahead_count: int | None = None
    terminal_level_exists: bool | None = None
    terminal_level_size: int | None = None
    terminal_marketable: bool | None = None
    fill_trigger: str | None = None
    fill_mbo_action: str | None = None
    fill_mbo_order_id: int | None = None
    lob_sequence_raw_top5: list[list[float]] = field(default_factory=list)
    toxicity_sequence: list[list[float]] = field(default_factory=list)
    post_trade_windows_ms: list[int] = field(
        default_factory=lambda: list(CONFIG.labeling.tox_post_trade_move_windows_ms)
    )
    post_trade_bbo: dict[int, tuple[int | None, int | None]] = field(default_factory=dict)

    @property
    def order_id(self) -> Any:
        return self.row.get("order_id")

    @property
    def is_terminal(self) -> bool:
        return self.status not in {"PENDING_DECISION", "ACTIVE", "PENDING_CANCEL"}

    @property
    def is_active(self) -> bool:
        return self.status in {"ACTIVE", "PENDING_CANCEL"}

    @property
    def needs_post_trade(self) -> bool:
        return self.status == "FILLED" and len(self.post_trade_bbo) < len(
            self.post_trade_windows_ms
        )

    def set_entry_bbo(self, book: Any) -> None:
        bid, ask = best_bid_ask(book)
        self.best_bid_at_entry = getattr(bid, "price", None)
        self.best_ask_at_entry = getattr(ask, "price", None)

    def append_feature_step(self, lob_step: list[float], tox_step: list[float]) -> None:
        self.lob_sequence_raw_top5.append(list(lob_step))
        self.toxicity_sequence.append(list(tox_step))

    def submit(self, book: Any, ts_event: int) -> None:
        """Submit the order into the current raw book state."""
        self.submit_time = int(ts_event)
        bid, ask = best_bid_ask(book)
        self.best_bid_at_submission = getattr(bid, "price", None)
        self.best_ask_at_submission = getattr(ask, "price", None)
        self.submit_marketable = self._is_marketable(book)

        if self.side == "B" and ask is not None and self.limit_price >= int(ask.price):
            self.submit_crossed = True
            self._fill(book, ts_event, fill_price=int(ask.price), trigger="submit_crossed_ask")
            return
        if self.side == "A" and bid is not None and self.limit_price <= int(bid.price):
            self.submit_crossed = True
            self._fill(book, ts_event, fill_price=int(bid.price), trigger="submit_crossed_bid")
            return

        self.current_vahead, self.ids_ahead = queue_ahead_at_price(
            book,
            side=self.side,
            price=int(self.limit_price),
        )
        self.tracking_queue = bool(self.ids_ahead)
        self.initial_queue_ahead = int(self.current_vahead)
        self.initial_ids_ahead_count = int(len(self.ids_ahead))
        self.initial_tracking_queue = bool(self.tracking_queue)
        self.last_queue_ahead = int(self.current_vahead)
        self.last_ids_ahead_count = int(len(self.ids_ahead))
        self.initial_level_exists, self.initial_level_size = self._level_state(book)
        self.status = "ACTIVE"

    def request_cancel(self, request_ts: int, effective_ts: int) -> None:
        if self.status != "ACTIVE":
            return
        self.status = "PENDING_CANCEL"
        self.cancel_request_time = int(request_ts)
        self.cancel_effective_time = int(effective_ts)

    def apply_cancel_if_due(self, book: Any, ts_event: int) -> bool:
        if self.status != "PENDING_CANCEL":
            return False
        if self.cancel_effective_time is None or int(ts_event) < self.cancel_effective_time:
            return False
        self.end_time = int(ts_event)
        self._capture_terminal_queue_state(book)
        self._capture_opportunity_reference(book)
        self.status = "CANCELED_STRATEGY"
        return True

    def update_with_mbo(self, mbo: Any, book: Any) -> None:
        """Update queue state from one raw message."""
        if not self.is_active:
            return
        marketable_fill_price = self._marketable_fill_price(book)
        if marketable_fill_price is not None:
            self._record_marketable_after_submit(book, int(mbo.ts_event))
            if not self.tracking_queue or self.current_vahead <= 0:
                self._fill(
                    book,
                    int(mbo.ts_event),
                    fill_price=self.limit_price,
                    trigger="marketable_after_submit_no_queue",
                    mbo=mbo,
                )
                return
        if self.tracking_queue and self.current_vahead <= 0:
            self._fill(
                book,
                int(mbo.ts_event),
                fill_price=self.limit_price,
                trigger="queue_empty_before_update",
                mbo=mbo,
            )
            return

        order_id = getattr(mbo, "order_id", None)
        action = _action_value(getattr(mbo, "action", None))
        size = int(getattr(mbo, "size", 0) or 0)
        if self.tracking_queue and order_id in self.ids_ahead:
            old_size = int(self.ids_ahead[order_id])
            if action in ("C", "F"):
                loss = min(size, old_size)
                self.current_vahead -= loss
                new_size = old_size - loss
                if new_size > 0:
                    self.ids_ahead[order_id] = new_size
                else:
                    self.ids_ahead.pop(order_id, None)
            elif action == "M":
                if size > old_size:
                    self.current_vahead -= old_size
                    self.ids_ahead.pop(order_id, None)
                elif size < old_size:
                    diff = old_size - size
                    self.current_vahead -= diff
                    self.ids_ahead[order_id] = size
            self.last_queue_ahead = int(self.current_vahead)
            self.last_ids_ahead_count = int(len(self.ids_ahead))

        if self.tracking_queue and self.current_vahead <= 0:
            self._fill(
                book,
                int(mbo.ts_event),
                fill_price=self.limit_price,
                trigger="queue_depleted",
                mbo=mbo,
            )

    def censor(self, book: Any, ts_event: int, reason: str = "CENSORED_TIME") -> None:
        if self.is_terminal:
            return
        self.end_time = int(ts_event)
        self._capture_terminal_queue_state(book)
        self._capture_opportunity_reference(book)
        self.status = reason

    def skip(self, book: Any, ts_event: int) -> None:
        self.end_time = int(ts_event)
        self._capture_terminal_queue_state(book)
        self._capture_opportunity_reference(book)
        self.status = "SKIPPED"

    def record_post_trade(self, book: Any, ts_event: int) -> None:
        if not self.needs_post_trade or self.end_time is None:
            return
        elapsed_ms = (int(ts_event) - int(self.end_time)) / 1e6
        for window_ms in self.post_trade_windows_ms:
            if window_ms in self.post_trade_bbo or elapsed_ms < float(window_ms):
                continue
            bid, ask = best_bid_ask(book)
            self.post_trade_bbo[int(window_ms)] = (
                getattr(bid, "price", None),
                getattr(ask, "price", None),
            )

    def force_complete_post_trade(self, book: Any) -> None:
        if self.status != "FILLED":
            return
        bid, ask = best_bid_ask(book)
        bbo = (getattr(bid, "price", None), getattr(ask, "price", None))
        for window_ms in self.post_trade_windows_ms:
            self.post_trade_bbo.setdefault(int(window_ms), bbo)

    def to_metric_row(self) -> pd.Series:
        out = dict(self.row.to_dict())
        labeled_status_reason = out.get("status_reason")
        labeled_event_type = out.get("event_type")
        labeled_event = out.get("event")
        labeled_fill_price = out.get("fill_price")
        end_time = int(self.end_time or self.observation_time)
        duration_s = max(0.0, (end_time - int(self.observation_time)) / 1e9)
        is_filled = self.status == "FILLED"

        out.update(
            {
                "labeled_status_reason": labeled_status_reason,
                "labeled_event_type": labeled_event_type,
                "labeled_event": labeled_event,
                "labeled_fill_price": labeled_fill_price,
                "order_id": self.order_id,
                "entry_time": int(self.observation_time),
                "duration_s": duration_s,
                "event": 1 if is_filled else 0,
                "event_type": 1 if is_filled else 0,
                "status_reason": "FILLED" if is_filled else self.status,
                "side": self.side,
                "price": self.fill_price if is_filled and self.fill_price is not None else self.limit_price,
                "fill_price": self.fill_price,
                "best_bid_at_entry": self.best_bid_at_entry,
                "best_ask_at_entry": self.best_ask_at_entry,
                "best_bid_at_execution": self.best_bid_at_execution,
                "best_ask_at_execution": self.best_ask_at_execution,
                "opportunity_reference_bid": self.opportunity_reference_bid,
                "opportunity_reference_ask": self.opportunity_reference_ask,
                "lob_sequence_raw_top5": self.lob_sequence_raw_top5 or None,
                "toxicity_sequence": self.toxicity_sequence or None,
                "sequence_length": len(self.lob_sequence_raw_top5),
            }
        )
        for window_ms, (bid, ask) in self.post_trade_bbo.items():
            suffix = ms_to_suffix(int(window_ms))
            out[f"post_trade_best_bid_{suffix}"] = bid
            out[f"post_trade_best_ask_{suffix}"] = ask
        return pd.Series(out)

    def _fill(
        self,
        book: Any,
        ts_event: int,
        *,
        fill_price: int,
        trigger: str | None = None,
        mbo: Any | None = None,
    ) -> None:
        bid, ask = best_bid_ask(book)
        self.best_bid_at_execution = getattr(bid, "price", None)
        self.best_ask_at_execution = getattr(ask, "price", None)
        self.fill_price = int(fill_price)
        self.fill_trigger = trigger
        if mbo is not None:
            self.fill_mbo_action = _action_value(getattr(mbo, "action", None))
            mbo_order_id = getattr(mbo, "order_id", None)
            self.fill_mbo_order_id = int(mbo_order_id) if mbo_order_id is not None else None
        self.status = "FILLED"
        self.end_time = int(ts_event)

    def _capture_opportunity_reference(self, book: Any) -> None:
        bid, ask = best_bid_ask(book)
        self.opportunity_reference_bid = getattr(bid, "price", None)
        self.opportunity_reference_ask = getattr(ask, "price", None)
        snapshot = book_to_snapshot(book, int(self.end_time or self.observation_time))
        if snapshot is None:
            return
        raw_top = _raw_top5_from_snapshot(snapshot)
        if raw_top is not None:
            self.lob_sequence_raw_top5.append(raw_top)

    def _capture_terminal_queue_state(self, book: Any) -> None:
        queue_ahead, ids_ahead = queue_ahead_at_price(
            book,
            side=self.side,
            price=int(self.limit_price),
        )
        self.terminal_queue_ahead = int(queue_ahead)
        self.terminal_ids_ahead_count = int(len(ids_ahead))
        self.terminal_level_exists, self.terminal_level_size = self._level_state(book)
        self.terminal_marketable = self._is_marketable(book)
        self.last_queue_ahead = int(self.current_vahead)
        self.last_ids_ahead_count = int(len(self.ids_ahead))

    def _level_state(self, book: Any) -> tuple[bool, int | None]:
        level_map = getattr(book, "bids", None) if self.side == "B" else getattr(book, "offers", None)
        if level_map is None:
            return False, None
        try:
            level = level_map.get(int(self.limit_price))
        except AttributeError:
            level = level_map[int(self.limit_price)] if int(self.limit_price) in level_map else None
        if level is None:
            return False, None
        level_obj = getattr(level, "level", None)
        size = getattr(level_obj, "size", None)
        if size is None:
            size = sum(int(getattr(order, "size", 0) or 0) for order in getattr(level, "orders", []) or [])
        return True, int(size)

    def _is_marketable(self, book: Any) -> bool:
        return self._marketable_fill_price(book) is not None

    def _marketable_fill_price(self, book: Any) -> int | None:
        bid, ask = best_bid_ask(book)
        if self.side == "B":
            if ask is not None and int(self.limit_price) >= int(ask.price):
                return int(ask.price)
            return None
        if self.side == "A":
            if bid is not None and int(self.limit_price) <= int(bid.price):
                return int(bid.price)
            return None
        return None

    def _record_marketable_after_submit(self, book: Any, ts_event: int) -> None:
        if self.marketable_after_submit_seen or not self._is_marketable(book):
            return
        bid, ask = best_bid_ask(book)
        self.marketable_after_submit_seen = True
        self.marketable_after_submit_time = int(ts_event)
        self.marketable_after_submit_bid = getattr(bid, "price", None)
        self.marketable_after_submit_ask = getattr(ask, "price", None)


def _raw_top5_from_snapshot(snapshot) -> list[float] | None:
    bids, asks, _ = snapshot
    if not bids or not asks:
        return None
    values: list[float] = []
    bid_prices = sorted(bids.keys(), reverse=True)[:5]
    ask_prices = sorted(asks.keys())[:5]
    for idx in range(5):
        if idx < len(ask_prices):
            ask = ask_prices[idx]
            values.extend([float(ask), float(asks[ask])])
        else:
            values.extend([0.0, 0.0])

        if idx < len(bid_prices):
            bid = bid_prices[idx]
            values.extend([float(bid), float(bids[bid])])
        else:
            values.extend([0.0, 0.0])
    return values


def _action_value(action: Any) -> str | None:
    if action is None:
        return None
    value = getattr(action, "value", None)
    if value is not None:
        return str(value)
    return str(action)


def coerce_price(value: Any) -> int | None:
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(price):
        return None
    return int(price)
