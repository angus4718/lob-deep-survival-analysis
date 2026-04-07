"""
ExecutionCompetingRisksLabeler: Assigns event type and time bin for each simulated order.

- 0: CENSORED
- 1: FAVORABLE_FILL
- 2: TOXIC_FILL

Consistent with EventType enum in domain/enums.py.
"""

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from .base import BaseLabeler
from ..domain.enums import EventType
from ..config import CONFIG
from .utils import ms_to_suffix


@dataclass
class ExecutionCompetingRisksLabeler(BaseLabeler):
    """
    Labels completed orders into competing risk event types.
    Uses configurable thresholds from LabelingConfig for:
    - Toxic fill identification via spread-relative heuristics
    """

    tox_bps: float = None
    tox_spread_bps: float = None
    tox_duration_s: float = None
    selected_window: int = None

    def __post_init__(self):
        if self.tox_bps is None:
            self.tox_bps = CONFIG.labeling.tox_bps
        if self.tox_spread_bps is None:
            self.tox_spread_bps = CONFIG.labeling.tox_spread_bps
        if self.tox_duration_s is None:
            self.tox_duration_s = CONFIG.labeling.tox_duration_s

    def label(self, insertion_context: dict[str, Any]) -> dict[str, Any]:
        """
        Assign event type for a simulated order.

        Args:
            insertion_context: dict with keys like:
                - status_reason: "FILLED", "CENSORED_TIME", "CENSORED_END", etc.
                - duration_s: float, lifetime of the order in seconds
                - price: int, entry price (in price units)
                - side: str, "B" or "A"
                - best_bid_at_entry: int, best bid when order was placed
                - best_ask_at_entry: int, best ask when order was placed
                - best_bid_at_post_trade: int, best bid at the post-trade time window end after order filled
                - best_ask_at_post_trade: int, best ask at the post-trade time window end after order filled

        Returns:
            dict with keys:
                - event_type: int from EventType enum
                - extras: dict with additional fields
        """
        status_reason = insertion_context.get("status_reason", "UNKNOWN")

        # Determine event type and extras
        if status_reason == "FILLED":
            event_type, extra = self._classify_fill(insertion_context)
        elif status_reason in ["CENSORED_TIME", "CENSORED_END"]:
            event_type = EventType.CENSORED
            extra = {}
        else:
            event_type = EventType.CENSORED
            extra = {}

        return {
            "event_type": event_type,
            "extras": extra,
        }

    def _classify_fill(
        self, insertion_context: Dict[str, Any]
    ) -> Tuple[EventType, Dict[str, Any]]:
        """
        Classify a FILLED order as FAVORABLE_FILL or TOXIC_FILL using spread-relative toxicity.

        PnL-Based Heuristic (execution-price normalized):
        - Adverse move: calculated from execution_price (captures actual trading PnL)
        - Toxicity threshold: dynamic half-spread at fill time

        Threshold Selection:
        1. Preferred: use post-trade BBO as proxy for execution-time spread
        2. Fallback: use entry-time BBO only for ultra-fast fills (duration < 100ms)
        3. Invalid: long resting orders without execution-time spread → CENSORED (cannot label reliably)

        Missing Data: If critical fields are None (execution_price, entry BBO), mark as CENSORED.
        Rationale: entry spread is irrelevant for orders resting > 100ms; using stale market
        conditions teaches the model to confidently predict noise.
        """

        side = insertion_context.get("side", "")
        execution_price = insertion_context.get("price")
        duration_s = insertion_context.get("duration_s", 0.0)
        best_bid_at_entry = insertion_context.get("best_bid_at_entry")
        best_ask_at_entry = insertion_context.get("best_ask_at_entry")

        if self.selected_window is None:
            best_bid_at_post_trade = insertion_context.get("best_bid_at_post_trade")
            best_ask_at_post_trade = insertion_context.get("best_ask_at_post_trade")
        else:
            best_bid_at_post_trade = insertion_context.get(
                f"post_trade_best_bid_{ms_to_suffix(self.selected_window)}"
            )
            best_ask_at_post_trade = insertion_context.get(
                f"post_trade_best_ask_{ms_to_suffix(self.selected_window)}"
            )

        extra = {
            "post_trade_adverse_move_bps": None,
            "spread_threshold_bps": None,
            "spread_source": None,
            "post_trade_recorded": None,
            "labeling_valid": True,
        }

        # === CRITICAL DATA VALIDATION ===
        # Execution price required: cannot calculate PnL without it
        if execution_price is None:
            extra["labeling_valid"] = False
            return EventType.CENSORED, extra

        # Entry BBO required: baseline reference for spread thresholds
        if best_bid_at_entry is None or best_ask_at_entry is None:
            extra["labeling_valid"] = False
            return EventType.CENSORED, extra

        # === DETERMINE SPREAD THRESHOLD ===
        # Threshold 1 (Preferred): post-trade spread as proxy for execution-time spread
        if best_bid_at_post_trade is not None and best_ask_at_post_trade is not None:
            post_trade_half_spread = (
                best_ask_at_post_trade - best_bid_at_post_trade
            ) / 2.0
            threshold_bps = (post_trade_half_spread / execution_price) * 10000
            extra["spread_source"] = "execution (post_trade_proxy)"
            extra["post_trade_recorded"] = True
            extra["spread_threshold_bps"] = threshold_bps
            post_trade_mid = (best_bid_at_post_trade + best_ask_at_post_trade) / 2.0
        # Threshold 2 (Fallback): entry spread only for ultra-fast fills (< 100ms)
        elif duration_s < 0.1:
            entry_half_spread = (best_ask_at_entry - best_bid_at_entry) / 2.0
            threshold_bps = (entry_half_spread / execution_price) * 10000
            extra["spread_source"] = "entry (ultra-fast, duration < 100ms)"
            extra["post_trade_recorded"] = False
            extra["spread_threshold_bps"] = threshold_bps
            # Use entry mid as proxy (not ideal, but only fallback for very fast fills)
            post_trade_mid = (best_bid_at_entry + best_ask_at_entry) / 2.0
        # Threshold 3 (Invalid): long resting orders without execution-time spread
        else:
            # Cannot label: entry spread is stale market history for 100ms+ resting orders
            extra["labeling_valid"] = False
            extra["post_trade_recorded"] = False
            return EventType.CENSORED, extra

        # === CALCULATE ADVERSE MOVE (normalized by execution_price) ===
        if side == "B":
            # BUY: adverse when mid price fell after fill
            adverse_move_bps = (
                (execution_price - post_trade_mid) / execution_price
            ) * 10000
        elif side == "A":
            # SELL: adverse when mid price rose after fill
            adverse_move_bps = (
                (post_trade_mid - execution_price) / execution_price
            ) * 10000
        else:
            # Unknown side, cannot classify reliably
            extra["labeling_valid"] = False
            return EventType.CENSORED, extra

        extra["post_trade_adverse_move_bps"] = adverse_move_bps

        # === CLASSIFY ===
        if adverse_move_bps <= threshold_bps:
            return EventType.FAVORABLE_FILL, extra
        else:
            return EventType.TOXIC_FILL, extra
