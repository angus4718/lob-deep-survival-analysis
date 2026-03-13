"""
ExecutionCompetingRisksLabeler: Assigns event type and time bin for each simulated order.

- 0: CENSORED
- 1: FAVORABLE_FILL
- 2: TOXIC_FILL

Consistent with EventType enum in domain/enums.py.
"""

from dataclasses import dataclass
from typing import Any, Dict

from .base import BaseLabeler
from ..domain.enums import EventType
from ..config import CONFIG
from .time_binning import UniformTimeBinner, LogTimeBinner


@dataclass
class ExecutionCompetingRisksLabeler(BaseLabeler):
    """
    Labels completed orders into competing risk event types.
    Uses configurable thresholds from LabelingConfig for:
    - Toxic fill identification
    - Time binning (via BaseTimeBinner)
    """

    tox_bps: float = None
    tox_spread_bps: float = None
    tox_duration_s: float = None
    binning_strategy: str = None

    def __post_init__(self):
        if self.tox_bps is None:
            self.tox_bps = CONFIG.labeling.tox_bps
        if self.tox_spread_bps is None:
            self.tox_spread_bps = CONFIG.labeling.tox_spread_bps
        if self.tox_duration_s is None:
            self.tox_duration_s = CONFIG.labeling.tox_duration_s
        if self.binning_strategy is None:
            self.binning_strategy = CONFIG.labeling.binning_strategy

        if self.binning_strategy == "uniform":
            self.time_binner = UniformTimeBinner()
        elif self.binning_strategy == "log":
            self.time_binner = LogTimeBinner()
        else:
            raise ValueError(f"Unknown binning strategy: {self.binning_strategy}")

    def label(self, insertion_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assign event type and time bin for a simulated order.

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
                - event_time_bin: int, discretized time bin
                - extras: dict with additional fields
        """
        status_reason = insertion_context.get("status_reason", "UNKNOWN")
        duration_s = insertion_context.get("duration_s", 0.0)

        # Determine event type and extras
        if status_reason == "FILLED":
            event_type, extra = self._classify_fill(insertion_context)
        elif status_reason in ["CENSORED_TIME", "CENSORED_END"]:
            event_type = EventType.CENSORED
            extra = {}
        else:
            event_type = EventType.CENSORED
            extra = {}

        # Use the time binner for binning
        event_time_bin = self.time_binner.bin_time(duration_s)

        return {
            "event_type": event_type,
            "event_time_bin": event_time_bin,
            "extras": extra,
        }

    def _classify_fill(
        self, insertion_context: Dict[str, Any]
    ) -> tuple[EventType, Dict[str, Any]]:
        """
        Classify a FILLED order as FAVORABLE_FILL or TOXIC_FILL.

        Heuristics:
        - Quick fills (< toxicity horizon) with narrow spreads at entry suggest favorable fills
        - Slow fills with wide spreads or adverse mid movement suggest toxic fills
        """

        side = insertion_context.get("side", "")
        entry_price = insertion_context.get("price")
        duration_s = insertion_context.get("duration_s", 0.0)
        best_bid_at_entry = insertion_context.get("best_bid_at_entry")
        best_ask_at_entry = insertion_context.get("best_ask_at_entry")
        best_bid_at_post_trade = insertion_context.get("best_bid_at_post_trade")
        best_ask_at_post_trade = insertion_context.get("best_ask_at_post_trade")

        extra = {
            "post_trade_adverse_move_bps": None,
            "post_trade_spread_bps": None,
            "post_trade_recorded": None,
        }

        if best_bid_at_post_trade is None or best_ask_at_post_trade is None:
            extra["post_trade_recorded"] = False
        else:
            mid_price_at_post_trade = (
                best_bid_at_post_trade + best_ask_at_post_trade
            ) / 2.0
            extra["post_trade_spread_bps"] = (
                (best_ask_at_post_trade - best_bid_at_post_trade)
                / mid_price_at_post_trade
            ) * 10000
            extra["post_trade_recorded"] = True

        if best_bid_at_entry is None or best_ask_at_entry is None:
            return EventType.FAVORABLE_FILL, extra

        mid_price_at_entry = (best_bid_at_entry + best_ask_at_entry) / 2.0
        entry_spread_bps = (
            (best_ask_at_entry - best_bid_at_entry) / mid_price_at_entry
        ) * 10000

        if extra["post_trade_recorded"]:
            if side == "B":
                # Adverse for a buy: midprice dropped after the fill
                extra["post_trade_adverse_move_bps"] = (
                    (mid_price_at_entry - mid_price_at_post_trade) / mid_price_at_entry
                ) * 10000
            elif side == "A":
                # Adverse for a sell: midprice rose after the fill
                extra["post_trade_adverse_move_bps"] = (
                    (mid_price_at_post_trade - mid_price_at_entry) / mid_price_at_entry
                ) * 10000
            # Favorable if adverse move is less than threshold
            if (
                extra["post_trade_adverse_move_bps"] is not None
                and extra["post_trade_adverse_move_bps"] <= self.tox_bps
            ):
                return EventType.FAVORABLE_FILL, extra
        else:
            if duration_s < self.tox_duration_s:
                if entry_spread_bps < self.tox_spread_bps:
                    return EventType.FAVORABLE_FILL, extra

        return EventType.TOXIC_FILL, extra
