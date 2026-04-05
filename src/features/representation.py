"""Mid-price centered moving-window and market-depth representations.

Moving-window: A (2W+1) signed-volume vector on a price grid centered at mid-price.
Each entry at offset i stores the signed volume at center + i*tick_size, where ask
volumes are positive and bid volumes are negative.

Market-depth: Cumulative signed volume from the center outward on each side, also
on a (2W+1) grid. Provides a smoother spatial profile reflecting market's ability to
absorb orders at each price level.

Implementation based on Wu et al. (2022): "Towards Robust Representations for Machine
Learning Models in Limit Order Books."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, List, Dict, Optional, Tuple

import torch

from .base import BaseLOBTransform
from ..config import CONFIG
from ..lob_implementation import Book


@dataclass(slots=True)
class RepresentationTransform(BaseLOBTransform):
    """Mid-price centered moving-window or market-depth representation."""

    window: int = CONFIG.features.window
    tick_size: int = CONFIG.features.tick_size
    representation: str = CONFIG.features.representation

    def transform_sequence(self, lob_states: Sequence[Book]) -> torch.Tensor:
        """Convert a sequence of Book objects into a (T, 2W+1) tensor.

        Uses a single price grid centered on the most recent snapshot's
        mid-price so the time dimension aligns to a common reference.
        """
        if not lob_states:
            return torch.zeros((0, 2 * self.window + 1), dtype=torch.float32)

        center = self._center_from_state(lob_states[-1])
        if center is None:
            return torch.zeros(
                (len(lob_states), 2 * self.window + 1), dtype=torch.float32
            )

        features = [self._transform_with_center(state, center) for state in lob_states]
        return torch.stack(features, dim=0)

    def transform_sequence_from_dicts(
        self,
        snapshots: Sequence[tuple],
        n_lookback: int,
        pad_to_length: bool = True,
    ) -> torch.Tensor:
        """Convert bid/ask snapshots into a tensor on a common centered price grid.

        Uses the most recent snapshot's mid-price as the common center for all
        time steps. By default, if fewer than ``n_lookback`` snapshots are
        available the result is zero-padded at the beginning so the output has
        shape ``(n_lookback, 2W+1)`` or ``(n_lookback, 20)`` for raw_top5.
        When ``pad_to_length`` is False, the method returns only the available
        snapshots without padding/truncation.

        Args:
            snapshots: Sequence of ``(bids, asks)`` dicts where keys are integer
                prices and values are aggregate sizes at that level.
            n_lookback: Required number of time steps in the output.

        Returns:
            Float tensor of shape ``(n_lookback, 2W+1)`` or ``(n_lookback, 20)``
            when ``pad_to_length=True``; otherwise ``(T, 2W+1)`` or ``(T, 20)``
            where ``T`` is the number of available snapshots.
        """
        # Determine output width based on representation type
        if self.representation == "raw_top5":
            width = 20
        else:
            width = 2 * self.window + 1

        if not snapshots:
            if pad_to_length:
                return torch.zeros((n_lookback, width), dtype=torch.float32)
            return torch.zeros((0, width), dtype=torch.float32)

        bids_last, asks_last = snapshots[-1]
        if not bids_last or not asks_last:
            if pad_to_length:
                return torch.zeros((n_lookback, width), dtype=torch.float32)
            return torch.zeros((0, width), dtype=torch.float32)

        # For raw_top5, center is not needed
        center = None
        if self.representation != "raw_top5":
            best_bid = max(bids_last.keys())
            best_ask = min(asks_last.keys())
            mid = 0.5 * (best_bid + best_ask)
            center = self._anchor_mid(mid)

        features: List[torch.Tensor] = []
        for bids, asks in snapshots:
            if self.representation == "moving_window":
                vals = self._moving_window(bids, asks, center)
            elif self.representation == "market_depth":
                vals = self._market_depth(bids, asks, center)
            elif self.representation == "raw_top5":
                vals = self._raw_top5(bids, asks)
            else:
                raise ValueError(f"Unknown representation: {self.representation}")
            features.append(torch.tensor(vals, dtype=torch.float32))

        tensor = torch.stack(features, dim=0)

        if not pad_to_length:
            return tensor

        # Pad with zeros at the start when the buffer is not yet full
        if len(features) < n_lookback:
            pad = torch.zeros((n_lookback - len(features), width), dtype=torch.float32)
            tensor = torch.cat([pad, tensor], dim=0)
        elif len(features) > n_lookback:
            tensor = tensor[-n_lookback:, :]

        return tensor

    def transform_snapshot(self, lob_state: Book) -> torch.Tensor:
        """Convert one Book object into a centered (2W+1) vector."""
        center = self._center_from_state(lob_state)
        if center is None:
            return torch.zeros((2 * self.window + 1,), dtype=torch.float32)
        return self._transform_with_center(lob_state, center)

    def _transform_with_center(self, lob_state: Book, center: int) -> torch.Tensor:
        """Build a representation using a provided center price."""
        bid_levels, ask_levels = self._levels_from_book(lob_state)
        if not bid_levels or not ask_levels:
            # Return appropriately sized zero tensor based on representation type
            if self.representation == "raw_top5":
                return torch.zeros((20,), dtype=torch.float32)
            else:
                return torch.zeros((2 * self.window + 1,), dtype=torch.float32)

        if self.representation == "moving_window":
            values = self._moving_window(bid_levels, ask_levels, center)
        elif self.representation == "market_depth":
            values = self._market_depth(bid_levels, ask_levels, center)
        elif self.representation == "raw_top5":
            values = self._raw_top5(bid_levels, ask_levels)
        else:
            raise ValueError(f"Unknown representation: {self.representation}")

        return torch.tensor(values, dtype=torch.float32)

    def _center_from_state(self, lob_state: Book) -> Optional[int]:
        """Compute mid-price center anchored to the tick grid."""
        bid_levels, ask_levels = self._levels_from_book(lob_state)
        if not bid_levels or not ask_levels:
            return None
        best_bid = max(bid_levels.keys())
        best_ask = min(ask_levels.keys())
        mid = 0.5 * (best_bid + best_ask)
        return self._anchor_mid(mid)

    def _levels_from_book(
        self, book: Book
    ) -> Tuple[Dict[int, float], Dict[int, float]]:
        """Extract bid and ask price->size maps from a Book object."""
        bids = {px: level.level.size for px, level in book.bids.items()}
        asks = {px: level.level.size for px, level in book.offers.items()}
        return bids, asks

    def _anchor_mid(self, mid: float) -> int:
        """Snap the mid-price to the tick grid to define the center."""
        return int(round(mid / self.tick_size) * self.tick_size)

    def _moving_window(
        self,
        bids: Dict[int, float],
        asks: Dict[int, float],
        center: int,
    ) -> list[float]:
        """Return signed volumes on a centered price grid."""
        values: List[float] = []
        for offset in range(-self.window, self.window + 1):
            price = center + offset * self.tick_size
            ask_vol = asks.get(price, 0.0)
            bid_vol = bids.get(price, 0.0)
            signed = ask_vol - bid_vol
            values.append(self._encode_signed_size(signed))
        return values

    def _market_depth(
        self,
        bids: Dict[int, float],
        asks: Dict[int, float],
        center: int,
    ) -> List[float]:
        """Return cumulative signed depth from the center outwards."""
        values = [0.0 for _ in range(2 * self.window + 1)]
        cum_ask = 0.0
        cum_bid = 0.0
        for step in range(1, self.window + 1):
            ask_price = center + step * self.tick_size
            bid_price = center - step * self.tick_size
            cum_ask += asks.get(ask_price, 0.0)
            cum_bid += bids.get(bid_price, 0.0)
            values[self.window + step] = self._encode_signed_size(cum_ask)
            values[self.window - step] = self._encode_signed_size(-cum_bid)
        return values

    def _raw_top5(
        self,
        bids: Dict[int, float],
        asks: Dict[int, float],
    ) -> List[float]:
        """Return raw top 5 bid and ask price-volume pairs.

        Returns a 20-feature vector:
        [bid_price_1, bid_vol_1, ..., bid_price_5, bid_vol_5,
         ask_price_1, ask_vol_1, ..., ask_price_5, ask_vol_5]

        Missing levels are filled with 0.0 (price and volume).
        """
        values: List[float] = []

        # Top 5 bid levels (best to 5th best)
        bid_prices_sorted = sorted(bids.keys(), reverse=True)[:5]
        for i in range(5):
            if i < len(bid_prices_sorted):
                price = bid_prices_sorted[i]
                values.append(float(price))
                values.append(float(bids[price]))
            else:
                values.append(0.0)
                values.append(0.0)

        # Top 5 ask levels (best to 5th best)
        ask_prices_sorted = sorted(asks.keys())[:5]
        for i in range(5):
            if i < len(ask_prices_sorted):
                price = ask_prices_sorted[i]
                values.append(float(price))
                values.append(float(asks[price]))
            else:
                values.append(0.0)
                values.append(0.0)

        return values

    def _encode_signed_size(self, signed_size: float) -> float:
        """Keep sign and return absolute size magnitude."""
        sign = 1.0 if signed_size >= 0 else -1.0
        val = abs(signed_size)
        return sign * val
