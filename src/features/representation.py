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
from typing import Dict, Sequence

import torch

from .base import BaseLOBTransform
from ..config import FeatureConfig
from ..lob_implementation import Book


@dataclass
class RepresentationTransform(BaseLOBTransform):
    """Mid-price centered moving-window or market-depth representation."""

    window: int = FeatureConfig.window
    tick_size: int = FeatureConfig.tick_size
    representation: str = FeatureConfig.representation

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
            return torch.zeros((2 * self.window + 1,), dtype=torch.float32)

        if self.representation == "moving_window":
            values = self._moving_window(bid_levels, ask_levels, center)
        elif self.representation == "market_depth":
            values = self._market_depth(bid_levels, ask_levels, center)
        else:
            raise ValueError(f"Unknown representation: {self.representation}")

        return torch.tensor(values, dtype=torch.float32)

    def _center_from_state(self, lob_state: Book) -> int | None:
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
    ) -> tuple[Dict[int, float], Dict[int, float]]:
        """Extract bid and ask price->size maps from a Book object."""
        bids = {int(px): float(level.level.size) for px, level in book.bids.items()}
        asks = {int(px): float(level.level.size) for px, level in book.offers.items()}
        return bids, asks

    def _anchor_mid(self, mid: float) -> int:
        """Snap the mid-price to the tick grid to define the center."""
        if self.tick_size <= 0:
            return int(round(mid))
        return int(round(mid / self.tick_size) * self.tick_size)

    def _moving_window(
        self,
        bids: Dict[int, float],
        asks: Dict[int, float],
        center: int,
    ) -> list[float]:
        """Return signed volumes on a centered price grid."""
        values: list[float] = []
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
    ) -> list[float]:
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

    def _encode_signed_size(self, signed_size: float) -> float:
        """Keep sign and return absolute size magnitude."""
        sign = 1.0 if signed_size >= 0 else -1.0
        val = abs(signed_size)
        return sign * val
