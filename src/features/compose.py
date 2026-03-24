"""
compose.py
----------
Implements feature transforms for toxic fill prediction and composition utilities:
- ToxicityFeatures: Order book features predictive of favorable vs toxic fills
- ComposeTransforms: Chains multiple transforms and concatenates outputs
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, List

import torch

from .base import BaseLOBTransform
from ..lob_implementation import Book


@dataclass
class ToxicityFeatures(BaseLOBTransform):
    """
    Compute order book features predictive of toxic fills over the lookback window.

    Unlike spatial grid representations (e.g., RepresentationTransform), these
    features are NOT transformed or normalized - they are raw calculated metrics
    directly extracted from each order book snapshot in the lookback period.

    Features include:
    - Spread metrics (absolute and relative)
    - Volume imbalance (bid vs ask)
    - Depth imbalance (weighted by price distance)
    - Price pressure indicators
    - Liquidity concentration
    - VPIN (Volume-Synchronized Probability of Informed Trading)

    Output: (lookback, 18) tensor where each of the 18 metrics is a time series
    computed at each snapshot. RAW values with no spatial transformation.

    Example - Computing features over a 20-snapshot lookback window:
        >>> tox = ToxicityFeatures(n_levels=5)
        >>> snapshots = [(bids_dict_t1, asks_dict_t1), ..., (bids_dict_t20, asks_dict_t20)]
        >>> features = tox.transform_sequence_from_dicts(snapshots, n_lookback=20)
        >>> features.shape  # (20, 18) - each metric evolves over 20 timesteps

    Use with ComposeTransforms to concatenate with spatial representations:
        >>> composed = ComposeTransforms([
        ...     RepresentationTransform(window=10),  # Spatial grid (lookback, 21)
        ...     ToxicityFeatures(n_levels=5)          # Raw metrics (lookback, 18)
        ... ])
        >>> features = composed.transform_sequence_from_dicts(snapshots, n_lookback=20)
        >>> features.shape  # (20, 39) - both branches evolve over 20 timesteps
    """

    n_levels: int = 5  # Number of price levels to consider

    @staticmethod
    def queue_position_feature(current_vahead: Any) -> float:
        """Convert queue-ahead volume into a non-negative scalar feature."""
        try:
            return float(max(int(current_vahead), 0))
        except Exception:
            return 0.0

    def augment_row_with_queue_position(
        self,
        toxicity_row: Sequence[float],
        current_vahead: Any,
    ) -> List[float]:
        """Append queue-position to one toxicity feature row."""
        return list(toxicity_row) + [self.queue_position_feature(current_vahead)]

    def augment_rows_with_queue_position(
        self,
        toxicity_rows: Sequence[Sequence[float]],
        current_vahead: Any,
    ) -> List[List[float]]:
        """Append queue-position to each row in a toxicity feature sequence."""
        if not toxicity_rows:
            return []
        return [
            self.augment_row_with_queue_position(row, current_vahead)
            for row in toxicity_rows
        ]

    def transform_snapshot(self, lob_state: Book) -> torch.Tensor:
        """Extract toxicity-predictive features from a single LOB snapshot."""
        features = []

        # Extract basic book structure
        bids = {px: level.level.size for px, level in lob_state.bids.items()}
        asks = {px: level.level.size for px, level in lob_state.offers.items()}

        if not bids or not asks:
            # Return zeros if book is empty
            return torch.zeros(self._feature_dim(), dtype=torch.float32)

        best_bid = max(bids.keys())
        best_ask = min(asks.keys())
        mid = 0.5 * (best_bid + best_ask)

        # 1. SPREAD FEATURES (2 features)
        spread_abs = best_ask - best_bid
        spread_bps = (spread_abs / mid) * 10000 if mid > 0 else 0
        features.extend([spread_abs, spread_bps])

        # 2. TOP-OF-BOOK IMBALANCE (2 features)
        bid_size_tob = bids.get(best_bid, 0)
        ask_size_tob = asks.get(best_ask, 0)
        total_tob = bid_size_tob + ask_size_tob

        if total_tob > 0:
            imbalance_tob = (bid_size_tob - ask_size_tob) / total_tob
            bid_ratio_tob = bid_size_tob / total_tob
        else:
            imbalance_tob = 0.0
            bid_ratio_tob = 0.5

        features.extend([imbalance_tob, bid_ratio_tob])

        # 3. DEPTH FEATURES (4 features: total, bid, ask, imbalance)
        sorted_bids = sorted(bids.items(), reverse=True)[: self.n_levels]
        sorted_asks = sorted(asks.items())[: self.n_levels]

        total_bid_volume = sum(size for _, size in sorted_bids)
        total_ask_volume = sum(size for _, size in sorted_asks)
        total_depth = total_bid_volume + total_ask_volume

        if total_depth > 0:
            depth_imbalance = (total_bid_volume - total_ask_volume) / total_depth
        else:
            depth_imbalance = 0.0

        features.extend(
            [total_depth, total_bid_volume, total_ask_volume, depth_imbalance]
        )

        # 4. WEIGHTED DEPTH IMBALANCE (2 features)
        # Weight by distance from mid (closer = more important)
        weighted_bid_volume = 0.0
        weighted_ask_volume = 0.0

        for price, size in sorted_bids:
            distance = abs(price - mid)
            weight = 1.0 / (1.0 + distance / mid) if mid > 0 else 0
            weighted_bid_volume += size * weight

        for price, size in sorted_asks:
            distance = abs(price - mid)
            weight = 1.0 / (1.0 + distance / mid) if mid > 0 else 0
            weighted_ask_volume += size * weight

        total_weighted = weighted_bid_volume + weighted_ask_volume
        if total_weighted > 0:
            weighted_imbalance = (
                weighted_bid_volume - weighted_ask_volume
            ) / total_weighted
        else:
            weighted_imbalance = 0.0

        features.extend([weighted_imbalance, total_weighted])

        # 5. PRICE PRESSURE (2 features)
        # Volume-weighted average distance from mid
        if total_bid_volume > 0:
            bid_pressure = (
                sum((mid - price) * size for price, size in sorted_bids)
                / total_bid_volume
            )
        else:
            bid_pressure = 0.0

        if total_ask_volume > 0:
            ask_pressure = (
                sum((price - mid) * size for price, size in sorted_asks)
                / total_ask_volume
            )
        else:
            ask_pressure = 0.0

        features.extend([bid_pressure, ask_pressure])

        # 6. LIQUIDITY CONCENTRATION (2 features)
        # What % of total depth is at best bid/ask?
        if total_depth > 0:
            tob_concentration = (bid_size_tob + ask_size_tob) / total_depth
        else:
            tob_concentration = 0.0

        # Standard deviation of volume across levels (normalized)
        all_volumes = [size for _, size in sorted_bids] + [
            size for _, size in sorted_asks
        ]
        if all_volumes and total_depth > 0:
            mean_vol = total_depth / len(all_volumes)
            variance = sum((v - mean_vol) ** 2 for v in all_volumes) / len(all_volumes)
            vol_std = variance**0.5
            vol_cv = (
                vol_std / mean_vol if mean_vol > 0 else 0
            )  # coefficient of variation
        else:
            vol_cv = 0.0

        features.extend([tob_concentration, vol_cv])

        # 7. MICROPRICE (1 feature)
        # Volume-weighted mid price
        if total_tob > 0:
            microprice = (best_bid * ask_size_tob + best_ask * bid_size_tob) / total_tob
            microprice_offset = (microprice - mid) / mid * 10000 if mid > 0 else 0
        else:
            microprice_offset = 0.0

        features.append(microprice_offset)

        # 8. BOOK SHAPE (2 features)
        # How many levels have meaningful volume (>1% of top level)?
        threshold = (
            0.01 * max(bid_size_tob, ask_size_tob)
            if max(bid_size_tob, ask_size_tob) > 0
            else 0
        )
        significant_bid_levels = sum(1 for _, size in sorted_bids if size > threshold)
        significant_ask_levels = sum(1 for _, size in sorted_asks if size > threshold)

        features.extend([significant_bid_levels, significant_ask_levels])

        # 9. VPIN (1 feature)
        # Volume-Synchronized Probability of Informed Trading
        # Measures order flow toxicity via absolute volume imbalance.
        # Higher VPIN indicates stronger one-sided flow, suggesting informed trading.
        # Range: [0, 1] where 0 = perfectly balanced, 1 = completely one-sided
        if total_depth > 0:
            # Calculate absolute imbalance normalized by total volume
            vpin = abs(total_bid_volume - total_ask_volume) / total_depth
        else:
            vpin = 0.0

        features.append(vpin)

        return torch.tensor(features, dtype=torch.float32)

    def _extract_features_from_dicts(
        self, bids: dict[int, int], asks: dict[int, int]
    ) -> torch.Tensor:
        """Extract toxicity features from bid/ask dictionaries.

        Args:
            bids: Dict mapping price -> size for bid side
            asks: Dict mapping price -> size for ask side

        Returns:
            1D tensor of 18 features
        """
        features = []

        if not bids or not asks:
            return torch.zeros(self._feature_dim(), dtype=torch.float32)

        best_bid = max(bids.keys())
        best_ask = min(asks.keys())
        mid = 0.5 * (best_bid + best_ask)

        # 1. SPREAD FEATURES (2 features)
        spread_abs = best_ask - best_bid
        spread_bps = (spread_abs / mid) * 10000 if mid > 0 else 0
        features.extend([spread_abs, spread_bps])

        # 2. TOP-OF-BOOK IMBALANCE (2 features)
        bid_size_tob = bids.get(best_bid, 0)
        ask_size_tob = asks.get(best_ask, 0)
        total_tob = bid_size_tob + ask_size_tob

        if total_tob > 0:
            imbalance_tob = (bid_size_tob - ask_size_tob) / total_tob
            bid_ratio_tob = bid_size_tob / total_tob
        else:
            imbalance_tob = 0.0
            bid_ratio_tob = 0.5

        features.extend([imbalance_tob, bid_ratio_tob])

        # 3. DEPTH FEATURES (4 features: total, bid, ask, imbalance)
        sorted_bids = sorted(bids.items(), reverse=True)[: self.n_levels]
        sorted_asks = sorted(asks.items())[: self.n_levels]

        total_bid_volume = sum(size for _, size in sorted_bids)
        total_ask_volume = sum(size for _, size in sorted_asks)
        total_depth = total_bid_volume + total_ask_volume

        if total_depth > 0:
            depth_imbalance = (total_bid_volume - total_ask_volume) / total_depth
        else:
            depth_imbalance = 0.0

        features.extend(
            [total_depth, total_bid_volume, total_ask_volume, depth_imbalance]
        )

        # 4. DEPTH-WEIGHTED IMBALANCE (2 features)
        bid_weighted_volume = sum(
            size / (1 + abs(best_bid - px)) for px, size in sorted_bids
        )
        ask_weighted_volume = sum(
            size / (1 + abs(px - best_ask)) for px, size in sorted_asks
        )
        total_weighted = bid_weighted_volume + ask_weighted_volume

        if total_weighted > 0:
            weighted_imbalance = (
                bid_weighted_volume - ask_weighted_volume
            ) / total_weighted
        else:
            weighted_imbalance = 0.0

        features.extend([weighted_imbalance, total_weighted])

        # 5. PRICE PRESSURE (2 features)
        bid_pressure = bid_weighted_volume / (1 + spread_abs) if spread_abs > 0 else 0
        ask_pressure = ask_weighted_volume / (1 + spread_abs) if spread_abs > 0 else 0
        features.extend([bid_pressure, ask_pressure])

        # 6. LIQUIDITY CONCENTRATION (2 features)
        tob_concentration = (
            (bid_size_tob + ask_size_tob) / total_depth if total_depth > 0 else 0
        )

        all_volumes = [size for _, size in sorted_bids] + [
            size for _, size in sorted_asks
        ]
        if all_volumes and sum(all_volumes) > 0:
            mean_vol = sum(all_volumes) / len(all_volumes)
            variance = sum((v - mean_vol) ** 2 for v in all_volumes) / len(all_volumes)
            volume_cv = (variance**0.5) / mean_vol if mean_vol > 0 else 0
        else:
            volume_cv = 0.0

        features.extend([tob_concentration, volume_cv])

        # 7. MICROPRICE DEVIATION (1 feature)
        if total_tob > 0:
            microprice = (best_bid * ask_size_tob + best_ask * bid_size_tob) / total_tob
            microprice_offset_bps = ((microprice - mid) / mid) * 10000 if mid > 0 else 0
        else:
            microprice_offset_bps = 0.0

        features.append(microprice_offset_bps)

        # 8. BOOK SHAPE (2 features: count of significant levels on each side)
        threshold = 0.1 * total_depth if total_depth > 0 else 0
        significant_bid_levels = sum(1 for _, size in sorted_bids if size >= threshold)
        significant_ask_levels = sum(1 for _, size in sorted_asks if size >= threshold)
        features.extend([significant_bid_levels, significant_ask_levels])

        # 9. VPIN (1 feature)
        # Use top-of-book as proxy for recent trade direction
        if total_tob > 0:
            buy_volume = bid_size_tob  # Passive buy orders
            sell_volume = ask_size_tob  # Passive sell orders
            total_volume = buy_volume + sell_volume
            vpin = (
                abs(buy_volume - sell_volume) / total_volume
                if total_volume > 0
                else 0.0
            )
        else:
            vpin = 0.0

        features.append(vpin)

        return torch.tensor(features, dtype=torch.float32)

    def transform_sequence_from_dicts(
        self,
        snapshots: Sequence[tuple],
        n_lookback: int,
        pad_to_length: bool = True,
    ) -> torch.Tensor:
        """Convert bid/ask snapshots into toxicity feature time series.

        Computes 18 toxicity metrics for each snapshot in the lookback window,
        producing a time series where each feature evolves over the lookback period.

        Args:
            snapshots: Sequence of (bids, asks) dict tuples where keys are prices,
                      values are sizes. Typically from _lob_snapshot_buffer.
            n_lookback: Required number of time steps in output. If fewer snapshots
                       are provided, zero-pads at the beginning.

        Returns:
            Float tensor of shape (n_lookback, 18) where:
            - Each row is one snapshot in the lookback period
            - Each column is one of 18 toxicity metrics
            - All features are raw (unscaled) values

            If ``pad_to_length=False``, returns shape ``(T, 18)`` where ``T``
            is the number of available snapshots.
        """
        feature_dim = self._feature_dim()

        if not snapshots:
            if pad_to_length:
                return torch.zeros((n_lookback, feature_dim), dtype=torch.float32)
            return torch.zeros((0, feature_dim), dtype=torch.float32)

        # Extract features for each snapshot
        features_list = []
        for bids_dict, asks_dict in snapshots:
            features = self._extract_features_from_dicts(bids_dict, asks_dict)
            features_list.append(features)

        # Stack into (T, 18) tensor
        if features_list:
            result = torch.stack(features_list, dim=0)
        else:
            if pad_to_length:
                result = torch.zeros((n_lookback, feature_dim), dtype=torch.float32)
            else:
                result = torch.zeros((0, feature_dim), dtype=torch.float32)

        if not pad_to_length:
            return result

        # Pad or truncate to exactly n_lookback timesteps
        T = result.shape[0]
        if T < n_lookback:
            # Zero-pad at beginning
            padding = torch.zeros((n_lookback - T, feature_dim), dtype=torch.float32)
            result = torch.cat([padding, result], dim=0)
        elif T > n_lookback:
            # Take last n_lookback timesteps
            result = result[-n_lookback:, :]

        return result

    def _feature_dim(self) -> int:
        """Return the dimensionality of the feature vector."""
        # 2 (spread) + 2 (tob_imbalance) + 4 (depth) + 2 (weighted_imbalance)
        # + 2 (pressure) + 2 (concentration) + 1 (microprice) + 2 (shape) + 1 (vpin) = 18
        return 18

    def get_feature_names(self) -> List[str]:
        """Return human-readable names for each feature."""
        return [
            "spread_abs",
            "spread_bps",
            "imbalance_tob",
            "bid_ratio_tob",
            "total_depth",
            "total_bid_volume",
            "total_ask_volume",
            "depth_imbalance",
            "weighted_imbalance",
            "total_weighted_volume",
            "bid_pressure",
            "ask_pressure",
            "tob_concentration",
            "volume_cv",
            "microprice_offset_bps",
            "significant_bid_levels",
            "significant_ask_levels",
            "vpin",
        ]


@dataclass
class ComposeTransforms(BaseLOBTransform):
    """
    Chain multiple LOB transforms and concatenate their outputs along the feature dimension.

    Each transform produces its own (lookback, D_i) tensor independently with no interaction.
    Outputs are concatenated along the feature axis to form (lookback, ΣD_i).

    **Multi-Head Architecture Support:**
    This composition naturally supports dual-branch neural networks:
    - **CNN Branch**: Market depth grid (spatial features) → shape (lookback, 21)
    - **MLP Branch**: Toxicity metrics + metadata → shape (lookback, 18)
    - **Fusion**: Concatenate both outputs → shape (lookback, 39) as input

    Example - Combining spatial grid with raw toxicity metrics over 20-timestep lookback:
        >>> from src.features.representation import RepresentationTransform
        >>> from src.features.compose import ToxicityFeatures, ComposeTransforms
        >>>
        >>> composed = ComposeTransforms(transforms=[
        ...     RepresentationTransform(window=10, representation="market_depth"),  # 21D spatial
        ...     ToxicityFeatures(n_levels=5)                                        # 18D raw metrics
        ... ])
        >>>
        >>> # Process 20-snapshot lookback window:
        >>> snapshots = [(bids_1, asks_1), ..., (bids_20, asks_20)]
        >>> features = composed.transform_sequence_from_dicts(snapshots, n_lookback=20)
        >>> features.shape  # (20, 39) - lookback × (21 spatial + 18 toxicity)
        >>> # First 21 columns: market depth evolving over 20 timesteps
        >>> # Last 18 columns: toxicity metrics evolving over 20 timesteps
    """

    transforms: List[BaseLOBTransform] = None

    def __post_init__(self):
        if self.transforms is None:
            self.transforms = []

    def transform_snapshot(self, lob_state: Any) -> torch.Tensor:
        """Apply all transforms and concatenate results."""
        if not self.transforms:
            return torch.empty(0, dtype=torch.float32)

        features = [t.transform_snapshot(lob_state) for t in self.transforms]
        return torch.cat(features, dim=0)

    def transform_sequence(self, lob_states: Sequence[Any]) -> torch.Tensor:
        """Apply all transforms to sequence and concatenate along feature dimension."""
        if not self.transforms:
            return torch.empty((len(lob_states), 0), dtype=torch.float32)

        # Each transform produces (T, D_i), concatenate to get (T, sum(D_i))
        feature_sequences = [t.transform_sequence(lob_states) for t in self.transforms]

        if not feature_sequences:
            return torch.empty((len(lob_states), 0), dtype=torch.float32)

        return torch.cat(feature_sequences, dim=1)

    def transform_sequence_from_dicts(
        self,
        snapshots: Sequence[tuple],
        n_lookback: int,
    ) -> torch.Tensor:
        """Apply all transforms to sequence of dict snapshots and concatenate.

        Args:
            snapshots: Sequence of (bids_dict, asks_dict) tuples
            n_lookback: Required number of time steps in output

        Returns:
            Float tensor of shape (n_lookback, total_feature_dim) where
            total_feature_dim is the sum of all transform dimensions
        """
        if not self.transforms:
            return torch.empty((n_lookback, 0), dtype=torch.float32)

        # Each transform produces (n_lookback, D_i), concatenate to get (n_lookback, sum(D_i))
        feature_sequences = []
        for transform in self.transforms:
            if hasattr(transform, "transform_sequence_from_dicts"):
                # Transform supports dict format
                features = transform.transform_sequence_from_dicts(
                    snapshots, n_lookback
                )
            else:
                # Fallback: return zeros if transform doesn't support dicts
                # This shouldn't happen if transforms are properly implemented
                feature_dim = getattr(transform, "_feature_dim", lambda: 0)()
                features = torch.zeros((n_lookback, feature_dim), dtype=torch.float32)

            feature_sequences.append(features)

        if not feature_sequences:
            return torch.empty((n_lookback, 0), dtype=torch.float32)

        return torch.cat(feature_sequences, dim=1)

    def get_feature_names(self) -> List[str]:
        """Return feature names from all composed transforms."""
        names = []
        for i, transform in enumerate(self.transforms):
            if hasattr(transform, "get_feature_names"):
                names.extend(transform.get_feature_names())
            else:
                # Generate generic names for transforms without names
                dim = (
                    transform.transform_snapshot(None).shape[0]
                    if hasattr(transform, "transform_snapshot")
                    else 0
                )
                names.extend([f"transform{i}_feat{j}" for j in range(dim)])
        return names
