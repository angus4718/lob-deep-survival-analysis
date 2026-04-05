"""
compose.py
----------
Feature transforms for toxicity modeling and transform composition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, List, Dict

import torch

from .base import BaseLOBTransform
from ..lob_implementation import Book


@dataclass
class ToxicityFeatures(BaseLOBTransform):
    """
    Build compact toxicity features from LOB snapshots.

    Core output is 10 features per snapshot. Sequence output appends
    `time_delta_ms` as an 11th column, and queue position can be appended
    later via `augment_row_with_queue_position`.
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
        """Extract the 10 core toxicity features from a single LOB snapshot."""
        bids = {px: level.level.size for px, level in lob_state.bids.items()}
        asks = {px: level.level.size for px, level in lob_state.offers.items()}
        return self._extract_features_from_dicts(bids, asks)

    def _extract_features_from_dicts(
        self, bids: Dict[int, int], asks: Dict[int, int]
    ) -> torch.Tensor:
        """Extract the 10 core toxicity features from bid/ask dictionaries."""
        features = []

        if not bids or not asks:
            return torch.zeros(self._feature_dim(), dtype=torch.float32)

        best_bid = max(bids.keys())
        best_ask = min(asks.keys())
        mid = 0.5 * (best_bid + best_ask)

        # 1. SPREAD (1 feature)
        spread_abs = best_ask - best_bid
        spread_bps = (spread_abs / mid) * 10000 if mid > 0 else 0
        features.extend([spread_bps])

        # 2. TOP-OF-BOOK IMBALANCE (1 feature)
        bid_size_tob = bids.get(best_bid, 0)
        ask_size_tob = asks.get(best_ask, 0)
        total_tob = bid_size_tob + ask_size_tob

        if total_tob > 0:
            imbalance_tob = (bid_size_tob - ask_size_tob) / total_tob
        else:
            imbalance_tob = 0.0

        features.extend([imbalance_tob])

        # 3. DEPTH IMBALANCE (1 feature)
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
            [depth_imbalance]
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

        # 5. LIQUIDITY CONCENTRATION (2 features)
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

        # 6. MICROPRICE DEVIATION (1 feature)
        if total_tob > 0:
            microprice = (best_bid * ask_size_tob + best_ask * bid_size_tob) / total_tob
            microprice_offset_bps = ((microprice - mid) / mid) * 10000 if mid > 0 else 0
        else:
            microprice_offset_bps = 0.0

        features.append(microprice_offset_bps)

        # 7. BOOK SHAPE (2 features: count of significant levels on each side)
        threshold = 0.1 * total_depth if total_depth > 0 else 0
        significant_bid_levels = sum(1 for _, size in sorted_bids if size >= threshold)
        significant_ask_levels = sum(1 for _, size in sorted_asks if size >= threshold)
        features.extend([significant_bid_levels, significant_ask_levels])

        return torch.tensor(features, dtype=torch.float32)

    def transform_sequence_from_dicts(
        self,
        snapshots: Sequence[tuple],
        n_lookback: int,
        pad_to_length: bool = True,
    ) -> torch.Tensor:
        """Build a toxicity feature sequence from snapshots.

        Returns `(n_lookback, 11)` when padded, or `(T, 11)` with
        `pad_to_length=False`. Columns are 10 core features plus `time_delta_ms`.
        """
        feature_dim = self._feature_dim() + 1  # +1 for time delta

        if not snapshots:
            if pad_to_length:
                return torch.zeros((n_lookback, feature_dim), dtype=torch.float32)
            return torch.zeros((0, feature_dim), dtype=torch.float32)

        # Extract features for each snapshot
        features_list = []
        prev_ts = None
        for snapshot in snapshots:
            if len(snapshot) == 3:
                bids_dict, asks_dict, ts = snapshot
            elif len(snapshot) == 2:
                bids_dict, asks_dict = snapshot
                ts = None
            else:
                continue

            # Compute time delta in milliseconds
            if prev_ts is not None and ts is not None:
                time_delta_ms = (ts - prev_ts) / 1e6  # ns to ms
            else:
                time_delta_ms = 0.0
            prev_ts = ts

            features = self._extract_features_from_dicts(bids_dict, asks_dict)
            # Append time delta as the last feature
            features_with_delta = torch.cat(
                [features, torch.tensor([time_delta_ms], dtype=torch.float32)]
            )
            features_list.append(features_with_delta)

        # Stack into (T, 11) tensor
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
        # 1 (spread_bps) + 1 (imbalance_tob) + 1 (depth_imbalance)
        # + 2 (weighted terms) + 2 (concentration terms)
        # + 1 (microprice offset) + 2 (book shape) = 10
        return 10

    def get_feature_names(self) -> List[str]:
        """Return human-readable names for each feature."""
        return [
            "spread_bps",
            "imbalance_tob",
            "depth_imbalance",
            "weighted_imbalance",
            "total_weighted_volume",
            "tob_concentration",
            "volume_cv",
            "microprice_offset_bps",
            "significant_bid_levels",
            "significant_ask_levels",
            "time_delta_ms",
            "queue_position",
        ]


@dataclass
class ComposeTransforms(BaseLOBTransform):
    """
    Chain transforms and concatenate outputs along the feature dimension.
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
        """Apply each transform to dict snapshots and concatenate features."""
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
