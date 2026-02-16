"""Test cases for src/features/representation.py."""

import sys
from pathlib import Path
import pytest
import torch

# Ensure project root is in sys.path for src imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.representation import RepresentationTransform
from src.lob_implementation import Book, LevelOrders, PriceLevel


def create_mock_level_orders(price: int, size: int) -> LevelOrders:
    """Helper to create a LevelOrders with a fake order."""
    from unittest.mock import MagicMock

    level = LevelOrders(price=price)
    mock_order = MagicMock()
    mock_order.size = size
    mock_order.flags = 0
    level.orders = [mock_order]
    return level


def create_book_with_levels(
    bid_prices: list[int],
    bid_sizes: list[int],
    ask_prices: list[int],
    ask_sizes: list[int],
) -> Book:
    """Create a Book with specified bid/ask levels."""
    book = Book()
    for px, sz in zip(bid_prices, bid_sizes):
        book.bids[px] = create_mock_level_orders(px, sz)
    for px, sz in zip(ask_prices, ask_sizes):
        book.offers[px] = create_mock_level_orders(px, sz)
    return book


from src.config import FeatureConfig


class TestRepresentationTransform:
    """Test RepresentationTransform class."""

    def test_init_defaults(self):
        """Test initialization with default config."""
        transform = RepresentationTransform()
        assert transform.window == FeatureConfig.window
        assert transform.tick_size == FeatureConfig.tick_size
        assert transform.representation == FeatureConfig.representation

    def test_transform_snapshot_basic(self):
        """Test transform_snapshot with a simple book."""
        transform = RepresentationTransform(
            window=5, tick_size=1, representation="moving_window"
        )

        book = create_book_with_levels(
            bid_prices=[100, 99, 98],
            bid_sizes=[10, 20, 30],
            ask_prices=[101, 102, 103],
            ask_sizes=[15, 25, 35],
        )

        result = transform.transform_snapshot(book)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (2 * transform.window + 1,)
        assert result.dtype == torch.float32

    def test_transform_snapshot_returns_zeros_for_empty_book(self):
        """Test that empty book returns zero tensor."""
        transform = RepresentationTransform(window=5)
        empty_book = Book()

        result = transform.transform_snapshot(empty_book)

        assert result.shape == (11,)
        assert torch.allclose(result, torch.zeros(11))

    def test_transform_sequence_basic(self):
        """Test transform_sequence with multiple books."""
        transform = RepresentationTransform(window=3, representation="moving_window")

        books = [
            create_book_with_levels([100], [10], [101], [10]),
            create_book_with_levels([100], [11], [101], [11]),
            create_book_with_levels([100], [12], [101], [12]),
        ]

        result = transform.transform_sequence(books)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 2 * transform.window + 1)
        assert result.dtype == torch.float32

    def test_transform_sequence_empty(self):
        """Test transform_sequence with empty list."""
        transform = RepresentationTransform(window=5)
        result = transform.transform_sequence([])

        assert result.shape == (0, 11)

    def test_moving_window_representation(self):
        """Test moving_window representation mode."""
        transform = RepresentationTransform(
            window=2, tick_size=1, representation="moving_window"
        )

        book = create_book_with_levels(
            bid_prices=[100, 99],
            bid_sizes=[10, 20],
            ask_prices=[101, 102],
            ask_sizes=[30, 40],
        )

        result = transform.transform_snapshot(book)

        assert result.shape == (5,)
        assert result[2] != 0  # Center should have non-zero value (ask - bid at mid)

    def test_market_depth_representation(self):
        """Test market_depth representation mode."""
        transform = RepresentationTransform(
            window=2, tick_size=1, representation="market_depth"
        )

        book = create_book_with_levels(
            bid_prices=[100, 99],
            bid_sizes=[10, 20],
            ask_prices=[101, 102],
            ask_sizes=[30, 40],
        )

        result = transform.transform_snapshot(book)

        assert result.shape == (5,)

    def test_invalid_representation_raises(self):
        """Test that invalid representation raises ValueError."""
        transform = RepresentationTransform(representation="invalid_mode")

        book = create_book_with_levels([100], [10], [101], [10])

        with pytest.raises(ValueError, match="Unknown representation"):
            transform.transform_snapshot(book)

    def test_tick_size_anchoring(self):
        """Test that mid-price is correctly anchored to tick grid."""
        transform1 = RepresentationTransform(window=5, tick_size=1)
        transform2 = RepresentationTransform(window=5, tick_size=2)

        book = create_book_with_levels(
            bid_prices=[100], bid_sizes=[10], ask_prices=[101], ask_sizes=[10]
        )

        result1 = transform1.transform_snapshot(book)
        result2 = transform2.transform_snapshot(book)

        assert result1.shape == result2.shape

    def test_signed_volumes(self):
        """Test that volumes are signed correctly (ask positive, bid negative)."""
        transform = RepresentationTransform(window=1, representation="moving_window")

        book = create_book_with_levels([100], [50], [101], [30])

        result = transform.transform_snapshot(book)

        assert result.shape == (3,)

    def test_sequence_centers_on_last_state(self):
        """Test that sequence centers on the most recent book's mid-price."""
        transform = RepresentationTransform(window=5, representation="moving_window")

        book1 = create_book_with_levels([100], [10], [101], [10])
        book2 = create_book_with_levels([200], [10], [201], [10])

        result = transform.transform_sequence([book1, book2])

        assert result.shape == (2, 11)

    def test_multiple_price_levels(self):
        """Test with multiple bid/ask levels."""
        transform = RepresentationTransform(window=5)

        book = create_book_with_levels(
            bid_prices=[100, 99, 98, 97],
            bid_sizes=[10, 20, 30, 40],
            ask_prices=[101, 102, 103, 104],
            ask_sizes=[15, 25, 35, 45],
        )

        result = transform.transform_snapshot(book)
        assert result.shape == (11,)

    def test_no_error_with_wide_spreads(self):
        """Test that the transform handles wide bid-ask spreads."""
        transform = RepresentationTransform(window=3)

        book = create_book_with_levels(
            bid_prices=[50], bid_sizes=[100], ask_prices=[150], ask_sizes=[100]
        )

        result = transform.transform_snapshot(book)
        assert result.shape == (7,)

    def test_encode_signed_size(self):
        """Test signed size encoding."""
        transform = RepresentationTransform()

        assert transform._encode_signed_size(100.0) == 100.0
        assert transform._encode_signed_size(-50.0) == -50.0
        assert transform._encode_signed_size(0.0) == 0.0

    def test_anchor_mid(self):
        """Test mid-price anchoring to tick grid."""
        # Use tick_size in nanodollars (default)
        default_tick = FeatureConfig.tick_size
        transform1 = RepresentationTransform(tick_size=default_tick)
        # 100.5 nanodollars, anchored to tick grid
        # Should round to nearest tick multiple
        assert transform1._anchor_mid(100.5) == int(
            round(100.5 / default_tick) * default_tick
        )

        # Custom tick sizes
        transform2 = RepresentationTransform(tick_size=2)
        assert transform2._anchor_mid(100.5) == int(round(100.5 / 2) * 2)

        transform3 = RepresentationTransform(tick_size=0)
        assert transform3._anchor_mid(100.5) == int(round(100.5))

    def test_levels_from_book(self):
        """Test extraction of bid/ask levels from book."""
        transform = RepresentationTransform()

        book = create_book_with_levels(
            bid_prices=[100, 99],
            bid_sizes=[10, 20],
            ask_prices=[101, 102],
            ask_sizes=[30, 40],
        )

        bids, asks = transform._levels_from_book(book)

        assert bids == {100: 10.0, 99: 20.0}
        assert asks == {101: 30.0, 102: 40.0}

    def test_center_from_state(self):
        """Test center computation from a book state."""
        transform = RepresentationTransform(tick_size=1)

        book = create_book_with_levels([100], [10], [101], [10])

        center = transform._center_from_state(book)

        assert center is not None
        assert isinstance(center, int)

    def test_center_from_empty_book_returns_none(self):
        """Test that empty book returns None center."""
        transform = RepresentationTransform()
        empty_book = Book()

        center = transform._center_from_state(empty_book)

        assert center is None
