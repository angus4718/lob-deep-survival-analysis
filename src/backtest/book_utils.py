"""Book-state helpers shared by raw backtest components."""

from __future__ import annotations

from typing import Any


BookSnapshot = tuple[dict[int, int], dict[int, int], int]


def book_to_snapshot(book: Any, ts_event: int) -> BookSnapshot | None:
    """Convert a book-like object into aggregate bid/ask snapshot dictionaries."""
    bids = levels_to_size_dict(getattr(book, "bids", None))
    asks = levels_to_size_dict(getattr(book, "offers", None))
    if not bids or not asks:
        return None
    return bids, asks, int(ts_event)


def best_bid_ask(book: Any) -> tuple[Any | None, Any | None]:
    """Return BBO for a book-like object, tolerating missing/invalid books."""
    try:
        return book.bbo()
    except Exception:
        return None, None


def queue_ahead_at_price(book: Any, *, side: str, price: int) -> tuple[int, dict[int, int]]:
    """Return displayed quantity and order ids ahead at ``price`` for ``side``."""
    level_map = getattr(book, "bids", None) if side == "B" else getattr(book, "offers", None)
    if level_map is None:
        return 0, {}
    try:
        level = level_map.get(int(price))
    except AttributeError:
        level = level_map[int(price)] if int(price) in level_map else None
    if level is None:
        return 0, {}

    orders = list(getattr(level, "orders", []) or [])
    ids_ahead: dict[int, int] = {}
    for order in orders:
        order_id = getattr(order, "order_id", None)
        size = int(getattr(order, "size", 0) or 0)
        if order_id is not None and size > 0:
            ids_ahead[int(order_id)] = size
    return int(sum(ids_ahead.values())), ids_ahead


def levels_to_size_dict(level_map: Any) -> dict[int, int]:
    if not level_map:
        return {}
    out: dict[int, int] = {}
    items = level_map.items() if hasattr(level_map, "items") else []
    for price, level in items:
        try:
            level_obj = getattr(level, "level", None)
            size = getattr(level_obj, "size", None)
            if size is None:
                orders = list(getattr(level, "orders", []) or [])
                size = sum(int(getattr(order, "size", 0) or 0) for order in orders)
            out[int(price)] = int(size)
        except Exception:
            continue
    return out
