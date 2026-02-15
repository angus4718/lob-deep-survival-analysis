"""Domain enums shared across labeling, modeling, and backtests.

Use EventType as the single source of truth for competing risk IDs in
datasets, model heads, losses, and evaluation/backtest logic.
"""

from enum import IntEnum


class EventType(IntEnum):
    CENSORED = 0
    FAVORABLE_FILL = 1
    TOXIC_FILL = 2
    RUNAWAY = 3

    @classmethod
    def is_valid(cls, value: int) -> bool:
        return value in cls._value2member_map_
