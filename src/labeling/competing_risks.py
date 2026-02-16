"""
ExecutionCompetingRisksLabeler: Assigns event type and time bin for each simulated order.

- 0: CENSORED
- 1: FAVORABLE_FILL
- 2: TOXIC_FILL
- 3: RUNAWAY

Consistent with EventType enum in domain/enums.py.
"""

from __future__ import annotations
from dataclasses import dataclass

from .base import BaseLabeler
from ..domain.enums import EventType


@dataclass
class ExecutionCompetingRisksLabeler(BaseLabeler):
    pass
