"""
BaseLabeler: Abstract base class for event/time labeling in survival datasets.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseLabeler(ABC):
    @abstractmethod
    def label(self, insertion_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assign event type and time bin for a simulated order.
        Args:
            insertion_context: dict with order metadata and replay access
        Returns:
            dict with keys: event_type, event_time_bin, extras
        """
        pass
