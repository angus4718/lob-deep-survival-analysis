# Placeholder for BaseTimeBinner and implementations.
"""
time_binning.py
---------------
Defines BaseTimeBinner (abstract) and implementations:
- bin_time(delta_seconds) -> int
- n_bins property
- UniformTimeBinner (linear bins)
- LogTimeBinner (logarithmic bins for fast events)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

from ..config import CONFIG
from .base import BaseTimeBinner


@dataclass
class UniformTimeBinner(BaseTimeBinner):
    bin_width_s: float = None
    max_time_s: float = None

    def __post_init__(self):
        if self.bin_width_s is None:
            self.bin_width_s = CONFIG.time_binning.bin_width_s
        if self.max_time_s is None:
            self.max_time_s = CONFIG.time_binning.max_time_s
        self.n_bins = int(np.ceil(self.max_time_s / self.bin_width_s))
		
    def bin_time(self, delta_seconds: float) -> int:
        idx = int(delta_seconds // self.bin_width_s)
        return min(idx, self.n_bins - 1)


@dataclass
class LogTimeBinner(BaseTimeBinner):
    min_time_s: float = None
    max_time_s: float = None
    n_bins: int = None
	
    def __post_init__(self):
        if self.min_time_s is None: 
            self.min_time_s = CONFIG.time_binning.min_time_s
        if self.max_time_s is None:
            self.max_time_s = CONFIG.time_binning.max_time_s
        if self.n_bins is None:
            self.n_bins = CONFIG.time_binning.n_bins
		# Logarithmic bin edges
        self.edges = np.logspace(np.log10(self.min_time_s), np.log10(self.max_time_s), self.n_bins + 1)

    def bin_time(self, delta_seconds: float) -> int:
        idx = np.digitize([delta_seconds], self.edges, right=False)[0] - 1
        return min(max(idx, 0), self.n_bins - 1)
