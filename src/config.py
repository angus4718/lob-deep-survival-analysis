"""Project-wide configuration defaults.

Import `CONFIG` where shared parameters are needed. Keep values in one place to
avoid drift across datasets, models, and backtests.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DataConfig:
    price_unit: int = 1e9  # DataBento prices are in 'nanodollars'.
    t_max_s: float = 10.0
    tox_horizon_s: float = 1.0
    runaway_delta_ticks: int = 5


@dataclass(frozen=True)
class FeatureConfig:
    window: int = 20
    tick_size: int = 0.01 * DataConfig.price_unit
    representation: str = "market_depth"  # "moving_window" or "market_depth"


@dataclass(frozen=True)
class ProjectConfig:
    data: DataConfig = DataConfig()
    features: FeatureConfig = FeatureConfig()


CONFIG = ProjectConfig()
