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


@dataclass(frozen=True)
class FeatureConfig:
    window: int = 20
    tick_size: int = 0.01 * DataConfig.price_unit
    representation: str = "market_depth"  # "moving_window" or "market_depth"
    interval_s: float = 0.1


@dataclass(frozen=True)
class LabelingConfig:
    tox_bps: float = 0.2  # bps, indicator of unfavorable fill
    tox_spread_bps: float = 0.2  # use for market condition checks
    tox_duration_s: float = 60
    tox_post_trade_move_window_ms: int = 100
    binning_strategy: str = "log"  # "uniform" or "log"


@dataclass(frozen=True)
class TimeBinningConfig:
    bin_width_s: float = 1.0
    min_time_s: float = 0.01
    max_time_s: float = 300
    n_bins: int = 20


@dataclass(frozen=True)
class ProjectConfig:
    data: DataConfig = DataConfig()
    features: FeatureConfig = FeatureConfig()
    labeling: LabelingConfig = LabelingConfig()
    time_binning: TimeBinningConfig = TimeBinningConfig()


CONFIG = ProjectConfig()
