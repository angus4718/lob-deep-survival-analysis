from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


NY_TZ = "America/New_York"
EVENT_CENSORED = 0


@dataclass
class OutputPaths:
    root: Path
    figures_dir: Path
    config_path: Path
    summary_path: Path
    daily_summary_path: Path
    trades_path: Path
    report_path: Path


def ensure_project_root(project_root: Path) -> Path:
    project_root = project_root.resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


def build_output_paths(output_dir: Path) -> OutputPaths:
    output_dir = output_dir.resolve()
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    return OutputPaths(
        root=output_dir,
        figures_dir=figures_dir,
        config_path=output_dir / "config.json",
        summary_path=output_dir / "summary.csv",
        daily_summary_path=output_dir / "daily_summary.csv",
        trades_path=output_dir / "trades.csv",
        report_path=output_dir / "report.md",
    )


def dump_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=_json_default))


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def side_to_sign(side: Any) -> float:
    side_str = str(side).upper()
    if side_str == "B":
        return 1.0
    if side_str == "A":
        return -1.0
    raise ValueError(f"Unsupported side value: {side!r}")


def ns_to_session_date(entry_time_ns):
    ts = pd.to_datetime(entry_time_ns, unit="ns", utc=True)
    if isinstance(ts, pd.Series):
        return ts.dt.tz_convert(NY_TZ).dt.normalize()
    return ts.tz_convert(NY_TZ).normalize()


def event_name_to_index(event_names: list[str]) -> dict[str, int]:
    return {name: idx for idx, name in enumerate(event_names)}


def horizon_index(time_grid: np.ndarray, horizon_s: float) -> int:
    idx = int(np.searchsorted(time_grid, horizon_s, side="left"))
    return min(max(idx, 0), len(time_grid) - 1)


def parse_window_suffix_ms(suffix: str) -> int:
    if suffix.endswith("ms"):
        return int(float(suffix[:-2]))
    if suffix.endswith("s"):
        return int(float(suffix[:-1]) * 1000)
    raise ValueError(f"Unsupported window suffix: {suffix}")


def detect_post_trade_windows_ms(columns: list[str]) -> dict[int, tuple[str, str]]:
    pairs: dict[int, tuple[str, str]] = {}
    bid_re = re.compile(r"^post_trade_best_bid_(.+)$")
    ask_lookup: dict[str, str] = {}
    for column in columns:
        if column.startswith("post_trade_best_ask_"):
            ask_lookup[column.removeprefix("post_trade_best_ask_")] = column
    for column in columns:
        match = bid_re.match(column)
        if not match:
            continue
        suffix = match.group(1)
        ask_col = ask_lookup.get(suffix)
        if ask_col is None:
            continue
        pairs[parse_window_suffix_ms(suffix)] = (column, ask_col)
    return dict(sorted(pairs.items()))


def safe_mean(series: pd.Series) -> float:
    if len(series) == 0:
        return float("nan")
    return float(series.mean())


def safe_rate(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def format_pct(value: float) -> str:
    if pd.isna(value):
        return "nan"
    return f"{100.0 * value:.2f}%"
