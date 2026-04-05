from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import pandas as pd
import numpy as np

from ..config import CONFIG
from .utils import ms_to_suffix


class BaseWindowSelector(ABC):
    """
    Base class for selecting a suitable markout horizon from an aggregated
    stats dataframe.

    Subclasses should implement `select_window`.
    """

    @abstractmethod
    def select_window(self, stats: pd.DataFrame) -> dict:
        """
        Select a window from a stats dataframe.

        Parameters
        ----------
        stats : pd.DataFrame
            Aggregated markout stats.

        Returns
        -------
        dict
            Result describing the selected window and related metadata.
        """
        raise NotImplementedError


@dataclass
class StabilizationWindowSelector(BaseWindowSelector):
    """
    Select the earliest horizon whose mean markout reaches a chosen fraction
    of the long-run mean markout magnitude, with matching sign.

    This avoids the sign-flip issue where an opposite-sign mean with similar
    absolute value is incorrectly treated as stabilized.
    """
    fraction: float = CONFIG.labeling.tox_markout_percentage_threshold
    mean_col: str = "mean"
    window_col: str = "window_ms"
    horizon_col: str = "horizon"
    min_abs_long_run_mean: float = 1e-12

    def select_window(self, stats: pd.DataFrame) -> dict:
        if stats.empty:
            return {
                "found": False,
                "reason": "stats is empty",
            }

        df = stats[[self.window_col, self.horizon_col, self.mean_col]].copy()
        df = df.replace([np.inf, -np.inf], np.nan).dropna(
            subset=[self.window_col, self.mean_col]
        )
        df = df.sort_values(self.window_col).reset_index(drop=True)

        if df.empty:
            return {
                "found": False,
                "reason": "no valid rows after dropping NaNs/Infs",
            }

        long_run_row = df.iloc[-1]
        long_run_mean = float(long_run_row[self.mean_col])
        long_run_abs = abs(long_run_mean)

        if long_run_abs < self.min_abs_long_run_mean:
            return {
                "found": False,
                "reason": "long-run mean markout is too close to zero",
                "long_run_window_ms": int(long_run_row[self.window_col]),
                "long_run_horizon": long_run_row[self.horizon_col],
                "long_run_mean": long_run_mean,
                "fraction": self.fraction,
            }

        threshold = self.fraction * long_run_abs
        long_run_sign = np.sign(long_run_mean)

        candidate_mask = df[self.mean_col].abs() >= threshold
        candidate_mask &= np.sign(df[self.mean_col]) == long_run_sign

        candidates = df.loc[candidate_mask]

        if candidates.empty:
            return {
                "found": False,
                "reason": "no horizon reaches the requested fraction with the same sign as long-run mean",
                "threshold_abs_mean": threshold,
                "long_run_window_ms": int(long_run_row[self.window_col]),
                "long_run_horizon": long_run_row[self.horizon_col],
                "long_run_mean": long_run_mean,
                "fraction": self.fraction,
            }

        chosen = candidates.iloc[0]

        return {
            "found": True,
            "method": self.__class__.__name__,
            "chosen_window_ms": int(chosen[self.window_col]),
            "chosen_horizon": chosen[self.horizon_col],
            "chosen_mean": float(chosen[self.mean_col]),
            "threshold_abs_mean": threshold,
            "long_run_window_ms": int(long_run_row[self.window_col]),
            "long_run_horizon": long_run_row[self.horizon_col],
            "long_run_mean": long_run_mean,
            "fraction": self.fraction,
        }



@dataclass
class MarkoutAnalyzer:
    """
    Compute post-trade markouts, aggregate stats, and select a final window.

    Parameters
    ----------
    windows : list[int]
        List of markout horizons in milliseconds.

    window_selector : BaseWindowSelector
        Strategy object for selecting the final horizon.

    use_fill_price_for_markout : bool
        If False, use entry mid as base. If True, use fill price.

    in_bps : bool
        If True, compute markouts in basis points.

    winsorize : bool
        Whether to winsorize markout columns before aggregation.

    winsor_lower / winsor_upper : float
        Quantiles used for winsorization.
    """

    window_selector: BaseWindowSelector

    windows: list[int] = CONFIG.labeling.tox_post_trade_move_windows_ms

    use_fill_price_for_markout: bool = False
    in_bps: bool = True
    winsorize: bool = False
    winsor_lower: float = 0.01
    winsor_upper: float = 0.99

    def compute_markouts_and_stats(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute signed post-trade markouts and aggregate statistics.
        """
        out = df.copy()

        out = out.dropna(subset=[col for col in out.columns if col.startswith("post_trade_best_")])

        out["mid_at_entry"] = (
            out["best_bid_at_entry"] + out["best_ask_at_entry"]
        ) / 2.0
        out["side_sign"] = np.where(out["side"].eq("B"), 1.0, -1.0)
        out["markout_base"] = (
            out["price"].astype(float)
            if self.use_fill_price_for_markout
            else out["mid_at_entry"].astype(float)
        )

        markout_cols: list[str] = []
        horizon_labels: list[str] = []

        for window in self.windows:
            suffix = ms_to_suffix(window)
            bid_col = f"post_trade_best_bid_{suffix}"
            ask_col = f"post_trade_best_ask_{suffix}"

            mid_col = f"post_trade_mid_{suffix}"
            markout_col = f"markout_{suffix}"

            out[mid_col] = (out[bid_col] + out[ask_col]) / 2.0

            signed_markout = out["side_sign"] * (out[mid_col] - out["markout_base"])

            if self.in_bps:
                out[markout_col] = 10000.0 * signed_markout / out["markout_base"]
            else:
                out[markout_col] = signed_markout

            markout_cols.append(markout_col)
            horizon_labels.append(suffix)

        if self.winsorize:
            for col in markout_cols:
                x = out[col]
                lower = x.quantile(self.winsor_lower)
                upper = x.quantile(self.winsor_upper)
                out[col] = x.clip(lower, upper)

        stats = self._aggregate_markout_stats(
            df=out,
            markout_cols=markout_cols,
            horizon_labels=horizon_labels,
        )

        return out, stats

    def _aggregate_markout_stats(
        self,
        df: pd.DataFrame,
        markout_cols: list[str],
        horizon_labels: list[str],
    ) -> pd.DataFrame:
        rows = []

        for window, label, col in zip(self.windows, horizon_labels, markout_cols):
            x = df[col].replace([np.inf, -np.inf], np.nan).dropna()
            n = len(x)

            if n == 0:
                rows.append({
                    "window_ms": window,
                    "horizon": label,
                    "n": 0,
                    "mean": np.nan,
                    "std": np.nan,
                    "se": np.nan,
                    "tstat": np.nan,
                    "median": np.nan,
                    "p05": np.nan,
                    "p25": np.nan,
                    "p75": np.nan,
                    "p95": np.nan,
                    "toxic_rate": np.nan,
                })
                continue

            mean = x.mean()
            std = x.std(ddof=1) if n > 1 else np.nan
            se = std / np.sqrt(n) if n > 1 else np.nan
            tstat = mean / se if pd.notna(se) and se > 0 else np.nan

            rows.append({
                "window_ms": window,
                "horizon": label,
                "n": n,
                "mean": mean,
                "std": std,
                "se": se,
                "tstat": tstat,
                "median": x.median(),
                "p05": x.quantile(0.05),
                "p25": x.quantile(0.25),
                "p75": x.quantile(0.75),
                "p95": x.quantile(0.95),
                "toxic_rate": (x < 0).mean(),
            })

        return pd.DataFrame(rows).sort_values("window_ms").reset_index(drop=True)

    def analyze(
        self,
        df: pd.DataFrame,
    ) -> dict:
        """
        Full workflow:
          1. compute markouts
          2. aggregate stats
          3. select final window

        Returns
        -------
        dict with keys:
            - df_markouts
            - stats
            - selected_window
        """
        df_markouts, stats = self.compute_markouts_and_stats(df)
        
        selected_window = self.window_selector.select_window(stats)

        return {
            "df_markouts": df_markouts,
            "stats": stats,
            "selected_window": selected_window,
        }

    def select_window_from_stats(self, stats: pd.DataFrame) -> dict:
        """
        Run only the window selection step on an existing stats dataframe.
        """
        return self.window_selector.select_window(stats)