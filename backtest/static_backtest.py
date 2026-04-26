from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from .artifact_adapter import LoadedArtifact
    from .common import (
        EVENT_CENSORED,
        detect_post_trade_windows_ms,
        horizon_index,
        safe_mean,
        safe_rate,
        side_to_sign,
    )
    from .data_adapter import DatasetBundle
except ImportError:
    from artifact_adapter import LoadedArtifact
    from common import (
        EVENT_CENSORED,
        detect_post_trade_windows_ms,
        horizon_index,
        safe_mean,
        safe_rate,
        side_to_sign,
    )
    from data_adapter import DatasetBundle


DEFAULT_STRATEGIES = [
    "model",
    "always_market",
    "always_limit",
    "training_prior",
]


@dataclass
class StaticBacktestConfig:
    horizon_s: float
    risk_aversion: float = 1.0
    decision_threshold: float = 0.0
    split_name: str = "test"
    markout_window_ms: int | None = None
    strategies: list[str] | None = None
    cleanup_price_mode: str = "entry_quote"
    market_future_mid_mode: str = "entry_mid"
    cleanup_future_mid_mode: str = "entry_mid"
    use_execution_quote_for_cleanup: bool = True

    def normalized_strategies(self) -> list[str]:
        return self.strategies or list(DEFAULT_STRATEGIES)


@dataclass
class BacktestResult:
    artifact_name: str
    trades: pd.DataFrame
    summary: pd.DataFrame
    daily_summary: pd.DataFrame
    config: dict[str, Any]


def run_static_backtest(
    bundle: DatasetBundle,
    artifact: LoadedArtifact,
    cif: np.ndarray,
    config: StaticBacktestConfig,
    artifact_name: str | None = None,
) -> BacktestResult:
    artifact_name = artifact_name or artifact.base_net_path.stem.removesuffix("_base_net")
    event_idx = artifact.metadata.event_index
    fav_idx = event_idx.get("FAVORABLE_FILL")
    tox_idx = event_idx.get("TOXIC_FILL")
    if fav_idx is None or tox_idx is None:
        raise ValueError(
            "This static backtester expects FAVORABLE_FILL and TOXIC_FILL in artifact metadata."
        )

    horizon_idx_value = horizon_index(artifact.metadata.time_grid, config.horizon_s)
    time_grid = artifact.metadata.time_grid
    post_trade_cols = _resolve_post_trade_columns(
        bundle.eval_frame.columns.tolist(),
        requested_window_ms=config.markout_window_ms,
    )

    rows: list[dict[str, Any]] = []
    strategies = config.normalized_strategies()
    eval_frame = bundle.eval_frame.reset_index(drop=True)

    for row_idx, row in eval_frame.iterrows():
        fav_prob = float(cif[fav_idx, horizon_idx_value, row_idx])
        tox_prob = float(cif[tox_idx, horizon_idx_value, row_idx])
        any_fill_prob = float(np.sum(cif[:, horizon_idx_value, row_idx]))
        other_prob = max(any_fill_prob - fav_prob - tox_prob, 0.0)

        prior_fav = bundle.training_priors["p_favorable_fill"]
        prior_tox = bundle.training_priors["p_toxic_fill"]
        entry_context = _build_entry_context(row=row)

        for strategy in strategies:
            decision = _decide_action(
                strategy_name=strategy,
                fav_prob=fav_prob,
                tox_prob=tox_prob,
                prior_fav=prior_fav,
                prior_tox=prior_tox,
                config=config,
            )
            trade_row = _simulate_trade(
                row=row,
                decision=decision,
                post_trade_cols=post_trade_cols,
                config=config,
                artifact_name=artifact_name,
                horizon_s=config.horizon_s,
                fav_prob=fav_prob,
                tox_prob=tox_prob,
                other_prob=other_prob,
                prior_fav=prior_fav,
                prior_tox=prior_tox,
                model_time_grid=time_grid,
                entry_context=entry_context,
            )
            rows.append(trade_row)

    trades = pd.DataFrame(rows)
    summary = summarize_trades(trades)
    daily_summary = summarize_daily(trades)

    result_config = {
        "artifact_name": artifact_name,
        "horizon_s": config.horizon_s,
        "risk_aversion": config.risk_aversion,
        "decision_threshold": config.decision_threshold,
        "split_name": config.split_name,
        "markout_window_ms": None if post_trade_cols is None else post_trade_cols["window_ms"],
        "strategies": strategies,
        "cleanup_price_mode": config.cleanup_price_mode,
        "market_future_mid_mode": config.market_future_mid_mode,
        "cleanup_future_mid_mode": config.cleanup_future_mid_mode,
        "use_execution_quote_for_cleanup": config.use_execution_quote_for_cleanup,
    }

    return BacktestResult(
        artifact_name=artifact_name,
        trades=trades,
        summary=summary,
        daily_summary=daily_summary,
        config=result_config,
    )


def summarize_trades(trades: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for strategy_name, group in trades.groupby("strategy", sort=False):
        passive_fills = group["exec_mode"].eq("limit_fill")
        cleanup_mask = group["exec_mode"].eq("cleanup_market")
        market_mask = group["exec_mode"].eq("market_entry")
        cancel_mask = group["exec_mode"].eq("cancel")
        toxic_passive = passive_fills & group["observed_toxic_fill"]
        rows.append(
            {
                "artifact_name": group["artifact_name"].iloc[0],
                "strategy": strategy_name,
                "num_trades": int(len(group)),
                "mean_implementation_shortfall": safe_mean(group["implementation_shortfall"]),
                "mean_realized_pnl": safe_mean(group["realized_pnl"]),
                "std_implementation_shortfall": float(group["implementation_shortfall"].std(ddof=0)),
                "std_realized_pnl": float(group["realized_pnl"].std(ddof=0)),
                "limit_decision_rate": safe_rate(float(group["decision_limit"].sum()), float(len(group))),
                "passive_fill_rate": safe_rate(float(passive_fills.sum()), float(len(group))),
                "cleanup_rate": safe_rate(float(cleanup_mask.sum()), float(len(group))),
                "market_entry_rate": safe_rate(float(market_mask.sum()), float(len(group))),
                "cancel_rate": safe_rate(float(cancel_mask.sum()), float(len(group))),
                "toxic_fill_rate_among_passive": safe_rate(
                    float(toxic_passive.sum()), float(passive_fills.sum())
                ),
                "toxic_fill_rate_overall": safe_rate(float(toxic_passive.sum()), float(len(group))),
                "avg_favorable_prob": safe_mean(group["favorable_prob"]),
                "avg_toxic_prob": safe_mean(group["toxic_prob"]),
                "avg_utility": safe_mean(group["utility"]),
            }
        )
    return pd.DataFrame(rows).sort_values("strategy").reset_index(drop=True)


def summarize_daily(trades: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        trades.groupby(["artifact_name", "strategy", "entry_date"], sort=True)
        .agg(
            num_trades=("strategy", "size"),
            mean_implementation_shortfall=("implementation_shortfall", "mean"),
            mean_realized_pnl=("realized_pnl", "mean"),
            passive_fill_rate=("exec_mode", lambda x: float((x == "limit_fill").mean())),
            cleanup_rate=("exec_mode", lambda x: float((x == "cleanup_market").mean())),
            cancel_rate=("exec_mode", lambda x: float((x == "cancel").mean())),
        )
        .reset_index()
    )
    return grouped


def _build_entry_context(row: pd.Series) -> dict[str, float]:
    side_sign = side_to_sign(row["side"])
    entry_bid = float(row["best_bid_at_entry"])
    entry_ask = float(row["best_ask_at_entry"])
    entry_mid = 0.5 * (entry_bid + entry_ask)
    limit_price = float(row.get("price", entry_bid if side_sign > 0 else entry_ask))
    market_price = entry_ask if side_sign > 0 else entry_bid
    return {
        "side_sign": side_sign,
        "entry_bid": entry_bid,
        "entry_ask": entry_ask,
        "entry_mid": entry_mid,
        "limit_price": limit_price,
        "market_price": market_price,
    }


def _decide_action(
    strategy_name: str,
    fav_prob: float,
    tox_prob: float,
    prior_fav: float,
    prior_tox: float,
    config: StaticBacktestConfig,
) -> dict[str, Any]:
    if strategy_name == "always_market":
        return {
            "strategy": strategy_name,
            "decision_limit": False,
            "decision_cancel": False,
            "utility": float("-inf"),
            "decision_source": "rule",
            "favorable_prob": 0.0,
            "toxic_prob": 0.0,
        }

    if strategy_name == "always_limit":
        return {
            "strategy": strategy_name,
            "decision_limit": True,
            "decision_cancel": False,
            "utility": float("inf"),
            "decision_source": "rule",
            "favorable_prob": fav_prob,
            "toxic_prob": tox_prob,
        }

    if strategy_name == "training_prior":
        utility = prior_fav - config.risk_aversion * prior_tox
        return {
            "strategy": strategy_name,
            "decision_limit": utility > config.decision_threshold,
            "decision_cancel": utility <= config.decision_threshold,
            "utility": utility,
            "decision_source": "training_prior",
            "favorable_prob": prior_fav,
            "toxic_prob": prior_tox,
        }

    if strategy_name == "model":
        utility = fav_prob - config.risk_aversion * tox_prob
        return {
            "strategy": strategy_name,
            "decision_limit": utility > config.decision_threshold,
            "decision_cancel": utility <= config.decision_threshold,
            "utility": utility,
            "decision_source": "artifact_cif",
            "favorable_prob": fav_prob,
            "toxic_prob": tox_prob,
        }

    raise ValueError(f"Unsupported strategy_name {strategy_name!r}.")


def _simulate_trade(
    row: pd.Series,
    decision: dict[str, Any],
    post_trade_cols: dict[str, Any] | None,
    config: StaticBacktestConfig,
    artifact_name: str,
    horizon_s: float,
    fav_prob: float,
    tox_prob: float,
    other_prob: float,
    prior_fav: float,
    prior_tox: float,
    model_time_grid: np.ndarray,
    entry_context: dict[str, float],
) -> dict[str, Any]:
    side_sign = entry_context["side_sign"]
    entry_bid = entry_context["entry_bid"]
    entry_ask = entry_context["entry_ask"]
    entry_mid = entry_context["entry_mid"]
    limit_price = entry_context["limit_price"]
    market_price = entry_context["market_price"]

    observed_event = int(row["event_type_competing"])
    observed_fill_within_h = observed_event in (1, 2) and float(row["duration_s"]) <= horizon_s

    exec_mode: str
    exec_price: float
    exec_time_s: float
    future_mid: float
    future_mid_source: str
    is_cleanup: bool

    if decision.get("decision_cancel", False):
        exec_mode = "cancel"
        exec_price = entry_mid
        exec_time_s = 0.0
        future_mid = entry_mid
        future_mid_source = "cancel_no_execution"
        is_cleanup = False
    elif not decision["decision_limit"]:
        exec_mode = "market_entry"
        exec_price = market_price
        exec_time_s = 0.0
        future_mid, future_mid_source = _resolve_non_passive_future_mid(
            row=row,
            entry_mid=entry_mid,
            mode=config.market_future_mid_mode,
        )
        is_cleanup = False
    else:
        if observed_fill_within_h:
            exec_mode = "limit_fill"
            exec_price = limit_price
            exec_time_s = float(row["duration_s"])
            future_mid, future_mid_source = _resolve_limit_future_mid(
                row=row,
                entry_mid=entry_mid,
                post_trade_cols=post_trade_cols,
            )
            is_cleanup = False
        else:
            exec_mode = "cleanup_market"
            exec_price, cleanup_source = _resolve_cleanup_price(
                row=row,
                side_sign=side_sign,
                entry_bid=entry_bid,
                entry_ask=entry_ask,
                fallback_market_price=market_price,
                mode=config.cleanup_price_mode,
                use_execution_quote=config.use_execution_quote_for_cleanup,
            )
            exec_time_s = min(float(row["duration_s"]), horizon_s)
            future_mid, future_mid_source = _resolve_non_passive_future_mid(
                row=row,
                entry_mid=entry_mid,
                mode=config.cleanup_future_mid_mode,
            )
            future_mid_source = f"{future_mid_source}; cleanup_price={cleanup_source}"
            is_cleanup = True

    implementation_shortfall = side_sign * (exec_price - entry_mid)
    realized_pnl = side_sign * (future_mid - exec_price)

    return {
        "artifact_name": artifact_name,
        "strategy": decision["strategy"],
        "decision_source": decision["decision_source"],
        "decision_limit": bool(decision["decision_limit"]),
        "decision_cancel": bool(decision.get("decision_cancel", False)),
        "entry_time": int(row["entry_time"]),
        "entry_date": pd.Timestamp(row["entry_date"]).date().isoformat(),
        "side": str(row["side"]),
        "duration_s": float(row["duration_s"]),
        "observed_event_type": observed_event,
        "observed_fill_within_horizon": bool(observed_fill_within_h),
        "observed_favorable_fill": observed_event == 1 and observed_fill_within_h,
        "observed_toxic_fill": observed_event == 2 and observed_fill_within_h,
        "entry_bid": entry_bid,
        "entry_ask": entry_ask,
        "entry_mid": entry_mid,
        "limit_price": limit_price,
        "market_price_entry": market_price,
        "exec_mode": exec_mode,
        "exec_time_s": exec_time_s,
        "exec_price": exec_price,
        "future_mid": future_mid,
        "future_mid_source": future_mid_source,
        "is_cleanup": is_cleanup,
        "favorable_prob": float(decision["favorable_prob"]),
        "toxic_prob": float(decision["toxic_prob"]),
        "model_favorable_prob": fav_prob,
        "model_toxic_prob": tox_prob,
        "model_other_prob": other_prob,
        "prior_favorable_prob": prior_fav,
        "prior_toxic_prob": prior_tox,
        "utility": float(decision["utility"]),
        "horizon_s": horizon_s,
        "model_time_grid_final_s": float(model_time_grid[-1]),
        "implementation_shortfall": float(implementation_shortfall),
        "realized_pnl": float(realized_pnl),
    }


def _resolve_post_trade_columns(
    columns: list[str],
    requested_window_ms: int | None,
) -> dict[str, Any] | None:
    window_pairs = detect_post_trade_windows_ms(columns)
    if not window_pairs:
        if "best_bid_at_post_trade" in columns and "best_ask_at_post_trade" in columns:
            return {
                "window_ms": None,
                "bid_col": "best_bid_at_post_trade",
                "ask_col": "best_ask_at_post_trade",
            }
        return None

    if requested_window_ms is None:
        selected_window = min(window_pairs)
    else:
        if requested_window_ms not in window_pairs:
            available = sorted(window_pairs)
            raise ValueError(
                f"Requested markout_window_ms={requested_window_ms} not available. "
                f"Available windows: {available}"
            )
        selected_window = requested_window_ms

    bid_col, ask_col = window_pairs[selected_window]
    return {
        "window_ms": selected_window,
        "bid_col": bid_col,
        "ask_col": ask_col,
    }


def _resolve_limit_future_mid(
    row: pd.Series,
    entry_mid: float,
    post_trade_cols: dict[str, Any] | None,
) -> tuple[float, str]:
    if post_trade_cols is not None:
        bid_val = row.get(post_trade_cols["bid_col"])
        ask_val = row.get(post_trade_cols["ask_col"])
        if pd.notna(bid_val) and pd.notna(ask_val):
            return 0.5 * (float(bid_val) + float(ask_val)), "post_trade_window"

    bid_val = row.get("best_bid_at_post_trade")
    ask_val = row.get("best_ask_at_post_trade")
    if pd.notna(bid_val) and pd.notna(ask_val):
        return 0.5 * (float(bid_val) + float(ask_val)), "best_bid_ask_at_post_trade"

    return entry_mid, "entry_mid_fallback"


def _resolve_non_passive_future_mid(
    row: pd.Series,
    entry_mid: float,
    mode: str,
) -> tuple[float, str]:
    if mode == "entry_mid":
        return entry_mid, "entry_mid"
    if mode == "execution_mid":
        bid_val = row.get("best_bid_at_execution")
        ask_val = row.get("best_ask_at_execution")
        if pd.notna(bid_val) and pd.notna(ask_val):
            return 0.5 * (float(bid_val) + float(ask_val)), "execution_mid"
        return entry_mid, "entry_mid_fallback"
    raise ValueError(f"Unsupported future mid mode {mode!r}.")


def _resolve_cleanup_price(
    row: pd.Series,
    side_sign: float,
    entry_bid: float,
    entry_ask: float,
    fallback_market_price: float,
    mode: str,
    use_execution_quote: bool,
) -> tuple[float, str]:
    if mode == "entry_quote":
        return fallback_market_price, "entry_quote"

    if mode == "execution_quote" and use_execution_quote:
        bid_val = row.get("best_bid_at_execution")
        ask_val = row.get("best_ask_at_execution")
        if pd.notna(bid_val) and pd.notna(ask_val):
            price = float(ask_val) if side_sign > 0 else float(bid_val)
            return price, "execution_quote"
        return fallback_market_price, "entry_quote_fallback"

    raise ValueError(f"Unsupported cleanup price mode {mode!r}.")
