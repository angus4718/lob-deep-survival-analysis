from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

try:
    from .common import horizon_index, safe_mean, safe_rate, side_to_sign
    from .dynamic_artifact_adapter import LoadedDynamicArtifact
    from .dynamic_data_adapter import DynamicDatasetBundle
    from .static_backtest import _resolve_limit_future_mid, _resolve_non_passive_future_mid, _resolve_post_trade_columns
except ImportError:
    from common import horizon_index, safe_mean, safe_rate, side_to_sign
    from dynamic_artifact_adapter import LoadedDynamicArtifact
    from dynamic_data_adapter import DynamicDatasetBundle
    from static_backtest import _resolve_limit_future_mid, _resolve_non_passive_future_mid, _resolve_post_trade_columns


DEFAULT_DYNAMIC_STRATEGIES = (
    "model",
    "always_market",
    "always_limit",
    "training_prior",
)


@dataclass
class DynamicBacktestConfig:
    horizon_s: float = 60.0
    risk_aversion: float = 1.0
    decision_threshold: float = 0.0
    split_name: str = "test"
    reeval_interval_s: float | None = None
    delta_proc_s: float = 0.0
    delta_lat_s: float = 0.0
    strategies: list[str] | tuple[str, ...] = DEFAULT_DYNAMIC_STRATEGIES
    markout_window_ms: int | None = None
    market_future_mid_mode: str = "entry_mid"
    cleanup_future_mid_mode: str = "entry_mid"

    def normalized_strategies(self) -> list[str]:
        seen: list[str] = []
        for strategy in self.strategies:
            strategy = str(strategy).strip()
            if strategy and strategy not in seen:
                seen.append(strategy)
        return seen


@dataclass
class DynamicBacktestResult:
    artifact_name: str
    trades: pd.DataFrame
    summary: pd.DataFrame
    daily_summary: pd.DataFrame
    config: dict[str, Any]


def run_dynamic_backtest(
    bundle: DynamicDatasetBundle,
    artifact: LoadedDynamicArtifact,
    cif: np.ndarray,
    config: DynamicBacktestConfig,
    artifact_name: str | None = None,
) -> DynamicBacktestResult:
    artifact_name = artifact_name or artifact.base_net_path.stem.removesuffix("_base_net")
    frame = bundle.eval_frame.copy().reset_index(drop=True)

    _attach_model_outputs(frame, bundle, artifact, cif, config)
    if "order_id" not in frame.columns:
        raise ValueError("Dynamic backtest requires an order_id column after dataset normalization.")

    post_trade_cols = _resolve_post_trade_columns(
        frame.columns.tolist(),
        requested_window_ms=config.markout_window_ms,
    )

    rows: list[dict[str, Any]] = []
    grouped = frame.groupby("order_id", sort=False)
    prior_fav = bundle.training_priors["p_favorable_fill"]
    prior_tox = bundle.training_priors["p_toxic_fill"]
    positive_events = [
        name for name in artifact.metadata.event_names if name not in {"TOXIC_FILL", "CENSORED"}
    ]

    for _, order_frame in grouped:
        order_frame = order_frame.sort_values(["decision_time_ns", "eval_step"]).reset_index(drop=True)
        for strategy in config.normalized_strategies():
            rows.append(
                _simulate_order(
                    order_frame=order_frame,
                    strategy_name=strategy,
                    prior_fav=prior_fav,
                    prior_tox=prior_tox,
                    config=config,
                    artifact_name=artifact_name,
                    post_trade_cols=post_trade_cols,
                    positive_events=positive_events,
                )
            )

    trades = pd.DataFrame(rows)
    summary = summarize_dynamic_trades(trades)
    daily_summary = summarize_dynamic_daily(trades)

    result_config = {
        "artifact_name": artifact_name,
        "horizon_s": config.horizon_s,
        "risk_aversion": config.risk_aversion,
        "decision_threshold": config.decision_threshold,
        "split_name": config.split_name,
        "reeval_interval_s": config.reeval_interval_s,
        "delta_proc_s": config.delta_proc_s,
        "delta_lat_s": config.delta_lat_s,
        "markout_window_ms": None if post_trade_cols is None else post_trade_cols["window_ms"],
        "strategies": config.normalized_strategies(),
        "market_future_mid_mode": config.market_future_mid_mode,
        "cleanup_future_mid_mode": config.cleanup_future_mid_mode,
        "proposal_action_rule": "utility > threshold => limit, otherwise market/cancel-and-take",
    }

    return DynamicBacktestResult(
        artifact_name=artifact_name,
        trades=trades,
        summary=summary,
        daily_summary=daily_summary,
        config=result_config,
    )


def _attach_model_outputs(
    frame: pd.DataFrame,
    bundle: DynamicDatasetBundle,
    artifact: LoadedDynamicArtifact,
    cif: np.ndarray,
    config: DynamicBacktestConfig,
) -> None:
    event_idx = artifact.metadata.event_index
    tox_idx = event_idx.get("TOXIC_FILL")
    if tox_idx is None:
        raise ValueError("Dynamic backtest expects TOXIC_FILL in artifact metadata.")

    positive_event_names = [
        name for name in artifact.metadata.event_names if name not in {"TOXIC_FILL", "CENSORED"}
    ]
    if not positive_event_names:
        raise ValueError("Dynamic backtest needs at least one non-toxic event for utility.")

    time_grid = artifact.metadata.time_grid
    favorable_probs: list[float] = []
    toxic_probs: list[float] = []
    utility_probs: list[float] = []
    horizon_remaining: list[float] = []

    for row_idx, row in frame.iterrows():
        remaining_s = max(config.horizon_s - float(row["elapsed_s_from_entry"]), 0.0)
        h_idx = horizon_index(time_grid, remaining_s)
        fav_prob = 0.0
        for event_name in positive_event_names:
            fav_prob += float(cif[event_idx[event_name], h_idx, row_idx])
        tox_prob = float(cif[tox_idx, h_idx, row_idx])
        utility = fav_prob - config.risk_aversion * tox_prob
        favorable_probs.append(fav_prob)
        toxic_probs.append(tox_prob)
        utility_probs.append(utility)
        horizon_remaining.append(remaining_s)

    frame["favorable_prob"] = favorable_probs
    frame["toxic_prob"] = toxic_probs
    frame["utility"] = utility_probs
    frame["remaining_horizon_s"] = horizon_remaining
    frame["observed_fill_time_ns"] = (
        frame["decision_time_ns"] + (frame["duration_s"].astype(float) * 1e9).round().astype(np.int64)
    )


def _simulate_order(
    order_frame: pd.DataFrame,
    strategy_name: str,
    prior_fav: float,
    prior_tox: float,
    config: DynamicBacktestConfig,
    artifact_name: str,
    post_trade_cols: dict[str, Any] | None,
    positive_events: list[str],
) -> dict[str, Any]:
    first_row = order_frame.iloc[0]
    side_sign = side_to_sign(first_row["side"])
    entry_bid = float(first_row["best_bid_at_entry"])
    entry_ask = float(first_row["best_ask_at_entry"])
    entry_mid = 0.5 * (entry_bid + entry_ask)
    limit_price = float(first_row["price"])
    horizon_end_ns = int(first_row["entry_time"] + round(config.horizon_s * 1e9))
    action_delay_ns = int(round((config.delta_proc_s + config.delta_lat_s) * 1e9))

    observation_rows = _select_observation_rows(order_frame, config.reeval_interval_s)
    observed_event = int(first_row["event_type_competing"])
    passive_fill_time_ns = None
    if observed_event in (1, 2):
        passive_fill_time_ns = int(first_row["entry_time"] + round(float(first_row["duration_s"]) * 1e9))

    if strategy_name == "always_market":
        exec_time_ns = min(
            int(first_row["decision_time_ns"]) + action_delay_ns,
            horizon_end_ns,
        )
        market_ref = _row_at_or_before(order_frame, exec_time_ns)
        exec_price = _market_price_from_row(market_ref, side_sign)
        future_mid, future_mid_source = _resolve_non_passive_future_mid(
            row=market_ref,
            entry_mid=entry_mid,
            mode=config.market_future_mid_mode,
        )
        return _build_trade_row(
            artifact_name=artifact_name,
            order_frame=order_frame,
            strategy_name=strategy_name,
            decision_source="rule",
            utility=float("-inf"),
            favorable_prob=0.0,
            toxic_prob=0.0,
            exec_mode="market_entry",
            exec_price=exec_price,
            exec_time_ns=exec_time_ns,
            future_mid=future_mid,
            future_mid_source=future_mid_source,
            entry_mid=entry_mid,
            side_sign=side_sign,
            observed_event=observed_event,
            observed_fill_time_ns=passive_fill_time_ns,
            horizon_end_ns=horizon_end_ns,
            num_revals=1,
            trigger_step=int(first_row["eval_step"]),
            trigger_time_ns=int(first_row["decision_time_ns"]),
            is_cleanup=False,
        )

    if strategy_name == "training_prior":
        trigger_utility = prior_fav - config.risk_aversion * prior_tox
        if trigger_utility <= config.decision_threshold:
            exec_time_ns = min(
                int(first_row["decision_time_ns"]) + action_delay_ns,
                horizon_end_ns,
            )
            if passive_fill_time_ns is not None and passive_fill_time_ns <= exec_time_ns:
                fill_row = _row_at_or_before(order_frame, passive_fill_time_ns)
                future_mid, future_mid_source = _resolve_limit_future_mid(
                    row=fill_row,
                    entry_mid=entry_mid,
                    post_trade_cols=post_trade_cols,
                )
                return _build_trade_row(
                    artifact_name=artifact_name,
                    order_frame=order_frame,
                    strategy_name=strategy_name,
                    decision_source="training_prior",
                    utility=trigger_utility,
                    favorable_prob=prior_fav,
                    toxic_prob=prior_tox,
                    exec_mode="limit_fill",
                    exec_price=limit_price,
                    exec_time_ns=passive_fill_time_ns,
                    future_mid=future_mid,
                    future_mid_source=future_mid_source,
                    entry_mid=entry_mid,
                    side_sign=side_sign,
                    observed_event=observed_event,
                    observed_fill_time_ns=passive_fill_time_ns,
                    horizon_end_ns=horizon_end_ns,
                    num_revals=1,
                    trigger_step=int(first_row["eval_step"]),
                    trigger_time_ns=int(first_row["decision_time_ns"]),
                    is_cleanup=False,
                )
            market_ref = _row_at_or_before(order_frame, exec_time_ns)
            exec_price = _market_price_from_row(market_ref, side_sign)
            future_mid, future_mid_source = _resolve_non_passive_future_mid(
                row=market_ref,
                entry_mid=entry_mid,
                mode=config.market_future_mid_mode,
            )
            return _build_trade_row(
                artifact_name=artifact_name,
                order_frame=order_frame,
                strategy_name=strategy_name,
                decision_source="training_prior",
                utility=trigger_utility,
                favorable_prob=prior_fav,
                toxic_prob=prior_tox,
                exec_mode="market_after_cancel",
                exec_price=exec_price,
                exec_time_ns=exec_time_ns,
                future_mid=future_mid,
                future_mid_source=future_mid_source,
                entry_mid=entry_mid,
                side_sign=side_sign,
                observed_event=observed_event,
                observed_fill_time_ns=passive_fill_time_ns,
                horizon_end_ns=horizon_end_ns,
                num_revals=1,
                trigger_step=int(first_row["eval_step"]),
                trigger_time_ns=int(first_row["decision_time_ns"]),
                is_cleanup=False,
            )

        # Otherwise behaves like always_limit for the whole order.
        return _finish_passive_strategy(
            artifact_name=artifact_name,
            order_frame=order_frame,
            strategy_name=strategy_name,
            decision_source="training_prior",
            utility=trigger_utility,
            favorable_prob=prior_fav,
            toxic_prob=prior_tox,
            entry_mid=entry_mid,
            limit_price=limit_price,
            side_sign=side_sign,
            observed_event=observed_event,
            passive_fill_time_ns=passive_fill_time_ns,
            horizon_end_ns=horizon_end_ns,
            post_trade_cols=post_trade_cols,
            config=config,
            num_revals=1,
            trigger_step=int(first_row["eval_step"]),
            trigger_time_ns=int(first_row["decision_time_ns"]),
        )

    if strategy_name == "always_limit":
        return _finish_passive_strategy(
            artifact_name=artifact_name,
            order_frame=order_frame,
            strategy_name=strategy_name,
            decision_source="rule",
            utility=float("inf"),
            favorable_prob=float(first_row["favorable_prob"]),
            toxic_prob=float(first_row["toxic_prob"]),
            entry_mid=entry_mid,
            limit_price=limit_price,
            side_sign=side_sign,
            observed_event=observed_event,
            passive_fill_time_ns=passive_fill_time_ns,
            horizon_end_ns=horizon_end_ns,
            post_trade_cols=post_trade_cols,
            config=config,
            num_revals=1,
            trigger_step=int(first_row["eval_step"]),
            trigger_time_ns=int(first_row["decision_time_ns"]),
        )

    # Dynamic proposal-aligned policy:
    # keep limit while utility > threshold; otherwise cancel and cross the spread.
    for obs_count, row in enumerate(observation_rows, start=1):
        decision_time_ns = int(row["decision_time_ns"])
        action_time_ns = min(decision_time_ns + action_delay_ns, horizon_end_ns)
        if passive_fill_time_ns is not None and passive_fill_time_ns <= action_time_ns:
            fill_row = _row_at_or_before(order_frame, passive_fill_time_ns)
            future_mid, future_mid_source = _resolve_limit_future_mid(
                row=fill_row,
                entry_mid=entry_mid,
                post_trade_cols=post_trade_cols,
            )
            return _build_trade_row(
                artifact_name=artifact_name,
                order_frame=order_frame,
                strategy_name=strategy_name,
                decision_source="artifact_cif",
                utility=float(row["utility"]),
                favorable_prob=float(row["favorable_prob"]),
                toxic_prob=float(row["toxic_prob"]),
                exec_mode="limit_fill",
                exec_price=limit_price,
                exec_time_ns=passive_fill_time_ns,
                future_mid=future_mid,
                future_mid_source=future_mid_source,
                entry_mid=entry_mid,
                side_sign=side_sign,
                observed_event=observed_event,
                observed_fill_time_ns=passive_fill_time_ns,
                horizon_end_ns=horizon_end_ns,
                num_revals=obs_count,
                trigger_step=int(row["eval_step"]),
                trigger_time_ns=decision_time_ns,
                is_cleanup=False,
            )

        if float(row["utility"]) <= config.decision_threshold:
            market_ref = _row_at_or_before(order_frame, action_time_ns)
            exec_price = _market_price_from_row(market_ref, side_sign)
            future_mid, future_mid_source = _resolve_non_passive_future_mid(
                row=market_ref,
                entry_mid=entry_mid,
                mode=config.market_future_mid_mode,
            )
            return _build_trade_row(
                artifact_name=artifact_name,
                order_frame=order_frame,
                strategy_name=strategy_name,
                decision_source="artifact_cif",
                utility=float(row["utility"]),
                favorable_prob=float(row["favorable_prob"]),
                toxic_prob=float(row["toxic_prob"]),
                exec_mode="market_after_cancel",
                exec_price=exec_price,
                exec_time_ns=action_time_ns,
                future_mid=future_mid,
                future_mid_source=future_mid_source,
                entry_mid=entry_mid,
                side_sign=side_sign,
                observed_event=observed_event,
                observed_fill_time_ns=passive_fill_time_ns,
                horizon_end_ns=horizon_end_ns,
                num_revals=obs_count,
                trigger_step=int(row["eval_step"]),
                trigger_time_ns=decision_time_ns,
                is_cleanup=False,
            )

    last_row = observation_rows[-1] if observation_rows else first_row
    return _finish_passive_strategy(
        artifact_name=artifact_name,
        order_frame=order_frame,
        strategy_name=strategy_name,
        decision_source="artifact_cif",
        utility=float(last_row["utility"]),
        favorable_prob=float(last_row["favorable_prob"]),
        toxic_prob=float(last_row["toxic_prob"]),
        entry_mid=entry_mid,
        limit_price=limit_price,
        side_sign=side_sign,
        observed_event=observed_event,
        passive_fill_time_ns=passive_fill_time_ns,
        horizon_end_ns=horizon_end_ns,
        post_trade_cols=post_trade_cols,
        config=config,
        num_revals=len(observation_rows),
        trigger_step=int(last_row["eval_step"]),
        trigger_time_ns=int(last_row["decision_time_ns"]),
    )


def _finish_passive_strategy(
    artifact_name: str,
    order_frame: pd.DataFrame,
    strategy_name: str,
    decision_source: str,
    utility: float,
    favorable_prob: float,
    toxic_prob: float,
    entry_mid: float,
    limit_price: float,
    side_sign: float,
    observed_event: int,
    passive_fill_time_ns: int | None,
    horizon_end_ns: int,
    post_trade_cols: dict[str, Any] | None,
    config: DynamicBacktestConfig,
    num_revals: int,
    trigger_step: int,
    trigger_time_ns: int,
) -> dict[str, Any]:
    if passive_fill_time_ns is not None and passive_fill_time_ns <= horizon_end_ns:
        fill_row = _row_at_or_before(order_frame, passive_fill_time_ns)
        future_mid, future_mid_source = _resolve_limit_future_mid(
            row=fill_row,
            entry_mid=entry_mid,
            post_trade_cols=post_trade_cols,
        )
        return _build_trade_row(
            artifact_name=artifact_name,
            order_frame=order_frame,
            strategy_name=strategy_name,
            decision_source=decision_source,
            utility=utility,
            favorable_prob=favorable_prob,
            toxic_prob=toxic_prob,
            exec_mode="limit_fill",
            exec_price=limit_price,
            exec_time_ns=passive_fill_time_ns,
            future_mid=future_mid,
            future_mid_source=future_mid_source,
            entry_mid=entry_mid,
            side_sign=side_sign,
            observed_event=observed_event,
            observed_fill_time_ns=passive_fill_time_ns,
            horizon_end_ns=horizon_end_ns,
            num_revals=num_revals,
            trigger_step=trigger_step,
            trigger_time_ns=trigger_time_ns,
            is_cleanup=False,
        )

    cleanup_row = _row_at_or_before(order_frame, horizon_end_ns)
    exec_price = _market_price_from_row(cleanup_row, side_sign)
    future_mid, future_mid_source = _resolve_non_passive_future_mid(
        row=cleanup_row,
        entry_mid=entry_mid,
        mode=config.cleanup_future_mid_mode,
    )
    return _build_trade_row(
        artifact_name=artifact_name,
        order_frame=order_frame,
        strategy_name=strategy_name,
        decision_source=decision_source,
        utility=utility,
        favorable_prob=favorable_prob,
        toxic_prob=toxic_prob,
        exec_mode="cleanup_market",
        exec_price=exec_price,
        exec_time_ns=horizon_end_ns,
        future_mid=future_mid,
        future_mid_source=future_mid_source,
        entry_mid=entry_mid,
        side_sign=side_sign,
        observed_event=observed_event,
        observed_fill_time_ns=passive_fill_time_ns,
        horizon_end_ns=horizon_end_ns,
        num_revals=num_revals,
        trigger_step=trigger_step,
        trigger_time_ns=trigger_time_ns,
        is_cleanup=True,
    )


def _build_trade_row(
    artifact_name: str,
    order_frame: pd.DataFrame,
    strategy_name: str,
    decision_source: str,
    utility: float,
    favorable_prob: float,
    toxic_prob: float,
    exec_mode: str,
    exec_price: float,
    exec_time_ns: int,
    future_mid: float,
    future_mid_source: str,
    entry_mid: float,
    side_sign: float,
    observed_event: int,
    observed_fill_time_ns: int | None,
    horizon_end_ns: int,
    num_revals: int,
    trigger_step: int,
    trigger_time_ns: int,
    is_cleanup: bool,
) -> dict[str, Any]:
    first_row = order_frame.iloc[0]
    implementation_shortfall = side_sign * (exec_price - entry_mid)
    realized_pnl = side_sign * (future_mid - exec_price)
    return {
        "artifact_name": artifact_name,
        "strategy": strategy_name,
        "decision_source": decision_source,
        "order_id": first_row["order_id"],
        "entry_time": int(first_row["entry_time"]),
        "entry_date": pd.Timestamp(first_row["entry_date"]).date().isoformat(),
        "side": str(first_row["side"]),
        "exec_mode": exec_mode,
        "exec_time_ns": int(exec_time_ns),
        "exec_time_s_from_entry": float((exec_time_ns - int(first_row["entry_time"])) / 1e9),
        "exec_price": float(exec_price),
        "entry_bid": float(first_row["best_bid_at_entry"]),
        "entry_ask": float(first_row["best_ask_at_entry"]),
        "entry_mid": float(entry_mid),
        "future_mid": float(future_mid),
        "future_mid_source": future_mid_source,
        "observed_event_type": int(observed_event),
        "observed_fill_time_s": (
            float((observed_fill_time_ns - int(first_row["entry_time"])) / 1e9)
            if observed_fill_time_ns is not None
            else np.nan
        ),
        "observed_fill_within_horizon": bool(
            observed_fill_time_ns is not None
            and observed_fill_time_ns <= horizon_end_ns
        ),
        "observed_favorable_fill": observed_event == 1 and observed_fill_time_ns is not None,
        "observed_toxic_fill": observed_event == 2 and observed_fill_time_ns is not None,
        "num_revals": int(num_revals),
        "trigger_eval_step": int(trigger_step),
        "trigger_time_ns": int(trigger_time_ns),
        "favorable_prob": float(favorable_prob),
        "toxic_prob": float(toxic_prob),
        "utility": float(utility),
        "implementation_shortfall": float(implementation_shortfall),
        "realized_pnl": float(realized_pnl),
        "is_cleanup": bool(is_cleanup),
    }


def summarize_dynamic_trades(trades: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for strategy_name, group in trades.groupby("strategy", sort=False):
        rows.append(
            {
                "artifact_name": group["artifact_name"].iloc[0],
                "strategy": strategy_name,
                "num_trades": int(len(group)),
                "mean_implementation_shortfall": safe_mean(group["implementation_shortfall"]),
                "mean_realized_pnl": safe_mean(group["realized_pnl"]),
                "std_implementation_shortfall": float(group["implementation_shortfall"].std(ddof=0)),
                "std_realized_pnl": float(group["realized_pnl"].std(ddof=0)),
                "passive_fill_rate": safe_rate(float((group["exec_mode"] == "limit_fill").sum()), float(len(group))),
                "market_entry_rate": safe_rate(
                    float(group["exec_mode"].isin(["market_entry", "market_after_cancel"]).sum()),
                    float(len(group)),
                ),
                "cleanup_rate": safe_rate(float((group["exec_mode"] == "cleanup_market").sum()), float(len(group))),
                "toxic_fill_rate_overall": safe_rate(
                    float(((group["exec_mode"] == "limit_fill") & group["observed_toxic_fill"]).sum()),
                    float(len(group)),
                ),
                "avg_num_revals": safe_mean(group["num_revals"]),
                "avg_favorable_prob": safe_mean(group["favorable_prob"]),
                "avg_toxic_prob": safe_mean(group["toxic_prob"]),
                "avg_utility": safe_mean(group["utility"]),
            }
        )
    return pd.DataFrame(rows).sort_values("strategy").reset_index(drop=True)


def summarize_dynamic_daily(trades: pd.DataFrame) -> pd.DataFrame:
    return (
        trades.groupby(["artifact_name", "strategy", "entry_date"], sort=True)
        .agg(
            num_trades=("strategy", "size"),
            mean_implementation_shortfall=("implementation_shortfall", "mean"),
            mean_realized_pnl=("realized_pnl", "mean"),
            passive_fill_rate=("exec_mode", lambda x: float((x == "limit_fill").mean())),
            cleanup_rate=("exec_mode", lambda x: float((x == "cleanup_market").mean())),
            avg_num_revals=("num_revals", "mean"),
        )
        .reset_index()
    )


def _select_observation_rows(order_frame: pd.DataFrame, reeval_interval_s: float | None) -> list[pd.Series]:
    rows: list[pd.Series] = []
    last_eval_ns: int | None = None
    for _, row in order_frame.iterrows():
        current_ns = int(row["decision_time_ns"])
        if last_eval_ns is None:
            rows.append(row)
            last_eval_ns = current_ns
            continue
        if reeval_interval_s is None:
            rows.append(row)
            last_eval_ns = current_ns
            continue
        if (current_ns - last_eval_ns) / 1e9 >= reeval_interval_s - 1e-12:
            rows.append(row)
            last_eval_ns = current_ns
    return rows


def _row_at_or_before(order_frame: pd.DataFrame, target_time_ns: int) -> pd.Series:
    eligible = order_frame.loc[order_frame["decision_time_ns"] <= target_time_ns]
    if len(eligible) == 0:
        return order_frame.iloc[0]
    return eligible.iloc[-1]


def _market_price_from_row(row: pd.Series, side_sign: float) -> float:
    return float(row["best_ask_now"]) if side_sign > 0 else float(row["best_bid_now"])
