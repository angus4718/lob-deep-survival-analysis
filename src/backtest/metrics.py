"""Backtest metric implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.domain.enums import EventType
from src.labeling.utils import ms_to_suffix
from src.labeling.window_selecting import MarkoutAnalyzer, StabilizationWindowSelector

from .types import DecisionAction, TradingDecision


@dataclass(frozen=True)
class _OpportunityCostResult:
    cost: float | None = None
    entry_quote: float | None = None
    reference_quote: float | None = None
    quote_side: str | None = None
    reason: str | None = None


@dataclass(frozen=True)
class _FillAdjustedToxicCostResult:
    bps: float | None = None
    raw_cost: float | None = None
    reference_mid: float | None = None
    component: str = "unfilled_zero"
    status: str = "ok"
    missing_reason: str | None = None


@dataclass(frozen=True)
class ImplementationShortfallMetric:
    """Compute implementation shortfall from realized labeled-order outcomes."""

    selected_window_ms: int = int(CONFIG.data.tox_horizon_s * 1000)
    unfilled_lob_sequence_col: str = "lob_sequence_raw_top5"
    price_unit: float = CONFIG.data.price_unit
    window_selection: dict[str, Any] | None = None
    calibrate_toxic_window: bool = True

    @classmethod
    def from_labeled_orders(
        cls,
        df: pd.DataFrame,
        *,
        unfilled_lob_sequence_col: str = "lob_sequence_raw_top5",
        price_unit: float = CONFIG.data.price_unit,
        winsorize: bool = True,
    ) -> "ImplementationShortfallMetric":
        """Create a metric calibrated with the same markout window selector as labeling."""
        selection = select_toxic_cost_window(df, winsorize=winsorize)
        if not selection.get("found"):
            raise ValueError(
                "Unable to select toxic-cost markout window for backtest metric: "
                f"{selection.get('reason', 'unknown reason')}"
            )
        return cls(
            selected_window_ms=int(selection["chosen_window_ms"]),
            unfilled_lob_sequence_col=unfilled_lob_sequence_col,
            price_unit=price_unit,
            window_selection=selection,
            calibrate_toxic_window=False,
        )

    def evaluate(self, row: pd.Series, decision: TradingDecision) -> dict[str, Any]:
        side = str(row.get("side", "")).upper()
        limit_price = _coalesce_float(decision.limit_price, row.get("price"))
        action = DecisionAction(decision.action)
        event_type = _safe_int(row.get("event_type"))
        status_reason = str(row.get("status_reason", "")).upper()
        actual_is_filled = status_reason == "FILLED" or event_type in {
            int(EventType.FAVORABLE_FILL),
            int(EventType.TOXIC_FILL),
        }
        labeled_actual_is_filled = _labeled_actual_is_filled(row, actual_is_filled)
        submitted = action != DecisionAction.SKIP
        canceled = action == DecisionAction.CANCEL
        effective_filled = bool(submitted and actual_is_filled and not canceled)
        fill_adjusted = _fill_adjusted_toxic_cost(
            row,
            side=side,
            submitted=submitted,
            canceled=canceled,
            effective_filled=effective_filled,
            labeled_actual_is_filled=labeled_actual_is_filled,
            selected_window_ms=int(self.selected_window_ms),
        )

        if effective_filled:
            fill_price = _coalesce_float(row.get("fill_price"), limit_price, row.get("price"))
            limit_price = fill_price
            if limit_price is None:
                return self._failed(
                    reason="missing_limit_price",
                    submitted=submitted,
                    filled=effective_filled,
                    actual_filled=bool(actual_is_filled),
                    canceled=canceled,
                    cost_type="toxic_cost",
                    fill_adjusted=fill_adjusted,
                )
            bid_col, ask_col = self._selected_post_trade_cols()
            reference_mid = _mid_price(
                row.get(bid_col),
                row.get(ask_col),
            )
            reference_source = f"selected_post_trade_{ms_to_suffix(self.selected_window_ms)}"
            cost_type = "toxic_cost"
            if reference_mid is None:
                return self._failed(
                    reason="missing_reference_mid",
                    submitted=submitted,
                    filled=effective_filled,
                    actual_filled=bool(actual_is_filled),
                    canceled=canceled,
                    cost_type=cost_type,
                    reference_source=reference_source,
                    limit_price=limit_price,
                    fill_adjusted=fill_adjusted,
                )

            signed_cost = _side_cost(side, limit_price, reference_mid)
            if not np.isfinite(signed_cost):
                return self._failed(
                    reason="invalid_side",
                    submitted=submitted,
                    filled=effective_filled,
                    actual_filled=bool(actual_is_filled),
                    canceled=canceled,
                    cost_type=cost_type,
                    reference_source=reference_source,
                    limit_price=limit_price,
                    reference_mid=reference_mid,
                    fill_adjusted=fill_adjusted,
                )
            basis_price = reference_mid
            reference_price = reference_mid
            fill_adjusted = _FillAdjustedToxicCostResult(
                bps=float(signed_cost / basis_price * 10000.0) if basis_price else np.nan,
                raw_cost=float(signed_cost),
                reference_mid=reference_mid,
                component="realized_toxic_cost",
                status="ok",
            )
            extra_fields: dict[str, Any] = {
                "reference_mid": _price_to_dollars(reference_mid, self.price_unit),
                "reference_mid_raw": reference_mid,
            }
        else:
            later_bid, later_ask = _best_quotes_from_raw_top5_sequence(
                row.get(self.unfilled_lob_sequence_col),
                end_idx=_safe_int(decision.diagnostics.get("reference_end_idx")),
            )
            later_bid = _coalesce_float(row.get("opportunity_reference_bid"), later_bid)
            later_ask = _coalesce_float(row.get("opportunity_reference_ask"), later_ask)
            reference_source = self.unfilled_lob_sequence_col
            cost_type = "opportunity_cost"
            opportunity = _same_side_opportunity_cost(
                side,
                entry_bid=_coalesce_float(row.get("best_bid_at_entry")),
                entry_ask=_coalesce_float(row.get("best_ask_at_entry")),
                later_bid=later_bid,
                later_ask=later_ask,
            )
            if opportunity.reason is not None:
                return self._failed(
                    reason=opportunity.reason,
                    submitted=submitted,
                    filled=effective_filled,
                    actual_filled=bool(actual_is_filled),
                    canceled=canceled,
                    cost_type=cost_type,
                    reference_source=reference_source,
                    limit_price=limit_price,
                    fill_adjusted=fill_adjusted,
                )
            signed_cost = float(opportunity.cost)
            basis_price = float(opportunity.entry_quote)
            reference_price = float(opportunity.reference_quote)
            extra_fields = {
                "opportunity_quote_side": opportunity.quote_side,
                "opportunity_entry_quote": _price_to_dollars(
                    opportunity.entry_quote, self.price_unit
                ),
                "opportunity_reference_quote": _price_to_dollars(
                    opportunity.reference_quote, self.price_unit
                ),
                "opportunity_entry_quote_raw": opportunity.entry_quote,
                "opportunity_reference_quote_raw": opportunity.reference_quote,
            }

        cost_bps = signed_cost / basis_price * 10000.0 if basis_price else np.nan
        limit_price_dollars = (
            _price_to_dollars(limit_price, self.price_unit)
            if limit_price is not None
            else None
        )
        reference_price_dollars = _price_to_dollars(reference_price, self.price_unit)
        signed_cost_dollars = _price_to_dollars(signed_cost, self.price_unit)
        out = {
            "metric_status": "ok",
            "submitted": submitted,
            "filled": effective_filled,
            "actual_filled": bool(actual_is_filled),
            "canceled": canceled,
            "cost_type": cost_type,
            "selected_window_ms": int(self.selected_window_ms),
            "price_unit": float(self.price_unit),
            "limit_price": limit_price_dollars,
            "reference_price": reference_price_dollars,
            "implementation_shortfall": float(signed_cost_dollars),
            "limit_price_raw": limit_price,
            "reference_price_raw": reference_price,
            "implementation_shortfall_raw": float(signed_cost),
            "reference_source": reference_source,
            "implementation_shortfall_bps": float(cost_bps),
        }
        out.update(extra_fields)
        out.update(_fill_adjusted_fields(fill_adjusted, self.price_unit))
        return out

    def _selected_post_trade_cols(self) -> tuple[str, str]:
        return _post_trade_cols(int(self.selected_window_ms))

    def _failed(
        self,
        *,
        reason: str,
        submitted: bool,
        cost_type: str,
        filled: bool | None = None,
        actual_filled: bool | None = None,
        canceled: bool | None = None,
        reference_source: str | None = None,
        limit_price: float | None = None,
        reference_mid: float | None = None,
        fill_adjusted: "_FillAdjustedToxicCostResult | None" = None,
    ) -> dict[str, Any]:
        out = {
            "metric_status": "failed",
            "submitted": bool(submitted),
            "cost_type": cost_type,
            "missing_reason": reason,
            "selected_window_ms": int(self.selected_window_ms),
            "price_unit": float(self.price_unit),
            "implementation_shortfall": np.nan,
            "implementation_shortfall_raw": np.nan,
            "implementation_shortfall_bps": np.nan,
        }
        if filled is not None:
            out["filled"] = bool(filled)
        if actual_filled is not None:
            out["actual_filled"] = bool(actual_filled)
        if canceled is not None:
            out["canceled"] = bool(canceled)
        if reference_source is not None:
            out["reference_source"] = reference_source
        if limit_price is not None:
            out["limit_price"] = _price_to_dollars(limit_price, self.price_unit)
            out["limit_price_raw"] = limit_price
        if reference_mid is not None:
            out["reference_mid"] = _price_to_dollars(reference_mid, self.price_unit)
            out["reference_mid_raw"] = reference_mid
        if fill_adjusted is not None:
            out.update(_fill_adjusted_fields(fill_adjusted, self.price_unit))
        return out


def select_toxic_cost_window(
    df: pd.DataFrame,
    *,
    winsorize: bool = True,
) -> dict[str, Any]:
    """Select the toxic-cost post-trade window using the labeling workflow."""
    available_windows = _available_post_trade_windows(df)
    if not available_windows:
        return {
            "found": False,
            "reason": "no configured post-trade bid/ask window columns are present",
        }

    if "event" in df.columns:
        filled = df[df["event"] == 1].copy()
    elif "status_reason" in df.columns:
        filled = df[df["status_reason"].astype(str).str.upper() == "FILLED"].copy()
    else:
        filled = df.copy()

    if filled.empty:
        return {
            "found": False,
            "reason": "no filled rows available for toxic-cost window selection",
        }

    analyzer = MarkoutAnalyzer(
        window_selector=StabilizationWindowSelector(),
        windows=available_windows,
        winsorize=winsorize,
    )
    return analyzer.analyze(filled)["selected_window"]


def _available_post_trade_windows(df: pd.DataFrame) -> list[int]:
    windows: list[int] = []
    for window_ms in CONFIG.labeling.tox_post_trade_move_windows_ms:
        suffix = ms_to_suffix(int(window_ms))
        bid_col = f"post_trade_best_bid_{suffix}"
        ask_col = f"post_trade_best_ask_{suffix}"
        if bid_col in df.columns and ask_col in df.columns:
            windows.append(int(window_ms))
    return windows


def _side_cost(side: str, limit_price: float, reference_mid: float) -> float:
    side = str(side).upper()
    if side in {"B", "BUY", "BID"}:
        return float(limit_price - reference_mid)
    if side in {"A", "ASK", "SELL", "S"}:
        return float(reference_mid - limit_price)
    return float("nan")


def _fill_adjusted_toxic_cost(
    row: pd.Series,
    *,
    side: str,
    submitted: bool,
    canceled: bool,
    effective_filled: bool,
    labeled_actual_is_filled: bool,
    selected_window_ms: int,
) -> _FillAdjustedToxicCostResult:
    if effective_filled:
        return _FillAdjustedToxicCostResult(
            bps=np.nan,
            raw_cost=np.nan,
            component="realized_toxic_cost",
            status="pending_realized_toxic_cost",
        )
    if (not submitted or canceled) and labeled_actual_is_filled:
        theoretical = _theoretical_toxic_cost(
            row,
            side=side,
            selected_window_ms=selected_window_ms,
        )
        if theoretical.status != "ok" or theoretical.bps is None:
            return _FillAdjustedToxicCostResult(
                bps=np.nan,
                raw_cost=np.nan,
                reference_mid=theoretical.reference_mid,
                component="flipped_counterfactual_toxic_cost",
                status="failed",
                missing_reason=theoretical.missing_reason or "missing_counterfactual_toxic_cost",
            )
        return _FillAdjustedToxicCostResult(
            bps=-float(theoretical.bps),
            raw_cost=-float(theoretical.raw_cost) if theoretical.raw_cost is not None else np.nan,
            reference_mid=theoretical.reference_mid,
            component="flipped_counterfactual_toxic_cost",
            status="ok",
        )
    return _FillAdjustedToxicCostResult(
        bps=0.0,
        raw_cost=0.0,
        component="unfilled_zero",
        status="ok",
    )


def _theoretical_toxic_cost(
    row: pd.Series,
    *,
    side: str,
    selected_window_ms: int,
) -> _FillAdjustedToxicCostResult:
    fill_price = _coalesce_float(
        row.get("labeled_fill_price"),
        row.get("original_fill_price"),
        row.get("fill_price"),
        row.get("price"),
    )
    if fill_price is None:
        return _FillAdjustedToxicCostResult(
            component="counterfactual_toxic_cost",
            status="failed",
            missing_reason="missing_counterfactual_fill_price",
        )
    bid_col, ask_col = _post_trade_cols(selected_window_ms)
    reference_mid = _mid_price(row.get(bid_col), row.get(ask_col))
    if reference_mid is None:
        return _FillAdjustedToxicCostResult(
            component="counterfactual_toxic_cost",
            status="failed",
            missing_reason="missing_counterfactual_reference_mid",
        )
    signed_cost = _side_cost(side, fill_price, reference_mid)
    if not np.isfinite(signed_cost):
        return _FillAdjustedToxicCostResult(
            raw_cost=signed_cost,
            reference_mid=reference_mid,
            component="counterfactual_toxic_cost",
            status="failed",
            missing_reason="invalid_counterfactual_side",
        )
    bps = signed_cost / reference_mid * 10000.0 if reference_mid else np.nan
    return _FillAdjustedToxicCostResult(
        bps=float(bps),
        raw_cost=float(signed_cost),
        reference_mid=reference_mid,
        component="counterfactual_toxic_cost",
        status="ok",
    )


def _fill_adjusted_fields(
    result: _FillAdjustedToxicCostResult,
    price_unit: float,
) -> dict[str, Any]:
    raw_cost = (
        float(result.raw_cost)
        if result.raw_cost is not None and np.isfinite(float(result.raw_cost))
        else np.nan
    )
    reference_mid = (
        float(result.reference_mid)
        if result.reference_mid is not None and np.isfinite(float(result.reference_mid))
        else np.nan
    )
    out: dict[str, Any] = {
        "fill_adjusted_toxic_cost_status": result.status,
        "fill_adjusted_toxic_cost_component": result.component,
        "fill_adjusted_toxic_cost_bps": (
            float(result.bps)
            if result.bps is not None and np.isfinite(float(result.bps))
            else np.nan
        ),
        "fill_adjusted_toxic_cost_raw": raw_cost,
        "fill_adjusted_toxic_cost": (
            _price_to_dollars(raw_cost, price_unit) if np.isfinite(raw_cost) else np.nan
        ),
        "fill_adjusted_toxic_cost_reference_mid_raw": reference_mid,
        "fill_adjusted_toxic_cost_reference_mid": (
            _price_to_dollars(reference_mid, price_unit)
            if np.isfinite(reference_mid)
            else np.nan
        ),
    }
    if result.missing_reason is not None:
        out["fill_adjusted_toxic_cost_missing_reason"] = result.missing_reason
    return out


def _labeled_actual_is_filled(row: pd.Series, fallback: bool) -> bool:
    for status_col in ("labeled_status_reason", "original_status_reason"):
        if status_col in row:
            status = str(row.get(status_col, "")).upper()
            if status == "FILLED":
                return True
            if status and status not in {"NAN", "NONE"}:
                return False
    for event_col in ("labeled_event_type", "original_event_type"):
        event_type = _safe_int(row.get(event_col))
        if event_type in {int(EventType.FAVORABLE_FILL), int(EventType.TOXIC_FILL)}:
            return True
        if event_type is not None:
            return False
    return bool(fallback)


def _post_trade_cols(window_ms: int) -> tuple[str, str]:
    suffix = ms_to_suffix(int(window_ms))
    return f"post_trade_best_bid_{suffix}", f"post_trade_best_ask_{suffix}"


def _mid_price(bid, ask) -> float | None:
    bid_f = _coalesce_float(bid)
    ask_f = _coalesce_float(ask)
    if bid_f is None or ask_f is None:
        return None
    return float((bid_f + ask_f) / 2.0)


def _price_to_dollars(value: float, price_unit: float) -> float:
    unit = float(price_unit)
    if not np.isfinite(unit) or unit == 0.0:
        raise ValueError(f"price_unit must be finite and non-zero. Got {price_unit!r}")
    return float(value) / unit


def _same_side_opportunity_cost(
    side: str,
    *,
    entry_bid: float | None,
    entry_ask: float | None,
    later_bid: float | None,
    later_ask: float | None,
) -> _OpportunityCostResult:
    if side == "B":
        if entry_ask is None:
            return _OpportunityCostResult(reason="missing_entry_ask")
        if later_ask is None:
            return _OpportunityCostResult(reason="missing_reference_ask")
        return _OpportunityCostResult(
            cost=float(later_ask - entry_ask),
            entry_quote=float(entry_ask),
            reference_quote=float(later_ask),
            quote_side="ask",
        )
    if side == "A":
        if entry_bid is None:
            return _OpportunityCostResult(reason="missing_entry_bid")
        if later_bid is None:
            return _OpportunityCostResult(reason="missing_reference_bid")
        return _OpportunityCostResult(
            cost=float(entry_bid - later_bid),
            entry_quote=float(entry_bid),
            reference_quote=float(later_bid),
            quote_side="bid",
        )
    return _OpportunityCostResult(reason="invalid_side")


def _best_quotes_from_raw_top5_sequence(
    sequence,
    end_idx: int | None = None,
) -> tuple[float | None, float | None]:
    """Extract the latest best bid and ask from a raw-top5 LOB sequence.

    ``RepresentationTransform._raw_top5`` stores each snapshot as:
    [ask_px_1, ask_vol_1, bid_px_1, bid_vol_1, ...].
    """
    if sequence is None:
        return None, None
    try:
        if pd.isna(sequence):
            return None, None
    except (TypeError, ValueError):
        pass

    try:
        rows = list(sequence)
    except TypeError:
        return None, None

    if end_idx is None:
        candidate_rows = reversed(rows)
    elif 0 <= int(end_idx) < len(rows):
        candidate_rows = [rows[int(end_idx)]]
    else:
        return None, None

    for raw_row in candidate_rows:
        row = np.asarray(raw_row, dtype=np.float64).reshape(-1)
        if row.size < 4:
            continue
        ask = _coalesce_float(row[0])
        bid = _coalesce_float(row[2])
        ask_size = _coalesce_float(row[1])
        bid_size = _coalesce_float(row[3])
        if (
            ask is not None
            and bid is not None
            and ask > 0.0
            and bid > 0.0
            and ask >= bid
            and (ask_size is None or ask_size >= 0.0)
            and (bid_size is None or bid_size >= 0.0)
        ):
            return float(bid), float(ask)
    return None, None


def _coalesce_float(*values) -> float | None:
    for value in values:
        try:
            out = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(out):
            return out
    return None


def _safe_int(value) -> int | None:
    try:
        if value is None or not np.isfinite(float(value)):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None
