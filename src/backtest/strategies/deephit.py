"""Dynamic DeepHit model strategy adapter."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import pickle
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.backtest.strategies.base import BaseStrategy, DecisionLogic
from src.backtest.types import DecisionAction, MarketSnapshot, TradingDecision
from src.models import (
    DeepHitMambaCompeting,
    DeepHitRNNCompeting,
    DeepHitRNNTransformerCompeting,
    DeepHitTransformerCompeting,
)


class DeepHitPredictionCache:
    """In-memory CIF/PMF cache for repeated DeepHit threshold sweeps."""

    FORMAT_VERSION = 1

    def __init__(self) -> None:
        self._store: dict[tuple[Any, ...], dict[str, np.ndarray]] = {}
        self.hits = 0
        self.misses = 0

    def get(self, key: tuple[Any, ...]) -> dict[str, np.ndarray] | None:
        value = self._store.get(key)
        if value is None:
            self.misses += 1
            return None
        self.hits += 1
        return value

    def set(self, key: tuple[Any, ...], value: dict[str, np.ndarray]) -> None:
        self._store[key] = value

    def clear(self) -> None:
        self._store.clear()
        self.hits = 0
        self.misses = 0

    def save(self, path: str | Path) -> None:
        """Persist cache contents to a local pickle file."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": self.FORMAT_VERSION,
            "store": self._store,
            "hits": int(self.hits),
            "misses": int(self.misses),
        }
        with out_path.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: str | Path, *, merge: bool = False) -> None:
        """Load cache contents from a local pickle file."""
        in_path = Path(path)
        with in_path.open("rb") as f:
            payload = pickle.load(f)
        version = int(payload.get("format_version", -1))
        if version != self.FORMAT_VERSION:
            raise ValueError(
                f"Unsupported DeepHitPredictionCache format_version={version}; "
                f"expected {self.FORMAT_VERSION}."
            )
        store = payload.get("store", {})
        if not isinstance(store, dict):
            raise ValueError("Invalid DeepHitPredictionCache file: 'store' is not a dict.")
        if merge:
            self._store.update(store)
            self.hits += int(payload.get("hits", 0))
            self.misses += int(payload.get("misses", 0))
        else:
            self._store = store
            self.hits = int(payload.get("hits", 0))
            self.misses = int(payload.get("misses", 0))

    @classmethod
    def from_file(cls, path: str | Path) -> "DeepHitPredictionCache":
        cache = cls()
        cache.load(path)
        return cache

    def __len__(self) -> int:
        return len(self._store)


def build_dynamic_deephit_model(
    *,
    model_name: str,
    num_features: int,
    num_events: int = 2,
    num_time_steps: int = 30,
    seq_len: int = 500,
    model_kwargs: dict[str, Any] | None = None,
) -> torch.nn.Module:
    """Build a dynamic DeepHit backbone matching the notebook defaults."""

    kwargs = dict(model_kwargs or {})
    name = model_name.lower()
    if name == "gru":
        defaults = {
            "hidden_size": 160,
            "num_layers": 2,
            "rnn_dropout": 0.2,
            "fc_hidden": int(160 * 1.75),
            "fc_dropout": 0.2,
        }
        defaults.update(kwargs)
        return DeepHitRNNCompeting(num_features, num_events, num_time_steps, **defaults)
    if name == "gru_transformer":
        defaults = {
            "hidden_size": 96,
            "num_layers": 2,
            "rnn_dropout": 0.2,
            "transformer_layers": 1,
            "transformer_heads": 2,
            "transformer_ff_dim": int(96 * 2.0),
            "transformer_dropout": 0.1,
            "max_seq_len": seq_len,
            "fc_hidden": int(96 * 1.75),
            "fc_dropout": 0.2,
        }
        defaults.update(kwargs)
        return DeepHitRNNTransformerCompeting(num_features, num_events, num_time_steps, **defaults)
    if name == "transformer":
        defaults = {
            "hidden_size": 96,
            "num_layers": 2,
            "num_heads": 4,
            "transformer_ff_dim": int(96 * 2.0),
            "transformer_dropout": 0.1,
            "max_seq_len": seq_len,
            "fc_hidden": int(96 * 1.75),
            "fc_dropout": 0.2,
        }
        defaults.update(kwargs)
        return DeepHitTransformerCompeting(num_features, num_events, num_time_steps, **defaults)
    if name == "mamba":
        defaults = {
            "hidden_size": 144,
            "num_mamba_layers": 1,
            "d_state": 8,
            "d_conv": 4,
            "expand": 2,
            "mamba_dropout": 0.15,
            "fc_hidden": int(144 * 1.75),
            "fc_dropout": 0.2,
        }
        defaults.update(kwargs)
        return DeepHitMambaCompeting(num_features, num_events, num_time_steps, **defaults)
    raise ValueError(f"Unknown dynamic DeepHit model_name: {model_name}")


@dataclass(frozen=True)
class DeepHitThresholdDecisionLogic(DecisionLogic):
    """Decision rule combining conditional toxic risk with DeepHit CIFs.

    DeepHit produces cause-specific CIFs. The rule uses:
    - conditional toxic probability among predicted fills as a classification-style score;
    - cumulative fill/toxic/favorable probabilities as survival-horizon gates.
    """

    max_toxic_probability: float = 0.35
    min_fill_probability: float = 0.0
    max_toxic_cif: float | None = None
    min_favorable_cif: float | None = None
    horizon_index: int = -1
    eps: float = 1e-8

    def decide(self, prediction: dict[str, Any], snapshot: MarketSnapshot) -> TradingDecision:
        cif = np.asarray(prediction["cif"], dtype=np.float32)
        requested_horizon = int(self.horizon_index)
        horizon = _bounded_horizon_index(requested_horizon, cif.shape[1])
        toxic_cif = float(cif[1, horizon])
        favorable_cif = float(cif[0, horizon])
        fill_cif = float(toxic_cif + favorable_cif)
        survival_prob = float(max(0.0, 1.0 - fill_cif))
        toxic_probability = (
            float(toxic_cif / max(fill_cif, float(self.eps)))
            if fill_cif > float(self.eps)
            else 0.0
        )

        pass_threshold = (
            toxic_probability <= float(self.max_toxic_probability)
            and fill_cif >= float(self.min_fill_probability)
        )
        if self.max_toxic_cif is not None:
            pass_threshold = pass_threshold and toxic_cif <= float(self.max_toxic_cif)
        if self.min_favorable_cif is not None:
            pass_threshold = pass_threshold and favorable_cif >= float(self.min_favorable_cif)

        if snapshot.position_open:
            action = DecisionAction.HOLD if pass_threshold else DecisionAction.CANCEL
            reason = "threshold_hold" if pass_threshold else "threshold_cancel"
        else:
            action = DecisionAction.SUBMIT if pass_threshold else DecisionAction.SKIP
            reason = "threshold_pass" if pass_threshold else "threshold_fail"
        return TradingDecision(
            action=action,
            limit_price=_as_float(snapshot.row.get("price")),
            reason=reason,
            diagnostics={
                "favorable_probability": favorable_cif,
                "toxic_probability": toxic_probability,
                "fill_probability": fill_cif,
                "favorable_cif": favorable_cif,
                "toxic_cif": toxic_cif,
                "fill_cif": fill_cif,
                "survival_probability": survival_prob,
                "max_toxic_probability": float(self.max_toxic_probability),
                "min_fill_probability": float(self.min_fill_probability),
                "max_toxic_cif": (
                    float(self.max_toxic_cif)
                    if self.max_toxic_cif is not None
                    else None
                ),
                "min_favorable_cif": (
                    float(self.min_favorable_cif)
                    if self.min_favorable_cif is not None
                    else None
                ),
                "horizon_index": horizon,
                "update_idx": int(snapshot.update_idx),
                "end_idx": int(snapshot.end_idx) if snapshot.end_idx is not None else None,
                "is_initial": bool(snapshot.is_initial),
                "position_open": bool(snapshot.position_open),
            },
        )


@dataclass(frozen=True)
class DeepHitToxicCIFDecisionLogic(DecisionLogic):
    """One-threshold rule using the toxic cumulative incidence directly.

    This keeps the DeepHit survival output in its native form: submit or hold
    when the model's cumulative toxic-event probability by the selected horizon
    is below the configured threshold.
    """

    max_toxic_cif: float = 0.2
    horizon_index: int = -1
    eps: float = 1e-8

    def decide(self, prediction: dict[str, Any], snapshot: MarketSnapshot) -> TradingDecision:
        cif = np.asarray(prediction["cif"], dtype=np.float32)
        requested_horizon = int(self.horizon_index)
        horizon = _bounded_horizon_index(requested_horizon, cif.shape[1])
        toxic_cif = float(cif[1, horizon])
        favorable_cif = float(cif[0, horizon])
        fill_cif = float(toxic_cif + favorable_cif)
        survival_prob = float(max(0.0, 1.0 - fill_cif))
        toxic_probability = (
            float(toxic_cif / max(fill_cif, float(self.eps)))
            if fill_cif > float(self.eps)
            else 0.0
        )

        pass_threshold = toxic_cif <= float(self.max_toxic_cif)
        if snapshot.position_open:
            action = DecisionAction.HOLD if pass_threshold else DecisionAction.CANCEL
            reason = "toxic_cif_hold" if pass_threshold else "toxic_cif_cancel"
        else:
            action = DecisionAction.SUBMIT if pass_threshold else DecisionAction.SKIP
            reason = "toxic_cif_pass" if pass_threshold else "toxic_cif_fail"

        return TradingDecision(
            action=action,
            limit_price=_as_float(snapshot.row.get("price")),
            reason=reason,
            diagnostics={
                "score_name": "toxic_cif",
                "score": toxic_cif,
                "favorable_probability": favorable_cif,
                "toxic_probability": toxic_probability,
                "fill_probability": fill_cif,
                "favorable_cif": favorable_cif,
                "toxic_cif": toxic_cif,
                "fill_cif": fill_cif,
                "survival_probability": survival_prob,
                "max_toxic_cif": float(self.max_toxic_cif),
                "horizon_index": horizon,
                "requested_horizon_index": requested_horizon,
                "update_idx": int(snapshot.update_idx),
                "end_idx": int(snapshot.end_idx) if snapshot.end_idx is not None else None,
                "is_initial": bool(snapshot.is_initial),
                "position_open": bool(snapshot.position_open),
            },
        )


class DeepHitStrategy(BaseStrategy):
    """Strategy composed of a trained DeepHit model and custom decision logic."""

    def __init__(
        self,
        model: torch.nn.Module,
        decision_logic: DecisionLogic,
        *,
        device: str | torch.device | None = None,
        prediction_cache: DeepHitPredictionCache | None = None,
        cache_predictions: bool = True,
        model_name: str | None = None,
        strict_cache_key: bool = False,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.model.eval()
        self.model_name = str(model_name or self.model.__class__.__name__)
        self.decision_logic = decision_logic
        self.prediction_cache = prediction_cache
        self.cache_predictions = bool(cache_predictions)
        self.strict_cache_key = bool(strict_cache_key)

    @classmethod
    def from_saved_model(
        cls,
        model_path: str | Path,
        *,
        model_name: str = "gru",
        num_features: int = 34,
        num_events: int = 2,
        num_time_steps: int = 30,
        seq_len: int = 500,
        decision_logic: DecisionLogic | None = None,
        model_kwargs: dict[str, Any] | None = None,
        device: str | torch.device | None = None,
        prediction_cache: DeepHitPredictionCache | None = None,
        cache_predictions: bool = True,
        strict_cache_key: bool = False,
    ) -> "DeepHitStrategy":
        model = build_dynamic_deephit_model(
            model_name=model_name,
            num_features=num_features,
            num_events=num_events,
            num_time_steps=num_time_steps,
            seq_len=seq_len,
            model_kwargs=model_kwargs,
        )
        payload = torch.load(model_path, map_location="cpu")
        state_dict = payload.get("model_state", payload) if isinstance(payload, dict) else payload
        model.load_state_dict(state_dict)
        return cls(
            model,
            decision_logic or DeepHitThresholdDecisionLogic(),
            device=device,
            prediction_cache=prediction_cache,
            cache_predictions=cache_predictions,
            model_name=model_name,
            strict_cache_key=strict_cache_key,
        )

    def predict(self, snapshot: MarketSnapshot) -> dict[str, Any]:
        cache_key = self._cache_key(snapshot)
        if self.cache_predictions and self.prediction_cache is not None:
            cached = self.prediction_cache.get(cache_key)
            if cached is not None:
                return cached

        x = torch.from_numpy(snapshot.features[None, :, :]).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            pmf = torch.softmax(logits.reshape(logits.size(0), -1), dim=1).reshape_as(logits)
            cif = torch.cumsum(pmf, dim=2)[0].detach().cpu().numpy()
        prediction = {"cif": cif, "pmf": pmf[0].detach().cpu().numpy()}
        if self.cache_predictions and self.prediction_cache is not None:
            self.prediction_cache.set(cache_key, prediction)
        return prediction

    def _cache_key(self, snapshot: MarketSnapshot) -> tuple[Any, ...]:
        key = (
            "deephit",
            self.model_name,
            _cache_scalar(snapshot.order_id),
            _cache_scalar(snapshot.row.get("entry_time")),
            int(snapshot.end_idx) if snapshot.end_idx is not None else None,
            int(snapshot.update_idx),
            tuple(int(v) for v in snapshot.features.shape),
        )
        if self.strict_cache_key:
            key = (*key, _feature_digest(snapshot.features))
        return key

    def decide(self, snapshot: MarketSnapshot) -> TradingDecision:
        prediction = self.predict(snapshot)
        return self.decision_logic.decide(prediction, snapshot)


def _as_float(value) -> float | None:
    try:
        if value is None or not np.isfinite(float(value)):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _feature_digest(features: np.ndarray) -> str:
    arr = np.ascontiguousarray(features)
    return hashlib.blake2b(arr.view(np.uint8), digest_size=16).hexdigest()


def _bounded_horizon_index(horizon_index: int, num_steps: int) -> int:
    if int(num_steps) <= 0:
        raise ValueError("DeepHit CIF must contain at least one horizon.")
    horizon = int(horizon_index)
    if horizon < 0:
        return max(-int(num_steps), horizon)
    return min(horizon, int(num_steps) - 1)


def _cache_scalar(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, np.generic):
        return value.item()
    return value
