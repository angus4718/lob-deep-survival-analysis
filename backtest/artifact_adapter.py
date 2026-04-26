from __future__ import annotations

import io
import pickle
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from .common import ensure_project_root, event_name_to_index
except ImportError:
    from common import ensure_project_root, event_name_to_index


STANDARDIZED_MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "gru": {
        "hidden_size": 160,
        "num_layers": 2,
        "rnn_dropout": 0.2,
        "fc_hidden": 256,
        "fc_dropout": 0.2,
    },
    "gru_transformer": {
        "hidden_size": 96,
        "num_layers": 2,
        "rnn_dropout": 0.2,
        "transformer_layers": 2,
        "transformer_heads": 4,
        "transformer_ff_dim": 192,
        "transformer_dropout": 0.1,
        "fc_hidden": 128,
        "fc_dropout": 0.2,
    },
    "transformer": {
        "hidden_size": 96,
        "num_layers": 2,
        "num_heads": 4,
        "transformer_ff_dim": 320,
        "transformer_dropout": 0.1,
        "fc_hidden": 112,
        "fc_dropout": 0.2,
    },
    "mamba": {
        "hidden_size": 144,
        "num_mamba_layers": 2,
        "d_state": 16,
        "d_conv": 4,
        "expand": 2,
        "mamba_dropout": 0.15,
        "fc_dropout": 0.2,
    },
}


@dataclass
class ArtifactMetadata:
    model_name: str
    num_competing_events: int
    event_names: list[str]
    output_steps: int
    time_grid: np.ndarray
    lookback_steps: int
    learning_rate: float | None = None
    best_epoch: int | None = None
    best_val_loss: float | None = None
    raw: dict[str, Any] | None = None

    @property
    def event_index(self) -> dict[str, int]:
        return event_name_to_index(self.event_names)


@dataclass
class LoadedArtifact:
    base_net_path: Path
    meta_path: Path
    project_root: Path
    metadata: ArtifactMetadata
    model: torch.nn.Module
    device: torch.device
    init_kwargs: dict[str, Any]

    def predict_cif(self, x_np: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        """Return CIF with shape ``(num_events, num_time_steps, num_samples)``."""
        self.model.eval()
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(x_np), batch_size):
                batch_np = x_np[start : start + batch_size].astype(np.float32, copy=False)
                batch = torch.from_numpy(batch_np).to(self.device)
                logits = self.model(batch)  # (B, K, T)
                pmf = _logits_to_pmf(logits)
                cif = torch.cumsum(pmf, dim=2)  # (B, K, T)
                outputs.append(cif.cpu().numpy())
        combined = np.concatenate(outputs, axis=0)
        return np.transpose(combined, (1, 2, 0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_net_path": str(self.base_net_path),
            "meta_path": str(self.meta_path),
            "project_root": str(self.project_root),
            "device": str(self.device),
            "metadata": {
                "model_name": self.metadata.model_name,
                "num_competing_events": self.metadata.num_competing_events,
                "event_names": self.metadata.event_names,
                "output_steps": self.metadata.output_steps,
                "time_grid": self.metadata.time_grid.tolist(),
                "lookback_steps": self.metadata.lookback_steps,
                "learning_rate": self.metadata.learning_rate,
                "best_epoch": self.metadata.best_epoch,
                "best_val_loss": self.metadata.best_val_loss,
            },
            "init_kwargs": self.init_kwargs,
        }


def load_artifact(
    artifact_base_net_path: Path,
    artifact_meta_path: Path,
    project_root: Path,
    feature_dim: int,
    device: str = "cpu",
) -> LoadedArtifact:
    project_root = ensure_project_root(project_root)
    artifact_base_net_path = artifact_base_net_path.resolve()
    artifact_meta_path = artifact_meta_path.resolve()

    meta = _load_metadata(artifact_meta_path)
    model_cls = _resolve_model_class(project_root, meta.model_name)

    init_kwargs = _resolve_model_init_kwargs(meta, feature_dim)
    model = model_cls(**init_kwargs)

    state_dict = torch.load(artifact_base_net_path, map_location="cpu")
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)

    torch_device = torch.device(device)
    model.to(torch_device)
    model.eval()

    return LoadedArtifact(
        base_net_path=artifact_base_net_path,
        meta_path=artifact_meta_path,
        project_root=project_root,
        metadata=meta,
        model=model,
        device=torch_device,
        init_kwargs=init_kwargs,
    )


def load_artifact_metadata(meta_path: Path) -> ArtifactMetadata:
    return _load_metadata(meta_path.resolve())


def _load_metadata(meta_path: Path) -> ArtifactMetadata:
    stem = meta_path.stem
    with zipfile.ZipFile(meta_path) as zf:
        payload = zf.read(f"{stem}/data.pkl")
    raw = pickle.load(io.BytesIO(payload))

    time_grid = np.asarray(raw["time_grid"], dtype=np.float32)
    return ArtifactMetadata(
        model_name=str(raw["model_name"]),
        num_competing_events=int(raw["num_competing_events"]),
        event_names=list(raw["event_names"]),
        output_steps=int(raw["output_steps"]),
        time_grid=time_grid,
        lookback_steps=int(raw["lookback_steps"]),
        learning_rate=_optional_float(raw.get("learning_rate")),
        best_epoch=_optional_int(raw.get("best_epoch")),
        best_val_loss=_optional_float(raw.get("best_val_loss")),
        raw=raw,
    )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _resolve_model_class(project_root: Path, model_name: str):
    ensure_project_root(project_root)
    from src.models import (
        DeepHitMambaCompeting,
        DeepHitRNNCompeting,
        DeepHitRNNTransformerCompeting,
        DeepHitTransformerCompeting,
    )

    registry = {
        "gru": DeepHitRNNCompeting,
        "gru_transformer": DeepHitRNNTransformerCompeting,
        "transformer": DeepHitTransformerCompeting,
        "mamba": DeepHitMambaCompeting,
    }
    if model_name not in registry:
        raise ValueError(
            f"Unsupported model_name {model_name!r}. "
            f"Known names: {sorted(registry)}"
        )
    return registry[model_name]


def _resolve_model_init_kwargs(meta: ArtifactMetadata, feature_dim: int) -> dict[str, Any]:
    if meta.model_name not in STANDARDIZED_MODEL_CONFIGS:
        raise ValueError(
            f"No standardized config known for model_name {meta.model_name!r}."
        )

    kwargs = dict(STANDARDIZED_MODEL_CONFIGS[meta.model_name])
    kwargs["num_features"] = feature_dim
    kwargs["num_events"] = meta.num_competing_events
    kwargs["num_time_steps"] = meta.output_steps

    if meta.model_name in {"gru_transformer", "transformer"}:
        kwargs["max_seq_len"] = meta.lookback_steps

    return kwargs


def _logits_to_pmf(logits: torch.Tensor) -> torch.Tensor:
    batch_size, num_events, num_time_steps = logits.shape
    flat = logits.reshape(batch_size, num_events * num_time_steps)
    pad = torch.zeros((batch_size, 1), dtype=logits.dtype, device=logits.device)
    flat_padded = torch.cat([flat, pad], dim=1)
    pmf = torch.softmax(flat_padded, dim=1)[:, :-1]
    return pmf.reshape(batch_size, num_events, num_time_steps)
