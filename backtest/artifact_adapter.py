from __future__ import annotations

import io
import pickle
import re
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
        "fc_hidden": 192,
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
    state_dict = torch.load(artifact_base_net_path, map_location="cpu", weights_only=True)

    model_cls = _resolve_model_class(project_root, meta.model_name)
    init_kwargs = _resolve_model_init_kwargs(meta, feature_dim, state_dict)
    model = model_cls(**init_kwargs)
    model.load_state_dict(state_dict, strict=True)
    print("Loaded artifact state_dict with strict=True")

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
        candidate_names = (f"{stem}/data.pkl", "archive/data.pkl")
        payload = None
        for candidate in candidate_names:
            try:
                payload = zf.read(candidate)
                break
            except KeyError:
                continue
        if payload is None:
            raise KeyError(
                f"Could not find metadata payload in {meta_path}. "
                f"Tried: {', '.join(candidate_names)}"
            )
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


def _resolve_model_init_kwargs(
    meta: ArtifactMetadata,
    feature_dim: int,
    state_dict: dict[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    if meta.model_name not in STANDARDIZED_MODEL_CONFIGS:
        raise ValueError(
            f"No standardized config known for model_name {meta.model_name!r}."
        )

    kwargs = dict(STANDARDIZED_MODEL_CONFIGS[meta.model_name])
    kwargs.update(_extract_config_from_metadata(meta))
    if state_dict is not None:
        kwargs.update(_infer_config_from_state_dict(meta.model_name, state_dict))
    kwargs["num_features"] = feature_dim
    kwargs["num_events"] = meta.num_competing_events
    kwargs["num_time_steps"] = meta.output_steps

    if meta.model_name in {"gru_transformer", "transformer"}:
        kwargs["max_seq_len"] = meta.lookback_steps

    return kwargs


def _extract_config_from_metadata(meta: ArtifactMetadata) -> dict[str, Any]:
    if not meta.raw:
        return {}

    allowed = {
        "hidden_size",
        "num_layers",
        "rnn_dropout",
        "fc_hidden",
        "fc_dropout",
        "transformer_layers",
        "transformer_heads",
        "transformer_ff_dim",
        "transformer_dropout",
        "num_heads",
        "max_seq_len",
        "num_mamba_layers",
        "d_state",
        "d_conv",
        "expand",
        "mamba_dropout",
    }
    return {key: meta.raw[key] for key in allowed if meta.raw.get(key) is not None}


def _infer_config_from_state_dict(
    model_name: str,
    state_dict: dict[str, torch.Tensor],
) -> dict[str, Any]:
    if model_name == "gru":
        hidden_size = state_dict["rnn.weight_ih_l0"].shape[0] // 3
        return {
            "hidden_size": int(hidden_size),
            "num_layers": _count_indexed_layers(state_dict, r"^rnn\.weight_ih_l(\d+)$"),
            "fc_hidden": int(state_dict["cause_specific_heads.0.0.weight"].shape[0]),
        }

    if model_name == "transformer":
        return {
            "hidden_size": int(state_dict["input_projection.weight"].shape[0]),
            "num_layers": _count_indexed_layers(
                state_dict, r"^transformer\.layers\.(\d+)\."
            ),
            "transformer_ff_dim": int(
                state_dict["transformer.layers.0.linear1.weight"].shape[0]
            ),
            "fc_hidden": int(state_dict["cause_specific_heads.0.0.weight"].shape[0]),
            "max_seq_len": int(state_dict["positional_embedding"].shape[1]),
        }

    if model_name == "gru_transformer":
        hidden_size = state_dict["rnn.weight_ih_l0"].shape[0] // 3
        return {
            "hidden_size": int(hidden_size),
            "num_layers": _count_indexed_layers(state_dict, r"^rnn\.weight_ih_l(\d+)$"),
            "transformer_layers": _count_indexed_layers(
                state_dict, r"^transformer\.layers\.(\d+)\."
            ),
            "transformer_ff_dim": int(
                state_dict["transformer.layers.0.linear1.weight"].shape[0]
            ),
            "fc_hidden": int(state_dict["cause_specific_heads.0.0.weight"].shape[0]),
            "max_seq_len": int(state_dict["positional_embedding"].shape[1]),
        }

    if model_name == "mamba":
        inferred = {
            "hidden_size": int(state_dict["input_proj.weight"].shape[0]),
            "num_mamba_layers": _count_indexed_layers(
                state_dict, r"^ssm_layers\.(\d+)\."
            ),
            "fc_hidden": int(state_dict["cause_heads.0.0.weight"].shape[0]),
        }
        a_log = state_dict.get("ssm_layers.0.block.A_log")
        if a_log is not None:
            inferred["d_state"] = int(a_log.shape[-1])
        conv_weight = state_dict.get("ssm_layers.0.block.conv1d.weight")
        if conv_weight is not None:
            inferred["d_conv"] = int(conv_weight.shape[-1])
        in_proj = state_dict.get("ssm_layers.0.block.in_proj.weight")
        hidden_size = inferred["hidden_size"]
        if in_proj is not None and hidden_size > 0:
            inferred["expand"] = int(in_proj.shape[0] // (2 * hidden_size))
        return inferred

    return {}


def _count_indexed_layers(state_dict: dict[str, torch.Tensor], pattern: str) -> int:
    regex = re.compile(pattern)
    indices = {
        int(match.group(1))
        for key in state_dict
        if (match := regex.match(key)) is not None
    }
    if not indices:
        return 1
    return max(indices) + 1


def _logits_to_pmf(logits: torch.Tensor) -> torch.Tensor:
    batch_size, num_events, num_time_steps = logits.shape
    flat = logits.reshape(batch_size, num_events * num_time_steps)
    pad = torch.zeros((batch_size, 1), dtype=logits.dtype, device=logits.device)
    flat_padded = torch.cat([flat, pad], dim=1)
    pmf = torch.softmax(flat_padded, dim=1)[:, :-1]
    return pmf.reshape(batch_size, num_events, num_time_steps)
