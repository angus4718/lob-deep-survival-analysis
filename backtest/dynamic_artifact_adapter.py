from __future__ import annotations

import io
import pickle
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from .common import event_name_to_index
    from .dynamic_models import (
        DynamicDeepHitMambaCompeting,
        DynamicDeepHitRNNTransformerCompeting,
    )
except ImportError:
    from common import event_name_to_index
    from dynamic_models import (
        DynamicDeepHitMambaCompeting,
        DynamicDeepHitRNNTransformerCompeting,
    )


@dataclass
class DynamicArtifactMetadata:
    model_name: str
    framework: str
    num_competing_events: int
    event_names: list[str]
    output_steps: int
    time_grid: np.ndarray
    lookback_steps: int
    rolling_lookback_window: bool
    order_balanced_training: bool | None = None
    num_train_orders: int | None = None
    num_val_orders: int | None = None
    num_test_orders: int | None = None
    learning_rate: float | None = None
    best_epoch: int | None = None
    best_val_loss: float | None = None
    raw: dict[str, Any] | None = None

    @property
    def event_index(self) -> dict[str, int]:
        return event_name_to_index(self.event_names)


@dataclass
class LoadedDynamicArtifact:
    base_net_path: Path
    meta_path: Path
    metadata: DynamicArtifactMetadata
    model: torch.nn.Module
    device: torch.device
    init_kwargs: dict[str, Any]

    def predict_cif(self, x_np: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        self.model.eval()
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(x_np), batch_size):
                batch_np = x_np[start : start + batch_size].astype(np.float32, copy=False)
                batch = torch.from_numpy(batch_np).to(self.device)
                logits = self.model(batch)
                pmf = _logits_to_pmf(logits)
                cif = torch.cumsum(pmf, dim=2)
                outputs.append(cif.cpu().numpy())
        combined = np.concatenate(outputs, axis=0)
        return np.transpose(combined, (1, 2, 0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_net_path": str(self.base_net_path),
            "meta_path": str(self.meta_path),
            "device": str(self.device),
            "metadata": {
                "model_name": self.metadata.model_name,
                "framework": self.metadata.framework,
                "num_competing_events": self.metadata.num_competing_events,
                "event_names": self.metadata.event_names,
                "output_steps": self.metadata.output_steps,
                "time_grid": self.metadata.time_grid.tolist(),
                "lookback_steps": self.metadata.lookback_steps,
                "rolling_lookback_window": self.metadata.rolling_lookback_window,
                "order_balanced_training": self.metadata.order_balanced_training,
                "num_train_orders": self.metadata.num_train_orders,
                "num_val_orders": self.metadata.num_val_orders,
                "num_test_orders": self.metadata.num_test_orders,
                "learning_rate": self.metadata.learning_rate,
                "best_epoch": self.metadata.best_epoch,
                "best_val_loss": self.metadata.best_val_loss,
            },
            "init_kwargs": self.init_kwargs,
        }


def load_dynamic_artifact_metadata(meta_path: Path) -> DynamicArtifactMetadata:
    meta_path = meta_path.resolve()
    stem = meta_path.stem
    with zipfile.ZipFile(meta_path) as zf:
        payload = zf.read(f"{stem}/data.pkl")
    raw = pickle.load(io.BytesIO(payload))
    return DynamicArtifactMetadata(
        model_name=str(raw["model_name"]),
        framework=str(raw.get("framework", "dynamic_deephit")),
        num_competing_events=int(raw["num_competing_events"]),
        event_names=list(raw["event_names"]),
        output_steps=int(raw["output_steps"]),
        time_grid=np.asarray(raw["time_grid"], dtype=np.float32),
        lookback_steps=int(raw["lookback_steps"]),
        rolling_lookback_window=bool(raw.get("rolling_lookback_window", True)),
        order_balanced_training=_optional_bool(raw.get("order_balanced_training")),
        num_train_orders=_optional_int(raw.get("num_train_orders")),
        num_val_orders=_optional_int(raw.get("num_val_orders")),
        num_test_orders=_optional_int(raw.get("num_test_orders")),
        learning_rate=_optional_float(raw.get("learning_rate")),
        best_epoch=_optional_int(raw.get("best_epoch")),
        best_val_loss=_optional_float(raw.get("best_val_loss")),
        raw=raw,
    )


def load_dynamic_artifact(
    artifact_base_net_path: Path,
    artifact_meta_path: Path,
    device: str = "cpu",
) -> LoadedDynamicArtifact:
    artifact_base_net_path = artifact_base_net_path.resolve()
    artifact_meta_path = artifact_meta_path.resolve()
    metadata = load_dynamic_artifact_metadata(artifact_meta_path)
    if metadata.framework != "dynamic_deephit":
        raise ValueError(
            f"Expected dynamic_deephit framework, got {metadata.framework!r}."
        )

    state_dict = torch.load(artifact_base_net_path, map_location="cpu", weights_only=True)
    model_cls, init_kwargs = _resolve_dynamic_model(metadata, state_dict)
    model = model_cls(**init_kwargs)
    model.load_state_dict(state_dict, strict=True)

    torch_device = torch.device(device)
    model.to(torch_device)
    model.eval()

    return LoadedDynamicArtifact(
        base_net_path=artifact_base_net_path,
        meta_path=artifact_meta_path,
        metadata=metadata,
        model=model,
        device=torch_device,
        init_kwargs=init_kwargs,
    )


def _resolve_dynamic_model(
    metadata: DynamicArtifactMetadata,
    state_dict: dict[str, torch.Tensor],
) -> tuple[type[torch.nn.Module], dict[str, Any]]:
    if metadata.model_name == "gru_transformer":
        return DynamicDeepHitRNNTransformerCompeting, _infer_gru_transformer_kwargs(
            metadata,
            state_dict,
        )
    if metadata.model_name == "mamba":
        return DynamicDeepHitMambaCompeting, _infer_mamba_kwargs(metadata, state_dict)
    raise ValueError(
        f"Unsupported dynamic model_name {metadata.model_name!r}. "
        f"Known names: ['gru_transformer', 'mamba']"
    )


def _infer_gru_transformer_kwargs(
    metadata: DynamicArtifactMetadata,
    state_dict: dict[str, torch.Tensor],
) -> dict[str, Any]:
    hidden_size = int(state_dict["positional_embedding"].shape[2])
    num_layers = len([k for k in state_dict if k.startswith("rnn.weight_ih_l")])
    transformer_layers = len(
        {
            key.split(".")[2]
            for key in state_dict
            if key.startswith("transformer.layers.") and key.endswith("self_attn.in_proj_weight")
        }
    )
    return {
        "num_features": int(state_dict["rnn.weight_ih_l0"].shape[1]),
        "num_events": metadata.num_competing_events,
        "num_time_steps": metadata.output_steps,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "rnn_dropout": 0.2,
        "transformer_layers": transformer_layers,
        "transformer_heads": 4,
        "transformer_ff_dim": int(state_dict["transformer.layers.0.linear1.weight"].shape[0]),
        "transformer_dropout": 0.1,
        "max_seq_len": int(state_dict["positional_embedding"].shape[1]),
        "fc_hidden": int(state_dict["cause_specific_heads.0.0.weight"].shape[0]),
        "fc_dropout": 0.2,
    }


def _infer_mamba_kwargs(
    metadata: DynamicArtifactMetadata,
    state_dict: dict[str, torch.Tensor],
) -> dict[str, Any]:
    hidden_size = int(state_dict["input_proj.weight"].shape[0])
    num_layers = len(
        {
            key.split(".")[1]
            for key in state_dict
            if key.startswith("ssm_layers.") and key.endswith(".norm.weight")
        }
    )
    conv_shape = state_dict["ssm_layers.0.block.conv1d.weight"].shape
    expand = int(conv_shape[0] // hidden_size)
    return {
        "num_features": int(state_dict["input_proj.weight"].shape[1]),
        "num_events": metadata.num_competing_events,
        "num_time_steps": metadata.output_steps,
        "hidden_size": hidden_size,
        "num_mamba_layers": num_layers,
        "d_state": int(state_dict["ssm_layers.0.block.A_log"].shape[1]),
        "d_conv": int(conv_shape[2]),
        "expand": expand,
        "mamba_dropout": 0.15,
        "fc_hidden": int(state_dict["cause_heads.0.0.weight"].shape[0]),
        "fc_dropout": 0.2,
    }


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _logits_to_pmf(logits: torch.Tensor) -> torch.Tensor:
    batch_size, num_events, num_time_steps = logits.shape
    flat = logits.reshape(batch_size, num_events * num_time_steps)
    pad = torch.zeros((batch_size, 1), dtype=logits.dtype, device=logits.device)
    flat_padded = torch.cat([flat, pad], dim=1)
    pmf = torch.softmax(flat_padded, dim=1)[:, :-1]
    return pmf.reshape(batch_size, num_events, num_time_steps)
