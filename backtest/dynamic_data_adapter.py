from __future__ import annotations

import json
import pickle
import sys
from dataclasses import dataclass
from importlib import import_module
from io import BytesIO
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

try:
    from .common import (
        EVENT_CENSORED,
        detect_post_trade_windows_ms,
        ns_to_session_date,
        safe_rate,
    )
except ImportError:
    from common import EVENT_CENSORED, detect_post_trade_windows_ms, ns_to_session_date, safe_rate


LOB_FEATURE_DIM = 20
TOXICITY_RAW_DIM = 12


@dataclass
class DynamicDataPaths:
    dataset_prefix: Path
    split_parquet_path: Path
    train_parquet_path: Path
    val_parquet_path: Path
    test_parquet_path: Path
    preprocessed_npz_path: Path
    sample_manifest_path: Path
    manifest_parquet_path: Path
    manifest_meta_path: Path
    order_store_path: Path


@dataclass
class DynamicDatasetBundle:
    dataset_prefix: Path
    eval_frame: pd.DataFrame
    x_eval: np.ndarray
    y_eval: np.ndarray
    d_eval: np.ndarray
    seq_len: np.ndarray
    split_name: str
    time_grid: np.ndarray | None
    training_priors: dict[str, float]
    available_post_trade_windows_ms: list[int]
    paths: DynamicDataPaths
    feature_source: str = "unknown"
    manifest_meta: dict[str, Any] | None = None
    sample_manifest: pd.DataFrame | None = None
    order_store: Any | None = None
    schema_summary: dict[str, Any] | None = None

    @property
    def dataset_path(self) -> Path:
        return self.paths.split_parquet_path

    @property
    def feature_dim(self) -> int:
        return int(self.x_eval.shape[2])


def load_dynamic_dataset(
    dataset_prefix: Path,
    lookback_steps: int,
    split_name: str = "test",
    time_grid: np.ndarray | None = None,
    project_root: Path | None = None,
    verbose: bool = True,
) -> DynamicDatasetBundle:
    split_name = split_name.lower()
    if split_name not in {"train", "val", "test"}:
        raise ValueError("Dynamic dataset split must be one of train/val/test.")

    paths = resolve_dynamic_paths(dataset_prefix, split_name=split_name)
    _ensure_project_import_path(project_root=project_root, dataset_prefix=paths.dataset_prefix)
    if verbose:
        _print_path_summary(paths, split_name, project_root)

    eval_frame = _load_split_frame(paths.split_parquet_path)
    sample_manifest = _load_optional_manifest(paths)
    order_store = _load_optional_pickle(paths.order_store_path)
    manifest_meta = _load_optional_json(paths.manifest_meta_path)

    if verbose:
        _print_frame_summary("eval_split_raw", eval_frame)
        if sample_manifest is not None:
            _print_frame_summary("sample_manifest", sample_manifest)
        else:
            print("[dynamic_data] sample_manifest: not available", flush=True)
        if isinstance(order_store, pd.DataFrame):
            _print_frame_summary("order_store", order_store)
        elif order_store is not None:
            print(
                f"[dynamic_data] order_store: loaded object type={type(order_store).__name__}",
                flush=True,
            )
        else:
            print("[dynamic_data] order_store: not available", flush=True)
        if manifest_meta is not None:
            print(
                f"[dynamic_data] manifest_meta keys={list(manifest_meta.keys())[:20]}",
                flush=True,
            )
        else:
            print("[dynamic_data] manifest_meta: not available", flush=True)

    if sample_manifest is not None:
        eval_frame = _merge_sidecar_frame(eval_frame, sample_manifest)
    if order_store is not None:
        order_store_frame = _coerce_order_store_frame(order_store)
        if order_store_frame is not None:
            eval_frame = _merge_sidecar_frame(eval_frame, order_store_frame, on="order_id")
            if verbose:
                _print_frame_summary("order_store_frame", order_store_frame)
        elif verbose:
            print(
                "[dynamic_data] order_store merge skipped: could not coerce sidecar "
                "object into a tabular frame",
                flush=True,
            )

    eval_frame = _standardize_dynamic_frame(eval_frame)
    if verbose:
        _print_frame_summary("eval_split_standardized", eval_frame)

    train_frame = _load_split_frame(paths.train_parquet_path)
    train_frame = _standardize_dynamic_frame(train_frame)
    if verbose:
        _print_frame_summary("train_split_standardized", train_frame)

    x_eval, seq_len, feature_source = _load_or_build_eval_features(
        paths=paths,
        split_name=split_name,
        eval_frame=eval_frame,
        train_frame=train_frame,
        lookback_steps=lookback_steps,
        sample_manifest=sample_manifest,
        verbose=verbose,
    )

    if len(eval_frame) != len(x_eval):
        raise ValueError(
            "Dynamic feature/sample count mismatch: "
            f"eval_frame={len(eval_frame)} vs x_eval={len(x_eval)}."
        )

    y_eval = eval_frame["duration_s"].to_numpy(dtype=np.float32)
    d_eval = eval_frame["event_type_competing"].to_numpy(dtype=np.int64)

    training_priors = {
        "p_favorable_fill": safe_rate(
            float((train_frame["event_type_competing"] == 1).sum()),
            float(len(train_frame)),
        ),
        "p_toxic_fill": safe_rate(
            float((train_frame["event_type_competing"] == 2).sum()),
            float(len(train_frame)),
        ),
        "p_any_fill": safe_rate(
            float(np.isin(train_frame["event_type_competing"], [1, 2]).sum()),
            float(len(train_frame)),
        ),
    }
    available_windows = sorted(detect_post_trade_windows_ms(list(eval_frame.columns)).keys())
    schema_summary = _build_schema_summary(
        eval_frame=eval_frame,
        train_frame=train_frame,
        paths=paths,
        feature_source=feature_source,
        lookback_steps=lookback_steps,
        x_eval=x_eval,
        seq_len=seq_len,
    )

    if verbose:
        print(
            "[dynamic_data] feature_source="
            f"{feature_source} x_eval_shape={tuple(x_eval.shape)} "
            f"seq_len_stats=min={int(seq_len.min(initial=0))} "
            f"median={int(np.median(seq_len)) if len(seq_len) else 0} "
            f"max={int(seq_len.max(initial=0))}",
            flush=True,
        )
        print(
            "[dynamic_data] training_priors="
            f"fav={training_priors['p_favorable_fill']:.6f} "
            f"tox={training_priors['p_toxic_fill']:.6f} "
            f"any_fill={training_priors['p_any_fill']:.6f}",
            flush=True,
        )

    return DynamicDatasetBundle(
        dataset_prefix=paths.dataset_prefix,
        eval_frame=eval_frame.reset_index(drop=True),
        x_eval=x_eval.astype(np.float32, copy=False),
        y_eval=y_eval,
        d_eval=d_eval,
        seq_len=seq_len.astype(np.int64, copy=False),
        split_name=split_name,
        time_grid=None if time_grid is None else np.asarray(time_grid, dtype=np.float32),
        training_priors=training_priors,
        available_post_trade_windows_ms=available_windows,
        paths=paths,
        feature_source=feature_source,
        manifest_meta=manifest_meta,
        sample_manifest=sample_manifest,
        order_store=order_store,
        schema_summary=schema_summary,
    )


def resolve_dynamic_paths(dataset_prefix: Path, split_name: str) -> DynamicDataPaths:
    dataset_prefix = dataset_prefix.expanduser().resolve()

    if dataset_prefix.suffix == ".parquet" and dataset_prefix.stem.endswith(
        f"_{split_name}"
    ):
        base_prefix = dataset_prefix.with_name(dataset_prefix.stem[: -(len(split_name) + 1)])
    elif dataset_prefix.suffix:
        base_prefix = dataset_prefix.with_suffix("")
    else:
        base_prefix = dataset_prefix

    base_str = str(base_prefix)
    return DynamicDataPaths(
        dataset_prefix=base_prefix,
        split_parquet_path=Path(f"{base_str}_{split_name}.parquet"),
        train_parquet_path=Path(f"{base_str}_train.parquet"),
        val_parquet_path=Path(f"{base_str}_val.parquet"),
        test_parquet_path=Path(f"{base_str}_test.parquet"),
        preprocessed_npz_path=Path(f"{base_str}_dynamic_preprocessed.npz"),
        sample_manifest_path=Path(f"{base_str}_dynamic_sample_manifest.pkl"),
        manifest_parquet_path=Path(f"{base_str}_dynamic_manifest.parquet"),
        manifest_meta_path=Path(f"{base_str}_dynamic_manifest_meta.json"),
        order_store_path=Path(f"{base_str}_dynamic_order_store.pkl"),
    )


def _load_split_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dynamic split parquet not found: {path}")
    query = f"SELECT * FROM read_parquet('{path.as_posix()}')"
    return duckdb.query(query).df().reset_index(drop=True)


def _load_or_build_eval_features(
    paths: DynamicDataPaths,
    split_name: str,
    eval_frame: pd.DataFrame,
    train_frame: pd.DataFrame,
    lookback_steps: int,
    sample_manifest: pd.DataFrame | None,
    verbose: bool,
) -> tuple[np.ndarray, np.ndarray, str]:
    if paths.preprocessed_npz_path.exists():
        try:
            x_eval, seq_len = _load_eval_features_from_npz(
                paths.preprocessed_npz_path,
                split_name=split_name,
                eval_frame=eval_frame,
                sample_manifest=sample_manifest,
            )
            if verbose:
                print(
                    f"[dynamic_data] loaded features from npz: {paths.preprocessed_npz_path}",
                    flush=True,
                )
            return x_eval, seq_len, "dynamic_preprocessed.npz"
        except Exception as exc:
            if verbose:
                print(
                    "[dynamic_data] WARNING: failed to load normalized features from "
                    f"{paths.preprocessed_npz_path}. Falling back to parquet sequences. "
                    f"Reason: {exc}",
                    flush=True,
                )

    train_x_raw, _ = _build_dynamic_feature_tensor(train_frame, lookback_steps=lookback_steps)
    feat_mean = train_x_raw.mean(axis=(0, 1), keepdims=True)
    feat_std = train_x_raw.std(axis=(0, 1), keepdims=True) + 1e-8
    mask_col_idx = train_x_raw.shape[2] - 1
    feat_mean[..., mask_col_idx] = 0.0
    feat_std[..., mask_col_idx] = 1.0

    eval_x_raw, seq_len = _build_dynamic_feature_tensor(eval_frame, lookback_steps=lookback_steps)
    eval_x = ((eval_x_raw - feat_mean) / feat_std).astype(np.float32)
    eval_x[..., mask_col_idx] = eval_x_raw[..., mask_col_idx]
    return eval_x, seq_len, "parquet_sequence_fallback"


def _load_eval_features_from_npz(
    npz_path: Path,
    split_name: str,
    eval_frame: pd.DataFrame,
    sample_manifest: pd.DataFrame | None,
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(npz_path, allow_pickle=True) as data:
        x_arr = _resolve_npz_array(data, split_name, ["X", "x", "features"])
        len_arr = _resolve_npz_array(data, split_name, ["L", "lengths", "seq_len"], required=False)
        sample_idx_arr = _resolve_npz_array(
            data,
            split_name,
            ["sample_idx", "sample_index", "indices"],
            required=False,
        )

        if x_arr is None:
            raise ValueError(
                f"No feature array found in {npz_path}. Available keys: {list(data.files)}"
            )

        x_arr = np.asarray(x_arr)
        if sample_idx_arr is not None and len(eval_frame) != len(x_arr):
            sample_idx_arr = np.asarray(sample_idx_arr).astype(np.int64)
            if len(sample_idx_arr) == len(eval_frame):
                x_arr = x_arr[sample_idx_arr]

        if len(eval_frame) != len(x_arr) and sample_manifest is not None:
            sample_idx_col = _first_existing(
                sample_manifest.columns,
                ["sample_idx", "sample_index", "manifest_idx", "idx"],
            )
            if sample_idx_col is not None and len(sample_manifest) == len(eval_frame):
                sample_idx = sample_manifest[sample_idx_col].to_numpy(dtype=np.int64)
                if sample_idx.max(initial=-1) < len(x_arr):
                    x_arr = x_arr[sample_idx]

        if len_arr is None:
            mask = x_arr[:, :, -1]
            len_arr = mask.sum(axis=1).astype(np.int64)
        else:
            len_arr = np.asarray(len_arr).astype(np.int64)

        if len(eval_frame) != len(x_arr):
            manifest_cols = [] if sample_manifest is None else list(sample_manifest.columns)
            raise ValueError(
                "NPZ feature count does not match eval_frame rows after alignment: "
                f"eval_rows={len(eval_frame)} npz_rows={len(x_arr)} split={split_name}. "
                f"Available npz keys={list(data.files)} sample_manifest_cols={manifest_cols[:20]}"
            )

        return x_arr.astype(np.float32), len_arr


def _resolve_npz_array(
    data: np.lib.npyio.NpzFile,
    split_name: str,
    base_names: list[str],
    required: bool = True,
) -> np.ndarray | None:
    candidates: list[str] = []
    split_aliases = [split_name, split_name.upper(), split_name.capitalize()]
    for base in base_names:
        candidates.extend(
            [
                f"{base}_{split_name}",
                f"{base}_{split_name.lower()}",
                f"{base}_{split_name.upper()}",
                f"{base}{split_name.capitalize()}",
                base,
                base.lower(),
                base.upper(),
            ]
        )
    for key in candidates:
        if key in data:
            return data[key]
    if required:
        return None
    return None


def _load_optional_manifest(paths: DynamicDataPaths) -> pd.DataFrame | None:
    if paths.sample_manifest_path.exists():
        obj = _load_optional_pickle(paths.sample_manifest_path)
        return _coerce_manifest_frame(obj)
    if paths.manifest_parquet_path.exists():
        return _load_split_frame(paths.manifest_parquet_path)
    return None


def _load_optional_pickle(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        payload = path.read_bytes()
        return _SafeFallbackUnpickler(BytesIO(payload)).load()
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Failed to unpickle {path} because module {exc.name!r} was not importable. "
            "This usually means the project repo root containing the `src` package is not on "
            "sys.path. Re-run with the correct --project-root."
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to unpickle sidecar file {path}: {exc}") from exc


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _coerce_manifest_frame(obj: Any) -> pd.DataFrame | None:
    if obj is None:
        return None
    obj = _unwrap_sidecar_object(obj)
    if isinstance(obj, pd.DataFrame):
        return obj.reset_index(drop=True)
    if isinstance(obj, dict):
        if all(np.ndim(v) <= 1 for v in obj.values()):
            return pd.DataFrame(obj).reset_index(drop=True)
        if "data" in obj and isinstance(obj["data"], pd.DataFrame):
            return obj["data"].reset_index(drop=True)
        for key in ["data", "rows", "samples", "records", "items", "manifest"]:
            value = obj.get(key)
            frame = _coerce_manifest_frame(value)
            if frame is not None:
                return frame
    if isinstance(obj, list):
        return pd.DataFrame(obj).reset_index(drop=True)
    return None


def _coerce_order_store_frame(obj: Any) -> pd.DataFrame | None:
    if obj is None:
        return None
    obj = _unwrap_sidecar_object(obj)
    if isinstance(obj, pd.DataFrame):
        return obj.reset_index(drop=True)
    if isinstance(obj, dict):
        if all(isinstance(v, dict) for v in obj.values()):
            rows = []
            for key, value in obj.items():
                row = {"order_id": key}
                row.update(value)
                rows.append(row)
            return pd.DataFrame(rows).reset_index(drop=True)
        if all(_is_dataframe_compatible_leaf(v) for v in obj.values()):
            try:
                return pd.DataFrame(obj).reset_index(drop=True)
            except Exception:
                return None
        for key in ["data", "rows", "samples", "records", "items", "orders", "order_store"]:
            value = obj.get(key)
            frame = _coerce_order_store_frame(value)
            if frame is not None:
                return frame
    if isinstance(obj, list):
        try:
            return pd.DataFrame(obj).reset_index(drop=True)
        except Exception:
            return None
    return None


def _merge_sidecar_frame(
    base: pd.DataFrame,
    sidecar: pd.DataFrame,
    on: str | None = None,
) -> pd.DataFrame:
    if sidecar is None or sidecar.empty:
        return base

    merged = base.copy()
    if on is not None and on in base.columns and on in sidecar.columns:
        add_cols = [c for c in sidecar.columns if c not in base.columns or c == on]
        return merged.merge(sidecar[add_cols], on=on, how="left", suffixes=("", "_sidecar"))

    if len(base) != len(sidecar):
        return base

    for column in sidecar.columns:
        if column not in merged.columns:
            merged[column] = sidecar[column].to_numpy()
    return merged


def _standardize_dynamic_frame(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    if "event_type_competing" not in df.columns:
        source_col = _required_first_existing(df.columns, ["event_type_competing", "event_type"])
        df["event_type_competing"] = df[source_col].astype(np.int64)
    if "duration_s" not in df.columns:
        raise ValueError("Dynamic dataset frame is missing duration_s.")

    entry_time_col = _required_first_existing(
        df.columns,
        ["entry_time", "entry_time_ns", "entry_timestamp_ns"],
    )
    df["entry_time"] = df[entry_time_col].astype(np.int64)

    side_col = _required_first_existing(df.columns, ["side", "order_side"])
    df["side"] = df[side_col].astype(str)

    order_id_col = _first_existing(
        df.columns,
        ["order_id", "virtual_order_id", "sim_order_id", "sample_order_id"],
    )
    if order_id_col is None:
        df["order_id"] = np.arange(len(df), dtype=np.int64)
    else:
        df["order_id"] = df[order_id_col]

    eval_step_col = _first_existing(
        df.columns,
        ["eval_step", "step_idx", "sample_step", "window_step"],
    )
    if eval_step_col is None:
        df["eval_step"] = df.groupby("order_id").cumcount().astype(np.int64)
    else:
        df["eval_step"] = df[eval_step_col].astype(np.int64)

    decision_time_col = _first_existing(
        df.columns,
        [
            "decision_time_ns",
            "current_time_ns",
            "sample_time_ns",
            "eval_time_ns",
            "snapshot_time_ns",
            "timestamp_ns",
            "state_time_ns",
        ],
    )
    if decision_time_col is not None:
        df["decision_time_ns"] = df[decision_time_col].astype(np.int64)
    else:
        elapsed_col = _first_existing(df.columns, ["elapsed_s", "elapsed_time_s", "time_since_entry_s"])
        if elapsed_col is not None:
            df["decision_time_ns"] = df["entry_time"] + (df[elapsed_col].astype(float) * 1e9).round().astype(np.int64)
        else:
            df["decision_time_ns"] = df["entry_time"]

    elapsed_col = _first_existing(df.columns, ["elapsed_s", "elapsed_time_s", "time_since_entry_s"])
    if elapsed_col is not None:
        df["elapsed_s_from_entry"] = df[elapsed_col].astype(float)
    else:
        df["elapsed_s_from_entry"] = (df["decision_time_ns"] - df["entry_time"]) / 1e9

    bid_now_col = _required_first_existing(
        df.columns,
        ["best_bid_now", "best_bid", "current_best_bid", "best_bid_at_entry"],
    )
    ask_now_col = _required_first_existing(
        df.columns,
        ["best_ask_now", "best_ask", "current_best_ask", "best_ask_at_entry"],
    )
    df["best_bid_now"] = df[bid_now_col].astype(float)
    df["best_ask_now"] = df[ask_now_col].astype(float)

    entry_bid_col = _first_existing(df.columns, ["best_bid_at_entry", "entry_best_bid"])
    entry_ask_col = _first_existing(df.columns, ["best_ask_at_entry", "entry_best_ask"])
    df["best_bid_at_entry"] = (
        df[entry_bid_col].astype(float) if entry_bid_col is not None else df["best_bid_now"].astype(float)
    )
    df["best_ask_at_entry"] = (
        df[entry_ask_col].astype(float) if entry_ask_col is not None else df["best_ask_now"].astype(float)
    )

    if "price" not in df.columns:
        side_sign = df["side"].astype(str).str.upper().map({"B": 1.0, "A": -1.0}).fillna(1.0)
        df["price"] = np.where(side_sign > 0, df["best_bid_now"], df["best_ask_now"])

    df["entry_date"] = ns_to_session_date(df["entry_time"])
    return df.sort_values(["order_id", "decision_time_ns", "eval_step"]).reset_index(drop=True)


def _build_dynamic_feature_tensor(
    df: pd.DataFrame,
    lookback_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    lob_seq_col = _required_first_existing(
        df.columns,
        [
            "lob_sequence_raw_top5",
            "entry_representation_raw_top5",
            "lob_sequence",
            "entry_representation",
        ],
    )
    tox_seq_col = _required_first_existing(
        df.columns,
        [
            "toxicity_sequence",
            "toxicity_representation",
        ],
    )

    side_raw = df["side"]
    side_vals = (
        side_raw.astype(str)
        .str.upper()
        .map({"B": 1.0, "A": 0.0})
        .fillna(0.5)
        .astype(np.float32)
        .to_numpy()
    )

    rows: list[np.ndarray] = []
    lengths: list[int] = []
    for lob_rep, tox_rep, side_val in zip(df[lob_seq_col], df[tox_seq_col], side_vals):
        lob_raw = _safe_stack_representation(
            lob_rep,
            feat_dim=LOB_FEATURE_DIM,
            source_label=lob_seq_col,
        )
        tox_raw = _safe_stack_representation(
            tox_rep,
            feat_dim=TOXICITY_RAW_DIM,
            source_label=tox_seq_col,
        )
        raw_len = min(len(lob_raw), len(tox_raw))
        lob_raw = lob_raw[-raw_len:, :] if raw_len else np.empty((0, LOB_FEATURE_DIM), dtype=np.float32)
        tox_raw = tox_raw[-raw_len:, :] if raw_len else np.empty((0, TOXICITY_RAW_DIM), dtype=np.float32)

        lob_arr, valid_len = _left_pad_or_truncate(lob_raw, lookback_steps)
        tox_arr, _ = _left_pad_or_truncate(tox_raw, lookback_steps)

        mask_col = np.zeros((lookback_steps, 1), dtype=np.float32)
        if valid_len > 0:
            mask_col[-valid_len:, 0] = 1.0
        side_col = np.full((lookback_steps, 1), side_val, dtype=np.float32) * mask_col
        full = np.concatenate([lob_arr, tox_arr, side_col, mask_col], axis=1)
        rows.append(full)
        lengths.append(valid_len)

    return np.stack(rows, axis=0), np.asarray(lengths, dtype=np.int64)


def _safe_stack_representation(rep: Any, feat_dim: int, source_label: str) -> np.ndarray:
    if rep is None:
        return np.empty((0, feat_dim), dtype=np.float32)

    rows: list[np.ndarray] = []
    try:
        iterator = list(rep)
    except TypeError as exc:
        raise ValueError(
            f"Column {source_label!r}: representation is not iterable (type={type(rep).__name__})."
        ) from exc

    if len(iterator) == 0:
        return np.empty((0, feat_dim), dtype=np.float32)

    for row_idx, row in enumerate(iterator):
        row_arr = np.asarray(row, dtype=np.float32).reshape(-1)
        if row_arr.size != feat_dim:
            raise ValueError(
                f"Column {source_label!r}: row {row_idx} has flattened size {row_arr.size}, "
                f"expected feature dim {feat_dim}."
            )
        rows.append(row_arr)

    arr = np.stack(rows, axis=0)
    if arr.ndim != 2 or arr.shape[1] != feat_dim:
        raise ValueError(
            f"Column {source_label!r}: expected feature dim {feat_dim}, got shape {arr.shape}."
        )
    return arr.astype(np.float32, copy=False)


def _left_pad_or_truncate(arr: np.ndarray, lookback_steps: int) -> tuple[np.ndarray, int]:
    if arr.shape[0] >= lookback_steps:
        trimmed = arr[-lookback_steps:, :]
        return trimmed.astype(np.float32, copy=False), int(lookback_steps)

    padded = np.zeros((lookback_steps, arr.shape[1]), dtype=np.float32)
    if arr.shape[0] > 0:
        padded[-arr.shape[0] :, :] = arr
    return padded, int(arr.shape[0])


def _first_existing(columns: Any, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _required_first_existing(columns: Any, candidates: list[str]) -> str:
    match = _first_existing(columns, candidates)
    if match is None:
        col_list = list(columns)
        preview = col_list[:40]
        raise ValueError(
            "None of the required columns were found. "
            f"Required candidates={candidates}. Available columns preview={preview}"
        )
    return match


def _print_path_summary(
    paths: DynamicDataPaths,
    split_name: str,
    project_root: Path | None,
) -> None:
    print(f"[dynamic_data] dataset_prefix={paths.dataset_prefix}", flush=True)
    print(f"[dynamic_data] requested_split={split_name}", flush=True)
    if project_root is not None:
        print(f"[dynamic_data] project_root={project_root}", flush=True)
    for label, path in [
        ("split_parquet", paths.split_parquet_path),
        ("train_parquet", paths.train_parquet_path),
        ("val_parquet", paths.val_parquet_path),
        ("test_parquet", paths.test_parquet_path),
        ("preprocessed_npz", paths.preprocessed_npz_path),
        ("sample_manifest", paths.sample_manifest_path),
        ("manifest_parquet", paths.manifest_parquet_path),
        ("manifest_meta", paths.manifest_meta_path),
        ("order_store", paths.order_store_path),
    ]:
        print(
            f"[dynamic_data] {label} exists={path.exists()} path={path}",
            flush=True,
        )


def _print_frame_summary(label: str, df: pd.DataFrame) -> None:
    print(
        f"[dynamic_data] {label}: rows={len(df)} cols={len(df.columns)} "
        f"columns_preview={list(df.columns)[:20]}",
        flush=True,
    )


def _build_schema_summary(
    eval_frame: pd.DataFrame,
    train_frame: pd.DataFrame,
    paths: DynamicDataPaths,
    feature_source: str,
    lookback_steps: int,
    x_eval: np.ndarray,
    seq_len: np.ndarray,
) -> dict[str, Any]:
    return {
        "feature_source": feature_source,
        "lookback_steps": int(lookback_steps),
        "split_rows": int(len(eval_frame)),
        "train_rows": int(len(train_frame)),
        "x_eval_shape": tuple(int(v) for v in x_eval.shape),
        "seq_len_min": int(seq_len.min(initial=0)) if len(seq_len) else 0,
        "seq_len_median": int(np.median(seq_len)) if len(seq_len) else 0,
        "seq_len_max": int(seq_len.max(initial=0)) if len(seq_len) else 0,
        "eval_columns": list(eval_frame.columns),
        "train_columns": list(train_frame.columns),
        "paths": {
            "split_parquet": str(paths.split_parquet_path),
            "train_parquet": str(paths.train_parquet_path),
            "val_parquet": str(paths.val_parquet_path),
            "test_parquet": str(paths.test_parquet_path),
            "preprocessed_npz": str(paths.preprocessed_npz_path),
            "sample_manifest": str(paths.sample_manifest_path),
            "manifest_parquet": str(paths.manifest_parquet_path),
            "manifest_meta": str(paths.manifest_meta_path),
            "order_store": str(paths.order_store_path),
        },
    }


def _ensure_project_import_path(project_root: Path | None, dataset_prefix: Path) -> None:
    candidates: list[Path] = []
    if project_root is not None:
        candidates.append(project_root.expanduser().resolve())

    dataset_root = dataset_prefix.parent
    candidates.extend(
        [
            dataset_root.parent / "lob-deep-survival-analysis-main",
            dataset_root.parent.parent / "lob-deep-survival-analysis-main",
            Path.cwd() / "lob-deep-survival-analysis-main",
        ]
    )

    seen: set[str] = set()
    for candidate in candidates:
        candidate_str = str(candidate)
        if candidate_str in seen:
            continue
        seen.add(candidate_str)
        if candidate.exists() and (candidate / "src").exists():
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            break


def _unwrap_sidecar_object(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (pd.DataFrame, dict, list, tuple)):
        return obj
    if hasattr(obj, "__dict__"):
        return dict(vars(obj))
    return obj


def _is_dataframe_compatible_leaf(value: Any) -> bool:
    if value is None:
        return True
    if np.isscalar(value):
        return True
    if isinstance(value, np.ndarray):
        return value.ndim <= 1
    if isinstance(value, (list, tuple)):
        return all(np.isscalar(item) or item is None for item in value)
    return False


class _SidecarPlaceholder:
    def __init__(self, *args, **kwargs) -> None:
        self._pickle_args = args
        self._pickle_kwargs = kwargs

    def __setstate__(self, state: Any) -> None:
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.state = state


class _SafeFallbackUnpickler(pickle.Unpickler):
    _placeholder_cache: dict[tuple[str, str], type] = {}

    def find_class(self, module: str, name: str):  # type: ignore[override]
        try:
            imported = import_module(module)
            return getattr(imported, name)
        except Exception:
            key = (module, name)
            cls = self._placeholder_cache.get(key)
            if cls is None:
                cls = type(name, (_SidecarPlaceholder,), {"__module__": module})
                self._placeholder_cache[key] = cls
            return cls
