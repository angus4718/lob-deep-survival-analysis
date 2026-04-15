import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.preprocess_dynamic_deephit_dataset import (  # noqa: E402
    _prepare_runtime_artifacts,
    _resolve_manifest_output_paths,
)
from src.notebook_data import build_dynamic_samples_manifest  # noqa: E402


def _make_split_dynamic_df() -> pd.DataFrame:
    def _lob_seq(base: float) -> list[list[float]]:
        return [[base + i, base + 10 + i] for i in range(4)]

    def _tox_seq(base: float) -> list[list[float]]:
        # Last column stores log1p(time_delta_ms), matching tox_dim=3 convention.
        return [[0.1 + base, 0.2 + base, float(np.log1p(100.0))] for _ in range(4)]

    rows = []
    split_labels = ["train", "train", "train", "train", "val", "test"]
    event_codes = [1, 2, 1, 0, 2, 1]
    for idx, split_name in enumerate(split_labels):
        rows.append(
            {
                "order_id": 1000 + idx,
                "entry_time": 1_700_000_000_000_000_000 + idx,
                "duration_s": 2.0 + 0.2 * idx,
                "event_type": int(event_codes[idx]),
                "side": "B" if idx % 2 == 0 else "A",
                "lob_sequence": _lob_seq(float(idx)),
                "toxicity_sequence": _tox_seq(float(idx) * 0.01),
                "sequence_length": 4,
                "dataset_split": split_name,
            }
        )
    return pd.DataFrame(rows)


def test_prepare_runtime_artifacts_has_required_outputs():
    df = _make_split_dynamic_df()
    order_store, sample_manifest = build_dynamic_samples_manifest(
        df,
        lookback_steps=2,
        lob_dim=2,
        tox_dim=3,
        duration_col="duration_s",
        event_col="event_type",
        admin_censor_time=2.0,
        max_samples_per_order=None,
    )

    artifacts, runtime_counts = _prepare_runtime_artifacts(
        df_split=df,
        order_store=order_store,
        sample_manifest=sample_manifest,
        t_max=2.0,
        train_order_subsample_fraction=1.0,
        train_order_subsample_seed=4718,
        max_samples_per_source_row=2,
        split_cap_random_seed=4718,
        num_time_steps=5,
        normalizer_chunk_size=2,
    )

    expected_keys = {
        "train_sample_idx",
        "val_sample_idx",
        "Y_train_disc",
        "D_train_disc",
        "Y_val",
        "D_val",
        "Y_val_disc",
        "D_val_disc",
        "ORDER_KEYS_TRAIN",
        "UPDATE_IDX_TRAIN",
        "ORDER_KEYS_VAL",
        "UPDATE_IDX_VAL",
        "feat_mean",
        "feat_std",
        "time_grid",
    }
    assert set(artifacts.keys()) == expected_keys

    assert artifacts["train_sample_idx"].size > 0
    assert artifacts["val_sample_idx"].size > 0
    assert artifacts["feat_mean"].shape == artifacts["feat_std"].shape
    assert artifacts["time_grid"].shape[0] == 5
    assert runtime_counts["test_sample_count"] > 0
    assert runtime_counts["output_steps"] == 5


def test_prepare_runtime_artifacts_applies_train_order_subsampling():
    df = _make_split_dynamic_df()
    order_store, sample_manifest = build_dynamic_samples_manifest(
        df,
        lookback_steps=2,
        lob_dim=2,
        tox_dim=3,
        duration_col="duration_s",
        event_col="event_type",
        admin_censor_time=2.0,
        max_samples_per_order=None,
    )

    artifacts_full, counts_full = _prepare_runtime_artifacts(
        df_split=df,
        order_store=order_store,
        sample_manifest=sample_manifest,
        t_max=2.0,
        train_order_subsample_fraction=1.0,
        train_order_subsample_seed=4718,
        max_samples_per_source_row=2,
        split_cap_random_seed=4718,
        num_time_steps=5,
        normalizer_chunk_size=2,
    )
    artifacts_sub, counts_sub = _prepare_runtime_artifacts(
        df_split=df,
        order_store=order_store,
        sample_manifest=sample_manifest,
        t_max=2.0,
        train_order_subsample_fraction=0.5,
        train_order_subsample_seed=4718,
        max_samples_per_source_row=2,
        split_cap_random_seed=4718,
        num_time_steps=5,
        normalizer_chunk_size=2,
    )

    assert counts_full["train_order_count_full"] == 4
    assert counts_sub["train_order_count"] == 2
    assert artifacts_full["train_sample_idx"].size >= artifacts_sub["train_sample_idx"].size


def test_resolve_manifest_paths_includes_preprocessed_npz():
    output_dir = Path("/tmp")
    paths = _resolve_manifest_output_paths(output_dir=output_dir, manifest_prefix="foo")

    assert len(paths) == 5
    assert paths[-1].name == "foo_dynamic_preprocessed.npz"
