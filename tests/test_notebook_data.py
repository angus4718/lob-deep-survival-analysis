import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is in sys.path for src imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.notebook_data import (
    apply_dynamic_normalizer,
    DynamicSampleDataset,
    build_dynamic_samples,
    build_dynamic_samples_manifest,
    build_order_batch_indices,
    fit_dynamic_normalizer_from_manifest,
    group_indices_by_order,
    materialize_dynamic_samples_from_manifest,
    normalize_dynamic_sequences,
    select_manifest_indices_by_source_rows,
    select_manifest_indices_by_order_ids,
)


def _make_dynamic_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": [101, 202],
            "entry_time": [1_000_000_000, 2_000_000_000],
            "duration_s": [5.0, 3.5],
            "event_type": [1, 2],
            "side": ["B", "A"],
            "lob_sequence": [
                [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]],
                [[11.0, 1.0], [12.0, 2.0], [13.0, 3.0]],
            ],
            "toxicity_sequence": [
                [
                    [0.1, 0.2, float(np.log1p(1_000_000_000.0))],
                    [0.2, 0.3, float(np.log1p(1_000_000_000.0))],
                    [0.3, 0.4, float(np.log1p(1_000_000_000.0))],
                    [0.4, 0.5, float(np.log1p(1_000_000_000.0))],
                ],
                [
                    [0.5, 0.6, float(np.log1p(500_000_000.0))],
                    [0.6, 0.7, float(np.log1p(500_000_000.0))],
                    [0.7, 0.8, float(np.log1p(500_000_000.0))],
                ],
            ],
            "sequence_length": [4, 3],
        }
    )


def test_group_indices_by_order_maps_all_rows():
    order_ids = np.array([10, 10, 21, 21, 21, 35], dtype=np.int64)

    groups = group_indices_by_order(order_ids)

    assert set(groups.keys()) == {10, 21, 35}
    assert np.array_equal(groups[10], np.array([0, 1], dtype=np.int64))
    assert np.array_equal(groups[21], np.array([2, 3, 4], dtype=np.int64))
    assert np.array_equal(groups[35], np.array([5], dtype=np.int64))


def test_build_order_batch_indices_no_order_split_when_not_shuffled():
    order_ids = np.array([10, 10, 20, 20, 20, 30, 30, 40], dtype=np.int64)

    batches = build_order_batch_indices(order_ids, orders_per_batch=2, shuffle=False)

    assert len(batches) == 2
    assert np.array_equal(batches[0], np.array([0, 1, 2, 3, 4], dtype=np.int64))
    assert np.array_equal(batches[1], np.array([5, 6, 7], dtype=np.int64))


def test_build_order_batch_indices_covers_all_rows_once_with_shuffle():
    order_ids = np.array([1, 1, 1, 2, 2, 3, 4, 4, 5], dtype=np.int64)

    batches = build_order_batch_indices(
        order_ids,
        orders_per_batch=2,
        shuffle=True,
        seed=4718,
    )

    combined = np.concatenate(batches)
    assert np.array_equal(np.sort(combined), np.arange(order_ids.size, dtype=np.int64))

    # No order should appear in more than one batch.
    order_to_batch_count = {int(oid): 0 for oid in np.unique(order_ids)}
    for batch in batches:
        batch_orders = set(order_ids[batch].tolist())
        for oid in batch_orders:
            order_to_batch_count[int(oid)] += 1

    assert all(count == 1 for count in order_to_batch_count.values())


def test_build_order_batch_indices_invalid_batch_size_raises():
    order_ids = np.array([1, 1, 2], dtype=np.int64)

    with pytest.raises(ValueError, match="orders_per_batch must be positive"):
        build_order_batch_indices(order_ids, orders_per_batch=0)


def test_manifest_materialization_matches_eager_builder():
    df = _make_dynamic_df()
    kwargs = {
        "lookback_steps": 3,
        "lob_dim": 2,
        "tox_dim": 3,
        "duration_col": "duration_s",
        "event_col": "event_type",
        "admin_censor_time": 10.0,
        "max_samples_per_order": 5,
    }

    x_ref, y_ref, d_ref, oid_ref, et_ref, upd_ref = build_dynamic_samples(df, **kwargs)
    order_store, manifest = build_dynamic_samples_manifest(df, **kwargs)
    x, y, d, oid, et, upd = materialize_dynamic_samples_from_manifest(order_store, manifest)

    assert x.shape == x_ref.shape
    assert np.allclose(x, x_ref)
    assert np.allclose(y, y_ref)
    assert np.array_equal(d, d_ref)
    assert np.array_equal(oid, oid_ref)
    assert np.array_equal(et, et_ref)
    assert np.array_equal(upd, upd_ref)


def test_manifest_capped_sampling_spans_order_lifecycle():
    df = pd.DataFrame(
        {
            "order_id": [1],
            "entry_time": [1_000_000_000],
            "duration_s": [10.0],
            "event_type": [1],
            "side": ["B"],
            "lob_sequence": [
                [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0], [6.0, 6.0]],
            ],
            "toxicity_sequence": [
                [
                    [0.1, 0.1, float(np.log1p(100_000_000.0))],
                    [0.2, 0.2, float(np.log1p(100_000_000.0))],
                    [0.3, 0.3, float(np.log1p(100_000_000.0))],
                    [0.4, 0.4, float(np.log1p(100_000_000.0))],
                    [0.5, 0.5, float(np.log1p(100_000_000.0))],
                    [0.6, 0.6, float(np.log1p(100_000_000.0))],
                ],
            ],
            "sequence_length": [6],
        }
    )

    _, manifest = build_dynamic_samples_manifest(
        df,
        lookback_steps=3,
        lob_dim=2,
        tox_dim=3,
        duration_col="duration_s",
        event_col="event_type",
        max_samples_per_order=2,
    )

    # anchor_idx=2 and capped N=2 samples should span lifecycle => update_idx [0, 3]
    assert len(manifest) == 2
    assert np.array_equal(manifest.update_idx, np.array([0, 3], dtype=np.int32))


def test_manifest_uses_second_last_toxicity_column_for_time_delta_when_queue_present():
    def tox_row(delta_ms: float, queue_ahead: float) -> list[float]:
        return ([0.0] * 10) + [float(np.log1p(delta_ms)), float(np.log1p(queue_ahead))]

    df = pd.DataFrame(
        {
            "order_id": [42],
            "entry_time": [1_000_000_000],
            "duration_s": [4.0],
            "event_type": [1],
            "side": ["B"],
            "lob_sequence": [
                [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]],
            ],
            "toxicity_sequence": [
                [
                    tox_row(0.0, 5_000.0),
                    tox_row(1000.0, 10_000.0),
                    tox_row(1000.0, 20_000.0),
                    tox_row(1000.0, 40_000.0),
                ]
            ],
            "sequence_length": [4],
        }
    )

    _, manifest = build_dynamic_samples_manifest(
        df,
        lookback_steps=2,
        lob_dim=2,
        tox_dim=12,
        duration_col="duration_s",
        event_col="event_type",
    )

    # anchor_idx=1 => sample indices [1, 2, 3]
    assert np.array_equal(manifest.update_idx, np.array([0, 1, 2], dtype=np.int32))
    assert np.allclose(manifest.y, np.array([3.0, 2.0, 1.0], dtype=np.float32), atol=1e-5)


def test_manifest_source_row_selection_avoids_order_id_collision():
    df = pd.DataFrame(
        {
            "order_id": [7, 7, 7],
            "entry_time": [1_000_000_000, 2_000_000_000, 3_000_000_000],
            "duration_s": [3.0, 3.0, 3.0],
            "event_type": [1, 1, 2],
            "side": ["B", "B", "A"],
            "lob_sequence": [
                [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
                [[4.0, 4.0], [5.0, 5.0], [6.0, 6.0]],
                [[7.0, 7.0], [8.0, 8.0], [9.0, 9.0]],
            ],
            "toxicity_sequence": [
                [[0.1, 0.1, float(np.log1p(100_000_000.0))]] * 3,
                [[0.2, 0.2, float(np.log1p(100_000_000.0))]] * 3,
                [[0.3, 0.3, float(np.log1p(100_000_000.0))]] * 3,
            ],
            "sequence_length": [3, 3, 3],
        }
    )

    _, manifest = build_dynamic_samples_manifest(
        df,
        lookback_steps=2,
        lob_dim=2,
        tox_dim=3,
        duration_col="duration_s",
        event_col="event_type",
    )

    first_two_rows = select_manifest_indices_by_source_rows(
        manifest,
        np.array([0, 1], dtype=np.int64),
    )
    last_row = select_manifest_indices_by_source_rows(
        manifest,
        np.array([2], dtype=np.int64),
    )

    assert first_two_rows.size > 0
    assert last_row.size > 0
    assert np.intersect1d(first_two_rows, last_row).size == 0


def test_source_row_selection_supports_seeded_random_cap():
    def tox_seq(n_steps: int) -> list[list[float]]:
        return [[0.1, 0.2, float(np.log1p(100_000_000.0))] for _ in range(n_steps)]

    df = pd.DataFrame(
        {
            "order_id": [1001, 1002],
            "entry_time": [1_000_000_000, 2_000_000_000],
            "duration_s": [10.0, 9.0],
            "event_type": [1, 2],
            "side": ["B", "A"],
            "lob_sequence": [
                [[float(i), float(i)] for i in range(1, 7)],
                [[float(i), float(i)] for i in range(11, 16)],
            ],
            "toxicity_sequence": [
                tox_seq(6),
                tox_seq(5),
            ],
            "sequence_length": [6, 5],
        }
    )

    _, manifest = build_dynamic_samples_manifest(
        df,
        lookback_steps=2,
        lob_dim=2,
        tox_dim=3,
        duration_col="duration_s",
        event_col="event_type",
        max_samples_per_order=None,
    )

    full_idx = select_manifest_indices_by_source_rows(
        manifest,
        np.array([0, 1], dtype=np.int64),
    )
    capped_a = select_manifest_indices_by_source_rows(
        manifest,
        np.array([0, 1], dtype=np.int64),
        max_samples_per_source_row=2,
        seed=4718,
    )
    capped_b = select_manifest_indices_by_source_rows(
        manifest,
        np.array([0, 1], dtype=np.int64),
        max_samples_per_source_row=2,
        seed=4718,
    )

    assert full_idx.size > capped_a.size
    assert np.array_equal(capped_a, capped_b)

    capped_source_rows = manifest.source_row_idx[capped_a]
    counts = pd.Series(capped_source_rows).value_counts()
    assert set(counts.index.tolist()) == {0, 1}
    assert counts.max() <= 2


def test_dynamic_sample_dataset_returns_expected_shapes_and_types():
    df = _make_dynamic_df()
    order_store, manifest = build_dynamic_samples_manifest(
        df,
        lookback_steps=3,
        lob_dim=2,
        tox_dim=3,
        duration_col="duration_s",
        event_col="event_type",
        max_samples_per_order=3,
    )
    idx = select_manifest_indices_by_order_ids(manifest, np.array([101], dtype=np.int64))
    dataset = DynamicSampleDataset(order_store, manifest, sample_indices=idx)

    x, y, d, order_id, entry_time, update_idx = dataset[0]
    assert tuple(x.shape) == (3, 7)  # lookback_steps x (lob_dim + tox_dim + 2)
    assert str(y.dtype) == "torch.float32"
    assert str(d.dtype) == "torch.int64"
    assert str(order_id.dtype) == "torch.int64"
    assert str(entry_time.dtype) == "torch.int64"
    assert str(update_idx.dtype) == "torch.int64"


def test_manifest_normalizer_matches_eager_normalizer():
    df = _make_dynamic_df()
    order_store, manifest = build_dynamic_samples_manifest(
        df,
        lookback_steps=3,
        lob_dim=2,
        tox_dim=3,
        duration_col="duration_s",
        event_col="event_type",
        max_samples_per_order=3,
    )

    all_idx = np.arange(len(manifest), dtype=np.int64)
    x_all, _, _, _, _, _ = materialize_dynamic_samples_from_manifest(order_store, manifest, all_idx)

    x_ref, _, feat_mean_ref, feat_std_ref = normalize_dynamic_sequences(x_all, [])
    feat_mean, feat_std = fit_dynamic_normalizer_from_manifest(
        order_store,
        manifest,
        all_idx,
        chunk_size=2,
    )
    x_new = apply_dynamic_normalizer(x_all, feat_mean, feat_std)

    assert np.allclose(feat_mean, feat_mean_ref, atol=1e-5)
    assert np.allclose(feat_std, feat_std_ref, atol=1e-5)
    assert np.allclose(x_new, x_ref, atol=1e-5)
