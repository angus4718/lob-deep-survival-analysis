import sys
from pathlib import Path

import pytest
import torch
from torch import nn

# Ensure project root is in sys.path for src imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.notebook_losses import (
    dynamic_deephit_total_loss,
    l2_rank_order_avg,
    l3_aux_order_avg,
    logits_to_pmf,
    pmf_to_cif,
)


class DummyNet(nn.Module):
    def __init__(self, hidden_dim: int, pred_dim: int) -> None:
        super().__init__()
        self.aux_head = nn.Linear(hidden_dim, pred_dim, bias=False)
        self._cache: dict[str, torch.Tensor] = {}


def test_logits_to_pmf_and_cif_properties():
    logits = torch.tensor(
        [
            [[0.2, 0.1, -0.3], [0.4, -0.2, 0.0]],
            [[-1.0, 0.5, 0.3], [0.1, -0.4, 0.9]],
        ],
        dtype=torch.float32,
    )

    pmf = logits_to_pmf(logits)
    cif = pmf_to_cif(pmf)

    assert pmf.shape == logits.shape
    assert cif.shape == logits.shape
    assert torch.allclose(pmf.sum(dim=(1, 2)), torch.ones(2), atol=1e-6)
    assert torch.all(cif[:, :, 1:] >= cif[:, :, :-1] - 1e-6)


def test_dynamic_total_loss_components_are_finite_and_backpropagate():
    torch.manual_seed(4718)

    batch_size, num_events, num_bins = 6, 2, 5
    seq_len, feat_dim = 4, 8

    logits = torch.randn(batch_size, num_events, num_bins, requires_grad=True)
    y = torch.tensor([1, 2, 1, 3, 0, 2], dtype=torch.int64)
    d = torch.tensor([1, 1, 2, 0, 1, 2], dtype=torch.int64)
    order_ids = torch.tensor([10, 10, 11, 11, 12, 12], dtype=torch.int64)
    update_idx = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.int64)

    x_batch = torch.randn(batch_size, seq_len, feat_dim, dtype=torch.float32)
    x_batch[:, :, -1] = 1.0  # mask channel

    net = DummyNet(hidden_dim=7, pred_dim=feat_dim - 1)
    state_out = torch.randn(batch_size, seq_len, 7, dtype=torch.float32, requires_grad=True)
    net._cache = {"state_out": state_out, "mask": x_batch[:, :, -1]}

    total, components = dynamic_deephit_total_loss(
        logits,
        y,
        d,
        order_ids,
        update_idx,
        x_batch,
        net,
        alpha=0.7,
        sigma=0.1,
        beta_l3=0.2,
        aux_target_dim=6,
    )

    assert torch.isfinite(total)
    assert torch.isfinite(components["l1"])
    assert torch.isfinite(components["l2"])
    assert torch.isfinite(components["l3"])

    total.backward()
    assert logits.grad is not None
    assert net.aux_head.weight.grad is not None


def test_dynamic_total_loss_alpha_scales_l2_term():
    torch.manual_seed(123)

    batch_size, num_events, num_bins = 6, 2, 5
    seq_len, feat_dim = 4, 8

    logits = torch.randn(batch_size, num_events, num_bins, requires_grad=True)
    y = torch.tensor([1, 2, 1, 3, 0, 2], dtype=torch.int64)
    d = torch.tensor([1, 1, 2, 0, 1, 2], dtype=torch.int64)
    order_ids = torch.tensor([10, 10, 11, 11, 12, 12], dtype=torch.int64)
    update_idx = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.int64)

    x_batch = torch.randn(batch_size, seq_len, feat_dim, dtype=torch.float32)
    x_batch[:, :, -1] = 1.0

    net = DummyNet(hidden_dim=7, pred_dim=feat_dim - 1)
    state_out = torch.randn(batch_size, seq_len, 7, dtype=torch.float32, requires_grad=True)
    net._cache = {"state_out": state_out, "mask": x_batch[:, :, -1]}

    total_a0, parts_a0 = dynamic_deephit_total_loss(
        logits,
        y,
        d,
        order_ids,
        update_idx,
        x_batch,
        net,
        alpha=0.0,
        sigma=0.1,
        beta_l3=0.2,
        aux_target_dim=6,
    )
    total_a1, parts_a1 = dynamic_deephit_total_loss(
        logits,
        y,
        d,
        order_ids,
        update_idx,
        x_batch,
        net,
        alpha=1.0,
        sigma=0.1,
        beta_l3=0.2,
        aux_target_dim=6,
    )

    assert torch.allclose(parts_a0["l2"], parts_a1["l2"], atol=1e-6)
    assert torch.allclose((total_a1 - total_a0), parts_a1["l2"], atol=1e-6)


def test_dynamic_total_loss_rejects_invalid_hyperparameters():
    torch.manual_seed(99)

    batch_size, num_events, num_bins = 4, 2, 4
    seq_len, feat_dim = 3, 8

    logits = torch.randn(batch_size, num_events, num_bins)
    y = torch.tensor([1, 2, 0, 1], dtype=torch.int64)
    d = torch.tensor([1, 0, 2, 1], dtype=torch.int64)
    order_ids = torch.tensor([10, 10, 11, 11], dtype=torch.int64)
    update_idx = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
    x_batch = torch.randn(batch_size, seq_len, feat_dim, dtype=torch.float32)
    x_batch[:, :, -1] = 1.0

    net = DummyNet(hidden_dim=6, pred_dim=feat_dim - 1)
    net._cache = {
        "state_out": torch.randn(batch_size, seq_len, 6, dtype=torch.float32),
        "mask": x_batch[:, :, -1],
    }

    with pytest.raises(ValueError, match="alpha"):
        dynamic_deephit_total_loss(
            logits,
            y,
            d,
            order_ids,
            update_idx,
            x_batch,
            net,
            alpha=-0.1,
            sigma=0.1,
            beta_l3=0.1,
        )

    with pytest.raises(ValueError, match="sigma"):
        dynamic_deephit_total_loss(
            logits,
            y,
            d,
            order_ids,
            update_idx,
            x_batch,
            net,
            alpha=0.5,
            sigma=0.0,
            beta_l3=0.1,
        )

    with pytest.raises(ValueError, match="beta_l3"):
        dynamic_deephit_total_loss(
            logits,
            y,
            d,
            order_ids,
            update_idx,
            x_batch,
            net,
            alpha=0.5,
            sigma=0.1,
            beta_l3=-1.0,
        )

    with pytest.raises(ValueError, match="eps"):
        dynamic_deephit_total_loss(
            logits,
            y,
            d,
            order_ids,
            update_idx,
            x_batch,
            net,
            alpha=0.5,
            sigma=0.1,
            beta_l3=0.1,
            eps=0.0,
        )


def test_l2_rank_loss_zero_when_no_event_anchors():
    torch.manual_seed(7)

    logits = torch.randn(5, 2, 4)
    y = torch.tensor([0, 1, 2, 3, 1], dtype=torch.int64)
    d = torch.zeros(5, dtype=torch.int64)
    order_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64)
    update_idx = torch.tensor([0, 0, 0, 0, 0], dtype=torch.int64)

    loss_l2 = l2_rank_order_avg(logits, y, d, order_ids, update_idx, sigma=0.1)
    assert torch.allclose(loss_l2, torch.zeros_like(loss_l2), atol=1e-7)


def test_l3_aux_loss_zero_when_no_valid_transitions():
    torch.manual_seed(11)

    batch_size, seq_len, feat_dim = 3, 4, 8
    x_batch = torch.randn(batch_size, seq_len, feat_dim, dtype=torch.float32)

    # Mask yields no valid consecutive transitions.
    mask = torch.zeros(batch_size, seq_len, dtype=torch.float32)
    mask[:, 0] = 1.0
    x_batch[:, :, -1] = mask

    net = DummyNet(hidden_dim=6, pred_dim=feat_dim - 1)
    net._cache = {
        "state_out": torch.randn(batch_size, seq_len, 6, dtype=torch.float32),
        "mask": mask,
    }

    order_ids = torch.tensor([100, 101, 102], dtype=torch.int64)
    loss_l3 = l3_aux_order_avg(
        net,
        x_batch,
        order_ids,
        beta_l3=0.3,
        aux_target_dim=6,
    )

    assert torch.allclose(loss_l3, torch.zeros_like(loss_l3), atol=1e-7)
