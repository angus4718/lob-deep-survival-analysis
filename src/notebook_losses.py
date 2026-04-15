"""Custom Dynamic-DeepHit loss helpers for notebook workflows."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def logits_to_pmf(logits: torch.Tensor) -> torch.Tensor:
    """Convert competing-risk logits to a joint PMF over (event, time-bin)."""
    if logits.ndim != 3:
        raise ValueError("logits must have shape (batch, num_events, num_time_bins)")
    batch_size, num_events, num_bins = logits.shape
    flat = logits.reshape(batch_size, num_events * num_bins)
    return F.softmax(flat, dim=1).reshape(batch_size, num_events, num_bins)


def pmf_to_cif(pmf: torch.Tensor) -> torch.Tensor:
    """Compute cause-specific CIF by cumulative summation over time bins."""
    if pmf.ndim != 3:
        raise ValueError("pmf must have shape (batch, num_events, num_time_bins)")
    return torch.cumsum(pmf, dim=2)


def _order_average(values: torch.Tensor, order_ids: torch.Tensor) -> torch.Tensor:
    """Average per-sample values as mean over order-level means."""
    if values.ndim != 1:
        raise ValueError("values must be a 1D tensor")
    if order_ids.ndim != 1:
        raise ValueError("order_ids must be a 1D tensor")
    if values.size(0) != order_ids.size(0):
        raise ValueError("values and order_ids must have the same length")

    order_ids = order_ids.long().to(values.device)
    unique_orders, inverse = torch.unique(order_ids, sorted=True, return_inverse=True)

    order_sums = torch.zeros(
        unique_orders.numel(),
        device=values.device,
        dtype=values.dtype,
    )
    order_counts = torch.zeros_like(order_sums)

    order_sums.scatter_add_(0, inverse, values)
    order_counts.scatter_add_(0, inverse, torch.ones_like(values))

    order_means = order_sums / order_counts.clamp_min(1.0)
    return order_means.mean()


def l1_nll_order_avg_from_pmf(
    pmf: torch.Tensor,
    y: torch.Tensor,
    d: torch.Tensor,
    order_ids: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """L1 negative log-likelihood with exact order-level averaging."""
    batch_size, num_events, num_bins = pmf.shape
    tau = y.long().to(pmf.device).clamp(0, num_bins - 1)
    event_code = d.long().to(pmf.device)

    sample_nll = torch.zeros(batch_size, device=pmf.device, dtype=pmf.dtype)

    event_mask = event_code > 0
    if torch.any(event_mask):
        event_idx = (event_code[event_mask] - 1).clamp(0, num_events - 1)
        event_prob = pmf[event_mask, event_idx, tau[event_mask]]
        sample_nll[event_mask] = -torch.log(event_prob.clamp_min(eps))

    cens_mask = ~event_mask
    if torch.any(cens_mask):
        any_event_cif = torch.cumsum(pmf.sum(dim=1), dim=1)
        surv_prob = 1.0 - any_event_cif[cens_mask, tau[cens_mask]]
        sample_nll[cens_mask] = -torch.log(surv_prob.clamp_min(eps))

    return _order_average(sample_nll, order_ids)


def l1_nll_order_avg(
    logits: torch.Tensor,
    y: torch.Tensor,
    d: torch.Tensor,
    order_ids: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """L1 negative log-likelihood with exact order-level averaging."""
    pmf = logits_to_pmf(logits)
    return l1_nll_order_avg_from_pmf(pmf, y, d, order_ids, eps=eps)


def l2_rank_order_avg_from_cif(
    cif: torch.Tensor,
    y: torch.Tensor,
    d: torch.Tensor,
    order_ids: torch.Tensor,
    update_idx: torch.Tensor,
    *,
    sigma: float = 0.1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """L2 event-specific ranking loss with exact order-level averaging.

    The pairwise comparisons are built within each update-index slice (same S).
    """
    batch_size, num_events, num_bins = cif.shape
    if batch_size == 0:
        return torch.zeros((), device=cif.device, dtype=cif.dtype)

    eps_val = float(eps)
    if (not math.isfinite(eps_val)) or eps_val <= 0.0:
        raise ValueError(f"eps must be finite and > 0. Got {eps!r}")
    sigma_raw = float(sigma)
    if (not math.isfinite(sigma_raw)) or sigma_raw <= 0.0:
        raise ValueError(f"sigma must be finite and > 0. Got {sigma!r}")

    tau = y.long().to(cif.device).clamp(0, num_bins - 1)
    event_code = d.long().to(cif.device)
    order_ids = order_ids.long().to(cif.device)
    update_idx = update_idx.long().to(cif.device)

    sigma_val = max(sigma_raw, eps_val)
    sample_terms = torch.zeros(batch_size, device=cif.device, dtype=cif.dtype)

    unique_steps = torch.unique(update_idx, sorted=True)
    for step in unique_steps.tolist():
        idx = torch.nonzero(update_idx == step, as_tuple=False).squeeze(1)
        if idx.numel() <= 1:
            continue

        tau_s = tau[idx]
        order_s = order_ids[idx]
        event_s = event_code[idx]

        for event_id in range(1, num_events + 1):
            anchors_local = torch.nonzero(event_s == event_id, as_tuple=False).squeeze(1)
            if anchors_local.numel() == 0:
                continue

            for local_anchor in anchors_local.tolist():
                tau_anchor = tau_s[local_anchor]
                valid = (order_s != order_s[local_anchor]) & (tau_s > tau_anchor)
                if not torch.any(valid):
                    continue

                i_idx = idx[local_anchor]
                j_idx = idx[valid]

                f_i = cif[i_idx, event_id - 1, tau_anchor]
                f_j = cif[j_idx, event_id - 1, tau_anchor]
                pair_terms = torch.exp(-(f_i - f_j) / sigma_val)
                sample_terms[i_idx] = sample_terms[i_idx] + pair_terms.sum()

    return _order_average(sample_terms, order_ids)


def l2_rank_order_avg(
    logits: torch.Tensor,
    y: torch.Tensor,
    d: torch.Tensor,
    order_ids: torch.Tensor,
    update_idx: torch.Tensor,
    *,
    sigma: float = 0.1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """L2 event-specific ranking loss with exact order-level averaging."""
    pmf = logits_to_pmf(logits)
    cif = pmf_to_cif(pmf)
    return l2_rank_order_avg_from_cif(
        cif,
        y,
        d,
        order_ids,
        update_idx,
        sigma=sigma,
        eps=eps,
    )


def l3_aux_order_avg(
    base_net,
    x: torch.Tensor,
    order_ids: torch.Tensor,
    *,
    beta_l3: float = 0.1,
    aux_target_dim: int | None = None,
) -> torch.Tensor:
    """L3 auxiliary next-step prediction loss with exact order averaging."""
    zero = torch.zeros((), device=x.device, dtype=x.dtype)
    beta_val = float(beta_l3)
    if not math.isfinite(beta_val):
        raise ValueError(f"beta_l3 must be finite. Got {beta_l3!r}")
    if beta_val < 0.0:
        raise ValueError(f"beta_l3 must be >= 0. Got {beta_l3!r}")
    if beta_val == 0.0:
        return zero

    cache = getattr(base_net, "_cache", None)
    aux_head = getattr(base_net, "aux_head", None)
    if cache is None or aux_head is None:
        return zero

    state_out = cache.get("state_out") if isinstance(cache, dict) else None
    mask = cache.get("mask") if isinstance(cache, dict) else None
    if state_out is None or mask is None:
        return zero
    if x.size(1) <= 1:
        return zero

    pred_next = aux_head(state_out[:, :-1, :])

    max_target_dim = min(pred_next.size(-1), x.size(-1))
    if aux_target_dim is None:
        # Exclude side and mask channels by default.
        inferred = x.size(-1) - 2
        aux_target_dim = max(1, min(inferred, max_target_dim))
    aux_target_dim = int(max(1, min(aux_target_dim, max_target_dim)))

    pred_next = pred_next[:, :, :aux_target_dim]
    target_next = x[:, 1:, :aux_target_dim]

    pair_valid = (mask[:, :-1] > 0.5) & (mask[:, 1:] > 0.5)
    pair_err = (pred_next - target_next).pow(2).mean(dim=-1)

    sample_sum = (pair_err * pair_valid.float()).sum(dim=1)
    sample_count = pair_valid.float().sum(dim=1)

    order_ids = order_ids.long().to(x.device)
    unique_orders, inverse = torch.unique(order_ids, sorted=True, return_inverse=True)

    order_sum = torch.zeros(unique_orders.numel(), device=x.device, dtype=x.dtype)
    order_count = torch.zeros_like(order_sum)

    order_sum.scatter_add_(0, inverse, sample_sum)
    order_count.scatter_add_(0, inverse, sample_count)

    order_avg = torch.where(
        order_count > 0,
        order_sum / order_count.clamp_min(1.0),
        torch.zeros_like(order_sum),
    )
    return beta_val * order_avg.mean()


def dynamic_deephit_total_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    d: torch.Tensor,
    order_ids: torch.Tensor,
    update_idx: torch.Tensor,
    x_batch: torch.Tensor,
    base_net,
    *,
    alpha: float = 1.0,
    sigma: float = 0.1,
    beta_l3: float = 0.1,
    aux_target_dim: int | None = None,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute L_total = L1 + alpha * L2 + L3 with exact order-level averaging."""
    alpha_val = float(alpha)
    if (not math.isfinite(alpha_val)) or alpha_val < 0.0:
        raise ValueError(f"alpha must be finite and >= 0. Got {alpha!r}")

    eps_val = float(eps)
    if (not math.isfinite(eps_val)) or eps_val <= 0.0:
        raise ValueError(f"eps must be finite and > 0. Got {eps!r}")

    pmf = logits_to_pmf(logits)
    cif = pmf_to_cif(pmf)

    l1 = l1_nll_order_avg_from_pmf(pmf, y, d, order_ids, eps=eps_val)
    l2 = l2_rank_order_avg_from_cif(
        cif,
        y,
        d,
        order_ids,
        update_idx,
        sigma=sigma,
        eps=eps_val,
    )
    l3 = l3_aux_order_avg(
        base_net,
        x_batch,
        order_ids,
        beta_l3=beta_l3,
        aux_target_dim=aux_target_dim,
    )

    total = l1 + (alpha_val * l2) + l3
    return total, {"l1": l1, "l2": l2, "l3": l3}


__all__ = [
    "dynamic_deephit_total_loss",
    "l1_nll_order_avg",
    "l2_rank_order_avg",
    "l3_aux_order_avg",
    "logits_to_pmf",
    "pmf_to_cif",
]
