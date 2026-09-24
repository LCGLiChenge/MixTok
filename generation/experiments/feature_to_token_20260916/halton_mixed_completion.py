"""Training masks matched to random and Halton-prefix completion states.

Only Router-selected positions are targets. Unselected proxy anchors are never
masked or supervised. The sampling and model APIs stay identical to phase 3.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from experiments.feature_to_token_20260916.halton_official_completion import official_components


def halton_prefix_mask(indices: torch.Tensor, order_rank: torch.Tensor,
                       completed_rounds: torch.Tensor, steps: int = 32) -> torch.Tensor:
    """Mask selected positions not revealed after 0..steps-1 Halton rounds."""
    batch, k = indices.shape
    if k < 1 or completed_rounds.shape != (batch,):
        raise ValueError("expected nonempty homogeneous K buckets")
    if bool(((completed_rounds < 0) | (completed_rounds >= steps)).any()):
        raise ValueError("completed_rounds outside sampling prefix")
    fraction = 1 - torch.arccos(completed_rounds.float() / steps) / (math.pi * .5)
    revealed = torch.maximum((k * fraction).long(), completed_rounds)
    revealed = revealed.clamp(max=k - 1)
    selected_rank = order_rank.to(indices.device)[indices]
    position_in_prefix = selected_rank.argsort(dim=1).argsort(dim=1)
    return position_in_prefix >= revealed[:, None]


class MixedHaltonCompletionObjective(nn.Module):
    def __init__(self, core, upstream_root: Path):
        super().__init__()
        self.core = core
        self.get_mask_code, sampler = official_components(upstream_root)
        xy = sampler.build_halton_mask(16)
        order = xy[:, 0] * 16 + xy[:, 1]
        if order.numel() != 256 or order.unique().numel() != 256:
            raise RuntimeError("invalid official Halton order")
        self.register_buffer("order_rank", order.argsort(), persistent=False)

    def forward(self, proxy, gt_sparse, labels, indices, valid):
        if not bool(valid.all()) or indices.shape != gt_sparse.shape:
            raise ValueError("expected homogeneous K buckets without padding")
        if not bool((indices.sort(1).values[:, 1:] > indices.sort(1).values[:, :-1]).all()):
            raise ValueError("duplicate route coordinates")
        if not bool(((indices >= 0) & (indices < 256)).all()):
            raise ValueError("invalid route coordinates")
        batch, k = indices.shape
        # Match the official RNG order: class dropout precedes masking.
        drop = (torch.rand(batch) < .1).to(labels.device)
        _, random_mask, _ = self.get_mask_code(
            gt_sparse[:, None, :], mode="arccos", value=16384, codebook_size=16384)
        random_mask = random_mask[:, 0]
        empty = ~random_mask.any(dim=1)
        if bool(empty.any()):
            rows = torch.nonzero(empty, as_tuple=True)[0]
            random_mask[rows, torch.randint(k, (len(rows),), device=indices.device)] = True
        completed = torch.randint(0, 32, (batch,), device=indices.device)
        prefix_mask = halton_prefix_mask(indices, self.order_rank, completed)
        use_prefix = torch.rand(batch, device=indices.device) < .5
        mask = torch.where(use_prefix[:, None], prefix_mask, random_mask)
        if not bool(mask.any(dim=1).all()):
            raise RuntimeError("mask schedule left a sample without targets")
        sparse_input = gt_sparse.masked_fill(mask, 16384)
        inputs = proxy.clone().scatter(1, indices, sparse_input)
        target = gt_sparse.masked_fill(~mask, -100)
        logits = self.core(inputs, labels, drop, indices)
        if logits.shape[-1] != 16385:
            raise RuntimeError("training must retain the MASK output class")
        loss = F.cross_entropy(logits.flatten(0, 1), target.flatten(), ignore_index=-100)
        with torch.no_grad():
            nll = F.cross_entropy(logits.flatten(0, 1), gt_sparse.flatten(), reduction="none").view_as(gt_sparse)
            stats = torch.stack((loss.detach(), nll[mask].sum(), mask.sum(),
                                 mask.float().mean(), drop.float().mean()))
        return loss, stats
