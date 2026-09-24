#!/usr/bin/env python3
"""E92 architecture trained to correct frozen E87 top-K swap errors."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from one_d_code_prefix_router_k_selector_e92 import (
    BINARY_CANDIDATES,
    GRID_SIZE,
    NUM_1D_TOKENS,
    TITOK_CODEBOOK_SIZE,
    OneDCodePrefixRouterKSelectorE92,
)


CHECKPOINT_FORMAT = "one_d_error_focused_grid_corrector_e94_v1"
BOUNDARIES = (64, 96, 128)
INITIAL_PREFIX_SCALE = 0.5
INITIAL_CODE_SCALE = 1.5
SWAP_MARGIN = 0.25
SIGN_MARGIN = 0.10
SIGN_WEIGHT = 0.25
RESIDUAL_RMS_WEIGHT = 0.01


class OneDErrorFocusedGridCorrectorE94(OneDCodePrefixRouterKSelectorE92):
    """One-forward E92-compatible model with an E94 checkpoint identity."""

    input_contract = ("code_ids", "f_prefix", "f_1d", "x_base")

    def __init__(
        self,
        base_config: dict[str, object] | None = None,
        prefix_channels: int = 1024,
        codebook_size: int = TITOK_CODEBOOK_SIZE,
        num_1d_tokens: int = NUM_1D_TOKENS,
        code_depth: int = 2,
        code_attention_heads: int = 8,
        code_mlp_ratio: float = 4.0,
        code_dropout: float = 0.1,
    ) -> None:
        super().__init__(
            base_config=base_config,
            prefix_channels=prefix_channels,
            codebook_size=codebook_size,
            num_1d_tokens=num_1d_tokens,
            code_depth=code_depth,
            code_attention_heads=code_attention_heads,
            code_mlp_ratio=code_mlp_ratio,
            code_dropout=code_dropout,
        )


def initialize_from_e92(
    model: OneDErrorFocusedGridCorrectorE94,
    source: OneDCodePrefixRouterKSelectorE92,
    prefix_scale: float = INITIAL_PREFIX_SCALE,
    code_scale: float = INITIAL_CODE_SCALE,
) -> dict[str, object]:
    if not torch.isfinite(torch.tensor(prefix_scale)) or not torch.isfinite(
        torch.tensor(code_scale)
    ):
        raise ValueError("E94 initialization scales must be finite")
    model.load_state_dict(source.state_dict(), strict=True)
    with torch.no_grad():
        model.prefix_residual.head.weight.mul_(float(prefix_scale))
        model.prefix_residual.head.bias.mul_(float(prefix_scale))
        model.code_residual.head.weight.mul_(float(code_scale))
        model.code_residual.head.bias.mul_(float(code_scale))
    model.freeze_base()
    verify_model_contract(model)
    return {
        "kind": "E92_EMA_with_train_only_E93_branch_scales",
        "source_state": "model_ema",
        "prefix_head_scale": float(prefix_scale),
        "code_head_scale": float(code_scale),
        "base_exactly_inherited_and_frozen": True,
        "K_selector_exactly_inherited_and_frozen": True,
        "scale_source": "E93 ImageNet-train calibration only",
    }


def _standardize(flat: torch.Tensor) -> torch.Tensor:
    centered = flat.float() - flat.float().mean(dim=1, keepdim=True)
    return centered / centered.std(
        dim=1, keepdim=True, unbiased=False
    ).clamp_min(1e-5)


def _topk_mask(flat: torch.Tensor, k: int) -> torch.Tensor:
    indices = flat.detach().topk(k, dim=1, largest=True, sorted=False).indices
    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask.scatter_(1, indices, True)
    return mask


def _masked_per_image_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    counts = mask.sum(dim=tuple(range(1, mask.ndim)))
    sums = (values * mask.float()).sum(dim=tuple(range(1, values.ndim)))
    valid = counts > 0
    means = sums / counts.clamp_min(1).float()
    return means, valid


def error_focused_correction_loss(
    student_logits: torch.Tensor,
    base_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    boundaries: Iterable[int] = BOUNDARIES,
    swap_margin: float = SWAP_MARGIN,
    sign_margin: float = SIGN_MARGIN,
    sign_weight: float = SIGN_WEIGHT,
    residual_rms_weight: float = RESIDUAL_RMS_WEIGHT,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Train the residual on cells that E87 swaps across fixed-K boundaries."""

    student = student_logits.float().flatten(1)
    base = base_logits.detach().float().flatten(1)
    teacher = teacher_logits.detach().float().flatten(1)
    if student.shape != base.shape or student.shape != teacher.shape:
        raise ValueError("E94 student/base/teacher shape mismatch")
    if student.shape[1] != GRID_SIZE * GRID_SIZE:
        raise ValueError("E94 expects a 16x16 grid")
    frozen_boundaries = tuple(int(boundary) for boundary in boundaries)
    if frozen_boundaries != BOUNDARIES:
        raise ValueError("E94 boundaries changed")

    student_z = _standardize(student)
    base_scale = base.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-5)
    delta = (student - base) / base_scale
    delta = delta - delta.mean(dim=1, keepdim=True)
    pair_differences = student_z[:, :, None] - student_z[:, None, :]

    swap_losses: list[torch.Tensor] = []
    sign_losses: list[torch.Tensor] = []
    valid_fractions: list[torch.Tensor] = []
    metrics: dict[str, torch.Tensor] = {}
    for boundary in frozen_boundaries:
        teacher_mask = _topk_mask(teacher, boundary)
        base_mask = _topk_mask(base, boundary)
        false_negative = teacher_mask & ~base_mask
        false_positive = base_mask & ~teacher_mask
        pair_mask = false_negative[:, :, None] & false_positive[:, None, :]

        pair_values = F.softplus(float(swap_margin) - pair_differences)
        pair_means, pair_valid = _masked_per_image_mean(pair_values, pair_mask)
        if pair_valid.any():
            swap_losses.append(pair_means[pair_valid].mean())
        else:
            swap_losses.append(student.sum() * 0.0)

        positive_values = F.softplus(float(sign_margin) - delta)
        negative_values = F.softplus(float(sign_margin) + delta)
        positive_means, positive_valid = _masked_per_image_mean(
            positive_values, false_negative
        )
        negative_means, negative_valid = _masked_per_image_mean(
            negative_values, false_positive
        )
        valid = positive_valid & negative_valid
        if valid.any():
            sign_losses.append(
                0.5 * (positive_means[valid] + negative_means[valid]).mean()
            )
        else:
            sign_losses.append(student.sum() * 0.0)
        valid_fractions.append(valid.float().mean())

        student_mask = _topk_mask(student, boundary)
        intersection = (student_mask & teacher_mask).sum(dim=1).float()
        union = (student_mask | teacher_mask).sum(dim=1).float().clamp_min(1)
        metrics[f"error_iou_fixed{boundary}"] = (intersection / union).mean()
        mismatch_count = false_negative.sum(dim=1).float()
        metrics[f"error_base_mismatch_count_fixed{boundary}"] = mismatch_count.mean()

    swap_loss = torch.stack(swap_losses).mean()
    sign_loss = torch.stack(sign_losses).mean()
    residual_rms = delta.square().mean(dim=1).sqrt().mean()
    total = (
        swap_loss
        + float(sign_weight) * sign_loss
        + float(residual_rms_weight) * residual_rms
    )
    metrics.update(
        {
            "error_swap_loss": swap_loss,
            "error_sign_loss": sign_loss,
            "error_residual_rms": residual_rms,
            "error_valid_image_fraction": torch.stack(valid_fractions).mean(),
        }
    )
    return total, metrics


def verify_model_contract(model: OneDErrorFocusedGridCorrectorE94) -> None:
    if tuple(model.binary_candidates) != BINARY_CANDIDATES:
        raise AssertionError("E94 candidate set changed")
    if tuple(model.input_contract) != ("code_ids", "f_prefix", "f_1d", "x_base"):
        raise AssertionError("E94 input contract changed")
    if any(parameter.requires_grad for parameter in model.base.parameters()):
        raise AssertionError("E94 E87 base/K-selector must remain frozen")
    booster_ids = {id(parameter) for parameter in model.booster_parameters()}
    trainable_ids = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    if trainable_ids != booster_ids:
        raise AssertionError("E94 trainable set is not exactly the E92 boosters")
    if any(
        isinstance(module, nn.modules.batchnorm._BatchNorm)
        for module in model.modules()
    ):
        raise AssertionError("E94 must not contain BatchNorm")


def load_e94(
    checkpoint_path: str | Path,
    device: torch.device,
    use_ema: bool = True,
) -> tuple[OneDErrorFocusedGridCorrectorE94, dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"unsupported E94 checkpoint: {checkpoint.get('format')}")
    model = OneDErrorFocusedGridCorrectorE94(**checkpoint["model_config"]).to(device)
    state_name = "model_ema" if use_ema and "model_ema" in checkpoint else "model"
    model.load_state_dict(checkpoint[state_name], strict=True)
    model.freeze_base()
    verify_model_contract(model)
    model.eval()
    checkpoint["loaded_model_state"] = state_name
    return model, checkpoint
