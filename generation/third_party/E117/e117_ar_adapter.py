#!/usr/bin/env python3
"""AR-facing deployment adapter for the strict content-only E117 method."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from e84_ar_adapter import E84ARDecisionAdapter
from one_d_deep_spatial_residual_e113 import OneDDeepSpatialResidualE113


CHECKPOINT_FORMAT = "one_d_direct_reconstruction_spatial_e116_v1"
NUM_1D_TOKENS = 32
CONTENT_VOCABULARY_SIZE = 4096
K_THRESHOLD = 3.5894503537612965
SELECTED_GAMMA = 0.25
BETAS = {64: 1.0, 128: 4.0}


def standardize_grid_e117(values: torch.Tensor) -> torch.Tensor:
    flat = values.float().flatten(1)
    centered = flat - flat.mean(dim=1, keepdim=True)
    scale = (centered.square().mean(dim=1, keepdim=True) + 1e-6).sqrt()
    return (centered / scale).view_as(values)


def tie_adjusted_score_e117(
    k_score: torch.Tensor,
    grid_score: torch.Tensor,
) -> torch.Tensor:
    """Reproduce the frozen float64 tie break used by E117's K-selector."""

    primary = k_score.float().reshape(-1)
    grid = grid_score.float().reshape(primary.numel(), -1)
    if grid.shape[1] != 256:
        raise ValueError(f"E117 expected 256 grid scores, got {grid.shape}")
    positions = torch.arange(
        1, 257, device=grid.device, dtype=torch.float64
    )
    weights = torch.sin(positions * 12.9898) + torch.cos(positions * 78.233)
    grid64 = grid.double()
    projection = (grid64 * weights).sum(dim=1)
    normalizer = (grid64.abs() * weights.abs()).sum(dim=1).clamp_min(1e-12)
    tie_unit = (projection / normalizer).clamp(-1.0, 1.0)
    next_up = torch.nextafter(primary, torch.full_like(primary, float("inf")))
    adjusted = primary.double() + 0.25 * (next_up - primary).double() * tie_unit
    if not bool(torch.isfinite(adjusted).all()):
        raise FloatingPointError("E117 tie-adjusted K score is non-finite")
    return adjusted


def selected_tokens_e117(
    details: dict[str, torch.Tensor],
    threshold: float = K_THRESHOLD,
) -> torch.Tensor:
    adjusted = tie_adjusted_score_e117(
        details["k_score"], details["parent_grid"]
    )
    return torch.where(
        adjusted > float(threshold),
        torch.full_like(details["k_score"], 128, dtype=torch.long),
        torch.full_like(details["k_score"], 64, dtype=torch.long),
    )


def spatial_score_e117(
    details: dict[str, torch.Tensor],
    token_counts: torch.Tensor,
    gamma: float = SELECTED_GAMMA,
) -> torch.Tensor:
    """Apply the single frozen E117 spatial rule; no candidate search occurs."""

    if float(gamma) != SELECTED_GAMMA:
        raise ValueError(f"deployed E117 gamma must be {SELECTED_GAMMA}")
    parent = standardize_grid_e117(details["parent_grid"])
    boundary = details["boundary_logits"]
    if boundary.ndim != 4 or tuple(boundary.shape[1:]) != (3, 16, 16):
        raise ValueError(f"E117 boundary shape changed: {tuple(boundary.shape)}")
    tokens = token_counts.long().reshape(-1)
    if tokens.shape[0] != parent.shape[0] or not bool(
        torch.all((tokens == 64) | (tokens == 128))
    ):
        raise ValueError("E117 requires one K in {64,128} per image")
    score64 = parent + SELECTED_GAMMA * BETAS[64] * standardize_grid_e117(
        boundary[:, 0:1]
    )
    score128 = parent + SELECTED_GAMMA * BETAS[128] * standardize_grid_e117(
        boundary[:, 2:3]
    )
    score = torch.where(
        (tokens == 128).view(-1, 1, 1, 1), score128, score64
    )
    if not bool(torch.isfinite(score).all()):
        raise FloatingPointError("E117 spatial score is non-finite")
    return score


def load_e117(
    checkpoint_path: str | Path,
    device: torch.device,
    use_ema: bool = True,
) -> tuple[OneDDeepSpatialResidualE113, dict[str, object]]:
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"unsupported E117 checkpoint: {checkpoint.get('format')}")
    model = OneDDeepSpatialResidualE113(**checkpoint["model_config"]).to(device)
    state_name = "model_ema" if use_ema and "model_ema" in checkpoint else "model"
    model.load_state_dict(checkpoint[state_name], strict=True)
    model.eval().requires_grad_(False)
    checkpoint["loaded_model_state"] = state_name
    return model, checkpoint


class E117ARDecisionAdapter(nn.Module):
    """Map exactly 32 generated content IDs to one independent E117 decision."""

    def __init__(
        self,
        tokenizer_model: nn.Module,
        student: nn.Module,
    ) -> None:
        super().__init__()
        if not hasattr(tokenizer_model, "titok") or not hasattr(
            tokenizer_model.titok, "quantize"
        ):
            raise TypeError("tokenizer_model lacks the TiTok quantizer")
        if not hasattr(tokenizer_model, "latent_decoder") or not hasattr(
            tokenizer_model, "llamagen_vq"
        ):
            raise TypeError("tokenizer_model lacks the frozen 1D decode path")
        self.tokenizer_model = tokenizer_model.eval().requires_grad_(False)
        self.student = student.eval().requires_grad_(False)

    def _device(self) -> torch.device:
        parameter = next(self.student.parameters(), None)
        return parameter.device if parameter is not None else torch.device("cpu")

    def _quantized_from_codes(self, code_ids: torch.Tensor) -> torch.Tensor:
        if code_ids.ndim != 2 or tuple(code_ids.shape[1:]) != (NUM_1D_TOKENS,):
            raise ValueError("E117 AR expects content IDs with shape [B,32]")
        ids = code_ids.to(self._device(), dtype=torch.long)
        if ids.numel() == 0 or bool(torch.any(ids < 0)) or bool(
            torch.any(ids >= CONTENT_VOCABULARY_SIZE)
        ):
            raise ValueError("E117 content ID is outside [0,4095]")
        quantized = self.tokenizer_model.titok.quantize.get_codebook_entry(
            ids.reshape(-1)
        )
        return quantized.reshape(ids.shape[0], 1, NUM_1D_TOKENS, -1).permute(
            0, 3, 1, 2
        ).contiguous()

    def _decode_1d_bundle(
        self, z_quantized: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent_decoder = self.tokenizer_model.latent_decoder
        if getattr(latent_decoder, "head_mode", None) != "feature":
            raise ValueError("E117 requires the feature-mode latent decoder")
        f_prefix = latent_decoder.forward_backbone(z_quantized)
        f_1d = latent_decoder.lg_latent_head(f_prefix)
        x_base = self.tokenizer_model.llamagen_vq.decoder(f_1d)
        return f_prefix, f_1d, x_base

    @torch.inference_mode()
    def forward(
        self,
        code_ids: torch.Tensor,
        return_1d_features: bool = False,
    ) -> dict[str, Any]:
        ids = code_ids.to(self._device(), dtype=torch.long)
        z_quantized = self._quantized_from_codes(ids)
        f_prefix, f_1d, x_base = self._decode_1d_bundle(z_quantized)
        details = self.student.forward_details(ids, f_prefix, f_1d, x_base)
        token_counts = selected_tokens_e117(details)
        grid_scores = spatial_score_e117(details, token_counts)
        mask, selected_indices, selected_valid = (
            E84ARDecisionAdapter._selection_from_scores(
                grid_scores, token_counts
            )
        )
        result: dict[str, Any] = {
            "grid_scores": grid_scores,
            "k_score": details["k_score"],
            "token_counts": token_counts,
            "selected_mask": mask,
            "selected_indices_padded": selected_indices,
            "selected_indices_valid": selected_valid,
            "threshold": K_THRESHOLD,
            "gamma": SELECTED_GAMMA,
            "student_forwards": 1,
            "old_router_forwards": 0,
            "source_image_used": False,
            "f_2d_used": False,
            "probe_reconstructions": 0,
            "batch_statistics_used": False,
            "extra_route_fields_used": False,
        }
        if return_1d_features:
            result.update(
                {
                    "z_1d": z_quantized,
                    "f_prefix": f_prefix,
                    "f_1d": f_1d,
                    "x_base": x_base,
                }
            )
        return result
