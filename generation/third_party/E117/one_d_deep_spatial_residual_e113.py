#!/usr/bin/env python3
"""High-capacity batch-independent spatial residual from 1D-derived inputs."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from one_d_error_focused_grid_corrector_e94 import (
    BINARY_CANDIDATES,
    GRID_SIZE,
    NUM_1D_TOKENS,
    TITOK_CODEBOOK_SIZE,
    OneDErrorFocusedGridCorrectorE94,
)


CHECKPOINT_FORMAT = "one_d_deep_spatial_residual_e113_v1"
BOUNDARIES = (64, 96, 128)


class ConvNeXtSpatialBlock(nn.Module):
    def __init__(self, width: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        expanded = int(round(width * float(mlp_ratio)))
        self.depthwise = nn.Conv2d(
            width, width, kernel_size=7, padding=3, groups=width
        )
        self.norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, expanded),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expanded, width),
        )
        self.layer_scale = nn.Parameter(torch.full((width,), 1e-3))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        residual = self.depthwise(hidden).permute(0, 2, 3, 1)
        residual = self.mlp(self.norm(residual)) * self.layer_scale
        return hidden + residual.permute(0, 3, 1, 2)


class SpatialCodeAttentionBlock(nn.Module):
    def __init__(
        self,
        width: int,
        heads: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("E113 width must be divisible by attention heads")
        expanded = int(round(width * float(mlp_ratio)))
        self.self_norm = nn.LayerNorm(width)
        self.self_attention = nn.MultiheadAttention(
            width, heads, dropout=dropout, batch_first=True
        )
        self.cross_query_norm = nn.LayerNorm(width)
        self.cross_code_norm = nn.LayerNorm(width)
        self.cross_attention = nn.MultiheadAttention(
            width, heads, dropout=dropout, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, expanded),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expanded, width),
        )
        self.self_scale = nn.Parameter(torch.full((width,), 1e-3))
        self.cross_scale = nn.Parameter(torch.full((width,), 1e-3))
        self.ffn_scale = nn.Parameter(torch.full((width,), 1e-3))

    def forward(
        self, spatial_tokens: torch.Tensor, code_tokens: torch.Tensor
    ) -> torch.Tensor:
        normalized = self.self_norm(spatial_tokens)
        self_value = self.self_attention(
            normalized, normalized, normalized, need_weights=False
        )[0]
        spatial_tokens = spatial_tokens + self_value * self.self_scale
        cross_value = self.cross_attention(
            self.cross_query_norm(spatial_tokens),
            self.cross_code_norm(code_tokens),
            self.cross_code_norm(code_tokens),
            need_weights=False,
        )[0]
        spatial_tokens = spatial_tokens + cross_value * self.cross_scale
        return spatial_tokens + self.ffn(
            self.ffn_norm(spatial_tokens)
        ) * self.ffn_scale


class OneDDeepSpatialResidualE113(nn.Module):
    """Deep grid booster whose deployment inputs all derive from 32 1D codes."""

    binary_candidates = BINARY_CANDIDATES
    input_contract = ("code_ids", "f_prefix", "f_1d", "x_base")

    def __init__(
        self,
        parent_config: dict[str, object] | None = None,
        width: int = 256,
        depth: int = 6,
        heads: int = 8,
        code_depth: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if depth <= 0 or code_depth <= 0 or width % heads:
            raise ValueError("invalid E113 depth/width/heads")
        self.parent = OneDErrorFocusedGridCorrectorE94(**(parent_config or {}))
        self.parent.freeze_base()
        self.parent.eval().requires_grad_(False)
        self.width = int(width)
        self.depth = int(depth)
        self.heads = int(heads)
        self.code_depth = int(code_depth)
        self.mlp_ratio = float(mlp_ratio)
        self.dropout = float(dropout)
        prefix_channels = self.parent.prefix_channels
        latent_channels = self.parent.base.latent_channels
        base_hidden = self.parent.base.hidden_dim
        codebook_size = self.parent.codebook_size

        self.prefix_norm = nn.GroupNorm(32, prefix_channels, affine=False)
        self.prefix_proj = nn.Conv2d(prefix_channels, width, 1)
        self.f1d_norm = nn.GroupNorm(32, latent_channels, affine=False)
        self.f1d_proj = nn.Conv2d(latent_channels, width, 1)
        self.parent_hidden_proj = nn.Conv2d(base_hidden, width, 1)
        self.position = nn.Parameter(torch.zeros(1, width, GRID_SIZE, GRID_SIZE))
        nn.init.trunc_normal_(self.position, std=0.02)

        self.code_embedding = nn.Embedding(codebook_size, width)
        self.code_position = nn.Parameter(
            torch.zeros(1, NUM_1D_TOKENS, width)
        )
        nn.init.trunc_normal_(self.code_position, std=0.02)
        expanded = int(round(width * float(mlp_ratio)))
        self.code_encoder = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=width,
                    nhead=heads,
                    dim_feedforward=expanded,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(code_depth)
            ]
        )
        self.code_norm = nn.LayerNorm(width)
        self.local_blocks = nn.ModuleList(
            [
                ConvNeXtSpatialBlock(width, mlp_ratio, dropout)
                for _ in range(depth)
            ]
        )
        self.attention_blocks = nn.ModuleList(
            [
                SpatialCodeAttentionBlock(width, heads, mlp_ratio, dropout)
                for _ in range(depth)
            ]
        )
        self.output_norm = nn.LayerNorm(width)
        self.grid_head = nn.Linear(width, 1)
        self.boundary_head = nn.Linear(width, len(BOUNDARIES))
        nn.init.zeros_(self.grid_head.weight)
        nn.init.zeros_(self.grid_head.bias)
        nn.init.zeros_(self.boundary_head.weight)
        nn.init.zeros_(self.boundary_head.bias)

    def train(self, mode: bool = True) -> "OneDDeepSpatialResidualE113":
        super().train(mode)
        self.parent.eval()
        return self

    def residual_parameters(self):
        for name, parameter in self.named_parameters():
            if not name.startswith("parent."):
                yield parameter

    def _parent_components(
        self,
        code_ids: torch.Tensor,
        f_prefix: torch.Tensor,
        f_1d: torch.Tensor,
        x_base: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            base_hidden, base_tokens = self.parent.base.hidden_features(
                f_1d, x_base
            )
            base_grid = self.parent.base.grid_head(base_hidden)
            k_normalized = self.parent.base.k_head(
                self.parent.base._k_features(base_tokens, base_grid)
            ).squeeze(1).float()
            k_score = (
                self.parent.base.margin_mean
                + self.parent.base.margin_scale * k_normalized
            )
            parent_grid = (
                base_grid
                + self.parent.prefix_residual(f_prefix.detach()).to(base_grid.dtype)
                + self.parent.code_residual(
                    base_tokens.detach(), code_ids.detach()
                ).to(base_grid.dtype)
            )
        return parent_grid, k_score, base_hidden

    def forward_details(
        self,
        code_ids: torch.Tensor,
        f_prefix: torch.Tensor,
        f_1d: torch.Tensor,
        x_base: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if code_ids.ndim != 2 or code_ids.shape[1] != NUM_1D_TOKENS:
            raise ValueError("E113 expects code_ids [B,32]")
        parent_grid, k_score, parent_hidden = self._parent_components(
            code_ids, f_prefix, f_1d, x_base
        )
        hidden = (
            self.prefix_proj(self.prefix_norm(f_prefix.detach()))
            + self.f1d_proj(self.f1d_norm(f_1d.detach()))
            + self.parent_hidden_proj(parent_hidden.detach())
            + self.position.to(dtype=f_prefix.dtype)
        )
        code_tokens = self.code_embedding(code_ids.detach().long())
        code_tokens = code_tokens + self.code_position.to(code_tokens.dtype)
        for block in self.code_encoder:
            code_tokens = block(code_tokens)
        code_tokens = self.code_norm(code_tokens)
        for local, attention in zip(self.local_blocks, self.attention_blocks):
            hidden = local(hidden)
            spatial_tokens = hidden.flatten(2).transpose(1, 2)
            spatial_tokens = attention(spatial_tokens, code_tokens)
            hidden = spatial_tokens.transpose(1, 2).reshape(
                hidden.shape[0], self.width, GRID_SIZE, GRID_SIZE
            )
        normalized = self.output_norm(hidden.flatten(2).transpose(1, 2))
        grid_delta = self.grid_head(normalized).transpose(1, 2).reshape(
            hidden.shape[0], 1, GRID_SIZE, GRID_SIZE
        )
        boundary_logits = self.boundary_head(normalized).transpose(1, 2).reshape(
            hidden.shape[0], len(BOUNDARIES), GRID_SIZE, GRID_SIZE
        )
        grid = parent_grid + grid_delta.to(parent_grid.dtype)
        return {
            "grid": grid,
            "k_score": k_score,
            "parent_grid": parent_grid,
            "grid_delta": grid_delta,
            "boundary_logits": boundary_logits,
        }

    def forward(
        self,
        code_ids: torch.Tensor,
        f_prefix: torch.Tensor,
        f_1d: torch.Tensor,
        x_base: torch.Tensor,
        return_details: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | dict[str, torch.Tensor]:
        details = self.forward_details(code_ids, f_prefix, f_1d, x_base)
        if return_details:
            return details
        return details["grid"], details["k_score"]

    def config_dict(self) -> dict[str, object]:
        return {
            "parent_config": self.parent.config_dict(),
            "width": self.width,
            "depth": self.depth,
            "heads": self.heads,
            "code_depth": self.code_depth,
            "mlp_ratio": self.mlp_ratio,
            "dropout": self.dropout,
        }


def initialize_from_e103(
    model: OneDDeepSpatialResidualE113,
    source: OneDErrorFocusedGridCorrectorE94,
) -> dict[str, object]:
    model.parent.load_state_dict(source.state_dict(), strict=True)
    model.parent.eval().requires_grad_(False)
    verify_model_contract(model)
    return {
        "kind": "exact_E103_EMA_plus_zero_init_deep_spatial_residual",
        "parent_state": "model_ema",
        "parent_frozen": True,
        "grid_and_boundary_heads_zero_initialized": True,
        "K_selector_exactly_inherited_and_frozen": True,
    }


def verify_model_contract(model: OneDDeepSpatialResidualE113) -> None:
    if tuple(model.binary_candidates) != BINARY_CANDIDATES:
        raise AssertionError("E113 binary candidates changed")
    if tuple(model.input_contract) != (
        "code_ids",
        "f_prefix",
        "f_1d",
        "x_base",
    ):
        raise AssertionError("E113 input contract changed")
    if any(parameter.requires_grad for parameter in model.parent.parameters()):
        raise AssertionError("E113 parent must remain frozen")
    residual_ids = {id(parameter) for parameter in model.residual_parameters()}
    trainable_ids = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    if residual_ids != trainable_ids:
        raise AssertionError("E113 trainable set is not exactly the residual branch")
    if any(
        isinstance(module, nn.modules.batchnorm._BatchNorm)
        for module in model.modules()
    ):
        raise AssertionError("E113 must not contain BatchNorm")


def load_e113(
    checkpoint_path: str | Path,
    device: torch.device,
    use_ema: bool = True,
) -> tuple[OneDDeepSpatialResidualE113, dict[str, object]]:
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"unsupported E113 checkpoint: {checkpoint.get('format')}")
    model = OneDDeepSpatialResidualE113(**checkpoint["model_config"]).to(device)
    state_name = "model_ema" if use_ema and "model_ema" in checkpoint else "model"
    model.load_state_dict(checkpoint[state_name], strict=True)
    model.parent.eval().requires_grad_(False)
    verify_model_contract(model)
    model.eval()
    checkpoint["loaded_model_state"] = state_name
    return model, checkpoint

