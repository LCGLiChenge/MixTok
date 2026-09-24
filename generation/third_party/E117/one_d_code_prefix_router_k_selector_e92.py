#!/usr/bin/env python3
"""E87 plus code-only prefix/token residuals for AR-compatible grid selection."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from one_d_derived_router_k_selector_e87 import (
    BINARY_CANDIDATES,
    GRID_SIZE,
    OneDDerivedRouterKSelectorE87,
)


CHECKPOINT_FORMAT = "one_d_code_prefix_router_k_selector_e92_v1"
NUM_1D_TOKENS = 32
TITOK_CODEBOOK_SIZE = 4096


class PrefixGridResidual(nn.Module):
    """Predict a grid-logit residual from the pre-head 1D decoder feature."""

    def __init__(self, prefix_channels: int, hidden_dim: int) -> None:
        super().__init__()
        if prefix_channels % 32:
            raise ValueError("E92 prefix_channels must be divisible by 32")
        self.norm = nn.GroupNorm(32, prefix_channels, affine=False)
        self.proj = nn.Conv2d(prefix_channels, hidden_dim, 1)
        self.local = nn.Sequential(
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.head = nn.Conv2d(hidden_dim, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, f_prefix: torch.Tensor) -> torch.Tensor:
        return self.head(self.local(self.proj(self.norm(f_prefix))))


class CodeCrossGridResidual(nn.Module):
    """Let the 16x16 E87 queries attend directly to the 32 generated code IDs."""

    def __init__(
        self,
        codebook_size: int,
        num_tokens: int,
        hidden_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("E92 hidden_dim must be divisible by code attention heads")
        self.codebook_size = int(codebook_size)
        self.num_tokens = int(num_tokens)
        self.embedding = nn.Embedding(codebook_size, hidden_dim)
        self.position = nn.Parameter(torch.zeros(1, num_tokens, hidden_dim))
        nn.init.trunc_normal_(self.position, std=0.02)
        feedforward = int(round(hidden_dim * float(mlp_ratio)))
        self.encoder = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=feedforward,
                    dropout=float(dropout),
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(depth)
            ]
        )
        self.code_norm = nn.LayerNorm(hidden_dim)
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        spatial_queries: torch.Tensor,
        code_ids: torch.Tensor,
    ) -> torch.Tensor:
        if code_ids.ndim != 2 or code_ids.shape[1] != self.num_tokens:
            raise ValueError(
                f"E92 expects code_ids [B,{self.num_tokens}], got {tuple(code_ids.shape)}"
            )
        if code_ids.shape[0] != spatial_queries.shape[0]:
            raise ValueError("E92 code/spatial batch mismatch")
        if code_ids.numel() == 0:
            raise ValueError("E92 received an empty code batch")
        minimum = int(code_ids.detach().amin().item())
        maximum = int(code_ids.detach().amax().item())
        if minimum < 0 or maximum >= self.codebook_size:
            raise ValueError(
                f"E92 code outside [0,{self.codebook_size}): min={minimum}, max={maximum}"
            )
        codes = self.embedding(code_ids.long()) + self.position
        for block in self.encoder:
            codes = block(codes)
        codes = self.code_norm(codes)
        attended = self.cross_attention(
            self.query_norm(spatial_queries),
            codes,
            codes,
            need_weights=False,
        )[0]
        batch = spatial_queries.shape[0]
        return self.head(attended).transpose(1, 2).reshape(
            batch, 1, GRID_SIZE, GRID_SIZE
        )


class OneDCodePrefixRouterKSelectorE92(nn.Module):
    """Boost E87 spatial logits using information derived from the same 32 codes.

    The frozen E87 submodule supplies the unchanged K-selector and baseline grid.
    Only the prefix and direct-code spatial residuals are trainable.  Every input
    is deterministic from the generated 1D codes and the normal frozen base
    decode; no source image, GT, f_2d, probe reconstruction, or batch statistic is
    accepted.
    """

    binary_candidates = BINARY_CANDIDATES
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
        super().__init__()
        self.base = OneDDerivedRouterKSelectorE87(**(base_config or {}))
        self.prefix_channels = int(prefix_channels)
        self.codebook_size = int(codebook_size)
        self.num_1d_tokens = int(num_1d_tokens)
        self.code_depth = int(code_depth)
        self.code_attention_heads = int(code_attention_heads)
        self.code_mlp_ratio = float(code_mlp_ratio)
        self.code_dropout = float(code_dropout)
        self.prefix_residual = PrefixGridResidual(
            self.prefix_channels, self.base.hidden_dim
        )
        self.code_residual = CodeCrossGridResidual(
            self.codebook_size,
            self.num_1d_tokens,
            self.base.hidden_dim,
            self.code_depth,
            self.code_attention_heads,
            self.code_mlp_ratio,
            self.code_dropout,
        )
        self.freeze_base()

    def freeze_base(self) -> None:
        self.base.eval().requires_grad_(False)

    def train(self, mode: bool = True) -> "OneDCodePrefixRouterKSelectorE92":
        super().train(mode)
        self.base.eval()
        return self

    def _validate_prefix(
        self,
        f_prefix: torch.Tensor,
        f_1d: torch.Tensor,
    ) -> None:
        expected = (self.prefix_channels, GRID_SIZE, GRID_SIZE)
        if f_prefix.ndim != 4 or tuple(f_prefix.shape[1:]) != expected:
            raise ValueError(
                f"E92 expects f_prefix [B,{self.prefix_channels},16,16], "
                f"got {tuple(f_prefix.shape)}"
            )
        if f_prefix.shape[0] != f_1d.shape[0]:
            raise ValueError("E92 prefix/f_1d batch mismatch")

    def forward(
        self,
        code_ids: torch.Tensor,
        f_prefix: torch.Tensor,
        f_1d: torch.Tensor,
        x_base: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_prefix(f_prefix, f_1d)
        with torch.no_grad():
            base_hidden, base_tokens = self.base.hidden_features(f_1d, x_base)
            base_grid = self.base.grid_head(base_hidden)
            k_normalized = self.base.k_head(
                self.base._k_features(base_tokens, base_grid)
            ).squeeze(1).float()
            k_score = self.base.margin_mean + self.base.margin_scale * k_normalized
        prefix_delta = self.prefix_residual(f_prefix.detach())
        code_delta = self.code_residual(base_tokens.detach(), code_ids.detach())
        grid = base_grid + prefix_delta.to(base_grid.dtype) + code_delta.to(base_grid.dtype)
        expected_grid = (f_1d.shape[0], 1, GRID_SIZE, GRID_SIZE)
        if tuple(grid.shape) != expected_grid or k_score.shape != (f_1d.shape[0],):
            raise AssertionError("E92 output shape changed")
        return grid, k_score

    @torch.no_grad()
    def predict(
        self,
        code_ids: torch.Tensor,
        f_prefix: torch.Tensor,
        f_1d: torch.Tensor,
        x_base: torch.Tensor,
        threshold: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        grid, k_score = self(code_ids, f_prefix, f_1d, x_base)
        token_counts = torch.where(
            k_score > float(threshold),
            torch.full_like(k_score, 128, dtype=torch.long),
            torch.full_like(k_score, 64, dtype=torch.long),
        )
        return grid, token_counts

    def booster_parameters(self):
        yield from self.prefix_residual.parameters()
        yield from self.code_residual.parameters()

    def config_dict(self) -> dict[str, object]:
        return {
            "base_config": self.base.config_dict(),
            "prefix_channels": self.prefix_channels,
            "codebook_size": self.codebook_size,
            "num_1d_tokens": self.num_1d_tokens,
            "code_depth": self.code_depth,
            "code_attention_heads": self.code_attention_heads,
            "code_mlp_ratio": self.code_mlp_ratio,
            "code_dropout": self.code_dropout,
        }


def initialize_from_e87(
    model: OneDCodePrefixRouterKSelectorE92,
    source: OneDDerivedRouterKSelectorE87,
) -> dict[str, object]:
    model.base.load_state_dict(source.state_dict(), strict=True)
    model.freeze_base()
    if torch.count_nonzero(model.prefix_residual.head.weight).item() != 0:
        raise AssertionError("E92 prefix head is not zero-init")
    if torch.count_nonzero(model.code_residual.head.weight).item() != 0:
        raise AssertionError("E92 code head is not zero-init")
    return {
        "kind": "exact_E87_EMA_plus_zero_init_prefix_and_code_grid_residuals",
        "base_state": "model_ema",
        "base_frozen": True,
        "prefix_residual_zero_initialized": True,
        "code_residual_zero_initialized": True,
        "K_selector_exactly_inherited_and_frozen": True,
    }


def verify_model_contract(model: OneDCodePrefixRouterKSelectorE92) -> None:
    if tuple(model.binary_candidates) != BINARY_CANDIDATES:
        raise AssertionError("E92 candidate set changed")
    if tuple(model.input_contract) != ("code_ids", "f_prefix", "f_1d", "x_base"):
        raise AssertionError("E92 input contract changed")
    if any(parameter.requires_grad for parameter in model.base.parameters()):
        raise AssertionError("E92 E87 base must remain frozen")
    booster_ids = {id(parameter) for parameter in model.booster_parameters()}
    trainable_ids = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    if trainable_ids != booster_ids:
        raise AssertionError("E92 trainable set is not exactly the two boosters")
    if any(
        isinstance(module, nn.modules.batchnorm._BatchNorm)
        for module in model.modules()
    ):
        raise AssertionError("E92 must not contain BatchNorm")


def load_e92(
    checkpoint_path: str | Path,
    device: torch.device,
    use_ema: bool = True,
) -> tuple[OneDCodePrefixRouterKSelectorE92, dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"unsupported E92 checkpoint: {checkpoint.get('format')}")
    model = OneDCodePrefixRouterKSelectorE92(**checkpoint["model_config"]).to(device)
    state_name = "model_ema" if use_ema and "model_ema" in checkpoint else "model"
    model.load_state_dict(checkpoint[state_name], strict=True)
    model.freeze_base()
    verify_model_contract(model)
    model.eval()
    checkpoint["loaded_model_state"] = state_name
    return model, checkpoint
