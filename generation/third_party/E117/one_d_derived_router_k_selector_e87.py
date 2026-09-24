#!/usr/bin/env python3
"""One-forward Router/K-Selector from features decoded from 1D codes only."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


CHECKPOINT_FORMAT = "one_d_derived_router_k_selector_e87_v1"
BINARY_CANDIDATES = (64, 128)
GRID_SIZE = 16


class ZeroInitGlobalBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        mlp_dim = int(round(hidden_dim * float(mlp_ratio)))
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.attn_out = nn.Linear(hidden_dim, hidden_dim)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_dim),
        )
        nn.init.zeros_(self.attn_out.weight)
        nn.init.zeros_(self.attn_out.bias)
        nn.init.zeros_(self.ffn[-1].weight)
        nn.init.zeros_(self.ffn[-1].bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        normalized = self.attn_norm(tokens)
        attended = self.attn(
            normalized, normalized, normalized, need_weights=False
        )[0]
        tokens = tokens + self.attn_out(attended)
        return tokens + self.ffn(self.ffn_norm(tokens))


class XBasePyramid(nn.Module):
    """Compress the deterministic 1D base reconstruction from 256 to 16."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        channels = (32, 64, 96, hidden_dim)
        stages: list[nn.Module] = []
        in_channels = 3
        for out_channels in channels:
            groups = min(8, out_channels)
            stages.extend(
                [
                    nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1),
                    nn.GroupNorm(groups, out_channels),
                    nn.SiLU(inplace=True),
                ]
            )
            in_channels = out_channels
        self.stages = nn.Sequential(*stages)
        self.refine = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim * 2, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim * 2, hidden_dim, 1),
        )

    def forward(self, x_base: torch.Tensor) -> torch.Tensor:
        features = self.stages(x_base.float())
        if tuple(features.shape[-2:]) != (GRID_SIZE, GRID_SIZE):
            raise ValueError(
                f"E87 expects x_base to downsample to 16x16, got {tuple(features.shape)}"
            )
        return features + self.refine(features)


class OneDDerivedRouterKSelectorE87(nn.Module):
    """Predict spatial scores and `{64,128}` cost from `f_1d` and `x_base`.

    Both inputs are deterministic outputs of the frozen decode of the same 32
    discrete 1D codes.  The class accepts no source image, GT, or 2D feature.
    """

    binary_candidates = BINARY_CANDIDATES
    input_contract = ("f_1d", "x_base")

    def __init__(
        self,
        latent_channels: int = 256,
        hidden_dim: int = 128,
        conv_depth: int = 3,
        attention_depth: int = 2,
        attention_heads: int = 8,
        mlp_ratio: float = 4.0,
        k_hidden_dim: int = 512,
        dropout: float = 0.1,
        margin_mean: float = 3.5918153005480065,
        margin_scale: float = 0.8157468242326514,
        detach_inputs: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dim % 8:
            raise ValueError("hidden_dim must be divisible by 8")
        if conv_depth <= 0 or attention_depth < 0:
            raise ValueError("invalid E87 depth")
        if not torch.isfinite(torch.tensor(margin_mean)):
            raise ValueError("margin_mean must be finite")
        if not torch.isfinite(torch.tensor(margin_scale)) or margin_scale <= 0:
            raise ValueError("margin_scale must be finite and positive")

        self.latent_channels = int(latent_channels)
        self.hidden_dim = int(hidden_dim)
        self.conv_depth = int(conv_depth)
        self.attention_depth = int(attention_depth)
        self.attention_heads = int(attention_heads)
        self.mlp_ratio = float(mlp_ratio)
        self.k_hidden_dim = int(k_hidden_dim)
        self.dropout = float(dropout)
        self.detach_inputs = bool(detach_inputs)

        # Names intentionally match E84 for exact warm-starting.
        self.feat_proj = nn.Conv2d(latent_channels, hidden_dim, 1)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, hidden_dim, GRID_SIZE, GRID_SIZE)
        )
        conv_blocks: list[nn.Module] = []
        for _ in range(conv_depth):
            conv_blocks.extend(
                [
                    nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                    nn.GroupNorm(8, hidden_dim),
                    nn.SiLU(inplace=True),
                ]
            )
        self.trunk = nn.Sequential(*conv_blocks)
        self.global_blocks = nn.ModuleList(
            [
                ZeroInitGlobalBlock(
                    hidden_dim, attention_heads, mlp_ratio, dropout
                )
                for _ in range(attention_depth)
            ]
        )
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.grid_head = nn.Conv2d(hidden_dim, 1, 1)

        self.xbase_encoder = XBasePyramid(hidden_dim)
        self.xbase_pre_fusion = nn.Conv2d(hidden_dim, hidden_dim, 1)
        self.xbase_post_fusion = nn.Conv2d(hidden_dim, hidden_dim, 1)
        nn.init.zeros_(self.xbase_pre_fusion.weight)
        nn.init.zeros_(self.xbase_pre_fusion.bias)
        nn.init.zeros_(self.xbase_post_fusion.weight)
        nn.init.zeros_(self.xbase_post_fusion.bias)

        self.register_buffer(
            "grid_summary_indices",
            torch.tensor((0, 31, 63, 95, 127, 159, 191, 223, 255)),
            persistent=True,
        )
        k_input_dim = 3 * hidden_dim + 13
        self.k_head = nn.Sequential(
            nn.LayerNorm(k_input_dim),
            nn.Linear(k_input_dim, k_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(k_hidden_dim, k_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(k_hidden_dim, 1),
        )
        self.register_buffer(
            "margin_mean", torch.tensor(float(margin_mean), dtype=torch.float32)
        )
        self.register_buffer(
            "margin_scale", torch.tensor(float(margin_scale), dtype=torch.float32)
        )

        nn.init.normal_(self.grid_head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.grid_head.bias)
        nn.init.normal_(self.k_head[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.k_head[-1].bias)

    def hidden_features(
        self,
        f_1d: torch.Tensor,
        x_base: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected = (self.latent_channels, GRID_SIZE, GRID_SIZE)
        if f_1d.ndim != 4 or tuple(f_1d.shape[1:]) != expected:
            raise ValueError(
                f"E87 expects f_1d [B,{expected[0]},16,16], got {tuple(f_1d.shape)}"
            )
        if (
            x_base.ndim != 4
            or x_base.shape[0] != f_1d.shape[0]
            or x_base.shape[1:] != (3, 256, 256)
        ):
            raise ValueError(
                f"E87 expects x_base [B,3,256,256], got {tuple(x_base.shape)}"
            )
        if self.detach_inputs:
            f_1d = f_1d.detach()
            x_base = x_base.detach()
        x_features = self.xbase_encoder(x_base)
        hidden = self.trunk(
            self.feat_proj(f_1d)
            + self.pos_embed.to(dtype=f_1d.dtype)
            + self.xbase_pre_fusion(x_features).to(dtype=f_1d.dtype)
        )
        tokens = hidden.flatten(2).transpose(1, 2)
        for block in self.global_blocks:
            tokens = block(tokens)
        post = self.xbase_post_fusion(x_features).flatten(2).transpose(1, 2)
        tokens = self.token_norm(tokens + post.to(dtype=tokens.dtype))
        hidden = tokens.transpose(1, 2).reshape(
            f_1d.shape[0], self.hidden_dim, GRID_SIZE, GRID_SIZE
        )
        return hidden, tokens

    def _k_features(
        self, tokens: torch.Tensor, grid_logits: torch.Tensor
    ) -> torch.Tensor:
        token_mean = tokens.float().mean(dim=1)
        token_std = tokens.float().std(dim=1, unbiased=False)
        token_max = tokens.float().amax(dim=1)
        flat_grid = grid_logits.float().flatten(1)
        sorted_grid = flat_grid.sort(dim=1).values
        quantiles = sorted_grid.index_select(1, self.grid_summary_indices)
        grid_moments = torch.stack(
            (
                flat_grid.mean(dim=1),
                flat_grid.std(dim=1, unbiased=False),
                flat_grid.amin(dim=1),
                flat_grid.amax(dim=1),
            ),
            dim=1,
        )
        return torch.cat(
            (token_mean, token_std, token_max, quantiles, grid_moments), dim=1
        )

    def forward(
        self,
        f_1d: torch.Tensor,
        x_base: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, tokens = self.hidden_features(f_1d, x_base)
        grid_logits = self.grid_head(hidden)
        k_normalized = self.k_head(
            self._k_features(tokens, grid_logits)
        ).squeeze(1).float()
        k_score = self.margin_mean + self.margin_scale * k_normalized
        return grid_logits, k_score

    @torch.no_grad()
    def predict(
        self,
        f_1d: torch.Tensor,
        x_base: torch.Tensor,
        threshold: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        grid_logits, k_score = self(f_1d, x_base)
        tokens = torch.where(
            k_score > float(threshold),
            torch.full_like(k_score, 128, dtype=torch.long),
            torch.full_like(k_score, 64, dtype=torch.long),
        )
        return grid_logits, tokens

    def config_dict(self) -> dict[str, object]:
        return {
            "latent_channels": self.latent_channels,
            "hidden_dim": self.hidden_dim,
            "conv_depth": self.conv_depth,
            "attention_depth": self.attention_depth,
            "attention_heads": self.attention_heads,
            "mlp_ratio": self.mlp_ratio,
            "k_hidden_dim": self.k_hidden_dim,
            "dropout": self.dropout,
            "margin_mean": float(self.margin_mean.item()),
            "margin_scale": float(self.margin_scale.item()),
            "detach_inputs": self.detach_inputs,
        }


def initialize_from_e84(
    model: OneDDerivedRouterKSelectorE87,
    e84: nn.Module,
) -> dict[str, object]:
    source = e84.state_dict()
    target = model.state_dict()
    copied: list[str] = []
    for name, target_value in target.items():
        if name in source and source[name].shape == target_value.shape:
            target[name] = source[name].detach().to(
                device=target_value.device, dtype=target_value.dtype
            )
            copied.append(name)
    model.load_state_dict(target, strict=True)
    required = set(source)
    if not required.issubset(copied):
        missing = sorted(required.difference(copied))
        raise ValueError(f"E87 failed to warm-start all E84 tensors: {missing}")
    return {
        "kind": "exact_formal_E84_plus_zero_init_xbase_residuals",
        "copied_parameter_tensors": len(copied),
        "copied_names": copied,
        "xbase_fusions_zero_initialized": True,
    }


def verify_model_contract(model: OneDDerivedRouterKSelectorE87) -> None:
    if tuple(model.binary_candidates) != BINARY_CANDIDATES:
        raise AssertionError("E87 candidate set changed")
    if tuple(model.input_contract) != ("f_1d", "x_base"):
        raise AssertionError("E87 input contract changed")
    if any(
        isinstance(module, nn.modules.batchnorm._BatchNorm)
        for module in model.modules()
    ):
        raise AssertionError("E87 must not contain BatchNorm")
    for fusion in (model.xbase_pre_fusion, model.xbase_post_fusion):
        if not torch.count_nonzero(fusion.weight).item() == 0:
            raise AssertionError("fresh E87 x_base fusion is not zero-init")


def load_e87(
    checkpoint_path: str | Path,
    device: torch.device,
    use_ema: bool = True,
) -> tuple[OneDDerivedRouterKSelectorE87, dict[str, object]]:
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"unsupported E87 checkpoint: {checkpoint.get('format')}")
    model = OneDDerivedRouterKSelectorE87(**checkpoint["model_config"]).to(device)
    state_name = "model_ema" if use_ema and "model_ema" in checkpoint else "model"
    model.load_state_dict(checkpoint[state_name], strict=True)
    model.eval().requires_grad_(False)
    checkpoint["loaded_model_state"] = state_name
    return model, checkpoint
