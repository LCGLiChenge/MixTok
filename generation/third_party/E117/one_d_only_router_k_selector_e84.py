#!/usr/bin/env python3
"""Joint one-forward Router and binary K-Selector using only 1D-branch features."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


CHECKPOINT_FORMAT = "one_d_only_router_k_selector_e84_v1"
BINARY_CANDIDATES = (64, 128)
GRID_SIZE = 16


class ZeroInitGlobalBlock(nn.Module):
    """Global spatial mixing that starts as an exact identity branch."""

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
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
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
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )[0]
        tokens = tokens + self.attn_out(attended)
        return tokens + self.ffn(self.ffn_norm(tokens))


class OneDOnlyRouterKSelectorE84(nn.Module):
    """Predict 256 grid scores and one `{64,128}` score from `f_1d` only.

    `f_1d` is the 16x16 feature produced by the frozen 1D tokenizer branch.  It
    is a deterministic function of the 32 TiTok codes, so it is available after
    AR generation without the source image, 2D codes, GT, or a probe decode.
    """

    binary_candidates = BINARY_CANDIDATES
    input_contract = ("f_1d",)

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
        detach_input: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dim % 8:
            raise ValueError("hidden_dim must be divisible by 8 for GroupNorm")
        if conv_depth <= 0 or attention_depth < 0:
            raise ValueError("invalid E84 depth")
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
        self.detach_input = bool(detach_input)

        self.feat_proj = nn.Conv2d(latent_channels, hidden_dim, kernel_size=1)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, hidden_dim, GRID_SIZE, GRID_SIZE)
        )
        conv_blocks: list[nn.Module] = []
        for _ in range(conv_depth):
            conv_blocks.extend(
                [
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                    nn.GroupNorm(8, hidden_dim),
                    nn.SiLU(inplace=True),
                ]
            )
        self.trunk = nn.Sequential(*conv_blocks)
        self.global_blocks = nn.ModuleList(
            [
                ZeroInitGlobalBlock(
                    hidden_dim,
                    attention_heads,
                    mlp_ratio,
                    dropout,
                )
                for _ in range(attention_depth)
            ]
        )
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.grid_head = nn.Conv2d(hidden_dim, 1, kernel_size=1)

        # mean/std/max hidden summaries + 9 grid quantiles + 4 grid moments.
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

    def hidden_features(self, f_1d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        expected = (self.latent_channels, GRID_SIZE, GRID_SIZE)
        if f_1d.ndim != 4 or tuple(f_1d.shape[1:]) != expected:
            raise ValueError(
                f"E84 expects f_1d [B,{expected[0]},16,16], got {tuple(f_1d.shape)}"
            )
        if self.detach_input:
            f_1d = f_1d.detach()
        hidden = self.trunk(
            self.feat_proj(f_1d) + self.pos_embed.to(dtype=f_1d.dtype)
        )
        tokens = hidden.flatten(2).transpose(1, 2)
        for block in self.global_blocks:
            tokens = block(tokens)
        tokens = self.token_norm(tokens)
        hidden = tokens.transpose(1, 2).reshape(
            f_1d.shape[0], self.hidden_dim, GRID_SIZE, GRID_SIZE
        )
        return hidden, tokens

    def _k_features(
        self,
        tokens: torch.Tensor,
        grid_logits: torch.Tensor,
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, tokens = self.hidden_features(f_1d)
        grid_logits = self.grid_head(hidden)
        k_normalized = self.k_head(
            self._k_features(tokens, grid_logits)
        ).squeeze(1).float()
        k_score = self.margin_mean + self.margin_scale * k_normalized
        if grid_logits.shape != (f_1d.shape[0], 1, GRID_SIZE, GRID_SIZE):
            raise AssertionError("E84 grid output shape changed")
        if k_score.shape != (f_1d.shape[0],):
            raise AssertionError("E84 K output shape changed")
        return grid_logits, k_score

    @torch.no_grad()
    def predict(
        self,
        f_1d: torch.Tensor,
        threshold: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        grid_logits, k_score = self(f_1d)
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
            "detach_input": self.detach_input,
        }


def initialize_from_source_router(
    student: OneDOnlyRouterKSelectorE84,
    source_router: nn.Module,
) -> dict[str, object]:
    """Warm-start only exact f_1d-path tensors from the frozen source Router."""

    source = source_router.state_dict()
    target = student.state_dict()
    copied: list[str] = []
    aliases = {"grid_head.weight": "score_head.weight", "grid_head.bias": "score_head.bias"}
    for target_name, target_value in target.items():
        source_name = aliases.get(target_name, target_name)
        if source_name in source and source[source_name].shape == target_value.shape:
            target[target_name] = source[source_name].detach().to(
                device=target_value.device, dtype=target_value.dtype
            )
            copied.append(target_name)
    student.load_state_dict(target, strict=True)
    return {
        "kind": "source_Router_exact_f1d_path_only",
        "copied_parameter_tensors": len(copied),
        "copied_names": copied,
        "forbidden_source_branches_copied": False,
    }


def verify_model_contract(model: OneDOnlyRouterKSelectorE84) -> None:
    if tuple(model.binary_candidates) != BINARY_CANDIDATES:
        raise AssertionError("E84 candidate set changed")
    if tuple(model.input_contract) != ("f_1d",):
        raise AssertionError("E84 input contract changed")
    if model.grid_head.out_channels != 1 or model.k_head[-1].out_features != 1:
        raise AssertionError("E84 output heads changed")
    if any(
        isinstance(module, nn.modules.batchnorm._BatchNorm)
        for module in model.modules()
    ):
        raise AssertionError("E84 must not contain BatchNorm")


def load_e84(
    checkpoint_path: str | Path,
    device: torch.device,
    use_ema: bool = True,
) -> tuple[OneDOnlyRouterKSelectorE84, dict[str, object]]:
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"unsupported E84 checkpoint: {checkpoint.get('format')}")
    model = OneDOnlyRouterKSelectorE84(**checkpoint["model_config"]).to(device)
    verify_model_contract(model)
    state_name = "model_ema" if use_ema and "model_ema" in checkpoint else "model"
    model.load_state_dict(checkpoint[state_name], strict=True)
    model.eval().requires_grad_(False)
    checkpoint["loaded_model_state"] = state_name
    return model, checkpoint
