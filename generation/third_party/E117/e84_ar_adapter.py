#!/usr/bin/env python3
"""AR-facing E84 adapter: 32 discrete 1D codes -> K and spatial locations."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from one_d_only_router_k_selector_e84 import OneDOnlyRouterKSelectorE84


class E84ARDecisionAdapter(nn.Module):
    """Make the E84 decision from generated TiTok code IDs only.

    The tokenizer model is frozen and supplies the normal codebook lookup plus
    1D decode path.  The E84 student receives only the resulting `f_1d`.  No
    source RGB image, x_base, f_2d, old Router output, or probe is accepted by
    this interface.
    """

    def __init__(
        self,
        tokenizer_model: nn.Module,
        student: OneDOnlyRouterKSelectorE84,
        threshold: float,
        num_1d_tokens: int = 32,
        codebook_size: int = 4096,
    ) -> None:
        super().__init__()
        if num_1d_tokens <= 0 or codebook_size <= 1:
            raise ValueError("invalid TiTok code contract")
        if not torch.isfinite(torch.tensor(float(threshold))):
            raise ValueError("E84 threshold must be finite")
        if not hasattr(tokenizer_model, "titok") or not hasattr(
            tokenizer_model.titok, "quantize"
        ):
            raise TypeError("tokenizer_model lacks the TiTok quantizer")
        if not callable(getattr(tokenizer_model, "decode", None)):
            raise TypeError("tokenizer_model lacks the frozen 1D decode path")
        self.tokenizer_model = tokenizer_model.eval().requires_grad_(False)
        self.student = student.eval().requires_grad_(False)
        self.threshold = float(threshold)
        self.num_1d_tokens = int(num_1d_tokens)
        self.codebook_size = int(codebook_size)

    def _device(self) -> torch.device:
        parameter = next(self.student.parameters(), None)
        if parameter is not None:
            return parameter.device
        buffer = next(self.student.buffers(), None)
        return buffer.device if buffer is not None else torch.device("cpu")

    def quantized_from_codes(self, code_ids: torch.Tensor) -> torch.Tensor:
        if code_ids.ndim != 2 or code_ids.shape[1] != self.num_1d_tokens:
            raise ValueError(
                f"E84 AR expects code_ids [B,{self.num_1d_tokens}], "
                f"got {tuple(code_ids.shape)}"
            )
        code_ids = code_ids.to(device=self._device(), dtype=torch.long)
        if code_ids.numel() == 0:
            raise ValueError("E84 AR received an empty code batch")
        minimum = int(code_ids.amin().item())
        maximum = int(code_ids.amax().item())
        if minimum < 0 or maximum >= self.codebook_size:
            raise ValueError(
                f"TiTok code outside [0,{self.codebook_size}): "
                f"min={minimum}, max={maximum}"
            )
        batch = int(code_ids.shape[0])
        quantized = self.tokenizer_model.titok.quantize.get_codebook_entry(
            code_ids.reshape(-1)
        )
        if quantized.ndim != 2 or quantized.shape[0] != batch * self.num_1d_tokens:
            raise ValueError(
                f"unexpected TiTok embedding shape {tuple(quantized.shape)}"
            )
        return quantized.reshape(
            batch, 1, self.num_1d_tokens, quantized.shape[-1]
        ).permute(0, 3, 1, 2).contiguous()

    def f_1d_from_codes(
        self, code_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z_quantized = self.quantized_from_codes(code_ids)
        x_base, f_1d, _code_probs, _code_logits = self.tokenizer_model.decode(
            z_quantized
        )
        expected = (
            code_ids.shape[0],
            self.student.latent_channels,
            16,
            16,
        )
        if tuple(f_1d.shape) != expected:
            raise ValueError(
                f"E84 AR expected f_1d {expected}, got {tuple(f_1d.shape)}"
            )
        return f_1d, x_base

    @staticmethod
    def _selection_from_scores(
        grid_scores: torch.Tensor,
        token_counts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = grid_scores.float().flatten(1)
        if flat.shape[1] != 256 or token_counts.shape != (flat.shape[0],):
            raise ValueError("E84 AR grid/K shape mismatch")
        order = torch.argsort(flat, dim=1, descending=True, stable=True)
        ranks = torch.empty_like(order)
        rank_values = torch.arange(256, device=flat.device).view(1, -1)
        ranks.scatter_(1, order, rank_values.expand_as(order))
        mask_flat = ranks < token_counts.long()[:, None]
        padded_indices = order[:, :128].clone()
        valid = (
            torch.arange(128, device=flat.device).view(1, -1)
            < token_counts.long()[:, None]
        )
        padded_indices.masked_fill_(~valid, -1)
        return mask_flat.view(flat.shape[0], 1, 16, 16), padded_indices, valid

    @torch.inference_mode()
    def forward(
        self,
        code_ids: torch.Tensor,
        return_1d_features: bool = False,
    ) -> dict[str, Any]:
        f_1d, x_base = self.f_1d_from_codes(code_ids)
        grid_scores, k_score = self.student(f_1d)
        token_counts = torch.where(
            k_score > self.threshold,
            torch.full_like(k_score, 128, dtype=torch.long),
            torch.full_like(k_score, 64, dtype=torch.long),
        )
        mask, selected_indices, selected_valid = self._selection_from_scores(
            grid_scores, token_counts
        )
        result: dict[str, Any] = {
            "grid_scores": grid_scores,
            "k_score": k_score,
            "token_counts": token_counts,
            "selected_mask": mask,
            "selected_indices_padded": selected_indices,
            "selected_indices_valid": selected_valid,
            "threshold": self.threshold,
            "student_forwards": 1,
            "old_router_forwards": 0,
            "source_image_used": False,
            "f_2d_used": False,
            "probe_reconstructions": 0,
        }
        if return_1d_features:
            result["f_1d"] = f_1d
            result["x_base"] = x_base
        return result


def titok_indices_from_result(
    result_dict: dict[str, Any], batch_size: int
) -> torch.Tensor:
    """Extract the same `[B,32]` IDs used by existing AR data pipelines."""

    for key in ("min_encoding_indices", "encoding_indices", "indices"):
        value = result_dict.get(key)
        if value is not None:
            indices = value.reshape(batch_size, -1).long()
            if indices.shape[1] != 32:
                raise ValueError(
                    f"TiTok result {key} has {indices.shape[1]} tokens, expected 32"
                )
            return indices
    raise KeyError(
        "TiTok result lacks min_encoding_indices/encoding_indices/indices"
    )
