"""Hierarchical unified MaskGIT for TiTok-1D and E117-routed sparse 2D codes.

The model deliberately shares the same LLaMA/RandAR-style Transformer blocks
between two factorized masked-modeling stages:

1. class -> 32 TiTok content IDs;
2. class + completed TiTok IDs + E117-selected coordinates -> sparse 2D IDs.

E117 is external and frozen.  This module consumes only its selected 16x16
coordinates and valid mask; it never predicts or changes the routing decision.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


_RANDAR_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "RandAR"
if str(_RANDAR_ROOT) not in sys.path:
    sys.path.insert(0, str(_RANDAR_ROOT))

from RandAR.model.llamagen_gpt import LabelEmbedder, RMSNorm, precompute_freqs_cis_2d  # noqa: E402
from RandAR.model.randar_gpt import TransformerBlock  # noqa: E402


def _mask_from_counts(
    valid: torch.Tensor,
    counts: torch.Tensor,
    scores: torch.Tensor,
) -> torch.Tensor:
    """Select exactly ``counts[b]`` valid entries with the lowest scores."""

    if valid.dtype != torch.bool or valid.ndim != 2:
        raise ValueError("valid must be a [B,L] boolean tensor")
    if counts.shape != (valid.shape[0],):
        raise ValueError("counts must have shape [B]")
    if scores.shape != valid.shape:
        raise ValueError("scores must have the same shape as valid")
    valid_counts = valid.sum(dim=1)
    if bool(torch.any(counts < 0)) or bool(torch.any(counts > valid_counts)):
        raise ValueError("mask counts must be within each sample's valid length")

    ranked_scores = scores.masked_fill(~valid, float("inf"))
    order = ranked_scores.argsort(dim=1)
    ranks = torch.empty_like(order)
    ranks.scatter_(1, order, torch.arange(valid.shape[1], device=valid.device).expand_as(order))
    return (ranks < counts[:, None]) & valid


def sample_arccos_mask(
    valid: torch.Tensor,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Original TiTok/MaskGIT arccos training mask with variable valid length."""

    batch = valid.shape[0]
    device = valid.device
    timesteps = torch.rand(batch, device=device, generator=generator)
    ratios = torch.acos(timesteps) / (math.pi * 0.5)
    valid_counts = valid.sum(dim=1)
    counts = torch.round(valid_counts.float() * ratios).long().clamp_min(1)
    counts = torch.minimum(counts, valid_counts)
    scores = torch.rand(valid.shape, device=device, generator=generator)
    return _mask_from_counts(valid, counts, scores), ratios


def fixed_ratio_mask(
    valid: torch.Tensor,
    ratio: float,
    seed: int,
) -> torch.Tensor:
    """Deterministic fixed-ratio mask used by the feasibility holdout."""

    ratio = float(ratio)
    if not 0.0 < ratio <= 1.0:
        raise ValueError("ratio must be in (0,1]")
    generator = torch.Generator(device=valid.device)
    generator.manual_seed(int(seed))
    valid_counts = valid.sum(dim=1)
    counts = torch.round(valid_counts.float() * ratio).long().clamp_min(1)
    counts = torch.minimum(counts, valid_counts)
    scores = torch.rand(valid.shape, device=valid.device, generator=generator)
    return _mask_from_counts(valid, counts, scores)


def _per_sample_mean(values: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
    weights = selected.to(values.dtype)
    counts = weights.sum(dim=1)
    means = (values * weights).sum(dim=1) / counts.clamp_min(1.0)
    return torch.where(counts > 0, means, torch.zeros_like(means))


def masked_token_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    masked: torch.Tensor,
    valid: torch.Tensor,
    label_smoothing: float = 0.1,
    unmasked_weight: float = 0.1,
    loss_normalization: str = "per_sample",
) -> dict[str, torch.Tensor]:
    """Compute TiTok-weighted MLM loss and masked-token diagnostics.

    ``global_titok`` exactly matches TiTok's MLMLoss reduction.  ``per_sample``
    retains the original E117 implementation's equal-example reduction.  K=64
    and K=128 are bucketed into separate batches, so the TiTok reduction does
    not mix different valid sequence lengths within an optimizer micro-step.
    """

    if logits.shape[:2] != targets.shape or targets.shape != masked.shape or masked.shape != valid.shape:
        raise ValueError("logits/targets/masked/valid shapes do not agree")
    if bool(torch.any(masked & ~valid)):
        raise ValueError("masked positions must be valid")
    safe_targets = targets.masked_fill(~valid, 0)
    ce = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        safe_targets.reshape(-1),
        reduction="none",
        label_smoothing=float(label_smoothing),
    ).view_as(targets)
    masked_loss = _per_sample_mean(ce, masked).mean()
    unmasked = valid & ~masked
    unmasked_loss = _per_sample_mean(ce, unmasked).mean()
    weights = valid.to(ce.dtype) * torch.where(
        masked,
        torch.ones((), dtype=ce.dtype, device=ce.device),
        torch.full((), float(unmasked_weight), dtype=ce.dtype, device=ce.device),
    )
    if loss_normalization == "global_titok":
        loss = (ce * weights).sum() / weights.sum().clamp_min(1e-8)
    elif loss_normalization == "per_sample":
        per_sample = (ce * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1e-8)
        loss = per_sample.mean()
    else:
        raise ValueError("loss_normalization must be 'global_titok' or 'per_sample'")

    predictions = logits.argmax(dim=-1)
    correct = (predictions == safe_targets).float()
    topk = min(5, logits.shape[-1])
    top5 = (logits.topk(topk, dim=-1).indices == safe_targets[..., None]).any(dim=-1).float()
    masked_acc = _per_sample_mean(correct, masked).mean()
    masked_top5 = _per_sample_mean(top5, masked).mean()
    return {
        "loss": loss,
        "masked_loss": masked_loss.detach(),
        "unmasked_loss": unmasked_loss.detach(),
        "masked_acc": masked_acc.detach(),
        "masked_top5": masked_top5.detach(),
        "mask_ratio": (masked.sum(dim=1).float() / valid.sum(dim=1).clamp_min(1)).mean().detach(),
    }


class E117SparseUnifiedMaskGIT(nn.Module):
    """Shared bidirectional Transformer with separate 1D and 2D vocabularies."""

    def __init__(
        self,
        dim: int = 1024,
        n_layer: int = 24,
        n_head: int = 16,
        n_kv_head: int | None = None,
        multiple_of: int = 256,
        ffn_dim_multiplier: float | None = None,
        rope_base: float = 10000.0,
        norm_eps: float = 1e-5,
        initializer_range: float = 0.02,
        attn_dropout_p: float = 0.0,
        resid_dropout_p: float = 0.1,
        ffn_dropout_p: float = 0.1,
        drop_path_rate: float = 0.0,
        token_dropout_p: float = 0.1,
        num_classes: int = 1000,
        class_dropout_prob: float = 0.1,
        titok_vocab_size: int = 4096,
        titok_num_tokens: int = 32,
        llamagen_vocab_size: int = 16384,
        grid_size: int = 16,
        max_sparse_tokens: int = 128,
        grad_checkpointing: bool = True,
        zero_init_output: bool = True,
        output_bias: bool = False,
        backbone_type: str = "llama",
        bert_intermediate_size: int = 3072,
        bert_layer_norm_eps: float = 1e-12,
    ) -> None:
        super().__init__()
        if backbone_type not in ("llama", "bert"):
            raise ValueError("backbone_type must be 'llama' or 'bert'")
        if dim % n_head != 0:
            raise ValueError("dim must be divisible by n_head")
        if backbone_type == "llama" and (dim // n_head) % 4 != 0:
            raise ValueError("head dimension must be divisible by four for 2D RoPE")
        if titok_num_tokens > grid_size * grid_size:
            raise ValueError("1D RoPE compatibility requires <= grid_size**2 tokens")

        self.dim = int(dim)
        self.n_layer = int(n_layer)
        self.n_head = int(n_head)
        self.titok_vocab_size = int(titok_vocab_size)
        self.titok_num_tokens = int(titok_num_tokens)
        self.llamagen_vocab_size = int(llamagen_vocab_size)
        self.grid_size = int(grid_size)
        self.grid_tokens = self.grid_size * self.grid_size
        self.max_sparse_tokens = int(max_sparse_tokens)
        self.grad_checkpointing = bool(grad_checkpointing)
        self.initializer_range = float(initializer_range)
        self.backbone_type = str(backbone_type)
        if self.backbone_type == "bert" and self.grad_checkpointing:
            raise ValueError("BERT gradient checkpointing is not implemented in this shared wrapper")

        self.mask_token_1d = self.titok_vocab_size
        self.mask_token_2d = self.llamagen_vocab_size
        self.pad_token_2d = self.llamagen_vocab_size + 1

        self.class_embedding = LabelEmbedder(num_classes, dim, class_dropout_prob)
        self.embedding_1d = nn.Embedding(self.titok_vocab_size + 1, dim)
        self.embedding_2d = nn.Embedding(self.llamagen_vocab_size + 2, dim, padding_idx=self.pad_token_2d)
        self.pos_embedding_1d = nn.Embedding(self.titok_num_tokens, dim)
        self.pos_embedding_2d = nn.Embedding(self.grid_tokens, dim)
        self.modality_embedding = nn.Embedding(2, dim)
        # 0 = no sparse stage, 1 = K64, 2 = K128.
        self.budget_embedding = nn.Embedding(3, dim)
        self.token_dropout = nn.Dropout(token_dropout_p)

        if self.backbone_type == "llama":
            dpr = torch.linspace(0, drop_path_rate, n_layer).tolist()
            self.layers = nn.ModuleList(
                [
                    TransformerBlock(
                        dim=dim,
                        n_layer=n_layer,
                        n_head=n_head,
                        n_kv_head=n_kv_head,
                        multiple_of=multiple_of,
                        ffn_dim_multiplier=ffn_dim_multiplier,
                        rope_base=rope_base,
                        norm_eps=norm_eps,
                        attn_dropout_p=attn_dropout_p,
                        resid_dropout_p=resid_dropout_p,
                        ffn_dropout_p=ffn_dropout_p,
                        drop_path=dpr[layer_id],
                    )
                    for layer_id in range(n_layer)
                ]
            )
            self.norm = RMSNorm(dim, eps=norm_eps)
            freqs = precompute_freqs_cis_2d(
                self.grid_size,
                self.dim // self.n_head,
                rope_base,
                cls_token_num=1,
            )
            self.register_buffer("freqs_cis", freqs, persistent=False)
            self.bert_input_norm = None
            self.bert_input_dropout = None
            self.class_pos_embedding = None
        else:
            os.environ.setdefault("USE_TF", "0")
            from transformers import BertConfig
            from transformers.models.bert.modeling_bert import BertEncoder

            bert_config = BertConfig(
                vocab_size=1,
                hidden_size=dim,
                num_hidden_layers=n_layer,
                num_attention_heads=n_head,
                intermediate_size=int(bert_intermediate_size),
                hidden_act="gelu",
                hidden_dropout_prob=float(resid_dropout_p),
                attention_probs_dropout_prob=float(attn_dropout_p),
                layer_norm_eps=float(bert_layer_norm_eps),
                position_embedding_type="absolute",
                add_cross_attention=False,
            )
            self.layers = BertEncoder(bert_config)
            self.bert_input_norm = nn.LayerNorm(dim, eps=float(bert_layer_norm_eps))
            self.bert_input_dropout = nn.Dropout(float(resid_dropout_p))
            self.class_pos_embedding = nn.Parameter(torch.empty(1, 1, dim))
            self.norm = nn.Identity()
        self.output_1d = nn.Linear(dim, self.titok_vocab_size, bias=output_bias)
        self.output_2d = nn.Linear(dim, self.llamagen_vocab_size, bias=output_bias)
        self.apply(self._init_weights)
        if self.class_pos_embedding is not None:
            nn.init.normal_(self.class_pos_embedding, std=self.initializer_range)
        with torch.no_grad():
            self.embedding_2d.weight[self.pad_token_2d].zero_()
        if zero_init_output:
            nn.init.zeros_(self.output_1d.weight)
            nn.init.zeros_(self.output_2d.weight)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=self.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _class_tokens(
        self,
        labels: torch.Tensor,
        force_drop_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        class_tokens = self.class_embedding(
            labels,
            train=self.training,
            force_drop_ids=force_drop_ids,
        )[:, :1]
        if self.class_pos_embedding is not None:
            class_tokens = class_tokens + self.class_pos_embedding
        return class_tokens

    def _one_d_embeddings(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2 or tokens.shape[1] != self.titok_num_tokens:
            raise ValueError(f"1D tokens must be [B,{self.titok_num_tokens}]")
        if bool(torch.any(tokens < 0)) or bool(torch.any(tokens > self.mask_token_1d)):
            raise ValueError("1D token is outside the content+mask vocabulary")
        positions = torch.arange(self.titok_num_tokens, device=tokens.device)
        return (
            self.embedding_1d(tokens)
            + self.pos_embedding_1d(positions)[None]
            + self.modality_embedding.weight[0][None, None]
        )

    def _sequence_freqs_1d(self, batch: int, device: torch.device) -> torch.Tensor:
        freqs = self.freqs_cis.to(device)
        one_d = freqs[1 : 1 + self.titok_num_tokens]
        return torch.cat((freqs[:1], one_d), dim=0)[None].expand(batch, -1, -1, -1)

    def _run_backbone(
        self,
        hidden: torch.Tensor,
        freqs: torch.Tensor | None,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        if hidden.shape[:2] != valid.shape:
            raise ValueError("hidden/valid sequence shapes do not agree")
        if self.backbone_type == "bert":
            if freqs is not None:
                raise ValueError("BERT backbone does not consume RoPE frequencies")
            hidden = self.bert_input_dropout(self.bert_input_norm(hidden))
            attention_mask = torch.zeros(
                (valid.shape[0], 1, 1, valid.shape[1]),
                dtype=hidden.dtype,
                device=hidden.device,
            )
            attention_mask.masked_fill_(~valid[:, None, None, :], torch.finfo(hidden.dtype).min)
            hidden = self.layers(
                hidden,
                attention_mask=attention_mask,
                return_dict=False,
            )[0]
            return hidden * valid[..., None].to(hidden.dtype)
        if freqs is None or freqs.shape[:2] != valid.shape:
            raise ValueError("LLaMA freqs/valid sequence shapes do not agree")
        # Passing an explicit mask is essential: RandAR uses causal SDPA only
        # when mask=None.  This all-direction valid-pair mask makes MaskGIT
        # genuinely bidirectional and excludes padded K64 slots.
        attention_mask = valid[:, None, :, None] & valid[:, None, None, :]
        hidden = hidden * valid[..., None].to(hidden.dtype)
        for layer in self.layers:
            if self.grad_checkpointing and self.training:
                hidden = checkpoint(
                    layer,
                    hidden,
                    freqs,
                    None,
                    attention_mask,
                    use_reentrant=False,
                )
            else:
                hidden = layer(hidden, freqs, None, attention_mask)
        return self.norm(hidden) * valid[..., None].to(hidden.dtype)

    def forward_1d(
        self,
        input_tokens: torch.Tensor,
        labels: torch.Tensor,
        force_drop_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch = input_tokens.shape[0]
        class_h = self._class_tokens(labels, force_drop_ids)
        one_d_h = self.token_dropout(self._one_d_embeddings(input_tokens))
        hidden = torch.cat((class_h, one_d_h), dim=1)
        valid = torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
        freqs = (
            self._sequence_freqs_1d(batch, hidden.device)
            if self.backbone_type == "llama"
            else None
        )
        hidden = self._run_backbone(hidden, freqs, valid)
        return self.output_1d(hidden[:, 1:]).float()

    def _validate_sparse_inputs(
        self,
        input_tokens: torch.Tensor,
        route_indices: torch.Tensor,
        route_valid: torch.Tensor,
    ) -> None:
        if input_tokens.shape != route_indices.shape or input_tokens.shape != route_valid.shape:
            raise ValueError("sparse token/index/valid shapes must agree")
        if input_tokens.ndim != 2 or input_tokens.shape[1] > self.max_sparse_tokens:
            raise ValueError("sparse input must be [B,K<=max_sparse_tokens]")
        if route_valid.dtype != torch.bool:
            raise ValueError("route_valid must be boolean")
        if bool(torch.any(route_indices[route_valid] < 0)) or bool(
            torch.any(route_indices[route_valid] >= self.grid_tokens)
        ):
            raise ValueError("E117 route index is outside the 16x16 grid")
        valid_tokens = input_tokens[route_valid]
        if bool(torch.any(valid_tokens < 0)) or bool(torch.any(valid_tokens > self.mask_token_2d)):
            raise ValueError("valid 2D token is outside the content+mask vocabulary")
        if bool(torch.any(input_tokens[~route_valid] != self.pad_token_2d)):
            raise ValueError("invalid sparse slots must contain the 2D pad token")

    def forward_2d(
        self,
        completed_1d: torch.Tensor,
        input_tokens: torch.Tensor,
        route_indices: torch.Tensor,
        route_valid: torch.Tensor,
        labels: torch.Tensor,
        force_drop_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._validate_sparse_inputs(input_tokens, route_indices, route_valid)
        batch, sparse_len = input_tokens.shape
        class_h = self._class_tokens(labels, force_drop_ids)
        one_d_h = self._one_d_embeddings(completed_1d)
        safe_indices = route_indices.clamp(0, self.grid_tokens - 1)
        sparse_h = (
            self.embedding_2d(input_tokens)
            + self.pos_embedding_2d(safe_indices)
            + self.modality_embedding.weight[1][None, None]
        )
        counts = route_valid.sum(dim=1)
        if not bool(torch.all((counts == 64) | (counts == 128))):
            raise ValueError("E117 sparse stage requires K in {64,128}")
        budget_ids = torch.where(counts == 64, 1, 2)
        sparse_h = sparse_h + self.budget_embedding(budget_ids)[:, None]
        sparse_h = self.token_dropout(sparse_h)
        hidden = torch.cat((class_h, one_d_h, sparse_h), dim=1)
        prefix_valid = torch.ones(
            (batch, 1 + self.titok_num_tokens), dtype=torch.bool, device=hidden.device
        )
        valid = torch.cat((prefix_valid, route_valid), dim=1)

        if self.backbone_type == "llama":
            base_freqs = self.freqs_cis.to(hidden.device)
            class_freqs = base_freqs[:1][None].expand(batch, -1, -1, -1)
            one_d_freqs = base_freqs[1 : 1 + self.titok_num_tokens][None].expand(batch, -1, -1, -1)
            sparse_freqs = base_freqs[1:][safe_indices]
            freqs = torch.cat((class_freqs, one_d_freqs, sparse_freqs), dim=1)
        else:
            freqs = None
        hidden = self._run_backbone(hidden, freqs, valid)
        sparse_hidden = hidden[:, 1 + self.titok_num_tokens : 1 + self.titok_num_tokens + sparse_len]
        return self.output_2d(sparse_hidden).float()

    def forward(self, stage: str, **kwargs) -> torch.Tensor:
        """Dispatch through ``nn.Module.__call__`` so DDP hooks remain active."""

        if stage == "1d":
            return self.forward_1d(**kwargs)
        if stage == "2d":
            return self.forward_2d(**kwargs)
        raise ValueError(f"unknown stage: {stage!r}")

    @staticmethod
    def _cfg_logits(
        conditional: torch.Tensor,
        unconditional: torch.Tensor,
        scale: float,
        formula: str = "standard",
    ) -> torch.Tensor:
        if formula == "standard":
            return unconditional + float(scale) * (conditional - unconditional)
        if formula == "titok":
            # This deliberately mirrors ImageBert.generate in the TiTok repo.
            return conditional + float(scale) * (conditional - unconditional)
        raise ValueError("cfg_formula must be 'standard' or 'titok'")

    @torch.no_grad()
    def _iterative_sample(
        self,
        tokens: torch.Tensor,
        valid: torch.Tensor,
        mask_token_id: int,
        num_steps: int,
        randomize_temperature: float,
        forward_logits: Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor],
        cfg_scale: float,
        guidance_decay: str = "constant",
        cfg_formula: str = "standard",
        softmax_temperature_annealing: bool = False,
    ) -> torch.Tensor:
        if num_steps < 1:
            raise ValueError("num_steps must be positive")
        if guidance_decay not in ("constant", "linear"):
            raise ValueError("guidance_decay must be 'constant' or 'linear'")
        if bool(torch.any(tokens[valid] != mask_token_id)):
            raise ValueError("iterative sampling must start fully masked")
        valid_counts = valid.sum(dim=1)
        for step in range(num_steps):
            ratio = float(step + 1) / float(num_steps)
            annealed_temp = float(randomize_temperature) * (1.0 - ratio)
            currently_masked = (tokens == mask_token_id) & valid
            cond_logits = forward_logits(tokens, torch.zeros(tokens.shape[0], device=tokens.device, dtype=torch.long))
            step_cfg_scale = (
                float(cfg_scale) * float(step) / float(num_steps)
                if guidance_decay == "linear"
                else float(cfg_scale)
            )
            no_guidance_scale = 1.0 if cfg_formula == "standard" else 0.0
            if step_cfg_scale != no_guidance_scale:
                uncond_logits = forward_logits(tokens, torch.ones(tokens.shape[0], device=tokens.device, dtype=torch.long))
                logits = self._cfg_logits(cond_logits, uncond_logits, step_cfg_scale, cfg_formula)
            else:
                logits = cond_logits
            if softmax_temperature_annealing:
                logits = logits / (0.5 + 0.8 * (1.0 - ratio))

            uniform = torch.rand_like(logits).clamp_(1e-6, 1.0 - 1e-6)
            gumbel = -torch.log(-torch.log(uniform))
            sampled = (logits + annealed_temp * gumbel).argmax(dim=-1)
            sampled = torch.where(currently_masked, sampled, tokens)
            safe_sampled = sampled.masked_fill(~valid, 0)
            sampled_confidence = logits.gather(-1, safe_sampled[..., None]).squeeze(-1).float()
            confidence_noise = torch.rand_like(sampled_confidence).clamp_(1e-6, 1.0 - 1e-6)
            confidence_noise = -torch.log(-torch.log(confidence_noise))
            confidence = sampled_confidence + annealed_temp * confidence_noise
            confidence = confidence.masked_fill(~currently_masked, float("inf"))

            if step == num_steps - 1:
                tokens = sampled
                continue
            remain_ratio = math.acos(ratio) / (math.pi * 0.5)
            target_counts = torch.floor(valid_counts.float() * remain_ratio).long()
            current_counts = currently_masked.sum(dim=1)
            target_counts = torch.minimum(target_counts, (current_counts - 1).clamp_min(0))
            remask = _mask_from_counts(valid, target_counts, confidence)
            tokens = torch.where(remask, mask_token_id, sampled)

        if bool(torch.any(tokens[valid] == mask_token_id)):
            raise RuntimeError("MaskGIT sampling left valid mask tokens behind")
        return tokens

    @torch.no_grad()
    def generate_1d(
        self,
        labels: torch.Tensor,
        num_steps: int = 8,
        cfg_scale: float = 4.5,
        randomize_temperature: float = 9.5,
        guidance_decay: str = "linear",
        cfg_formula: str = "titok",
        softmax_temperature_annealing: bool = True,
    ) -> torch.Tensor:
        tokens = torch.full(
            (labels.shape[0], self.titok_num_tokens),
            self.mask_token_1d,
            dtype=torch.long,
            device=labels.device,
        )
        valid = torch.ones_like(tokens, dtype=torch.bool)
        return self._iterative_sample(
            tokens=tokens,
            valid=valid,
            mask_token_id=self.mask_token_1d,
            num_steps=num_steps,
            randomize_temperature=randomize_temperature,
            forward_logits=lambda ids, force: self.forward_1d(ids, labels, force),
            cfg_scale=cfg_scale,
            guidance_decay=guidance_decay,
            cfg_formula=cfg_formula,
            softmax_temperature_annealing=softmax_temperature_annealing,
        )

    @torch.no_grad()
    def generate_2d(
        self,
        completed_1d: torch.Tensor,
        route_indices: torch.Tensor,
        route_valid: torch.Tensor,
        labels: torch.Tensor,
        num_steps: int = 8,
        cfg_scale: float = 4.5,
        randomize_temperature: float = 1.0,
        guidance_decay: str = "constant",
        cfg_formula: str = "standard",
        softmax_temperature_annealing: bool = False,
    ) -> torch.Tensor:
        tokens = torch.full_like(route_indices, self.mask_token_2d)
        tokens = torch.where(route_valid, tokens, torch.full_like(tokens, self.pad_token_2d))
        return self._iterative_sample(
            tokens=tokens,
            valid=route_valid,
            mask_token_id=self.mask_token_2d,
            num_steps=num_steps,
            randomize_temperature=randomize_temperature,
            forward_logits=lambda ids, force: self.forward_2d(
                completed_1d, ids, route_indices, route_valid, labels, force
            ),
            cfg_scale=cfg_scale,
            guidance_decay=guidance_decay,
            cfg_formula=cfg_formula,
            softmax_temperature_annealing=softmax_temperature_annealing,
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
