#!/usr/bin/env python3
"""Adapt LlamaGen decoder with an f2d-aware end-to-end dynamic-budget STE router."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets.folder import IMG_EXTENSIONS, default_loader
from tqdm import tqdm

from models import TiTokLlamaGenStage2
from train_titok_llamagen_recon import (
    TrainableEMA,
    adamw_param_groups,
    autocast_dtype,
    build_discriminator,
    build_dino_feature_loss,
    build_clip_feature_loss,
    build_perceptual_loss,
    chw_to_pil,
    compute_feature_moment_loss,
    compute_image_loss,
    compute_lecam_loss,
    compute_lr,
    compute_perceptual_loss,
    discriminator_feature_matching_loss,
    convert_image_range,
    denorm_to_uint8,
    discriminator_input,
    gan_factor_for_step,
    gan_g_loss_from_logits,
    get_perceptual_weight,
    hinge_d_loss,
    image_to_zero_one,
    load_llamagen_vq,
    load_titok,
    make_transform,
    split_discriminator_logits,
    set_optimizer_lr,
    set_requires_grad,
    weighted_logits_mean,
)


def _lowpass_upsample_01(x, low_size):
    low_size = int(low_size)
    if low_size <= 0 or (x.shape[-2] == low_size and x.shape[-1] == low_size):
        return x.float()
    low = F.interpolate(x.float(), size=(low_size, low_size), mode="bilinear", align_corners=False, antialias=True)
    return F.interpolate(low, size=x.shape[-2:], mode="bilinear", align_corners=False, antialias=True)


def prepare_gan_inputs(fake, real, image_range, args, for_generator=False):
    fake_01 = discriminator_input(fake, image_range)
    real_01 = discriminator_input(real, image_range)
    if args.gan_input_filter == "none":
        return fake_01, real_01
    if args.gan_input_filter == "highfreq_composite":
        low_real = _lowpass_upsample_01(real_01, args.gan_highpass_size)
        low_fake = _lowpass_upsample_01(fake_01, args.gan_highpass_size)
        fake_hf = fake_01 - low_fake
        fake_composite = (low_real.detach() + fake_hf).clamp(0.0, 1.0)
        return fake_composite, real_01
    if args.gan_input_filter == "highfreq_grad_only":
        if for_generator:
            low_fake = _lowpass_upsample_01(fake_01, args.gan_highpass_size)
            fake_01 = fake_01 + low_fake.detach() - low_fake
        return fake_01, real_01
    raise ValueError(f"unsupported gan_input_filter {args.gan_input_filter}")


def compute_lowfreq_anchor_loss(pred, target, image_range, low_size):
    pred_01 = discriminator_input(pred, image_range)
    target_01 = discriminator_input(target, image_range)
    low_size = int(low_size)
    if low_size > 0:
        pred_01 = F.interpolate(pred_01.float(), size=(low_size, low_size), mode="bilinear", align_corners=False, antialias=True)
        target_01 = F.interpolate(target_01.float(), size=(low_size, low_size), mode="bilinear", align_corners=False, antialias=True)
    return F.l1_loss(pred_01.float(), target_01.float())

def gan_g_factor_for_step(args, step, d_warmup_end_step):
    if not getattr(args, "gan_g_ramp_after_d_warmup", False):
        return gan_factor_for_step(args, step)
    if args.lambda_gan <= 0.0:
        return 0.0
    if step <= d_warmup_end_step:
        return 0.0
    if args.gan_ramp_steps <= 0:
        return args.discriminator_factor
    ramp = min(1.0, max(0.0, float(step - d_warmup_end_step) / float(args.gan_ramp_steps)))
    return args.discriminator_factor * ramp


def format_run_header(args, dataset_len, world_size, trainable_params):
    effective_batch = args.batch_size * world_size * args.accum_steps
    summary = (
        f"dataset={dataset_len} world_size={world_size} batch_size_per_gpu={args.batch_size} "
        f"accum_steps={args.accum_steps} global_batch={effective_batch} trainable_params={trainable_params} "
        f"train_adapter={args.train_adapter} train_llamagen_encoder={args.train_llamagen_encoder} "
        f"train_llamagen_quant_conv={args.train_llamagen_quant_conv} train_llamagen_quantizer={args.train_llamagen_quantizer} "
        f"train_post_quant_conv={args.train_post_quant_conv} train_llamagen_decoder={args.train_llamagen_decoder} "
        f"llamagen_decoder_train_last_n={args.llamagen_decoder_train_last_n} "
        f"lr_adapter={args.lr} lr_lg_encoder={args.lr_llamagen_encoder} lr_llamagen={args.lr_llamagen} "
        f"loss=image({args.image_loss}) base:{args.lambda_base},mix:{args.lambda_mix},native:{args.lambda_native},"
        f"mix_native:{args.lambda_mix_native}@{args.lambda_mix_native_perceptual}/{args.mix_native_teacher},"
        f"perceptual({args.perceptual_loss}):{get_perceptual_weight(args)},feat:{args.lambda_feat},"
        f"feat_moment:{args.lambda_feat_moment},dino_feat:{args.lambda_dino_feat}/{args.dino_feat_loss},"
        f"clip_feat:{args.lambda_clip_feat}/{args.clip_model}/{args.clip_pretrained}/{args.clip_feat_loss},"
        f"disc_fm:{args.lambda_disc_feature_matching},lowfreq_anchor:{args.lambda_lowfreq_anchor}@{args.lowfreq_anchor_size},"
        f"gan:{args.lambda_gan}@{args.gan_start_step}+ramp{args.gan_ramp_steps}/"
        f"g_after_dwarm:{getattr(args, 'gan_g_ramp_after_d_warmup', False)}/{args.gan_input_filter}@{args.gan_highpass_size},"
        f"disc:{args.discriminator_type}/scales={args.disc_scales}/weights={args.disc_loss_weights},"
        f"dino:{args.dino_model}@{args.dino_loss_weight},"
        f"d_every:{args.d_every},d_warmup:{args.d_warmup_steps},g_freeze:{args.g_freeze_steps},lecam:{args.lecam_regularization_weight} "
        f"mask_selection:{args.mask_selection} mask_ratio:{args.mask_ratio} "
        f"mask_score_weights:error={args.mask_error_weight},gradient={args.mask_gradient_weight},"
        f"variance={args.mask_variance_weight} "
        f"selector_threshold={args.selector_threshold},target_mask_ratio={args.target_mask_ratio},"
        f"threshold_controller={args.use_threshold_ema_controller},"
        f"selector_lr={args.selector_threshold_lr},selector_ema={args.selector_ratio_ema_decay},"
        f"score_normalize_scope={args.score_normalize_scope} "
        f"mask_ratio_range:{args.mask_ratio_min}-{args.mask_ratio_max} "
        f"augment=random_crop:{args.random_crop},random_flip:{args.random_flip} "
        f"ema:{args.use_ema}@{args.ema_decay} adapter_init={args.adapter_init} "
        f"router_only_steps:{args.router_only_steps},router_only_disable_gan:{args.router_only_disable_gan} "
        f"router_ratio_target:lambda={args.lambda_router_ratio_target},spread={args.router_ratio_target_spread}"
    )
    resolved_args = {key: value for key, value in sorted(vars(args).items())}
    lines = [
        "",
        "=" * 80,
        f"run_start={datetime.now().isoformat(timespec='seconds')}",
        f"cwd={Path.cwd()}",
        f"config_file={args.config}",
        summary,
        "resolved_args_json=",
        json.dumps(resolved_args, indent=2, sort_keys=True, default=str),
        "=" * 80,
    ]
    return summary, "\n".join(lines) + "\n"


def distributed_setup():
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rank = 0
        world_size = 1
        local_rank = 0
    return distributed, rank, world_size, local_rank, device


def cycle(loader, sampler=None):
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


def set_module_trainable(module: nn.Module | None, requires_grad: bool):
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad_(requires_grad)


def set_llamagen_decoder_trainable(decoder: nn.Module | None, train_decoder: bool, train_last_n: int):
    if decoder is None:
        return
    set_module_trainable(decoder, False)
    if not train_decoder or train_last_n == 0:
        return
    if train_last_n < 0 or not hasattr(decoder, "conv_blocks"):
        set_module_trainable(decoder, True)
        return

    conv_blocks = getattr(decoder, "conv_blocks")
    for block in list(conv_blocks)[-train_last_n:]:
        set_module_trainable(block, True)
    set_module_trainable(getattr(decoder, "conv_out", None), True)


def configure_trainable_parts(model: TiTokLlamaGenStage2, args):
    model.titok.eval().requires_grad_(False)
    model.latent_decoder.requires_grad_(args.train_adapter)
    model.llamagen_vq.eval().requires_grad_(False)
    set_module_trainable(getattr(model.llamagen_vq, "encoder", None), args.train_llamagen_encoder)
    set_module_trainable(getattr(model.llamagen_vq, "quantize", None), args.train_llamagen_quantizer)
    set_module_trainable(getattr(model.llamagen_vq, "quant_conv", None), args.train_llamagen_quant_conv)
    set_module_trainable(getattr(model.llamagen_vq, "post_quant_conv", None), args.train_post_quant_conv)
    set_llamagen_decoder_trainable(
        getattr(model.llamagen_vq, "decoder", None),
        args.train_llamagen_decoder,
        args.llamagen_decoder_train_last_n,
    )
    set_module_trainable(getattr(model, "router", None), args.train_router)
    return model


def set_adapt_train_mode(model, args):
    core = model.module if isinstance(model, DDP) else model
    core.titok.eval()
    core.latent_decoder.train(args.train_adapter)
    core.llamagen_vq.eval()
    if args.train_llamagen_encoder and hasattr(core.llamagen_vq, "encoder"):
        core.llamagen_vq.encoder.train()
    if args.train_llamagen_quantizer and hasattr(core.llamagen_vq, "quantize"):
        core.llamagen_vq.quantize.train()
    if args.train_llamagen_quant_conv and hasattr(core.llamagen_vq, "quant_conv"):
        core.llamagen_vq.quant_conv.train()
    if args.train_post_quant_conv and hasattr(core.llamagen_vq, "post_quant_conv"):
        core.llamagen_vq.post_quant_conv.train()
    if args.train_llamagen_decoder and hasattr(core.llamagen_vq, "decoder"):
        core.llamagen_vq.decoder.train()
    if hasattr(core, "router"):
        core.router.train(args.train_router)


def apply_router_only_trainability(core: TiTokLlamaGenStage2, args, router_only_active: bool):
    if not router_only_active:
        configure_trainable_parts(core, args)
        return
    core.titok.eval().requires_grad_(False)
    set_module_trainable(core.latent_decoder, False)
    core.llamagen_vq.eval()
    set_module_trainable(getattr(core.llamagen_vq, "encoder", None), False)
    set_module_trainable(getattr(core.llamagen_vq, "quantize", None), False)
    set_module_trainable(getattr(core.llamagen_vq, "quant_conv", None), False)
    set_module_trainable(getattr(core.llamagen_vq, "post_quant_conv", None), False)
    set_module_trainable(getattr(core.llamagen_vq, "decoder", None), False)
    set_module_trainable(getattr(core, "router", None), args.train_router)


def sync_module_grads(module: nn.Module | None, world_size: int):
    if module is None or world_size <= 1:
        return
    for param in module.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad.div_(world_size)


def trainable_param_groups(core: TiTokLlamaGenStage2, args):
    groups = []
    if args.train_adapter:
        for group in adamw_param_groups(core.latent_decoder, args.weight_decay):
            group["lr"] = args.lr
            group["lr_role"] = "adapter"
            groups.append(group)
    if args.train_llamagen_encoder and hasattr(core.llamagen_vq, "encoder"):
        for group in adamw_param_groups(core.llamagen_vq.encoder, args.weight_decay_llamagen):
            group["lr"] = args.lr_llamagen_encoder
            group["lr_role"] = "llamagen_encoder"
            groups.append(group)
    if args.train_llamagen_quant_conv and hasattr(core.llamagen_vq, "quant_conv"):
        for group in adamw_param_groups(core.llamagen_vq.quant_conv, args.weight_decay_llamagen):
            group["lr"] = args.lr_llamagen_encoder
            group["lr_role"] = "llamagen_encoder"
            groups.append(group)
    if args.train_llamagen_quantizer and hasattr(core.llamagen_vq, "quantize"):
        for group in adamw_param_groups(core.llamagen_vq.quantize, args.weight_decay_llamagen):
            group["lr"] = args.lr_llamagen_encoder
            group["lr_role"] = "llamagen_encoder"
            groups.append(group)
    if args.train_post_quant_conv and hasattr(core.llamagen_vq, "post_quant_conv"):
        for group in adamw_param_groups(core.llamagen_vq.post_quant_conv, args.weight_decay_llamagen):
            group["lr"] = args.lr_llamagen
            group["lr_role"] = "llamagen"
            groups.append(group)
    if args.train_llamagen_decoder and hasattr(core.llamagen_vq, "decoder"):
        for group in adamw_param_groups(core.llamagen_vq.decoder, args.weight_decay_llamagen):
            group["lr"] = args.lr_llamagen
            group["lr_role"] = "llamagen"
            groups.append(group)
    if args.train_router and hasattr(core, "router"):
        for group in adamw_param_groups(core.router, args.weight_decay_router):
            group["lr"] = args.lr_router
            group["lr_role"] = "router"
            groups.append(group)
    if not groups:
        raise ValueError("no trainable parameter groups; enable at least one train_* flag")
    return groups


def trainable_params(core: TiTokLlamaGenStage2):
    return [param for param in core.parameters() if param.requires_grad]


def load_discriminator_resume(discriminator, state_dict, strict=True):
    if discriminator is None:
        return "none", [], []
    wrapper_prefixes = (
        "discriminators.",
        "patch_discriminator.",
        "dino_discriminator.",
        "imagenet_mean",
        "imagenet_std",
    )
    is_wrapped_state = any(str(key).startswith(wrapper_prefixes) for key in state_dict)
    if hasattr(discriminator, "load_primary_state_dict") and not is_wrapped_state:
        missing, unexpected = discriminator.load_primary_state_dict(state_dict, strict=strict)
        return getattr(discriminator, "primary_load_name", "primary_scale"), missing, unexpected
    missing, unexpected = discriminator.load_state_dict(state_dict, strict=strict)
    return "full", missing, unexpected


def native_llamagen_feature(vq_model, x_lg, codebook_embed_dim, allow_encoder_grad=False):
    if allow_encoder_grad:
        quant, _, info = vq_model.encode(x_lg)
    else:
        with torch.no_grad():
            quant, _, info = vq_model.encode(x_lg)
            quant = quant.detach()
    code_indices = None
    if info is not None and len(info) >= 3 and info[2] is not None:
        code_indices = info[2].view(x_lg.shape[0], quant.shape[-2], quant.shape[-1]).long()
    if quant.ndim != 4 or quant.shape[-2:] != (16, 16):
        raise ValueError(f"unexpected LlamaGen encode output shape: {tuple(quant.shape)}")
    if quant.shape[1] == codebook_embed_dim:
        return vq_model.post_quant_conv(quant), code_indices
    if quant.shape[1] == 256:
        return quant, code_indices
    raise ValueError(f"unexpected LlamaGen encode channels {tuple(quant.shape)}")


@torch.no_grad()
def spatial_error_score(x_base, target):
    err = (x_base.detach().float() - target.detach().float()).abs().mean(dim=1, keepdim=True)
    return F.adaptive_avg_pool2d(err, (16, 16)).flatten(1)


@torch.no_grad()
def spatial_error_score_grid(x_base, target, grid_hw=16):
    err = (x_base.detach().float() - target.detach().float()).abs().mean(dim=1, keepdim=True)
    return F.adaptive_avg_pool2d(err, (grid_hw, grid_hw))


@torch.no_grad()
def gradient_score_from_image_grid(x, grid_hw=16):
    gray = x.detach().float().mean(dim=1, keepdim=True)
    dx = F.pad((gray[:, :, :, 1:] - gray[:, :, :, :-1]).abs(), (0, 1, 0, 0))
    dy = F.pad((gray[:, :, 1:, :] - gray[:, :, :-1, :]).abs(), (0, 0, 0, 1))
    return F.adaptive_avg_pool2d(dx + dy, (grid_hw, grid_hw))


@torch.no_grad()
def local_variance_score_from_image_grid(x, grid_hw=16):
    xf = x.detach().float()
    mean = F.avg_pool2d(xf, kernel_size=7, stride=1, padding=3)
    mean_sq = F.avg_pool2d(xf * xf, kernel_size=7, stride=1, padding=3)
    var = (mean_sq - mean * mean).clamp_min(0.0).mean(dim=1, keepdim=True)
    return F.adaptive_avg_pool2d(var, (grid_hw, grid_hw))


@torch.no_grad()
def normalize_score_global_pool(score, eps=1e-6):
    score = score.detach().float()
    min_val = score.amin()
    max_val = score.amax()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(min_val, op=dist.ReduceOp.MIN)
        dist.all_reduce(max_val, op=dist.ReduceOp.MAX)
    return (score - min_val) / (max_val - min_val + eps)


@torch.no_grad()
def normalize_score_per_image(score, eps=1e-6):
    score = score.detach().float()
    min_val = score.amin(dim=(2, 3), keepdim=True)
    max_val = score.amax(dim=(2, 3), keepdim=True)
    return (score - min_val) / (max_val - min_val + eps)


@torch.no_grad()
def global_mean_value(value):
    value = value.detach().float()
    stats = torch.stack([value.sum(), value.new_tensor(float(value.numel()))])
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    return float((stats[0] / stats[1].clamp_min(1.0)).item())


@torch.no_grad()
def normalize_score_parts(raw_parts, scope):
    if scope == "global_pool":
        normalize = normalize_score_global_pool
    elif scope == "per_image":
        normalize = normalize_score_per_image
    else:
        raise ValueError(f"unknown score_normalize_scope: {scope}")
    return {
        "error": normalize(raw_parts["error"]),
        "gradient": normalize(raw_parts["gradient"]),
        "variance": normalize(raw_parts["variance"]),
    }


@torch.no_grad()
def mixed_score_from_normalized_parts(norm_parts, args):
    return (
        float(args.mask_error_weight) * norm_parts["error"]
        + float(args.mask_gradient_weight) * norm_parts["gradient"]
        + float(args.mask_variance_weight) * norm_parts["variance"]
    )


@torch.no_grad()
def mixed_score_from_raw_global_pool(raw_parts, args):
    norm_parts = normalize_score_parts(raw_parts, "global_pool")
    score = mixed_score_from_normalized_parts(norm_parts, args)
    return score, norm_parts


@torch.no_grad()
def mixed_score_from_raw_parts(raw_parts, args, scope):
    norm_parts = normalize_score_parts(raw_parts, scope)
    score = mixed_score_from_normalized_parts(norm_parts, args)
    return score, norm_parts


@torch.no_grad()
def oracle_error_mask(x_base, target, ratio_min, ratio_max):
    err_16 = spatial_error_score(x_base, target)
    ratios = torch.empty(err_16.shape[0], device=err_16.device).uniform_(ratio_min, ratio_max)
    masks = []
    for sample_err, ratio in zip(err_16, ratios):
        k = int(round(float(ratio) * sample_err.numel()))
        flat = torch.zeros_like(sample_err)
        if k > 0:
            topk = torch.topk(sample_err, k).indices
            flat[topk] = 1.0
        masks.append(flat.view(1, 16, 16))
    return torch.stack(masks, dim=0)


@torch.no_grad()
def global_top_ratio_mask_from_score(score, ratio):
    local_flat = score.detach().float().flatten()
    if dist.is_available() and dist.is_initialized():
        gathered = [torch.empty_like(local_flat) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, local_flat)
        all_flat = torch.cat(gathered, dim=0)
        local_start = dist.get_rank() * local_flat.numel()
    else:
        all_flat = local_flat
        local_start = 0

    k = int(round(float(ratio) * all_flat.numel()))
    k = max(0, min(k, all_flat.numel()))
    all_mask = torch.zeros_like(all_flat)
    if k > 0:
        all_mask[torch.topk(all_flat, k, largest=True).indices] = 1.0
    local_mask = all_mask[local_start:local_start + local_flat.numel()]
    return local_mask.view_as(score).view(score.shape[0], 1, 16, 16)


@torch.no_grad()
def threshold_mask_from_score(score, selector_threshold):
    return (score.detach().float() > float(selector_threshold)).to(dtype=torch.float32)


class DynamicBudgetRouter(nn.Module):
    def __init__(
        self,
        latent_channels=256,
        hidden_dim=128,
        depth=3,
        target_ratio=0.5,
        min_ratio=0.05,
        max_ratio=0.9,
        detach_inputs=True,
    ):
        super().__init__()
        self.detach_inputs = detach_inputs
        self.feat_proj = nn.Conv2d(latent_channels, hidden_dim, kernel_size=1)
        self.f2d_proj = nn.Conv2d(latent_channels, hidden_dim, kernel_size=1)
        self.delta_proj = nn.Conv2d(latent_channels, hidden_dim, kernel_size=1)
        self.abs_delta_proj = nn.Conv2d(latent_channels, hidden_dim, kernel_size=1)
        self.base_proj = nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, hidden_dim, 16, 16))
        blocks = []
        for _ in range(depth):
            blocks.extend([
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GroupNorm(8, hidden_dim),
                nn.SiLU(inplace=True),
            ])
        self.trunk = nn.Sequential(*blocks)
        self.score_head = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        self.ratio_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.normal_(self.score_head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.score_head.bias)
        nn.init.zeros_(self.f2d_proj.weight)
        nn.init.zeros_(self.f2d_proj.bias)
        nn.init.zeros_(self.delta_proj.weight)
        nn.init.zeros_(self.delta_proj.bias)
        nn.init.zeros_(self.abs_delta_proj.weight)
        nn.init.zeros_(self.abs_delta_proj.bias)
        ratio01 = (float(target_ratio) - float(min_ratio)) / max(float(max_ratio) - float(min_ratio), 1e-6)
        ratio01 = min(max(ratio01, 1e-4), 1.0 - 1e-4)
        nn.init.zeros_(self.ratio_head[-1].weight)
        nn.init.constant_(self.ratio_head[-1].bias, math.log(ratio01 / (1.0 - ratio01)))

    def forward(self, f_1d, x_base, f_2d=None):
        if f_2d is None:
            f_2d = torch.zeros_like(f_1d)
        if self.detach_inputs:
            f_1d = f_1d.detach()
            x_base = x_base.detach()
            f_2d = f_2d.detach()
        x_low = F.adaptive_avg_pool2d(x_base.float(), (16, 16)).to(dtype=f_1d.dtype)
        delta = f_2d - f_1d
        h = (
            self.feat_proj(f_1d)
            + self.f2d_proj(f_2d)
            + self.delta_proj(delta)
            + self.abs_delta_proj(delta.abs())
            + self.base_proj(x_low)
            + self.pos_embed.to(dtype=f_1d.dtype)
        )
        h = self.trunk(h)
        return self.score_head(h), self.ratio_head(h)


def calibrate_ratio_logits_to_target_mean(ratio_logits, args):
    if not bool(args.router_ratio_target_mean_calibrate):
        return ratio_logits
    target = float(args.router_target_mean_ratio)
    min_ratio = float(args.router_min_ratio)
    max_ratio = float(args.router_max_ratio)
    target = min(max(target, min_ratio), max_ratio)
    if max_ratio <= min_ratio:
        return ratio_logits

    logits = ratio_logits.float()
    lo = logits.new_full((), -30.0)
    hi = logits.new_full((), 30.0)
    for _ in range(max(int(args.router_ratio_target_calibrate_iters), 1)):
        mid = (lo + hi) * 0.5
        ratio = min_ratio + (max_ratio - min_ratio) * torch.sigmoid(logits + mid)
        if float(ratio.mean().item()) < target:
            lo = mid
        else:
            hi = mid
    return ratio_logits + ((lo + hi) * 0.5).to(dtype=ratio_logits.dtype)


def adjust_token_counts_to_total(k, ratio_soft, min_tokens, max_tokens, target_total):
    k = k.clone()
    bsz = k.numel()
    if bsz == 0:
        return k
    target_total = int(round(float(target_total)))
    target_total = max(int(min_tokens) * bsz, min(int(max_tokens) * bsz, target_total))
    diff = target_total - int(k.sum().item())
    if diff == 0:
        return k

    order = torch.argsort(ratio_soft.detach().float().flatten(), descending=diff > 0)
    while diff != 0:
        changed = 0
        for idx in order.tolist():
            if diff > 0:
                if int(k[idx].item()) >= int(max_tokens):
                    continue
                k[idx] += 1
                diff -= 1
            else:
                if int(k[idx].item()) <= int(min_tokens):
                    continue
                k[idx] -= 1
                diff += 1
            changed += 1
            if diff == 0:
                break
        if changed == 0:
            break
    return k


def make_dynamic_budget_ste_mask(logits, ratio_logits, args, budget_debt_tokens=0.0):
    bsz, _channels, height, width = logits.shape
    tokens = height * width
    ratio01 = torch.sigmoid(ratio_logits.float()).flatten()
    ratio_soft = float(args.router_min_ratio) + (float(args.router_max_ratio) - float(args.router_min_ratio)) * ratio01
    min_tokens = args.router_min_tokens if args.router_min_tokens >= 0 else int(round(float(args.router_min_ratio) * tokens))
    max_tokens = args.router_max_tokens if args.router_max_tokens >= 0 else int(round(float(args.router_max_ratio) * tokens))
    min_tokens = max(0, min(int(min_tokens), tokens))
    max_tokens = max(min_tokens, min(int(max_tokens), tokens))
    k = torch.round(ratio_soft.detach() * tokens).to(dtype=torch.long).clamp(min=min_tokens, max=max_tokens)
    budget_mode = getattr(args, "router_budget_mode", "soft")
    if budget_mode in {"batch_exact", "bank"}:
        target_ratio = float(getattr(args, "router_exact_budget_ratio", -1.0))
        if target_ratio < 0.0:
            target_ratio = float(args.router_target_mean_ratio)
        per_step_target = float(bsz * tokens) * target_ratio
        if budget_mode == "batch_exact":
            target_total = per_step_target
        else:
            max_debt = max(0.0, float(getattr(args, "router_budget_bank_max_debt_tokens", 0.0)))
            raw_total = float(k.sum().item())
            projected_debt = float(budget_debt_tokens) + raw_total - per_step_target
            target_total = raw_total
            if projected_debt > max_debt:
                target_total -= projected_debt - max_debt
            elif projected_debt < -max_debt:
                target_total += -max_debt - projected_debt
        k = adjust_token_counts_to_total(k, ratio_soft, min_tokens, max_tokens, target_total)

    flat_logits = logits.flatten(1)
    hard_flat = torch.zeros_like(flat_logits, dtype=torch.float32)
    for idx in range(bsz):
        ki = int(k[idx].item())
        if ki > 0:
            hard_flat[idx, torch.topk(flat_logits[idx].float(), ki, largest=True).indices] = 1.0
    hard_mask = hard_flat.view_as(logits)

    logits_float = logits.float()
    logits_norm = (logits_float - logits_float.mean(dim=(2, 3), keepdim=True)) / (
        logits_float.std(dim=(2, 3), keepdim=True, unbiased=False) + 1e-6
    )
    ratio_for_bias = ratio_soft.clamp(1e-4, 1.0 - 1e-4).view(bsz, 1, 1, 1)
    ratio_bias = torch.log(ratio_for_bias / (1.0 - ratio_for_bias))
    soft_mask = torch.sigmoid((logits_norm + ratio_bias) / max(float(args.router_tau), 1e-4))
    mask = hard_mask + soft_mask - soft_mask.detach()
    return mask.to(dtype=logits.dtype), hard_mask, soft_mask, ratio_soft


def make_top_p_ste_mask(logits, args):
    """Build an exact hard top-p mask with a differentiable prefix surrogate."""
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError(f"top-p router logits must have shape [B,1,H,W], got {tuple(logits.shape)}")
    top_p = float(args.router_top_p)
    temperature = float(args.router_top_p_temperature)
    ste_tau = float(args.router_top_p_ste_tau)
    if not 0.0 < top_p <= 1.0:
        raise ValueError(f"router_top_p must be in (0,1], got {top_p}")
    if temperature <= 0.0:
        raise ValueError(f"router_top_p_temperature must be positive, got {temperature}")
    if ste_tau <= 0.0:
        raise ValueError(f"router_top_p_ste_tau must be positive, got {ste_tau}")

    flat_logits = logits.float().flatten(1)
    probabilities = torch.softmax(flat_logits / temperature, dim=1)
    sorted_probabilities, sorted_indices = torch.sort(probabilities, dim=1, descending=True)
    cumulative_before = torch.cumsum(sorted_probabilities, dim=1) - sorted_probabilities

    # c_{k-1} < p selects the minimal prefix ending at the first c_k >= p.
    hard_sorted = (cumulative_before < top_p).to(dtype=torch.float32)
    soft_sorted = torch.sigmoid((top_p - cumulative_before) / ste_tau)

    hard_flat = torch.zeros_like(probabilities)
    soft_flat = torch.zeros_like(probabilities)
    hard_flat.scatter_(1, sorted_indices, hard_sorted)
    soft_flat.scatter_(1, sorted_indices, soft_sorted)
    hard_mask = hard_flat.view_as(logits)
    soft_mask = soft_flat.view_as(logits)
    mask = hard_mask + soft_mask - soft_mask.detach()
    soft_ratio = soft_flat.mean(dim=1)
    return mask.to(dtype=logits.dtype), hard_mask, soft_mask, soft_ratio


@torch.no_grad()
def grid_mse_gain_score(x_base, x_native, target, grid_hw=16):
    base_err = (x_base.detach().float() - target.detach().float()).pow(2).mean(dim=1, keepdim=True)
    native_err = (x_native.detach().float() - target.detach().float()).pow(2).mean(dim=1, keepdim=True)
    return F.adaptive_avg_pool2d(base_err - native_err, (grid_hw, grid_hw)).clamp_min(0.0)


@torch.no_grad()
def per_image_top_ratio_mask_from_score(score, ratio):
    flat_score = score.detach().float().flatten(1)
    tokens = flat_score.shape[1]
    k = max(0, min(tokens, int(round(float(ratio) * tokens))))
    flat_mask = torch.zeros_like(flat_score)
    if k > 0:
        flat_mask.scatter_(1, torch.topk(flat_score, k, dim=1, largest=True).indices, 1.0)
    return flat_mask.view_as(score)


@torch.no_grad()
def per_image_variable_top_ratio_mask_from_score(score, ratios):
    flat_score = score.detach().float().flatten(1)
    ratios = ratios.detach().float().flatten()
    bsz, tokens = flat_score.shape
    if ratios.numel() != bsz:
        raise ValueError(f"ratios must have shape ({bsz},), got {tuple(ratios.shape)}")
    flat_mask = torch.zeros_like(flat_score)
    for idx in range(bsz):
        k = max(0, min(tokens, int(round(float(ratios[idx].item()) * tokens))))
        if k > 0:
            flat_mask[idx].scatter_(0, torch.topk(flat_score[idx], k, dim=0, largest=True).indices, 1.0)
    return flat_mask.view_as(score)


@torch.no_grad()
def image_ratio_target_from_gain_score(gain_score, args):
    local_score = gain_score.detach().float().flatten(1).mean(dim=1)
    bsz = local_score.shape[0]
    target = float(args.router_target_mean_ratio)
    spread = float(args.router_ratio_target_spread)
    if bsz == 0 or spread <= 0.0:
        return local_score.new_full((bsz,), target)

    if dist.is_available() and dist.is_initialized():
        gathered = [torch.empty_like(local_score) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, local_score)
        all_score = torch.cat(gathered, dim=0)
        local_start = dist.get_rank() * local_score.numel()
    else:
        all_score = local_score
        local_start = 0

    if all_score.numel() <= 1:
        all_target = all_score.new_full(all_score.shape, target)
    else:
        order = torch.argsort(all_score)
        ranks = torch.empty_like(all_score)
        ranks[order] = torch.linspace(0.0, 1.0, all_score.numel(), device=all_score.device, dtype=all_score.dtype)
        centered = ranks - ranks.mean()
        all_target = target + 2.0 * spread * centered
        min_ratio = max(0.0, float(args.router_min_ratio))
        max_ratio = min(1.0, float(args.router_max_ratio))
        all_target = all_target.clamp(min=min_ratio, max=max_ratio)
        all_target = (all_target + (target - all_target.mean())).clamp(min=min_ratio, max=max_ratio)

    return all_target[local_start:local_start + bsz].to(device=gain_score.device)


def save_adapt_grid(path, target, x_base, x_mix, x_native, max_images=8, image_range="minus1_1"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = [
        denorm_to_uint8(target[:max_images], image_range),
        denorm_to_uint8(x_base[:max_images], image_range),
        denorm_to_uint8(x_mix[:max_images], image_range),
        denorm_to_uint8(x_native[:max_images], image_range),
    ]
    headers = ["target", "base", "mix", "native"]
    n = tensors[0].shape[0]
    cell = tensors[0].shape[-1]
    label_h = 18
    canvas = Image.new("RGB", (len(tensors) * cell, n * (cell + label_h)), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for i in range(n):
        for j, batch in enumerate(tensors):
            x0 = j * cell
            y0 = i * (cell + label_h)
            draw.text((x0 + 4, y0 + 2), headers[j], fill=(0, 0, 0))
            canvas.paste(chw_to_pil(batch[i]), (x0, y0 + label_h))
    canvas.save(path)


def load_adapter_init(model: TiTokLlamaGenStage2, ckpt_path: str, use_ema: bool, strict: bool = True):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if use_ema and isinstance(ckpt, dict) and "model_ema" in ckpt:
        state = ckpt["model_ema"]
        key = "model_ema"
    else:
        state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        key = "model"
    if isinstance(state, dict) and any(k.startswith("latent_decoder.") for k in state):
        state = {k[len("latent_decoder."):]: v for k, v in state.items() if k.startswith("latent_decoder.")}
    model.load_trainable_state_dict(state, strict=strict)
    step = int(ckpt.get("step", -1)) if isinstance(ckpt, dict) else -1
    return key, step


def collect_trainable_state(core: TiTokLlamaGenStage2):
    return {
        name: param.detach().cpu().clone()
        for name, param in core.named_parameters()
        if param.requires_grad
    }


def parse_prefix_list(value):
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    if isinstance(value, (list, tuple)):
        return tuple(str(part).strip() for part in value if str(part).strip())
    raise TypeError(f"prefix list must be a string or sequence, got {type(value).__name__}")


def load_adapt_resume(core: TiTokLlamaGenStage2, ckpt, strict=True, skip_prefixes=(), use_ema_model=False):
    if use_ema_model:
        if not isinstance(ckpt, dict) or "model_ema" not in ckpt:
            raise ValueError("resume_use_ema_model requested but checkpoint has no model_ema")
        state = ckpt["model_ema"]
        fallback_state = None
    else:
        state = ckpt.get("model", ckpt)
        fallback_state = ckpt.get("model_ema") if isinstance(ckpt, dict) else None
    if not isinstance(state, dict):
        raise ValueError("resume checkpoint does not contain a state dict")
    if fallback_state is not None and not isinstance(fallback_state, dict):
        fallback_state = None
    skip_prefixes = parse_prefix_list(skip_prefixes)
    current = dict(core.named_parameters())
    missing = []
    unexpected = []
    skipped = []
    filled_from_ema = []
    loaded = set()
    with torch.no_grad():
        for name, value in state.items():
            if any(name == prefix or name.startswith(prefix + ".") for prefix in skip_prefixes):
                skipped.append(name)
                continue
            if name not in current:
                unexpected.append(name)
                continue
            current[name].copy_(value.to(device=current[name].device, dtype=current[name].dtype))
            loaded.add(name)
        if fallback_state is not None:
            for name, param in current.items():
                if name in loaded or name not in fallback_state:
                    continue
                if any(name == prefix or name.startswith(prefix + ".") for prefix in skip_prefixes):
                    continue
                value = fallback_state[name]
                param.copy_(value.to(device=param.device, dtype=param.dtype))
                loaded.add(name)
                filled_from_ema.append(name)
        if strict:
            for name, param in current.items():
                if param.requires_grad and name not in loaded and not any(name == prefix or name.startswith(prefix + ".") for prefix in skip_prefixes):
                    missing.append(name)
    if strict and (missing or unexpected):
        raise RuntimeError(f"resume mismatch missing={missing} unexpected={unexpected}")
    return missing, unexpected, skipped, filled_from_ema


def compute_path_losses(perceptual, perceptual_name, pred, target, image_loss):
    image = compute_image_loss(pred, target, image_loss)
    perc = compute_perceptual_loss(perceptual, perceptual_name, pred, target)
    mse01 = F.mse_loss(image_to_zero_one(pred, "minus1_1").float(), image_to_zero_one(target, "minus1_1").float())
    return image, perc, mse01



def epoch_fraction_to_steps(value, steps_per_epoch, *, add_one_after=False):
    value = float(value)
    if value <= 0.0:
        return 0
    steps = max(1, int(round(value * steps_per_epoch)))
    return steps + (1 if add_one_after else 0)


def optimizer_steps_per_epoch(loader_steps_per_epoch, accum_steps):
    return max(1, int(loader_steps_per_epoch) // max(1, int(accum_steps)))


def compute_lr_for_scheduler(args, step, base_lr, scheduler, min_lr=0.0, start_step=0, max_steps=None):
    if scheduler == "constant":
        return base_lr
    if scheduler != "cosine":
        raise ValueError(f"unsupported lr scheduler {scheduler}")
    max_steps = args.max_steps if max_steps is None else max_steps
    rel_step = max(step - start_step, 0)
    total_steps = max(max_steps - start_step, 1)
    if args.warmup_steps > 0 and rel_step < args.warmup_steps:
        return base_lr * rel_step / args.warmup_steps
    denom = max(total_steps - args.warmup_steps, 1)
    progress = min(max((rel_step - args.warmup_steps) / denom, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


def apply_epoch_native_schedule(args, start_step, steps_per_epoch):
    if float(getattr(args, "epochs", 0.0)) > 0.0:
        args.max_steps = start_step + epoch_fraction_to_steps(args.epochs, steps_per_epoch)
    if float(getattr(args, "warmup_epochs", 0.0)) > 0.0:
        args.warmup_steps = epoch_fraction_to_steps(args.warmup_epochs, steps_per_epoch)
    if float(getattr(args, "router_only_epochs", 0.0)) > 0.0:
        args.router_only_steps = epoch_fraction_to_steps(args.router_only_epochs, steps_per_epoch)
    if float(getattr(args, "d_warmup_epochs", 0.0)) > 0.0:
        args.d_warmup_steps = epoch_fraction_to_steps(args.d_warmup_epochs, steps_per_epoch)
    if float(getattr(args, "g_freeze_epochs", 0.0)) > 0.0:
        args.g_freeze_steps = epoch_fraction_to_steps(args.g_freeze_epochs, steps_per_epoch)
    if float(getattr(args, "gan_start_epoch", -1.0)) >= 0.0:
        args.gan_start_step = start_step + epoch_fraction_to_steps(args.gan_start_epoch, steps_per_epoch, add_one_after=True)


def init_wandb_run(args, out_dir, dataset_len, world_size, trainable_count, is_main):
    if not is_main or not args.wandb:
        return None
    try:
        import wandb
    except Exception as exc:
        print(f"wandb disabled: failed to import wandb ({exc})", flush=True)
        return None

    config = vars(args).copy()
    config.update({
        "dataset_len": dataset_len,
        "world_size": world_size,
        "global_batch_size": args.batch_size * world_size * args.accum_steps,
        "trainable_params": trainable_count,
    })
    kwargs = {
        "project": args.wandb_project,
        "name": args.wandb_name or out_dir.name,
        "config": config,
        "dir": str(out_dir),
    }
    if args.wandb_entity:
        kwargs["entity"] = args.wandb_entity
    if args.wandb_mode:
        kwargs["mode"] = args.wandb_mode
    if args.wandb_tags:
        kwargs["tags"] = args.wandb_tags
    try:
        return wandb.init(**kwargs)
    except Exception as exc:
        print(f"wandb disabled: init failed ({exc})", flush=True)
        return None

_DATASET_MANIFEST_VERSION = 2


class ManifestImageFolder(Dataset):
    """ImageFolder-compatible dataset backed by a precomputed sample manifest."""

    def __init__(self, root, samples, classes, class_to_idx, transform=None):
        self.root = str(root)
        self.samples = [(str(path), int(target)) for path, target in samples]
        self.imgs = self.samples
        self.targets = [target for _, target in self.samples]
        self.classes = list(classes)
        self.class_to_idx = dict(class_to_idx)
        self.transform = transform
        self.target_transform = None
        self.loader = default_loader

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def _load_dataset_manifest(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {"version", "root", "samples", "classes", "class_to_idx"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError(f"invalid dataset manifest schema: {path}")
    return payload


def _scan_flat_imagefolder(root):
    """Scan ImageNet-style class directories without per-image NFS stat calls."""
    with os.scandir(root) as entries:
        classes = sorted(
            entry.name
            for entry in entries
            if not entry.name.startswith(".") and entry.is_dir(follow_symlinks=False)
        )
    if not classes:
        raise FileNotFoundError(f"no class directories found in {root}")

    class_to_idx = {class_name: index for index, class_name in enumerate(classes)}
    samples = []
    empty_classes = []
    for class_name in classes:
        class_dir = root / class_name
        with os.scandir(class_dir) as entries:
            filenames = sorted(
                entry.name
                for entry in entries
                if not entry.name.startswith(".") and entry.name.lower().endswith(IMG_EXTENSIONS)
            )
        if not filenames:
            empty_classes.append(class_name)
            continue
        target = class_to_idx[class_name]
        samples.extend((str(class_dir / filename), target) for filename in filenames)

    if empty_classes:
        raise FileNotFoundError(f"found no valid images in classes: {empty_classes[:8]}")
    return samples, classes, class_to_idx


def build_cached_imagefolder(args, distributed, rank):
    root = Path(args.data_path).resolve()
    manifest_path = Path(args.dataset_manifest) if args.dataset_manifest else root / ".mot_imagefolder_manifest_v2.pt"
    manifest_path = manifest_path.resolve()
    build_error = [None]

    if rank == 0:
        try:
            rebuild = bool(args.rebuild_dataset_manifest)
            if manifest_path.exists() and not rebuild:
                try:
                    payload = _load_dataset_manifest(manifest_path)
                    rebuild = payload["version"] != _DATASET_MANIFEST_VERSION or payload["root"] != str(root)
                except Exception:
                    rebuild = True
            else:
                rebuild = True

            if rebuild:
                samples, classes, class_to_idx = _scan_flat_imagefolder(root)
                payload = {
                    "version": _DATASET_MANIFEST_VERSION,
                    "root": str(root),
                    "samples": samples,
                    "classes": classes,
                    "class_to_idx": class_to_idx,
                }
                manifest_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = manifest_path.with_name(f".{manifest_path.name}.tmp.{os.getpid()}")
                try:
                    torch.save(payload, tmp_path)
                    os.replace(tmp_path, manifest_path)
                finally:
                    if tmp_path.exists():
                        tmp_path.unlink()
                print(f"saved dataset manifest with {len(samples):,} samples to {manifest_path}", flush=True)
            else:
                print(f"using cached dataset manifest {manifest_path}", flush=True)
        except Exception as exc:
            build_error[0] = f"{type(exc).__name__}: {exc}"

    if distributed:
        dist.broadcast_object_list(build_error, src=0)
    if build_error[0] is not None:
        raise RuntimeError(f"failed to prepare dataset manifest: {build_error[0]}")
    if distributed:
        dist.barrier()

    payload = _load_dataset_manifest(manifest_path)
    if payload["version"] != _DATASET_MANIFEST_VERSION or payload["root"] != str(root):
        raise RuntimeError(f"stale dataset manifest: {manifest_path}")
    transform = make_transform(args.image_size, args.random_crop, args.random_flip)
    return ManifestImageFolder(
        root,
        payload["samples"],
        payload["classes"],
        payload["class_to_idx"],
        transform=transform,
    )


def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training")
    distributed, rank, world_size, local_rank, device = distributed_setup()
    is_main = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    dataset = build_cached_imagefolder(args, distributed, rank)
    if args.limit_samples > 0:
        dataset = Subset(dataset, list(range(min(args.limit_samples, len(dataset)))))
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True) if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    data_iter = cycle(loader, sampler)
    loader_steps_per_epoch = max(1, len(loader))
    steps_per_epoch = optimizer_steps_per_epoch(loader_steps_per_epoch, args.accum_steps)

    titok = load_titok(args.titok_root, args.titok_config, args.titok_ckpt, device)
    vq_model = load_llamagen_vq(args.llamagen_root, args.llamagen_ckpt, device, args.codebook_size, args.codebook_embed_dim)
    native_teacher_vq = None
    if args.mix_native_teacher == "frozen":
        if float(args.lambda_mix_native) <= 0.0:
            raise ValueError("mix_native_teacher=frozen requires lambda_mix_native > 0")
        native_teacher_vq = load_llamagen_vq(
            args.llamagen_root,
            args.llamagen_ckpt,
            device,
            args.codebook_size,
            args.codebook_embed_dim,
        )
        native_teacher_vq.eval().requires_grad_(False)
    perceptual = build_perceptual_loss(args, device)
    dino_feature_loss = build_dino_feature_loss(args, device)
    clip_feature_loss = build_clip_feature_loss(args, device)

    model = TiTokLlamaGenStage2(
        titok,
        vq_model,
        lg_latent_channels=args.lg_latent_channels,
        head_channels=args.lg_head_channels,
        head_mode=args.latent_head_mode,
        codebook_size=args.codebook_size,
        codebook_temperature=args.codebook_temperature,
    ).to(device)
    if args.latent_head_mode != "feature":
        raise ValueError("decoder adaptation currently expects latent_head_mode=feature")
    model.router = DynamicBudgetRouter(
        latent_channels=args.lg_latent_channels,
        hidden_dim=args.router_hidden_dim,
        depth=args.router_depth,
        target_ratio=args.router_target_mean_ratio,
        min_ratio=args.router_min_ratio,
        max_ratio=args.router_max_ratio,
        detach_inputs=args.router_detach_inputs,
    ).to(device)

    router_modes = {"router_e2e_dynamic", "router_e2e_top_p"}
    if args.mask_selection in router_modes and not args.train_router:
        raise ValueError(f"{args.mask_selection} requires --train-router")

    if args.adapter_init:
        key, init_step = load_adapter_init(model, args.adapter_init, args.adapter_init_ema, strict=True)
        if is_main:
            print(f"loaded adapter init from {args.adapter_init} ({key}, step {init_step})", flush=True)

    configure_trainable_parts(model, args)
    core = model
    optimizer = torch.optim.AdamW(trainable_param_groups(core, args), betas=(0.9, 0.999))
    params = trainable_params(core)
    ema = TrainableEMA(core, decay=args.ema_decay) if args.use_ema else None
    discriminator = build_discriminator(args, device)
    disc_params = [p for p in discriminator.parameters() if p.requires_grad] if discriminator is not None else []
    optimizer_d = (
        torch.optim.AdamW(
            adamw_param_groups(discriminator, args.weight_decay_llamagen),
            lr=args.lr_d,
            betas=(0.5, 0.9),
        )
        if discriminator is not None
        else None
    )
    lecam_ema_real = torch.zeros((), device=device)
    lecam_ema_fake = torch.zeros((), device=device)
    router_budget_debt_tokens = 0.0

    start_step = 0
    selector_threshold_state = float(args.selector_threshold)
    selector_ratio_ema = float(args.target_mask_ratio)
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        missing, unexpected, skipped, filled_from_ema = load_adapt_resume(
            core,
            ckpt,
            strict=not args.resume_non_strict,
            skip_prefixes=args.resume_skip_prefixes,
            use_ema_model=args.resume_use_ema_model,
        )
        if args.resume_use_ema_model:
            print("[resume] loaded model weights from checkpoint model_ema", flush=True)
        start_step = int(ckpt.get("step", 0)) if isinstance(ckpt, dict) else 0
        if isinstance(ckpt, dict) and "optimizer" in ckpt and not args.reset_optimizer:
            optimizer.load_state_dict(ckpt["optimizer"])
        discriminator_load_mode = "none"
        if (
            discriminator is not None
            and isinstance(ckpt, dict)
            and "discriminator" in ckpt
            and not args.reset_discriminator
        ):
            discriminator_load_mode, _missing_d, _unexpected_d = load_discriminator_resume(
                discriminator,
                ckpt["discriminator"],
                strict=True,
            )
        if (
            optimizer_d is not None
            and isinstance(ckpt, dict)
            and "optimizer_d" in ckpt
            and not args.reset_optimizer
            and not args.reset_discriminator
        ):
            optimizer_d.load_state_dict(ckpt["optimizer_d"])
        if isinstance(ckpt, dict):
            lecam_ema_real = ckpt.get("lecam_ema_real", lecam_ema_real.detach().cpu()).to(device)
            lecam_ema_fake = ckpt.get("lecam_ema_fake", lecam_ema_fake.detach().cpu()).to(device)
            if not args.reset_selector_state:
                selector_threshold_state = float(ckpt.get("selector_threshold_state", selector_threshold_state))
                selector_ratio_ema = float(ckpt.get("selector_ratio_ema", selector_ratio_ema))
                router_budget_debt_tokens = float(ckpt.get("router_budget_debt_tokens", router_budget_debt_tokens))
        if ema is not None:
            if isinstance(ckpt, dict) and "model_ema" in ckpt:
                ema.load_state_dict(ckpt["model_ema"], device=device)
                if is_main:
                    print("loaded EMA from resume checkpoint", flush=True)
            else:
                ema = TrainableEMA(core, decay=args.ema_decay)
                if is_main:
                    print("initialized EMA from resumed model weights", flush=True)
        if is_main:
            if skipped:
                print(f"skipped {len(skipped)} resume params with prefixes {parse_prefix_list(args.resume_skip_prefixes)}", flush=True)
            if filled_from_ema:
                print(f"filled {len(filled_from_ema)} resume params from model_ema fallback", flush=True)
            if discriminator_load_mode != "none":
                print(f"loaded discriminator from resume ({discriminator_load_mode})", flush=True)
            print(f"resumed decoder-adapt checkpoint from {args.resume} at step {start_step}", flush=True)

    apply_epoch_native_schedule(args, start_step, steps_per_epoch)

    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        if discriminator is not None:
            discriminator = DDP(
                discriminator,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
                broadcast_buffers=False,
            )

    out_dir = Path(args.output_dir)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()
    log_path = out_dir / "log.txt"

    n_trainable = 0
    if is_main:
        core = model.module if distributed else model
        n_trainable = sum(p.numel() for p in core.parameters() if p.requires_grad)
        msg, run_header = format_run_header(args, len(dataset), world_size, n_trainable)
        epoch_msg = (
            f"loader_steps_per_epoch={loader_steps_per_epoch} optimizer_steps_per_epoch={steps_per_epoch} "
            f"epochs={args.epochs} max_steps={args.max_steps} "
            f"router_only_steps={args.router_only_steps} gan_start_step={args.gan_start_step} "
            f"d_warmup_steps={args.d_warmup_steps} latest_every_epoch={args.latest_every_epoch} "
            f"save_epoch_every={args.save_epoch_every} save_epoch_fraction_every={args.save_epoch_fraction_every}"
        )
        print(msg, flush=True)
        print(epoch_msg, flush=True)
        with log_path.open("a") as f:
            f.write(run_header)
            f.write(epoch_msg + "\n")

    wandb_run = init_wandb_run(args, out_dir, len(dataset), world_size, n_trainable, is_main)

    running = {
        "loss": 0.0,
        "base": 0.0,
        "base_lp": 0.0,
        "base_mse01": 0.0,
        "mix": 0.0,
        "mix_lp": 0.0,
        "mix_mse01": 0.0,
        "mix_native": 0.0,
        "mix_native_lp": 0.0,
        "native": 0.0,
        "native_lp": 0.0,
        "native_mse01": 0.0,
        "feat": 0.0,
        "feat_moment": 0.0,
        "dino_feat": 0.0,
        "clip_feat": 0.0,
        "gan_g": 0.0,
        "disc_fm": 0.0,
        "lowfreq_anchor": 0.0,
        "d_loss": 0.0,
        "lecam": 0.0,
        "logits_real": 0.0,
        "logits_fake": 0.0,
        "mask": 0.0,
        "mask_tokens": 0.0,
        "mask_tokens_std": 0.0,
        "mask_tokens_min": 0.0,
        "mask_tokens_max": 0.0,
        "score": 0.0,
        "score_error": 0.0,
        "score_gradient": 0.0,
        "score_variance": 0.0,
        "selector_threshold": 0.0,
        "selector_ratio_ema": 0.0,
        "router_budget": 0.0,
        "router_binary": 0.0,
        "router_budget_debt": 0.0,
        "router_aux": 0.0,
        "router_ratio_target": 0.0,
        "router_ratio_target_std": 0.0,
        "router_ratio_target_loss": 0.0,
        "router_ratio": 0.0,
        "router_ratio_std": 0.0,
        "router_soft": 0.0,
        "grad": 0.0,
    }
    count = 0
    start_time = time.time()
    pbar_desc = f"Epoch 0.00/{args.epochs:g}" if float(args.epochs) > 0.0 else f"Steps {start_step + 1}-{args.max_steps}"
    pbar = tqdm(total=max(0, args.max_steps - start_step), desc=pbar_desc, disable=not is_main, dynamic_ncols=True, mininterval=2)
    d_warmup_start_step = max(start_step + 1, args.gan_start_step) if args.reset_discriminator else 0
    d_warmup_end_step = d_warmup_start_step + max(0, args.d_warmup_steps) - 1 if args.reset_discriminator and args.d_warmup_steps > 0 else 0

    for step in range(start_step + 1, args.max_steps + 1):
        set_adapt_train_mode(model, args)
        core = model.module if distributed else model
        current_lr = compute_lr(args, step, base_lr=args.lr, start_step=start_step)
        current_lr_lg = compute_lr(args, step, base_lr=args.lr_llamagen, start_step=start_step)
        current_lr_d = compute_lr_for_scheduler(
            args,
            step,
            base_lr=args.lr_d,
            scheduler=args.lr_d_scheduler,
            min_lr=args.min_lr_d,
            start_step=args.gan_start_step,
        )
        router_only_active = args.mask_selection in router_modes and args.router_only_steps > 0 and step <= start_step + args.router_only_steps
        apply_router_only_trainability(core, args, router_only_active)
        g_freeze_active = args.g_freeze_steps > 0 and step <= start_step + args.g_freeze_steps
        for group in optimizer.param_groups:
            role = group.get("lr_role")
            if g_freeze_active or (router_only_active and role != "router"):
                group["lr"] = 0.0
            elif role == "llamagen":
                group["lr"] = current_lr_lg
            elif role == "llamagen_encoder":
                group["lr"] = compute_lr(args, step, base_lr=args.lr_llamagen_encoder, start_step=start_step)
            elif role == "router":
                group["lr"] = compute_lr(args, step, base_lr=args.lr_router, start_step=start_step)
            else:
                group["lr"] = current_lr
        if optimizer_d is not None:
            set_optimizer_lr(optimizer_d, 0.0 if (router_only_active and args.router_only_disable_gan) else current_lr_d)
        gan_factor = gan_factor_for_step(args, step)
        if router_only_active and args.router_only_disable_gan:
            gan_factor = 0.0
        d_warmup_active = (
            discriminator is not None
            and args.d_warmup_steps > 0
            and d_warmup_start_step <= step <= d_warmup_end_step
        )
        gan_g_factor = gan_g_factor_for_step(args, step, d_warmup_end_step)
        if router_only_active and args.router_only_disable_gan:
            gan_g_factor = 0.0
        train_discriminator = discriminator is not None and gan_factor > 0.0 and step % args.d_every == 0
        optimizer.zero_grad(set_to_none=True)
        if optimizer_d is not None:
            optimizer_d.zero_grad(set_to_none=True)

        metric_sums = {key: 0.0 for key in running if key != "grad"}
        last = {}

        accum_batches = []
        for _micro_step in range(args.accum_steps):
            x_01, _ = next(data_iter)
            x_01 = x_01.to(device, non_blocking=True)
            x_titok = convert_image_range(x_01, args.titok_input_range)
            x_lg = convert_image_range(x_01, args.llamagen_input_range)
            accum_batches.append((x_01, x_titok, x_lg))

        precomputed_masks = [None] * len(accum_batches)
        score_metric_values = {
            "score": 0.0,
            "score_error": 0.0,
            "score_gradient": 0.0,
            "score_variance": 0.0,
            "selector_threshold": float(selector_threshold_state),
            "selector_ratio_ema": float(selector_ratio_ema),
            "router_budget_debt_tokens": float(router_budget_debt_tokens),
        }
        if args.mask_selection == "accum_global_error":
            score_chunks = []
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=autocast_dtype(args.mixed_precision), enabled=args.mixed_precision != "none"):
                    for _x_01_scan, x_titok_scan, x_lg_scan in accum_batches:
                        x_base_scan, _extra_scan = core(x_titok_scan)
                        score_chunks.append(spatial_error_score(x_base_scan, x_lg_scan))
            mask_cat = global_top_ratio_mask_from_score(torch.cat(score_chunks, dim=0), args.mask_ratio)
            precomputed_masks = list(torch.split(mask_cat, [score.shape[0] for score in score_chunks], dim=0))
        elif args.mask_selection == "accum_global_mixed_score":
            raw_chunks = {"error": [], "gradient": [], "variance": []}
            lengths = []
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=autocast_dtype(args.mixed_precision), enabled=args.mixed_precision != "none"):
                    for _x_01_scan, x_titok_scan, x_lg_scan in accum_batches:
                        x_base_scan, _extra_scan = core(x_titok_scan)
                        raw_chunks["error"].append(spatial_error_score_grid(x_base_scan, x_lg_scan, grid_hw=16))
                        raw_chunks["gradient"].append(gradient_score_from_image_grid(x_lg_scan, grid_hw=16))
                        raw_chunks["variance"].append(local_variance_score_from_image_grid(x_lg_scan, grid_hw=16))
                        lengths.append(x_lg_scan.shape[0])
            raw_cat = {key: torch.cat(chunks, dim=0) for key, chunks in raw_chunks.items()}
            score_cat, norm_parts = mixed_score_from_raw_global_pool(raw_cat, args)
            mask_cat = global_top_ratio_mask_from_score(score_cat, args.mask_ratio)
            precomputed_masks = list(torch.split(mask_cat, lengths, dim=0))
            score_metric_values = {
                "score": global_mean_value(score_cat),
                "score_error": global_mean_value(norm_parts["error"]),
                "score_gradient": global_mean_value(norm_parts["gradient"]),
                "score_variance": global_mean_value(norm_parts["variance"]),
                "selector_threshold": float(selector_threshold_state),
                "selector_ratio_ema": float(selector_ratio_ema),
                "router_budget_debt_tokens": float(router_budget_debt_tokens),
            }
        elif args.mask_selection == "accum_global_mixed_score_threshold":
            raw_chunks = {"error": [], "gradient": [], "variance": []}
            lengths = []
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=autocast_dtype(args.mixed_precision), enabled=args.mixed_precision != "none"):
                    for _x_01_scan, x_titok_scan, x_lg_scan in accum_batches:
                        x_base_scan, _extra_scan = core(x_titok_scan)
                        raw_chunks["error"].append(spatial_error_score_grid(x_base_scan, x_lg_scan, grid_hw=16))
                        raw_chunks["gradient"].append(gradient_score_from_image_grid(x_lg_scan, grid_hw=16))
                        raw_chunks["variance"].append(local_variance_score_from_image_grid(x_lg_scan, grid_hw=16))
                        lengths.append(x_lg_scan.shape[0])
            raw_cat = {key: torch.cat(chunks, dim=0) for key, chunks in raw_chunks.items()}
            score_cat, norm_parts = mixed_score_from_raw_parts(raw_cat, args, args.score_normalize_scope)
            threshold_used = float(selector_threshold_state)
            mask_cat = threshold_mask_from_score(score_cat, threshold_used)
            global_mask_ratio = global_mean_value(mask_cat)
            if args.use_threshold_ema_controller:
                selector_ratio_ema = (
                    float(args.selector_ratio_ema_decay) * float(selector_ratio_ema)
                    + (1.0 - float(args.selector_ratio_ema_decay)) * global_mask_ratio
                )
                selector_threshold_state += float(args.selector_threshold_lr) * (
                    selector_ratio_ema - float(args.target_mask_ratio)
                )
                selector_threshold_state = max(
                    float(args.selector_threshold_min),
                    min(float(args.selector_threshold_max), float(selector_threshold_state)),
                )
            precomputed_masks = list(torch.split(mask_cat, lengths, dim=0))
            score_metric_values = {
                "score": global_mean_value(score_cat),
                "score_error": global_mean_value(norm_parts["error"]),
                "score_gradient": global_mean_value(norm_parts["gradient"]),
                "score_variance": global_mean_value(norm_parts["variance"]),
                "selector_threshold": threshold_used,
                "selector_ratio_ema": float(selector_ratio_ema),
            }

        for _micro_step, (_x_01, x_titok, x_lg) in enumerate(accum_batches):
            sync_micro_step = _micro_step == len(accum_batches) - 1
            model_for_forward = core if router_only_active else model
            model_sync_context = (
                model.no_sync()
                if distributed and not router_only_active and not sync_micro_step and isinstance(model, DDP)
                else contextlib.nullcontext()
            )
            disc_sync_context = (
                discriminator.no_sync()
                if distributed and not sync_micro_step and isinstance(discriminator, DDP)
                else contextlib.nullcontext()
            )
            with model_sync_context, disc_sync_context, torch.autocast(
                device_type="cuda", dtype=autocast_dtype(args.mixed_precision), enabled=args.mixed_precision != "none"
            ):
                x_base, extra = model_for_forward(x_titok)
                f_1d_lg = extra["f_1d_lg"]
                f_2d_lg, _ = native_llamagen_feature(
                    core.llamagen_vq,
                    x_lg,
                    args.codebook_embed_dim,
                    allow_encoder_grad=args.train_llamagen_encoder,
                )
                if f_1d_lg.shape[1:] != (args.lg_latent_channels, 16, 16):
                    raise ValueError(f"bad f_1d_lg shape {tuple(f_1d_lg.shape)}")
                if f_2d_lg.shape[1:] != (args.lg_latent_channels, 16, 16):
                    raise ValueError(f"bad f_2d_lg shape {tuple(f_2d_lg.shape)}")

                router_logits = None
                router_ratio_soft = None
                router_soft_mask = None
                router_budget_loss = x_lg.new_zeros(())
                router_binary_loss = x_lg.new_zeros(())
                router_aux_loss = x_lg.new_zeros(())
                router_ratio_target = None
                router_ratio_target_loss = x_lg.new_zeros(())
                local_score_metric_values = score_metric_values
                if args.mask_selection in router_modes:
                    router_logits, router_ratio_logits = core.router(f_1d_lg, x_base, f_2d_lg)
                    if args.mask_selection == "router_e2e_top_p":
                        mask, hard_mask, router_soft_mask, router_ratio_soft = make_top_p_ste_mask(
                            router_logits, args
                        )
                    else:
                        router_ratio_logits = calibrate_ratio_logits_to_target_mean(router_ratio_logits, args)
                        mask, hard_mask, router_soft_mask, router_ratio_soft = make_dynamic_budget_ste_mask(
                            router_logits, router_ratio_logits, args, budget_debt_tokens=router_budget_debt_tokens
                        )
                    if args.mask_selection == "router_e2e_dynamic" and getattr(args, "router_budget_mode", "soft") == "bank":
                        target_ratio_for_bank = float(getattr(args, "router_exact_budget_ratio", -1.0))
                        if target_ratio_for_bank < 0.0:
                            target_ratio_for_bank = float(args.router_target_mean_ratio)
                        router_budget_debt_tokens += float(hard_mask.detach().float().sum().item()) - (
                            hard_mask.shape[0] * hard_mask.shape[2] * hard_mask.shape[3] * target_ratio_for_bank
                        )
                        max_debt_for_bank = max(0.0, float(getattr(args, "router_budget_bank_max_debt_tokens", 0.0)))
                        if max_debt_for_bank > 0.0:
                            router_budget_debt_tokens = max(-max_debt_for_bank, min(max_debt_for_bank, router_budget_debt_tokens))
                    mask = mask.to(dtype=f_1d_lg.dtype)
                    router_soft_mean = router_soft_mask.float().mean()
                    router_ratio_mean = router_ratio_soft.float().mean()
                    if args.mask_selection == "router_e2e_top_p":
                        router_budget_loss = (router_soft_mean - float(args.router_target_mean_ratio)).pow(2)
                        # Keep the legacy ratio head visible to DDP without using it to choose top-p K.
                        router_budget_loss = router_budget_loss + router_ratio_logits.float().sum() * 0.0
                    else:
                        router_budget_loss = (router_soft_mean - float(args.router_target_mean_ratio)).pow(2) + (
                            router_ratio_mean - float(args.router_target_mean_ratio)
                        ).pow(2)
                    router_binary_loss = (router_soft_mask.float() * (1.0 - router_soft_mask.float())).mean()
                    local_score_metric_values = {
                        "score": global_mean_value(torch.sigmoid(router_logits.detach().float())),
                        "score_error": 0.0,
                        "score_gradient": global_mean_value(router_soft_mask.detach().float()),
                        "score_variance": 0.0,
                        "selector_threshold": global_mean_value(router_ratio_soft.detach().float()),
                        "selector_ratio_ema": global_mean_value(hard_mask.detach().float()),
                    }
                elif precomputed_masks[_micro_step] is None:
                    mask = oracle_error_mask(x_base, x_lg, args.mask_ratio_min, args.mask_ratio_max).to(dtype=f_1d_lg.dtype)
                else:
                    mask = precomputed_masks[_micro_step].to(device=f_1d_lg.device, dtype=f_1d_lg.dtype)
                f_mix = (1.0 - mask) * f_1d_lg + mask * f_2d_lg
                x_mix = core.llamagen_vq.decoder(f_mix)
                needs_native_grad = float(args.lambda_native) > 0.0
                if needs_native_grad:
                    x_native = core.llamagen_vq.decoder(f_2d_lg)
                else:
                    with torch.no_grad():
                        x_native = core.llamagen_vq.decoder(f_2d_lg.detach())
                x_native_teacher = x_native.detach()
                if native_teacher_vq is not None:
                    with torch.no_grad():
                        f_2d_teacher, _ = native_llamagen_feature(
                            native_teacher_vq,
                            x_lg,
                            args.codebook_embed_dim,
                            allow_encoder_grad=False,
                        )
                        x_native_teacher = native_teacher_vq.decoder(f_2d_teacher).detach()

                if float(args.lambda_base) > 0.0:
                    base_img, base_lp, base_mse01 = compute_path_losses(
                        perceptual, args.perceptual_loss, x_base, x_lg, args.image_loss
                    )
                else:
                    with torch.no_grad():
                        base_img, base_lp, base_mse01 = compute_path_losses(
                            perceptual, args.perceptual_loss, x_base.detach(), x_lg, args.image_loss
                        )
                mix_img, mix_lp, mix_mse01 = compute_path_losses(perceptual, args.perceptual_loss, x_mix, x_lg, args.image_loss)
                if float(args.lambda_native) > 0.0:
                    native_img, native_lp, native_mse01 = compute_path_losses(
                        perceptual, args.perceptual_loss, x_native, x_lg, args.image_loss
                    )
                else:
                    with torch.no_grad():
                        native_img, native_lp, native_mse01 = compute_path_losses(
                            perceptual, args.perceptual_loss, x_native.detach(), x_lg, args.image_loss
                        )
                if float(args.lambda_mix_native) > 0.0:
                    mix_native_img, mix_native_lp, _ = compute_path_losses(
                        perceptual, args.perceptual_loss, x_mix, x_native_teacher, args.image_loss
                    )
                else:
                    with torch.no_grad():
                        mix_native_img, mix_native_lp, _ = compute_path_losses(
                            perceptual, args.perceptual_loss, x_mix.detach(), x_native_teacher, args.image_loss
                        )
                feat_loss = F.l1_loss(f_1d_lg.float(), f_2d_lg.detach().float())
                feat_moment = compute_feature_moment_loss(f_1d_lg.float(), f_2d_lg.detach().float())
                dino_feat_loss = x_mix.new_zeros(())
                if dino_feature_loss is not None and float(args.lambda_dino_feat) > 0.0:
                    dino_feat_loss = dino_feature_loss(
                        x_mix,
                        x_lg,
                        pred_range=args.llamagen_input_range,
                        target_range=args.llamagen_input_range,
                    )
                clip_feat_loss = x_mix.new_zeros(())
                if clip_feature_loss is not None and float(args.lambda_clip_feat) > 0.0:
                    clip_feat_loss = clip_feature_loss(
                        x_mix,
                        x_lg,
                        pred_range=args.llamagen_input_range,
                        target_range=args.llamagen_input_range,
                    )
                lowfreq_anchor_loss = x_mix.new_zeros(())
                if float(args.lambda_lowfreq_anchor) > 0.0:
                    lowfreq_anchor_loss = compute_lowfreq_anchor_loss(
                        x_mix,
                        x_lg,
                        args.llamagen_input_range,
                        args.lowfreq_anchor_size,
                    )
                router_aux_weight = float(args.router_gain_aux_weight)
                if args.router_gain_aux_decay_steps > 0:
                    router_aux_weight *= max(0.0, 1.0 - float(step - start_step) / float(args.router_gain_aux_decay_steps))
                needs_ratio_target = router_logits is not None and (
                    router_aux_weight > 0.0 or float(args.lambda_router_ratio_target) > 0.0
                )
                if needs_ratio_target:
                    with torch.no_grad():
                        gain_score = grid_mse_gain_score(x_base.detach(), x_native.detach(), x_lg, grid_hw=16)
                    if float(args.router_ratio_target_spread) > 0.0 or float(args.lambda_router_ratio_target) > 0.0:
                        router_ratio_target = image_ratio_target_from_gain_score(gain_score, args)
                    if router_aux_weight > 0.0:
                        if router_ratio_target is not None and float(args.router_ratio_target_spread) > 0.0:
                            gain_target = per_image_variable_top_ratio_mask_from_score(gain_score, router_ratio_target)
                        else:
                            gain_target = per_image_top_ratio_mask_from_score(gain_score, args.router_gain_aux_target_ratio)
                        router_aux_loss = F.binary_cross_entropy_with_logits(router_logits.float(), gain_target.float())
                    if router_ratio_soft is not None and float(args.lambda_router_ratio_target) > 0.0:
                        if router_ratio_target is None:
                            router_ratio_target = image_ratio_target_from_gain_score(gain_score, args)
                        router_ratio_target_loss = F.smooth_l1_loss(router_ratio_soft.float(), router_ratio_target.float())

                perceptual_weight = get_perceptual_weight(args)
                base_loss = (
                    args.lambda_base * (base_img + perceptual_weight * base_lp)
                    if float(args.lambda_base) > 0.0
                    else x_mix.new_zeros(())
                )
                mix_loss = args.lambda_mix * (mix_img + perceptual_weight * mix_lp)
                native_loss = (
                    args.lambda_native * (native_img + perceptual_weight * native_lp)
                    if float(args.lambda_native) > 0.0
                    else x_mix.new_zeros(())
                )
                mix_native_loss = (
                    args.lambda_mix_native * (mix_native_img + args.lambda_mix_native_perceptual * mix_native_lp)
                    if float(args.lambda_mix_native) > 0.0
                    else x_mix.new_zeros(())
                )
                gan_g_loss = x_mix.new_zeros(())
                disc_fm_loss = x_mix.new_zeros(())
                if discriminator is not None and gan_g_factor > 0.0 and not d_warmup_active:
                    set_requires_grad(discriminator, False)
                    fake_for_g, real_for_g = prepare_gan_inputs(
                        x_mix,
                        x_lg,
                        args.llamagen_input_range,
                        args,
                        for_generator=True,
                    )
                    logits_fake_for_g = discriminator(fake_for_g)
                    gan_g_loss = gan_g_loss_from_logits(logits_fake_for_g)
                    if args.lambda_disc_feature_matching > 0.0 and not g_freeze_active:
                        disc_fm_loss = discriminator_feature_matching_loss(discriminator, fake_for_g, real_for_g)
                loss = (
                    base_loss
                    + mix_loss
                    + native_loss
                    + mix_native_loss
                    + args.lambda_feat * feat_loss
                    + args.lambda_feat_moment * feat_moment
                    + args.lambda_dino_feat * dino_feat_loss
                    + args.lambda_clip_feat * clip_feat_loss
                    + args.lambda_lowfreq_anchor * lowfreq_anchor_loss
                    + args.lambda_disc_feature_matching * disc_fm_loss
                    + args.lambda_gan * gan_g_factor * gan_g_loss
                    + args.lambda_router_budget * router_budget_loss
                    + args.lambda_router_binary * router_binary_loss
                    + args.lambda_router_ratio_target * router_ratio_target_loss
                    + router_aux_weight * router_aux_loss
                )

            if args.train_llamagen_quantizer and hasattr(core.llamagen_vq, "quantize"):
                quantizer_zero_loss = None
                for quantizer_param in core.llamagen_vq.quantize.parameters():
                    if quantizer_param.requires_grad:
                        quantizer_term = quantizer_param.float().sum() * 0.0
                        quantizer_zero_loss = quantizer_term if quantizer_zero_loss is None else quantizer_zero_loss + quantizer_term
                if quantizer_zero_loss is not None:
                    loss = loss + quantizer_zero_loss

            (loss / args.accum_steps).backward()

            d_loss = x_mix.new_zeros(())
            lecam_loss = x_mix.new_zeros(())
            logits_real_mean = x_mix.new_zeros(())
            logits_fake_mean = x_mix.new_zeros(())
            if train_discriminator:
                set_requires_grad(discriminator, True)
                fake_for_d, real_for_d = prepare_gan_inputs(
                    x_mix.detach(),
                    x_lg,
                    args.llamagen_input_range,
                    args,
                    for_generator=False,
                )
                real_for_d = real_for_d.detach()
                fake_for_d = fake_for_d.detach()
                logits_real = discriminator(real_for_d)
                logits_fake = discriminator(fake_for_d)
                logits_real_mean = weighted_logits_mean(logits_real)
                logits_fake_mean = weighted_logits_mean(logits_fake)
                d_loss = gan_factor * hinge_d_loss(logits_real, logits_fake)
                if args.lecam_regularization_weight > 0.0:
                    lecam_loss = compute_lecam_loss(
                        logits_real_mean,
                        logits_fake_mean,
                        lecam_ema_real,
                        lecam_ema_fake,
                    ) * args.lecam_regularization_weight
                    d_loss = d_loss + lecam_loss
                    lecam_ema_real = lecam_ema_real * args.lecam_ema_decay + logits_real_mean.detach() * (1.0 - args.lecam_ema_decay)
                    lecam_ema_fake = lecam_ema_fake * args.lecam_ema_decay + logits_fake_mean.detach() * (1.0 - args.lecam_ema_decay)
                (d_loss / args.accum_steps).backward()

            metric_sums["loss"] += loss.detach().float().item()
            metric_sums["base"] += base_img.detach().float().item()
            metric_sums["base_lp"] += base_lp.detach().float().item()
            metric_sums["base_mse01"] += base_mse01.detach().float().item()
            metric_sums["mix"] += mix_img.detach().float().item()
            metric_sums["mix_lp"] += mix_lp.detach().float().item()
            metric_sums["mix_mse01"] += mix_mse01.detach().float().item()
            metric_sums["mix_native"] += mix_native_img.detach().float().item()
            metric_sums["mix_native_lp"] += mix_native_lp.detach().float().item()
            metric_sums["native"] += native_img.detach().float().item()
            metric_sums["native_lp"] += native_lp.detach().float().item()
            metric_sums["native_mse01"] += native_mse01.detach().float().item()
            metric_sums["feat"] += feat_loss.detach().float().item()
            metric_sums["feat_moment"] += feat_moment.detach().float().item()
            metric_sums["dino_feat"] += dino_feat_loss.detach().float().item()
            metric_sums["clip_feat"] += clip_feat_loss.detach().float().item()
            metric_sums["gan_g"] += (args.lambda_gan * gan_g_factor * gan_g_loss).detach().float().item()
            metric_sums["disc_fm"] += (args.lambda_disc_feature_matching * disc_fm_loss).detach().float().item()
            metric_sums["lowfreq_anchor"] += (args.lambda_lowfreq_anchor * lowfreq_anchor_loss).detach().float().item()
            metric_sums["d_loss"] += d_loss.detach().float().item()
            metric_sums["lecam"] += lecam_loss.detach().float().item()
            metric_sums["router_budget"] += router_budget_loss.detach().float().item()
            metric_sums["router_binary"] += router_binary_loss.detach().float().item()
            metric_sums["router_budget_debt"] += float(router_budget_debt_tokens)
            metric_sums["router_aux"] += router_aux_loss.detach().float().item()
            metric_sums["router_ratio_target"] += 0.0 if router_ratio_target is None else router_ratio_target.detach().float().mean().item()
            metric_sums["router_ratio_target_std"] += 0.0 if router_ratio_target is None else router_ratio_target.detach().float().std(unbiased=False).item()
            metric_sums["router_ratio_target_loss"] += router_ratio_target_loss.detach().float().item()
            metric_sums["router_ratio"] += 0.0 if router_ratio_soft is None else router_ratio_soft.detach().float().mean().item()
            metric_sums["router_ratio_std"] += 0.0 if router_ratio_soft is None else router_ratio_soft.detach().float().std(unbiased=False).item()
            metric_sums["router_soft"] += 0.0 if router_soft_mask is None else router_soft_mask.detach().float().mean().item()
            mask_tokens = mask.detach().float().flatten(1).sum(dim=1)
            metric_sums["logits_real"] += logits_real_mean.detach().float().item()
            metric_sums["logits_fake"] += logits_fake_mean.detach().float().item()
            metric_sums["mask"] += mask.detach().float().mean().item()
            metric_sums["mask_tokens"] += mask_tokens.mean().item()
            metric_sums["mask_tokens_std"] += mask_tokens.std(unbiased=False).item()
            metric_sums["mask_tokens_min"] += mask_tokens.min().item()
            metric_sums["mask_tokens_max"] += mask_tokens.max().item()
            for score_key, score_value in local_score_metric_values.items():
                metric_sums[score_key] += score_value
            last = {
                "x_lg": x_lg,
                "x_base": x_base,
                "x_mix": x_mix,
                "x_native": x_native,
                "f_1d_lg": f_1d_lg,
                "f_2d_lg": f_2d_lg,
                "mask": mask,
            }

        if distributed and router_only_active:
            sync_module_grads(getattr(core, "router", None), world_size)

        grad_norm = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
        if not g_freeze_active:
            optimizer.step()
            if ema is not None:
                ema.update(model.module if distributed else model)
        if train_discriminator:
            torch.nn.utils.clip_grad_norm_(disc_params, args.max_grad_norm)
            optimizer_d.step()

        vals = torch.tensor(
            [
                metric_sums["loss"] / args.accum_steps,
                metric_sums["base"] / args.accum_steps,
                metric_sums["base_lp"] / args.accum_steps,
                metric_sums["base_mse01"] / args.accum_steps,
                metric_sums["mix"] / args.accum_steps,
                metric_sums["mix_lp"] / args.accum_steps,
                metric_sums["mix_mse01"] / args.accum_steps,
                metric_sums["mix_native"] / args.accum_steps,
                metric_sums["mix_native_lp"] / args.accum_steps,
                metric_sums["native"] / args.accum_steps,
                metric_sums["native_lp"] / args.accum_steps,
                metric_sums["native_mse01"] / args.accum_steps,
                metric_sums["feat"] / args.accum_steps,
                metric_sums["feat_moment"] / args.accum_steps,
                metric_sums["dino_feat"] / args.accum_steps,
                metric_sums["clip_feat"] / args.accum_steps,
                metric_sums["gan_g"] / args.accum_steps,
                metric_sums["disc_fm"] / args.accum_steps,
                metric_sums["lowfreq_anchor"] / args.accum_steps,
                metric_sums["d_loss"] / args.accum_steps,
                metric_sums["lecam"] / args.accum_steps,
                metric_sums["logits_real"] / args.accum_steps,
                metric_sums["logits_fake"] / args.accum_steps,
                metric_sums["mask"] / args.accum_steps,
                metric_sums["mask_tokens"] / args.accum_steps,
                metric_sums["mask_tokens_std"] / args.accum_steps,
                metric_sums["mask_tokens_min"] / args.accum_steps,
                metric_sums["mask_tokens_max"] / args.accum_steps,
                metric_sums["score"] / args.accum_steps,
                metric_sums["score_error"] / args.accum_steps,
                metric_sums["score_gradient"] / args.accum_steps,
                metric_sums["score_variance"] / args.accum_steps,
                metric_sums["selector_threshold"] / args.accum_steps,
                metric_sums["selector_ratio_ema"] / args.accum_steps,
                metric_sums["router_budget"] / args.accum_steps,
                metric_sums["router_binary"] / args.accum_steps,
                metric_sums["router_budget_debt"] / args.accum_steps,
                metric_sums["router_aux"] / args.accum_steps,
                metric_sums["router_ratio_target"] / args.accum_steps,
                metric_sums["router_ratio_target_std"] / args.accum_steps,
                metric_sums["router_ratio_target_loss"] / args.accum_steps,
                metric_sums["router_ratio"] / args.accum_steps,
                metric_sums["router_ratio_std"] / args.accum_steps,
                metric_sums["router_soft"] / args.accum_steps,
                grad_norm.detach().float().item(),
            ],
            device=device,
            dtype=torch.float32,
        )
        if distributed:
            dist.all_reduce(vals, op=dist.ReduceOp.AVG)
        keys = [
            "loss", "base", "base_lp", "base_mse01", "mix", "mix_lp", "mix_mse01",
            "mix_native", "mix_native_lp",
            "native", "native_lp", "native_mse01", "feat", "feat_moment", "dino_feat", "clip_feat",
            "gan_g", "disc_fm", "lowfreq_anchor", "d_loss", "lecam", "logits_real", "logits_fake",
            "mask", "mask_tokens", "mask_tokens_std", "mask_tokens_min", "mask_tokens_max",
            "score", "score_error", "score_gradient", "score_variance",
            "selector_threshold", "selector_ratio_ema", "router_budget", "router_binary", "router_budget_debt",
            "router_aux", "router_ratio_target", "router_ratio_target_std", "router_ratio_target_loss",
            "router_ratio", "router_ratio_std", "router_soft", "grad",
        ]
        for key, value in zip(keys, vals.tolist()):
            running[key] += value
        count += 1
        denom = max(count, 1)
        sec = (time.time() - start_time) / denom
        base_mse = running["base_mse01"] / denom
        mix_mse = running["mix_mse01"] / denom
        native_mse = running["native_mse01"] / denom
        loss_avg = running["loss"] / denom
        gan_avg = running["gan_g"] / denom
        perceptual_weight_log = get_perceptual_weight(args)
        recon_avg = (
            args.lambda_base * (running["base"] / denom + perceptual_weight_log * running["base_lp"] / denom)
            + args.lambda_mix * (running["mix"] / denom + perceptual_weight_log * running["mix_lp"] / denom)
            + args.lambda_native * (running["native"] / denom + perceptual_weight_log * running["native_lp"] / denom)
            + args.lambda_mix_native
            * (running["mix_native"] / denom + args.lambda_mix_native_perceptual * running["mix_native_lp"] / denom)
        )
        gan_abs_frac = abs(gan_avg) / max(abs(loss_avg), 1e-12)
        gan_recon_frac = abs(gan_avg) / max(abs(recon_avg), 1e-12)

        if is_main:
            pbar.update(1)
            epoch_progress = float(step - start_step) / float(steps_per_epoch)
            if float(args.epochs) > 0.0:
                pbar.set_description(f"Epoch {epoch_progress:.2f}/{args.epochs:g}")
            pbar.set_postfix({
                "step": step,
                "epoch": f"{epoch_progress:.2f}",
                "mix_l1": f"{running['mix'] / denom:.3f}",
                "mix_lp": f"{running['mix_lp'] / denom:.3f}",
                "psnr": f"{(-10.0 * math.log10(max(mix_mse, 1e-12))):.2f}",
                "base": f"{(-10.0 * math.log10(max(base_mse, 1e-12))):.2f}",
                "mask": f"{running['mask'] / denom:.2f}",
                "tok": f"{running['mask_tokens'] / denom:.0f}",
                "thr": f"{running['selector_threshold'] / denom:.3f}",
                "rb": f"{running['router_budget'] / denom:.3f}",
                "debt": f"{running['router_budget_debt'] / denom:.0f}",
                "rr": f"{running['router_ratio'] / denom:.2f}",
                "gan": f"{gan_avg:.3f}",
                "g%": f"{100.0 * gan_abs_frac:.1f}",
                "gr%": f"{100.0 * gan_recon_frac:.1f}",
                "fm": f"{running['disc_fm'] / denom:.3f}",
                "lf": f"{running['lowfreq_anchor'] / denom:.3f}",
                "d": f"{running['d_loss'] / denom:.3f}",
                "phase": "g_freeze" if g_freeze_active else ("router" if router_only_active else "joint"),
            })

        if is_main and (step == 1 or step % args.log_every == 0):
            with torch.no_grad():
                f_1d_lg = last["f_1d_lg"]
                f_2d_lg = last["f_2d_lg"]
                stats = (
                    f"f1d mean/std {f_1d_lg.float().mean().item():.4f}/{f_1d_lg.float().std().item():.4f} "
                    f"f2d mean/std {f_2d_lg.float().mean().item():.4f}/{f_2d_lg.float().std().item():.4f} "
                    f"base std {last['x_base'].float().std().item():.4f} mix std {last['x_mix'].float().std().item():.4f} "
                    f"native std {last['x_native'].float().std().item():.4f} "
                    f"mask {running['mask']/denom:.3f} tokens {running['mask_tokens']/denom:.1f} "
                    f"tok_std {running['mask_tokens_std']/denom:.1f} tok_min {running['mask_tokens_min']/denom:.0f} "
                    f"tok_max {running['mask_tokens_max']/denom:.0f} "
                    f"selector_threshold {running['selector_threshold']/denom:.4f} "
                    f"selector_ratio_ema {running['selector_ratio_ema']/denom:.4f} "
                    f"sel_score {running['score']/denom:.3f} err_score {running['score_error']/denom:.3f} "
                    f"grad_score {running['score_gradient']/denom:.3f} var_score {running['score_variance']/denom:.3f} "
                    f"router_budget {running['router_budget']/denom:.5f} router_binary {running['router_binary']/denom:.5f} "
                    f"router_debt {running['router_budget_debt']/denom:.1f} "
                    f"router_aux {running['router_aux']/denom:.5f} router_ratio {running['router_ratio']/denom:.3f} "
                    f"ratio_tgt {running['router_ratio_target']/denom:.3f} "
                    f"ratio_std {running['router_ratio_std']/denom:.3f} "
                    f"ratio_tgt_std {running['router_ratio_target_std']/denom:.3f} "
                    f"ratio_tgt_loss {running['router_ratio_target_loss']/denom:.5f} "
                    f"router_soft {running['router_soft']/denom:.3f}"
                )
            msg = (
                f"Step {step:08d} | loss {running['loss']/denom:.5f} | "
                f"base {running['base']/denom:.5f} lp {running['base_lp']/denom:.5f} psnr {(-10.0 * math.log10(max(base_mse, 1e-12))):.2f} | "
                f"mix {running['mix']/denom:.5f} lp {running['mix_lp']/denom:.5f} psnr {(-10.0 * math.log10(max(mix_mse, 1e-12))):.2f} | "
                f"native {running['native']/denom:.5f} lp {running['native_lp']/denom:.5f} psnr {(-10.0 * math.log10(max(native_mse, 1e-12))):.2f} | "
                f"mix_native {running['mix_native']/denom:.5f} lp {running['mix_native_lp']/denom:.5f} | "
                f"feat {running['feat']/denom:.5f} feat_moment {running['feat_moment']/denom:.5f} "
                f"dino_feat {running['dino_feat']/denom:.5f} clip_feat {running['clip_feat']/denom:.5f} | "
                f"gan_g {gan_avg:.5f} gan_abs_frac {gan_abs_frac:.5f} gan_recon_frac {gan_recon_frac:.5f} "
                f"disc_fm {running['disc_fm']/denom:.5f} lowfreq_anchor {running['lowfreq_anchor']/denom:.5f} | d {running['d_loss']/denom:.5f} | "
                f"router_budget {running['router_budget']/denom:.5f} router_binary {running['router_binary']/denom:.5f} "
                f"router_debt {running['router_budget_debt']/denom:.1f} router_aux {running['router_aux']/denom:.5f} "
                f"ratio_tgt_std {running['router_ratio_target_std']/denom:.3f} "
                f"ratio_tgt_loss {running['router_ratio_target_loss']/denom:.5f} | "
                f"lecam {running['lecam']/denom:.5f} | d_real {running['logits_real']/denom:.4f} | "
                f"d_fake {running['logits_fake']/denom:.4f} | "
                f"grad {running['grad']/denom:.4f} | lr {current_lr:.6g} lr_lg {current_lr_lg:.6g} "
                f"lr_router {compute_lr(args, step, base_lr=args.lr_router, start_step=start_step):.6g} lr_d {current_lr_d:.6g} "
                f"phase {'g_freeze' if g_freeze_active else ('router_only' if router_only_active else 'joint')} | {sec:.3f}s/step | {stats}"
            )
            with log_path.open("a") as f:
                f.write(msg + "\n")
            if wandb_run is not None:
                log_metrics = {f"train/{key}": running[key] / denom for key in keys}
                log_metrics.update({
                    "train/gan_abs_frac": gan_abs_frac,
                    "train/gan_recon_frac": gan_recon_frac,
                    "train/base_psnr": -10.0 * math.log10(max(base_mse, 1e-12)),
                    "train/mix_psnr": -10.0 * math.log10(max(mix_mse, 1e-12)),
                    "train/native_psnr": -10.0 * math.log10(max(native_mse, 1e-12)),
                    "train/lr": current_lr,
                    "train/lr_lg": current_lr_lg,
                    "train/lr_router": compute_lr(args, step, base_lr=args.lr_router, start_step=start_step),
                    "train/lr_d": current_lr_d,
                    "train/sec_per_step": sec,
                    "train/phase_router_only": float(router_only_active),
                    "train/phase_g_freeze": float(g_freeze_active),
                })
                wandb_run.log(log_metrics, step=step)
            running = {key: 0.0 for key in running}
            count = 0
            start_time = time.time()

        if is_main and args.sample_every > 0 and (step == 1 or step % args.sample_every == 0):
            save_adapt_grid(
                out_dir / "samples" / f"step_{step:08d}.png",
                last["x_lg"],
                last["x_base"],
                last["x_mix"],
                last["x_native"],
                args.sample_images,
                args.llamagen_input_range,
            )

        save_steps = set(args.save_steps)
        trained_steps = step - start_step
        epoch_boundary_due = (
            args.latest_every_epoch
            and trained_steps > 0
            and trained_steps % steps_per_epoch == 0
        )
        fraction_save_every_steps = epoch_fraction_to_steps(args.save_epoch_fraction_every, steps_per_epoch)
        epoch_fraction_due = (
            fraction_save_every_steps > 0
            and trained_steps > 0
            and trained_steps % fraction_save_every_steps == 0
        )
        completed_epochs = trained_steps // steps_per_epoch if trained_steps > 0 else 0
        save_epoch_due = (
            args.save_epoch_every > 0
            and epoch_boundary_due
            and completed_epochs > 0
            and completed_epochs % args.save_epoch_every == 0
        )
        save_latest_due = (args.save_every > 0 and step % args.save_every == 0) or epoch_boundary_due or epoch_fraction_due
        save_periodic_step_due = (
            args.save_step_checkpoints
            and ((args.save_every > 0 and step % args.save_every == 0) or epoch_fraction_due)
        )
        save_explicit_step_due = step in save_steps
        if is_main and (save_latest_due or save_periodic_step_due or save_explicit_step_due or save_epoch_due):
            core = model.module if distributed else model
            payload = {
                "model": collect_trainable_state(core),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
                "step": step,
                "lecam_ema_real": lecam_ema_real.detach().cpu(),
                "lecam_ema_fake": lecam_ema_fake.detach().cpu(),
                "selector_threshold_state": float(selector_threshold_state),
                "selector_ratio_ema": float(selector_ratio_ema),
            }
            if ema is not None:
                payload["model_ema"] = ema.state_dict()
            core_d = discriminator.module if distributed and discriminator is not None else discriminator
            if core_d is not None:
                payload["discriminator"] = core_d.state_dict()
                payload["optimizer_d"] = optimizer_d.state_dict()
            if save_latest_due:
                torch.save(payload, out_dir / "latest.pt")
            if save_periodic_step_due or save_explicit_step_due:
                torch.save(payload, out_dir / f"step_{step:08d}.pt")
            if save_epoch_due:
                torch.save(payload, out_dir / f"epoch_{completed_epochs:04d}_step_{step:08d}.pt")

    if is_main:
        pbar.close()
        core = model.module if distributed else model
        payload = {
            "model": collect_trainable_state(core),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "step": args.max_steps,
            "lecam_ema_real": lecam_ema_real.detach().cpu(),
            "lecam_ema_fake": lecam_ema_fake.detach().cpu(),
            "selector_threshold_state": float(selector_threshold_state),
            "selector_ratio_ema": float(selector_ratio_ema),
        }
        if ema is not None:
            payload["model_ema"] = ema.state_dict()
        core_d = discriminator.module if distributed and discriminator is not None else discriminator
        if core_d is not None:
            payload["discriminator"] = core_d.state_dict()
            payload["optimizer_d"] = optimizer_d.state_dict()
        if args.final_save:
            torch.save(payload, out_dir / "latest.pt")
            print(f"saved {out_dir / 'latest.pt'}", flush=True)
        else:
            print("skipped final checkpoint save (--no-final-save)", flush=True)
    if wandb_run is not None:
        wandb_run.finish()
    if distributed:
        dist.destroy_process_group()


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--data-path", type=str, default="../ImageNet/train")
    parser.add_argument("--dataset-manifest", type=str, default="")
    parser.add_argument("--rebuild-dataset-manifest", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output-dir", type=str, default="results/titok_llamagen_decoder_adapt_mix")
    parser.add_argument("--adapter-init", type=str, default="")
    parser.add_argument("--adapter-init-ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--resume-use-ema-model", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume-non-strict", action="store_true", default=False)
    parser.add_argument("--resume-skip-prefixes", nargs="*", default=[])
    parser.add_argument("--reset-optimizer", action="store_true", default=False)
    parser.add_argument("--reset-discriminator", action="store_true", default=False)
    parser.add_argument("--reset-selector-state", action="store_true", default=False)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--random-crop", action="store_true", default=False)
    parser.add_argument("--random-flip", action="store_true", default=False)
    parser.add_argument("--titok-input-range", type=str, default="zero_1", choices=["zero_1", "minus1_1"])
    parser.add_argument("--llamagen-input-range", type=str, default="minus1_1", choices=["zero_1", "minus1_1"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--epochs", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--limit-samples", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.0)
    parser.add_argument("--lr-llamagen", type=float, default=1e-5)
    parser.add_argument("--lr-llamagen-encoder", type=float, default=1e-6)
    parser.add_argument("--lr-router", type=float, default=1e-4)
    parser.add_argument("--lr-scheduler", type=str, default="constant", choices=["constant", "cosine"])
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--warmup-epochs", type=float, default=0.0)
    parser.add_argument("--min-lr", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--weight-decay-llamagen", type=float, default=0.0)
    parser.add_argument("--weight-decay-router", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--mixed-precision", type=str, default="bf16", choices=["bf16", "fp16", "none"])
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--train-adapter", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--train-llamagen-encoder", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--train-llamagen-quant-conv", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--train-llamagen-quantizer", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--train-post-quant-conv", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-llamagen-decoder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--llamagen-decoder-train-last-n", type=int, default=-1)
    parser.add_argument("--train-router", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--image-loss", type=str, default="l1", choices=["l1", "l2"])
    parser.add_argument("--perceptual-loss", type=str, default="lpips", choices=["lpips", "convnext_s", "none"])
    parser.add_argument("--lambda-perceptual", type=float, default=0.3)
    parser.add_argument("--lambda-lpips", type=float, default=0.3)
    parser.add_argument("--lambda-base", type=float, default=1.0)
    parser.add_argument("--lambda-mix", type=float, default=1.0)
    parser.add_argument("--lambda-native", type=float, default=0.5)
    parser.add_argument("--lambda-mix-native", type=float, default=0.0)
    parser.add_argument("--lambda-mix-native-perceptual", type=float, default=0.0)
    parser.add_argument("--mix-native-teacher", type=str, default="shared", choices=["shared", "frozen"])
    parser.add_argument("--lambda-feat", type=float, default=0.0)
    parser.add_argument("--lambda-feat-moment", type=float, default=0.0)
    parser.add_argument("--lambda-disc-feature-matching", type=float, default=0.0)
    parser.add_argument("--lambda-lowfreq-anchor", type=float, default=0.0)
    parser.add_argument("--lowfreq-anchor-size", type=int, default=32)
    parser.add_argument("--lambda-dino-feat", type=float, default=0.0)
    parser.add_argument("--lambda-clip-feat", type=float, default=0.0)
    parser.add_argument("--clip-model", type=str, default="ViT-B-32")
    parser.add_argument("--clip-pretrained", type=str, default="openai")
    parser.add_argument("--clip-cache-dir", type=str, default=None)
    parser.add_argument("--clip-prefer-hf-hub", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clip-weights-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clip-feat-loss", type=str, default="l1", choices=["l1", "l2"])
    parser.add_argument("--clip-feat-input-size", type=int, default=None)
    parser.add_argument("--clip-feat-normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dino-feat-loss", type=str, default="l1", choices=["l1", "l2"])
    parser.add_argument("--dino-feat-input-size", type=int, default=224)
    parser.add_argument("--dino-feat-use-patch-tokens", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dino-feat-normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lambda-router-budget", type=float, default=10.0)
    parser.add_argument("--lambda-router-binary", type=float, default=0.01)
    parser.add_argument("--lambda-router-ratio-target", type=float, default=0.0)
    parser.add_argument("--lambda-gan", type=float, default=0.0)
    parser.add_argument("--gan-start-step", type=int, default=0)
    parser.add_argument("--gan-start-epoch", type=float, default=-1.0)
    parser.add_argument("--gan-ramp-steps", type=int, default=0)
    parser.add_argument("--gan-g-ramp-after-d-warmup", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gan-input-filter", type=str, default="none", choices=["none", "highfreq_composite", "highfreq_grad_only"])
    parser.add_argument("--gan-highpass-size", type=int, default=64)
    parser.add_argument("--discriminator-factor", type=float, default=1.0)
    parser.add_argument("--lr-d", type=float, default=1.0e-5)
    parser.add_argument("--lr-d-scheduler", type=str, default="constant", choices=["constant", "cosine"])
    parser.add_argument("--min-lr-d", type=float, default=0.0)
    parser.add_argument("--d-every", type=int, default=1)
    parser.add_argument("--d-warmup-steps", type=int, default=0)
    parser.add_argument("--d-warmup-epochs", type=float, default=0.0)
    parser.add_argument("--g-freeze-steps", type=int, default=0)
    parser.add_argument("--g-freeze-epochs", type=float, default=0.0)
    parser.add_argument("--lecam-regularization-weight", type=float, default=0.001)
    parser.add_argument("--lecam-ema-decay", type=float, default=0.999)
    parser.add_argument("--discriminator-type", type=str, default="patch", choices=["patch", "multiscale_patch", "dino", "dinov1s", "patch_dino", "multiscale_patch_dino", "projected_convnext", "projected_gan", "official_projected_gan", "stylegan"])
    parser.add_argument("--stylegan-channel-multiplier", type=int, default=1)
    parser.add_argument("--projected-input-size", type=int, default=224)
    parser.add_argument("--projected-head-hidden", type=int, default=256)
    parser.add_argument("--projected-head-depth", type=int, default=2)
    parser.add_argument("--projected-backbone", type=str, default="convnext_small", choices=["convnext_small", "convnext_base"])
    parser.add_argument("--projected-channels", type=int, default=256)
    parser.add_argument("--projected-normalize-features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--projected-loss-weights", type=float, nargs="*", default=[1.0, 1.0, 1.0, 1.0])
    parser.add_argument("--projected-pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--official-pg-input-size", type=int, default=224)
    parser.add_argument("--official-pg-im-res", type=int, default=256)
    parser.add_argument("--official-pg-cout", type=int, default=64)
    parser.add_argument("--official-pg-expand", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--official-pg-proj-type", type=int, default=2, choices=[0, 1, 2])
    parser.add_argument("--official-pg-num-discs", type=int, default=4, choices=[1, 2, 3, 4])
    parser.add_argument("--official-pg-separable", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--official-pg-patch", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--official-pg-pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--official-pg-diffaug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--official-pg-diffaug-policy", type=str, default="color,translation,cutout")
    parser.add_argument("--disc-scales", type=float, nargs="*", default=[1.0])
    parser.add_argument("--disc-loss-weights", type=float, nargs="*", default=[1.0])
    parser.add_argument("--disc-hidden-channels", type=int, default=128)
    parser.add_argument("--disc-num-stages", type=int, default=3)
    parser.add_argument("--dino-repo", type=str, default="../.cache/torch/hub/facebookresearch_dinov2_main")
    parser.add_argument("--dino-model", type=str, default="dinov2_vits14")
    parser.add_argument("--dino-ckpt", type=str, default=None)
    parser.add_argument("--dino-input-size", type=int, default=224)
    parser.add_argument("--dino-loss-weight", type=float, default=0.25)
    parser.add_argument("--dino-head-hidden", type=int, default=256)
    parser.add_argument("--dino-use-patch-tokens", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mask-ratio-min", type=float, default=0.1)
    parser.add_argument("--mask-ratio-max", type=float, default=0.5)
    parser.add_argument(
        "--mask-selection",
        type=str,
        default="per_image_random",
        choices=["per_image_random", "accum_global_error", "accum_global_mixed_score", "accum_global_mixed_score_threshold", "router_e2e_dynamic", "router_e2e_top_p"],
    )
    parser.add_argument("--mask-ratio", type=float, default=0.5)
    parser.add_argument("--mask-error-weight", type=float, default=0.6)
    parser.add_argument("--mask-gradient-weight", type=float, default=0.3)
    parser.add_argument("--mask-variance-weight", type=float, default=0.1)
    parser.add_argument("--selector-threshold", type=float, default=0.3)
    parser.add_argument("--target-mask-ratio", type=float, default=0.5)
    parser.add_argument("--use-threshold-ema-controller", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--selector-threshold-lr", type=float, default=0.005)
    parser.add_argument("--selector-ratio-ema-decay", type=float, default=0.95)
    parser.add_argument("--selector-threshold-min", type=float, default=0.0)
    parser.add_argument("--selector-threshold-max", type=float, default=1.0)
    parser.add_argument("--score-normalize-scope", type=str, default="global_pool", choices=["global_pool", "per_image"])
    parser.add_argument("--router-hidden-dim", type=int, default=128)
    parser.add_argument("--router-depth", type=int, default=3)
    parser.add_argument("--router-min-ratio", type=float, default=0.05)
    parser.add_argument("--router-max-ratio", type=float, default=0.90)
    parser.add_argument("--router-target-mean-ratio", type=float, default=0.50)
    parser.add_argument("--router-budget-mode", type=str, default="soft", choices=["soft", "batch_exact", "bank"])
    parser.add_argument("--router-exact-budget-ratio", type=float, default=-1.0)
    parser.add_argument("--router-budget-bank-max-debt-tokens", type=float, default=1024.0)
    parser.add_argument("--router-ratio-target-mean-calibrate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--router-ratio-target-calibrate-iters", type=int, default=24)
    parser.add_argument("--router-min-tokens", type=int, default=-1)
    parser.add_argument("--router-max-tokens", type=int, default=-1)
    parser.add_argument("--router-tau", type=float, default=0.7)
    parser.add_argument("--router-top-p", type=float, default=0.95)
    parser.add_argument("--router-top-p-temperature", type=float, default=1.0)
    parser.add_argument("--router-top-p-ste-tau", type=float, default=0.01)
    parser.add_argument("--router-detach-inputs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--router-gain-aux-weight", type=float, default=0.1)
    parser.add_argument("--router-gain-aux-decay-steps", type=int, default=5000)
    parser.add_argument("--router-gain-aux-target-ratio", type=float, default=0.5)
    parser.add_argument("--router-ratio-target-spread", type=float, default=0.0)
    parser.add_argument("--router-only-steps", type=int, default=0)
    parser.add_argument("--router-only-epochs", type=float, default=0.0)
    parser.add_argument("--router-only-disable-gan", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wandb-project", type=str, default="MoT")
    parser.add_argument("--wandb-name", type=str, default="")
    parser.add_argument("--wandb-entity", type=str, default="")
    parser.add_argument("--wandb-mode", type=str, default="")
    parser.add_argument("--wandb-tags", nargs="*", default=[])
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--latest-every-epoch", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-epoch-every", type=int, default=0)
    parser.add_argument("--save-epoch-fraction-every", type=float, default=0.0)
    parser.add_argument("--save-step-checkpoints", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--final-save", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-steps", type=int, nargs="*", default=[])
    parser.add_argument("--sample-every", type=int, default=200)
    parser.add_argument("--sample-images", type=int, default=8)
    parser.add_argument("--lg-latent-channels", type=int, default=256)
    parser.add_argument("--lg-head-channels", type=int, default=256)
    parser.add_argument("--latent-head-mode", type=str, default="feature", choices=["feature", "codebook"])
    parser.add_argument("--codebook-temperature", type=float, default=1.0)
    parser.add_argument("--titok-root", type=str, default="../1d-tokenizer")
    parser.add_argument("--titok-config", type=str, default="../1d-tokenizer/configs/infer/TiTok/titok_l32.yaml")
    parser.add_argument("--titok-ckpt", type=str, default="../1d-tokenizer/tokenizer_titok_l32.bin")
    parser.add_argument("--llamagen-root", type=str, default="../LlamaGen")
    parser.add_argument("--llamagen-ckpt", type=str, default="../LlamaGen/pretrained_models/vq_ds16_c2i.pt")
    parser.add_argument("--codebook-size", type=int, default=16384)
    parser.add_argument("--codebook-embed-dim", type=int, default=8)
    return parser


def parse_args():
    parser = build_parser()
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None)
    config_args, remaining = config_parser.parse_known_args()

    defaults = vars(parser.parse_args([]))
    config_values = {}
    if config_args.config is not None:
        config = OmegaConf.to_container(OmegaConf.load(config_args.config), resolve=True)
        if not isinstance(config, dict):
            raise ValueError(f"config must contain a mapping, got {type(config).__name__}")
        config_values = {key.replace("-", "_"): value for key, value in config.items()}
        unknown = sorted(set(config_values) - set(defaults))
        if unknown:
            raise ValueError(f"unknown config keys: {unknown}")
    parser.set_defaults(**config_values)
    args = parser.parse_args(remaining)
    args.config = config_args.config
    return args


if __name__ == "__main__":
    main(parse_args())
