#!/usr/bin/env python3
"""Adapt LlamaGen decoder with global gain-first texture grid selection."""

from __future__ import annotations

import argparse
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
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from models import TiTokLlamaGenStage2
from train_titok_llamagen_recon import (
    TrainableEMA,
    adamw_param_groups,
    autocast_dtype,
    build_discriminator,
    build_perceptual_loss,
    chw_to_pil,
    compute_feature_moment_loss,
    compute_image_loss,
    compute_lecam_loss,
    compute_lr,
    compute_perceptual_loss,
    convert_image_range,
    denorm_to_uint8,
    discriminator_input,
    gan_factor_for_step,
    get_perceptual_weight,
    hinge_d_loss,
    image_to_zero_one,
    load_llamagen_vq,
    load_titok,
    make_transform,
    set_optimizer_lr,
    set_requires_grad,
)


def format_run_header(args, dataset_len, world_size, trainable_params):
    effective_batch = args.batch_size * world_size * args.accum_steps
    summary = (
        f"dataset={dataset_len} world_size={world_size} batch_size_per_gpu={args.batch_size} "
        f"accum_steps={args.accum_steps} global_batch={effective_batch} trainable_params={trainable_params} "
        f"train_adapter={args.train_adapter} train_llamagen_encoder={args.train_llamagen_encoder} "
        f"train_llamagen_quant_conv={args.train_llamagen_quant_conv} train_llamagen_quantizer={args.train_llamagen_quantizer} "
        f"train_post_quant_conv={args.train_post_quant_conv} train_llamagen_decoder={args.train_llamagen_decoder} "
        f"lr_adapter={args.lr} lr_lg_encoder={args.lr_llamagen_encoder} lr_llamagen={args.lr_llamagen} "
        f"loss=image({args.image_loss}) base:{args.lambda_base},mix:{args.lambda_mix},native:{args.lambda_native},"
        f"perceptual({args.perceptual_loss}):{get_perceptual_weight(args)},feat:{args.lambda_feat},"
        f"feat_moment:{args.lambda_feat_moment} gan:{args.lambda_gan}@{args.gan_start_step}+ramp{args.gan_ramp_steps},"
        f"d_every:{args.d_every},d_warmup:{args.d_warmup_steps},lecam:{args.lecam_regularization_weight} "
        f"mask_selection:{args.mask_selection} mask_ratio:{args.mask_ratio} "
        f"mask_score_weights:error={args.mask_error_weight},gradient={args.mask_gradient_weight},"
        f"variance={args.mask_variance_weight} "
        f"gain_texture_alpha={args.gain_texture_alpha} "
        f"gain_texture_weights:gradient={args.gain_gradient_weight},variance={args.gain_variance_weight} "
        f"blend_gain_weight={args.blend_gain_weight} "
        f"mask_ratio_range:{args.mask_ratio_min}-{args.mask_ratio_max} "
        f"augment=random_crop:{args.random_crop},random_flip:{args.random_flip} "
        f"ema:{args.use_ema}@{args.ema_decay} adapter_init={args.adapter_init}"
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


def configure_trainable_parts(model: TiTokLlamaGenStage2, args):
    model.titok.eval().requires_grad_(False)
    model.latent_decoder.requires_grad_(args.train_adapter)
    model.llamagen_vq.eval().requires_grad_(False)
    set_module_trainable(getattr(model.llamagen_vq, "encoder", None), args.train_llamagen_encoder)
    set_module_trainable(getattr(model.llamagen_vq, "quantize", None), args.train_llamagen_quantizer)
    set_module_trainable(getattr(model.llamagen_vq, "quant_conv", None), args.train_llamagen_quant_conv)
    set_module_trainable(getattr(model.llamagen_vq, "post_quant_conv", None), args.train_post_quant_conv)
    set_module_trainable(getattr(model.llamagen_vq, "decoder", None), args.train_llamagen_decoder)
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
    if not groups:
        raise ValueError("no trainable parameter groups; enable at least one train_* flag")
    return groups


def trainable_params(core: TiTokLlamaGenStage2):
    return [param for param in core.parameters() if param.requires_grad]


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
def global_mean_value(value):
    value = value.detach().float()
    stats = torch.stack([value.sum(), value.new_tensor(float(value.numel()))])
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    return float((stats[0] / stats[1].clamp_min(1.0)).item())


@torch.no_grad()
def mixed_score_from_raw_global_pool(raw_parts, args):
    norm_parts = {
        "error": normalize_score_global_pool(raw_parts["error"]),
        "gradient": normalize_score_global_pool(raw_parts["gradient"]),
        "variance": normalize_score_global_pool(raw_parts["variance"]),
    }
    score = (
        float(args.mask_error_weight) * norm_parts["error"]
        + float(args.mask_gradient_weight) * norm_parts["gradient"]
        + float(args.mask_variance_weight) * norm_parts["variance"]
    )
    return score, norm_parts


@torch.no_grad()
def grid_mse_gain_score(x_base, x_native, target, grid_hw=16):
    base_err = (x_base.detach().float() - target.detach().float()).pow(2).mean(dim=1, keepdim=True)
    native_err = (x_native.detach().float() - target.detach().float()).pow(2).mean(dim=1, keepdim=True)
    return F.adaptive_avg_pool2d(base_err - native_err, (grid_hw, grid_hw)).clamp_min(0.0)


@torch.no_grad()
def gain_first_texture_score_from_raw_global_pool(raw_parts, args):
    norm_parts = {
        "gain": normalize_score_global_pool(raw_parts["gain"]),
        "gradient": normalize_score_global_pool(raw_parts["gradient"]),
        "variance": normalize_score_global_pool(raw_parts["variance"]),
    }
    texture = (
        float(args.gain_gradient_weight) * norm_parts["gradient"]
        + float(args.gain_variance_weight) * norm_parts["variance"]
    )
    score = norm_parts["gain"] * (1.0 + float(args.gain_texture_alpha) * texture)
    norm_parts["texture"] = texture
    return score, norm_parts


@torch.no_grad()
def gain_mixed_blend_score_from_raw_global_pool(raw_parts, args):
    mixed_raw_parts = {
        "error": raw_parts["error"],
        "gradient": raw_parts["gradient"],
        "variance": raw_parts["variance"],
    }
    mixed_score, mixed_norm_parts = mixed_score_from_raw_global_pool(mixed_raw_parts, args)
    gain_raw_parts = {
        "gain": raw_parts["gain"],
        "gradient": raw_parts["gradient"],
        "variance": raw_parts["variance"],
    }
    gain_score, gain_norm_parts = gain_first_texture_score_from_raw_global_pool(gain_raw_parts, args)
    score = (
        (1.0 - float(args.blend_gain_weight)) * normalize_score_global_pool(mixed_score)
        + float(args.blend_gain_weight) * normalize_score_global_pool(gain_score)
    )
    return score, {
        "error": mixed_norm_parts["error"],
        "gradient": mixed_norm_parts["gradient"],
        "variance": mixed_norm_parts["variance"],
        "gain": gain_norm_parts["gain"],
    }


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
def per_image_top_ratio_mask_from_score(score, ratio):
    flat_score = score.detach().float().flatten(1)
    batch_size, num_tokens = flat_score.shape
    k = int(round(float(ratio) * num_tokens))
    k = max(0, min(k, num_tokens))
    flat_mask = torch.zeros_like(flat_score)
    if k > 0:
        topk = torch.topk(flat_score, k, dim=1, largest=True).indices
        flat_mask.scatter_(1, topk, 1.0)
    return flat_mask.view_as(score).view(batch_size, 1, 16, 16)


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


def load_adapt_resume(core: TiTokLlamaGenStage2, ckpt, strict=True):
    state = ckpt.get("model", ckpt)
    if not isinstance(state, dict):
        raise ValueError("resume checkpoint does not contain a state dict")
    current = dict(core.named_parameters())
    missing = []
    unexpected = []
    with torch.no_grad():
        for name, value in state.items():
            if name not in current:
                unexpected.append(name)
                continue
            current[name].copy_(value.to(device=current[name].device, dtype=current[name].dtype))
        if strict:
            for name, param in current.items():
                if param.requires_grad and name not in state:
                    missing.append(name)
    if strict and (missing or unexpected):
        raise RuntimeError(f"resume mismatch missing={missing} unexpected={unexpected}")
    return missing, unexpected


def compute_path_losses(perceptual, perceptual_name, pred, target, image_loss):
    image = compute_image_loss(pred, target, image_loss)
    perc = compute_perceptual_loss(perceptual, perceptual_name, pred, target)
    mse01 = F.mse_loss(image_to_zero_one(pred, "minus1_1").float(), image_to_zero_one(target, "minus1_1").float())
    return image, perc, mse01


def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training")
    distributed, rank, world_size, local_rank, device = distributed_setup()
    is_main = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    dataset = ImageFolder(args.data_path, transform=make_transform(args.image_size, args.random_crop, args.random_flip))
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

    titok = load_titok(args.titok_root, args.titok_config, args.titok_ckpt, device)
    vq_model = load_llamagen_vq(args.llamagen_root, args.llamagen_ckpt, device, args.codebook_size, args.codebook_embed_dim)
    perceptual = build_perceptual_loss(args, device)

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

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        load_adapt_resume(core, ckpt, strict=not args.resume_non_strict)
        start_step = int(ckpt.get("step", 0)) if isinstance(ckpt, dict) else 0
        if isinstance(ckpt, dict) and "optimizer" in ckpt and not args.reset_optimizer:
            optimizer.load_state_dict(ckpt["optimizer"])
        if (
            discriminator is not None
            and isinstance(ckpt, dict)
            and "discriminator" in ckpt
            and not args.reset_discriminator
        ):
            discriminator.load_state_dict(ckpt["discriminator"], strict=True)
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
        if ema is not None and isinstance(ckpt, dict) and "model_ema" in ckpt:
            ema.load_state_dict(ckpt["model_ema"], device=device)
        if is_main:
            print(f"resumed decoder-adapt checkpoint from {args.resume} at step {start_step}", flush=True)

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

    if is_main:
        core = model.module if distributed else model
        n_trainable = sum(p.numel() for p in core.parameters() if p.requires_grad)
        msg, run_header = format_run_header(args, len(dataset), world_size, n_trainable)
        print(msg, flush=True)
        with log_path.open("a") as f:
            f.write(run_header)

    running = {
        "loss": 0.0,
        "base": 0.0,
        "base_lp": 0.0,
        "base_mse01": 0.0,
        "mix": 0.0,
        "mix_lp": 0.0,
        "mix_mse01": 0.0,
        "native": 0.0,
        "native_lp": 0.0,
        "native_mse01": 0.0,
        "feat": 0.0,
        "feat_moment": 0.0,
        "gan_g": 0.0,
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
        "grad": 0.0,
    }
    count = 0
    start_time = time.time()
    pbar = tqdm(total=max(0, args.max_steps - start_step), desc=f"Steps {start_step + 1}-{args.max_steps}", disable=not is_main, dynamic_ncols=True, mininterval=2)
    d_warmup_end_step = start_step + max(0, args.d_warmup_steps) if args.reset_discriminator else 0

    for step in range(start_step + 1, args.max_steps + 1):
        set_adapt_train_mode(model, args)
        core = model.module if distributed else model
        current_lr = compute_lr(args, step, base_lr=args.lr)
        current_lr_lg = compute_lr(args, step, base_lr=args.lr_llamagen)
        current_lr_d = compute_lr(args, step, base_lr=args.lr_d, start_step=args.gan_start_step)
        for group in optimizer.param_groups:
            if group.get("lr_role") == "llamagen":
                group["lr"] = current_lr_lg
            elif group.get("lr_role") == "llamagen_encoder":
                group["lr"] = compute_lr(args, step, base_lr=args.lr_llamagen_encoder)
            else:
                group["lr"] = current_lr
        if optimizer_d is not None:
            set_optimizer_lr(optimizer_d, current_lr_d)
        gan_factor = gan_factor_for_step(args, step)
        d_warmup_active = discriminator is not None and args.d_warmup_steps > 0 and step <= d_warmup_end_step
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
            }
        elif args.mask_selection in ("accum_global_gain_first_texture", "per_image_gain_first_texture"):
            raw_chunks = {"gain": [], "gradient": [], "variance": []}
            lengths = []
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=autocast_dtype(args.mixed_precision), enabled=args.mixed_precision != "none"):
                    for _x_01_scan, x_titok_scan, x_lg_scan in accum_batches:
                        x_base_scan, _extra_scan = core(x_titok_scan)
                        f_2d_scan, _ = native_llamagen_feature(
                            core.llamagen_vq,
                            x_lg_scan,
                            args.codebook_embed_dim,
                            allow_encoder_grad=False,
                        )
                        x_native_scan = core.llamagen_vq.decoder(f_2d_scan)
                        raw_chunks["gain"].append(grid_mse_gain_score(x_base_scan, x_native_scan, x_lg_scan, grid_hw=16))
                        raw_chunks["gradient"].append(gradient_score_from_image_grid(x_lg_scan, grid_hw=16))
                        raw_chunks["variance"].append(local_variance_score_from_image_grid(x_lg_scan, grid_hw=16))
                        lengths.append(x_lg_scan.shape[0])
            raw_cat = {key: torch.cat(chunks, dim=0) for key, chunks in raw_chunks.items()}
            score_cat, norm_parts = gain_first_texture_score_from_raw_global_pool(raw_cat, args)
            if args.mask_selection == "per_image_gain_first_texture":
                mask_cat = per_image_top_ratio_mask_from_score(score_cat, args.mask_ratio)
            else:
                mask_cat = global_top_ratio_mask_from_score(score_cat, args.mask_ratio)
            precomputed_masks = list(torch.split(mask_cat, lengths, dim=0))
            score_metric_values = {
                "score": global_mean_value(score_cat),
                "score_error": global_mean_value(norm_parts["gain"]),
                "score_gradient": global_mean_value(norm_parts["gradient"]),
                "score_variance": global_mean_value(norm_parts["variance"]),
            }
        elif args.mask_selection in ("accum_global_gain_mixed_blend", "per_image_gain_mixed_blend"):
            raw_chunks = {"error": [], "gain": [], "gradient": [], "variance": []}
            lengths = []
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=autocast_dtype(args.mixed_precision), enabled=args.mixed_precision != "none"):
                    for _x_01_scan, x_titok_scan, x_lg_scan in accum_batches:
                        x_base_scan, _extra_scan = core(x_titok_scan)
                        f_2d_scan, _ = native_llamagen_feature(
                            core.llamagen_vq,
                            x_lg_scan,
                            args.codebook_embed_dim,
                            allow_encoder_grad=False,
                        )
                        x_native_scan = core.llamagen_vq.decoder(f_2d_scan)
                        raw_chunks["error"].append(spatial_error_score_grid(x_base_scan, x_lg_scan, grid_hw=16))
                        raw_chunks["gain"].append(grid_mse_gain_score(x_base_scan, x_native_scan, x_lg_scan, grid_hw=16))
                        raw_chunks["gradient"].append(gradient_score_from_image_grid(x_lg_scan, grid_hw=16))
                        raw_chunks["variance"].append(local_variance_score_from_image_grid(x_lg_scan, grid_hw=16))
                        lengths.append(x_lg_scan.shape[0])
            raw_cat = {key: torch.cat(chunks, dim=0) for key, chunks in raw_chunks.items()}
            score_cat, norm_parts = gain_mixed_blend_score_from_raw_global_pool(raw_cat, args)
            if args.mask_selection == "per_image_gain_mixed_blend":
                mask_cat = per_image_top_ratio_mask_from_score(score_cat, args.mask_ratio)
            else:
                mask_cat = global_top_ratio_mask_from_score(score_cat, args.mask_ratio)
            precomputed_masks = list(torch.split(mask_cat, lengths, dim=0))
            score_metric_values = {
                "score": global_mean_value(score_cat),
                "score_error": global_mean_value(norm_parts["error"]),
                "score_gradient": global_mean_value(norm_parts["gradient"]),
                "score_variance": global_mean_value(norm_parts["variance"]),
            }

        for _micro_step, (_x_01, x_titok, x_lg) in enumerate(accum_batches):
            with torch.autocast(device_type="cuda", dtype=autocast_dtype(args.mixed_precision), enabled=args.mixed_precision != "none"):
                x_base, extra = model(x_titok)
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

                if precomputed_masks[_micro_step] is None:
                    mask = oracle_error_mask(x_base, x_lg, args.mask_ratio_min, args.mask_ratio_max).to(dtype=f_1d_lg.dtype)
                else:
                    mask = precomputed_masks[_micro_step].to(device=f_1d_lg.device, dtype=f_1d_lg.dtype)
                f_mix = (1.0 - mask) * f_1d_lg + mask * f_2d_lg
                x_mix = core.llamagen_vq.decoder(f_mix)
                x_native = core.llamagen_vq.decoder(f_2d_lg)

                base_img, base_lp, base_mse01 = compute_path_losses(perceptual, args.perceptual_loss, x_base, x_lg, args.image_loss)
                mix_img, mix_lp, mix_mse01 = compute_path_losses(perceptual, args.perceptual_loss, x_mix, x_lg, args.image_loss)
                native_img, native_lp, native_mse01 = compute_path_losses(perceptual, args.perceptual_loss, x_native, x_lg, args.image_loss)
                feat_loss = F.l1_loss(f_1d_lg.float(), f_2d_lg.detach().float())
                feat_moment = compute_feature_moment_loss(f_1d_lg.float(), f_2d_lg.detach().float())

                perceptual_weight = get_perceptual_weight(args)
                base_loss = args.lambda_base * (base_img + perceptual_weight * base_lp)
                mix_loss = args.lambda_mix * (mix_img + perceptual_weight * mix_lp)
                native_loss = args.lambda_native * (native_img + perceptual_weight * native_lp)
                gan_g_loss = x_mix.new_zeros(())
                if discriminator is not None and gan_factor > 0.0 and not d_warmup_active:
                    set_requires_grad(discriminator, False)
                    logits_fake_for_g = discriminator(discriminator_input(x_mix, args.llamagen_input_range))
                    gan_g_loss = -torch.mean(logits_fake_for_g)
                loss = (
                    base_loss
                    + mix_loss
                    + native_loss
                    + args.lambda_feat * feat_loss
                    + args.lambda_feat_moment * feat_moment
                    + args.lambda_gan * gan_factor * gan_g_loss
                )

            (loss / args.accum_steps).backward()

            d_loss = x_mix.new_zeros(())
            lecam_loss = x_mix.new_zeros(())
            logits_real_mean = x_mix.new_zeros(())
            logits_fake_mean = x_mix.new_zeros(())
            if train_discriminator:
                set_requires_grad(discriminator, True)
                real_for_d = discriminator_input(x_lg, args.llamagen_input_range).detach()
                fake_for_d = discriminator_input(x_mix.detach(), args.llamagen_input_range)
                logits_both = discriminator(torch.cat([real_for_d, fake_for_d], dim=0))
                logits_real, logits_fake = logits_both.chunk(2, dim=0)
                logits_real_mean = torch.mean(logits_real)
                logits_fake_mean = torch.mean(logits_fake)
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
            metric_sums["native"] += native_img.detach().float().item()
            metric_sums["native_lp"] += native_lp.detach().float().item()
            metric_sums["native_mse01"] += native_mse01.detach().float().item()
            metric_sums["feat"] += feat_loss.detach().float().item()
            metric_sums["feat_moment"] += feat_moment.detach().float().item()
            metric_sums["gan_g"] += (args.lambda_gan * gan_factor * gan_g_loss).detach().float().item()
            metric_sums["d_loss"] += d_loss.detach().float().item()
            metric_sums["lecam"] += lecam_loss.detach().float().item()
            mask_tokens = mask.detach().float().flatten(1).sum(dim=1)
            metric_sums["logits_real"] += logits_real_mean.detach().float().item()
            metric_sums["logits_fake"] += logits_fake_mean.detach().float().item()
            metric_sums["mask"] += mask.detach().float().mean().item()
            metric_sums["mask_tokens"] += mask_tokens.mean().item()
            metric_sums["mask_tokens_std"] += mask_tokens.std(unbiased=False).item()
            metric_sums["mask_tokens_min"] += mask_tokens.min().item()
            metric_sums["mask_tokens_max"] += mask_tokens.max().item()
            for score_key, score_value in score_metric_values.items():
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

        grad_norm = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
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
                metric_sums["native"] / args.accum_steps,
                metric_sums["native_lp"] / args.accum_steps,
                metric_sums["native_mse01"] / args.accum_steps,
                metric_sums["feat"] / args.accum_steps,
                metric_sums["feat_moment"] / args.accum_steps,
                metric_sums["gan_g"] / args.accum_steps,
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
                grad_norm.detach().float().item(),
            ],
            device=device,
            dtype=torch.float32,
        )
        if distributed:
            dist.all_reduce(vals, op=dist.ReduceOp.AVG)
        keys = [
            "loss", "base", "base_lp", "base_mse01", "mix", "mix_lp", "mix_mse01",
            "native", "native_lp", "native_mse01", "feat", "feat_moment",
            "gan_g", "d_loss", "lecam", "logits_real", "logits_fake",
            "mask", "mask_tokens", "mask_tokens_std", "mask_tokens_min", "mask_tokens_max",
            "score", "score_error", "score_gradient", "score_variance", "grad",
        ]
        for key, value in zip(keys, vals.tolist()):
            running[key] += value
        count += 1
        denom = max(count, 1)
        sec = (time.time() - start_time) / denom
        base_mse = running["base_mse01"] / denom
        mix_mse = running["mix_mse01"] / denom
        native_mse = running["native_mse01"] / denom

        if is_main:
            pbar.update(1)
            pbar.set_postfix({
                "step": step,
                "mix_l1": f"{running['mix'] / denom:.3f}",
                "mix_lp": f"{running['mix_lp'] / denom:.3f}",
                "psnr": f"{(-10.0 * math.log10(max(mix_mse, 1e-12))):.2f}",
                "base": f"{(-10.0 * math.log10(max(base_mse, 1e-12))):.2f}",
                "mask": f"{running['mask'] / denom:.2f}",
                "tok": f"{running['mask_tokens'] / denom:.0f}",
                "gan": f"{running['gan_g'] / denom:.3f}",
                "d": f"{running['d_loss'] / denom:.3f}",
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
                    f"sel_score {running['score']/denom:.3f} err_score {running['score_error']/denom:.3f} "
                    f"grad_score {running['score_gradient']/denom:.3f} var_score {running['score_variance']/denom:.3f}"
                )
            msg = (
                f"Step {step:08d} | loss {running['loss']/denom:.5f} | "
                f"base {running['base']/denom:.5f} lp {running['base_lp']/denom:.5f} psnr {(-10.0 * math.log10(max(base_mse, 1e-12))):.2f} | "
                f"mix {running['mix']/denom:.5f} lp {running['mix_lp']/denom:.5f} psnr {(-10.0 * math.log10(max(mix_mse, 1e-12))):.2f} | "
                f"native {running['native']/denom:.5f} lp {running['native_lp']/denom:.5f} psnr {(-10.0 * math.log10(max(native_mse, 1e-12))):.2f} | "
                f"feat {running['feat']/denom:.5f} feat_moment {running['feat_moment']/denom:.5f} | "
                f"gan_g {running['gan_g']/denom:.5f} | d {running['d_loss']/denom:.5f} | "
                f"lecam {running['lecam']/denom:.5f} | d_real {running['logits_real']/denom:.4f} | "
                f"d_fake {running['logits_fake']/denom:.4f} | "
                f"grad {running['grad']/denom:.4f} | lr {current_lr:.6g} lr_lg {current_lr_lg:.6g} lr_d {current_lr_d:.6g} | {sec:.3f}s/step | {stats}"
            )
            with log_path.open("a") as f:
                f.write(msg + "\n")
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
        save_latest_due = args.save_every > 0 and step % args.save_every == 0
        save_periodic_step_due = save_latest_due and args.save_step_checkpoints
        save_step_due = save_periodic_step_due or step in save_steps
        if is_main and (save_latest_due or save_step_due):
            core = model.module if distributed else model
            payload = {
                "model": collect_trainable_state(core),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
                "step": step,
                "lecam_ema_real": lecam_ema_real.detach().cpu(),
                "lecam_ema_fake": lecam_ema_fake.detach().cpu(),
            }
            if ema is not None:
                payload["model_ema"] = ema.state_dict()
            core_d = discriminator.module if distributed and discriminator is not None else discriminator
            if core_d is not None:
                payload["discriminator"] = core_d.state_dict()
                payload["optimizer_d"] = optimizer_d.state_dict()
            if save_latest_due:
                torch.save(payload, out_dir / "latest.pt")
            if save_step_due:
                torch.save(payload, out_dir / f"step_{step:08d}.pt")

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
        }
        if ema is not None:
            payload["model_ema"] = ema.state_dict()
        core_d = discriminator.module if distributed and discriminator is not None else discriminator
        if core_d is not None:
            payload["discriminator"] = core_d.state_dict()
            payload["optimizer_d"] = optimizer_d.state_dict()
        torch.save(payload, out_dir / "latest.pt")
        print(f"saved {out_dir / 'latest.pt'}", flush=True)
    if distributed:
        dist.destroy_process_group()


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--data-path", type=str, default="../ImageNet/train")
    parser.add_argument("--output-dir", type=str, default="results/titok_llamagen_decoder_adapt_mix")
    parser.add_argument("--adapter-init", type=str, default="")
    parser.add_argument("--adapter-init-ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--resume-non-strict", action="store_true", default=False)
    parser.add_argument("--reset-optimizer", action="store_true", default=False)
    parser.add_argument("--reset-discriminator", action="store_true", default=False)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--random-crop", action="store_true", default=False)
    parser.add_argument("--random-flip", action="store_true", default=False)
    parser.add_argument("--titok-input-range", type=str, default="zero_1", choices=["zero_1", "minus1_1"])
    parser.add_argument("--llamagen-input-range", type=str, default="minus1_1", choices=["zero_1", "minus1_1"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--limit-samples", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.0)
    parser.add_argument("--lr-llamagen", type=float, default=1e-5)
    parser.add_argument("--lr-llamagen-encoder", type=float, default=1e-6)
    parser.add_argument("--lr-scheduler", type=str, default="constant", choices=["constant", "cosine"])
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--min-lr", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--weight-decay-llamagen", type=float, default=0.0)
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
    parser.add_argument("--image-loss", type=str, default="l1", choices=["l1", "l2"])
    parser.add_argument("--perceptual-loss", type=str, default="lpips", choices=["lpips", "convnext_s", "none"])
    parser.add_argument("--lambda-perceptual", type=float, default=0.3)
    parser.add_argument("--lambda-lpips", type=float, default=0.3)
    parser.add_argument("--lambda-base", type=float, default=1.0)
    parser.add_argument("--lambda-mix", type=float, default=1.0)
    parser.add_argument("--lambda-native", type=float, default=0.5)
    parser.add_argument("--lambda-feat", type=float, default=0.0)
    parser.add_argument("--lambda-feat-moment", type=float, default=0.0)
    parser.add_argument("--lambda-gan", type=float, default=0.0)
    parser.add_argument("--gan-start-step", type=int, default=0)
    parser.add_argument("--gan-ramp-steps", type=int, default=0)
    parser.add_argument("--discriminator-factor", type=float, default=1.0)
    parser.add_argument("--lr-d", type=float, default=1.0e-5)
    parser.add_argument("--d-every", type=int, default=1)
    parser.add_argument("--d-warmup-steps", type=int, default=0)
    parser.add_argument("--lecam-regularization-weight", type=float, default=0.001)
    parser.add_argument("--lecam-ema-decay", type=float, default=0.999)
    parser.add_argument("--disc-hidden-channels", type=int, default=128)
    parser.add_argument("--disc-num-stages", type=int, default=3)
    parser.add_argument("--mask-ratio-min", type=float, default=0.1)
    parser.add_argument("--mask-ratio-max", type=float, default=0.5)
    parser.add_argument(
        "--mask-selection",
        type=str,
        default="per_image_random",
        choices=[
            "per_image_random",
            "accum_global_error",
            "accum_global_mixed_score",
            "accum_global_gain_first_texture",
            "accum_global_gain_mixed_blend",
            "per_image_gain_first_texture",
            "per_image_gain_mixed_blend",
        ],
    )
    parser.add_argument("--mask-ratio", type=float, default=0.5)
    parser.add_argument("--mask-error-weight", type=float, default=0.6)
    parser.add_argument("--mask-gradient-weight", type=float, default=0.3)
    parser.add_argument("--mask-variance-weight", type=float, default=0.1)
    parser.add_argument("--gain-texture-alpha", type=float, default=0.2)
    parser.add_argument("--gain-gradient-weight", type=float, default=0.5)
    parser.add_argument("--gain-variance-weight", type=float, default=0.5)
    parser.add_argument("--blend-gain-weight", type=float, default=0.8)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--save-step-checkpoints", action=argparse.BooleanOptionalAction, default=False)
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
