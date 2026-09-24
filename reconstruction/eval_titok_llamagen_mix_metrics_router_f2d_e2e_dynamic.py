#!/usr/bin/env python3
"""Evaluate TiTok/LlamaGen mix checkpoints with f2d-aware dynamic Router masks."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from tqdm import tqdm

from models import TiTokLlamaGenStage2
from train_titok_llamagen_decoder_adapt_router_f2d_e2e_dynamic import (
    DynamicBudgetRouter,
    make_dynamic_budget_ste_mask,
    make_top_p_ste_mask,
)
from train_titok_llamagen_decoder_adapt_global_gain_texture import (
    global_mean_value,
    gain_first_texture_score_from_raw_global_pool,
    global_top_ratio_mask_from_score,
    gradient_score_from_image_grid,
    grid_mse_gain_score,
    load_adapter_init,
    local_variance_score_from_image_grid,
    mixed_score_from_raw_global_pool,
    native_llamagen_feature,
    normalize_score_global_pool,
    oracle_error_mask,
    spatial_error_score_grid,
)
from train_titok_llamagen_recon import (
    add_path,
    autocast_dtype,
    convert_image_range,
    image_to_zero_one,
    load_llamagen_vq,
    load_titok,
)

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class EvalImageDataset(Dataset):
    def __init__(self, root: str, image_size: int, center_crop_arr, dataset_manifest: str = ""):
        self.root = Path(root)
        if dataset_manifest:
            payload = torch.load(dataset_manifest, map_location="cpu", weights_only=False)
            required = {"root", "samples"}
            if not isinstance(payload, dict) or not required.issubset(payload):
                raise ValueError(f"invalid dataset manifest schema: {dataset_manifest}")
            if Path(payload["root"]).resolve() != self.root.resolve():
                raise ValueError(
                    f"stale dataset manifest root: expected {self.root.resolve()}, got {payload['root']}"
                )
            self.paths = [Path(path) for path, _target in payload["samples"]]
        else:
            self.paths = sorted(
                p for p in self.root.rglob("*")
                if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS
            )
        if not self.paths:
            raise FileNotFoundError(f"No images found under {self.root}")
        self.transform = transforms.Compose([
            transforms.Lambda(lambda pil: center_crop_arr(pil, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index: int):
        return self.transform(Image.open(self.paths[index]).convert("RGB")), 0


def to_uint8(x: torch.Tensor, image_range: str) -> torch.Tensor:
    return (image_to_zero_one(x, image_range).clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)


def psnr_sum(pred: torch.Tensor, target: torch.Tensor, image_range: str) -> torch.Tensor:
    pred01 = image_to_zero_one(pred, image_range).clamp(0.0, 1.0)
    target01 = image_to_zero_one(target, image_range).clamp(0.0, 1.0)
    mse = (pred01 - target01).pow(2).flatten(1).mean(dim=1)
    return (-10.0 * torch.log10(mse + 1e-8)).sum()


def ssim_sum(pred: torch.Tensor, target: torch.Tensor, image_range: str, structural_similarity) -> float:
    pred_np = image_to_zero_one(pred, image_range).clamp(0.0, 1.0).detach().cpu().permute(0, 2, 3, 1).numpy()
    target_np = image_to_zero_one(target, image_range).clamp(0.0, 1.0).detach().cpu().permute(0, 2, 3, 1).numpy()
    return sum(float(structural_similarity(t, p, data_range=1.0, channel_axis=2)) for p, t in zip(pred_np, target_np))


def build_lpips_metric(args, device):
    if args.lpips_net == "llamagen_vgg":
        add_path(args.llamagen_root)
        from tokenizer.tokenizer_image.lpips import LPIPS

        return LPIPS().to(device).eval().requires_grad_(False)
    try:
        import lpips
    except Exception as exc:
        raise ImportError("lpips is required for LPIPS evaluation unless --lpips-net llamagen_vgg is used.") from exc
    return lpips.LPIPS(net=args.lpips_net).to(device).eval().requires_grad_(False)


def calibrate_ratio_logits_to_target_mean(ratio_logits, args):
    target = float(args.router_eval_target_mean_ratio)
    if target < 0.0:
        return ratio_logits
    min_ratio = float(args.router_min_ratio)
    max_ratio = float(args.router_max_ratio)
    target = min(max(target, min_ratio), max_ratio)
    if max_ratio <= min_ratio:
        return ratio_logits

    logits = ratio_logits.float()
    lo = logits.new_full((), -30.0)
    hi = logits.new_full((), 30.0)
    for _ in range(max(int(args.router_eval_target_calibrate_iters), 1)):
        mid = (lo + hi) * 0.5
        ratio = min_ratio + (max_ratio - min_ratio) * torch.sigmoid(logits + mid)
        if float(ratio.mean().item()) < target:
            lo = mid
        else:
            hi = mid
    return ratio_logits + ((lo + hi) * 0.5).to(dtype=ratio_logits.dtype)


CHECKPOINT_FILENAME = re.compile(r"epoch_(\d+)_step_(\d+)\.pt$")


def checkpoint_identity(checkpoint, ckpt_path: str):
    ckpt_step = int(checkpoint.get("step", -1)) if isinstance(checkpoint, dict) else -1
    ckpt_epoch = int(checkpoint.get("epoch", -1)) if isinstance(checkpoint, dict) else -1
    match = CHECKPOINT_FILENAME.search(Path(ckpt_path).name)
    if match is not None:
        filename_epoch = int(match.group(1))
        filename_step = int(match.group(2))
        if ckpt_step >= 0 and ckpt_step != filename_step:
            raise RuntimeError(
                f"checkpoint step mismatch: metadata={ckpt_step} filename={filename_step}: {ckpt_path}"
            )
        if ckpt_epoch >= 0 and ckpt_epoch != filename_epoch:
            raise RuntimeError(
                f"checkpoint epoch mismatch: metadata={ckpt_epoch} filename={filename_epoch}: {ckpt_path}"
            )
        ckpt_step = filename_step
        ckpt_epoch = filename_epoch
    return ckpt_step, ckpt_epoch


def load_trainable_params(model: TiTokLlamaGenStage2, ckpt_path: str, require_latent_decoder: bool = False, use_model_ema: bool = False):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if use_model_ema:
        if not isinstance(ckpt, dict) or "model_ema" not in ckpt:
            raise RuntimeError(f"--use-model-ema requested but checkpoint has no model_ema: {ckpt_path}")
        state = ckpt["model_ema"]
    else:
        state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    params = dict(model.named_parameters())
    copied = 0
    missing = []
    unexpected = []
    with torch.no_grad():
        for name, value in state.items():
            if name not in params:
                unexpected.append(name)
                continue
            params[name].copy_(value.to(device=params[name].device, dtype=params[name].dtype))
            copied += 1
        if require_latent_decoder:
            for name, param in params.items():
                if name.startswith("latent_decoder") and name not in state:
                    missing.append(name)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing[:10]} unexpected={unexpected[:10]}")
    ckpt_step, ckpt_epoch = checkpoint_identity(ckpt, ckpt_path)
    selector_threshold_state = ckpt.get("selector_threshold_state") if isinstance(ckpt, dict) else None
    selector_ratio_ema = ckpt.get("selector_ratio_ema") if isinstance(ckpt, dict) else None
    return ckpt_step, ckpt_epoch, copied, selector_threshold_state, selector_ratio_ema


@torch.no_grad()
def normalize_score_per_image(score, eps=1e-6):
    score = score.detach().float()
    min_val = score.amin(dim=(2, 3), keepdim=True)
    max_val = score.amax(dim=(2, 3), keepdim=True)
    return (score - min_val) / (max_val - min_val + eps)


@torch.no_grad()
def mixed_score_from_raw_parts(raw_parts, args, scope):
    if scope == "global_pool":
        normalize = normalize_score_global_pool
    elif scope == "per_image":
        normalize = normalize_score_per_image
    else:
        raise ValueError(f"unsupported score_normalize_scope {scope}")
    norm_parts = {
        "error": normalize(raw_parts["error"]),
        "gradient": normalize(raw_parts["gradient"]),
        "variance": normalize(raw_parts["variance"]),
    }
    score = (
        float(args.mask_error_weight) * norm_parts["error"]
        + float(args.mask_gradient_weight) * norm_parts["gradient"]
        + float(args.mask_variance_weight) * norm_parts["variance"]
    )
    return score, norm_parts


@torch.no_grad()
def threshold_mask_from_score(score, selector_threshold):
    return (score.detach().float() > float(selector_threshold)).to(dtype=torch.float32)


class PathMetricAccumulator:
    def __init__(self, compute_ssim: bool = True):
        self.compute_ssim = bool(compute_ssim)
        self.psnr = 0.0
        self.ssim = 0.0
        self.lpips = 0.0
        self.l1 = 0.0
        self.mse01 = 0.0
        self.count = 0

    def update(self, pred, target, image_range, lpips_fn, structural_similarity, lpips_only=False):
        bsz = pred.shape[0]
        self.lpips += float(lpips_fn(pred.float(), target.float()).mean().item()) * bsz
        if not lpips_only:
            self.psnr += float(psnr_sum(pred, target, image_range).item())
            if self.compute_ssim:
                self.ssim += ssim_sum(pred, target, image_range, structural_similarity)
            self.l1 += float(F.l1_loss(pred.float(), target.float(), reduction="mean").item()) * bsz
            pred01 = image_to_zero_one(pred, image_range)
            target01 = image_to_zero_one(target, image_range)
            self.mse01 += float(F.mse_loss(pred01.float(), target01.float(), reduction="mean").item()) * bsz
        self.count += bsz

    def compute(self, fid_value):
        count = max(self.count, 1)
        mse01 = self.mse01 / count
        psnr_avg_image = self.psnr / count
        lpips_only = self.mse01 == 0.0 and self.psnr == 0.0 and self.ssim == 0.0 and self.l1 == 0.0
        return {
            "fid": None if fid_value is None else float(fid_value),
            "lpips": self.lpips / count,
            "psnr": None if lpips_only else psnr_avg_image,
            "psnr_avg_image": None if lpips_only else psnr_avg_image,
            "psnr_from_mse01": None if lpips_only else -10.0 * math.log10(max(mse01, 1e-12)),
            "ssim": None if (lpips_only or not self.compute_ssim) else self.ssim / count,
            "l1": None if lpips_only else self.l1 / count,
            "mse01": None if lpips_only else mse01,
        }


def main(args):
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    FrechetInceptionDistance = None
    if not (args.no_fid or args.lpips_only):
        try:
            from torchmetrics.image.fid import FrechetInceptionDistance
        except Exception as exc:
            raise ImportError("torchmetrics with FID support is required for FID evaluation.") from exc
    structural_similarity = None
    if not (args.lpips_only or args.no_ssim):
        try:
            from skimage.metrics import structural_similarity
        except Exception as exc:
            raise ImportError("scikit-image is required for SSIM evaluation.") from exc
    add_path(args.llamagen_root)
    from dataset.augmentation import center_crop_arr

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    titok = load_titok(args.titok_root, args.titok_config, args.titok_ckpt, device)
    vq_model = load_llamagen_vq(args.llamagen_root, args.llamagen_ckpt, device, args.codebook_size, args.codebook_embed_dim)
    model = TiTokLlamaGenStage2(
        titok,
        vq_model,
        lg_latent_channels=args.lg_latent_channels,
        head_channels=args.lg_head_channels,
        head_mode="feature",
        codebook_size=args.codebook_size,
        codebook_temperature=1.0,
    ).to(device)
    model.router = DynamicBudgetRouter(
        latent_channels=args.lg_latent_channels,
        hidden_dim=args.router_hidden_dim,
        depth=args.router_depth,
        target_ratio=args.router_target_mean_ratio,
        min_ratio=args.router_min_ratio,
        max_ratio=args.router_max_ratio,
        detach_inputs=args.router_detach_inputs,
    ).to(device)
    adapter_state = ""
    adapter_step = -1
    if args.adapter_init:
        adapter_state, adapter_step = load_adapter_init(model, args.adapter_init, args.adapter_init_ema, strict=True)
    base_ckpt_step = -1
    base_ckpt_epoch = -1
    base_copied = 0
    if args.base_ckpt:
        base_ckpt_step, base_ckpt_epoch, base_copied, _, _ = load_trainable_params(
            model,
            args.base_ckpt,
            require_latent_decoder=(args.require_latent_decoder or not bool(args.adapter_init)),
        )
    ckpt_step, ckpt_epoch, copied, ckpt_selector_threshold, ckpt_selector_ratio_ema = load_trainable_params(
        model,
        args.ckpt,
        require_latent_decoder=(args.require_latent_decoder or not bool(args.adapter_init)) and not bool(args.base_ckpt),
        use_model_ema=args.use_model_ema,
    )
    selector_threshold = args.selector_threshold
    if args.use_checkpoint_selector_threshold and ckpt_selector_threshold is not None:
        selector_threshold = float(ckpt_selector_threshold)
    if selector_threshold is None:
        selector_threshold = float(args.mask_ratio)
    model.eval().requires_grad_(False)

    dataset = EvalImageDataset(
        args.data_path,
        args.image_size,
        center_crop_arr,
        dataset_manifest=args.dataset_manifest,
    )
    if args.num_images > 0:
        dataset = Subset(dataset, range(min(args.num_images, len(dataset))))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    eval_paths = list(dict.fromkeys(args.eval_paths))
    no_fid = args.no_fid or args.lpips_only
    fids = {} if no_fid else {
        name: FrechetInceptionDistance(feature=args.fid_feature, normalize=False).to(device)
        for name in eval_paths
    }
    acc = {name: PathMetricAccumulator(compute_ssim=not args.no_ssim) for name in eval_paths}
    lpips_fn = build_lpips_metric(args, device)
    dtype = autocast_dtype(args.mixed_precision)
    autocast_enabled = device.type == "cuda" and args.mixed_precision != "none"
    mask_sum = 0.0
    mask_tokens_sum = 0.0
    mask_tokens_sq_sum = 0.0
    mask_tokens_min = None
    mask_tokens_max = None
    score_sum = 0.0
    score_error_sum = 0.0
    score_gradient_sum = 0.0
    score_variance_sum = 0.0
    feat_l1_sum = 0.0
    count = 0
    top_p_grid_stats = {
        (float(temperature), float(top_p)): {"count": 0, "sum": 0.0, "sq_sum": 0.0, "min": None, "max": None}
        for temperature in args.router_top_p_temperature_grid
        for top_p in args.router_top_p_grid
    }

    with torch.no_grad():
        pbar = tqdm(loader, desc="eval_mix", dynamic_ncols=True)
        for x01, _ in pbar:
            x01 = x01.to(device, non_blocking=True)
            x_titok = convert_image_range(x01, args.titok_input_range)
            target = convert_image_range(x01, args.llamagen_input_range)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
                x_base, extra = model(x_titok)
                f_1d = extra["f_1d_lg"]
                f_2d, _ = native_llamagen_feature(model.llamagen_vq, target, args.codebook_embed_dim, allow_encoder_grad=False)
                x_native = None
                if args.mask_selection == "per_image_error_random":
                    mask = oracle_error_mask(x_base, target, args.mask_ratio_min, args.mask_ratio_max).to(dtype=f_1d.dtype)
                    score = spatial_error_score_grid(x_base, target, grid_hw=16)
                    zeros = torch.zeros_like(score)
                    norm_parts = {
                        "error": normalize_score_global_pool(score),
                        "gradient": zeros,
                        "variance": zeros,
                    }
                    score_primary_key = "error"
                elif args.mask_selection == "batch_global_mixed_score":
                    raw_parts = {
                        "error": spatial_error_score_grid(x_base, target, grid_hw=16),
                        "gradient": gradient_score_from_image_grid(target, grid_hw=16),
                        "variance": local_variance_score_from_image_grid(target, grid_hw=16),
                    }
                    score, norm_parts = mixed_score_from_raw_global_pool(raw_parts, args)
                    mask = global_top_ratio_mask_from_score(score, args.mask_ratio).to(dtype=f_1d.dtype)
                    score_primary_key = "error"
                elif args.mask_selection == "accum_global_mixed_score_threshold":
                    raw_parts = {
                        "error": spatial_error_score_grid(x_base, target, grid_hw=16),
                        "gradient": gradient_score_from_image_grid(target, grid_hw=16),
                        "variance": local_variance_score_from_image_grid(target, grid_hw=16),
                    }
                    score, norm_parts = mixed_score_from_raw_parts(raw_parts, args, args.score_normalize_scope)
                    mask = threshold_mask_from_score(score, selector_threshold).to(dtype=f_1d.dtype)
                    score_primary_key = "error"
                elif args.mask_selection == "router_e2e_dynamic":
                    router_logits, router_ratio_logits = model.router(f_1d, x_base, f_2d)
                    router_ratio_logits = calibrate_ratio_logits_to_target_mean(router_ratio_logits, args)
                    mask, hard_mask, router_soft_mask, router_ratio_soft = make_dynamic_budget_ste_mask(
                        router_logits, router_ratio_logits, args
                    )
                    mask = mask.to(dtype=f_1d.dtype)
                    score = torch.sigmoid(router_logits.detach().float())
                    zeros = torch.zeros_like(score)
                    norm_parts = {
                        "error": zeros,
                        "gradient": router_soft_mask.detach().float(),
                        "variance": zeros,
                    }
                    score_primary_key = "gradient"
                elif args.mask_selection == "router_e2e_top_p":
                    router_logits, _router_ratio_logits = model.router(f_1d, x_base, f_2d)
                    _mask, hard_mask, router_soft_mask, _router_ratio_soft = make_top_p_ste_mask(
                        router_logits, args
                    )
                    mask = hard_mask.to(dtype=f_1d.dtype)
                    score = torch.softmax(
                        router_logits.detach().float().flatten(1) / float(args.router_top_p_temperature), dim=1
                    ).view_as(router_logits)
                    zeros = torch.zeros_like(score)
                    norm_parts = {
                        "error": zeros,
                        "gradient": router_soft_mask.detach().float(),
                        "variance": zeros,
                    }
                    score_primary_key = "gradient"
                    if top_p_grid_stats:
                        flat_logits = router_logits.detach().float().flatten(1)
                        for temperature in args.router_top_p_temperature_grid:
                            probabilities = torch.softmax(flat_logits / float(temperature), dim=1)
                            sorted_probabilities = torch.sort(probabilities, dim=1, descending=True).values
                            cumulative_before = torch.cumsum(sorted_probabilities, dim=1) - sorted_probabilities
                            for top_p in args.router_top_p_grid:
                                token_counts = (cumulative_before < float(top_p)).sum(dim=1).float()
                                stats = top_p_grid_stats[(float(temperature), float(top_p))]
                                stats["count"] += int(token_counts.numel())
                                stats["sum"] += float(token_counts.sum().item())
                                stats["sq_sum"] += float((token_counts * token_counts).sum().item())
                                batch_min = float(token_counts.min().item())
                                batch_max = float(token_counts.max().item())
                                stats["min"] = batch_min if stats["min"] is None else min(stats["min"], batch_min)
                                stats["max"] = batch_max if stats["max"] is None else max(stats["max"], batch_max)
                elif args.mask_selection == "gain_first_texture":
                    x_native = model.llamagen_vq.decoder(f_2d)
                    raw_parts = {
                        "gain": grid_mse_gain_score(x_base, x_native, target, grid_hw=16),
                        "gradient": gradient_score_from_image_grid(target, grid_hw=16),
                        "variance": local_variance_score_from_image_grid(target, grid_hw=16),
                    }
                    score, norm_parts = gain_first_texture_score_from_raw_global_pool(raw_parts, args)
                    mask = global_top_ratio_mask_from_score(score, args.mask_ratio).to(dtype=f_1d.dtype)
                    score_primary_key = "gain"
                elif args.mask_selection == "gain_mixed_blend":
                    x_native = model.llamagen_vq.decoder(f_2d)
                    mixed_raw_parts = {
                        "error": spatial_error_score_grid(x_base, target, grid_hw=16),
                        "gradient": gradient_score_from_image_grid(target, grid_hw=16),
                        "variance": local_variance_score_from_image_grid(target, grid_hw=16),
                    }
                    mixed_score, mixed_norm_parts = mixed_score_from_raw_global_pool(mixed_raw_parts, args)
                    gain_raw_parts = {
                        "gain": grid_mse_gain_score(x_base, x_native, target, grid_hw=16),
                        "gradient": mixed_raw_parts["gradient"],
                        "variance": mixed_raw_parts["variance"],
                    }
                    gain_score, gain_norm_parts = gain_first_texture_score_from_raw_global_pool(gain_raw_parts, args)
                    score = (
                        (1.0 - float(args.blend_gain_weight)) * normalize_score_global_pool(mixed_score)
                        + float(args.blend_gain_weight) * normalize_score_global_pool(gain_score)
                    )
                    norm_parts = {
                        "error": mixed_norm_parts["error"],
                        "gradient": mixed_norm_parts["gradient"],
                        "variance": mixed_norm_parts["variance"],
                        "gain": gain_norm_parts["gain"],
                    }
                    mask = global_top_ratio_mask_from_score(score, args.mask_ratio).to(dtype=f_1d.dtype)
                    score_primary_key = "gain"
                else:
                    raise ValueError(f"unsupported mask_selection {args.mask_selection}")
                f_mix = (1.0 - mask) * f_1d + mask * f_2d
                x_mix = model.llamagen_vq.decoder(f_mix)
                if x_native is None and "native" in eval_paths:
                    x_native = model.llamagen_vq.decoder(f_2d)
            paths = {}
            if "base" in eval_paths:
                paths["base"] = x_base.float()
            if "mix" in eval_paths:
                paths["mix"] = x_mix.float()
            if "native" in eval_paths:
                paths["native"] = x_native.float()
            target = target.float()
            real_uint8 = to_uint8(target, args.llamagen_input_range)
            for name, pred in paths.items():
                if not no_fid:
                    fids[name].update(real_uint8, real=True)
                    fids[name].update(to_uint8(pred, args.llamagen_input_range), real=False)
                acc[name].update(pred, target, args.llamagen_input_range, lpips_fn, structural_similarity, lpips_only=args.lpips_only)
            bsz = target.shape[0]
            mask_float = mask.float()
            mask_tokens = mask_float.flatten(1).sum(dim=1)
            mask_sum += float(mask_float.mean().item()) * bsz
            mask_tokens_sum += float(mask_tokens.sum().item())
            mask_tokens_sq_sum += float((mask_tokens * mask_tokens).sum().item())
            batch_token_min = float(mask_tokens.min().item())
            batch_token_max = float(mask_tokens.max().item())
            mask_tokens_min = batch_token_min if mask_tokens_min is None else min(mask_tokens_min, batch_token_min)
            mask_tokens_max = batch_token_max if mask_tokens_max is None else max(mask_tokens_max, batch_token_max)
            score_sum += global_mean_value(score) * bsz
            score_error_sum += global_mean_value(norm_parts[score_primary_key]) * bsz
            score_gradient_sum += global_mean_value(norm_parts["gradient"]) * bsz
            score_variance_sum += global_mean_value(norm_parts["variance"]) * bsz
            feat_l1_sum += float(F.l1_loss(f_1d.float(), f_2d.float(), reduction="mean").item()) * bsz
            count += bsz
            pbar.set_postfix(mix_psnr=f"{acc['mix'].psnr/max(count,1):.2f}", mix_lpips=f"{acc['mix'].lpips/max(count,1):.4f}")

    count_for_token_stats = max(count, 1)
    mask_tokens_mean = mask_tokens_sum / count_for_token_stats
    mask_tokens_var = max(mask_tokens_sq_sum / count_for_token_stats - mask_tokens_mean * mask_tokens_mean, 0.0)
    metrics = {
        "ckpt": str(args.ckpt),
        "ckpt_step": ckpt_step,
        "ckpt_epoch": ckpt_epoch,
        "copied_params": copied,
        "base_ckpt": str(args.base_ckpt),
        "base_ckpt_step": base_ckpt_step,
        "base_ckpt_epoch": base_ckpt_epoch,
        "base_copied_params": base_copied,
        "adapter_init": str(args.adapter_init),
        "adapter_init_state": adapter_state,
        "adapter_init_step": adapter_step,
        "use_model_ema": bool(args.use_model_ema),
        "data_path": str(args.data_path),
        "dataset_manifest": str(args.dataset_manifest),
        "num_images": count,
        "image_size": args.image_size,
        "seed": args.seed,
        "mask_selection": args.mask_selection,
        "mask_ratio": args.mask_ratio,
        "mask_ratio_min": args.mask_ratio_min,
        "mask_ratio_max": args.mask_ratio_max,
        "mask_error_weight": args.mask_error_weight,
        "mask_gradient_weight": args.mask_gradient_weight,
        "mask_variance_weight": args.mask_variance_weight,
        "gain_texture_alpha": args.gain_texture_alpha,
        "gain_gradient_weight": args.gain_gradient_weight,
        "gain_variance_weight": args.gain_variance_weight,
        "blend_gain_weight": args.blend_gain_weight,
        "selector_threshold": selector_threshold,
        "ckpt_selector_threshold_state": None if ckpt_selector_threshold is None else float(ckpt_selector_threshold),
        "ckpt_selector_ratio_ema": None if ckpt_selector_ratio_ema is None else float(ckpt_selector_ratio_ema),
        "score_normalize_scope": args.score_normalize_scope,
        "router_hidden_dim": args.router_hidden_dim,
        "router_depth": args.router_depth,
        "router_min_ratio": args.router_min_ratio,
        "router_max_ratio": args.router_max_ratio,
        "router_target_mean_ratio": args.router_target_mean_ratio,
        "router_eval_target_mean_ratio": args.router_eval_target_mean_ratio,
        "router_eval_target_calibrate_iters": args.router_eval_target_calibrate_iters,
        "router_tau": args.router_tau,
        "router_top_p": args.router_top_p,
        "router_top_p_temperature": args.router_top_p_temperature,
        "router_top_p_ste_tau": args.router_top_p_ste_tau,
        "top_p_grid": {
            f"temperature={temperature:g},p={top_p:g}": {
                "count": stats["count"],
                "mean": stats["sum"] / max(stats["count"], 1),
                "std": math.sqrt(max(stats["sq_sum"] / max(stats["count"], 1) - (stats["sum"] / max(stats["count"], 1)) ** 2, 0.0)),
                "min": stats["min"],
                "max": stats["max"],
            }
            for (temperature, top_p), stats in top_p_grid_stats.items()
        },
        "mask_mean": mask_sum / max(count, 1),
        "mask_tokens_mean": mask_tokens_mean,
        "mask_tokens_std": math.sqrt(mask_tokens_var),
        "mask_tokens_min": mask_tokens_min,
        "mask_tokens_max": mask_tokens_max,
        "score_mean": score_sum / max(count, 1),
        "score_error_mean": score_error_sum / max(count, 1),
        "score_gradient_mean": score_gradient_sum / max(count, 1),
        "score_variance_mean": score_variance_sum / max(count, 1),
        "feat_l1": feat_l1_sum / max(count, 1),
        "eval_paths": eval_paths,
        "lpips_net": args.lpips_net,
        "no_fid": bool(no_fid),
        "fid_feature": None if no_fid else int(args.fid_feature),
        "fid_normalize": False,
        "no_ssim": bool(args.no_ssim),
        "lpips_only": bool(args.lpips_only),
        "reconstruction": {
            name: acc[name].compute(None if no_fid else fids[name].compute().item())
            for name in eval_paths
        },
    }
    output_json = Path(args.output_json)
    suffix = ""
    if ckpt_epoch >= 0:
        suffix += f"_epoch{ckpt_epoch:06d}"
    if ckpt_step >= 0:
        suffix += f"_step{ckpt_step:08d}"
    if suffix:
        output_json = output_json.with_name(f"{output_json.stem}{suffix}{output_json.suffix}")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"Saved metrics to {output_json}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate TiTok/LlamaGen mix reconstruction metrics with batch-global masks.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--base-ckpt", type=str, default="")
    parser.add_argument("--use-model-ema", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--require-latent-decoder", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--adapter-init", type=str, default="")
    parser.add_argument("--adapter-init-ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--data-path", type=str, default="../ImageNet/validation")
    parser.add_argument("--dataset-manifest", type=str, default="")
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--num-images", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--titok-input-range", type=str, default="zero_1", choices=["zero_1", "minus1_1"])
    parser.add_argument("--llamagen-input-range", type=str, default="minus1_1", choices=["zero_1", "minus1_1"])
    parser.add_argument("--mixed-precision", type=str, default="bf16", choices=["bf16", "fp16", "none"])
    parser.add_argument("--mask-selection", type=str, default="batch_global_mixed_score", choices=["per_image_error_random", "batch_global_mixed_score", "accum_global_mixed_score_threshold", "gain_first_texture", "gain_mixed_blend", "router_e2e_dynamic", "router_e2e_top_p"])
    parser.add_argument("--mask-ratio-min", type=float, default=0.1)
    parser.add_argument("--mask-ratio-max", type=float, default=0.9)
    parser.add_argument("--mask-ratio", type=float, default=0.5)
    parser.add_argument("--mask-error-weight", type=float, default=0.6)
    parser.add_argument("--mask-gradient-weight", type=float, default=0.3)
    parser.add_argument("--mask-variance-weight", type=float, default=0.1)
    parser.add_argument("--selector-threshold", type=float, default=None)
    parser.add_argument("--use-checkpoint-selector-threshold", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--score-normalize-scope", type=str, default="global_pool", choices=["global_pool", "per_image"])
    parser.add_argument("--router-hidden-dim", type=int, default=128)
    parser.add_argument("--router-depth", type=int, default=3)
    parser.add_argument("--router-min-ratio", type=float, default=0.05)
    parser.add_argument("--router-max-ratio", type=float, default=0.90)
    parser.add_argument("--router-target-mean-ratio", type=float, default=0.50)
    parser.add_argument("--router-eval-target-mean-ratio", type=float, default=-1.0)
    parser.add_argument("--router-eval-target-calibrate-iters", type=int, default=24)
    parser.add_argument("--router-min-tokens", type=int, default=-1)
    parser.add_argument("--router-max-tokens", type=int, default=-1)
    parser.add_argument("--router-tau", type=float, default=0.7)
    parser.add_argument("--router-top-p", type=float, default=0.95)
    parser.add_argument("--router-top-p-temperature", type=float, default=1.0)
    parser.add_argument("--router-top-p-ste-tau", type=float, default=0.01)
    parser.add_argument("--router-top-p-grid", type=float, nargs="*", default=[])
    parser.add_argument("--router-top-p-temperature-grid", type=float, nargs="*", default=[])
    parser.add_argument("--router-detach-inputs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gain-texture-alpha", type=float, default=0.2)
    parser.add_argument("--gain-gradient-weight", type=float, default=0.5)
    parser.add_argument("--gain-variance-weight", type=float, default=0.5)
    parser.add_argument("--blend-gain-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fid-feature", type=int, default=2048)
    parser.add_argument("--lpips-net", type=str, default="alex")
    parser.add_argument("--no-fid", action="store_true", default=False)
    parser.add_argument("--no-ssim", action="store_true", default=False)
    parser.add_argument("--lpips-only", action="store_true", default=False)
    parser.add_argument("--eval-paths", type=str, nargs="+", default=["base", "mix", "native"], choices=["base", "mix", "native"])
    parser.add_argument("--lg-latent-channels", type=int, default=256)
    parser.add_argument("--lg-head-channels", type=int, default=256)
    parser.add_argument("--titok-root", type=str, default="../1d-tokenizer")
    parser.add_argument("--titok-config", type=str, default="../1d-tokenizer/configs/infer/TiTok/titok_l32.yaml")
    parser.add_argument("--titok-ckpt", type=str, default="../1d-tokenizer/tokenizer_titok_l32.bin")
    parser.add_argument("--llamagen-root", type=str, default="../LlamaGen")
    parser.add_argument("--llamagen-ckpt", type=str, default="../LlamaGen/pretrained_models/vq_ds16_c2i.pt")
    parser.add_argument("--codebook-size", type=int, default=16384)
    parser.add_argument("--codebook-embed-dim", type=int, default=8)

    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None)
    config_args, remaining = config_parser.parse_known_args()
    defaults = vars(parser.parse_args([]))
    if config_args.config is not None:
        config = OmegaConf.to_container(OmegaConf.load(config_args.config), resolve=True)
        if not isinstance(config, dict):
            raise ValueError(f"config must contain a mapping, got {type(config).__name__}")
        values = {key.replace("-", "_"): value for key, value in config.items()}
        unknown = sorted(set(values) - set(defaults))
        if unknown:
            raise ValueError(f"unknown config keys: {unknown}")
        parser.set_defaults(**values)
    args = parser.parse_args(remaining)
    args.config = config_args.config
    if not args.ckpt:
        raise ValueError("--ckpt is required, either in YAML or CLI")
    if not args.output_json:
        raise ValueError("--output-json is required, either in YAML or CLI")
    return args


if __name__ == "__main__":
    main(parse_args())
