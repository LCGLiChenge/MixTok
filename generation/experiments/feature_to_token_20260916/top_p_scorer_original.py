#!/usr/bin/env python3
"""Train a 1D-only top-p Grid Scorer by distilling the top-p recon teacher.

The teacher is the top-p end-to-end recon checkpoint.  It may use x_base and
native 2D features only to produce offline targets.  The student receives
only the top-p decoder's 1D feature map, so the deployed scorer is genuinely
1D-only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def setup_dist():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    if world > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    return rank, world, local_rank, torch.device("cuda", local_rank)


class OneDTopPGridScorer(nn.Module):
    """The old Grid Scorer trunk with all x_base/f2d paths removed."""

    def __init__(self, channels=256, hidden=128, depth=3):
        super().__init__()
        self.f_proj = nn.Conv2d(channels, hidden, kernel_size=1)
        self.pos_embed = nn.Parameter(torch.zeros(1, hidden, 16, 16))
        blocks = []
        for _ in range(depth):
            blocks.extend([
                nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
                nn.GroupNorm(8, hidden),
                nn.SiLU(inplace=True),
            ])
        self.trunk = nn.Sequential(*blocks)
        self.score_head = nn.Conv2d(hidden, 1, kernel_size=1)
        nn.init.normal_(self.score_head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.score_head.bias)

    def forward(self, f_1d):
        if f_1d.ndim != 4 or f_1d.shape[1:] != (256, 16, 16):
            raise ValueError(f"expected [B,256,16,16], got {tuple(f_1d.shape)}")
        h = self.f_proj(f_1d) + self.pos_embed.to(dtype=f_1d.dtype)
        h = self.trunk(h)
        return self.score_head(h)


def top_p_mask(logits, p, temperature):
    """Return exact [B,256] prefix masks and their token counts."""
    flat = logits.float().flatten(1) / float(temperature)
    probs = flat.softmax(dim=1)
    sorted_probs, order = probs.sort(dim=1, descending=True)
    before = sorted_probs.cumsum(dim=1) - sorted_probs
    sorted_mask = before < float(p)
    mask = torch.zeros_like(sorted_mask)
    mask.scatter_(1, order, sorted_mask)
    return mask, mask.sum(dim=1)


def load_models(args, device, source_root):
    sys.path.insert(0, str(source_root))
    import train_titok_llamagen_decoder_adapt_router_f2d_e2e_dynamic as top
    from models import TiTokLlamaGenStage2

    titok = top.load_titok(args.titok_root, args.titok_config, args.titok_ckpt, device)
    os.environ["MOT_SKIP_LLAMAGEN_INITIAL_WEIGHTS"] = "1"
    vq = top.load_llamagen_vq(
        args.llamagen_root,
        "/dev/null",
        device,
        codebook_size=16384,
        codebook_embed_dim=8,
    )
    model = TiTokLlamaGenStage2(
        titok,
        vq,
        lg_latent_channels=256,
        head_channels=256,
        head_mode="feature",
        codebook_size=16384,
        codebook_temperature=1.0,
    ).to(device)
    model.router = top.DynamicBudgetRouter(
        latent_channels=256,
        hidden_dim=128,
        depth=3,
        target_ratio=0.375,
        min_ratio=0.0,
        max_ratio=1.0,
        detach_inputs=True,
    ).to(device)
    ckpt = torch.load(args.teacher_ckpt, map_location="cpu", weights_only=False)
    state = ckpt.get("model_ema", ckpt.get("model", ckpt))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if args.rank == 0:
        print(f"teacher loaded step={ckpt.get('step', -1)} missing={len(missing)} unexpected={len(unexpected)}", flush=True)
        print(f"teacher checkpoint={args.teacher_ckpt}", flush=True)
    model.eval().requires_grad_(False)
    return top, model


def make_loader(args, top, distributed, rank, world):
    payload = top._load_dataset_manifest(args.dataset_manifest)
    transform = top.make_transform(256, random_crop=False, random_flip=False)
    ds = top.ManifestImageFolder(
        payload["root"], payload["samples"], payload["classes"], payload["class_to_idx"], transform=transform
    )
    if args.limit_samples > 0:
        ds = torch.utils.data.Subset(ds, range(min(args.limit_samples, len(ds))))
    start = int(getattr(args, "start_index", 0))
    if start < 0 or start >= len(ds):
        raise ValueError(f"invalid start_index={start} for dataset size {len(ds)}")
    if args.num_images < 0:
        raise ValueError("num_images must be non-negative; 0 means all remaining images")
    stop = len(ds) if args.num_images == 0 else min(start + args.num_images, len(ds))
    ds = torch.utils.data.Subset(ds, range(start, stop))
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True, drop_last=True) if distributed else None
    loader = DataLoader(
        ds,
        batch_size=args.micro_batch,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    return loader, sampler


@torch.no_grad()
def evaluate_student(student, model, loader, device, p_grid, teacher_temperature):
    student.eval()
    counts = {str(p): [] for p in p_grid}
    kl_sum = 0.0
    n = 0
    for x01, _ in loader:
        x01 = x01.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, extra = model(x01)
            f1d = extra["f_1d_lg"]
            logits = student(f1d.float())
        for p in p_grid:
            _, k = top_p_mask(logits, p, teacher_temperature)
            counts[str(p)].extend(k.cpu().tolist())
        n += x01.shape[0]
        if n >= 2048:
            break
    return {p: {"mean_k": float(sum(v) / max(len(v), 1)), "n": len(v)} for p, v in counts.items()}


def main(args):
    rank, world, local_rank, device = setup_dist()
    args.rank = rank
    args.world = world
    source_root = Path(args.source_root).resolve()
    output = Path(args.output_dir)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        (output / "config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True))

    top, teacher_model = load_models(args, device, source_root)
    loader, sampler = make_loader(args, top, world > 1, rank, world)

    student = OneDTopPGridScorer(hidden=args.hidden, depth=args.depth).to(device)
    if world > 1:
        student = DDP(student, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=False)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    step = 0
    epoch = 0
    last_log = time.time()
    student.train()
    while step < args.steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for x01, _ in loader:
            if step >= args.steps:
                break
            x01 = x01.to(device, non_blocking=True)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                _, extra = teacher_model(x01)
                f1d = extra["f_1d_lg"]
                x_lg = x01.float() * 2.0 - 1.0
                f2d, _ = top.native_llamagen_feature(teacher_model.llamagen_vq, x_lg, 8)
                x_base = teacher_model.llamagen_vq.decoder(f1d)
                teacher_logits, _ = teacher_model.router(f1d, x_base, f2d)
                teacher_prob = (teacher_logits.float().flatten(1) / args.teacher_temperature).softmax(dim=1)
            student_logits = student(f1d.float())
            student_logprob = F.log_softmax(student_logits.float().flatten(1) / args.student_temperature, dim=1)
            loss_kl = F.kl_div(student_logprob, teacher_prob, reduction="batchmean")
            teacher_center = teacher_logits.float().flatten(1) - teacher_logits.float().flatten(1).mean(dim=1, keepdim=True)
            student_center = student_logits.float().flatten(1) - student_logits.float().flatten(1).mean(dim=1, keepdim=True)
            loss_shape = F.smooth_l1_loss(student_center, teacher_center.detach())
            loss = loss_kl + args.logit_loss_weight * loss_shape
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            step += 1
            if rank == 0 and (step == 1 or step % args.log_every == 0):
                now = time.time()
                print(f"step={step}/{args.steps} loss={loss.item():.6f} kl={loss_kl.item():.6f} shape={loss_shape.item():.6f} sec={now-last_log:.2f}", flush=True)
                last_log = now
            if step % args.save_every == 0 or step == args.steps:
                if rank == 0:
                    module = student.module if isinstance(student, DDP) else student
                    torch.save({"model": module.state_dict(), "step": step, "args": vars(args)}, output / "latest.pt")
                    print(f"saved {output / 'latest.pt'}", flush=True)
        epoch += 1
    if world > 1:
        dist.barrier()
    if rank == 0:
        module = student.module if isinstance(student, DDP) else student
        stats = evaluate_student(module, teacher_model, loader, device, args.p_grid, args.teacher_temperature)
        (output / "mean_k.json").write_text(json.dumps(stats, indent=2, sort_keys=True))
        print(json.dumps(stats, indent=2), flush=True)
    if world > 1:
        dist.destroy_process_group()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source-root", required=True)
    p.add_argument("--teacher-ckpt", required=True)
    p.add_argument("--dataset-manifest", required=True)
    p.add_argument("--titok-root", required=True)
    p.add_argument("--titok-config", required=True)
    p.add_argument("--titok-ckpt", required=True)
    p.add_argument("--llamagen-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--micro-batch", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=12)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--teacher-temperature", type=float, default=2.0)
    p.add_argument("--student-temperature", type=float, default=2.0)
    p.add_argument("--logit-loss-weight", type=float, default=0.05)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--limit-samples", type=int, default=0)
    p.add_argument("--num-images", type=int, default=0,
                   help="Use this many images after start_index (0: all remaining images)")
    p.add_argument("--p-grid", type=float, nargs="+", default=[0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995])
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
