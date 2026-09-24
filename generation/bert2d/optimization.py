"""Adam SDE scaling (arxiv:2205.10287) + first-order AdamW decay scaling
(arxiv:2307.13813, App. C.2). Not a guarantee of equal MaskGIT FID.
"""
from __future__ import annotations
import argparse
import json
import math

REFERENCE_BATCH = 448


def add_arguments(parser):
    parser.add_argument("--batch-scaling", choices=("adamw-sde", "legacy"), default="adamw-sde",
                        help="joint scaling from batch 448; legacy reproduces old runs")
    parser.add_argument("--lr-new", type=float, default=1e-4,
                        help="UNSCALED fresh LR at reference global batch 448")
    parser.add_argument("--lr-pretrained", type=float, default=1e-5,
                        help="UNSCALED pretrained LR at reference global batch 448")


def resolve(global_batch, mode="adamw-sde", lr_new=1e-4, lr_pretrained=1e-5):
    if not isinstance(global_batch, int) or global_batch < 1:
        raise ValueError("global batch must be a positive integer")
    if mode not in ("adamw-sde", "legacy"):
        raise ValueError("unknown batch scaling mode")
    if any(not math.isfinite(v) or v <= 0 for v in (lr_new, lr_pretrained)):
        raise ValueError("reference learning rates must be finite and positive")
    k = global_batch / REFERENCE_BATCH if mode == "adamw-sde" else 1.
    scale = math.sqrt(k)
    betas = [.9, .96] if k == 1 else [1-k*(1-b) for b in (.9, .96)]
    if not all(0 <= b < 1 for b in betas):
        raise ValueError("Adam SDE scaling produces invalid betas; reduce global batch. "
                         "No silent clipping or fallback.")
    return dict(version=1, mode=mode, reference_global_batch=REFERENCE_BATCH,
                global_batch=global_batch, kappa=k, lr_multiplier=scale,
                lr_new=lr_new*scale, lr_pretrained=lr_pretrained*scale,
                betas=betas, eps=1e-8/scale, weight_decay=.03*scale,
                schedule_clock="reference_updates" if mode == "adamw-sde" else "optimizer_updates",
                fresh_warmup=50., pretrained_freeze=20., pretrained_warmup=100.,
                grad_clip=1., decay_rule="sqrt_first_order" if mode == "adamw-sde" else "fixed")


def from_args(args):
    return resolve(args.global_batch, args.batch_scaling, args.lr_new, args.lr_pretrained)


def cli_arguments(args):
    return ["--batch-scaling", args.batch_scaling, "--lr-new", str(args.lr_new),
            "--lr-pretrained", str(args.lr_pretrained)]


def factor(recipe, group_name, step):
    clock = step * recipe["kappa"]
    if group_name == "fresh":
        return min(1., clock / recipe["fresh_warmup"])
    if group_name == "pretrained":
        return max(0., min(1., (clock-recipe["pretrained_freeze"])/recipe["pretrained_warmup"]))
    raise ValueError("unknown optimizer parameter group")


def make_optimizer(core, audit, recipe):
    import torch
    fresh = set(audit["fresh_parameters"])
    groups = []
    for name, is_new, key in (("fresh", True, "lr_new"), ("pretrained", False, "lr_pretrained")):
        params = [p for n, p in core.named_parameters() if (n in fresh) == is_new]
        if params:
            groups.append(dict(params=params, name=name, lr=0., base_lr=recipe[key]))
    return torch.optim.AdamW(groups, betas=tuple(recipe["betas"]), eps=recipe["eps"],
                             weight_decay=recipe["weight_decay"])


def set_learning_rates(optimizer, recipe, step):
    for group in optimizer.param_groups:
        group["lr"] = group["base_lr"] * factor(recipe, group["name"], step)


def validate_saved_config(config, recipe):
    previous = config.get("optimization")
    if previous is None:
        if recipe["mode"] != "legacy":
            raise ValueError("old checkpoint requires --batch-scaling legacy; new SDE recipe "
                             "requires a fresh OUTPUT, not silent optimizer migration")
        previous = resolve(config["global_batch"], "legacy", config["lr_new"], config["lr_pretrained"])
    if previous != recipe:
        raise ValueError("resume optimization recipe mismatch; use original recipe or a fresh OUTPUT")


def validate_saved_groups(groups, recipe):
    for group in groups:
        key = {"fresh": "lr_new", "pretrained": "lr_pretrained"}.get(group.get("name"))
        if key is None:
            raise ValueError("unknown saved optimizer group")
        expected = dict(base_lr=recipe[key], eps=recipe["eps"], weight_decay=recipe["weight_decay"])
        if list(group["betas"]) != recipe["betas"] or any(group.get(k) != v for k, v in expected.items()):
            raise ValueError("saved optimizer hyperparameters disagree with resolved recipe")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--global-batch", type=int, default=448)
    add_arguments(p)
    print(json.dumps(from_args(p.parse_args()), indent=2, allow_nan=False))
