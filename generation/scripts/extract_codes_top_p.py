#!/usr/bin/env python3
"""Extract paired TiTok-L32 and MoT-adapted LlamaGen VQ-16 codes.

The output is written directly in the packed format consumed by
``train_titok_llamagen_unified_ar.py``.  Multiple torchrun ranks write disjoint
ImageNet indices into shared ``.npy`` memmaps, avoiding millions of small files.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from PIL import Image
from safetensors.torch import load_file as load_safetensors_file
from torch.utils.data import DataLoader, Sampler
from torchvision import datasets, transforms
from tqdm import tqdm


def center_crop_arr(pil_image, image_size):
    """ADM center crop used by the verified packed-code extraction."""

    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(value // 2 for value in pil_image.size),
            resample=Image.Resampling.BOX,
        )
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(value * scale) for value in pil_image.size),
        resample=Image.Resampling.BICUBIC,
    )
    array = np.asarray(pil_image)
    crop_y = (array.shape[0] - image_size) // 2
    crop_x = (array.shape[1] - image_size) // 2
    return Image.fromarray(array[crop_y : crop_y + image_size, crop_x : crop_x + image_size])


class ImageFolderWithIndex(datasets.ImageFolder):
    def __init__(self, root, transform, limit_samples=0, image_list=None):
        if image_list is None:
            super().__init__(root, transform=transform)
        else:
            # This is the same sorted path/class order produced by ImageFolder.
            # The full label array is audited against the historical cache.
            root_path = Path(root).resolve()
            paths = sorted(Path(image_list).read_text().splitlines())
            if not paths or any(not Path(path).is_relative_to(root_path) for path in paths):
                raise ValueError("image list is empty or contains paths outside data root")
            self.root = str(root_path)
            self.transform = transform
            self.target_transform = None
            self.loader = datasets.folder.default_loader
            self.classes = sorted({Path(path).parent.name for path in paths})
            self.class_to_idx = {name: index for index, name in enumerate(self.classes)}
            self.samples = [(path, self.class_to_idx[Path(path).parent.name]) for path in paths]
            self.imgs = self.samples
            self.targets = [target for _, target in self.samples]
        self.full_length = len(self.samples)
        self.limit_samples = min(int(limit_samples), self.full_length) if limit_samples else self.full_length

    def __len__(self):
        return self.limit_samples

    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        return sample, target, index


class PendingStridedSampler(Sampler[int]):
    """Assign each real dataset index to exactly one rank, without padding."""

    def __init__(self, dataset_size, rank, world_size, written=None):
        indices = range(rank, dataset_size, world_size)
        if written is None:
            self.indices = list(indices)
        else:
            self.indices = [index for index in indices if int(written[index]) == 0]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class PairedTransform:
    def __init__(self, image_size):
        self.image_size = int(image_size)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
            inplace=False,
        )

    def __call__(self, image):
        image = center_crop_arr(image, self.image_size)
        titok_image = self.to_tensor(image)
        llamagen_image = self.normalize(titok_image.clone())
        return titok_image, llamagen_image


def distributed_context():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    return rank, world_size, local_rank


def barrier(world_size):
    if world_size > 1:
        dist.barrier()


def broadcast_rank0_error(error, rank, world_size):
    """Propagate rank-0 setup failures so peers do not hang at a barrier."""
    if world_size > 1:
        payload = [error if rank == 0 else None]
        dist.broadcast_object_list(payload, src=0)
        error = payload[0]
    if error is not None:
        raise RuntimeError(error)


def load_titok(args, device):
    sys.path.insert(0, str(args.titok_root))
    from modeling.titok import TiTok

    tokenizer = TiTok(OmegaConf.load(args.titok_config))
    if str(args.titok_ckpt).endswith(".safetensors"):
        state = load_safetensors_file(str(args.titok_ckpt), device="cpu")
    else:
        state = torch.load(args.titok_ckpt, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
    tokenizer.load_state_dict(state, strict=True)
    return tokenizer.to(device).eval().requires_grad_(False)


def load_llamagen(args, device):
    sys.path.insert(0, str(args.llamagen_root))
    from vq_model import VQ_models

    tokenizer = VQ_models["VQ-16"](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim,
    )
    mot_ckpt = torch.load(
        args.mot_ckpt,
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    if args.mot_state_key not in mot_ckpt:
        raise KeyError(
            f"{args.mot_state_key!r} not found in {args.mot_ckpt}; "
            f"available={sorted(mot_ckpt)}"
        )
    prefix = "llamagen_vq."
    current = tokenizer.state_dict()
    parameter_names = set(dict(tokenizer.named_parameters()))
    adapted = {
        key[len(prefix) :]: value
        for key, value in mot_ckpt[args.mot_state_key].items()
        if key.startswith(prefix) and key[len(prefix) :] in current
    }
    missing_adapted_parameters = sorted(parameter_names - set(adapted))
    if args.require_full_llamagen_state and missing_adapted_parameters:
        raise RuntimeError(
            "MoT checkpoint does not fully cover LlamaGen VQ parameters; "
            f"missing={missing_adapted_parameters[:20]}"
        )
    missing, unexpected = tokenizer.load_state_dict(adapted, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected adapted LlamaGen keys: {unexpected[:20]}")
    missing_parameters = sorted(key for key in missing if key in parameter_names)
    if missing_parameters:
        raise RuntimeError(f"Adapted LlamaGen parameters were not loaded: {missing_parameters[:20]}")
    unrestored_buffers = sorted(key for key in missing if key not in parameter_names)
    allowed_runtime_buffers = ["quantize.codebook_used"]
    if unrestored_buffers != allowed_runtime_buffers:
        raise RuntimeError(
            "Unexpected LlamaGen buffers missing from MoT checkpoint: "
            f"{unrestored_buffers}; expected only {allowed_runtime_buffers}"
        )

    checkpoint_args = mot_ckpt.get("args", {})
    checkpoint_args = vars(checkpoint_args) if hasattr(checkpoint_args, "__dict__") else checkpoint_args
    info = {
        "step": int(mot_ckpt.get("step", -1)),
        "loaded_keys": len(adapted),
        "unrestored_buffers": unrestored_buffers,
        "mask_ratio": checkpoint_args.get("mask_ratio") if isinstance(checkpoint_args, dict) else None,
        "router_min_tokens": checkpoint_args.get("router_min_tokens") if isinstance(checkpoint_args, dict) else None,
        "router_max_tokens": checkpoint_args.get("router_max_tokens") if isinstance(checkpoint_args, dict) else None,
    }
    del mot_ckpt, adapted, current, parameter_names
    return tokenizer.to(device).eval().requires_grad_(False), info


def create_or_open_arrays(output_root, n, num_aug, rank, world_size, resume):
    output_root = Path(output_root)
    paths = {
        "titok": output_root / "titok_codes.npy",
        "llamagen": output_root / "llamagen_codes.npy",
        "labels": output_root / "labels.npy",
        "written": output_root / "written.npy",
    }
    rank0_error = None
    if rank == 0:
        try:
            output_root.mkdir(parents=True, exist_ok=True)
            existing = [str(path) for path in paths.values() if path.exists()]
            if existing and not resume:
                raise FileExistsError(
                    "Output arrays already exist; choose a new --output-root or pass --resume: "
                    + ", ".join(existing)
                )
            if not resume:
                np.lib.format.open_memmap(paths["titok"], mode="w+", dtype=np.uint16, shape=(n, num_aug, 32)).flush()
                np.lib.format.open_memmap(paths["llamagen"], mode="w+", dtype=np.uint16, shape=(n, num_aug, 256)).flush()
                np.lib.format.open_memmap(paths["labels"], mode="w+", dtype=np.uint16, shape=(n,)).flush()
                np.lib.format.open_memmap(paths["written"], mode="w+", dtype=np.uint8, shape=(n,)).flush()
        except Exception as exc:
            rank0_error = f"rank-0 output initialization failed: {type(exc).__name__}: {exc}"
    broadcast_rank0_error(rank0_error, rank, world_size)
    barrier(world_size)

    arrays = {name: np.load(path, mmap_mode="r+") for name, path in paths.items()}
    expected = {
        "titok": (n, num_aug, 32),
        "llamagen": (n, num_aug, 256),
        "labels": (n,),
        "written": (n,),
    }
    for name, shape in expected.items():
        if arrays[name].shape != shape:
            raise ValueError(f"{paths[name]} has shape {arrays[name].shape}, expected {shape}")
    return arrays


def make_aug_batch(images, aug_mode):
    if aug_mode == "none":
        return images, 1
    if aug_mode != "adm":
        raise ValueError(f"Unsupported augmentation mode: {aug_mode}")
    flipped = torch.flip(images, dims=[-1])
    output = torch.empty((images.shape[0] * 2, *images.shape[1:]), dtype=images.dtype, device=images.device)
    output[0::2] = images
    output[1::2] = flipped
    return output, 2


def write_metadata(args, output_root, dataset, mot_info, world_size, completed, written_count):
    metadata = {
        "format_version": 1,
        "completed": bool(completed),
        "written_samples": int(written_count),
        "num_samples": len(dataset),
        "num_aug": 2 if args.aug_mode == "adm" else 1,
        "titok_width": 32,
        "titok_vocab_size": 4096,
        "llamagen_width": 256,
        "llamagen_vocab_size": args.codebook_size,
        "dtype": "uint16",
        "data_path": str(Path(args.data_path).resolve()),
        "image_list": str(args.image_list.resolve()) if args.image_list else None,
        "image_list_sha256": sha256_file(args.image_list) if args.image_list else None,
        "sample_order": "sorted full paths, matching ImageFolder" if args.image_list else "ImageFolder",
        "image_size": args.image_size,
        "augmentation": args.aug_mode,
        "preprocessing": {
            "crop": "RandAR center_crop_arr",
            "titok_range": "zero_1",
            "llamagen_range": "minus1_1",
        },
        "titok_config": str(Path(args.titok_config).resolve()),
        "titok_ckpt": str(Path(args.titok_ckpt).resolve()),
        "mot_ckpt": str(Path(args.mot_ckpt).resolve()),
        "mot_state_key": args.mot_state_key,
        "mot_step": mot_info["step"],
        "mot_mask_ratio": mot_info["mask_ratio"],
        "mot_router_min_tokens": mot_info["router_min_tokens"],
        "mot_router_max_tokens": mot_info["router_max_tokens"],
        "llamagen_loaded_keys": mot_info["loaded_keys"],
        "llamagen_unrestored_buffers": mot_info["unrestored_buffers"],
        "extraction_precision": args.mixed_precision,
        "world_size": world_size,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (Path(output_root) / "meta.json").write_text(json.dumps(metadata, indent=2) + "\n")


def sha256_file(path, chunk_size=16 * 1024 * 1024):
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(output_root):
    root = Path(output_root)
    names = (
        "titok_codes.npy",
        "llamagen_codes.npy",
        "labels.npy",
        "written.npy",
        "meta.json",
    )
    lines = [f"{sha256_file(root / name)}  {name}" for name in names]
    (root / "manifest.sha256").write_text("\n".join(lines) + "\n")

def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for tokenizer code extraction")
    rank, world_size, local_rank = distributed_context()
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    transform = PairedTransform(args.image_size)
    dataset = ImageFolderWithIndex(args.data_path, transform, args.limit_samples,
                                   args.image_list)
    if len(dataset.classes) != args.num_classes:
        raise ValueError(f"Expected {args.num_classes} ImageNet classes, found {len(dataset.classes)}")

    titok = load_titok(args, device)
    llamagen, mot_info = load_llamagen(args, device)
    if rank == 0:
        print(f"Loaded MoT step {mot_info['step']} {args.mot_state_key}: {mot_info['loaded_keys']} LlamaGen keys", flush=True)

    num_aug = 2 if args.aug_mode == "adm" else 1
    arrays = create_or_open_arrays(args.output_root, len(dataset), num_aug, rank, world_size, args.resume)
    sampler = PendingStridedSampler(len(dataset), rank, world_size, arrays["written"] if args.resume else None)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    progress = tqdm(total=len(sampler), desc="extract packed codes", unit="img", disable=rank != 0)
    autocast_enabled = args.mixed_precision != "none"
    autocast_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16

    with torch.inference_mode():
        for (titok_images, llamagen_images), labels, indices in loader:
            titok_images = titok_images.to(device, non_blocking=True)
            llamagen_images = llamagen_images.to(device, non_blocking=True)
            titok_aug, observed_num_aug = make_aug_batch(titok_images, args.aug_mode)
            llamagen_aug, _ = make_aug_batch(llamagen_images, args.aug_mode)
            with torch.autocast("cuda", dtype=autocast_dtype, enabled=autocast_enabled):
                titok_codes = titok.encode(titok_aug)[1]["min_encoding_indices"]
                _llamagen_quant, _llamagen_loss, llamagen_info = llamagen.encode(llamagen_aug)
                if llamagen_info is None or len(llamagen_info) < 3 or llamagen_info[2] is None:
                    raise RuntimeError("LlamaGen encode did not return code indices in info[2]")
                llamagen_codes = llamagen_info[2]
            batch_size = indices.shape[0]
            titok_codes = titok_codes.reshape(batch_size, observed_num_aug, -1)
            llamagen_codes = llamagen_codes.reshape(batch_size, observed_num_aug, -1)
            if titok_codes.shape[-1] != 32 or llamagen_codes.shape[-1] != 256:
                raise RuntimeError(
                    f"Unexpected code shapes: TiTok={tuple(titok_codes.shape)}, "
                    f"LlamaGen={tuple(llamagen_codes.shape)}"
                )
            if int(titok_codes.min()) < 0 or int(titok_codes.max()) >= 4096:
                raise RuntimeError("TiTok token id is outside [0, 4096)")
            if int(llamagen_codes.min()) < 0 or int(llamagen_codes.max()) >= args.codebook_size:
                raise RuntimeError(f"LlamaGen token id is outside [0, {args.codebook_size})")

            index_np = indices.numpy()
            arrays["titok"][index_np] = titok_codes.cpu().numpy().astype(np.uint16, copy=False)
            arrays["llamagen"][index_np] = llamagen_codes.cpu().numpy().astype(np.uint16, copy=False)
            arrays["labels"][index_np] = labels.numpy().astype(np.uint16, copy=False)
            arrays["written"][index_np] = 1
            progress.update(batch_size)

    progress.close()
    for array in arrays.values():
        array.flush()
    barrier(world_size)
    if rank == 0:
        written_count = int(np.count_nonzero(arrays["written"]))
        completed = written_count == len(dataset)
        write_metadata(args, args.output_root, dataset, mot_info, world_size, completed, written_count)
        if not completed:
            raise RuntimeError(f"Only {written_count:,}/{len(dataset):,} samples were written")
        write_manifest(args.output_root)
        print(f"Completed packed extraction: {args.output_root} ({written_count:,} samples)", flush=True)
    barrier(world_size)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--image-list", type=Path)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--mot-ckpt", required=True)
    parser.add_argument("--mot-state-key", default="model_ema", choices=["model", "model_ema"])
    parser.add_argument("--require-full-llamagen-state", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--aug-mode", default="adm", choices=["adm", "none"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--limit-samples", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--mixed-precision", default="none", choices=["none", "bf16", "fp16"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--titok-root", required=True)
    parser.add_argument("--titok-config", required=True)
    parser.add_argument("--titok-ckpt", required=True)
    parser.add_argument("--llamagen-root", required=True)
    parser.add_argument("--codebook-size", type=int, default=16384)
    parser.add_argument("--codebook-embed-dim", type=int, default=8)
    main(parser.parse_args())
