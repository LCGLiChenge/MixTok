"""Packed-code dataset utilities for E117 sparse MaskGIT training."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


class PackedOneDCodeDataset(Dataset):
    """Flatten all packed 1D augmentations with a source-disjoint hash split."""

    def __init__(
        self,
        packed_root: str | Path,
        split: str,
        eval_fraction: float = 0.002,
        split_seed: int = 20260901,
        excluded_source_indices: np.ndarray | list[int] | None = None,
    ) -> None:
        if split not in {"train", "eval", "all"}:
            raise ValueError("split must be train, eval, or all")
        self.packed_root = Path(packed_root).resolve()
        metadata = json.loads((self.packed_root / "meta.json").read_text())
        if not metadata.get("completed", False) or int(metadata.get("format_version", -1)) != 1:
            raise RuntimeError("packed code cache is incomplete or unsupported")
        self.titok = np.load(self.packed_root / "titok_codes.npy", mmap_mode="r")
        self.labels = np.load(self.packed_root / "labels.npy", mmap_mode="r")
        if self.titok.ndim != 3 or self.titok.shape[2] != 32:
            raise ValueError("packed TiTok codes must have shape [source, augmentation, 32]")
        if self.labels.shape != (self.titok.shape[0],):
            raise ValueError("packed labels do not match TiTok source count")
        self.num_aug = int(self.titok.shape[1])
        if self.num_aug != int(metadata.get("num_aug", -1)):
            raise ValueError("packed augmentation count does not match metadata")

        source_indices = np.arange(self.titok.shape[0], dtype=np.uint64)
        hashed = source_indices * np.uint64(6364136223846793005) + np.uint64(int(split_seed))
        eval_cutoff = int(round(float(eval_fraction) * 10000))
        if split != "all" and not 1 <= eval_cutoff <= 9999:
            raise ValueError("eval_fraction must map to a cutoff in [1, 9999]/10000")
        is_eval = (hashed % np.uint64(10000)) < np.uint64(eval_cutoff)
        selected = is_eval if split == "eval" else ~is_eval if split == "train" else np.ones_like(is_eval)
        if excluded_source_indices is not None:
            excluded = np.asarray(excluded_source_indices, dtype=np.int64).reshape(-1)
            if excluded.size and (excluded.min() < 0 or excluded.max() >= self.titok.shape[0]):
                raise ValueError("excluded source index is outside the packed dataset")
            selected = selected.copy()
            selected[excluded] = False
        self.source_indices = np.flatnonzero(selected).astype(np.int64)
        if self.source_indices.size == 0:
            raise RuntimeError(f"empty {split} split")

    def __len__(self) -> int:
        return int(self.source_indices.size * self.num_aug)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        source = int(self.source_indices[int(index) // self.num_aug])
        augmentation = int(index) % self.num_aug
        z1d = np.array(self.titok[source, augmentation], dtype=np.int64, copy=True)
        return {
            "z1d": torch.from_numpy(z1d),
            "label": torch.tensor(int(self.labels[source]), dtype=torch.long),
            "source_index": torch.tensor(source, dtype=torch.long),
            "augmentation": torch.tensor(augmentation, dtype=torch.long),
        }


class E117SparseCodeDataset(Dataset):
    """Flatten paired augmentations while keeping source images split-disjoint."""

    def __init__(
        self,
        packed_root: str | Path,
        route_cache: str | Path,
        split: str,
        eval_fraction: float = 0.1,
        split_seed: int = 20260901,
    ) -> None:
        if split not in {"train", "eval", "all"}:
            raise ValueError("split must be train, eval, or all")
        self.packed_root = Path(packed_root).resolve()
        self.route_cache = Path(route_cache).resolve()
        metadata = json.loads((self.route_cache / "meta.json").read_text())
        if metadata.get("format") != "e117_sparse_route_cache_v1" or not metadata.get("completed", False):
            raise RuntimeError("E117 route cache is incomplete or unsupported")
        self.titok = np.load(self.packed_root / "titok_codes.npy", mmap_mode="r")
        self.llamagen = np.load(self.packed_root / "llamagen_codes.npy", mmap_mode="r")
        self.labels = np.load(self.packed_root / "labels.npy", mmap_mode="r")
        self.route_k = np.load(self.route_cache / "route_k.npy", mmap_mode="r")
        self.route_indices = np.load(self.route_cache / "route_indices.npy", mmap_mode="r")
        self.source_indices = np.load(self.route_cache / "source_indices.npy", mmap_mode="r")
        if self.route_k.shape != self.route_indices.shape[:2] or self.route_indices.shape[-1] != 128:
            raise ValueError("route cache arrays have inconsistent shapes")
        if self.route_k.shape[0] != self.source_indices.shape[0]:
            raise ValueError("route cache source index shape mismatch")
        if self.route_k.shape[1] != self.titok.shape[1]:
            raise ValueError("route cache augmentation count mismatch")

        eval_fraction = float(eval_fraction)
        if not 0.0 < eval_fraction < 1.0 and split != "all":
            raise ValueError("eval_fraction must be in (0,1)")
        # Integer hashing avoids augmentation leakage and has no global RNG state.
        hashed = (
            (self.source_indices.astype(np.uint64) * np.uint64(6364136223846793005))
            + np.uint64(int(split_seed))
        )
        eval_cutoff = int(round(eval_fraction * 10000))
        is_eval = (hashed % np.uint64(10000)) < np.uint64(eval_cutoff)
        if split == "train":
            selected = ~is_eval
        elif split == "eval":
            selected = is_eval
        else:
            selected = np.ones_like(is_eval, dtype=np.bool_)
        self.local_rows = np.flatnonzero(selected).astype(np.int64)
        self.num_aug = int(self.route_k.shape[1])
        if self.local_rows.size == 0:
            raise RuntimeError(f"empty {split} split")
        k_values = np.asarray(self.route_k[self.local_rows]).reshape(-1)
        if not np.all((k_values == 64) | (k_values == 128)):
            raise RuntimeError("route cache contains K outside {64,128}")
        self.k_values = k_values.astype(np.uint8, copy=True)

    def __len__(self) -> int:
        return int(self.local_rows.size * self.num_aug)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        local_row = int(self.local_rows[index // self.num_aug])
        aug = int(index % self.num_aug)
        source = int(self.source_indices[local_row])
        k = int(self.route_k[local_row, aug])
        indices = np.array(self.route_indices[local_row, aug, :k], dtype=np.int64, copy=True)
        if indices.shape != (k,) or np.unique(indices).size != k or np.any(indices[:-1] >= indices[1:]):
            raise RuntimeError("cached E117 coordinates must be unique and strictly sorted")
        z1d = np.array(self.titok[source, aug], dtype=np.int64, copy=True)
        full_2d = np.asarray(self.llamagen[source, aug])
        selected_2d = np.array(full_2d[indices], dtype=np.int64, copy=True)
        return {
            "z1d": torch.from_numpy(z1d),
            "z2d": torch.from_numpy(selected_2d),
            "route_indices": torch.from_numpy(indices),
            "k": torch.tensor(k, dtype=torch.long),
            "label": torch.tensor(int(self.labels[source]), dtype=torch.long),
            "source_index": torch.tensor(source, dtype=torch.long),
            "augmentation": torch.tensor(aug, dtype=torch.long),
        }


def collate_e117_sparse(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    max_k = max(int(item["k"]) for item in batch)
    if max_k not in (64, 128):
        raise ValueError("batch maximum K must be 64 or 128")
    size = len(batch)
    z2d = torch.zeros(size, max_k, dtype=torch.long)
    indices = torch.zeros(size, max_k, dtype=torch.long)
    valid = torch.zeros(size, max_k, dtype=torch.bool)
    for row, item in enumerate(batch):
        k = int(item["k"])
        z2d[row, :k] = item["z2d"]
        indices[row, :k] = item["route_indices"]
        valid[row, :k] = True
    return {
        "z1d": torch.stack([item["z1d"] for item in batch]),
        "z2d": z2d,
        "route_indices": indices,
        "route_valid": valid,
        "k": torch.stack([item["k"] for item in batch]),
        "label": torch.stack([item["label"] for item in batch]),
        "source_index": torch.stack([item["source_index"] for item in batch]),
        "augmentation": torch.stack([item["augmentation"] for item in batch]),
    }


class KBucketBatchSampler(Sampler[list[int]]):
    """Homogeneous K64/K128 batches so padding does not erase sparse speedups."""

    def __init__(
        self,
        dataset: E117SparseCodeDataset,
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        seed: int,
        interleave: bool = False,
    ) -> None:
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.interleave = bool(interleave)
        self.epoch = 0
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.by_k = {
            64: np.flatnonzero(dataset.k_values == 64).astype(np.int64),
            128: np.flatnonzero(dataset.k_values == 128).astype(np.int64),
        }
        if not len(self.by_k[64]) or not len(self.by_k[128]):
            raise RuntimeError("both E117 K buckets are required")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        batches_by_k: dict[int, list[list[int]]] = {}
        for k in (64, 128):
            bucket_batches: list[list[int]] = []
            values = self.by_k[k].copy()
            if self.shuffle:
                rng.shuffle(values)
            stop = len(values) if not self.drop_last else len(values) - len(values) % self.batch_size
            for start in range(0, stop, self.batch_size):
                current = values[start : start + self.batch_size]
                if len(current) == self.batch_size or not self.drop_last:
                    bucket_batches.append(current.tolist())
            batches_by_k[k] = bucket_batches
        if self.interleave:
            batches = []
            for offset in range(max(len(batches_by_k[64]), len(batches_by_k[128]))):
                for k in (64, 128):
                    if offset < len(batches_by_k[k]):
                        batches.append(batches_by_k[k][offset])
        else:
            batches = batches_by_k[64] + batches_by_k[128]
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        if self.drop_last:
            return sum(len(values) // self.batch_size for values in self.by_k.values())
        return sum(math.ceil(len(values) / self.batch_size) for values in self.by_k.values())
