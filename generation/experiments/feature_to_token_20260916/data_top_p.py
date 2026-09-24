"""Top-p cache adapter for the unchanged Halton completion objective.

The sampler is the historical KBucketBatchSampler and RankBucketSampler with
only the hard-coded (64, 128) bucket list generalized to actual K values.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from bert2d.paths import sha256


NUM_SOURCES = 1_281_167
NUM_AUG = 2
NUM_ROWS = NUM_SOURCES * NUM_AUG
VOCABULARY = 16_384


class TopPProxyRouteDataset(Dataset):
    def __init__(self, source_root: Path, gt_root: Path, cache_root: Path):
        source_root, gt_root, cache_root = map(Path, (source_root, gt_root, cache_root))
        source_meta = json.loads((source_root / "meta.json").read_text())
        gt_meta = json.loads((gt_root / "meta.json").read_text())
        config = json.loads((cache_root / "config.json").read_text())
        summary = json.loads((cache_root / "summary.json").read_text())
        if (source_meta.get("completed") is not True
                or gt_meta.get("completed") is not True
                or summary.get("status") != "complete"
                or summary.get("complete_coverage") is not True):
            raise RuntimeError("source, GT, or top-p final condition cache incomplete")
        if config.get("format") != "top_p_mapper_route_cache_v1":
            raise RuntimeError("wrong top-p final condition format")
        if config.get("shape") != [NUM_ROWS, 256]:
            raise RuntimeError("top-p condition row count differs")
        if int(gt_meta.get("num_samples", -1)) != NUM_SOURCES or int(gt_meta.get("num_aug", -1)) != NUM_AUG:
            raise RuntimeError("top-p GT shape metadata differs")
        if (int(gt_meta.get("mot_step", -1)) != 247482
                or gt_meta.get("mot_state_key") != "model_ema"
                or Path(gt_meta.get("mot_ckpt", "")).resolve() != Path(config["top_p_checkpoint"]).resolve()
                or Path(config["source_root"]).resolve() != source_root.resolve()
                or config["source_titok_sha256"] != sha256(source_root / "titok_codes.npy")
                or summary.get("config_sha256") != sha256(cache_root / "config.json")):
            raise RuntimeError("top-p GT, 1D codes, and final conditions have different provenance")
        self.z1d = np.load(source_root / "titok_codes.npy", mmap_mode="r", allow_pickle=False)
        self.labels = np.load(source_root / "labels.npy", mmap_mode="r", allow_pickle=False)
        gt_labels = np.load(gt_root / "labels.npy", mmap_mode="r", allow_pickle=False)
        self.gt2d = np.load(gt_root / "llamagen_codes.npy", mmap_mode="r", allow_pickle=False)
        self.proxy = np.load(cache_root / "proxy_codes.npy", mmap_mode="r", allow_pickle=False)
        self.indices = np.load(cache_root / "route_indices.npy", mmap_mode="r", allow_pickle=False)
        self.route_k = np.load(cache_root / "route_k.npy", mmap_mode="r", allow_pickle=False)
        if (self.z1d.shape != (NUM_SOURCES, NUM_AUG, 32)
                or self.gt2d.shape != (NUM_SOURCES, NUM_AUG, 256)
                or self.proxy.shape != (NUM_ROWS, 256)
                or self.indices.shape != (NUM_ROWS, 256)
                or self.route_k.shape != (NUM_ROWS,)
                or self.labels.shape != (NUM_SOURCES,)
                or not np.array_equal(self.labels, gt_labels)):
            raise RuntimeError("top-p cache row, view, or label alignment differs")
        if (self.z1d.dtype != np.uint16 or self.gt2d.dtype != np.uint16
                or self.proxy.dtype != np.uint16 or self.indices.dtype != np.uint16
                or self.route_k.dtype != np.uint16):
            raise RuntimeError("top-p cache dtype differs")
        self.k_values = np.asarray(self.route_k).copy()
        if not np.all((self.k_values >= 1) & (self.k_values <= 256)):
            raise RuntimeError("top-p K outside [1,256]")
        self.proxy_sha256 = summary.get("proxy_codes_sha256")
        if not self.proxy_sha256 or not summary.get("route_indices_sha256"):
            raise RuntimeError("top-p condition cache lacks hashes")

    def __len__(self):
        return NUM_ROWS

    def __getitem__(self, index):
        flat = int(index)
        source, augmentation = divmod(flat, NUM_AUG)
        k = int(self.route_k[flat])
        indices = np.array(self.indices[flat, :k], dtype=np.int64, copy=True)
        if (indices.size != k or (indices < 0).any() or (indices >= 256).any()
                or (indices[:-1] >= indices[1:]).any()):
            raise RuntimeError("top-p route has invalid coordinates")
        selected_gt = np.array(self.gt2d[source, augmentation, indices],
                               dtype=np.int64, copy=True)
        return {
            "z1d": torch.from_numpy(np.array(self.z1d[source, augmentation], dtype=np.int64, copy=True)),
            "z2d": torch.from_numpy(selected_gt),
            "route_indices": torch.from_numpy(indices),
            "k": torch.tensor(k, dtype=torch.long),
            "label": torch.tensor(int(self.labels[source]), dtype=torch.long),
            "source_index": torch.tensor(source, dtype=torch.long),
            "augmentation": torch.tensor(augmentation, dtype=torch.long),
            "proxy": torch.from_numpy(np.array(self.proxy[flat], dtype=np.int64, copy=True)),
        }


def collate_top_p_proxy_route(rows):
    k = int(rows[0]["k"])
    if any(int(row["k"]) != k for row in rows):
        raise RuntimeError("a Halton batch contains mixed K")
    return {
        "z1d": torch.stack([row["z1d"] for row in rows]),
        "z2d": torch.stack([row["z2d"] for row in rows]),
        "route_indices": torch.stack([row["route_indices"] for row in rows]),
        "route_valid": torch.ones(len(rows), k, dtype=torch.bool),
        "k": torch.stack([row["k"] for row in rows]),
        "label": torch.stack([row["label"] for row in rows]),
        "source_index": torch.stack([row["source_index"] for row in rows]),
        "augmentation": torch.stack([row["augmentation"] for row in rows]),
        "proxy": torch.stack([row["proxy"] for row in rows]),
    }


class DynamicKBucketBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle, drop_last, seed):
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.by_k = {int(k): np.flatnonzero(dataset.k_values == k).astype(np.int64)
                     for k in np.unique(dataset.k_values)}
        if not self.by_k:
            raise RuntimeError("no nonempty top-p K bucket")

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        batches = []
        for k in sorted(self.by_k):
            values = self.by_k[k].copy()
            if self.shuffle:
                rng.shuffle(values)
            stop = len(values) if not self.drop_last else len(values) - len(values) % self.batch_size
            for start in range(0, stop, self.batch_size):
                current = values[start:start + self.batch_size]
                if len(current) == self.batch_size or not self.drop_last:
                    batches.append(current.tolist())
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches

    def __len__(self):
        if self.drop_last:
            return sum(len(values) // self.batch_size for values in self.by_k.values())
        return sum((len(values) + self.batch_size - 1) // self.batch_size
                   for values in self.by_k.values())

    def coverage(self):
        return {str(k): {"raw": len(values), "used": (len(values) // self.batch_size) * self.batch_size,
                         "dropped": len(values) % self.batch_size}
                for k, values in self.by_k.items()}


class RankDynamicBucketSampler(Sampler):
    def __init__(self, dataset, micro, world, rank, seed):
        if micro < 1 or not 0 <= rank < world:
            raise ValueError("invalid rank/micro batch")
        self.base = DynamicKBucketBatchSampler(dataset, micro * world, True, True, seed)
        self.micro, self.rank = micro, rank

    def set_epoch(self, epoch):
        self.base.set_epoch(epoch)

    def __len__(self):
        return len(self.base)

    def __iter__(self):
        start = self.rank * self.micro
        for global_rows in self.base:
            yield global_rows[start:start + self.micro]
