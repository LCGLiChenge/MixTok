"""Dynamic-K batch samplers for sparse embedding generation."""
import numpy as np
from torch.utils.data import Sampler

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
