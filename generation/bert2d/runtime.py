"""Shared objective, sampler and latest-only checkpoint implementation."""
import os,shutil
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Sampler
from safetensors import safe_open
from safetensors.torch import save_file
from .paths import atomic_json,sha256
from h20.data import KBucketBatchSampler
class RankBucketSampler(Sampler):
    """Generate identical global K-batches on all ranks, then disjoint slices."""
    def __init__(self, dataset, micro, world, rank, seed):
        if micro < 1 or not 0 <= rank < world:
            raise ValueError('invalid rank/micro batch')
        self.base = KBucketBatchSampler(dataset, micro*world, True, True, seed)
        self.micro, self.rank = micro, rank

    def set_epoch(self, epoch):
        self.base.set_epoch(epoch)

    def __len__(self):
        return len(self.base)

    def __iter__(self):
        start = self.rank*self.micro
        for global_rows in self.base:
            yield global_rows[start:start+self.micro]

def optimizer_description(optimizer, core):
    names = {id(p): n for n, p in core.named_parameters()}
    return [{**{k: v for k, v in g.items() if k != 'params'},
             'parameter_names': [names[id(p)] for p in g['params']]} for g in optimizer.param_groups]

class OffsetSampler:
    def __init__(self, base):
        self.base = base
        self.offset = 0

    def set_epoch(self, epoch):
        self.base.set_epoch(epoch)

    def __len__(self):
        return max(0, len(self.base) - self.offset)

    def __iter__(self):
        from itertools import islice
        return islice(iter(self.base), self.offset, None)

def smoothed_objective(logits, targets, masked, valid):
    logp = F.log_softmax(logits.float(), dim=-1)
    nll = -logp.gather(-1, targets[..., None]).squeeze(-1)
    ce = 0.9 * nll - 0.1 * logp.mean(-1)
    weights = torch.where(masked, 1.0, 0.1) * valid
    loss = ((ce * weights).sum(-1) / weights.sum(-1).clamp_min(1)).mean()
    hard = ((nll * masked).sum(-1) / masked.sum(-1).clamp_min(1)).mean()
    return loss, hard.detach()

def save_latest(output: Path, core: nn.Module, ema: nn.Module | None, optimizer: torch.optim.Optimizer, rng_states: list, metadata: dict) -> dict:
    if shutil.disk_usage(output).free < 10 * 1024**3:
        raise RuntimeError("less than 10GiB checkpoint safety headroom")
    tensors = {"raw/" + name: value.detach().cpu().contiguous() for name, value in core.state_dict().items()}
    if ema is not None:
        tensors.update({"ema/" + name: value.detach().cpu().contiguous() for name, value in ema.state_dict().items()})
    for name, parameter in core.named_parameters():
        for key, value in optimizer.state.get(parameter, {}).items():
            if torch.is_tensor(value):
                tensors["adam/" + name + "/" + key] = value.detach().cpu().contiguous()
    for rank, state in enumerate(rng_states):
        tensors[f"rng/{rank}/cpu"] = state["cpu"]
        tensors[f"rng/{rank}/cuda"] = state["cuda"]
    temporary = output / "latest.pt.tmp"
    format_name = metadata.get("config", {}).get("style", "fullctx_unified_maskgit_v1")
    save_file(tensors, str(temporary), metadata={"format": str(format_name)})
    with safe_open(str(temporary), framework="pt", device="cpu") as reader:
        if set(reader.keys()) != set(tensors):
            raise ValueError("checkpoint key mismatch")
        for name, expected in tensors.items():
            actual = reader.get_tensor(name)
            if actual.dtype != expected.dtype or not torch.equal(actual, expected):
                raise ValueError("checkpoint round-trip mismatch: " + name)
    result = dict(
        metadata,
        format=str(format_name),
        model_class=core.__class__.__module__ + "." + core.__class__.__name__,
        optimizer=optimizer_description(optimizer, core),
        tensors=len(tensors),
        bytes=temporary.stat().st_size,
        sha256=sha256(temporary),
        round_trip_exact=True,
    )
    os.replace(temporary, output / "latest.pt")
    atomic_json(output / "latest.json", result)
    return {key: result[key] for key in ("step", "sha256", "bytes", "tensors", "round_trip_exact")}
