"""Bounded, full-TRAIN clean-prefix baseline; exact TiTok initialization.

Independent of known-good trainers. New latest.safetensors only; no input
checkpoint edits. A four-update DDP smoke is mandatory before the pilot.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from copy import deepcopy
from datetime import timedelta
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

os.environ.setdefault('USE_TF', '0')
import numpy as np
import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler
from safetensors import safe_open
from safetensors.torch import save_file

from h20.model import TiTokSparseImageBert, load_official_state
from h20.assets import digest
from h20.data import E117SparseCodeDataset, KBucketBatchSampler, collate_e117_sparse
from h20.base_model import sample_arccos_mask

NEW = ('embedding_2d.', 'pos_embedding_2d.', 'budget_embedding.', 'output_2d.')
SOURCE_N = 1281167
STAT_NAMES = ('loss1d', 'loss2d', 'masked_nll1d', 'masked_nll2d', 'mask_ratio1d', 'mask_ratio2d')


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
    os.replace(temporary, path)


def fingerprint(model, shared_only=False):
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        if shared_only and name.startswith(NEW):
            continue
        h.update(name.encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


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


def smoothed_objective(logits, targets, masked, valid):
    """Existing per_sample objective, without expensive unused top-5 sorting."""
    logp = F.log_softmax(logits.float(), dim=-1)
    nll = -logp.gather(-1, targets[..., None]).squeeze(-1)
    ce = .9*nll - .1*logp.mean(-1)
    weights = torch.where(masked, 1., .1)*valid
    loss = ((ce*weights).sum(-1)/weights.sum(-1).clamp_min(1)).mean()
    hard_nll = ((nll*masked).sum(-1)/masked.sum(-1).clamp_min(1)).mean()
    return loss, hard_nll.detach()


class TrainingForward(nn.Module):
    def __init__(self, core):
        super().__init__()
        self.core = core

    def forward(self, batch, joint):
        core = self.core
        z1, z2, valid = batch['z1d'], batch['z2d'], batch['route_valid']
        valid1 = torch.ones_like(z1, dtype=torch.bool)
        mask1, _ = sample_arccos_mask(valid1)
        mask2, _ = sample_arccos_mask(valid)
        logits1 = core(stage='1d', input_tokens=torch.where(mask1, core.mask_token_1d, z1),
                       labels=batch['label'])
        logits2 = core(stage='2d', completed_1d=z1,
                       input_tokens=torch.where(mask2, core.mask_token_2d, z2),
                       route_indices=batch['route_indices'], route_valid=valid, labels=batch['label'])
        loss1, nll1 = smoothed_objective(logits1, z1, mask1, valid1)
        loss2, nll2 = smoothed_objective(logits2, z2, mask2, valid)
        loss = (1.5 if joint else 0.)*loss1 + loss2
        stats = torch.stack((loss1.detach(), loss2.detach(), nll1, nll2,
                             mask1.float().mean(), mask2.float().mean()))
        return loss, stats


def optimizer_for(core):
    return torch.optim.AdamW([dict(params=[p for n, p in core.named_parameters() if n.startswith(NEW)],
                                  lr=1e-4, name='new2d')],
                             betas=(.9, .96), weight_decay=.03)


def enter_joint(optimizer, core):
    if len(optimizer.param_groups) != 1:
        raise ValueError('joint phase entered twice')
    optimizer.add_param_group(dict(params=[p for n, p in core.named_parameters() if not n.startswith(NEW)],
                                   lr=1e-5, name='pretrained'))


def set_lrs(optimizer, step, warmup):
    for group in optimizer.param_groups:
        if group['name'] == 'new2d':
            group['lr'] = 1e-4*min(1., step/50)
        else:
            group['lr'] = 1e-5*min(1., (step-warmup)/100)


@torch.no_grad()
def update_ema(ema, core):
    for target, source in zip(ema.parameters(), core.parameters()):
        target.lerp_(source, .001)
    for target, source in zip(ema.buffers(), core.buffers()):
        target.copy_(source)


def fixed_masks(source, k):
    rng = np.random.default_rng(20260922 + int(source))
    order = rng.permutation(32)
    masks = []
    for count in (8, 16, 32):
        mask = np.zeros(32, dtype=bool)
        mask[order[:count]] = True
        masks.append(mask)
    # Same arccos distribution, at least one masked position.
    count2 = min(k, max(1, int(np.rint(np.arccos(rng.uniform())*2/np.pi*k))))
    mask2 = np.zeros(k, dtype=bool)
    mask2[rng.permutation(k)[:count2]] = True
    return np.stack(masks), mask2


@torch.no_grad()
def evaluate(core, dataset, selected):
    was_training = core.training
    core.eval()
    sums = np.zeros(3, dtype=np.float64)
    sums2 = {64: [], 128: []}
    # No RNG perturbation of dropout, masking or future DataLoader seeds.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        for k in (64, 128):
            rows = [int(i) for i in selected if int(dataset.k_values[i]) == k]
            for start in range(0, len(rows), 32):
                chosen = rows[start:start+32]
                cpu = collate_e117_sparse([dataset[i] for i in chosen])
                batch = {key: val.cuda(non_blocking=True) for key, val in cpu.items()}
                mask_rows = [fixed_masks(int(s), k) for s in cpu['source_index']]
                mask1 = torch.from_numpy(np.stack([m[0] for m in mask_rows])).cuda()
                mask2 = torch.from_numpy(np.stack([m[1] for m in mask_rows])).cuda()
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    for regime in range(3):
                        mask = mask1[:, regime]
                        out = core.forward_1d(torch.where(mask, core.mask_token_1d, batch['z1d']), batch['label'])
                        nll = F.cross_entropy(out.float().transpose(1, 2), batch['z1d'], reduction='none')
                        sums[regime] += float(((nll*mask).sum(-1)/mask.sum(-1)).sum())
                    out = core.forward_2d(batch['z1d'], torch.where(mask2, core.mask_token_2d, batch['z2d']),
                                          batch['route_indices'], batch['route_valid'], batch['label'])
                    nll = F.cross_entropy(out.float().transpose(1, 2), batch['z2d'], reduction='none')
                    values = ((nll*mask2).sum(-1)/mask2.sum(-1)).float().cpu().tolist()
                    sums2[k].extend(values)
    core.train(was_training)
    result = {f'nll1d_mask{count}': float(value/len(selected)) for count, value in zip((8, 16, 32), sums)}
    result['nll1d'] = float(sums.mean()/len(selected))
    result['nll2d'] = float(np.mean(sums2[64]+sums2[128]))
    for k in (64, 128):
        if sums2[k]:
            result[f'nll2d_k{k}'] = float(np.mean(sums2[k]))
            result[f'n_k{k}'] = len(sums2[k])
    if not all(np.isfinite(v) for v in result.values()):
        raise ValueError('nonfinite development metric')
    return result


def optimizer_description(optimizer, core):
    names = {id(p): n for n, p in core.named_parameters()}
    return [{**{k: v for k, v in g.items() if k != 'params'},
             'parameter_names': [names[id(p)] for p in g['params']]} for g in optimizer.param_groups]


def save_latest(output, core, ema, optimizer, rng_states, metadata):
    """Tensor-only state with exact immediate readback; atomic own-output writes."""
    if shutil.disk_usage(output).free < 10*1024**3:
        raise RuntimeError('less than 10GiB checkpoint safety headroom')
    tensors = {}
    for label, model in (('raw', core), ('ema', ema)):
        tensors.update({label+'/'+n: p.detach().cpu().contiguous() for n, p in model.state_dict().items()})
    for name, p in core.named_parameters():
        for key, val in optimizer.state.get(p, {}).items():
            if not torch.is_tensor(val):
                raise TypeError('unexpected non-tensor Adam state')
            tensors['adam/'+name+'/'+key] = val.detach().cpu().contiguous()
    for rank, state in enumerate(rng_states):
        tensors[f'rng/{rank}/cpu'] = state['cpu']
        tensors[f'rng/{rank}/cuda'] = state['cuda']
    temporary = output/'latest.safetensors.tmp'
    save_file(tensors, str(temporary), metadata={'format': 'titok_sparse_bert_clean_prefix_v1'})
    with safe_open(str(temporary), framework='pt', device='cpu') as reader:
        if set(reader.keys()) != set(tensors):
            raise ValueError('checkpoint keys differ on readback')
        for name, expected in tensors.items():
            if not torch.equal(reader.get_tensor(name), expected):
                raise ValueError('checkpoint round-trip mismatch: '+name)
    metadata = dict(metadata, optimizer=optimizer_description(optimizer, core),
                    tensors=len(tensors), bytes=temporary.stat().st_size,
                    sha256=digest(temporary), round_trip_exact=True,
                    format='titok_sparse_bert_clean_prefix_v1',
                    model_class='h20.model.TiTokSparseImageBert',
                    attention_implementation=core.attention_implementation,
                    inference_1d_sampler='upstream ImageBert.generate(OfficialSamplingView(core), ...)')
    os.replace(temporary, output/'latest.safetensors')
    atomic_json(output/'latest.json', metadata)
    return {k: metadata[k] for k in ('sha256', 'bytes', 'tensors', 'round_trip_exact', 'step')}
