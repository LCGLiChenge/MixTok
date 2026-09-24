"""Official Halton components restricted to Router-selected grid positions.

Network and scheduler imports come from the pinned upstream checkout. The
training reduction matches its unweighted all-position CE within selected K.
Sampling follows the official Halton order, arccos reveal schedule, temperature,
and CFG convention; MASK is excluded only from the categorical sampling support.
"""
import math
import ast
import os
import types
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.feature_to_token_20260916.halton_proxy import upstream_transformer


def official_components(root: Path):
    upstream_transformer(root)
    from Utils.masking_scheduler import get_mask_code
    from Sampler.halton_sampler import HaltonSampler
    return get_mask_code, HaltonSampler


def official_training_api(root: Path, model, lr=1e-4, warm_up=2500, max_iter=1000000):
    """Execute upstream's three training helpers verbatim, without FID imports.

    The upstream module imports its entire CLIP/FID/dataset stack at import time.
    Selecting these method ASTs leaves the optimizer/loss/schedule code intact
    and avoids introducing unrelated evaluation dependencies into every rank.
    """
    source = Path(root) / 'Trainer/abstract_trainer.py'
    tree = ast.parse(source.read_text(), filename=str(source))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Trainer')
    names = {'get_optim', 'get_loss', 'adapt_learning_rate'}
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in cls.body} != names:
        raise RuntimeError('official training API changed')
    module = ast.Module(body=[cls], type_ignores=[])
    namespace = dict(os=os, np=np, torch=torch, nn=nn, optim=torch.optim)
    exec(compile(ast.fix_missing_locations(module), str(source), 'exec'), namespace)
    trainer = namespace['Trainer']()
    trainer.args = types.SimpleNamespace(resume=False, lr=lr, warm_up=warm_up,
                                         max_iter=max_iter, iter=0)
    trainer.optim = trainer.get_optim(model, lr, betas=(.9, .999), weight_decay=.03)
    return trainer


class HaltonCompletionObjective(nn.Module):
    def __init__(self, core, upstream_root):
        super().__init__()
        self.core = core
        self.get_mask_code, _ = official_components(upstream_root)

    def forward(self, proxy, gt_sparse, labels, indices, valid):
        # K-bucketing is deliberate: official CE weights every token equally.
        if not bool(valid.all()) or indices.shape != gt_sparse.shape:
            raise ValueError('expected homogeneous K buckets without padding')
        if not bool((indices.sort(1).values[:, 1:] > indices.sort(1).values[:, :-1]).all()):
            raise ValueError('duplicate route coordinates')
        if not bool(((indices >= 0) & (indices < 256)).all()):
            raise ValueError('invalid route coordinates')
        # Match upstream random-number order: class dropout precedes masking.
        drop = (torch.rand(len(labels)) < .1).to(labels.device)
        masked, mask, loss_mask = self.get_mask_code(
            gt_sparse[:, None, :], mode='arccos', value=16384, codebook_size=16384)
        inputs = proxy.clone().scatter(1, indices, masked[:, 0])
        target = torch.where(loss_mask[:, 0], gt_sparse.detach(), -100)
        logits = self.core(inputs, labels, drop, indices)
        if logits.shape[-1] != 16385:
            raise RuntimeError('training must retain the MASK output class')
        loss = F.cross_entropy(logits.flatten(0, 1), target.flatten(), ignore_index=-100)
        with torch.no_grad():
            nll = F.cross_entropy(logits.flatten(0, 1), gt_sparse.flatten(), reduction='none')
            hidden = mask[:, 0].flatten()
            # Return sums/counts so distributed masked NLL is token-weighted.
            stats = torch.stack((loss.detach(), nll[hidden].sum(), hidden.sum(),
                                 hidden.float().mean(), drop.float().mean()))
        return loss, stats


@torch.no_grad()
def sample_halton_completion(core, proxy, selected, labels, upstream_root,
                             steps=32, cfg_w=1.5, temperature=1., trace=None):
    if core.training:
        raise ValueError('sampling requires eval mode')
    if proxy.shape != selected.shape or selected.dtype != torch.bool:
        raise ValueError('invalid selection')
    if not bool(((proxy >= 0) & (proxy < 16384)).all()):
        raise ValueError('proxy must contain real code IDs')
    if steps < 1:
        raise ValueError('positive sampling steps required')
    _, HaltonSampler = official_components(Path(upstream_root))
    xy = HaltonSampler.build_halton_mask(16).to(proxy.device)
    order = xy[:, 0] * 16 + xy[:, 1]
    if len(order) != 256 or order.unique().numel() != 256:
        raise RuntimeError('invalid official Halton permutation')
    filtered = [order[selected[row, order]] for row in range(len(proxy))]
    counts = selected.sum(1)
    prev = torch.zeros_like(counts)
    code = proxy.masked_fill(selected, 16384)
    for step in range(steps):
        ratio = (step + 1) / steps
        # Same arccos progression and minimum one position/round as upstream.
        fraction = 1 - torch.arccos(torch.tensor(ratio)).item() / (math.pi * .5)
        end = torch.minimum(torch.maximum((counts * fraction).long(),
                             torch.full_like(counts, step + 1)), counts)
        if step == steps - 1:
            end = counts
        update = torch.zeros_like(selected)
        for row, positions in enumerate(filtered):
            update[row, positions[int(prev[row]):int(end[row])]] = True
        if bool(update.any()):
            if cfg_w != 0:
                drop = torch.cat((torch.zeros_like(labels, dtype=torch.bool),
                                  torch.ones_like(labels, dtype=torch.bool)))
                cond, uncond = core(torch.cat((code, code)), torch.cat((labels, labels)), drop).chunk(2)
                logits = (1 + cfg_w) * cond - cfg_w * uncond
            else:
                logits = core(code, labels)
            logits = logits[..., :16384]  # sampling-only restriction
            predicted = torch.distributions.Categorical(logits=logits * temperature).sample()
            code[update] = predicted[update]
        if not torch.equal(code[~selected], proxy[~selected]):
            raise RuntimeError('proxy anchors changed')
        if trace is not None:
            trace.append(dict(step=step, updated=int(update.sum()),
                              remaining=int((code == 16384).sum()), anchors_exact=True))
        prev = end
    if bool((code == 16384).any()):
        raise RuntimeError('Halton completion left MASK tokens')
    return code
