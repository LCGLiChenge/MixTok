"""Official Halton parameters with continuous unselected input contexts."""
import math
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from experiments.continuous_halton_embedding_20260923.halton_upstream import (
    official_components, upstream_transformer,
)


class ContinuousHalton(nn.Module):
    vocabulary = 16384
    mask_token = 16384

    def __init__(self, upstream_root: Path, dropout: float = 0.1):
        super().__init__()
        transformer = upstream_transformer(upstream_root)
        self.model = transformer(input_size=16, hidden_dim=768, codebook_size=16384,
                                 depth=12, heads=12, mlp_dim=3072, dropout=dropout,
                                 nclass=1000, register=1, proj=1)

    def forward(self, context, tokens, selected, labels, force_drop_ids=None, output_indices=None):
        if context.shape != (len(tokens), 256, 768) or tokens.shape != selected.shape:
            raise ValueError('continuous input shape differs')
        if tokens.dtype != torch.long or selected.dtype != torch.bool or tokens.shape[1] != 256:
            raise ValueError('bad discrete/route input')
        m = self.model
        if m.proj != 1:
            raise ValueError('only audited official proj=1 is supported')
        if force_drop_ids is None:
            force_drop_ids = torch.zeros_like(labels, dtype=torch.bool)
        y = m.cls_emb(torch.where(force_drop_ids.bool(), torch.full_like(labels, m.nclass), labels))
        x = torch.where(selected[..., None], m.tok_emb(tokens), context)
        pos = m.pos_emb(torch.arange(256, device=tokens.device))
        x = x + pos
        if m.register > 0:
            reg = m.reg_tokens(torch.arange(m.register, device=tokens.device))
            x = torch.cat((x, reg.expand(len(tokens), m.register, m.hidden_dim)), dim=1)
        x = m.transformer(x, y, mask=None)
        x = x[:, :256].contiguous()
        logits = m.head(m.last_norm(x, y))
        if output_indices is not None:
            logits = logits.gather(1, output_indices[..., None].expand(-1, -1, self.vocabulary+1))
        return logits.float()


def halton_prefix_mask(indices: torch.Tensor, order_rank: torch.Tensor,
                       completed_rounds: torch.Tensor, steps: int = 32) -> torch.Tensor:
    """Mask selected positions not revealed after 0..steps-1 Halton rounds."""
    batch, k = indices.shape
    if k < 1 or completed_rounds.shape != (batch,):
        raise ValueError("expected nonempty homogeneous K buckets")
    if bool(((completed_rounds < 0) | (completed_rounds >= steps)).any()):
        raise ValueError("completed_rounds outside sampling prefix")
    fraction = 1 - torch.arccos(completed_rounds.float() / steps) / (math.pi * .5)
    revealed = torch.maximum((k * fraction).long(), completed_rounds)
    revealed = revealed.clamp(max=k - 1)
    selected_rank = order_rank.to(indices.device)[indices]
    position_in_prefix = selected_rank.argsort(dim=1).argsort(dim=1)
    return position_in_prefix >= revealed[:, None]


class ContinuousObjective(nn.Module):
    def __init__(self, core, upstream_root):
        super().__init__()
        self.core = core
        self.get_mask_code, sampler = official_components(Path(upstream_root))
        xy = sampler.build_halton_mask(16)
        order = xy[:, 0]*16 + xy[:, 1]
        if order.numel() != 256 or order.unique().numel() != 256:
            raise RuntimeError('invalid official Halton order')
        self.register_buffer('order_rank', order.argsort(), persistent=False)

    def forward(self, context, gt_sparse, labels, indices, valid):
        if not bool(valid.all()) or indices.shape != gt_sparse.shape:
            raise ValueError('expected homogeneous K buckets without padding')
        if not bool((indices.sort(1).values[:,1:] > indices.sort(1).values[:,:-1]).all()):
            raise ValueError('duplicate route coordinates')
        if not bool(((indices >= 0) & (indices < 256)).all()):
            raise ValueError('invalid route coordinates')
        batch, k = indices.shape
        drop = (torch.rand(batch) < .1).to(labels.device)
        _, random_mask, _ = self.get_mask_code(gt_sparse[:,None,:], mode='arccos', value=16384, codebook_size=16384)
        random_mask = random_mask[:,0]
        empty = ~random_mask.any(dim=1)
        if bool(empty.any()):
            rows = torch.nonzero(empty, as_tuple=True)[0]
            random_mask[rows, torch.randint(k,(len(rows),),device=indices.device)] = True
        completed = torch.randint(0,32,(batch,),device=indices.device)
        prefix_mask = halton_prefix_mask(indices,self.order_rank,completed)
        use_prefix = torch.rand(batch,device=indices.device) < .5
        mask = torch.where(use_prefix[:,None],prefix_mask,random_mask)
        if not bool(mask.any(dim=1).all()):
            raise RuntimeError('no supervised tokens')
        sparse_input = gt_sparse.masked_fill(mask,16384)
        tokens = torch.zeros(batch,256,dtype=torch.long,device=indices.device).scatter(1,indices,sparse_input)
        selected = torch.zeros_like(tokens,dtype=torch.bool).scatter(1,indices,True)
        logits = self.core(context,tokens,selected,labels,drop,indices)
        target = gt_sparse.masked_fill(~mask,-100)
        loss = F.cross_entropy(logits.flatten(0,1),target.flatten(),ignore_index=-100)
        with torch.no_grad():
            nll = F.cross_entropy(logits.flatten(0,1),gt_sparse.flatten(),reduction='none').view_as(gt_sparse)
            stats = torch.stack((loss.detach(),nll[mask].sum(),mask.sum(),mask.float().mean(),drop.float().mean()))
        return loss,stats


@torch.no_grad()
def sample_continuous_completion(core, context, selected, labels, upstream_root,
                                 steps=8,cfg_w=2.75,temperature=1.,trace=None):
    if core.training or steps < 1 or selected.dtype != torch.bool:
        raise ValueError('invalid sampling state')
    if context.shape != (len(selected),256,768):
        raise ValueError('bad continuous context')
    _, sampler = official_components(Path(upstream_root))
    xy = sampler.build_halton_mask(16).to(context.device)
    order = xy[:,0]*16 + xy[:,1]
    filtered = [order[selected[row,order]] for row in range(len(context))]
    counts = selected.sum(1)
    prev = torch.zeros_like(counts)
    code = torch.zeros_like(selected,dtype=torch.long).masked_fill(selected,16384)
    anchor_copy = context[~selected].clone()
    for step in range(steps):
        ratio = (step+1)/steps
        fraction = 1-torch.arccos(torch.tensor(ratio)).item()/(math.pi*.5)
        end = torch.minimum(torch.maximum((counts*fraction).long(),torch.full_like(counts,step+1)),counts)
        if step == steps-1:
            end = counts
        update = torch.zeros_like(selected)
        for row,positions in enumerate(filtered):
            update[row,positions[int(prev[row]):int(end[row])]] = True
        if bool(update.any()):
            if cfg_w != 0:
                drop = torch.cat((torch.zeros_like(labels,dtype=torch.bool),torch.ones_like(labels,dtype=torch.bool)))
                cond, uncond = core(torch.cat((context,context)),torch.cat((code,code)),
                    torch.cat((selected,selected)),torch.cat((labels,labels)),drop).chunk(2)
                logits = (1+cfg_w)*cond-cfg_w*uncond
            else:
                logits = core(context,code,selected,labels)
            predicted = torch.distributions.Categorical(logits=logits[...,:16384]*temperature).sample()
            code[update] = predicted[update]
        if not torch.equal(context[~selected],anchor_copy):
            raise RuntimeError('continuous anchors changed')
        if trace is not None:
            trace.append(dict(step=step,updated=int(update.sum()),remaining=int(((code==16384)&selected).sum()),anchors_exact=True))
        prev = end
    if bool((code[selected]==16384).any()):
        raise RuntimeError('selected MASK tokens remain')
    return code
