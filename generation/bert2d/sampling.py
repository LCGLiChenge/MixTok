"""Follow-up hypotheses locked before fresh-cohort confirmation; original sampler untouched."""
from dataclasses import dataclass,asdict,replace
from functools import lru_cache
import math
import torch
from .paths import enable_h20
enable_h20()
from h20.base_model import _mask_from_counts
from .sampling_base import Sampling,selected_confidence

@dataclass(frozen=True)
class Followup(Sampling):
    draw_scale: float=1.
    confidence_scale: float=1.
    blind_confidence: bool=False
    blend: float=1.
    order: str='confidence'

BASE=Followup()
MARGIN4=replace(BASE,name='margin4',confidence='margin',steps=4)
GREEDY_MARGIN4=replace(MARGIN4,name='greedy_margin4',draw_scale=0.)
VARIANTS=[BASE,replace(BASE,name='margin',confidence='margin'),MARGIN4,
    replace(MARGIN4,name='margin4_cfg1',cfg=1.),
    replace(MARGIN4,name='margin4_cfg2',cfg=2.),
    GREEDY_MARGIN4,
    replace(GREEDY_MARGIN4,name='greedy_margin4_cfg1',cfg=1.),
    replace(GREEDY_MARGIN4,name='greedy_margin4_cfg2',cfg=2.),
    replace(GREEDY_MARGIN4,name='greedy_margin4_cfg3',cfg=3.),
    replace(GREEDY_MARGIN4,name='greedy_margin4_cfg6',cfg=6.),
    replace(GREEDY_MARGIN4,name='greedy_margin4_cfg8',cfg=8.),
    replace(GREEDY_MARGIN4,name='greedy_margin4_cfgramp',cfg_ramp=True),
    replace(GREEDY_MARGIN4,name='greedy_margin2',steps=2),
    replace(GREEDY_MARGIN4,name='greedy_margin8',steps=8),
    replace(GREEDY_MARGIN4,name='greedy_margin4_cosine',schedule='cosine'),
    replace(GREEDY_MARGIN4,name='greedy_probability4',confidence='logprob'),
    replace(GREEDY_MARGIN4,name='greedy_logit4',confidence='logit'),
    replace(MARGIN4,name='halfdraw_margin4',draw_scale=.5),
    replace(MARGIN4,name='categorical05_margin4',draw='categorical',draw_scale=.5),
    replace(MARGIN4,name='halton_margin4',order='halton'),
    replace(MARGIN4,name='halton_fixed_margin4',order='halton_fixed'),
    replace(MARGIN4,name='raster_margin4',order='raster'),
    replace(BASE,name='blind_probability',confidence='logprob',remask_known=True,blind_confidence=True),
    replace(MARGIN4,name='margin4_blend075',blend=.75),
    replace(MARGIN4,name='margin4_blend050',blend=.5),
    replace(MARGIN4,name='margin4_blend025',blend=.25)]


@lru_cache(maxsize=1)
def halton_grid():
    """First-visit Halton order over the original 16x16 grid."""
    def radical(n,base):
        value,scale=0.,1.
        while n:
            n,digit=divmod(n,base);scale/=base;value+=digit*scale
        return value
    seen=[];used=set()
    for n in range(1,10001):
        cell=int(radical(n,2)*16)*16+int(radical(n,3)*16)
        if cell not in used:
            used.add(cell);seen.append(cell)
            if len(seen)==256:return tuple(seen)
    raise RuntimeError('halton order did not cover 16x16 grid')

def spatial_order_ranks(index,valid,z1,labels,order):
    if order=='confidence':return None
    if order not in ('halton','halton_fixed','raster'):
        raise ValueError('unknown spatial order')
    if index.shape!=valid.shape or index.ndim!=2 or valid.dtype!=torch.bool:
        raise ValueError('index/valid shape mismatch')
    if bool(((index[valid]<0)|(index[valid]>=256)).any()):
        raise ValueError('valid grid index outside 16x16 range')
    base=torch.arange(256,dtype=torch.long) if order=='raster' else torch.tensor(halton_grid(),dtype=torch.long)
    rows=[]
    zhash=z1.long().sum(1).detach().cpu()
    lhash=labels.long().detach().cpu()
    for b in range(index.shape[0]):
        seq=base
        if order=='halton':
            seq=base.roll(int((zhash[b]*131+lhash[b]*17+b*29).remainder(256)))
        inverse=torch.empty_like(seq)
        inverse[seq]=torch.arange(256)
        rows.append(inverse)
    ranks=torch.stack(rows).to(index.device).gather(1,index.clamp(0,255))
    return ranks.masked_fill(~valid,256)

def blend_features(base,mixed,alpha):
    if base.shape!=mixed.shape or not 0<=alpha<=1:raise ValueError('invalid latent blend')
    if alpha==0:return base
    if alpha==1:return mixed
    return base+alpha*(mixed-base)

@torch.no_grad()
def generate(core,z1,index,valid,labels,config,trace=None):
    if config.steps<1 or config.draw not in ['annealed','categorical'] or config.schedule not in ['arccos','cosine']:
        raise ValueError('invalid sampling configuration')
    tokens=torch.where(valid,torch.full_like(index,core.mask_token_2d),torch.full_like(index,core.pad_token_2d))
    counts=valid.sum(1)
    order=getattr(config,'order','confidence')
    spatial_ranks=spatial_order_ranks(index,valid,z1,labels,order)
    for step in range(config.steps):
        ratio=(step+1)/config.steps;temperature=config.noise*(1-ratio)
        unknown=(tokens==core.mask_token_2d)&valid
        cond=core.forward_2d(z1,tokens,index,valid,labels,torch.zeros_like(labels))
        scale=1.+(config.cfg-1.)*step/max(1,config.steps-1) if config.cfg_ramp else config.cfg
        if scale!=1.:
            uncond=core.forward_2d(z1,tokens,index,valid,labels,torch.ones_like(labels))
            logits=core._cfg_logits(cond,uncond,scale,'standard')
        else:logits=cond
        uniform=torch.rand_like(logits).clamp_(1e-6,1.-1e-6)
        gumbel=-torch.log(-torch.log(uniform))
        draw_temp=temperature if config.draw=='annealed' else 1.
        sampled=(logits+draw_temp*getattr(config,'draw_scale',1.)*gumbel).argmax(-1)
        sampled=torch.where(unknown,sampled,tokens)
        safe=sampled.masked_fill(~valid,0)
        score_logits=logits
        if getattr(config,'blind_confidence',False) and step>0:
            # A known token must not score itself while visible. This inexpensive
            # all-MASK probe deliberately removes all2D context for BOTH known
            # and unknown positions, so confidence scales stay comparable.
            probe=torch.where(valid,core.mask_token_2d,core.pad_token_2d)
            score_logits=core.forward_2d(z1,probe,index,valid,labels,torch.zeros_like(labels))
            if scale!=1.:
                blind_uncond=core.forward_2d(z1,probe,index,valid,labels,torch.ones_like(labels))
                score_logits=core._cfg_logits(score_logits,blind_uncond,scale,'standard')
        confidence=selected_confidence(score_logits,safe,config.confidence)
        noise=torch.rand_like(confidence).clamp_(1e-6,1.-1e-6)
        confidence=confidence+temperature*getattr(config,'confidence_scale',1.)*(-torch.log(-torch.log(noise)))
        if spatial_ranks is not None:
            # Keep token draw and confidence-noise RNG budget, but choose which
            # positions stay masked from a spatial order. Lowest scores are
            # re-masked, so negative rank reveals low-rank cells first.
            confidence=-spatial_ranks.float()
        confidence=confidence.masked_fill(~(valid if config.remask_known else unknown),float('inf'))
        if step==config.steps-1:
            tokens=sampled
            if trace is not None:trace.append(dict(step=step,remaining=0,previously_known_remasked=0))
            continue
        remain=math.acos(ratio)/(math.pi*.5) if config.schedule=='arccos' else math.cos(ratio*math.pi*.5)
        target=torch.floor(counts.float()*remain).long()
        if not config.remask_known:target=torch.minimum(target,(unknown.sum(1)-1).clamp_min(0))
        remask=_mask_from_counts(valid,target,confidence)
        if trace is not None:trace.append(dict(step=step,remaining=int(remask.sum()),
            previously_known_remasked=int((remask&~unknown&valid).sum())))
        tokens=torch.where(remask,core.mask_token_2d,sampled)
    if bool(((tokens[valid]<0)|(tokens[valid]>=core.llamagen_vocab_size)).any()):
        raise RuntimeError('sampler returned non-content token')
    if bool((tokens[~valid]!=core.pad_token_2d).any()):raise RuntimeError('sampler altered padding')
    return tokens
