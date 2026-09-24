"""Original four-step margin sampler; token/RNG parity must be verified."""
from dataclasses import dataclass,replace
import math
import torch
from h20.base_model import _mask_from_counts

@dataclass(frozen=True)
class Sampling:
    name: str='baseline'
    steps: int=8
    cfg: float=4.5
    confidence: str='logit'
    draw: str='annealed'
    noise: float=1.
    remask_known: bool=False
    schedule: str='arccos'
    cfg_ramp: bool=False


def selected_confidence(logits,selected,mode):
    chosen=logits.gather(-1,selected[...,None]).squeeze(-1).float()
    if mode=='logit':return chosen
    if mode=='logprob':return chosen-torch.logsumexp(logits.float(),-1)
    if mode=='margin':
        top=logits.topk(2,-1)
        competitor=torch.where(top.indices[...,0]==selected,top.values[...,1],top.values[...,0])
        return chosen-competitor
    raise ValueError('unknown confidence definition')

@dataclass(frozen=True)
class Followup(Sampling):
    draw_scale: float=1.
    confidence_scale: float=1.
    blind_confidence: bool=False
    blend: float=1.


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


MARGIN4=replace(Followup(),name="margin4",confidence="margin",steps=4)
