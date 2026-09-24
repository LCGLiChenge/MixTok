"""Alternating joint optimization. Hard sampled-image loss trains fusion only."""
from contextlib import contextmanager
import torch
from torch import nn
from h20.training import smoothed_objective,NEW
from h20.base_model import sample_arccos_mask
from h20.model import OfficialSamplingView
from h20_joint.sampling import generate,MARGIN4
from h20_joint.assets import load_vendor

STAT_NAMES=('loss1d','loss2d','masked_nll1d','masked_nll2d','fusion_mse','half_mse','mean_alpha')


@contextmanager
def sampling_mode(generator):
    was=generator.training;generator.eval()
    try:yield
    finally:generator.train(was)


@torch.no_grad()
def online_conditions(generator,provider,labels,seed):
    load_vendor()
    from modeling.maskgit import ImageBert
    device=labels.device
    with torch.random.fork_rng(devices=[device.index]),sampling_mode(generator):
        torch.manual_seed(seed)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            ids=ImageBert.generate(OfficialSamplingView(generator),condition=labels,num_sample_steps=8,
                guidance_scale=4.5,randomize_temperature=9.5,guidance_decay='linear',softmax_temperature_annealing=False)
        bundle=provider.bundle(ids)
        def current_features(query):
            if not torch.equal(query,ids):raise ValueError('wrong generated1D prefix')
            return bundle['base']
        original=generator._feature_provider;generator.set_feature_provider(current_features)
        try:
            torch.manual_seed(seed+1)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                codes=generate(generator,ids,bundle['index'],bundle['valid'],labels,MARGIN4)
        finally:generator.set_feature_provider(original)
        # Construct target only AFTER sampling; it never enters token generation.
        mixed=provider.mixed(bundle['base'],codes,bundle['index'],bundle['valid'])
        target=provider.teacher(ids)
    return dict(bundle,ids=ids.detach(),codes=codes.detach(),mixed=mixed.detach(),target=target.detach())


class TrainingForward(nn.Module):
    def __init__(self,system,provider):
        super().__init__();self.system=system;object.__setattr__(self,'provider',provider)

    def forward(self,batch,joint,fusion=None):
        if batch is None:
            if fusion is None:raise ValueError('one objective is required')
            with torch.autocast(fusion['base'].device.type,enabled=False):
                fused,alpha=self.system.fusion(fusion['base'],fusion['mixed'],fusion['scores'],fusion['ids'],fusion['index'],fusion['valid'])
                image=self.provider.render(fused)
                fmse=(image-fusion['target']).square().mean()
                with torch.no_grad():
                    half=self.provider.render(fusion['base']+.5*(fusion['mixed']-fusion['base']))
                    hmse=(half-fusion['target']).square().mean()
                alpha_mean=alpha.sum()/fusion['valid'].sum()
            # DDP's final synchronized micro includes every parameter. Previous
            # CE gradients remain accumulated; sampled image loss adds zero to
            # the generator, never a straight-through approximation.
            loss=fmse+sum(p.sum()*0 for p in self.system.generator.parameters())
            zero=fmse.detach()*0
            return loss,torch.stack((zero,zero,zero,zero,fmse.detach(),hmse.detach(),alpha_mean.detach()))
        if fusion is not None:raise ValueError('CE and fusion have separate accumulation normalizers')
        generator=self.system.generator;z1=batch['z1d'];z2=batch['z2d'];valid=batch['route_valid']
        valid1=torch.ones_like(z1,dtype=torch.bool);mask1,_=sample_arccos_mask(valid1);mask2,_=sample_arccos_mask(valid)
        features=self.provider.features(z1)
        logits1=generator.forward_1d(torch.where(mask1,generator.mask_token_1d,z1),batch['label'])
        tokens=torch.where(valid,torch.where(mask2,generator.mask_token_2d,z2),generator.pad_token_2d)
        logits2=generator.forward_2d(z1,tokens,batch['route_indices'],valid,batch['label'],base_features=features)
        loss1,nll1=smoothed_objective(logits1,z1,mask1,valid1)
        loss2,nll2=smoothed_objective(logits2,z2,mask2,valid)
        # No alpha-weighting: poor2D predictions cannot escape CE via alpha=0.
        loss=(1.5 if joint else 0.)*loss1+loss2
        zero=loss.detach()*0
        # Keep DDP parameter participation identical on accumulation micros.
        loss=loss+sum(p.sum()*0 for p in self.system.fusion.parameters())
        return loss,torch.stack((loss1.detach(),loss2.detach(),nll1,nll2,zero,zero,zero))


def groups(system):
    fresh=[];pretrained=[]
    for name,p in system.generator.named_parameters():
        (fresh if name.startswith(NEW+('spatial_projection.','enhancement.')) else pretrained).append(p)
    return fresh,pretrained,list(system.fusion.parameters())


def optimizer_for(system):
    fresh,_,fusion=groups(system)
    return torch.optim.AdamW([dict(params=fresh,lr=1e-4,name='new2d'),dict(params=fusion,lr=3e-3,name='fusion',weight_decay=0.,betas=(.9,.999))],
                             betas=(.9,.96),weight_decay=.03)


def enter_joint(optimizer,system):
    if len(optimizer.param_groups)!=2:raise ValueError('joint phase entered twice')
    optimizer.add_param_group(dict(params=groups(system)[1],lr=1e-5,name='pretrained'))


def set_lrs(optimizer,step,warmup):
    for group in optimizer.param_groups:
        if group['name']=='new2d':group['lr']=1e-4*min(1.,step/50)
        elif group['name']=='fusion':group['lr']=3e-3*min(1.,max(0,step-warmup)/50)
        else:group['lr']=1e-5*min(1.,(step-warmup)/100)
