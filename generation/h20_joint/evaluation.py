"""Paired image-MSE diagnostics, explicitly NOT FID or a held-out quality claim."""
import torch
from h20_joint.objective import online_conditions


@torch.no_grad()
def images(system,provider,packet):
    fused,alpha=system.fusion(packet['base'],packet['mixed'],packet['scores'],packet['ids'],packet['index'],packet['valid'])
    return dict(pure1d=provider.render(packet['base']),full2d=provider.render(packet['mixed']),
        half=provider.render(packet['base']+.5*(packet['mixed']-packet['base'])),
        learned=provider.render(fused),native1d_teacher=packet['target']),alpha


@torch.no_grad()
def development(system,provider):
    was=system.training;system.eval();result={}
    try:
        # Diverse fixed classes/seed, disjoint from the training seed domain.
        for group,values in enumerate(((9,281),(530,963))):
            labels=torch.tensor(values,device=next(system.parameters()).device)
            packet=online_conditions(system.generator,provider,labels,82000000+group*100)
            rendered,alpha=images(system,provider,packet)
            for name,image in rendered.items():
                if name=='native1d_teacher':continue
                value=float((image-packet['target']).square().mean())
                result[f'online_{name}_teacher_mse']=result.get(f'online_{name}_teacher_mse',0)+value/2
            result['online_mean_alpha']=result.get('online_mean_alpha',0)+float(alpha.sum()/packet['valid'].sum())/2
    finally:system.train(was)
    return result
