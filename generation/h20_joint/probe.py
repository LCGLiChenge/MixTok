"""Real joint-path capacity test: frozen assets, online generation, CE, image backward, Adam, EMA."""
import argparse
from copy import deepcopy
from pathlib import Path
import time
import torch
from h20.training import atomic_json,update_ema
from h20_joint.assets import FrozenAssets
from h20_joint.model import create
from h20_joint.objective import TrainingForward,optimizer_for,enter_joint,online_conditions


def run(args):
    torch.set_num_threads(4);torch.cuda.set_device(0);torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    system,initialization=create(args.assets,args.memory);system=system.cuda().train()
    ema=deepcopy(system).eval().requires_grad_(False)
    provider=FrozenAssets(args.assets,torch.device('cuda',0),chunk=4)
    system.generator.set_feature_provider(provider.features);ema.generator.set_feature_provider(provider.features)
    wrapper=TrainingForward(system,provider);optimizer=optimizer_for(system);enter_joint(optimizer,system)
    b=args.micro
    batch=dict(z1d=torch.randint(4096,(b,32),device='cuda'),z2d=torch.randint(16384,(b,128),device='cuda'),
        route_indices=torch.arange(128,device='cuda').expand(b,-1),route_valid=torch.ones(b,128,dtype=torch.bool,device='cuda'),
        label=torch.arange(b,device='cuda')%1000)
    tick=time.monotonic();rows=[]
    for step in range(args.updates):
        wrapper.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss,ce=wrapper(batch,True)
        loss.backward()
        labels=torch.arange(args.fusion_micro,device='cuda')%1000
        packet=online_conditions(system.generator,provider,labels,910000+step*100)
        image_loss,im=wrapper(None,True,packet);image_loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in system.parameters()):
            raise ValueError('missing/nonfinite joint gradient')
        if not system.generator.spatial_projection.weight.grad.abs().sum()>0:raise ValueError('spatial input receives no gradient')
        if not system.fusion.output.weight.grad.abs().sum()>0:raise ValueError('fusion receives no gradient')
        provider.assert_frozen()
        for group in optimizer.param_groups:torch.nn.utils.clip_grad_norm_(group['params'],1.,error_if_nonfinite=True)
        optimizer.step();update_ema(ema,system)
        rows.append(dict(step=step+1,ce=float(loss.detach()),image_mse=float(image_loss.detach()),half_mse=float(im[5]),mean_alpha=float(im[6])))
        del packet
    # Fixed development uses up to32 examples; include its separate peak even
    # when the selected training micro is smaller than32.
    wrapper.zero_grad(set_to_none=True);system.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        z1=batch['z1d'][:1].expand(32,-1);z2=batch['z2d'][:1].expand(32,-1)
        index=batch['route_indices'][:1].expand(32,-1);valid=batch['route_valid'][:1].expand(32,-1)
        for model in (system.generator,ema.generator):
            out=model.forward_2d(z1,z2,index,valid,torch.arange(32,device='cuda'))
            if not torch.isfinite(out).all():raise ValueError('nonfinite development forward')
    torch.cuda.synchronize();peak=torch.cuda.max_memory_reserved();total=torch.cuda.get_device_properties(0).total_memory
    result=dict(status='pass' if peak/total<=args.memory_fraction else 'over_margin',memory=args.memory,
        micro=b,fusion_micro=args.fusion_micro,peak_reserved_gib=peak/1024**3,total_gib=total/1024**3,
        fraction=peak/total,seconds=time.monotonic()-tick,updates=rows,initialization=initialization,frozen=provider.audit,
        full_checkpoint_tested=False,hardware=torch.cuda.get_device_name(0))
    atomic_json(args.output,result);print(result,flush=True)
    return 0 if result['status']=='pass' else 3


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--micro',type=int,required=True);p.add_argument('--fusion-micro',type=int,default=2)
    p.add_argument('--memory',choices=('local','full'),default='local')
    p.add_argument('--memory-fraction',type=float,default=.92);p.add_argument('--updates',type=int,default=3)
    args=p.parse_args()
    if min(args.micro,args.fusion_micro,args.updates)<1:p.error('positive budgets required')
    try:code=run(args)
    except torch.cuda.OutOfMemoryError:
        atomic_json(args.output,dict(status='oom',micro=args.micro,fusion_micro=args.fusion_micro));code=2
    raise SystemExit(code)
