"""Two-rank tiny CPU DDP versus single-process gradients, with CE/fusion accumulation."""
from contextlib import nullcontext
from copy import deepcopy
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from unittest.mock import patch
from h20_joint.test_joint import tiny,packet,Provider
from h20_joint.objective import TrainingForward


def fixed_mask(valid):
    mask=(torch.arange(valid.shape[1],device=valid.device)%2==0)[None].expand_as(valid)&valid
    return mask,mask.float().mean(1)


def main():
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE'])
    if world!=2:raise ValueError('this test requires two CPU ranks')
    torch.set_num_threads(2);dist.init_process_group('gloo')
    try:
        torch.manual_seed(42);system=tiny();reference=deepcopy(system)
        fusion=packet(4)
        batch=dict(z1d=torch.randint(17,(4,32)),z2d=torch.randint(23,(4,128)),
            route_indices=fusion['index'],route_valid=fusion['valid'],label=torch.arange(4))
        ddp=DDP(TrainingForward(system,Provider()),broadcast_buffers=False,gradient_as_bucket_view=True)
        wanted=TrainingForward(reference,Provider())
        with patch('h20_joint.objective.sample_arccos_mask',fixed_mask):
            wanted(batch,True)[0].backward();wanted(None,True,fusion)[0].backward()
            for i in range(2):
                offset=rank*2+i;local={k:v[offset:offset+1] for k,v in batch.items()}
                with ddp.no_sync():(ddp(local,True)[0]/2).backward()
            for i in range(2):
                offset=rank*2+i;local={k:v[offset:offset+1] for k,v in fusion.items()}
                with nullcontext() if i==1 else ddp.no_sync():(ddp(None,True,local)[0]/2).backward()
        maximum=0.
        for (name,a),(_,b) in zip(system.named_parameters(),reference.named_parameters()):
            if a.grad is None or b.grad is None:raise AssertionError('missing gradient: '+name)
            torch.testing.assert_close(a.grad,b.grad,atol=3e-6,rtol=1e-4,msg=lambda msg:name+': '+msg)
            maximum=max(maximum,float((a.grad-b.grad).abs().max()))
        if rank==0:print(dict(status='pass',world=world,CE_micros=2,fusion_micros=2,max_gradient_abs_error=maximum),flush=True)
        dist.barrier()
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
