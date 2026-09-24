"""Low-memory official-weight equivalence and gradient-boundary smoke."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch
from torch import nn
from bert2d.paths import atomic_json
from experiments.feature_to_token_20260916.halton_proxy import HaltonProxy,load_pretrained
from experiments.feature_to_token_20260916.halton_mixed_completion import MixedHaltonCompletionObjective
from experiments.continuous_halton_embedding_20260923.continuous_halton import ContinuousHalton,ContinuousObjective,sample_continuous_completion
from experiments.continuous_halton_embedding_20260923.embedding_adapter import EmbeddingAdapter,official_embedding,regression_loss


class LegacyView(nn.Module):
    def __init__(self,core):
        super().__init__()
        self.core=core
    def forward(self,*args):
        return HaltonProxy.forward(self.core,*args)


def run(args):
    torch.set_num_threads(4)
    device=torch.device(args.device)
    if device.type=='cuda':
        torch.cuda.set_per_process_memory_fraction(.06,device)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(2409)
    core=ContinuousHalton(args.upstream_root)
    init=load_pretrained(core,args.pretrained,True)
    core=core.to(device).eval()
    ids=torch.randint(0,16384,(1,256),device=device)
    labels=torch.tensor([7],device=device)
    indices=torch.arange(0,256,3,device=device)[None]
    selected=torch.zeros_like(ids,dtype=torch.bool).scatter(1,indices,True)
    drop=torch.tensor([False],device=device)
    with torch.no_grad():
        context=core.model.tok_emb(ids).detach()
        baseline=HaltonProxy.forward(core,ids,labels,drop)
        actual=core(context,ids,selected,labels,drop)
        torch.testing.assert_close(actual,baseline,rtol=1e-5,atol=1e-5)
        diff=float((actual-baseline).abs().max())
        # Same sparse training masks and dropout RNG, same model parameters.
        old_objective=MixedHaltonCompletionObjective(LegacyView(core),args.upstream_root)
        new_objective=ContinuousObjective(core,args.upstream_root)
        valid=torch.ones_like(indices,dtype=torch.bool)
        gt=ids.gather(1,indices)
        torch.manual_seed(42)
        old_loss,old_stats=old_objective(ids,gt,labels,indices,valid)
        torch.manual_seed(42)
        new_loss,new_stats=new_objective(context,gt,labels,indices,valid)
        torch.testing.assert_close(old_loss,new_loss,rtol=1e-5,atol=1e-5)
        torch.testing.assert_close(old_stats,new_stats,rtol=1e-5,atol=1e-5)
    del baseline,actual
    core.train()
    loss,_=new_objective(context,gt,labels,indices,valid)
    loss.backward()
    if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in core.parameters()):
        raise RuntimeError('missing/nonfinite Halton gradients')
    if context.requires_grad or context.grad is not None:
        raise RuntimeError('context received gradients')
    core.zero_grad(set_to_none=True)
    core.eval()
    trace=[]
    with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
        completed=sample_continuous_completion(core,context,selected,labels,args.upstream_root,steps=4,trace=trace)
    if bool((completed[selected]>=16384).any()):
        raise RuntimeError('bad generated ID')
    table,variance=official_embedding(args.pretrained)
    adapter=EmbeddingAdapter().to(device)
    features=torch.randn(1,256,16,16,device=device)
    a_loss=regression_loss(adapter(features),ids,table.to(device),variance.to(device))
    a_loss.backward()
    if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in adapter.parameters()):
        raise RuntimeError('invalid adapter gradient')
    atomic_json(args.output,dict(status='passed',logits_max_abs_diff=diff,
        objective_loss=float(new_loss),adapter_loss=float(a_loss),initialization=init,
        adapter_parameters=sum(p.numel() for p in adapter.parameters()),
        context_gradients=False,sampling_trace=trace,
        peak_reserved_gib=torch.cuda.max_memory_reserved(device)/1024**3 if device.type=='cuda' else None))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--upstream-root',type=Path,required=True)
    p.add_argument('--pretrained',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    run(p.parse_args())
