"""Isolated worst-case K128 memory probe; never writes a checkpoint."""
import argparse,json,time,math
from pathlib import Path
import torch
from .model import BertSparse2D, load_titok_backbone
from .features import PortableBaseFeatures
from .train import TrainingForward
from .paths import ASSETS,atomic_json
from .optimization import add_arguments, from_args, make_optimizer, set_learning_rates
def main(a):
    recipe=from_args(a)
    torch.set_num_threads(4);torch.manual_seed(0)
    try:
        core=BertSparse2D()
        audit=load_titok_backbone(core,a.assets_root/"weights/generator_titok_l32.bin")
        core=core.cuda().train()
        if a.recompute_layers:core.enable_recompute(a.recompute_layers)
        core.set_feature_provider(PortableBaseFeatures("cuda",a.feature_chunk,assets_root=a.assets_root))
        f=TrainingForward(core);opt=make_optimizer(core,audit,recipe)
        b=a.micro
        batch=dict(z1d=torch.randint(4096,(b,32),device="cuda"),z2d=torch.randint(16384,(b,128),device="cuda"),
            route_indices=torch.arange(128,device="cuda")[None].expand(b,-1),
            route_valid=torch.ones(b,128,dtype=torch.bool,device="cuda"),label=torch.arange(b,device="cuda")%1000)
        for i in range(3):
            # Post-warmup stress test; same groups/hyperparameters as formal training.
            set_learning_rates(opt,recipe,max(1,math.ceil(120/recipe["kappa"]))+i)
            batch["z1d"]=torch.randint(4096,(b,32),device="cuda")
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16):loss,_=f(batch)
            if not torch.isfinite(loss):raise RuntimeError("nonfinite capacity loss")
            loss.backward();nnorm=torch.nn.utils.clip_grad_norm_(core.parameters(),1.,error_if_nonfinite=True)
            opt.step();torch.cuda.synchronize()
        peak=torch.cuda.max_memory_reserved()/1024**2
        row=dict(status="ok" if peak+1536<=a.limit_mib else "over_budget",
                 optimization=recipe,micro=b,peak_reserved_mib=peak,ddp_context_margin_mib=1536,limit_mib=a.limit_mib)
        atomic_json(a.output,row);print(json.dumps(row))
    except torch.OutOfMemoryError:
        atomic_json(a.output,dict(status="oom",micro=a.micro));raise SystemExit(2)
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--micro",type=int,required=True)
    p.add_argument("--global-batch",type=int,default=448)
    add_arguments(p)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--assets-root",type=Path,default=ASSETS)
    p.add_argument("--recompute-layers",type=int,default=10)
    p.add_argument("--feature-chunk",type=int,default=8)
    p.add_argument("--limit-mib",type=float,required=True)
    main(p.parse_args())
