"""Actual K128 two-stage forward/backward/AdamW/EMA memory probe."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time
import torch
from h20.assets import official_model
from h20.training import TrainingForward,optimizer_for,enter_joint,update_ema

def run(args):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_device(0)
    core,_=official_model(args.assets); core=core.cuda().train()
    ema=deepcopy(core).eval().requires_grad_(False)
    model=TrainingForward(core); optimizer=optimizer_for(core); enter_joint(optimizer,core)
    b=args.micro
    batch=dict(z1d=torch.randint(4096,(b,32),device='cuda'),z2d=torch.randint(16384,(b,128),device='cuda'),
        route_indices=torch.arange(128,device='cuda').expand(b,-1),route_valid=torch.ones(b,128,dtype=torch.bool,device='cuda'),
        label=torch.randint(1000,(b,),device='cuda'))
    torch.cuda.synchronize(); started=time.monotonic()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16): loss,_=model(batch,True)
        loss.backward(); torch.nn.utils.clip_grad_norm_(core.parameters(),1.,error_if_nonfinite=True)
        optimizer.step(); update_ema(ema,core)
    torch.cuda.synchronize()
    peak=torch.cuda.max_memory_reserved(); total=torch.cuda.get_device_properties(0).total_memory
    result=dict(micro=b,peak_reserved_gib=peak/1024**3,total_gib=total/1024**3,
                fraction=peak/total,seconds_per_step=(time.monotonic()-started)/3,
                status='pass' if peak/total<=args.memory_fraction else 'over_margin')
    args.output.write_text(json.dumps(result,indent=2)); print(json.dumps(result),flush=True)
    return 0 if result['status']=='pass' else 3

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',type=Path,required=True);p.add_argument('--micro',type=int,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--memory-fraction',type=float,default=.92)
    args=p.parse_args()
    try: code=run(args)
    except torch.cuda.OutOfMemoryError:
        args.output.write_text(json.dumps({'status':'oom','micro':args.micro})); code=2
    raise SystemExit(code)
