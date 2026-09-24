"""Preflight, isolated capacity probes, then an explicit torchrun invocation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from h20.preflight import environment,devices,data
from h20.assets import verify
from h20.checkpoint import metadata

def candidates(global_batch,world,maximum=None):
    return [n for n in range(global_batch//world,0,-1)
            if global_batch%(n*world)==0 and (maximum is None or n<=maximum)]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--init',choices=['resume','official'],default='resume')
    p.add_argument('--resume',type=Path);p.add_argument('--steps',type=int,default=50000)
    p.add_argument('--max-micro',type=int);p.add_argument('--memory-fraction',type=float,default=.92)
    p.add_argument('--allow-non-h20',action='store_true');p.add_argument('--smoke',action='store_true')
    p.add_argument('--workers',type=int,default=2);p.add_argument('--plan-only',action='store_true')
    args=p.parse_args()
    if not .5<=args.memory_fraction<=.95:p.error('memory fraction must be between .5 and .95')
    env=environment();gpus=devices(args.allow_non_h20);world=len(gpus)
    root=args.assets.resolve();output=args.output.resolve()
    verify(root,('train',));dataset=data(root)
    resume=args.resume.resolve() if args.resume else output if (output/'latest.json').exists() else root/'resume' if args.init=='resume' else None
    if args.smoke:
        if args.resume:p.error('smoke is fresh only; use h20.train for a targeted resume smoke')
        resume=None
    meta=metadata(resume) if resume else None
    if output.exists() and output!=resume:raise FileExistsError('new output required')
    if resume and resume==root/'resume':verify(root,('resume',))
    global_batch=world*min(args.max_micro or 2,2) if args.smoke else 2048
    steps=4 if args.smoke else args.steps;warmup=2 if args.smoke else 600
    if meta and steps<=meta['step']:raise ValueError('target steps already reached')
    plan=dict(world=world,gpus=gpus,environment=env,dataset=dataset,global_batch=global_batch,
        total_steps=steps,warmup=warmup,resume=str(resume) if resume else None,
        initial_step=meta['step'] if meta else 0,candidates=candidates(global_batch,world,args.max_micro),
        target_examples=steps*global_batch,source_equivalent_epochs=steps*global_batch/1281167,
        memory_fraction=args.memory_fraction)
    print(json.dumps(plan,indent=2),flush=True)
    if args.plan_only:return
    import wandb
    if os.environ.get('WANDB_MODE','online')!='online':raise ValueError('W&B online required')
    if not wandb.login():raise RuntimeError('run wandb login before launch')
    output.mkdir(parents=True,exist_ok=True)
    probe_root=output/'capacity';probe_root.mkdir(exist_ok=True)
    probe_gpu=min(gpus,key=lambda gpu:gpu['total_gib'])
    plan['probe_gpu']=probe_gpu
    environment_variables=dict(os.environ,CUDA_VISIBLE_DEVICES=str(probe_gpu['physical']))
    selected=None
    for micro in plan['candidates']:
        report=probe_root/f'micro{micro}.json'
        if report.exists():report=probe_root/f'micro{micro}-retry{len(list(probe_root.iterdir()))}.json'
        command=[sys.executable,'-m','h20.probe','--assets',str(root),'--micro',str(micro),
                 '--output',str(report),'--memory-fraction',str(args.memory_fraction)]
        result=subprocess.run(command,env=environment_variables)
        if result.returncode not in [0,2,3]:raise RuntimeError('probe failed for a reason other than OOM/margin')
        record=json.loads(report.read_text())
        if result.returncode==0 and record['status']=='pass':selected=micro;break
    if selected is None:raise RuntimeError('no safe batch size; no training started')
    plan.update(micro=selected,accumulation=global_batch//(selected*world),probe=str(report))
    plan_path=output/'run_plan.json'
    if plan_path.exists():plan_path=output/f'resume_plan_step{plan["initial_step"]}.json'
    plan_path.write_text(json.dumps(plan,indent=2))
    command=[sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={world}',
        '-m','h20.train','--assets',str(root),'--output',str(output),'--micro',str(selected),
        '--global-batch',str(global_batch),'--steps',str(steps),'--warmup',str(warmup),
        '--workers',str(args.workers),'--prepared-output']
    if resume:command+=['--resume',str(resume)]
    if args.smoke:command+=['--smoke','--eval-n','16','--eval-every','2']
    print('Launch: '+' '.join(command),flush=True)
    # Replace the launcher so SIGINT/SIGTERM reaches the owned torchrun process.
    os.execv(sys.executable,command)

if __name__=='__main__':main()
