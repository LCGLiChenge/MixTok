"""H20 delivery: capacity probe -> fresh DDP smoke -> full resume smoke -> official fresh formal run."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from h20.assets import verify
from h20.preflight import environment,devices,data
from h20.launch import candidates
from h20.training import atomic_json,SOURCE_N
from h20_joint.checkpoint import metadata
from h20_joint.train import source_identity


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=80);p.add_argument('--resume',type=Path)
    p.add_argument('--memory',choices=('local','full'),default='local')
    p.add_argument('--max-micro',type=int);p.add_argument('--fusion-micro',type=int,default=2)
    p.add_argument('--memory-fraction',type=float,default=.92)
    p.add_argument('--allow-non-h20',action='store_true');p.add_argument('--smoke-only',action='store_true')
    p.add_argument('--plan-only',action='store_true');p.add_argument('--workers',type=int,default=2)
    args=p.parse_args()
    if args.epochs<1 or not .5<=args.memory_fraction<=.95:p.error('invalid epoch budget/memory fraction')
    env=environment();gpus=devices(args.allow_non_h20);world=len(gpus)
    if args.fusion_micro<1 or 16%(args.fusion_micro*world):p.error('fusion micro * world must divide16')
    root=args.assets.resolve();output=args.output.resolve()
    verify(root,('train','inference'));dataset=data(root)
    resume=args.resume.resolve() if args.resume else None
    meta=metadata(resume) if resume else None
    if output.exists() and output!=resume:raise FileExistsError('fresh output required; use --resume explicitly')
    if meta and meta['config']['memory']!=args.memory:raise ValueError('resume spatial architecture mismatch')
    if meta and output==resume:
        if meta['config']['world']!=world or meta['config']['fusion_micro']!=args.fusion_micro:
            raise ValueError('changed layout needs a NEW --output and the old --resume')
        if args.max_micro is not None and args.max_micro<meta['config']['micro']:
            raise ValueError('new micro cap excludes saved layout; use a new output')
    steps=math.ceil(args.epochs*SOURCE_N/2048)
    if meta and steps<=meta['step']:raise ValueError('requested total budget already reached')
    cap=output if output.exists() else output.parent
    if not cap.exists():raise FileNotFoundError('create the output parent first: '+str(cap))
    if shutil.disk_usage(cap).free<20*1024**3:raise RuntimeError('need20GiB free for smoke and atomic full checkpoints')
    plan=dict(world=world,gpus=gpus,environment=env,dataset=dataset,memory=args.memory,global_batch=2048,
        fusion_global_batch=16,fusion_micro=args.fusion_micro,epochs=args.epochs,total_steps=steps,
        source_equivalent_epochs=steps*2048/SOURCE_N,memory_fraction=args.memory_fraction,
        resume=str(resume) if resume else None,candidates=candidates(2048,world,args.max_micro))
    if meta and output==resume:plan['candidates']=[meta['config']['micro']]
    print(json.dumps(plan,indent=2),flush=True)
    if args.plan_only:return
    import wandb
    if os.environ.get('WANDB_MODE','online')!='online' or not wandb.login():raise RuntimeError('W&B online login required')
    output.mkdir(parents=True,exist_ok=True)
    capacity=output/('capacity_resume_'+str(meta['step']) if meta else 'capacity')
    capacity.mkdir(exist_ok=False)
    probe_gpu=min(gpus,key=lambda v:v['total_gib'])
    probe_env=dict(os.environ,CUDA_VISIBLE_DEVICES=probe_gpu['physical'])
    selected=None
    for micro in plan['candidates']:
        report=capacity/f'micro{micro}.json'
        command=[sys.executable,'-m','h20_joint.probe','--assets',str(root),'--micro',str(micro),
            '--fusion-micro',str(args.fusion_micro),'--memory',args.memory,'--memory-fraction',str(args.memory_fraction),'--output',str(report)]
        result=subprocess.run(command,env=probe_env)
        if result.returncode not in (0,2,3):raise RuntimeError('capacity test failed for a reason other than memory')
        if result.returncode==0 and json.loads(report.read_text())['status']=='pass':selected=micro;break
    if selected is None:raise RuntimeError('no safe microbatch; formal training not started')
    plan.update(micro=selected,accumulation=2048//(selected*world),probe=report.name)
    atomic_json(capacity/'plan.json',plan)
    launch=[sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={world}','-m','h20_joint.train']
    common=['--assets',str(root),'--memory',args.memory,'--micro',str(selected),
        '--fusion-micro',str(args.fusion_micro),'--global-batch','2048','--workers',str(args.workers)]
    # Independent output: smoke weights are NEVER the formal initialization.
    smoke=capacity/'distributed_smoke'
    smoke_args=common+['--output',str(smoke),'--smoke','--warmup','2','--eval-n','16','--eval-every','2']
    subprocess.run(launch+smoke_args+['--steps','4'],check=True)
    first=metadata(smoke)
    if first['step']!=4 or not first['round_trip_exact']:raise RuntimeError('smoke save failed')
    subprocess.run(launch+smoke_args+['--steps','5','--resume',str(smoke)],check=True)
    second=metadata(smoke);summary=json.loads((smoke/'summary.json').read_text())
    if second['step']!=5 or summary['start_step']!=4 or summary['status']!='complete':raise RuntimeError('resume smoke failed')
    atomic_json(capacity/'verified.json',dict(status='pass',sources=source_identity(),plan=plan,
        fresh_step=4,resumed_step=5,resume_sha256=second['sha256'],formal_initialization='official' if not resume else 'explicit_joint_resume'))
    if args.smoke_only:return
    command=launch+common+['--output',str(output),'--prepared-output','--steps',str(steps)]
    if resume:command+=['--resume',str(resume)]
    print('Formal launch: '+' '.join(command),flush=True)
    os.execv(sys.executable,command)


if __name__=='__main__':main()
