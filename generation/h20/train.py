"""Portable formal training; same objectives as the audited TiTok-BERT run."""
import argparse
from contextlib import nullcontext
from copy import deepcopy
from datetime import timedelta
import json
import os
from pathlib import Path
import signal
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from h20.assets import official_model,digest
from h20.checkpoint import metadata,restore,stream_origin
from h20.data import E117SparseCodeDataset,collate_e117_sparse
from h20.training import (TrainingForward,RankBucketSampler,optimizer_for,enter_joint,set_lrs,
    update_ema,evaluate,save_latest,atomic_json,STAT_NAMES,SOURCE_N,fingerprint)

def main(args):
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
    if args.global_batch % (args.micro*world): raise ValueError('global batch must divide micro * world exactly')
    accumulation=args.global_batch//(args.micro*world)
    if accumulation<1: raise ValueError('microbatch exceeds global batch')
    torch.cuda.set_device(local);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    dist.init_process_group('nccl',timeout=timedelta(minutes=15),device_id=torch.device('cuda',local))
    run=None;step=0;stop=[False];began=time.monotonic()
    for sig in [signal.SIGINT,signal.SIGTERM]: signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    output=args.output.resolve();root=args.assets.resolve()
    try:
        meta=metadata(args.resume) if args.resume else None
        if meta and meta['config']['global_batch']!=args.global_batch:
            raise ValueError('resume must keep the original global batch')
        if meta and meta['config']['warmup']!=args.warmup:
            raise ValueError('resume must keep the original warmup')
        if meta:
            for key,expected in dict(ema_decay=.999,generated_prefix=False,lr_new=1e-4,
                    lr_pretrained=1e-5,loss1d_weight=1.5,loss_normalization='per_sample',precision='bf16').items():
                if meta['config'].get(key)!=expected:raise ValueError('incompatible resume objective: '+key)
        step=int(meta['step']) if meta else 0;start_step=step
        if args.steps<=step: raise ValueError('--steps is the total target, and must exceed restored step')
        config=dict(seed=0,micro=args.micro,world=world,accumulation=accumulation,global_batch=args.global_batch,
            steps=args.steps,warmup=args.warmup,eval_n=args.eval_n,eval_every=args.eval_every,
            precision='bf16',ema_decay=.999,generated_prefix=False,lr_new=1e-4,lr_pretrained=1e-5,
            loss1d_weight=1.5,loss_normalization='per_sample',mode='formal')
        origin=(0,0,False) if meta is None else stream_origin(meta,micro=args.micro,world=world,accumulation=accumulation)
        native=bool(meta and 'handoff_version' in meta and origin[2])
        config['resume_replay']='native_same_layout_rng' if native else 'cross_layout_next_packed_pass' if meta and not origin[2] else 'legacy_same_layout_rng' if meta else 'official_initialization'
        if rank==0:
            same_output=args.resume and output==args.resume.resolve()
            if output.exists() and not same_output and not args.prepared_output:
                raise FileExistsError('fresh output required, or explicitly resume its own checkpoint')
            output.mkdir(parents=True,exist_ok=True)
            old_config=output/'config.json'
            if same_output and old_config.exists():
                previous=json.loads(old_config.read_text())
                for key in ['micro','world','accumulation','global_batch','warmup','eval_n']:
                    if previous[key]!=config[key]: raise ValueError('changed layout/config needs a new output: '+key)
            atomic_json(old_config,config)
            import wandb
            prior=json.loads((output/'wandb_run.json').read_text()) if same_output and (output/'wandb_run.json').exists() else None
            run=wandb.init(project=os.environ.get('WANDB_PROJECT','mot-maskgit-review'),
                entity=os.environ.get('WANDB_ENTITY') or None,group='titok-bert-h20-20260910',name=output.name,
                id=prior['id'] if prior else None,resume='must' if prior else None,mode='online',dir=str(output),config=None if prior else config,
                settings=wandb.Settings(disable_code=True,disable_git=True,save_code=False,console='off'))
            if prior:run.config.update(config,allow_val_change=True)
            run.define_metric('optimizer_step')
            for prefix in ['train/*','dev/*','system/*']:run.define_metric(prefix,step_metric='optimizer_step')
            atomic_json(output/'wandb_run.json',dict(id=run.id,url=run.url))
            sources={str(p.relative_to(Path(__file__).resolve().parents[1])):digest(p)
                     for p in Path(__file__).resolve().parent.glob('*.py')}
            atomic_json(output/'manifest.json',dict(config=config,sources=sources,
                restored_checkpoint_sha256=meta['sha256'] if meta else None,
                resumed_step=step,stream_origin=origin,cuda=torch.version.cuda,torch=torch.__version__))
        dist.barrier();torch.manual_seed(0)
        core,init_report=official_model(root);core=core.cuda().train()
        optimizer=optimizer_for(core)
        ema=deepcopy(core).eval().requires_grad_(False) if rank==0 else None
        ddp=DDP(TrainingForward(core),device_ids=[local],broadcast_buffers=False,gradient_as_bucket_view=True)
        train=E117SparseCodeDataset(root/'codes/train',root/'routes/train',split='all')
        sampler=RankBucketSampler(train,args.micro,world,rank,0)
        loader=DataLoader(train,batch_sampler=sampler,collate_fn=collate_e117_sparse,
            num_workers=args.workers,pin_memory=True,generator=torch.Generator().manual_seed(90000+rank),
            **({'persistent_workers':True,'prefetch_factor':2} if args.workers else {}))
        if len(loader)<1:raise ValueError('no full training microbatches')
        def stream():
            epoch,skip,_=origin
            while True:
                sampler.set_epoch(epoch)
                for offset,batch in enumerate(loader):
                    if offset<skip:continue
                    yield batch,dict(packed_pass=epoch,next_microbatch_offset=offset+1)
                epoch+=1;skip=0
        batches=stream()
        torch.manual_seed(10000+rank)
        if meta:
            restore(args.resume,core,ema,optimizer,meta,rank=rank,world=world,exact_rng=origin[2])
        elif args.warmup==0:
            enter_joint(optimizer,core)
        selected=np.random.default_rng(20260922).choice(50000,args.eval_n,replace=False)
        val=E117SparseCodeDataset(root/'codes/val',root/'routes/val',split='all') if rank==0 else None
        history=[]
        if rank==0:
            old=output/'development.json'
            if old.exists():history=json.loads(old.read_text())
            atomic_json(output/'dev_cohort.json',dict(source_indices=selected.tolist(),mask_seed=20260922,
                split='ImageNet validation development',augmentation=0))
            atomic_json(output/'initialization.json',init_report)
        def development():
            if rank==0:
                row=dict(step=step,raw=evaluate(core,val,selected),ema=evaluate(ema,val,selected))
                history.append(row);atomic_json(output/'development.json',history)
                run.log(dict(optimizer_step=step,**{f'dev/{state}_{key}':v
                    for state in ['raw','ema'] for key,v in row[state].items()}))
                print(json.dumps({'stage':'development',**row}),flush=True)
            dist.barrier()
        development()
        cursor=None;previous_epoch=int(step*args.global_batch/SOURCE_N);seen=set();checkpoint=None
        for step in range(start_step+1,args.steps+1):
            tick=time.monotonic();joint=step>args.warmup
            if joint and len(optimizer.param_groups)==1:enter_joint(optimizer,core)
            set_lrs(optimizer,step,args.warmup);core.train();ddp.zero_grad(set_to_none=True)
            stats=torch.zeros(len(STAT_NAMES),device='cuda')
            for micro in range(accumulation):
                cpu,cursor=next(batches)
                if not cpu['route_valid'].all() or cpu['k'].unique().numel()!=1:raise ValueError('mixed K bucket')
                if args.smoke:
                    pairs=list(zip(cpu['source_index'].tolist(),cpu['augmentation'].tolist()))
                    all_pairs=[None]*world;dist.all_gather_object(all_pairs,pairs)
                    flat=[p for group in all_pairs for p in group]
                    if len(set(flat))!=len(flat) or seen.intersection(flat):raise ValueError('duplicate DDP samples')
                    seen.update(flat)
                batch={k:v.cuda(non_blocking=True) for k,v in cpu.items()}
                with ddp.no_sync() if micro+1<accumulation else nullcontext():
                    with torch.autocast('cuda',dtype=torch.bfloat16):loss,metrics=ddp(batch,joint)
                    if not torch.isfinite(loss):raise ValueError('nonfinite loss')
                    (loss/accumulation).backward()
                stats+=metrics/accumulation
            active=[p for group in optimizer.param_groups for p in group['params']]
            norm=torch.nn.utils.clip_grad_norm_(active,1.,error_if_nonfinite=True)
            if any(p.grad is None for p in core.parameters()):raise ValueError('missing gradient')
            all_norm=torch.stack([p.grad.float().norm() for p in core.parameters()]).norm()
            if not torch.isfinite(all_norm):raise ValueError('nonfinite gradient')
            optimizer.step()
            if rank==0:update_ema(ema,core)
            dist.all_reduce(stats);stats/=world
            if rank==0 and (step%10==0 or args.smoke):
                row=dict(step=step,phase='joint' if joint else 'new2d_warmup',
                    **{k:float(v) for k,v in zip(STAT_NAMES,stats)},
                    source_equivalent_epochs=step*args.global_batch/SOURCE_N,
                    grad_norm=float(norm),seconds_per_step=time.monotonic()-tick,
                    peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)
                atomic_json(output/'status.json',dict(status='running',**row));print(json.dumps(row),flush=True)
                run.log(dict(optimizer_step=step,**{f'train/{k}':row[k] for k in STAT_NAMES},
                    **{'train/source_equivalent_epochs':row['source_equivalent_epochs'],
                       'train/grad_norm':row['grad_norm'],'train/lr_new':optimizer.param_groups[0]['lr'],
                       'train/lr_pretrained':optimizer.param_groups[1]['lr'] if joint else 0.,
                       'system/seconds_per_step':row['seconds_per_step'],'system/peak_reserved_gib':row['peak_reserved_gib']}))
            ddp.zero_grad(set_to_none=True)
            if step%args.eval_every==0 or step==args.steps:development()
            stop_tensor=torch.tensor(int(stop[0]),device='cuda');dist.all_reduce(stop_tensor,op=dist.ReduceOp.MAX)
            stopped=bool(stop_tensor.item());source_epoch=int(step*args.global_batch/SOURCE_N)
            if source_epoch>previous_epoch or step==args.steps or stopped:
                rng=[None]*world;dist.all_gather_object(rng,dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state()))
                if rank==0:
                    checkpoint=save_latest(output,core,ema,optimizer,rng,dict(step=step,config=config,cursor=cursor,
                        handoff_version=1,stop_reason='signal' if stopped else None,
                        source_equivalent_epochs=step*args.global_batch/SOURCE_N))
                dist.barrier();previous_epoch=source_epoch
            if stopped:break
        if rank==0:
            summary=dict(status='stopped_safely' if stop[0] else 'complete',step=step,start_step=start_step,
                checkpoint=checkpoint,config=config,elapsed_seconds=time.monotonic()-began,
                disjoint_smoke_pairs=len(seen),parameter_sha256=fingerprint(core))
            atomic_json(output/'summary.json',summary);run.summary['outcome']=summary['status'];run.finish()
        dist.barrier()
    except BaseException as exc:
        if output.exists():atomic_json(output/f'failure_rank{rank}.json',dict(error=str(exc),type=type(exc).__name__,step=step))
        if run is not None:run.finish(exit_code=1)
        raise
    finally:
        dist.destroy_process_group()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',type=Path);p.add_argument('--micro',type=int,required=True)
    p.add_argument('--global-batch',type=int,default=2048);p.add_argument('--steps',type=int,default=50000)
    p.add_argument('--warmup',type=int,default=600);p.add_argument('--eval-every',type=int,default=100)
    p.add_argument('--eval-n',type=int,default=1024);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--smoke',action='store_true');p.add_argument('--prepared-output',action='store_true')
    args=p.parse_args()
    if not 0<=args.warmup<args.steps or args.micro<1 or args.eval_every<1 or not 1<=args.eval_n<=50000:
        p.error('invalid training budget or batch configuration')
    main(args)
