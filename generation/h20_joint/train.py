"""Full-data spatial MaskGIT + online fusion, two independently averaged losses."""
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
from h20.assets import ROOT,digest
from h20.data import E117SparseCodeDataset,collate_e117_sparse
from h20.training import RankBucketSampler,update_ema,evaluate,atomic_json,SOURCE_N
from h20_joint.assets import FrozenAssets
from h20_joint.model import create
from h20_joint.objective import TrainingForward,optimizer_for,enter_joint,set_lrs,online_conditions,STAT_NAMES
from h20_joint.checkpoint import metadata,restore,stream_origin,save_latest
from h20_joint.evaluation import development as image_development


def source_identity():
    return {str(p.relative_to(ROOT)):digest(p) for folder in ('h20','h20_joint','third_party')
            for p in sorted((ROOT/folder).rglob('*.py'))}


def main(args):
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
    if args.global_batch%(args.micro*world):raise ValueError('CE global batch must divide micro * world')
    if args.fusion_global_batch%(args.fusion_micro*world):raise ValueError('fusion global batch must divide fusion micro * world')
    accumulation=args.global_batch//(args.micro*world)
    fusion_accum=args.fusion_global_batch//(args.fusion_micro*world)
    if min(accumulation,fusion_accum)<1:raise ValueError('microbatch exceeds global batch')
    torch.cuda.set_device(local);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    dist.init_process_group('nccl',timeout=timedelta(minutes=30),device_id=torch.device('cuda',local))
    output=args.output.resolve();root=args.assets.resolve();run=None;step=0;stop=[False]
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    began=time.monotonic()
    try:
        meta=metadata(args.resume) if args.resume else None
        step=int(meta['step']) if meta else 0;start=step
        config=dict(seed=0,memory=args.memory,micro=args.micro,world=world,accumulation=accumulation,
            global_batch=args.global_batch,fusion_global_batch=args.fusion_global_batch,
            fusion_micro=args.fusion_micro,fusion_accumulation=fusion_accum,steps=args.steps,warmup=args.warmup,
            eval_n=args.eval_n,eval_every=args.eval_every,precision='bf16_CE_fp32_decoder',ema_decay=.999,
            lr_new=1e-4,lr_pretrained=1e-5,lr_fusion=3e-3,loss1d_weight=1.5,loss2d_weight=1.,
            image_weight=1.,token_prefix='real_packed',fusion_prefix='online_current_raw_generated',
            fusion_target='frozen_native_TiTok_decode_same_generated_1d',fusion_score_input='zero',
            sampler_1d='upstream8_cfg4.5_linear_temp9.5',sampler_2d='margin4_cfg4.5_constant',
            loss_normalization='CE_and_fusion_global_batches_independent',smoke=args.smoke,in_memory_smoke=args.in_memory_smoke)
        immutable=[k for k in config if k not in ('micro','world','accumulation','fusion_micro',
            'fusion_accumulation','steps','eval_every')]
        if meta:
            for key in immutable:
                if meta['config'].get(key)!=config[key]:raise ValueError('incompatible resume: '+key)
        if args.steps<=step:raise ValueError('--steps is a total target, already reached')
        origin=stream_origin(meta,micro=args.micro,world=world,accumulation=accumulation) if meta else (0,0,False)
        config['resume_replay']='same_layout_rng_cursor' if meta and origin[2] else 'cross_layout_next_packed_pass' if meta else 'official_fresh'
        same_output=bool(args.resume and output==args.resume.resolve())
        if rank==0:
            if output.exists() and not same_output and not args.prepared_output:raise FileExistsError('fresh output required')
            output.mkdir(parents=True,exist_ok=True)
            previous=output/'config.json'
            if previous.exists() and not same_output:raise FileExistsError('will not overwrite another run config')
            if same_output and previous.exists():
                old=json.loads(previous.read_text())
                for key in ('micro','world','accumulation','fusion_micro'):
                    if old[key]!=config[key]:raise ValueError('layout migration requires a new output: '+key)
            atomic_json(previous,config)
            if os.environ.get('WANDB_MODE','online')!='online':raise ValueError('W&B online is required')
            import wandb
            prior=json.loads((output/'wandb_run.json').read_text()) if same_output and (output/'wandb_run.json').exists() else None
            run=wandb.init(project=os.environ.get('WANDB_PROJECT','mot-maskgit-review'),
                entity=os.environ.get('WANDB_ENTITY') or None,group='titok-spatial-online-fusion-v1',name=output.name,
                id=prior['id'] if prior else None,resume='must' if prior else None,mode='online',dir=str(output),
                config=None if prior else config,
                settings=wandb.Settings(disable_code=True,disable_git=True,save_code=False,console='off'))
            if prior:run.config.update(config,allow_val_change=True)
            run.define_metric('optimizer_step')
            for prefix in ('train/*','dev/*','system/*'):run.define_metric(prefix,step_metric='optimizer_step')
            atomic_json(output/'wandb_run.json',dict(id=run.id,url=run.url))
            atomic_json(output/('resume_manifest.json' if meta else 'manifest.json'),dict(config=config,
                sources=source_identity(),restored_sha256=meta['sha256'] if meta else None,
                resumed_step=step,stream_origin=origin,torch=torch.__version__,cuda=torch.version.cuda))
        dist.barrier();torch.manual_seed(0)
        system,init_report=create(root,args.memory);system=system.cuda().train()
        optimizer=optimizer_for(system)
        ema=deepcopy(system).eval().requires_grad_(False) if rank==0 else None
        provider=FrozenAssets(root,torch.device('cuda',local),chunk=4)
        system.generator.set_feature_provider(provider.features)
        if ema:ema.generator.set_feature_provider(provider.features)
        ddp=DDP(TrainingForward(system,provider),device_ids=[local],broadcast_buffers=False,gradient_as_bucket_view=True)
        train=E117SparseCodeDataset(root/'codes/train',root/'routes/train',split='all')
        sampler=RankBucketSampler(train,args.micro,world,rank,0)
        loader=DataLoader(train,batch_sampler=sampler,collate_fn=collate_e117_sparse,num_workers=args.workers,
            pin_memory=True,generator=torch.Generator().manual_seed(90000+rank),
            **({'persistent_workers':True,'prefetch_factor':2} if args.workers else {}))
        if not len(loader):raise ValueError('no full microbatches')
        def stream():
            epoch,skip,_=origin
            while True:
                sampler.set_epoch(epoch)
                for offset,batch in enumerate(loader):
                    if offset<skip:continue
                    yield batch,dict(packed_pass=epoch,next_microbatch_offset=offset+1)
                epoch+=1;skip=0
        batches=stream();torch.manual_seed(10000+rank)
        if meta:restore(args.resume,system,ema,optimizer,meta,rank=rank,world=world,exact_rng=origin[2])
        elif args.warmup==0:enter_joint(optimizer,system)
        selected=np.random.default_rng(20260922).choice(50000,args.eval_n,replace=False)
        val=E117SparseCodeDataset(root/'codes/val',root/'routes/val',split='all') if rank==0 else None
        if rank==0:
            atomic_json(output/'initialization.json',dict(model=init_report,frozen=provider.audit,resumed=bool(meta)))
            atomic_json(output/'dev_cohort.json',dict(indices=selected.tolist(),split='validation development',mask_seed=20260922))
        def development():
            if rank==0:
                row=dict(step=step,raw=evaluate(system.generator,val,selected),ema=evaluate(ema.generator,val,selected))
                row['raw'].update(image_development(system,provider));row['ema'].update(image_development(ema,provider))
                atomic_json(output/'development_latest.json',row)
                run.log(dict(optimizer_step=step,**{f'dev/{state}_{key}':v for state in ('raw','ema') for key,v in row[state].items()}))
                print(json.dumps({'stage':'development',**row}),flush=True)
            dist.barrier()
        development();cursor=None;previous_epoch=int(step*args.global_batch/SOURCE_N);seen=set();checkpoint=None
        for step in range(start+1,args.steps+1):
            tick=time.monotonic();joint=step>args.warmup
            if joint and len(optimizer.param_groups)==2:enter_joint(optimizer,system)
            set_lrs(optimizer,step,args.warmup);system.train();ddp.zero_grad(set_to_none=True)
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
                sync=not joint and micro+1==accumulation
                with nullcontext() if sync else ddp.no_sync():
                    with torch.autocast('cuda',dtype=torch.bfloat16):loss,metrics=ddp(batch,joint)
                    if not torch.isfinite(loss):raise ValueError('nonfinite CE')
                    (loss/accumulation).backward()
                stats+=metrics/accumulation
            if joint:
                for fmicro in range(fusion_accum):
                    # Counter-derived labels/seeds do not depend on the CE RNG.
                    first=((step-1)*args.fusion_global_batch+rank*(args.fusion_global_batch//world)+fmicro*args.fusion_micro)
                    labels=torch.arange(first,first+args.fusion_micro,device='cuda')%1000
                    packet=online_conditions(system.generator,provider,labels,730000000+first)
                    with nullcontext() if fmicro+1==fusion_accum else ddp.no_sync():
                        loss,metrics=ddp(None,joint,packet)
                        if not torch.isfinite(loss):raise ValueError('nonfinite image loss')
                        (loss/fusion_accum).backward()
                    stats+=metrics/fusion_accum
                    del packet
            if any(p.grad is None for p in system.parameters()):raise ValueError('missing DDP gradient')
            # Separate clipping: image-gradient scale must not suppress CE.
            active_gen=[p for g in optimizer.param_groups if g['name']!='fusion' for p in g['params']]
            norm=torch.nn.utils.clip_grad_norm_(active_gen,1.,error_if_nonfinite=True)
            fnorm=torch.nn.utils.clip_grad_norm_(system.fusion.parameters(),1.,error_if_nonfinite=True)
            if not torch.stack([torch.isfinite(p.grad).all() for p in system.parameters()]).all():raise ValueError('nonfinite gradient')
            provider.assert_frozen();optimizer.step()
            if rank==0:update_ema(ema,system)
            dist.all_reduce(stats);stats/=world
            if rank==0 and (step%10==0 or args.smoke):
                row=dict(step=step,phase='joint' if joint else 'new2d_warmup',**{k:float(v) for k,v in zip(STAT_NAMES,stats)},
                    source_equivalent_epochs=step*args.global_batch/SOURCE_N,grad_norm=float(norm),fusion_grad_norm=float(fnorm),
                    seconds_per_step=time.monotonic()-tick,peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)
                atomic_json(output/'status.json',dict(status='running',**row));print(json.dumps(row),flush=True)
                run.log(dict(optimizer_step=step,**{f'train/{k}':row[k] for k in STAT_NAMES},
                    **{f'train/{k}':row[k] for k in ('source_equivalent_epochs','grad_norm','fusion_grad_norm')},
                    **{f'train/lr_{g["name"]}':g['lr'] for g in optimizer.param_groups},
                    **{f'system/{k}':row[k] for k in ('seconds_per_step','peak_reserved_gib')}))
            ddp.zero_grad(set_to_none=True)
            if step%args.eval_every==0 or step==args.steps:development()
            stop_tensor=torch.tensor(int(stop[0]),device='cuda');dist.all_reduce(stop_tensor,op=dist.ReduceOp.MAX)
            stopped=bool(stop_tensor.item());epoch=int(step*args.global_batch/SOURCE_N)
            if epoch>previous_epoch or step==args.steps or stopped:
                rng=[None]*world;dist.all_gather_object(rng,dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state()))
                if rank==0 and not args.in_memory_smoke:checkpoint=save_latest(output,system,ema,optimizer,rng,dict(step=step,config=config,cursor=cursor,
                    handoff_version=2,stop_reason='signal' if stopped else None,source_equivalent_epochs=step*args.global_batch/SOURCE_N))
                dist.barrier();previous_epoch=epoch
            if stopped:break
        if rank==0:
            summary=dict(status='stopped_safely' if stopped else 'complete',step=step,start_step=start,
                checkpoint=checkpoint,config=config,elapsed_seconds=time.monotonic()-began,disjoint_smoke_pairs=len(seen),
                full_checkpoint_tested=checkpoint is not None)
            atomic_json(output/'summary.json',summary);run.summary['outcome']=summary['status'];run.finish()
        dist.barrier()
    except BaseException as exc:
        if output.exists():atomic_json(output/f'failure_rank{rank}.json',dict(error=str(exc),type=type(exc).__name__,step=step))
        if run is not None:run.finish(exit_code=1)
        raise
    finally:dist.destroy_process_group()


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',type=Path);p.add_argument('--micro',type=int,required=True)
    p.add_argument('--memory',choices=('local','full'),default='local')
    p.add_argument('--global-batch',type=int,default=2048);p.add_argument('--fusion-global-batch',type=int,default=16)
    p.add_argument('--fusion-micro',type=int,default=2);p.add_argument('--steps',type=int,default=50046)
    p.add_argument('--warmup',type=int,default=600);p.add_argument('--eval-every',type=int,default=100)
    p.add_argument('--eval-n',type=int,default=1024);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--smoke',action='store_true');p.add_argument('--prepared-output',action='store_true')
    p.add_argument('--in-memory-smoke',action='store_true',help='bounded developer test ONLY; never satisfies the H20 launch gate')
    return p


if __name__=='__main__':
    p=parser();args=p.parse_args()
    if not 0<=args.warmup<args.steps or min(args.micro,args.fusion_micro,args.eval_every)<1 or not 1<=args.eval_n<=50000:
        p.error('invalid budget or batch')
    if args.in_memory_smoke and (not args.smoke or args.resume or args.steps>5 or args.global_batch>8 or args.micro>2
            or args.fusion_global_batch>16 or args.eval_n>16 or int(os.environ.get('WORLD_SIZE',1))>2):
        p.error('in-memory test requires fresh --smoke, <=5 updates, CE global<=8, fusion global<=16, micro<=2, eval<=16, world<=2')
    main(args)
