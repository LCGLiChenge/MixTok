"""Up to eight GPUs; preserve the original four logical shards and RNG batches."""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import time

from bert2d.paths import ASSETS,EVALUATOR,atomic_json,output_path,sha256


def batch_plan(n,reference_total=None):
    import numpy as np
    total=n if reference_total is None else reference_total
    if reference_total is not None and (total!=50000 or n not in (8,32,64,5000,10000)):
        raise ValueError('50k-prefix mode supports 8/32/64 smoke or 5k/10k')
    batches=[ids[offset:offset+8] for ids in np.array_split(np.arange(total),4)
             for offset in range(0,len(ids),8)]
    if reference_total is None:
        return batches
    prefix=[ids for ids in batches if int(ids[0])<n]
    if any(int(ids[-1])>=n for ids in prefix):
        raise ValueError('prefix boundary cuts through a 50k RNG batch')
    return prefix


def worker_batches(n,workers,worker,reference_total=None):
    import numpy as np
    batches=batch_plan(n,reference_total)
    if not 0<=worker<workers or workers>len(batches): raise ValueError('invalid worker plan')
    return [batches[i] for i in np.array_split(np.arange(len(batches)),workers)[worker]]


def choose_gpus(allowed,count=8):
    fields='index,uuid,memory.free,gpu_recovery_action'
    command=['nvidia-smi',f'--query-gpu={fields}','--format=csv,noheader,nounits']
    try:
        raw=subprocess.check_output(command,text=True,timeout=10)
    except subprocess.CalledProcessError:
        # Older drivers may not provide the recovery-action field.
        command[1]='--query-gpu=index,uuid,memory.free'
        raw=subprocess.check_output(command,text=True,timeout=10)
    cards=[]
    for line in raw.splitlines():
        values=[v.strip() for v in line.split(',')]
        index,uuid,free=values[:3]
        recovery=values[3].lower() if len(values)==4 else 'n/a'
        if recovery not in ('none','n/a'):
            continue  # nvidia-smi can report free memory on a GPU needing reset.
        if (index in allowed or uuid in allowed) and int(free)>=15*1024:
            cards.append((int(free),uuid))
    return [uuid for _,uuid in sorted(cards,reverse=True)[:count]]


def verify_shards(out,n,digest,workers=8,reference_total=None):
    import numpy as np
    manifests=[];receipts=[]
    for shard in range(workers):
        ids=np.concatenate(worker_batches(n,workers,shard,reference_total))
        receipt=json.loads((out/f'shard{shard}.json').read_text())
        manifest=json.loads((out/f'manifest_shard{shard}.json').read_text())
        if (receipt['status']!='complete' or receipt['checkpoint_sha256']!=digest
            or receipt['ids']!=ids.tolist() or not receipt['anchors_unchanged']
            or manifest['checkpoint_sha256']!=digest or manifest['n']!=n):
            raise RuntimeError(f'shard {shard} identity/coverage mismatch')
        route_counts=receipt.get('route_counts')
        if (not isinstance(route_counts,dict) or route_counts.get('n')!=len(ids)
            or not len(ids)<=route_counts.get('sum_k',-1)<=256*len(ids)
            or not 1<=route_counts.get('min_k',0)<=route_counts.get('max_k',257)<=256):
            raise RuntimeError(f'shard {shard} Router token counts are missing or invalid')
        histogram=np.asarray(route_counts.get('histogram',[]),dtype=np.int64)
        if (histogram.shape!=(257,) or histogram.sum()!=len(ids)
            or int(histogram @ np.arange(257))!=route_counts['sum_k']
            or histogram[0]!=0):
            raise RuntimeError(f'shard {shard} Router token histogram is invalid')
        for arm in ('base','full_refine'):
            path=out/f'{arm}_shard{shard}.npy'
            if sha256(path)!=receipt['hashes'][arm]: raise RuntimeError('feature hash mismatch')
            features=np.load(path,mmap_mode='r')
            if features.shape!=(len(ids),2048) or features.dtype!=np.float32 or not np.isfinite(features).all():
                raise RuntimeError('invalid feature array')
        if manifests:
            for key in ('step','seed','batch','stage1','stage2','router','context',
                        'decoder_identities','reference','source_sha256','screen_batch_plan_total'):
                if manifest[key]!=manifests[0][key]: raise RuntimeError(f'shard protocol mismatch: {key}')
        manifests.append(manifest);receipts.append(receipt)
    all_ids=np.concatenate([np.asarray(row['ids'],dtype=np.int64) for row in receipts])
    if not np.array_equal(np.sort(all_ids),np.arange(n,dtype=np.int64)):
        raise RuntimeError('global image ID coverage mismatch')
    return manifests,receipts


def run(args):
    if args.n not in (8,32,64,5000,10000,50000): raise ValueError('unsupported evaluation size')
    worker_count=min(args.workers,len(batch_plan(args.n,args.reference_total)))
    out=output_path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    digest=sha256(args.checkpoint)
    while True:
        gpus=choose_gpus(args.allowed_gpus.split(','),worker_count)
        if len(gpus)==worker_count: break
        atomic_json(out/'status.json',dict(status='waiting_for_gpus',workers_required=worker_count,completed=0,total=args.n,
                    eligible_gpus=gpus,seconds=time.monotonic()-started))
        time.sleep(10)
    atomic_json(out/'parallel_config.json',dict(workers=worker_count,gpus=gpus,checkpoint_sha256=digest,
                source_sha256=sha256(Path(__file__)),seed=args.seed,n=args.n,
                batch=8,logical_shards=4,stage2_steps=args.stage2_steps,
                screen_batch_plan_total=(args.reference_total or args.n),
                stage1_guidance_scale=args.stage1_guidance_scale,
                stage1_randomize_temperature=args.stage1_randomize_temperature,
                context_scale=args.context_scale,
                stage2_logit_scale=args.stage2_logit_scale,
                router_p=args.router_p,
                prune_unused_one_d=True,
                adapter_override=args.allow_adapter_override,
                adapter_step=args.adapter_step,
                adapter_checkpoint=(str(args.adapter_checkpoint.resolve()) if args.adapter_checkpoint else None),
                scorer_checkpoint=(str(args.scorer_checkpoint.resolve()) if args.scorer_checkpoint else None),
                top_p_checkpoint=(str(args.top_p_checkpoint.resolve()) if args.top_p_checkpoint else None),
                sampling_unchanged=(args.stage1_guidance_scale==4.75 and
                                    args.stage1_randomize_temperature==9.5 and
                                    args.context_scale==1.0 and
                                    args.reference_total is None and
                                    args.stage2_steps==8 and
                                    args.stage2_logit_scale==1.0 and args.router_p==0.75)))
    workers=[]
    try:
        for shard,gpu in enumerate(gpus):
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpu
            env['OMP_NUM_THREADS']='4';env['OPENBLAS_NUM_THREADS']='4'
            for name in ('RANK','WORLD_SIZE','LOCAL_RANK','LOCAL_WORLD_SIZE','GROUP_RANK','ROLE_RANK','ROLE_WORLD_SIZE'):
                env.pop(name,None)
            log=(out/f'worker_shard{shard}.log').open('w')
            command=[sys.executable,str(Path(__file__).with_name('eval_worker.py')),
                     '--checkpoint',str(args.checkpoint),'--output',str(out),'--n',str(args.n),
                     '--seed',str(args.seed),'--cfg-w',str(args.cfg_w),
                     '--stage1-guidance-scale',str(args.stage1_guidance_scale),
                     '--stage1-randomize-temperature',str(args.stage1_randomize_temperature),
                     '--context-scale',str(args.context_scale),
                     '--stage2-steps',str(args.stage2_steps),
                     '--stage2-logit-scale',str(args.stage2_logit_scale),
                     '--router-p',str(args.router_p),
                     '--worker-shard',str(shard),'--worker-count',str(worker_count)]
            if args.reference_total is not None:
                command.extend(('--reference-total',str(args.reference_total)))
            if args.adapter_checkpoint is not None:
                command.extend(('--adapter-checkpoint',str(args.adapter_checkpoint)))
            if args.allow_adapter_override:
                command.extend(('--allow-adapter-override','--adapter-step',str(args.adapter_step)))
            if args.scorer_checkpoint is not None:
                command.extend(('--scorer-checkpoint',str(args.scorer_checkpoint)))
            if args.top_p_checkpoint is not None:
                command.extend(('--top-p-checkpoint',str(args.top_p_checkpoint)))
            try: child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            except BaseException: log.close();raise
            workers.append((child,log))
        while True:
            codes=[child.poll() for child,_ in workers]
            if any(code is not None and code!=0 for code in codes):
                raise RuntimeError(f'evaluation worker failed: exit codes {codes}')
            states=[]
            for shard in range(worker_count):
                file=out/f'status_shard{shard}.json'
                states.append(json.loads(file.read_text()) if file.exists() else {})
            atomic_json(out/'status.json',dict(status='running',completed=sum(s.get('completed',0) for s in states),
                total=args.n,workers=worker_count,gpus=gpus,seconds=time.monotonic()-started))
            if all(code==0 for code in codes): break
            time.sleep(2)
    finally:
        for child,_ in workers:
            if child.poll() is None: child.terminate()
        for child,log in workers:
            child.wait();log.close()
    manifests,receipts=verify_shards(out,args.n,digest,worker_count,args.reference_total)
    import numpy as np
    histogram=np.sum([np.asarray(row['route_counts']['histogram'],dtype=np.int64)
                      for row in receipts],axis=0)
    sum_k=sum(row['route_counts']['sum_k'] for row in receipts)
    if not args.n<=sum_k<=256*args.n: raise RuntimeError('global Router token count mismatch')
    mean_2d=sum_k/args.n
    expanded=np.repeat(np.arange(257),histogram)
    tokens=dict(n=args.n,sum_k=sum_k,min_k=min(row['route_counts']['min_k'] for row in receipts),
                max_k=max(row['route_counts']['max_k'] for row in receipts),mean_1d=32,
                mean_generated_2d=mean_2d,mean_generated_total=32+mean_2d,
                p10_k=float(np.quantile(expanded,.1)),p50_k=float(np.quantile(expanded,.5)),
                p90_k=float(np.quantile(expanded,.9)),p99_k=float(np.quantile(expanded,.99)),
                k_histogram=histogram.tolist(),continuous_context_grid_positions=256,
                mean_unselected_context_positions=256-mean_2d)
    if manifests[0]['seed']!=args.seed: raise RuntimeError('worker seed mismatch')
    manifest=dict(manifests[0]);manifest.pop('physical_gpu',None)
    manifest.update(physical_gpus=gpus,workers=worker_count,coordinator_sha256=sha256(Path(__file__)),
                    tokens=tokens)
    atomic_json(out/'manifest.json',manifest)
    atomic_json(out/'status.json',dict(status='computing_fid',completed=args.n,total=args.n,workers=worker_count,
                                     seconds=time.monotonic()-started))
    metrics={}
    if args.n in (5000,10000,50000):
        # Same ADM FIDStatistics and same numpy moments as Evaluator.compute_statistics.
        # All Inception features were extracted on GPUs; only the matrix statistic
        # aggregation is CPU-side, as it was in the previous evaluator.
        os.environ['CUDA_VISIBLE_DEVICES']=''
        import numpy as np
        spec=importlib.util.spec_from_file_location('halton_parallel_adm',EVALUATOR)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with np.load(ASSETS/'fid/VIRTUAL_imagenet256_labeled.npz') as reference:
            ref=module.FIDStatistics(reference['mu'],reference['sigma'])
            for arm in ('base','full_refine'):
                features=np.concatenate([np.load(out/f'{arm}_shard{s}.npy') for s in range(worker_count)])
                stats=module.FIDStatistics(np.mean(features,axis=0),np.cov(features,rowvar=False))
                metrics[arm]=dict(fid=float(stats.frechet_distance(ref)))
    result=dict(status='complete',n=args.n,step=manifest['step'],metrics=metrics,tokens=tokens,checkpoint_sha256=digest,
        seconds=time.monotonic()-started,smoke=args.n not in (5000,10000,50000),anchors_unchanged=True,workers=worker_count,
        peak_reserved_gib=max(r['peak_reserved_gib'] for r in receipts))
    atomic_json(out/'summary.json',result)
    atomic_json(out/'status.json',dict(status='complete',completed=args.n,total=args.n,workers=worker_count,
                                     seconds=result['seconds']))
    print(json.dumps(result),flush=True)


def cli():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--n',type=int,default=50000)
    p.add_argument('--reference-total',type=int,choices=(50000,),
                   help='generate exact first n images using 50k logical RNG batches')
    p.add_argument('--seed',type=int,default=20260914)
    p.add_argument('--cfg-w',type=float,default=3.0)
    p.add_argument('--stage1-guidance-scale',type=float,default=4.75,
                   help='TiTok 1D MaskGIT guidance scale; paper value is 4.75')
    p.add_argument('--stage1-randomize-temperature',type=float,default=9.5,
                   help='TiTok 1D Gumbel randomization amplitude; original value is 9.5')
    p.add_argument('--context-scale',type=float,default=1.0,
                   help='multiply continuous 768D context before Halton; original value is 1.0')
    p.add_argument('--stage2-steps',type=int,choices=(1,2,4,6,8,16,32,48,64),default=8)
    p.add_argument('--stage2-logit-scale',type=float,default=1.0,
                   help='multiply sampled 2D logits; conventional temperature is its reciprocal')
    p.add_argument('--router-p',type=float,default=0.75,
                   help='inference-time top-p refinement mass; training used 0.75')
    p.add_argument('--scorer-checkpoint',type=Path,required=True)
    p.add_argument('--top-p-checkpoint',type=Path,required=True)
    p.add_argument('--adapter-checkpoint',type=Path)
    p.add_argument('--allow-adapter-override',action='store_true',
                   help='explicitly evaluate a fine-tuned adapter descended from this checkpoint')
    p.add_argument('--adapter-step',type=int,
                   help='absolute adapter step; required with --allow-adapter-override')
    p.add_argument('--workers',type=int,choices=(1,2,4,8),default=8)
    # This deployment is explicitly authorized on all eight H20s; the parent
    # trainer's old one-GPU env is not the complete task GPU allocation.
    p.add_argument('--allowed-gpus',default=os.environ.get('MOTAR_EVAL_ALLOWED_GPUS','0,1,2,3,4,5,6,7'))
    args=p.parse_args()
    if args.n==10000 and args.reference_total!=50000:
        p.error('10k screens require --reference-total 50000')
    if args.reference_total is not None and args.n not in (8,32,64,5000,10000):
        p.error('--reference-total 50000 supports 8/32/64 smoke or 5k/10k')
    if not math.isfinite(args.stage1_guidance_scale) or args.stage1_guidance_scale<=0:
        p.error('--stage1-guidance-scale must be finite and positive')
    if not math.isfinite(args.stage1_randomize_temperature) or args.stage1_randomize_temperature<0:
        p.error('--stage1-randomize-temperature must be finite and nonnegative')
    if not math.isfinite(args.context_scale) or args.context_scale<=0:
        p.error('--context-scale must be finite and positive')
    if not math.isfinite(args.stage2_logit_scale) or args.stage2_logit_scale<=0:
        p.error('--stage2-logit-scale must be finite and positive')
    if not math.isfinite(args.router_p) or not 0<args.router_p<=1:
        p.error('--router-p must be in (0, 1]')
    if args.allow_adapter_override != (args.adapter_step is not None):
        p.error('--allow-adapter-override and --adapter-step must be supplied together')
    if args.allow_adapter_override and args.adapter_step<=20000:
        p.error('overridden adapter step must exceed the original 20000')
    try: run(args)
    except BaseException as exc:
        if args.output.exists(): atomic_json(args.output/'failure.json',dict(error=repr(exc)))
        raise
    return 0


if __name__ == '__main__':
    raise SystemExit(cli())
