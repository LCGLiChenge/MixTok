"""Up to eight GPUs; preserve the original four logical shards and RNG batches."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from bert2d.paths import ASSETS,EVALUATOR,atomic_json,output_path,sha256


def batch_plan(n):
    import numpy as np
    return [ids[offset:offset+8] for ids in np.array_split(np.arange(n),4)
            for offset in range(0,len(ids),8)]


def worker_batches(n,workers,worker):
    import numpy as np
    batches=batch_plan(n)
    if not 0<=worker<workers or workers>len(batches): raise ValueError('invalid worker plan')
    return [batches[i] for i in np.array_split(np.arange(len(batches)),workers)[worker]]


def choose_gpus(allowed,count=8):
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free',
                                 '--format=csv,noheader,nounits'],text=True,timeout=10)
    cards=[]
    for line in raw.splitlines():
        index,uuid,free=(v.strip() for v in line.split(','))
        if (index in allowed or uuid in allowed) and int(free)>=15*1024:
            cards.append((int(free),uuid))
    return [uuid for _,uuid in sorted(cards,reverse=True)[:count]]


def verify_shards(out,n,digest,workers=8):
    import numpy as np
    manifests=[];receipts=[]
    for shard in range(workers):
        ids=np.concatenate(worker_batches(n,workers,shard))
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
            for key in ('step','seed','batch','stage1','stage2','router','proxy','decoder_identities','reference','source_sha256'):
                if manifest[key]!=manifests[0][key]: raise RuntimeError(f'shard protocol mismatch: {key}')
        manifests.append(manifest);receipts.append(receipt)
    return manifests,receipts


def run(args):
    if args.n not in (8,32,64,5000,50000): raise ValueError('unsupported evaluation size')
    worker_count=min(args.workers,len(batch_plan(args.n)))
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
                mapper_checkpoint=(str(args.mapper_checkpoint.resolve()) if args.mapper_checkpoint else None),
                scorer_checkpoint=(str(args.scorer_checkpoint.resolve()) if args.scorer_checkpoint else None),
                top_p_checkpoint=(str(args.top_p_checkpoint.resolve()) if args.top_p_checkpoint else None),
                sampling_unchanged=args.stage2_steps==8))
    workers=[]
    try:
        for shard,gpu in enumerate(gpus):
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpu
            env['OMP_NUM_THREADS']='4';env['OPENBLAS_NUM_THREADS']='4'
            for name in ('RANK','WORLD_SIZE','LOCAL_RANK','LOCAL_WORLD_SIZE','GROUP_RANK','ROLE_RANK','ROLE_WORLD_SIZE'):
                env.pop(name,None)
            log=(out/f'worker_shard{shard}.log').open('w')
            command=[sys.executable,str(Path(__file__).with_name('eval_top_p_worker.py')),
                     '--checkpoint',str(args.checkpoint),'--output',str(out),'--n',str(args.n),
                     '--seed',str(args.seed),'--cfg-w',str(args.cfg_w),
                     '--stage2-steps',str(args.stage2_steps),
                     '--worker-shard',str(shard),'--worker-count',str(worker_count)]
            if args.mapper_checkpoint is not None:
                command.extend(('--mapper-checkpoint',str(args.mapper_checkpoint)))
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
    manifests,receipts=verify_shards(out,args.n,digest,worker_count)
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
                k_histogram=histogram.tolist(),proxy_context_2d_tokens=256)
    if manifests[0]['seed']!=args.seed: raise RuntimeError('worker seed mismatch')
    manifest=dict(manifests[0]);manifest.pop('physical_gpu',None)
    manifest.update(physical_gpus=gpus,workers=worker_count,coordinator_sha256=sha256(Path(__file__)),
                    tokens=tokens)
    atomic_json(out/'manifest.json',manifest)
    atomic_json(out/'status.json',dict(status='computing_fid',completed=args.n,total=args.n,workers=worker_count,
                                     seconds=time.monotonic()-started))
    metrics={}
    if args.n in (5000,50000):
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
        seconds=time.monotonic()-started,smoke=args.n not in (5000,50000),anchors_unchanged=True,workers=worker_count,
        peak_reserved_gib=max(r['peak_reserved_gib'] for r in receipts))
    atomic_json(out/'summary.json',result)
    atomic_json(out/'status.json',dict(status='complete',completed=args.n,total=args.n,workers=worker_count,
                                     seconds=result['seconds']))
    print(json.dumps(result),flush=True)


def cli():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--n',type=int,default=5000)
    p.add_argument('--seed',type=int,default=20260914)
    p.add_argument('--cfg-w',type=float,default=2.75)
    p.add_argument('--stage2-steps',type=int,choices=(1,2,4,6,8,16,32),default=8)
    p.add_argument('--scorer-checkpoint',type=Path,required=True)
    p.add_argument('--top-p-checkpoint',type=Path,required=True)
    p.add_argument('--mapper-checkpoint',type=Path)
    p.add_argument('--workers',type=int,choices=(1,2,4,8),default=8)
    # This deployment is explicitly authorized on all eight H20s; the parent
    # trainer's old one-GPU env is not the complete task GPU allocation.
    p.add_argument('--allowed-gpus',default=os.environ.get('MOTAR_EVAL_ALLOWED_GPUS','0,1,2,3,4,5,6,7'))
    args=p.parse_args()
    try: run(args)
    except BaseException as exc:
        if args.output.exists(): atomic_json(args.output/'failure.json',dict(error=repr(exc)))
        raise
    return 0


if __name__ == '__main__':
    raise SystemExit(cli())
