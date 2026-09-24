"""Explicit-GPU memory probe + save/resume gate + 40-epoch latest-only training."""
import argparse,json,os,shutil,subprocess,sys,tempfile
from pathlib import Path
from .paths import ASSETS,RESULT_ROOT,ROOT,atomic_json,output_path
from .assets import verify
from .periodic_eval import run_schedule
from .optimization import add_arguments, from_args, cli_arguments, validate_saved_config, validate_saved_groups
def run(cmd,env):subprocess.run(cmd,cwd=ROOT,env=env,check=True)
def main(a):
    recipe=from_args(a)
    if not 0<=a.test_epochs<=a.epochs:
        raise ValueError("test-epochs must be zero or at most the total epochs")
    if a.print_optimization:
        print(json.dumps(recipe,indent=2));return
    out=output_path(a.output)
    if (out/"latest.pt").exists():
        meta=json.loads((out/"latest.json").read_text())
        validate_saved_config(meta["config"],recipe)
        validate_saved_groups(meta["optimizer"],recipe)
    visible=os.environ.get("CUDA_VISIBLE_DEVICES","")
    gpus=[s.strip() for s in visible.split(",") if s.strip()]
    if len(gpus) not in (1,2,4,8) or len(set(gpus))!=len(gpus):
        raise ValueError("set explicitly authorized CUDA_VISIBLE_DEVICES (1/2/4/8 distinct GPUs)")
    n=len(gpus)
    if a.eval_every<1 or not 1<=a.eval_batch<=32:raise ValueError("positive eval cadence and batch 1..32 required")
    if a.epochs<1 or a.global_batch<1 or a.micro<0:raise ValueError("positive epoch/global batch and nonnegative micro required")
    if a.global_batch%n:raise ValueError("global batch must divide number of GPUs")
    if a.micro and a.global_batch%(a.micro*n):raise ValueError("micro*world must divide global batch")
    out=output_path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(out.parent).free<20*1024**3:raise RuntimeError("at least 20 GiB free output space required before startup")
    env=os.environ.copy();env.update(USE_TF="0",PYTORCH_ALLOC_CONF="expandable_segments:True")
    if not a.skip_asset_verify:verify(a.assets_root,include_fid=True)
    run([sys.executable,"-c","import tensorflow as tf; import scipy; assert tf.config.list_physical_devices('GPU'), 'ADM TensorFlow GPU unavailable'"],dict(env,CUDA_VISIBLE_DEVICES=gpus[0]))
    rows=subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid,memory.free","--format=csv,noheader,nounits"],text=True)
    free={}
    for row in rows.strip().splitlines():
        idx,uid,value=[x.strip() for x in row.split(",")]
        free[idx]=int(value);free[uid]=int(value)
    if any(g not in free for g in gpus):raise ValueError("GPU identifiers must be nvidia-smi index or full UUID")
    reserve=a.eval_reserve_gib*1024 if a.eval_mode=="async" else 0.
    if not 0<a.eval_reserve_gib or not 0<a.eval_load_timeout<=a.eval_timeout:
        raise ValueError("positive eval memory reservation and valid timeouts required")
    limited=min(gpus,key=lambda x:free[x]);limit=free[limited]*.92-reserve
    if limit<=1536:raise RuntimeError("not enough memory after async-eval reservation")
    resume=out if (out/"latest.pt").exists() else None
    if out.exists() and resume is None:raise FileExistsError("existing output has no checkpoint; inspect before reusing")
    if resume:
        old=json.loads((resume/"config.json").read_text())
        micro=old["micro"]
        if old["world"]!=n or old["global_batch"]!=a.global_batch:raise ValueError("resume DDP layout changed")
        if a.micro and a.micro!=micro:raise ValueError("resume microbatch changed")
    else:micro=a.micro
    candidates=[micro] if micro else [v for v in range(a.global_batch//n,0,-1) if a.global_batch%(v*n)==0]
    probe_dir=Path(tempfile.mkdtemp(prefix="bert2d_startup_",dir=out.parent))
    chosen=None
    for value in candidates:
        result=probe_dir/f"memory_m{value}.json"
        probe_env=dict(env,CUDA_VISIBLE_DEVICES=limited)
        cmd=[sys.executable,"-m","bert2d.probe","--micro",str(value),"--output",str(result),
             "--assets-root",str(a.assets_root),"--recompute-layers",str(a.recompute_layers),"--limit-mib",str(limit),
             "--global-batch",str(a.global_batch)]+cli_arguments(a)
        rc=subprocess.run(cmd,cwd=ROOT,env=probe_env).returncode
        if rc not in (0,2):raise RuntimeError("capacity probe failed; not a recoverable capacity result")
        row=json.loads(result.read_text())
        if row["status"]=="ok":chosen=value;break
    if chosen is None:raise RuntimeError("no allowed microbatch fits; no training started")
    base=[sys.executable,"-m","torch.distributed.run","--standalone",f"--nproc_per_node={n}","-m","bert2d.train"]
    common=["--micro",str(chosen),"--global-batch",str(a.global_batch),"--assets-root",str(a.assets_root),
            "--recompute-layers",str(a.recompute_layers)]+cli_arguments(a)
    if not resume:
        smoke=probe_dir/"save_resume_smoke"
        run(base+common+["--output",str(smoke),"--smoke-updates","2","--wandb-mode","disabled"],env)
        run(base+common+["--output",str(smoke),"--resume",str(smoke),"--smoke-updates","3","--wandb-mode","disabled"],env)
        report=json.loads((smoke/"latest.json").read_text())
        if report["step"]!=3 or not report["round_trip_exact"]:raise RuntimeError("save/resume gate failed")
        weight=smoke/"latest.pt"
        if weight.is_symlink() or weight.resolve().parent!=smoke.resolve():raise ValueError("unsafe smoke cleanup")
        atomic_json(probe_dir/"cleanup.json",dict(path=str(weight),sha256=report["sha256"],bytes=weight.stat().st_size,status="deleting"))
        weight.unlink()
        atomic_json(probe_dir/"cleanup.json",dict(path=str(weight),sha256=report["sha256"],status="deleted_after_verified_resume"))
    # Probe/smoke allocated only the explicitly requested GPUs. Recheck at the handoff boundary.
    now=subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid,memory.free","--format=csv,noheader,nounits"],text=True)
    for row in now.strip().splitlines():
        idx,uid,value=[x.strip() for x in row.split(",")]
        if (idx in gpus or uid in gpus) and int(value)<row_peak(row=None,probe_dir=probe_dir,micro=chosen)+reserve:
            raise RuntimeError("GPU memory availability changed before training")
    cmd=base+common+["--output",str(out),"--epochs",str(a.epochs),
        "--wandb-project",a.wandb_project,"--wandb-mode",a.wandb_mode]
    if a.eval_mode=="async":
        cmd += ["--async-eval","--eval-every",str(a.eval_every),"--eval-batch",str(a.eval_batch),
                "--eval-seed",str(a.eval_seed),"--eval-load-timeout",str(a.eval_load_timeout),
                "--eval-timeout",str(a.eval_timeout)]
        if resume:cmd += ["--resume",str(out)]
        if a.test_epochs:cmd += ["--stop-after-epoch",str(a.test_epochs)]
    atomic_json(probe_dir/"launch.json",dict(command=cmd,global_batch=a.global_batch,micro=chosen,world=n,
        epochs=a.epochs,test_epochs=a.test_epochs,optimization=recipe,eval_mode=a.eval_mode,
        eval_every=a.eval_every,eval_reserve_gib=a.eval_reserve_gib))
    a.output=out
    # Bounded test stops the orchestrator, but trainer keeps its full epoch target.
    if a.eval_mode=="async":run(cmd,env)
    else:run_schedule(schedule_arguments(a),cmd,env,gpus)
def schedule_arguments(a):
    if not 0<=a.test_epochs<=a.epochs:raise ValueError("invalid test epoch limit")
    limited=argparse.Namespace(**vars(a))
    if a.test_epochs:limited.epochs=a.test_epochs
    return limited
def row_peak(row,probe_dir,micro):
    d=json.loads((probe_dir/f"memory_m{micro}.json").read_text())
    return d["peak_reserved_mib"]+1536
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=RESULT_ROOT/"bert_sparse2d_40epoch")
    p.add_argument("--assets-root",type=Path,default=ASSETS)
    p.add_argument("--epochs",type=int,default=40)
    p.add_argument("--test-epochs",type=int,default=0,help="bounded test: stop after N epochs; keep full trainer target; 0 disables")
    p.add_argument("--eval-every",type=int,default=1,help="run paired 5k FID every N completed epochs")
    p.add_argument("--eval-mode",choices=("async","serial"),default="async")
    p.add_argument("--eval-reserve-gib",type=float,default=12,help="per-GPU headroom reserved during training probe for async eval")
    p.add_argument("--eval-load-timeout",type=float,default=600)
    p.add_argument("--eval-timeout",type=float,default=1800)
    p.add_argument("--eval-batch",type=int,default=8)
    p.add_argument("--eval-seed",type=int,default=20260914)
    p.add_argument("--global-batch",type=int,default=448)
    add_arguments(p)
    p.add_argument("--print-optimization",action="store_true",help="print resolved recipe and exit before any GPU or asset checks")
    p.add_argument("--micro",type=int,default=0,help="0 probes divisors while preserving global batch")
    p.add_argument("--recompute-layers",type=int,default=10)
    p.add_argument("--wandb-project",default="motar-bert-sparse2d")
    p.add_argument("--wandb-mode",choices=("online","offline","disabled"),default="online")
    p.add_argument("--skip-asset-verify",action="store_true",help="only after explicit separate successful verification")
    main(p.parse_args())
