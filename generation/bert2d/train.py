"""40-epoch portable BERT sparse-2D trainer. Only latest.pt is retained."""
from __future__ import annotations
import argparse, json, math, os, signal, time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from .paths import ASSETS, atomic_json, output_path, sha256
from .periodic_eval import define_axes
from .async_eval import AsyncEvaluator
from .model import BertSparse2D, load_titok_backbone
from .features import PortableBaseFeatures
from .runtime import RankBucketSampler, OffsetSampler, smoothed_objective, save_latest
from .resume import load_resume, restore_rng, resumed_stream
from .optimization import (add_arguments, from_args, make_optimizer, set_learning_rates,
                           validate_saved_config, validate_saved_groups)
from h20.base_model import sample_arccos_mask
from h20.data import E117SparseCodeDataset, collate_e117_sparse

class TrainingForward(nn.Module):
    def __init__(self,core):
        super().__init__();self.core=core
    def forward(self,b):
        z2,valid=b["z2d"],b["route_valid"]
        mask,_=sample_arccos_mask(valid)
        drop=torch.rand(len(z2),device=z2.device)<.1
        logits=self.core.forward_2d(b["z1d"],torch.where(mask,16384,z2),
            b["route_indices"],valid,b["label"],force_drop_ids=drop)
        loss,nll=smoothed_objective(logits,z2,mask,valid)
        return loss,torch.stack((loss.detach(),nll,mask.sum()/valid.sum(),drop.float().mean()))

class SyntheticDataset(Dataset):
    """Deterministic tiny CPU/DDP test; never used by formal training."""
    def __init__(self): self.k_values=torch.tensor([64]*8+[128]*8,dtype=torch.uint8).numpy()
    def __len__(self):return 16
    def __getitem__(self,i):
        g=torch.Generator().manual_seed(1234+i)
        k=int(self.k_values[i])
        return dict(z1d=torch.randint(4096,(32,),generator=g),z2d=torch.randint(16384,(k,),generator=g),
                    route_indices=torch.arange(k),label=torch.tensor(i%1000),
                    source_index=torch.tensor(i),augmentation=torch.tensor(0),k=torch.tensor(k))

def synthetic_features(z1):
    x=torch.arange(256*256,device=z1.device,dtype=torch.float32).reshape(1,256,16,16)
    return torch.sin(x*.01+z1[:,0,None,None,None].float()).detach()

def validate_resume(directory,args,world):
    meta=json.loads((directory/"latest.json").read_text())
    cfg=meta["config"]
    recipe=from_args(args)
    validate_saved_config(cfg,recipe)
    validate_saved_groups(meta["optimizer"],recipe)
    if cfg["style"]!="bert_fullctx_sparse2d_v1":raise ValueError("wrong checkpoint model")
    for key in ("micro","global_batch","seed","lr_new","lr_pretrained","recompute_layers"):
        if cfg[key]!=getattr(args,key):raise ValueError("resume config mismatch: "+key)
    if cfg["world"]!=world:raise ValueError("resume requires identical world size; no silent rank/data reset")
    if bool(cfg.get("synthetic",False))!=args.synthetic:raise ValueError("smoke/formal checkpoints cannot be mixed")
    if not args.synthetic and Path(cfg["assets_root"]).resolve()!=args.assets_root.resolve():
        raise ValueError("resume assets changed; verify and explicitly migrate metadata before resuming")
    if sha256(directory/"latest.pt")!=meta["sha256"]:raise ValueError("checkpoint checksum mismatch")
    return meta

def main(args):
    recipe=from_args(args)
    if args.async_eval and (args.synthetic or args.smoke_updates):
        raise ValueError("async FID requires a formal checkpoint, not trainer synthetic/smoke mode")
    if args.eval_every<1 or not 1<=args.eval_batch<=32:
        raise ValueError("invalid async eval cadence/batch")
    if not 0<args.eval_load_timeout<=args.eval_timeout:
        raise ValueError("invalid eval timeouts")
    rank,world,local=(int(os.environ.get(k,d)) for k,d in (("RANK",0),("WORLD_SIZE",1),("LOCAL_RANK",0)))
    if world not in (1,2,4,8):raise ValueError("supported world sizes: 1,2,4,8")
    if args.global_batch%(args.micro*world):raise ValueError("global batch must divide micro*world")
    if not 0 <= args.stop_after_epoch <= args.epochs:raise ValueError("stop-after-epoch outside total target")
    if args.epochs<1 or args.micro<1:raise ValueError("positive epochs/micro required")
    if args.synthetic and not 1<=args.smoke_updates<=8:raise ValueError("synthetic mode only permits a bounded smoke")
    if args.device=="cpu" and not args.synthetic:raise ValueError("CPU mode is only a synthetic implementation test")
    if args.smoke_updates and not 1<=args.smoke_updates<=8:raise ValueError("smoke capped at 8 updates")
    if not 0<=args.recompute_layers<=24:raise ValueError("recompute layers must be 0..24")
    device=torch.device("cuda",local) if args.device=="cuda" else torch.device("cpu")
    if device.type=="cuda":torch.cuda.set_device(local)
    torch.set_num_threads(4)
    dist.init_process_group("nccl" if device.type=="cuda" else "gloo",timeout=timedelta(seconds=max(1200,args.eval_timeout+120)))
    output=output_path(args.output)
    step=0;started=time.monotonic();run=None;stopping=False;evaluator=None
    def eval_call(action):
        error=[None]
        if rank==0:
            try:action()
            except Exception as exc:error[0]=f"{type(exc).__name__}: {exc}"
        dist.broadcast_object_list(error,src=0)
        if error[0]:raise RuntimeError(error[0])
    def stop_request(*_):
        nonlocal stopping
        stopping=True
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop_request)
    try:
        if rank==0:
            if output.exists() and args.resume is None:raise FileExistsError("fresh init refuses an existing output")
            output.mkdir(parents=True,exist_ok=True)
        dist.barrier()
        meta=validate_resume(args.resume,args,world) if args.resume else None
        torch.manual_seed(args.seed)
        core=BertSparse2D(dim=32,depth=2,heads=4,dropout=.1) if args.synthetic else BertSparse2D()
        if meta:audit=meta["init_audit"]
        elif args.synthetic:
            audit=dict(fresh_parameters=list(core.state_dict()),synthetic_only=True)
        else:audit=load_titok_backbone(core,args.assets_root/"weights/generator_titok_l32.bin")
        core.to(device).train()
        if args.recompute_layers:core.enable_recompute(min(args.recompute_layers,len(core.encoder.layer)))
        provider=synthetic_features if args.synthetic else PortableBaseFeatures(str(device),args.feature_chunk,assets_root=args.assets_root)
        core.set_feature_provider(provider)
        opt=make_optimizer(core,audit,recipe)
        restored=load_resume(core,opt,args.resume,meta,rank) if meta else None
        start=int(meta["step"]) if meta else 0
        data=SyntheticDataset() if args.synthetic else E117SparseCodeDataset(args.assets_root/"codes/train",args.assets_root/"routes/train",split="all")
        if not args.synthetic and len(data)!=2562334:raise ValueError("expected full paired ImageNet TRAIN cache")
        accum=args.global_batch//(args.micro*world)
        updates_per_epoch=math.ceil(len(data)/args.global_batch)
        target=args.smoke_updates or args.epochs*updates_per_epoch
        if start>target or (start==target and not args.async_eval):raise ValueError("checkpoint already reached requested TOTAL target")
        step=start
        cfg={k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(args).items()}
        cfg.update(optimization=recipe,style="bert_fullctx_sparse2d_v1",dim=core.dim,depth=len(core.encoder.layer),
            heads=4 if args.synthetic else 16,dropout=.1,world=world,accumulation=accum,
            updates_per_epoch=updates_per_epoch,total_updates=target,packed_examples=len(data),
            epoch_definition="ceil(packed_examples/global_batch) consecutive updates; bucket drop_last",
            sequence="[class|feature256|sparse2dK]",objective="2D-only",class_dropout=.1,
            smoothing=.1,visible_weight=.1,direct_replace=True,eval_state="raw",
            eval_1d_source="frozen official TiTok",resume_step=start,
            resume_sha256=meta["sha256"] if meta else None,
            source_image_equivalent_epochs=target*args.global_batch/1281167)
        if rank==0:
            atomic_json(output/"config.json",cfg)
            atomic_json(output/"init_audit.json",audit)
            if args.wandb_mode!="disabled":
                import wandb
                id_path=output/"wandb_run.json"
                wandb_id=json.loads(id_path.read_text())["id"] if id_path.exists() else wandb.util.generate_id()
                run=wandb.init(project=args.wandb_project,name=output.name,dir=str(output),mode=args.wandb_mode,
                               config=cfg,id=wandb_id,resume="allow")
                atomic_json(id_path,dict(id=wandb_id,project=args.wandb_project))
                define_axes(run)
        sampler=OffsetSampler(RankBucketSampler(data,args.micro,world,rank,args.seed))
        loader=DataLoader(data,batch_sampler=sampler,collate_fn=collate_e117_sparse,num_workers=args.workers,
            pin_memory=device.type=="cuda",generator=torch.Generator().manual_seed(90000+rank))
        cursor=meta["cursor"] if meta else dict(packed_pass=0,next_microbatch_offset=0)
        stream=resumed_stream(loader,sampler,cursor)
        ddp=DDP(TrainingForward(core),device_ids=[local] if device.type=="cuda" else None,
                broadcast_buffers=False,gradient_as_bucket_view=True)
        if restored:
            restore_rng(restored)
            atomic_json(output/f"resume_audit_rank{rank}.json",{k:v for k,v in restored.items() if not k.startswith("rng_")})
        else:torch.manual_seed(args.seed+1000+rank)
        window=torch.zeros(4,device=device);count=0;clipped=0;tick=time.monotonic()
        segment_target = min(target, args.stop_after_epoch * updates_per_epoch) if args.stop_after_epoch else target
        if segment_target < start or (segment_target==start and not args.async_eval):
            raise ValueError("segment boundary already reached")
        if args.async_eval:
            def initialize_evaluator():
                nonlocal evaluator
                gpus=[x.strip() for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
                if len(gpus)!=world or len(set(gpus))!=world:
                    raise ValueError("async eval requires explicit same GPUs as training")
                evaluator=AsyncEvaluator(output,args.assets_root,gpus,run,batch=args.eval_batch,
                    seed=args.eval_seed,mode=args.wandb_mode,load_timeout=args.eval_load_timeout,
                    timeout=args.eval_timeout)
                if meta and start%updates_per_epoch==0 and (start//updates_per_epoch)%args.eval_every==0:
                    evaluator.start(meta)
            eval_call(initialize_evaluator)
            tick=time.monotonic()
        for step in range(start+1,segment_target+1):
            set_learning_rates(opt,recipe,step)
            opt.zero_grad(set_to_none=True);stats=torch.zeros(4,device=device)
            for micro in range(accum):
                cpu,cursor=next(stream);batch={k:v.to(device,non_blocking=True) for k,v in cpu.items()}
                with ddp.no_sync() if micro+1<accum else nullcontext():
                    with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):
                        loss,values=ddp(batch)
                    if not bool(torch.isfinite(loss)):raise RuntimeError("nonfinite loss")
                    (loss/accum).backward()
                stats+=values/accum
            grad=nn.utils.clip_grad_norm_(core.parameters(),recipe["grad_clip"],error_if_nonfinite=True);opt.step()
            clipped+=int(float(grad)>recipe["grad_clip"])
            dist.all_reduce(stats);window+=stats/world;count+=1
            flag=torch.tensor(int(stopping),device=device);dist.all_reduce(flag,op=dist.ReduceOp.MAX)
            stopping=bool(flag.item())
            if args.async_eval and (step%10==0 or step==segment_target or stopping):
                eval_call(lambda:evaluator.poll())
            if step<=start+3 or step%10==0 or step==segment_target or stopping:
                row=dict(step=step,epoch=step/updates_per_epoch,status="running",loss2d=float(window[0]/count),
                    masked_nll2d=float(window[1]/count),mask_ratio=float(window[2]/count),class_drop_fraction=float(window[3]/count),
                    grad_norm=float(grad),seconds_per_step=(time.monotonic()-tick)/count,
                    peak_reserved_gib=torch.cuda.max_memory_reserved(device)/1024**3 if device.type=="cuda" else 0.,
                    lr_new=opt.param_groups[0]["lr"],samples_seen=step*args.global_batch,
                    lr_pretrained=next((g["lr"] for g in opt.param_groups if g["name"]=="pretrained"),0.),
                    clip_fraction=clipped/count,reference_updates=step*args.global_batch/448,
                    beta1=recipe["betas"][0],beta2=recipe["betas"][1],
                    optimizer_eps=recipe["eps"],weight_decay=recipe["weight_decay"],
                    samples_per_second=count*args.global_batch/max(time.monotonic()-tick,1e-9))
                if rank==0:
                    atomic_json(output/"status.json",row)
                    with (output/"metrics.jsonl").open("a") as f:f.write(json.dumps(row)+"\n")
                    print(json.dumps(row),flush=True)
                    if run:run.log(row)
                window.zero_();count=0;clipped=0;tick=time.monotonic()
            if step%updates_per_epoch==0 or step==segment_target or stopping:
                if args.async_eval:
                    # Usually already complete; wait only if evaluation fell behind.
                    eval_call(lambda:evaluator.finish())
                rng=[None]*world
                cuda_rng=torch.cuda.get_rng_state(device) if device.type=="cuda" else torch.empty(0,dtype=torch.uint8)
                dist.all_gather_object(rng,dict(cpu=torch.get_rng_state(),cuda=cuda_rng))
                if rank==0:
                    saved=save_latest(output,core,None,opt,rng,dict(step=step,config=cfg,cursor=cursor,init_audit=audit))
                    print(json.dumps(dict(stage="latest",**saved)),flush=True)
                dist.barrier()
                if args.async_eval and not stopping and step%updates_per_epoch==0 and (step//updates_per_epoch)%args.eval_every==0:
                    eval_call(lambda:evaluator.start(json.loads((output/"latest.json").read_text())))
                tick=time.monotonic()
            if stopping:break
        if args.async_eval:eval_call(lambda:evaluator.finish())
        if not args.synthetic:provider.assert_frozen()
        if rank==0:
            summary=dict(status="interrupted" if stopping else ("awaiting_eval" if step<target else "complete"),step=step,target=target,
                         epoch=step/updates_per_epoch,elapsed_seconds=time.monotonic()-started,config=cfg)
            atomic_json(output/"status.json",summary);atomic_json(output/"summary.json",summary)
            if run:run.finish()
        dist.barrier()
    except BaseException as exc:
        if output.exists():atomic_json(output/f"failure_rank{rank}.json",dict(step=step,error=str(exc),type=type(exc).__name__))
        raise
    finally:
        try:
            if evaluator is not None:evaluator.cancel()
        finally:dist.destroy_process_group()

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--assets-root",type=Path,default=ASSETS)
    p.add_argument("--resume",type=Path,help="checkpoint DIRECTORY; restores raw+Adam+rank RNG+cursor")
    p.add_argument("--epochs",type=int,default=40)
    p.add_argument("--stop-after-epoch",type=int,default=0,help="save and exit at this boundary without changing total schedule")
    p.add_argument("--micro",type=int,default=56)
    p.add_argument("--global-batch",type=int,default=448)
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--workers",type=int,default=2)
    p.add_argument("--feature-chunk",type=int,default=8)
    add_arguments(p)
    p.add_argument("--recompute-layers",type=int,default=10)
    p.add_argument("--wandb-project",default="motar-bert-sparse2d")
    p.add_argument("--wandb-mode",choices=("online","offline","disabled"),default="online")
    p.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    p.add_argument("--async-eval",action="store_true",help="background FID, same GPUs, no snapshot; enabled by launcher")
    p.add_argument("--eval-every",type=int,default=1)
    p.add_argument("--eval-batch",type=int,default=8)
    p.add_argument("--eval-seed",type=int,default=20260914)
    p.add_argument("--eval-load-timeout",type=float,default=600)
    p.add_argument("--eval-timeout",type=float,default=1800)
    p.add_argument("--synthetic",action="store_true")
    p.add_argument("--smoke-updates",type=int,default=0,help="bounded TOTAL target for startup tests, not epochs")
    return p
if __name__=="__main__":main(parser().parse_args())
