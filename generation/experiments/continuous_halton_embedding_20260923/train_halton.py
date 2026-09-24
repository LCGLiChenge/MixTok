"""Stage 2: train official Halton with a frozen continuous embedding adapter."""
from __future__ import annotations

import argparse
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("USE_TF", "0")
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from bert2d.paths import atomic_json, output_path, sha256
from experiments.continuous_halton_embedding_20260923.data import (
    RankDynamicBucketSampler, ContinuousRouteDataset as TopPProxyRouteDataset,
    collate_continuous as collate_top_p_proxy_route,
)
from experiments.feature_to_token_20260916.halton_official_completion import (
    HaltonCompletionObjective, official_training_api, sample_halton_completion,
)
from experiments.feature_to_token_20260916.halton_proxy import load_pretrained
from experiments.continuous_halton_embedding_20260923.continuous_halton import (
    ContinuousHalton as HaltonProxy, ContinuousObjective as MixedHaltonCompletionObjective,
    sample_continuous_completion as sample_halton_completion)
from experiments.continuous_halton_embedding_20260923.embedding_adapter import load_adapter
from experiments.feature_to_token_20260916.assets_top_p import TopPFrozenAssets
from experiments.feature_to_token_20260916.train_proxy_converter import make_features
from bert2d.paths import ASSETS
from experiments.continuous_halton_embedding_20260923.async_eval import AsyncTopPEval


def parse_extra_eval_steps(value: str) -> list[int]:
    steps = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        step = int(item)
        if step < 1:
            raise ValueError("extra eval steps must be positive")
        steps.append(step)
    return sorted(set(steps))


def set_budget_cosine_lr(optimizer, step: int, peak_lr: float, warmup: int,
                         hold_until: int, decay_until: int, min_lr: float) -> float:
    """Warm up, hold, then cosine-decay on the actual update budget."""
    if not (1 <= step <= decay_until):
        raise ValueError("step must be within the budget-cosine horizon")
    if not (0 <= warmup <= hold_until < decay_until):
        raise ValueError("expected 0 <= warmup <= hold_until < decay_until")
    if not (0.0 <= min_lr <= peak_lr):
        raise ValueError("expected 0 <= min_lr <= peak_lr")
    if warmup and step <= warmup:
        lr = peak_lr * step / warmup
    elif step <= hold_until:
        lr = peak_lr
    else:
        progress = (step - hold_until) / (decay_until - hold_until)
        lr = min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def save_checkpoint(out: Path, core: HaltonProxy, optimizer, step: int, config: dict, cursor: dict, rng_states: list):
    """Atomic rolling latest; torch serialization retains tied embedding storage."""
    import shutil
    if shutil.disk_usage(out).free < 10 * 1024**3:
        raise RuntimeError("less than 10 GiB free for checkpoint")
    state = dict(step=step, config=config, cursor=cursor, rng_states=rng_states,
                 model={k: v.detach().cpu() for k, v in core.state_dict().items()},
                 optimizer=optimizer.state_dict())
    temporary = out / "latest.pt.tmp"
    torch.save(state, temporary)
    check = torch.load(temporary, map_location="cpu", mmap=True, weights_only=True)
    if check["step"] != step or set(check["model"]) != set(state["model"]):
        raise RuntimeError("checkpoint round-trip header mismatch")
    for key in state["model"]:
        if not torch.equal(check["model"][key], state["model"][key]):
            raise RuntimeError(f"checkpoint round-trip mismatch: {key}")
    digest = sha256(temporary)
    temporary.replace(out / "latest.pt")
    atomic_json(out / "latest.json", dict(step=step, sha256=digest,
                bytes=(out / "latest.pt").stat().st_size, round_trip_exact=True,
                config=config))


def run(args):
    rank, world, local = (int(os.environ[k]) for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    if world not in (4, 8):
        raise RuntimeError("this run requires four or eight H20 GPUs")
    if args.epochs < 1 or (args.updates is not None and not args.smoke):
        raise ValueError('updates override is smoke-only; formal horizon is fixed by epochs')
    if args.stop_after is not None and (args.stop_after < 1 or args.smoke):
        raise ValueError('stop-after must be positive and is for non-smoke comparisons')
    out = output_path(args.output)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite experiment: {out}")
    torch.set_num_threads(4)
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    if "H20" not in torch.cuda.get_device_name(device):
        raise RuntimeError("expected H20 GPU")
    # Eight-H20 smoke measured 9.84 GiB reserved at micro32. Evaluation has
    # its own admission check; do not require its headroom on every rank.
    if torch.cuda.mem_get_info(device)[0] < args.min_free_gib * 1024**3:
        raise RuntimeError("insufficient free memory for continuous training")
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.skip_smoke_fid and not args.smoke:
        raise ValueError("only small smoke may skip GPU FID")
    dist.init_process_group("nccl", timeout=timedelta(minutes=15))
    # RNG/header objects are CPU metadata; avoid creating extra CUDA gather
    # communicators while the low-memory smoke shares GPUs with another job.
    metadata_group = dist.new_group(backend="gloo", timeout=timedelta(minutes=15))
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    if ((not args.smoke and args.stop_after != 20000)
            or args.lr_schedule != "budget-cosine"
            or args.mask_objective != "mixed-prefix"):
        raise ValueError("formal top-p run requires 20k budget-cosine updates")
    cache_config_path = args.cache_root / "config.json"
    cache_summary_path = args.cache_root / "summary.json"
    cache_config = json.loads(cache_config_path.read_text())
    cache_summary = json.loads(cache_summary_path.read_text())
    if (cache_config.get("format") != "top_p_continuous_route_cache_v1"
            or cache_summary.get("status") != "complete"
            or cache_summary.get("complete_coverage") is not True):
        raise RuntimeError("final top-p condition cache is incomplete")
    verified = [None]
    if rank == 0:
        pretrained_sha = sha256(args.pretrained)
        if pretrained_sha != "7fe25cb80b05743e8b42dacc61d88792a9ab5217d6bd756bef28821e0a5fc68f":
            raise RuntimeError("Halton pretrained checkpoint SHA256 mismatch")
        if cache_config.get("top_p_checkpoint_sha256") != sha256(args.top_p_ckpt):
            raise RuntimeError("cache and top-p tokenizer differ")
        if not args.mapper_ckpt.is_file():
            raise RuntimeError("continuous adapter checkpoint missing")
        if cache_config.get("scorer_sha256") != sha256(args.scorer_ckpt):
            raise RuntimeError("cache and accepted scorer differ")
        if cache_summary.get("config_sha256") != sha256(cache_config_path):
            raise RuntimeError("final condition cache config hash differs")
        verified[0] = pretrained_sha
    dist.broadcast_object_list(verified, src=0, group=metadata_group)
    cache_config["mapper_sha256"] = sha256(args.mapper_ckpt)
    cache_config["context_kind"] = "frozen_continuous_adapter"
    transfer = True
    extra_eval_steps = parse_extra_eval_steps(args.extra_eval_steps)
    if args.lr_schedule == "budget-cosine":
        schedule_budget = args.lr_decay_until if args.smoke else args.stop_after
        if schedule_budget != args.lr_decay_until:
            raise ValueError("budget-cosine requires the execution budget == lr-decay-until")
        if not (0 <= args.warmup <= args.lr_hold_until < args.lr_decay_until):
            raise ValueError("invalid budget-cosine phase boundaries")
    config = dict(style="top_p_halton_continuous_embedding_v1", seed=args.seed,
                  world=world, micro=args.micro, global_batch=world * args.micro,
                  epochs=args.epochs, lr=args.lr, warmup=args.warmup,
                  lr_schedule=args.lr_schedule, lr_hold_until=args.lr_hold_until,
                  lr_decay_until=args.lr_decay_until, min_lr=args.min_lr,
                  weight_decay=.03, betas=[.9,.999],
                  loss=('masked-only CE on selected positions; 16385 classes'
                        if args.mask_objective == 'mixed-prefix' else
                        'official equal-weight CE on all selected positions; 16385 classes'),
                  mask=('half official arccos random, half Halton-prefix including full-MASK'
                        if args.mask_objective == 'mixed-prefix' else
                        'official get_mask_code arccos on K positions'),
                  mask_objective=args.mask_objective,
                  sampling=dict(order='official Halton filtered to selection', steps=32,
                                cfg_w=args.eval_cfg_w, temperature=1., exclude_mask_at_sampling=True),
                  initialization='all official Halton parameters; fresh optimizer and task iteration zero',
                  init_model_sha256=None,
                  route_root=str(args.cache_root.resolve()),
                  route_meta_sha256=sha256(cache_config_path),
                  eval_router_mode='top-p-1d-only-scorer',
                  eval_stage2_steps=args.eval_stage2_steps,
                  upstream_hashes={str(p):sha256(args.upstream_root / p) for p in (
                      'Network/transformer.py','Utils/masking_scheduler.py',
                      'Sampler/halton_sampler.py','Trainer/abstract_trainer.py',
                      'Trainer/cls_trainer.py','launch/run_cls_to_img.sh')},
                  extra_eval_steps=extra_eval_steps,eval_every_epoch=True,
                  eval_n=8 if args.smoke else 5000,eval_gpu=str(args.eval_gpu),
                  log_every=args.log_every,
                  proxy_root=str(args.cache_root.resolve()),
                  eval_mapper_checkpoint=str(args.mapper_ckpt.resolve()),
                  eval_scorer_checkpoint=str(args.scorer_ckpt.resolve()),
                  source_root=str(args.source_root.resolve()),
                  top_p_checkpoint=str(args.top_p_ckpt.resolve()),
                  adapter_sha256=cache_config['mapper_sha256'],
                  adapter_step=args.adapter_step if args.smoke else 20000,
                  condition_precision='f1d FP32 noTF32; frozen adapter BF16; context before position embedding',
                  upstream_root=str(args.upstream_root.resolve()),
                  pretrained_sha256=verified[0],
                  top_p_condition_audit=dict(config=cache_config,summary=cache_summary),
                  source_sha256={name: sha256(Path(__file__).with_name(name))
                                  for name in ("continuous_halton.py", "embedding_adapter.py", "data.py",
                                              "async_eval.py", "eval_worker.py",
                                              "parallel_eval.py", Path(__file__).name)})
    data = TopPProxyRouteDataset(args.source_root, args.source_root, args.cache_root)
    if len(data) != 2562334:
        raise RuntimeError("unexpected dataset size")
    config["condition_kind"] = "continuous_768D_before_positional_embedding"
    sampler = RankDynamicBucketSampler(data, args.micro, world, rank, args.seed)
    coverage = sampler.base.coverage()
    config['k_bucket_coverage'] = coverage
    config['k_bucket_dropped_total'] = sum(row['dropped'] for row in coverage.values())
    horizon = len(sampler) * args.epochs
    updates = args.updates if args.smoke and args.updates is not None else horizon
    if args.stop_after is not None:
        if args.stop_after > horizon:
            raise ValueError('stop-after exceeds schedule horizon')
        updates = args.stop_after
    if updates < 1 or updates > horizon: raise ValueError('invalid update count')
    config.update(updates=updates,stop_after=args.stop_after,schedule_horizon=horizon,updates_per_epoch=len(sampler),
                  dataset_rows=len(data),epoch_definition='one pass over both cached augmentations; drop last per K bucket',
                  smoke=args.smoke)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=False)
        atomic_json(out / "config.json", config)
    dist.barrier()
    core = HaltonProxy(args.upstream_root)
    init = load_pretrained(core, args.pretrained, transfer, verify_hash=False)
    if rank == 0:
        atomic_json(out / "initialization.json", init)
    core.to(device).train()
    official = official_training_api(args.upstream_root, core, args.lr, args.warmup, horizon)
    optimizer = official.optim
    objective_class = (MixedHaltonCompletionObjective if args.mask_objective == 'mixed-prefix'
                       else HaltonCompletionObjective)
    objective = DDP(objective_class(core, args.upstream_root), device_ids=[local], broadcast_buffers=False)
    loader = DataLoader(data, batch_sampler=sampler, collate_fn=collate_top_p_proxy_route,
                        num_workers=2, pin_memory=True, persistent_workers=True)
    stream = iter(loader)
    # Loading frozen modules must not advance the baseline masking/dropout RNG.
    with torch.random.fork_rng(devices=[local]):
        frozen = TopPFrozenAssets(ASSETS, args.top_p_ckpt, "cpu", chunk=args.micro)
        quantizer = frozen.native.quantize.to(device)
        latent_decoder = frozen.shell.latent_decoder.to(device)
        adapter, adapter_config = load_adapter(args.mapper_ckpt, device,
            cache_config["top_p_checkpoint_sha256"], args.adapter_step if args.smoke else 20000)
    config["adapter_training"] = adapter_config
    if rank == 0:
        atomic_json(out / "config.json", config)
    epoch, epoch_batch = 0, 0
    run_wandb=None;evaluator=None
    if rank == 0:
        import wandb
        run_wandb=wandb.init(project=args.wandb_project,name=out.name,dir=str(out),
                             mode='disabled' if args.smoke else 'online',config=config)
        run_wandb.define_metric('train/step')
        run_wandb.define_metric('train/*',step_metric='train/step')
        run_wandb.define_metric('eval/checkpoint_step')
        run_wandb.define_metric('eval/*',step_metric='eval/checkpoint_step')
        atomic_json(out/'wandb.json',dict(id=run_wandb.id,url=run_wandb.url,smoke=args.smoke))
        evaluator=AsyncTopPEval(out,args.eval_gpu,n=config['eval_n'],cfg_w=args.eval_cfg_w,
                                mapper_checkpoint=args.mapper_ckpt,
                                scorer_checkpoint=args.scorer_ckpt,
                                top_p_checkpoint=args.top_p_ckpt,
                                stage2_steps=args.eval_stage2_steps)
    def publish(results):
        for result in results:
            values={'eval/checkpoint_step':result['step'],'eval/epoch':result['epoch'],
                    'eval/succeeded':int(result['status']=='complete')}
            for name,metrics in result.get('metrics',{}).items():
                values[f'eval/{name}_fid5k']=metrics['fid']
            if result.get('tokens'):
                values['eval/mean_generated_2d_tokens']=result['tokens']['mean_generated_2d']
                values['eval/mean_generated_total_tokens']=result['tokens']['mean_generated_total']
            run_wandb.log(values)
            print(json.dumps(dict(evaluation=result)),flush=True)
            if result['status'] != 'complete':
                raise RuntimeError(f"asynchronous FID evaluation failed at step {result['step']}: {result}")
    if not args.smoke:
        initial_states = [None] * world if rank == 0 else None
        dist.gather_object(dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(device)),
                           initial_states, dst=0, group=metadata_group)
        if rank == 0:
            save_checkpoint(out, core, optimizer, 0, config,
                            dict(epoch=0, batch_in_epoch=0), initial_states)
            evaluator.enqueue(0, 0.0)
            publish(evaluator.poll())
        dist.barrier()
    # Save/load and sampling are exercised by a separate three-update smoke.
    started = time.monotonic()
    for step in range(1, updates + 1):
        if args.max_seconds and time.monotonic() - started > args.max_seconds:
            raise TimeoutError("explicit runtime limit reached")
        tic = time.monotonic()
        try:
            cpu = next(stream)
        except StopIteration:
            epoch += 1
            epoch_batch = 0
            sampler.set_epoch(epoch)
            stream = iter(loader)
            cpu = next(stream)
        batch = {key: value.to(device, non_blocking=True) for key, value in cpu.items()}
        epoch_batch += 1
        features = make_features(batch["z1d"], quantizer, latent_decoder)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            proxy = adapter(features)
        if proxy.requires_grad or not bool(torch.isfinite(proxy).all()):
            raise RuntimeError("invalid frozen continuous context")
        del features
        indices, valid = batch["route_indices"], batch["route_valid"]
        if args.lr_schedule == "budget-cosine":
            set_budget_cosine_lr(optimizer, step, args.lr, args.warmup,
                                 args.lr_hold_until, args.lr_decay_until, args.min_lr)
        else:
            official.args.iter = step - 1
            official.adapt_learning_rate()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, metrics = objective(proxy, batch['z2d'], batch["label"], indices, valid)
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite training loss")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(core.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if step <= 2 or step % args.log_every == 0 or step == updates:
            dist.all_reduce(metrics)
            row = dict(step=step, epoch=epoch + epoch_batch / len(sampler),
                       loss=float(metrics[0] / world), masked_nll=float(metrics[1] / metrics[2].clamp_min(1)),
                       mask_ratio=float(metrics[3] / world), class_dropout=float(metrics[4] / world),
                       selected_k=int(indices.shape[1]),
                       lr=optimizer.param_groups[0]['lr'],
                       grad_norm=float(grad_norm), seconds_per_step=time.monotonic()-tic,
                       elapsed_seconds=time.monotonic()-started,
                       peak_reserved_gib=torch.cuda.max_memory_reserved(device)/1024**3)
            atomic_json(out / f"rank{rank}.json", row)
            if rank == 0:
                with (out / "metrics.jsonl").open("a") as handle:
                    handle.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
                run_wandb.log({'train/'+key:value for key,value in row.items()})
                publish(evaluator.poll())
        eval_due=epoch_batch==len(sampler) or step in config['extra_eval_steps'] or step==updates
        if eval_due:
            states=[None]*world if rank==0 else None
            dist.gather_object(dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(device)),states,dst=0,group=metadata_group)
            if rank == 0:
                save_checkpoint(out, core, optimizer, step, config,
                                dict(epoch=epoch,batch_in_epoch=epoch_batch),states)
                if not args.skip_smoke_fid:
                    evaluator.enqueue(step,epoch+epoch_batch/len(sampler))
                publish(evaluator.poll())
            dist.barrier()
    if args.smoke:
        core.eval()
        selected = torch.zeros(len(proxy),256,dtype=torch.bool,device=device).scatter(1,indices,True)
        anchor_copy = proxy[:1][~selected[:1]].clone()
        trace=[]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            result=sample_halton_completion(core,proxy[:1],selected[:1],batch['label'][:1],
                args.upstream_root,steps=4,trace=trace)
        if not torch.equal(proxy[:1][~selected[:1]],anchor_copy) or bool((result[selected[:1]]==16384).any()):
            raise RuntimeError('smoke anchor corruption')
        atomic_json(out / f'sampling_smoke_rank{rank}.json',dict(status='passed',trace=trace))
        dist.barrier()
    frozen.assert_frozen()
    if any(p.requires_grad or p.grad is not None for p in adapter.parameters()):
        raise RuntimeError("adapter was updated in stage 2")
    if rank == 0:
        atomic_json(out / "training_complete.json", dict(status="complete", step=updates))
    dist.destroy_process_group()
    if rank == 0:
        while evaluator.busy:
            publish(evaluator.poll())
            if evaluator.busy: time.sleep(2)
        if args.smoke and not args.skip_smoke_fid:
            receipt=json.loads((out/f'eval_receipt_step{updates}.json').read_text())
            if receipt['status']!='complete': raise RuntimeError('smoke evaluation failed')
        atomic_json(out / "complete.json", dict(status="complete", step=updates))
        run_wandb.finish()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream-root", type=Path, required=True)
    p.add_argument("--pretrained", type=Path, required=True)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--cache-root", type=Path, required=True)
    p.add_argument("--top-p-ckpt", type=Path, required=True)
    p.add_argument("--mapper-ckpt", type=Path, required=True)
    p.add_argument("--scorer-ckpt", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument('--epochs',type=int,default=50)
    p.add_argument("--updates", type=int, default=None)
    p.add_argument('--stop-after',type=int,default=None,
                   help='Stop a controlled trial early without changing the epochs-based LR horizon')
    p.add_argument("--micro", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument('--lr',type=float,default=2e-5)
    p.add_argument('--warmup',type=int,default=2500)
    p.add_argument('--lr-schedule',choices=('official','budget-cosine'),default='budget-cosine')
    p.add_argument('--lr-hold-until',type=int,default=4000)
    p.add_argument('--lr-decay-until',type=int,default=20000)
    p.add_argument('--min-lr',type=float,default=2e-6)
    p.add_argument('--smoke',action='store_true')
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument('--eval-gpu',default='auto')
    p.add_argument('--eval-cfg-w',type=float,default=2.75)
    p.add_argument('--eval-stage2-steps',type=int,choices=(1,2,4,6,8,16,32),default=8)
    p.add_argument('--mask-objective',choices=('official','mixed-prefix'),default='mixed-prefix')
    p.add_argument('--extra-eval-steps',default='2000,4000,8000,12000,16000,20000',
                   help='Comma-separated update numbers for additional async 5k evals')
    p.add_argument('--wandb-project',default='mixtok-continuous-halton')
    p.add_argument("--min-free-gib", type=float, default=40.)
    p.add_argument("--memory-fraction", type=float, default=.92)
    p.add_argument("--adapter-step", type=int, default=20000)
    p.add_argument("--skip-smoke-fid", action="store_true")
    p.add_argument("--max-seconds", type=int, default=0)
    args = p.parse_args()
    run(args)
