"""Stage 1: fresh embedding adapter, official fixed targets, eight H20 GPUs."""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('USE_TF', '0')
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from bert2d.paths import ASSETS, atomic_json, output_path, sha256
from experiments.continuous_halton_embedding_20260923.assets_top_p import TopPFrozenAssets, TOP_P_SHA256
from experiments.continuous_halton_embedding_20260923.adapter_data import (
    load_data_contract, source_split, flat_rows, training_order, make_features, save_checkpoint)
from experiments.continuous_halton_embedding_20260923.embedding_adapter import (
    EmbeddingAdapter, FORMAT, OFFICIAL_SHA256, official_embedding, regression_loss)


@torch.no_grad()
def evaluate(core, z1, teacher_ids, rows, batch, device, quantizer, decoder, table, variance, rank, world):
    was_training = core.training
    core.eval()
    totals = torch.zeros(4, dtype=torch.float64, device=device)
    local_rows = rows[rank::world]
    for start in range(0, len(local_rows), batch):
        flat = local_rows[start:start + batch]
        source, view = np.divmod(flat, 2)
        ids = torch.from_numpy(np.array(z1[source, view], dtype=np.int64)).to(device)
        targets = torch.from_numpy(np.array(teacher_ids[flat], dtype=np.int64)).to(device)
        features = make_features(ids, quantizer, decoder)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            pred = core(features)
        target = F.embedding(targets, table)
        totals[0] += (pred.float() - target).square().sum() / variance
        totals[1] += pred.numel()
        totals[2] += F.cosine_similarity(pred.float(), target, dim=-1).sum()
        totals[3] += targets.numel()
    if world > 1:
        dist.all_reduce(totals)
    core.train(was_training)
    return dict(normalized_mse=float(totals[0] / totals[1]),
                cosine=float(totals[2] / totals[3]), tokens=int(totals[3]))


def run(args):
    rank, local, world = (int(os.getenv(k, d)) for k, d in
                           [('RANK', '0'), ('LOCAL_RANK', '0'), ('WORLD_SIZE', '1')])
    if not 0 < args.stop_after <= args.steps:
        raise ValueError('invalid update budget')
    torch.set_num_threads(4)
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    if 'H20' not in torch.cuda.get_device_name(device):
        raise RuntimeError('H20-only experiment')
    free, _ = torch.cuda.mem_get_info(device)
    if free < args.min_free_gib * 1024**3:
        raise RuntimeError('insufficient free GPU memory')
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    if world > 1:
        dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=15))
    out = output_path(args.output)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=False)
    if world > 1:
        dist.barrier()
    start = time.monotonic()
    status_path = out / f'status_rank{rank}.json'
    atomic_json(status_path, dict(status='loading', pid=os.getpid(), rank=rank))
    if sha256(args.top_p_ckpt) != TOP_P_SHA256:
        raise RuntimeError('top-p tokenizer mismatch')
    z1, teacher_ids, audit = load_data_contract(args.source_root, args.teacher_root, TOP_P_SHA256)
    train_sources, eval_sources = source_split(args.split_seed, args.eval_fraction)
    chosen = np.sort(np.random.default_rng(args.eval_seed).choice(eval_sources, args.eval_sources, replace=False))
    eval_rows = flat_rows(chosen)
    # Deterministic source ordering across the full training horizon.
    order = training_order(flat_rows(train_sources), args.steps * args.batch * world, args.seed, True)
    frozen = TopPFrozenAssets(args.assets_root, args.top_p_ckpt, 'cpu', chunk=args.batch)
    quantizer = frozen.native.quantize.to(device)
    decoder = frozen.shell.latent_decoder.to(device)
    table, variance = official_embedding(args.pretrained)
    table, variance = table.to(device), variance.to(device)
    core = EmbeddingAdapter().to(device)
    optimizer = torch.optim.AdamW(core.parameters(), lr=args.lr, betas=(.9, .999), weight_decay=.01)
    model = DDP(core, device_ids=[local], broadcast_buffers=False) if world > 1 else core
    config = dict(format=FORMAT, steps=args.steps, stop_after=args.stop_after,
        top_p_checkpoint_sha256=TOP_P_SHA256, official_halton_sha256=OFFICIAL_SHA256,
        micro_batch=args.batch, global_batch=args.batch * world, world=world, seed=args.seed,
        lr=args.lr, warmup=args.warmup, weight_decay=.01, model_config=core.config,
        parameters=sum(p.numel() for p in core.parameters()),
        input='f1d only', target='official Halton embedding of top-p RGB-reencoded teacher IDs',
        loss='all-256-position MSE / fixed global variance of official real-token embedding table',
        target_variance=float(variance), target_table_sha256=hashlib.sha256(table.cpu().numpy().tobytes()).hexdigest(),
        precision='frozen f1d FP32 noTF32; adapter train/eval BF16; loss FP32',
        split_seed=args.split_seed, eval_fraction=args.eval_fraction, eval_seed=args.eval_seed,
        eval_sources=args.eval_sources, train_rows_sha256=hashlib.sha256(order.tobytes()).hexdigest(),
        eval_rows_sha256=hashlib.sha256(eval_rows.tobytes()).hexdigest(),
        initialization='fresh Conv2d projection; zero-init residual last layer',
        frozen_audit=frozen.audit, data_audit=audit, smoke=args.smoke,
        source_sha256={name:sha256(Path(__file__).with_name(name)) for name in ('train_adapter.py','embedding_adapter.py')})
    wb = None
    if rank == 0:
        atomic_json(out / 'config.json', config)
        import wandb
        wb = wandb.init(project=args.wandb_project, name=out.name, dir=str(out), config=config,
                        mode='disabled' if args.smoke else args.wandb_mode)
        wb.define_metric('train/step'); wb.define_metric('train/*', step_metric='train/step')
        wb.define_metric('eval/step'); wb.define_metric('eval/*', step_metric='eval/step')
        atomic_json(out / 'wandb.json', dict(id=wb.id, url=wb.url, smoke=args.smoke))

    def publish_eval(step):
        metrics = evaluate(core, z1, teacher_ids, eval_rows, args.eval_batch, device,
                           quantizer, decoder, table, variance, rank, world)
        if rank == 0:
            atomic_json(out / f'eval_step{step}.json', dict(step=step, **metrics))
            wb.log({'eval/step':step, **{f'eval/{k}':v for k,v in metrics.items()}})
            print(json.dumps(dict(evaluation=metrics, step=step)), flush=True)
        return metrics

    initial = publish_eval(0)
    ticks = []
    for step in range(1, args.stop_after + 1):
        tick = time.monotonic()
        offset = ((step-1) * world + rank) * args.batch
        flat = order[offset:offset + args.batch]
        source, view = np.divmod(flat, 2)
        ids = torch.from_numpy(np.array(z1[source, view], dtype=np.int64)).to(device)
        target_ids = torch.from_numpy(np.array(teacher_ids[flat], dtype=np.int64)).to(device)
        features = make_features(ids, quantizer, decoder)
        progress = (step-1) / max(args.steps-1, 1)
        lr = args.lr * min(step/max(args.warmup,1),1.) * (.1+.9*.5*(1+math.cos(math.pi*progress)))
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            prediction = model(features)
        loss = regression_loss(prediction, target_ids, table, variance)
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite embedding loss')
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(core.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        values = torch.stack((loss.detach(), grad.detach()))
        if world > 1:
            dist.all_reduce(values); values /= world
        ticks.append(time.monotonic()-tick)
        status = dict(status='running', pid=os.getpid(), rank=rank, step=step,
            total=args.stop_after, loss=float(values[0]), grad_norm=float(values[1]), lr=lr,
            seconds=time.monotonic()-start, seconds_per_step=float(np.mean(ticks[-20:])),
            peak_reserved_gib=torch.cuda.max_memory_reserved(device)/1024**3)
        atomic_json(status_path, status)
        if rank == 0 and (step == 1 or step % 20 == 0 or step == args.stop_after):
            wb.log({'train/step':step, **{f'train/{k}':status[k] for k in ('loss','grad_norm','lr','seconds_per_step')}})
            print(json.dumps(status), flush=True)
        if step % args.eval_every == 0 or step == args.stop_after:
            last_eval = publish_eval(step)
        if step % args.save_every == 0 or step == args.stop_after:
            if world > 1:
                dist.barrier()
            if rank == 0:
                # Keep the small smoke adapter until the stage-2 loading smoke has
                # consumed it; the pipeline removes it after both checks pass.
                receipt = save_checkpoint(out, core, optimizer, step, config, False)
            if world > 1:
                dist.barrier()
        del features, prediction, loss
    frozen.assert_frozen()
    if table.requires_grad or table.grad is not None:
        raise RuntimeError('target embedding received gradients')
    if rank == 0:
        atomic_json(out / 'summary.json', dict(status='complete', step=step, initial_eval=initial,
            final_eval=last_eval, checkpoint=receipt, seconds=time.monotonic()-start))
        wb.finish()
    atomic_json(status_path, dict(status='complete', step=step, pid=os.getpid(), rank=rank))
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source-root','teacher-root','top-p-ckpt','pretrained','output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--assets-root', type=Path, default=ASSETS)
    p.add_argument('--steps', type=int, default=40000)
    p.add_argument('--stop-after', type=int, default=20000)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--eval-batch', type=int, default=64)
    p.add_argument('--eval-every', type=int, default=500)
    p.add_argument('--eval-sources', type=int, default=512)
    p.add_argument('--eval-fraction', type=float, default=.002)
    p.add_argument('--split-seed', type=int, default=20260901)
    p.add_argument('--eval-seed', type=int, default=20260917)
    p.add_argument('--seed', type=int, default=20260918)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--warmup', type=int, default=500)
    p.add_argument('--save-every', type=int, default=1000)
    p.add_argument('--memory-fraction', type=float, default=.92)
    p.add_argument('--min-free-gib', type=float, default=75.)
    p.add_argument('--wandb-mode',choices=('offline','online','disabled'),default='offline')
    p.add_argument('--wandb-project', default='mixtok-continuous-halton')
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    try:
        run(args)
    except BaseException as error:
        if args.output.exists():
            atomic_json(args.output / f'failure_rank{os.getenv("RANK","0")}.json', dict(error=repr(error), pid=os.getpid()))
        raise
