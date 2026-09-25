"""Build the top-p route-only cache for continuous embedding generation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from bert2d.paths import atomic_json, sha256
from experiments.continuous_halton_embedding_20260923.adapter_data import make_features
from experiments.continuous_halton_embedding_20260923.assets_top_p import TopPFrozenAssets, TOP_P_SHA256, verified_top_p_artifact, verify_source_meta
from experiments.continuous_halton_embedding_20260923.data import FORMAT, NUM_SOURCES, NUM_AUG
from experiments.continuous_halton_embedding_20260923.top_p_scorer import OneDTopPGridScorer, top_p_mask

NUM_ROWS = NUM_SOURCES * NUM_AUG


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def source_codes(root):
    meta = json.loads((root / 'meta.json').read_text())
    verify_source_meta(meta, root)
    require(meta.get('completed') is True and int(meta.get('num_samples', -1)) == NUM_SOURCES,
            'paired top-p source cache is incomplete')
    require(int(meta.get('num_aug', -1)) == NUM_AUG and int(meta.get('mot_step', -1)) == 247482
            and meta.get('mot_state_key') == 'model_ema', 'source cache provenance differs')
    codes = np.load(root / 'titok_codes.npy', mmap_mode='r', allow_pickle=False)
    require(codes.shape == (NUM_SOURCES, NUM_AUG, 32) and codes.dtype == np.uint16,
            '1D cache shape or dtype differs')
    return codes


def prepare(args):
    codes = source_codes(args.source_root)
    require(args.p == .75 and args.temperature == 2., 'use accepted p=.75 and T=2')
    top_p_artifact_sha256 = verified_top_p_artifact(args.top_p_ckpt)
    scorer_state = torch.load(args.scorer_ckpt, map_location='cpu', weights_only=True)
    require(scorer_state.get('step') == 20000 and 'model' in scorer_state and 'args' in scorer_state,
            'accepted 20k scorer checkpoint required')
    del scorer_state
    config = dict(format=FORMAT, shape=[NUM_ROWS, 256], dtype='uint16',
                  flat_layout='flat_id=source_index*num_aug+augmentation',
                  p=args.p, temperature=args.temperature, world_size=args.world_size,
                  batch_size=args.batch,
                  source_root=str(args.source_root.resolve()),
                  source_meta_sha256=sha256(args.source_root / 'meta.json'),
                  source_titok_sha256=sha256(args.source_root / 'titok_codes.npy'),
                  source_gt2d_sha256=sha256(args.source_root / 'llamagen_codes.npy'),
                  source_labels_sha256=sha256(args.source_root / 'labels.npy'),
                  source_shape=list(codes.shape),
                  top_p_checkpoint_sha256=TOP_P_SHA256,
                  top_p_artifact_sha256=top_p_artifact_sha256,
                  scorer_sha256=sha256(args.scorer_ckpt))
    args.output.mkdir(parents=True, exist_ok=False)
    for name, shape in (('route_indices.npy', (NUM_ROWS, 256)), ('route_k.npy', (NUM_ROWS,))):
        array = np.lib.format.open_memmap(args.output / name, mode='w+', dtype=np.uint16, shape=shape)
        array.flush()
    for rank in range(args.world_size):
        count = len(range(rank, NUM_ROWS, args.world_size))
        written = np.lib.format.open_memmap(args.output / f'written_rank{rank}.npy',
                                             mode='w+', dtype=np.bool_, shape=(count,))
        written[:] = False
        written.flush()
    atomic_json(args.output / 'config.json', config)
    print(json.dumps(dict(status='prepared', rows=NUM_ROWS, output=str(args.output))), flush=True)


def load_models(args, config, device):
    require(verified_top_p_artifact(args.top_p_ckpt) ==
            config.get('top_p_artifact_sha256', TOP_P_SHA256)
            and sha256(args.scorer_ckpt) == config['scorer_sha256'],
            'frozen checkpoint identity changed after prepare')
    frozen = TopPFrozenAssets(args.assets_root, args.top_p_ckpt, device, chunk=args.batch)
    state = torch.load(args.scorer_ckpt, map_location='cpu', weights_only=True)
    scorer = OneDTopPGridScorer(hidden=int(state['args']['hidden']),
                                depth=int(state['args']['depth'])).to(device)
    scorer.load_state_dict(state['model'], strict=True)
    return frozen, scorer.eval().requires_grad_(False)


@torch.inference_mode()
def extract(args):
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    device = torch.device('cuda', 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = json.loads((args.output / 'config.json').read_text())
    require(config['format'] == FORMAT and config['world_size'] == args.world_size
            and config.get('batch_size') == args.batch
            and config['p'] == args.p and config['temperature'] == args.temperature,
            'route configuration or extraction batch size changed')
    source = source_codes(args.source_root)
    require(sha256(args.source_root / 'titok_codes.npy') == config['source_titok_sha256'],
            '1D cache identity changed after prepare')
    route = np.load(args.output / 'route_indices.npy', mmap_mode='r+', allow_pickle=False)
    route_k = np.load(args.output / 'route_k.npy', mmap_mode='r+', allow_pickle=False)
    written = np.load(args.output / f'written_rank{args.rank}.npy', mmap_mode='r+', allow_pickle=False)
    flat_ids = np.arange(args.rank, NUM_ROWS, args.world_size, dtype=np.int64)
    require(written.shape == (len(flat_ids),), 'rank partition changed')
    pending = np.flatnonzero(~np.asarray(written))
    frozen, scorer = load_models(args, config, device)
    started = time.monotonic()
    for begin in range(0, len(pending), args.batch):
        local = pending[begin:begin + args.batch]
        rows = flat_ids[local]
        image, view = np.divmod(rows, NUM_AUG)
        ids = torch.from_numpy(np.array(source[image, view], dtype=np.int64)).to(device)
        features = make_features(ids, frozen.native.quantize, frozen.shell.latent_decoder)
        selected, k_tensor = top_p_mask(scorer(features.float()), args.p, args.temperature)
        for j, row in enumerate(rows):
            chosen = torch.nonzero(selected[j], as_tuple=False).flatten().cpu().numpy()
            k = int(k_tensor[j])
            require(k == len(chosen) and 1 <= k <= 256, 'invalid top-p K')
            route[row, :k] = chosen.astype(np.uint16, copy=False)
            route_k[row] = k
        written[local] = True
        done = begin + len(local)
        if done % (args.batch * args.flush_every) == 0 or done == len(pending):
            route.flush(); route_k.flush(); written.flush()
            atomic_json(args.output / f'status_rank{args.rank}.json',
                        dict(status='running', rank=args.rank, completed=int(written.sum()),
                             rank_total=len(written), elapsed_seconds=time.monotonic()-started,
                             peak_reserved_gib=torch.cuda.max_memory_reserved(device)/1024**3))
    route.flush(); route_k.flush(); written.flush()
    frozen.assert_frozen()
    atomic_json(args.output / f'status_rank{args.rank}.json',
                dict(status='complete', rank=args.rank, completed=int(written.sum()),
                     rank_total=len(written), elapsed_seconds=time.monotonic()-started))


def finalize(args):
    config_path = args.output / 'config.json'
    config = json.loads(config_path.read_text())
    require(config['format'] == FORMAT and config['world_size'] == args.world_size,
            'route configuration changed')
    for rank in range(args.world_size):
        written = np.load(args.output / f'written_rank{rank}.npy', mmap_mode='r', allow_pickle=False)
        require(bool(written.all()), f'rank {rank} did not finish every row')
    route = np.load(args.output / 'route_indices.npy', mmap_mode='r', allow_pickle=False)
    route_k = np.load(args.output / 'route_k.npy', mmap_mode='r', allow_pickle=False)
    require(route.shape == (NUM_ROWS, 256) and route_k.shape == (NUM_ROWS,)
            and route.dtype == route_k.dtype == np.uint16, 'route layout differs')
    histogram = np.zeros(257, dtype=np.int64)
    for begin in range(0, NUM_ROWS, 8192):
        end = min(begin + 8192, NUM_ROWS)
        r = np.asarray(route[begin:end]); k = np.asarray(route_k[begin:end])
        require(bool(np.all((k >= 1) & (k <= 256))), 'route K outside [1,256]')
        valid = np.arange(256)[None, :] < k[:, None]
        require(bool(np.all(r[valid] < 256)), 'route index outside grid')
        pairs = np.arange(255)[None, :] < (k[:, None] - 1)
        require(not bool(np.any((np.diff(r.astype(np.int32), axis=1) <= 0) & pairs)),
                'selected route indices are not strictly sorted')
        histogram += np.bincount(k.astype(np.int64), minlength=257)
    require(int(histogram.sum()) == NUM_ROWS, 'route coverage differs')
    summary = dict(status='complete', complete_coverage=True, rows=NUM_ROWS,
                   mean_k=float(route_k.mean()), min_k=int(route_k.min()), max_k=int(route_k.max()),
                   k_histogram=histogram.tolist(), scorer_sha256=config['scorer_sha256'],
                   top_p_checkpoint_sha256=config['top_p_checkpoint_sha256'],
                   route_indices_sha256=sha256(args.output / 'route_indices.npy'),
                   route_k_sha256=sha256(args.output / 'route_k.npy'),
                   config_sha256=sha256(config_path))
    atomic_json(args.output / 'summary.json', summary)
    print(json.dumps(dict(status='complete', rows=NUM_ROWS, mean_k=summary['mean_k'])), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'extract', 'finalize'))
    parser.add_argument('--assets-root', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--top-p-ckpt', type=Path, required=True)
    parser.add_argument('--scorer-ckpt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--world-size', type=int, default=8)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--flush-every', type=int, default=25)
    parser.add_argument('--p', type=float, default=.75)
    parser.add_argument('--temperature', type=float, default=2.)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    require(args.world_size > 0 and 0 <= args.rank < args.world_size
            and args.batch > 0 and args.flush_every > 0, 'invalid rank or batch')
    if args.mode == 'prepare':
        prepare(args)
    elif args.mode == 'extract':
        extract(args)
    else:
        finalize(args)
