"""Post-20k fixed-5k CFG search, followed by one 50k confirmation eval."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from bert2d.paths import ASSETS, atomic_json, output_path, sha256


def score(root: Path, cfg: float, args, identity: dict) -> dict:
    name = f"cfg_{cfg:.3f}".replace('.', 'p')
    destination = root / name
    command = [sys.executable, str(Path(__file__).with_name('parallel_eval.py')),
               '--checkpoint', str(args.checkpoint), '--output', str(destination),
               '--n', '5000', '--seed', '20260914', '--cfg-w', str(cfg),
               '--stage2-steps', '8', '--workers', '8',
               '--mapper-checkpoint', str(args.mapper_checkpoint),
               '--scorer-checkpoint', str(args.scorer_checkpoint),
               '--top-p-checkpoint', str(args.top_p_checkpoint)]
    with (root / f'{name}.log').open('w') as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    result = json.loads((destination / 'summary.json').read_text())
    if (result['status'] != 'complete' or result['n'] != 5000
            or result['checkpoint_sha256'] != identity['halton']):
        raise RuntimeError(f'5k eval failed identity: {name}')
    return {'cfg': cfg, 'fid': result['metrics']['full_refine']['fid'],
            'base_fid': result['metrics']['base']['fid'],
            'tokens': result['tokens'], 'output': str(destination)}


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    model = json.loads((args.checkpoint.parent / 'complete.json').read_text())
    if model.get('status') != 'complete' or model.get('step') != 20000:
        raise RuntimeError('formal 20k Halton training has not completed')
    identities = {name: sha256(path) for name, path in (
        ('halton', args.checkpoint), ('mapper', args.mapper_checkpoint),
        ('scorer', args.scorer_checkpoint), ('top_p', args.top_p_checkpoint),
        ('adm_reference', ASSETS / 'fid/VIRTUAL_imagenet256_labeled.npz'),
        ('adm_graph', ASSETS / 'fid/classify_image_graph_def.pb'))}
    root = output_path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    atomic_json(root / 'config.json', dict(identities=identities,
        cohort='image IDs 0..4999, labels ID modulo 1000, four logical shards',
        seed=20260914, stage1='official TiTok 8 steps, CFG 4.5 linear, temperature 9.5',
        stage2='Halton 8 steps, original CFG formula, direct selected-only replacement',
        coarse=[1., 1.5, 2., 2.5, 2.75, 3., 3.5, 4., 4.5],
        fine_rule='best coarse ±0.25 at 0.125 spacing; extend 0.5 outside if edge'))
    results = {}
    for cfg in (1., 1.5, 2., 2.5, 2.75, 3., 3.5, 4., 4.5):
        results[cfg] = score(root, cfg, args, identities)
        atomic_json(root / 'sweep_progress.json', list(results.values()))
    coarse_best = min(results, key=lambda value: results[value]['fid'])
    candidates = {round(coarse_best + delta, 3)
                  for delta in (-.25, -.125, 0., .125, .25)}
    if coarse_best == min(results):
        candidates.update({round(coarse_best - delta, 3) for delta in (.375, .5)})
    if coarse_best == max(results):
        candidates.update({round(coarse_best + delta, 3) for delta in (.375, .5)})
    for cfg in sorted(value for value in candidates if value > 0 and value not in results):
        results[cfg] = score(root, cfg, args, identities)
        atomic_json(root / 'sweep_progress.json', list(results.values()))
    route = results[coarse_best]['tokens']
    for row in results.values():
        if row['tokens'] != route:
            raise RuntimeError('fixed cohort route/token statistics changed across CFG values')
    if any(sha256(path) != identities[name] for name, path in (
            ('halton', args.checkpoint), ('mapper', args.mapper_checkpoint),
            ('scorer', args.scorer_checkpoint), ('top_p', args.top_p_checkpoint))):
        raise RuntimeError('component checkpoint changed during sweep')
    chosen = min(results, key=lambda value: results[value]['fid'])
    atomic_json(root / 'choice.json', dict(cfg=chosen, reason='minimum fixed-5k full-refine gFID',
                                         cohort_seed=20260914, identities=identities,
                                         candidates=sorted(results.values(), key=lambda row: row['cfg'])))
    destination = root / 'final_50k'
    command = [sys.executable, str(Path(__file__).with_name('parallel_eval.py')),
               '--checkpoint', str(args.checkpoint), '--output', str(destination),
               '--n', '50000', '--seed', '20260914', '--cfg-w', str(chosen),
               '--stage2-steps', '8', '--workers', '8',
               '--mapper-checkpoint', str(args.mapper_checkpoint),
               '--scorer-checkpoint', str(args.scorer_checkpoint),
               '--top-p-checkpoint', str(args.top_p_checkpoint)]
    with (root / 'final_50k.log').open('w') as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    final = json.loads((destination / 'summary.json').read_text())
    if (final['status'] != 'complete' or final['n'] != 50000
            or final['checkpoint_sha256'] != identities['halton']):
        raise RuntimeError('final 50k FID identity/completion check failed')
    atomic_json(root / 'summary.json', dict(status='complete', cfg=chosen,
                fid50k=final['metrics']['full_refine']['fid'],
                base_fid50k=final['metrics']['base']['fid'],
                tokens50k=final['tokens'], checkpoint_sha256=identities['halton']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--mapper-checkpoint', type=Path, required=True)
    parser.add_argument('--scorer-checkpoint', type=Path, required=True)
    parser.add_argument('--top-p-checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
