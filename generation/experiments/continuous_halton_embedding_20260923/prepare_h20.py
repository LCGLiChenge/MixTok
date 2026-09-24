"""Create an isolated runtime and route-only cache from a verified baseline."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(16*1024**2),b''):
            h.update(b)
    return h.hexdigest()


def write(p,obj):
    p.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')


def main(args):
    baseline=args.baseline_root.resolve()
    if not baseline.is_dir():
        raise FileNotFoundError(baseline)
    root=args.root.resolve()
    if root.exists():
        raise FileExistsError(root)
    config=json.loads((baseline/'results/halton20k_final_top_p_seed0/config.json').read_text())
    summary=json.loads((baseline/'results/halton20k_cfg_sweep_and_50k_retry1/summary.json').read_text())
    if summary['status']!='complete' or 'checkpoint_sha256' not in summary:
        raise RuntimeError('baseline evaluation is incomplete')
    checkpoint=baseline/'results/halton20k_final_top_p_seed0/latest.pt'
    if digest(checkpoint)!=summary['checkpoint_sha256']:
        raise RuntimeError('baseline checkpoint identity differs')
    old=baseline/'top_p_port/experiments/feature_to_token_20260916'
    hashes={name:digest(old/name) for name in config['source_sha256']}
    differences={name:dict(recorded=value,current=hashes[name]) for name,value in config['source_sha256'].items() if hashes[name]!=value}
    critical={'train_halton_top_p.py','assets_top_p.py','halton_proxy.py','halton_mixed_completion.py','data_top_p.py','halton_official_completion.py'}
    if critical.intersection(differences):
        raise RuntimeError(f'baseline training source changed: {differences}')
    root.mkdir(parents=True)
    shutil.copytree(baseline/'top_p_port',root/'source',symlinks=True,
                    ignore=shutil.ignore_patterns('.git','__pycache__'))
    package=root/'source/experiments/continuous_halton_embedding_20260923'
    shutil.copytree(Path(__file__).resolve().parent,package,
                    ignore=shutil.ignore_patterns('__pycache__'))
    (root/'results').mkdir()
    route_root=root/'cache/routes'
    route_root.mkdir(parents=True)
    original=baseline/'cache/final_conditions_top_p_train'
    rc=json.loads((original/'config.json').read_text())
    rs=json.loads((original/'summary.json').read_text())
    if rs['status']!='complete' or not rs['complete_coverage']:
        raise RuntimeError('baseline routes incomplete')
    route_hashes={}
    for name in ('route_indices.npy','route_k.npy'):
        route_hashes[name]=digest(original/name)
        key=name.replace('.npy','_sha256')
        if rs.get(key) and rs[key]!=route_hashes[name]:
            raise RuntimeError('route hash mismatch')
        (route_root/name).symlink_to(original/name)
    new_config=dict(format='top_p_continuous_route_cache_v1',shape=rc['shape'],
        top_p_checkpoint_sha256=rc['top_p_checkpoint_sha256'],
        source_titok_sha256=rc['source_titok_sha256'],scorer_sha256=rc['scorer_sha256'],
        p=.75,temperature=2.,source_root=rc['source_root'],
        baseline_route_config_sha256=digest(original/'config.json'),
        route_hashes=route_hashes,context='continuous adapter evaluated online; no learned proxy ID cache')
    write(route_root/'config.json',new_config)
    write(route_root/'summary.json',dict(status='complete',complete_coverage=True,
        config_sha256=digest(route_root/'config.json'),rows=2562334,**route_hashes))
    write(root/'baseline_audit.json',dict(status='verified',baseline_root=str(baseline),
        fid50k=summary['fid50k'],cfg=summary['cfg'],tokens50k=summary['tokens50k'],
        checkpoint_sha256=summary['checkpoint_sha256'],halton_config=config,
        mapper_config=json.loads((baseline/'results/mapper20k_codebook8_seed20260918/config.json').read_text()),
        runtime_source_hashes=hashes,non_training_source_changes=differences,
        baseline_task_weights_loaded=False))
    print(json.dumps(dict(status='prepared',root=str(root),fid50k=summary['fid50k'],non_training_source_changes=differences)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--baseline-root',type=Path,required=True)
    main(p.parse_args())
