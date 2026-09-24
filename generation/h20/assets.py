"""Pinned, relocatable assets; no original-server paths are required."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(16*1024**2), b''): h.update(block)
    return h.hexdigest()

def specifications():
    return json.loads((ROOT/'configs/h20_assets.json').read_text())['files']

def verify(root, profiles=('train', 'resume')):
    root = Path(root)
    checked = 0
    for spec in specifications():
        if spec['profile'] not in profiles: continue
        path = root/spec['local_path']
        if not path.is_file(): raise FileNotFoundError(path)
        if path.stat().st_size != spec['size'] or digest(path) != spec['sha256']:
            raise RuntimeError('asset identity mismatch: '+str(path))
        checked += 1
    if not checked: raise ValueError('no matching assets')
    return checked

def official_model(root):
    import os, sys
    os.environ.setdefault('USE_TF', '0')
    sys.path.insert(0, str(ROOT/'third_party/TiTok'))
    import torch
    from omegaconf import OmegaConf
    from modeling.maskgit import ImageBert
    from h20.model import TiTokSparseImageBert, load_official_state
    reference = ImageBert(OmegaConf.load(ROOT/'third_party/TiTok/configs/infer/TiTok/titok_l32.yaml'))
    core = TiTokSparseImageBert(attention_implementation=reference.model.config._attn_implementation)
    state = torch.load(Path(root)/'weights/generator_titok_l32.bin', map_location='cpu',
                       mmap=True, weights_only=True)
    report = load_official_state(core, state, reference.model.config)
    return core, report

def materialize(spec, destination, fetch):
    """Transport shards are concatenated byte-for-byte, never torch-converted."""
    import shutil
    if not spec.get('repo_id'):
        raise FileNotFoundError('This asset has no anonymous remote source; place it under --root at '+spec['local_path'])
    destination=Path(destination)
    if destination.exists():raise FileExistsError(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_name(destination.name+'.download')
    parts=spec.get('parts',[spec])
    with temporary.open('wb') as output:
        for part in parts:
            source=Path(fetch(repo_id=spec['repo_id'],filename=part['filename'],revision=spec['revision']))
            if source.stat().st_size!=part['size'] or digest(source)!=part['sha256']:
                raise RuntimeError('downloaded part hash mismatch: '+part['filename'])
            with source.open('rb') as f:shutil.copyfileobj(f,output,length=16*1024**2)
    if temporary.stat().st_size!=spec['size'] or digest(temporary)!=spec['sha256']:
        raise RuntimeError('assembled asset identity mismatch: '+spec['local_path'])
    temporary.replace(destination)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--profile', choices=['train','resume','joint','all'], default='train')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    profiles = ['train'] + (['resume'] if args.profile in ['resume','all'] else [])
    if args.profile in ('joint','all'): profiles.append('inference')
    if not args.verify_only:
        from huggingface_hub import hf_hub_download
        for spec in specifications():
            if spec['profile'] not in profiles: continue
            destination = args.root/spec['local_path']
            if destination.exists():
                if destination.stat().st_size != spec['size'] or digest(destination) != spec['sha256']:
                    raise RuntimeError('refusing to overwrite mismatched asset: '+str(destination))
                continue
            materialize(spec,destination,hf_hub_download)
    print(json.dumps({'status':'verified','files':verify(args.root,profiles),'root':str(args.root.resolve())}))

if __name__=='__main__': main()
