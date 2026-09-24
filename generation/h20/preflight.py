"""Fail before training for missing assets, incompatible versions or GPU leases."""
import argparse
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import numpy as np
from h20.assets import verify

def environment():
    import torch, transformers
    expected={'torch':'2.10.0','torchvision':'0.25.0','numpy':'1.26.2',
              'omegaconf':'2.3.0','safetensors':'0.7.0','wandb':'0.25.0'}
    for name, version in expected.items():
        if metadata.version(name).split('+')[0] != version:
            raise RuntimeError(f'{name} must be {version}; run install_h20_environment.sh')
    direct=metadata.distribution('transformers').read_text('direct_url.json')
    if not direct or json.loads(direct).get('vcs_info',{}).get('commit_id') != 'a957b7911a758d54597914b4479fe6e81424d64f':
        raise RuntimeError('transformers must be the pinned Git commit, not an arbitrary release')
    return dict(torch=torch.__version__,cuda=torch.version.cuda,transformers=transformers.__version__)

def devices(allow_non_h20=False):
    import torch
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    if not visible or any(not s.isdigit() for s in visible) or len(set(visible))!=len(visible):
        raise ValueError('set explicit distinct numeric CUDA_VISIBLE_DEVICES; never claim all cards automatically')
    if len(visible)!=torch.cuda.device_count() or len(visible) not in [1,2,4,8]:
        raise ValueError('supported world sizes: 1, 2, 4, 8')
    info=[]
    for local, physical in enumerate(visible):
        name=torch.cuda.get_device_name(local)
        if not allow_non_h20 and 'H20' not in name:
            raise ValueError('this handoff targets H20; test other GPUs explicitly with --allow-non-h20')
        if torch.cuda.get_device_capability(local)[0]<8:
            raise ValueError('bf16-capable GPU required')
        pids=subprocess.check_output(['nvidia-smi','--id='+physical,'--query-compute-apps=pid',
                                      '--format=csv,noheader,nounits'],text=True).splitlines()
        if any(s.strip().isdigit() and int(s)!=os.getpid() for s in pids):
            raise RuntimeError('selected GPU already has a compute process: '+physical)
        info.append(dict(physical=physical,name=name,total_gib=torch.cuda.get_device_properties(local).total_memory/1024**3))
    return info

def data(root):
    from h20.data import E117SparseCodeDataset
    result={}
    for split,n,aug in [('train',1281167,2),('val',50000,1)]:
        packed=Path(root)/'codes'/split; routes=Path(root)/'routes'/split
        pm=json.loads((packed/'meta.json').read_text()); rm=json.loads((routes/'meta.json').read_text())
        if not pm['completed'] or not rm['completed'] or pm['num_samples']!=n or pm['num_aug']!=aug:
            raise ValueError('incomplete/wrong packed dataset')
        if rm['num_samples']!=n or rm['num_aug']!=aug or rm['e117_checkpoint_sha256']!='a5b84689d2b29f579d2442da7594ac093292b6386760867a0668ca02f82e6156':
            raise ValueError('wrong E117 route provenance')
        if pm['mot_state_key']!='model_ema' or pm['mot_step']!=199440: raise ValueError('wrong decoder codebook')
        if not np.load(packed/'written.npy',mmap_mode='r').all() or not np.load(routes/'written.npy',mmap_mode='r').all():
            raise ValueError('unwritten cache entries')
        source=np.load(routes/'source_indices.npy',mmap_mode='r')
        if not np.array_equal(source,np.arange(n)): raise ValueError('source ordering mismatch')
        d=E117SparseCodeDataset(packed,routes,split='all')
        if len(d)!=n*aug: raise ValueError('wrong full dataset length')
        if d.titok.shape!=(n,aug,32) or d.llamagen.shape!=(n,aug,256) or d.labels.shape!=(n,):
            raise ValueError('packed array shape mismatch')
        result[split]=dict(sources=n,augmentations=aug,examples=len(d))
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--environment-only',action='store_true')
    p.add_argument('--assets',type=Path)
    p.add_argument('--allow-non-h20',action='store_true')
    args=p.parse_args(); result={'environment':environment()}
    if not args.environment_only:
        if args.assets is None: p.error('--assets is required')
        result['verified_assets']=verify(args.assets,('train',))
        result['data']=data(args.assets)
        result['devices']=devices(args.allow_non_h20)
        import wandb
        if os.environ.get('WANDB_MODE','online')!='online': raise ValueError('W&B online required')
        if not wandb.login(): raise RuntimeError('run wandb login first; never put the key in this repo')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
