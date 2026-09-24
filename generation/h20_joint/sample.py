"""Same-token paired PNG previews from a joint checkpoint. This does not calculate FID."""
import argparse
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from PIL import Image
from h20.training import atomic_json
from h20_joint.checkpoint import metadata
from h20_joint.model import create
from h20_joint.assets import FrozenAssets
from h20_joint.objective import online_conditions
from h20_joint.evaluation import images


@torch.no_grad()
def main(args):
    if args.output.exists():raise FileExistsError('new preview output required')
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required')
    torch.cuda.set_device(0);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    meta=metadata(args.checkpoint);system,_=create(args.assets,meta['config']['memory'])
    with safe_open(str(args.checkpoint/'latest.safetensors'),framework='pt',device='cpu') as reader:
        prefix=args.state+'/'
        system.load_state_dict({k[len(prefix):]:reader.get_tensor(k) for k in reader.keys() if k.startswith(prefix)},strict=True)
    system=system.cuda().eval().requires_grad_(False)
    provider=FrozenAssets(args.assets,torch.device('cuda',0),chunk=4)
    system.generator.set_feature_provider(provider.features)
    args.output.mkdir(parents=True)
    records=[]
    for first in range(0,args.num_images,args.batch):
        size=min(args.batch,args.num_images-first)
        labels=(torch.arange(first,first+size,device='cuda')*37+9)%1000
        packet=online_conditions(system.generator,provider,labels,args.seed+first*100)
        rendered,alpha=images(system,provider,packet)
        for arm,values in rendered.items():
            folder=args.output/arm;folder.mkdir(exist_ok=True)
            pixels=values.clamp(0,1).mul(255).round().byte().permute(0,2,3,1).cpu().numpy()
            for i,pixel in enumerate(pixels):Image.fromarray(pixel).save(folder/f'{first+i:06d}.png')
        records.append(dict(first=first,labels=labels.cpu().tolist(),mean_alpha=float(alpha.sum()/packet['valid'].sum()),
                            k=packet['valid'].sum(1).cpu().tolist()))
        np.savez_compressed(args.output/f'tokens_{first:06d}.npz',ids=packet['ids'].cpu().numpy(),
            codes=packet['codes'].cpu().numpy(),index=packet['index'].cpu().numpy(),valid=packet['valid'].cpu().numpy())
    atomic_json(args.output/'manifest.json',dict(checkpoint_sha256=meta['sha256'],step=meta['step'],
        state=args.state,seed=args.seed,batch=args.batch,num_images=args.num_images,
        paired_same_tokens=True,fid_calculated=False,records=records))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--state',choices=('raw','ema'),default='ema')
    p.add_argument('--num-images',type=int,default=16);p.add_argument('--batch',type=int,default=2)
    p.add_argument('--seed',type=int,default=20260911)
    args=p.parse_args()
    if not 1<=args.num_images<=50000 or args.batch<1:p.error('invalid sample budget')
    main(args)
