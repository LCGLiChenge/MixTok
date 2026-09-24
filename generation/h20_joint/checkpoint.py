"""Joint-only tensor checkpoint; legacy checkpoints cannot masquerade as joint."""
import json
import os
from pathlib import Path
import shutil
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from h20.assets import digest
from h20.training import atomic_json,optimizer_description
from h20.checkpoint import restore,stream_origin

FORMAT='titok_spatial_online_fusion_v1'


def metadata(directory):
    directory=Path(directory);result=json.loads((directory/'latest.json').read_text())
    path=directory/'latest.safetensors'
    if result['format']!=FORMAT:raise ValueError('requires a joint checkpoint, not a baseline/pilot')
    if path.stat().st_size!=result['bytes'] or digest(path)!=result['sha256']:
        raise ValueError('checkpoint hash/size mismatch')
    with safe_open(str(path),framework='pt',device='cpu') as reader:
        if reader.metadata().get('format')!=FORMAT:raise ValueError('tensor format mismatch')
    return result


def save_latest(output,core,ema,optimizer,rng_states,meta):
    if shutil.disk_usage(output).free<10*1024**3:
        raise RuntimeError('less than 10GiB checkpoint safety headroom')
    tensors={}
    for label,model in (('raw',core),('ema',ema)):
        tensors.update({label+'/'+n:p.detach().cpu().contiguous() for n,p in model.state_dict().items()})
    for name,p in core.named_parameters():
        for key,val in optimizer.state.get(p,{}).items():
            if not torch.is_tensor(val):raise TypeError('non-tensor Adam state')
            tensors['adam/'+name+'/'+key]=val.detach().cpu().contiguous()
    for rank,state in enumerate(rng_states):
        tensors[f'rng/{rank}/cpu']=state['cpu'];tensors[f'rng/{rank}/cuda']=state['cuda']
    temporary=output/'latest.safetensors.tmp'
    save_file(tensors,str(temporary),metadata={'format':FORMAT})
    with safe_open(str(temporary),framework='pt',device='cpu') as reader:
        if set(reader.keys())!=set(tensors):raise ValueError('checkpoint key mismatch')
        for name,expected in tensors.items():
            if not torch.equal(reader.get_tensor(name),expected):raise ValueError('round-trip mismatch: '+name)
    meta=dict(meta,optimizer=optimizer_description(optimizer,core),tensors=len(tensors),
              bytes=temporary.stat().st_size,sha256=digest(temporary),round_trip_exact=True,
              format=FORMAT,model_class='h20_joint.model.JointSystem',
              attention_implementation=core.generator.attention_implementation)
    os.replace(temporary,output/'latest.safetensors');atomic_json(output/'latest.json',meta)
    return {k:meta[k] for k in ('sha256','bytes','tensors','round_trip_exact','step')}
