"""Restore raw model, EMA and named AdamW state from the audited tensor format."""
import json
from pathlib import Path
import torch
from safetensors import safe_open
from h20.assets import digest

def metadata(directory):
    directory = Path(directory)
    result = json.loads((directory/'latest.json').read_text())
    path = directory/'latest.safetensors'
    if path.stat().st_size != result['bytes'] or digest(path) != result['sha256']:
        raise ValueError('checkpoint hash/size mismatch')
    if result['format'] != 'titok_sparse_bert_clean_prefix_v1':
        raise ValueError('unsupported checkpoint format')
    return result

def restore(directory, core, ema, optimizer, meta, *, rank, world, exact_rng=False):
    names = dict(core.named_parameters())
    with safe_open(str(Path(directory)/'latest.safetensors'), framework='pt', device='cpu') as reader:
        keys = set(reader.keys())
        for label, model in [('raw',core),('ema',ema)]:
            if model is None: continue
            state = {k[len(label)+1:]:reader.get_tensor(k) for k in keys if k.startswith(label+'/')}
            model.load_state_dict(state, strict=True)
        expected_groups = meta['optimizer']
        optimizer.param_groups.clear()
        optimizer.state.clear()
        for group in expected_groups:
            g = {k:v for k,v in group.items() if k!='parameter_names'}
            g['params'] = [names[n] for n in group['parameter_names']]
            optimizer.add_param_group(g)
        for group in expected_groups:
            for name in group['parameter_names']:
                p = names[name]
                values = {}
                for key in ['step','exp_avg','exp_avg_sq']:
                    full = 'adam/'+name+'/'+key
                    if full not in keys: raise ValueError('missing optimizer tensor: '+full)
                    value = reader.get_tensor(full)
                    values[key] = value if key=='step' else value.to(p.device)
                if values['exp_avg'].shape != p.shape or values['exp_avg_sq'].shape != p.shape:
                    raise ValueError('optimizer shape mismatch: '+name)
                optimizer.state[p] = values
        if exact_rng:
            if meta['config']['world'] != world: raise ValueError('exact RNG restore requires unchanged world size')
            torch.set_rng_state(reader.get_tensor(f'rng/{rank}/cpu'))
            torch.cuda.set_rng_state(reader.get_tensor(f'rng/{rank}/cuda'))

def stream_origin(meta, *, micro, world, accumulation):
    old = meta['config']
    same_layout = (old['micro'],old['world'],old['accumulation']) == (micro,world,accumulation)
    cursor = meta['cursor']
    # A changed K-bucket microbatch layout changes ordering. Start the next
    # deterministic packed pass, explicitly recorded as a non-bitwise migration.
    return (int(cursor['packed_pass']), int(cursor['next_microbatch_offset']), True) if same_layout else (
        int(cursor['packed_pass'])+1, 0, False)
