"""Strict state/cursor restoration for the unchanged BERT sparse-2D probe."""
from __future__ import annotations
import json
from pathlib import Path
import torch
from safetensors import safe_open
from .paths import sha256

def restore_payload(core, optimizer, tensors, metadata, rank, restore_cuda=True):
    raw={k[4:]:v for k,v in tensors.items() if k.startswith("raw/")}
    core.load_state_dict(raw,strict=True)
    named=dict(core.named_parameters())
    name_by_id={id(p):n for n,p in named.items()}
    if len(optimizer.param_groups)!=len(metadata["optimizer"]):
        raise ValueError("optimizer group count mismatch")
    for group,saved in zip(optimizer.param_groups,metadata["optimizer"]):
        names=[name_by_id[id(p)] for p in group["params"]]
        if names != saved["parameter_names"]:
            raise ValueError("optimizer parameter order mismatch")
        for key,value in saved.items():
            if key=="parameter_names": continue
            group[key]=tuple(value) if key=="betas" else value
    optimizer.state.clear()
    count=0
    for name,parameter in named.items():
        prefix="adam/"+name+"/"
        state={k[len(prefix):]:v for k,v in tensors.items() if k.startswith(prefix)}
        if set(state)!={"step","exp_avg","exp_avg_sq"}:
            raise ValueError("incomplete Adam state: "+name)
        if int(state["step"].item())!=int(metadata["step"]):
            raise ValueError("Adam step mismatch: "+name)
        for key,value in state.items():
            if key!="step" and value.shape!=parameter.shape:
                raise ValueError("Adam shape mismatch: "+name)
        # Non-capturable AdamW keeps its scalar counter on CPU.
        optimizer.state[parameter]={k:v.clone().to("cpu" if k=="step" else parameter.device) for k,v in state.items()}
        count+=1
    rng_cpu=tensors[f"rng/{rank}/cpu"].clone().cpu()
    rng_cuda=tensors[f"rng/{rank}/cuda"].clone().cpu() if restore_cuda else None
    return dict(optimizer_states=count,resume_step=int(metadata["step"]),
                cursor=metadata["cursor"],model_tensors=len(raw),
                rng_cpu=rng_cpu,rng_cuda=rng_cuda)

def load_resume(core,optimizer,directory,metadata,rank):
    with safe_open(str(Path(directory)/"latest.pt"),framework="pt",device="cpu") as reader:
        tensors={k:reader.get_tensor(k) for k in reader.keys() if not k.startswith("rng/") or k.startswith(f"rng/{rank}/")}
    return restore_payload(core,optimizer,tensors,metadata,rank,restore_cuda=next(core.parameters()).is_cuda)

def restore_rng(state):
    torch.set_rng_state(state["rng_cpu"])
    if state["rng_cuda"] is not None:
        torch.cuda.set_rng_state(state["rng_cuda"])
    if not torch.equal(torch.get_rng_state(),state["rng_cpu"]):
        raise ValueError("CPU RNG restoration failed")
    if state["rng_cuda"] is not None and not torch.equal(torch.cuda.get_rng_state(),state["rng_cuda"]):
        raise ValueError("CUDA RNG restoration failed")

def resumed_stream(loader,sampler,cursor):
    epoch,skip=int(cursor["packed_pass"]),int(cursor["next_microbatch_offset"])
    while True:
        sampler.set_epoch(epoch)
        if not 0 <= skip <= len(sampler.base):
            raise ValueError("data cursor outside sampler")
        sampler.offset=skip
        for offset,batch in enumerate(loader,start=skip):
            yield batch,dict(packed_pass=epoch,next_microbatch_offset=offset+1)
        epoch+=1
        skip=0
