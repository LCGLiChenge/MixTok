"""Fixed-cohort top-p generation evaluation with the ADM protocol."""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('USE_TF','0')
# The running trainer launches this file by path for every evaluation. Dispatch
# before Torch/TF imports so the live job can adopt parallel eval without a restart.
if __name__=='__main__' and not any(v.startswith('--worker-shard') for v in sys.argv):
    from experiments.continuous_halton_embedding_20260923.parallel_eval_compact import cli
    raise SystemExit(cli())
import numpy as np
import torch
from bert2d.paths import ASSETS,ONE_D,EVALUATOR,atomic_json,output_path,sha256
from bert2d.eval import ImageBert,OfficialSamplingView,to_uint8
from h20.assets import official_model
from experiments.continuous_halton_embedding_20260923.assets_top_p import (TopPFrozenAssets, TOP_P_SHA256, TOP_P_INFERENCE_SHA256)
from experiments.continuous_halton_embedding_20260923.continuous_halton import ContinuousHalton
from experiments.continuous_halton_embedding_20260923.halton_upstream import resolve_upstream_root
from experiments.continuous_halton_embedding_20260923.continuous_halton import sample_continuous_completion
from experiments.continuous_halton_embedding_20260923.parallel_eval import worker_batches
from experiments.continuous_halton_embedding_20260923.embedding_adapter import load_adapter, FORMAT
from experiments.continuous_halton_embedding_20260923.top_p_scorer import OneDTopPGridScorer, top_p_mask
from experiments.continuous_halton_embedding_20260923.adapter_data import make_features


def render(assets, features):
    with torch.autocast('cuda', enabled=False):
        return ((assets.shell.llamagen_vq.decoder(features.float()) + 1) * .5).clamp(0, 1)


def mixed_direct(assets, base, completed, selected):
    result=base.flatten(2).transpose(1,2).clone()
    batch,spatial=selected.nonzero(as_tuple=True)
    vq=assets.shell.llamagen_vq
    with torch.autocast('cuda',enabled=False):
        emb=vq.quantize.get_codebook_entry(completed[selected])
        projected=vq.post_quant_conv(emb.t()[None,:,None,:]).squeeze(0).squeeze(1).t()
    result[batch,spatial]=projected.to(result.dtype)
    return result.transpose(1,2).reshape_as(base)


def main(args):
    if args.n not in (8,32,64,5000,10000,50000): raise ValueError('unsupported evaluation size')
    if len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))!=1: raise ValueError('one isolated evaluation GPU required')
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    device=torch.device('cuda:0')
    out=output_path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    shard_id=args.worker_shard
    batches=worker_batches(args.n,args.worker_count,shard_id,args.reference_total)
    status_path=out/f'status_shard{shard_id}.json'
    if status_path.exists(): raise FileExistsError(status_path)
    started=time.monotonic()
    atomic_json(status_path,dict(status='loading',pid=os.getpid(),completed=0))
    digest=sha256(args.checkpoint)
    state=torch.load(args.checkpoint,map_location='cpu',mmap=True,weights_only=True)
    config=state['config'];step=state['step']
    upstream_root=resolve_upstream_root(config)
    model=ContinuousHalton(upstream_root)
    model.load_state_dict(state['model'],strict=True)
    model=model.to(device).eval().requires_grad_(False)
    del state
    top_p_artifact_digest=sha256(args.top_p_checkpoint)
    top_p_digest=TOP_P_SHA256
    if (top_p_artifact_digest not in (TOP_P_SHA256,TOP_P_INFERENCE_SHA256)
            or top_p_digest!=config['top_p_condition_audit']['config']['top_p_checkpoint_sha256']):
        raise RuntimeError('evaluation tokenizer differs from training tokenizer')
    assets=TopPFrozenAssets(ASSETS,args.top_p_checkpoint,device,chunk=4)
    scorer_digest=sha256(args.scorer_checkpoint)
    if scorer_digest!=config['top_p_condition_audit']['config']['scorer_sha256']:
        raise RuntimeError('evaluation scorer differs from training route cache')
    scorer_state=torch.load(args.scorer_checkpoint,map_location='cpu',weights_only=True)
    scorer_args=scorer_state['args']
    scorer=OneDTopPGridScorer(hidden=int(scorer_args['hidden']),
                              depth=int(scorer_args['depth'])).to(device)
    scorer.load_state_dict(scorer_state['model'],strict=True)
    scorer=scorer.eval().requires_grad_(False)
    del scorer_state
    adapter_digest=sha256(args.adapter_checkpoint)
    parent_adapter_sha=config['adapter_sha256']
    joint_finetune=bool(config.get('joint_finetune',False))
    if joint_finetune:
        if args.allow_adapter_override or adapter_digest!=config.get('joint_adapter_sha256'):
            raise RuntimeError('joint Halton requires its paired fine-tuned adapter')
        adapter_state=torch.load(args.adapter_checkpoint,map_location='cpu',mmap=True,weights_only=True)
        candidate_config=adapter_state['train_config']
        if (candidate_config.get('joint_finetune') is not True
                or candidate_config.get('parent_adapter_sha256')!=parent_adapter_sha
                or candidate_config.get('parent_halton_sha256')
                    !=config.get('parent_halton_checkpoint_sha256')):
            raise RuntimeError('joint adapter has the wrong parent model lineage')
        del adapter_state
        required_step=config['adapter_step']
    elif args.allow_adapter_override:
        adapter_state=torch.load(args.adapter_checkpoint,map_location='cpu',mmap=True,weights_only=True)
        candidate_config=adapter_state['train_config']
        if (candidate_config.get('parent_adapter_sha256')!=parent_adapter_sha
                or candidate_config.get('parent_halton_sha256')!=digest):
            raise RuntimeError('adapter override does not descend from this Halton/adapter pair')
        del adapter_state
        required_step=args.adapter_step
    else:
        if adapter_digest!=parent_adapter_sha or args.adapter_step is not None:
            raise RuntimeError('evaluation adapter differs from training checkpoint')
        required_step=config['adapter_step']
    adapter, adapter_config = load_adapter(args.adapter_checkpoint, device, top_p_digest, required_step)
    adapter_audit=dict(checkpoint=str(args.adapter_checkpoint.resolve()),sha256=adapter_digest,
                      step=required_step,format=FORMAT,
                      inference_scale=args.context_scale,training_scale=1.0,
                      override_of_training_adapter=args.allow_adapter_override,
                      joint_finetune=joint_finetune,
                      parent_training_adapter_sha256=parent_adapter_sha,
                      parameters=sum(p.numel() for p in adapter.parameters()))
    router_audit=dict(mode='top-p-1d-only-scorer',checkpoint=str(args.scorer_checkpoint.resolve()),
                      sha256=scorer_digest,p=args.router_p,training_p=.75,temperature=2.,
                      selection='sorted probability prefix where cumulative-before < p')
    one_d,one_d_audit=official_model(ASSETS)
    before=sum(p.numel() for p in one_d.parameters())
    removed=0
    for name in ('embedding_2d','pos_embedding_2d','budget_embedding','output_2d'):
        module=getattr(one_d,name,None)
        if module is None:
            raise RuntimeError(f'missing unused 1D wrapper module: {name}')
        removed+=sum(p.numel() for p in module.parameters())
        delattr(one_d,name)
    after=sum(p.numel() for p in one_d.parameters())
    if (before,removed,after)!=(202585088,25382656,177202432):
        raise RuntimeError('unexpected 1D wrapper parameter ledger after pruning')
    one_d_audit=dict(one_d_audit,unused_2d_interfaces_pruned=True,
                     removed_parameters=removed,parameters_before=before,
                     parameters_after=after)
    one_d=one_d.to(device).eval().requires_grad_(False)
    view=OfficialSamplingView(one_d).eval()
    stage1_sampling=dict(ONE_D,guidance_scale=args.stage1_guidance_scale,
                         randomize_temperature=args.stage1_randomize_temperature)
    from bert2d.assets import FID,check
    for item in FID: check(ASSETS/item['local_path'],item)
    spec=importlib.util.spec_from_file_location('halton_adm',EVALUATOR)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.INCEPTION_V3_PATH=str(ASSETS/'fid/classify_image_graph_def.pb')
    tf=module.tf;tf.disable_eager_execution()
    tf.config.experimental.enable_tensor_float_32_execution(False)
    cfg=tf.ConfigProto(allow_soft_placement=True,intra_op_parallelism_threads=4,inter_op_parallelism_threads=2)
    cfg.gpu_options.allow_growth=True;cfg.gpu_options.visible_device_list='0'
    decoder_identities=dict(top_p_checkpoint_sha256=top_p_digest,
                            top_p_artifact_sha256=top_p_artifact_digest,
                            native_titok_checkpoint_sha256=sha256(ASSETS/'weights/tokenizer_titok_l32.bin'),
                            state_key='model_ema',top_p_step=247482)
    manifest=dict(n=args.n,step=step,checkpoint_sha256=digest,batch=8,logical_shards=4,seed=args.seed,
        screen_batch_plan_total=args.reference_total,
        stage1=stage1_sampling,one_d_audit=one_d_audit,router=router_audit,
        stage2={**config['sampling'],'cfg_w':args.cfg_w,'steps':args.stage2_steps,
                'temperature':args.stage2_logit_scale,
                'logit_scale':args.stage2_logit_scale,
                'conventional_temperature':1.0/args.stage2_logit_scale},
        context=dict(mode='continuous_1d_spatial_context',precision='BF16 adapter / FP32 f1d',**adapter_audit),
        decoder_identities=decoder_identities,primary='full_refine',
        full_refine='unselected continuous 1D features unchanged; selected directly replaced by 2D',
        no_gt_images_or_2d_used=True,source_sha256=sha256(Path(__file__)),
        reference=str(ASSETS/'fid/VIRTUAL_imagenet256_labeled.npz'))
    manifest['generation_parameters'] = {name:sum(p.numel() for p in module.parameters()) for name,module in
        [('one_d',one_d),('halton',model),('adapter',adapter),('scorer',scorer)]}
    manifest['generation_parameters']['total'] = sum(manifest['generation_parameters'].values())
    manifest['physical_gpu']=os.environ['CUDA_VISIBLE_DEVICES']
    atomic_json(out/f'manifest_shard{shard_id}.json',manifest)
    arms=('base','full_refine')
    completed_count=0
    route_sum_k=0;route_min_k=256;route_max_k=0
    route_hist=np.zeros(257,dtype=np.int64)
    with torch.inference_mode(),tf.Session(config=cfg) as session:
        evaluator=module.Evaluator(session,batch_size=8)
        gpu_checked=False
        for shard,ids_all in [(shard_id,np.concatenate(batches))]:
            pools={name:np.lib.format.open_memmap(out/f'{name}_shard{shard}.npy',mode='w+',dtype=np.float32,
                   shape=(len(ids_all),2048)) for name in arms}
            offset=0
            for ids in batches:
                labels=torch.as_tensor(ids%1000,device=device)
                torch.manual_seed(args.seed+2*int(ids[0]))
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    z1=ImageBert.generate(view,condition=labels,**stage1_sampling)
                base=make_features(z1,assets.native.quantize,assets.shell.latent_decoder)
                selected,route_counts=top_p_mask(scorer(base.float()),args.router_p,2.)
                if not bool(((route_counts>=1)&(route_counts<=256)).all()):
                    raise RuntimeError('unexpected top-p router budget')
                route_sum_k+=int(route_counts.sum())
                route_min_k=min(route_min_k,int(route_counts.min()))
                route_max_k=max(route_max_k,int(route_counts.max()))
                route_hist+=torch.bincount(route_counts,minlength=257).cpu().numpy()
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    context=torch.cat([adapter(part.float()) for part in base.split(4)])
                if args.context_scale!=1.0:
                    context=context*args.context_scale
                anchors=context[~selected].clone()
                rgb=torch.cat([render(assets,part) for part in base.split(4)])
                torch.manual_seed(args.seed+2*int(ids[0])+1)
                trace=[]
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    completed=sample_continuous_completion(
                        model, context, selected, labels, upstream_root,
                        steps=args.stage2_steps, cfg_w=args.cfg_w,
                        temperature=args.stage2_logit_scale, trace=trace)
                if not torch.equal(context[~selected],anchors): raise RuntimeError('continuous anchors changed')
                mixed=mixed_direct(assets,base,completed,selected)
                pixels=dict(base=to_uint8(rgb),full_refine=to_uint8(torch.cat([render(assets,part) for part in mixed.split(4)])))
                if offset==0:
                    np.savez_compressed(out/f'first_shard{shard}.npz',ids=ids,context=context.float().cpu().numpy(),
                                        completed=completed.cpu().numpy(),selected=selected.cpu().numpy(),**pixels)
                    atomic_json(out/f'trace_shard{shard}.json',trace)
                for name,images in pixels.items():
                    kwargs={}
                    if not gpu_checked:
                        metadata=tf.RunMetadata()
                        kwargs=dict(options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),run_metadata=metadata)
                    features=session.run(evaluator.pool_features,{evaluator.image_input:images.astype(np.float32)},**kwargs).reshape(len(ids),2048)
                    if not gpu_checked:
                        devices=[d.device for d in metadata.step_stats.dev_stats if 'GPU' in d.device.upper()
                                 and any('conv' in n.node_name.lower() for n in d.node_stats)]
                        if not devices: raise RuntimeError('ADM convolutions not on GPU')
                        atomic_json(out/f'adm_gpu_shard{shard}.json',dict(devices=devices));gpu_checked=True
                    if not np.isfinite(features).all(): raise RuntimeError('nonfinite FID features')
                    pools[name][offset:offset+len(ids)]=features
                completed_count+=len(ids)
                offset+=len(ids)
                atomic_json(status_path,dict(status='running',completed=completed_count,total=len(ids_all),
                    sum_k=route_sum_k,min_k=route_min_k,max_k=route_max_k,
                    seconds=time.monotonic()-started,peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3))
            for pool in pools.values(): pool.flush()
            result=dict(status='complete',shard=shard,n=len(ids_all),ids=ids_all.tolist(),step=step,
                checkpoint_sha256=digest,anchors_unchanged=True,seconds=time.monotonic()-started,
                route_counts=dict(n=completed_count,sum_k=route_sum_k,
                                  min_k=route_min_k,max_k=route_max_k,
                                  histogram=route_hist.tolist()),
                peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
                hashes={name:sha256(out/f'{name}_shard{shard}.npy') for name in arms})
            atomic_json(out/f'shard{shard}.json',result)
            atomic_json(status_path,dict(status='complete',completed=completed_count,total=len(ids_all),
                seconds=time.monotonic()-started,peak_reserved_gib=result['peak_reserved_gib']))
        return  # the CPU coordinator verifies all workers and computes FID


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--n',type=int,default=5000)
    p.add_argument('--reference-total',type=int,default=None)
    p.add_argument('--seed',type=int,default=20260914)
    p.add_argument('--cfg-w',type=float,default=3.375)
    p.add_argument('--stage1-guidance-scale',type=float,default=4.75)
    p.add_argument('--stage1-randomize-temperature',type=float,default=9.5)
    p.add_argument('--context-scale',type=float,default=1.0)
    p.add_argument('--stage2-steps',type=int,choices=(1,2,4,6,8,16,32,48,64),default=8)
    p.add_argument('--stage2-logit-scale',type=float,default=1.0)
    p.add_argument('--router-p',type=float,default=0.75)
    p.add_argument('--scorer-checkpoint',type=Path,required=True)
    p.add_argument('--top-p-checkpoint',type=Path,required=True)
    p.add_argument('--adapter-checkpoint',type=Path,required=True)
    p.add_argument('--allow-adapter-override',action='store_true')
    p.add_argument('--adapter-step',type=int)
    p.add_argument('--worker-shard',type=int,choices=range(8),required=True)
    p.add_argument('--worker-count',type=int,choices=(1,2,4,8),default=4)
    args=p.parse_args()
    if args.n==10000 and args.reference_total!=50000:
        p.error('10k screens require --reference-total 50000')
    if args.reference_total is not None and (
            args.reference_total!=50000 or args.n not in (8,32,64,5000,10000)):
        p.error('50k-prefix mode supports 8/32/64 smoke or 5k/10k')
    if not math.isfinite(args.stage1_guidance_scale) or args.stage1_guidance_scale<=0:
        p.error('--stage1-guidance-scale must be finite and positive')
    if not math.isfinite(args.stage1_randomize_temperature) or args.stage1_randomize_temperature<0:
        p.error('--stage1-randomize-temperature must be finite and nonnegative')
    if not math.isfinite(args.context_scale) or args.context_scale<=0:
        p.error('--context-scale must be finite and positive')
    if not math.isfinite(args.stage2_logit_scale) or args.stage2_logit_scale<=0:
        p.error('--stage2-logit-scale must be finite and positive')
    if not math.isfinite(args.router_p) or not 0<args.router_p<=1:
        p.error('--router-p must be in (0, 1]')
    if args.allow_adapter_override != (args.adapter_step is not None):
        p.error('--allow-adapter-override and --adapter-step must be supplied together')
    if args.allow_adapter_override and args.adapter_step<=20000:
        p.error('overridden adapter step must exceed the original 20000')
    try: main(args)
    except BaseException as exc:
        if args.output.exists(): atomic_json(args.output/f'failure_shard{args.worker_shard}.json',dict(error=repr(exc)))
        raise
