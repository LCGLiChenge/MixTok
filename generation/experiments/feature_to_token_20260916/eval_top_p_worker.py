"""Fixed-cohort top-p generation FID with historical 1.907 ADM protocol."""
import argparse
import importlib.util
import json
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
    from experiments.feature_to_token_20260916.parallel_top_p_eval import cli
    raise SystemExit(cli())
import numpy as np
import torch
from bert2d.paths import ASSETS,ONE_D,EVALUATOR,atomic_json,output_path,sha256
from bert2d.eval import ImageBert,OfficialSamplingView,to_uint8
from h20.assets import official_model
from experiments.feature_to_token_20260916.assets_top_p import TopPFrozenAssets
from experiments.feature_to_token_20260916.halton_proxy import HaltonProxy
from experiments.feature_to_token_20260916.halton_official_completion import sample_halton_completion
from experiments.feature_to_token_20260916.parallel_top_p_eval import worker_batches
from experiments.feature_to_token_20260916.converter_codebook import CodebookFeatureTokenConverter
from experiments.feature_to_token_20260916.top_p_scorer_original import OneDTopPGridScorer, top_p_mask
from experiments.feature_to_token_20260916.train_proxy_converter import make_features


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
    if args.n not in (8,32,64,5000,50000): raise ValueError('unsupported evaluation size')
    if len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))!=1: raise ValueError('one isolated evaluation GPU required')
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    device=torch.device('cuda:0')
    out=output_path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    shard_id=args.worker_shard
    batches=worker_batches(args.n,args.worker_count,shard_id)
    status_path=out/f'status_shard{shard_id}.json'
    if status_path.exists(): raise FileExistsError(status_path)
    started=time.monotonic()
    atomic_json(status_path,dict(status='loading',pid=os.getpid(),completed=0))
    digest=sha256(args.checkpoint)
    state=torch.load(args.checkpoint,map_location='cpu',mmap=True,weights_only=True)
    config=state['config'];step=state['step']
    model=HaltonProxy(Path(config['upstream_root']))
    model.load_state_dict(state['model'],strict=True)
    model=model.to(device).eval().requires_grad_(False)
    del state
    top_p_digest=sha256(args.top_p_checkpoint)
    if (top_p_digest!='61e6e22acf5e4f7dcf44158518e4061a3fc9d6ca481ddc753e97f822546cb819'
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
    mapper_digest=sha256(args.mapper_checkpoint)
    if mapper_digest!=config['top_p_condition_audit']['config']['mapper_sha256']:
        raise RuntimeError('evaluation mapper differs from training proxy cache')
    mapper_state=torch.load(args.mapper_checkpoint,map_location='cpu',mmap=True,weights_only=True)
    if (mapper_state.get('format')!='feature_proxy_token_converter_codebook_v1'
            or int(mapper_state['step'])!=20000
            or mapper_state['train_config']['top_p_checkpoint_sha256']!=top_p_digest):
        raise RuntimeError('mapper is not the fresh top-p 20k codebook run')
    mapper=CodebookFeatureTokenConverter(**mapper_state['model_config']).to(device)
    native=assets.shell.llamagen_vq.quantize.get_codebook_entry(
        torch.arange(16384,device=device))
    mapper.initialize_codebook(native.float(),assets.shell.llamagen_vq.post_quant_conv)
    mapper.load_state_dict(mapper_state['model'],strict=True)
    mapper=mapper.eval().requires_grad_(False)
    mapper_audit=dict(checkpoint=str(args.mapper_checkpoint.resolve()),sha256=mapper_digest,
                      step=int(mapper_state['step']),format=mapper_state['format'],
                      parameters=sum(p.numel() for p in mapper.parameters()))
    del mapper_state
    router_audit=dict(mode='top-p-1d-only-scorer',checkpoint=str(args.scorer_checkpoint.resolve()),
                      sha256=scorer_digest,p=.75,temperature=2.,
                      selection='sorted probability prefix where cumulative-before < p')
    one_d,one_d_audit=official_model(ASSETS)
    one_d=one_d.to(device).eval().requires_grad_(False)
    view=OfficialSamplingView(one_d).eval()
    from bert2d.assets import FID,check
    for item in FID: check(ASSETS/item['local_path'],item)
    spec=importlib.util.spec_from_file_location('halton_proxy_adm',EVALUATOR)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.INCEPTION_V3_PATH=str(ASSETS/'fid/classify_image_graph_def.pb')
    tf=module.tf;tf.disable_eager_execution()
    tf.config.experimental.enable_tensor_float_32_execution(False)
    cfg=tf.ConfigProto(allow_soft_placement=True,intra_op_parallelism_threads=4,inter_op_parallelism_threads=2)
    cfg.gpu_options.allow_growth=True;cfg.gpu_options.visible_device_list='0'
    decoder_identities=dict(top_p_checkpoint_sha256=top_p_digest,
                            native_titok_checkpoint_sha256=sha256(ASSETS/'weights/tokenizer_titok_l32.bin'),
                            state_key='model_ema',top_p_step=247482)
    manifest=dict(n=args.n,step=step,checkpoint_sha256=digest,batch=8,logical_shards=4,seed=args.seed,
        stage1=ONE_D,one_d_audit=one_d_audit,router=router_audit,
        stage2={**config['sampling'],'cfg_w':args.cfg_w,'steps':args.stage2_steps},
        proxy=dict(mode='learned_f1d_mapper',**mapper_audit),
        decoder_identities=decoder_identities,primary='full_refine',
        full_refine='unselected continuous 1D features unchanged; selected directly replaced by 2D',
        no_gt_images_or_2d_used=True,source_sha256=sha256(Path(__file__)),
        reference=str(ASSETS/'fid/VIRTUAL_imagenet256_labeled.npz'))
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
                    z1=ImageBert.generate(view,condition=labels,**ONE_D)
                base=make_features(z1,assets.native.quantize,assets.shell.latent_decoder)
                selected,route_counts=top_p_mask(scorer(base.float()),.75,2.)
                if not bool(((route_counts>=1)&(route_counts<=256)).all()):
                    raise RuntimeError('unexpected top-p router budget')
                route_sum_k+=int(route_counts.sum())
                route_min_k=min(route_min_k,int(route_counts.min()))
                route_max_k=max(route_max_k,int(route_counts.max()))
                route_hist+=torch.bincount(route_counts,minlength=257).cpu().numpy()
                # The historical final 1.907 evaluator calls mapper.tokens in FP32.
                proxy=torch.cat([mapper.tokens(part.float()) for part in base.split(4)]).long()
                rgb=torch.cat([render(assets,part) for part in base.split(4)])
                torch.manual_seed(args.seed+2*int(ids[0])+1)
                trace=[]
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    completed=sample_halton_completion(
                        model, proxy, selected, labels, Path(config['upstream_root']),
                        steps=args.stage2_steps, cfg_w=args.cfg_w, trace=trace)
                if not torch.equal(completed[~selected],proxy[~selected]): raise RuntimeError('anchors changed')
                mixed=mixed_direct(assets,base,completed,selected)
                pixels=dict(base=to_uint8(rgb),full_refine=to_uint8(torch.cat([render(assets,part) for part in mixed.split(4)])))
                if offset==0:
                    np.savez_compressed(out/f'first_shard{shard}.npz',ids=ids,proxy=proxy.cpu().numpy(),
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
    p.add_argument('--seed',type=int,default=20260914)
    p.add_argument('--cfg-w',type=float,default=2.75)
    p.add_argument('--stage2-steps',type=int,choices=(1,2,4,6,8,16,32),default=8)
    p.add_argument('--scorer-checkpoint',type=Path,required=True)
    p.add_argument('--top-p-checkpoint',type=Path,required=True)
    p.add_argument('--mapper-checkpoint',type=Path,required=True)
    p.add_argument('--worker-shard',type=int,choices=range(8),required=True)
    p.add_argument('--worker-count',type=int,choices=(1,2,4,8),default=4)
    args=p.parse_args()
    try: main(args)
    except BaseException as exc:
        if args.output.exists(): atomic_json(args.output/f'failure_shard{args.worker_shard}.json',dict(error=repr(exc)))
        raise
