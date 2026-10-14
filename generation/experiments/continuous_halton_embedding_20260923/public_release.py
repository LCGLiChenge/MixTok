"""Load the hash-pinned public MixTok generation and tokenizer exports.

Training-step metadata is not needed for inference. The public tokenizer omits
it, so its audit records None rather than inventing the original training step.
"""
from types import SimpleNamespace
import torch
from bert2d.paths import sha256
from bert2d.eval import OfficialSamplingView
from h20.assets import official_model
from .assets_top_p import TopPFrozenAssets, TOP_P_SHA256, PUBLIC_SHA256
from .continuous_halton import ContinuousHalton
from .halton_upstream import resolve_upstream_root
from .top_p_scorer import OneDTopPGridScorer

GENERATION_SHA256 = "9b0116b3cccf1c3033d594dfce2f7f982a9012fac590aedd17dfaeb3504e9b9d"

def load_public_release(args, assets_root, device):
    ASSETS = assets_root
    digest=sha256(args.checkpoint)
    if digest != "9b0116b3cccf1c3033d594dfce2f7f982a9012fac590aedd17dfaeb3504e9b9d":
        raise RuntimeError("public generation bundle checksum differs")
    if args.allow_adapter_override or args.adapter_step is not None:
        raise ValueError("public bundle does not support adapter overrides")
    if args.adapter_checkpoint is not None or args.scorer_checkpoint is not None:
        raise ValueError("public bundle already contains adapter and scorer; omit their checkpoint options")
    state=torch.load(args.checkpoint,map_location='cpu',mmap=True,weights_only=True)
    if state['format'] != 'mot_continuous_halton_compact_model_v1':
        raise ValueError('not the public compact generation bundle')
    expected_sources={
        'halton_checkpoint_sha256':'d42573c9511b70415160ed9288e11595d5ae022adc870020659663856e399aa6',
        'adapter_checkpoint_sha256':'9e1c0fdc1a182e6e793edf31ef1183e3ebd5e8663630b0c4cdf19b841405c207',
        'scorer_checkpoint_sha256':'52fddf824047fa9c207c046152275fc38715beb2020e0613ed9e7e73f2893a35',
        'one_d_asset_sha256':'3c5938a01658e8886d994fd9885b19d73bf4311cc5a86945d117be46d4b32233'}
    if any(state['sources'][k]!=v for k,v in expected_sources.items()):
        raise RuntimeError('bundle source lineage differs from release manifest')
    config=state['configs']['halton'];step=20000 # Verified exporter requires step 20000.
    adapter_config=state['configs']['adapter']
    if (config['adapter_sha256'] != expected_sources['adapter_checkpoint_sha256']
        or config['top_p_condition_audit']['config']['scorer_sha256'] != expected_sources['scorer_checkpoint_sha256']
        or config['top_p_condition_audit']['config']['top_p_checkpoint_sha256'] != TOP_P_SHA256
        or adapter_config['top_p_checkpoint_sha256'] != TOP_P_SHA256
        or adapter_config['official_halton_sha256'] != '7fe25cb80b05743e8b42dacc61d88792a9ab5217d6bd756bef28821e0a5fc68f'):
        raise RuntimeError('generation bundle training lineage is inconsistent')
    upstream_root=resolve_upstream_root(config)
    model=ContinuousHalton(upstream_root)
    model.load_state_dict(state['models']['halton'],strict=True)
    model=model.to(device).eval().requires_grad_(False)
    top_p_artifact_digest=sha256(args.top_p_checkpoint)
    if top_p_artifact_digest != PUBLIC_SHA256: raise RuntimeError('public tokenizer checksum differs')
    top_p_digest=top_p_artifact_digest
    assets=TopPFrozenAssets(ASSETS,args.top_p_checkpoint,device,chunk=4)
    scorer_args=state['configs']['scorer_args']
    scorer=OneDTopPGridScorer(hidden=int(scorer_args['hidden']),depth=int(scorer_args['depth']))
    scorer.load_state_dict(state['models']['scorer'],strict=True)
    scorer=scorer.to(device).eval().requires_grad_(False)
    from experiments.continuous_halton_embedding_20260923.embedding_adapter import EmbeddingAdapter
    adapter=EmbeddingAdapter(**state['configs']['adapter_model'])
    adapter.load_state_dict(state['models']['adapter'],strict=True)
    adapter=adapter.to(device).eval().requires_grad_(False)
    adapter_audit=dict(checkpoint=str(args.checkpoint.resolve()),sha256=digest,
        source_checkpoint_sha256=expected_sources['adapter_checkpoint_sha256'],
        step=20000,format=state['format'],inference_scale=args.context_scale,training_scale=1.0,
        override_of_training_adapter=False,joint_finetune=False,
        parameters=sum(p.numel() for p in adapter.parameters()))
    router_audit=dict(mode='top-p-1d-only-scorer',checkpoint=str(args.checkpoint.resolve()),sha256=digest,
        source_checkpoint_sha256=expected_sources['scorer_checkpoint_sha256'],p=args.router_p,training_p=.75,
        temperature=2.,selection='sorted probability prefix where cumulative-before < p')
    one_d,one_d_audit=official_model(ASSETS)
    for name in ('embedding_2d','pos_embedding_2d','budget_embedding','output_2d'):
        delattr(one_d,name)
    actual=one_d.state_dict(); bundled=state['models']['one_d']
    if actual.keys()!=bundled.keys() or any(not torch.equal(actual[k],bundled[k]) for k in actual):
        raise RuntimeError('bundle 1D tensors differ from pinned official generator')
    one_d.load_state_dict(bundled,strict=True)
    counts={name:sum(p.numel() for p in module.parameters()) for name,module in
        [('one_d',one_d),('halton',model),('adapter',adapter),('scorer',scorer)]}
    if counts != {'one_d':177202432,'halton':142224641,'adapter':673024,'scorer':509313}:
        raise RuntimeError('public bundle model parameter counts differ')
    one_d_audit=dict(one_d_audit,public_bundle_sha256=digest,exact_tensor_match=True,
        unused_2d_interfaces_pruned=True,removed_parameters=25382656,parameters_after=177202432)
    del state,actual,bundled
    one_d=one_d.to(device).eval().requires_grad_(False)
    view=OfficialSamplingView(one_d).eval()
    return SimpleNamespace(digest=digest, config=config, step=step, upstream_root=upstream_root, model=model, assets=assets, scorer=scorer, adapter=adapter, adapter_audit=adapter_audit, router_audit=router_audit, one_d=one_d, one_d_audit=one_d_audit, view=view, top_p_digest=top_p_digest, top_p_artifact_digest=top_p_artifact_digest)
