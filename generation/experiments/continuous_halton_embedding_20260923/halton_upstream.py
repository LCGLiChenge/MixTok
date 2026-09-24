"""Pinned official Halton model initialization and training helpers."""
from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path
import sys
import types
import numpy as np
import torch
from torch import nn

UPSTREAM_COMMIT = "f61b0a1314717004dc7487531fd16a8bb71e1888"
TRANSFORMER_SHA256 = "3a94ad2ac63bc085e4c6982a1b7fce28c3bb263637626e88d596c3857beed856"
PRETRAINED_SHA256 = "7fe25cb80b05743e8b42dacc61d88792a9ab5217d6bd756bef28821e0a5fc68f"

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()

def upstream_transformer(root: Path):
    root = root.resolve()
    source = root / "Network/transformer.py"
    if sha256(source) != TRANSFORMER_SHA256:
        raise RuntimeError("Halton Transformer source hash differs from audited upstream")
    # Importing the official class preserves its parameter names and tied head.
    sys.path.insert(0, str(root))
    from Network.transformer import Transformer
    return Transformer


def load_pretrained(core, path: Path, transfer_tokens: bool,
                    verify_hash: bool = True) -> dict:
    if verify_hash and sha256(path) != PRETRAINED_SHA256:
        raise RuntimeError("Halton pretrained checkpoint SHA256 mismatch")
    checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    raw = checkpoint["model_state_dict"]
    state = {}
    for key, value in raw.items():
        while key.startswith(("module.", "_orig_mod.")):
            key = key.split(".", 1)[1]
        state[key] = value
    expected = core.model.state_dict()
    if set(state) != set(expected):
        raise RuntimeError(f"upstream state keys differ: missing={set(expected)-set(state)}, extra={set(state)-set(expected)}")
    for key in expected:
        if state[key].shape != expected[key].shape:
            raise RuntimeError(f"upstream parameter shape differs: {key}")
    if not transfer_tokens:
        state["tok_emb.weight"] = expected["tok_emb.weight"].detach().clone()
        state["head.weight"] = state["tok_emb.weight"]
        state["head.bias"] = expected["head.bias"].detach().clone()
    core.model.load_state_dict(state, strict=True)
    for key in expected:
        if transfer_tokens or key not in ("tok_emb.weight", "head.weight", "head.bias"):
            if not torch.equal(core.model.state_dict()[key], state[key]):
                raise RuntimeError(f"pretrained load differs at {key}")
    total = sum(p.numel() for p in core.parameters())
    fresh = sum(p.numel() for name, p in core.named_parameters()
                if not transfer_tokens and name in ("model.tok_emb.weight", "model.head.bias"))
    return dict(source_sha256=PRETRAINED_SHA256, upstream_commit=UPSTREAM_COMMIT,
                upstream_transformer_sha256=TRANSFORMER_SHA256,
                pretrained_parameters=total-fresh, fresh_parameters=fresh,
                total_parameters=total, transfer_token_embedding=transfer_tokens,
                tied_head=core.model.head.weight is core.model.tok_emb.weight)


def official_components(root: Path):
    upstream_transformer(root)
    from Utils.masking_scheduler import get_mask_code
    from Sampler.halton_sampler import HaltonSampler
    return get_mask_code, HaltonSampler


def official_training_api(root: Path, model, lr=1e-4, warm_up=2500, max_iter=1000000):
    """Execute upstream's three training helpers verbatim, without FID imports.

    The upstream module imports its entire CLIP/FID/dataset stack at import time.
    Selecting these method ASTs leaves the optimizer/loss/schedule code intact
    and avoids introducing unrelated evaluation dependencies into every rank.
    """
    source = Path(root) / 'Trainer/abstract_trainer.py'
    tree = ast.parse(source.read_text(), filename=str(source))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Trainer')
    names = {'get_optim', 'get_loss', 'adapt_learning_rate'}
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in cls.body} != names:
        raise RuntimeError('official training API changed')
    module = ast.Module(body=[cls], type_ignores=[])
    namespace = dict(os=os, np=np, torch=torch, nn=nn, optim=torch.optim)
    exec(compile(ast.fix_missing_locations(module), str(source), 'exec'), namespace)
    trainer = namespace['Trainer']()
    trainer.args = types.SimpleNamespace(resume=False, lr=lr, warm_up=warm_up,
                                         max_iter=max_iter, iter=0)
    trainer.optim = trainer.get_optim(model, lr, betas=(.9, .999), weight_decay=.03)
    return trainer
