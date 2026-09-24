"""Exact upstream Halton-MaskGIT model with our fixed-proxy completion contract."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

from h20.data import E117SparseCodeDataset, collate_e117_sparse

UPSTREAM_COMMIT = "f61b0a1314717004dc7487531fd16a8bb71e1888"
TRANSFORMER_SHA256 = "3a94ad2ac63bc085e4c6982a1b7fce28c3bb263637626e88d596c3857beed856"
PRETRAINED_SHA256 = "7fe25cb80b05743e8b42dacc61d88792a9ab5217d6bd756bef28821e0a5fc68f"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
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


class HaltonProxy(nn.Module):
    vocabulary = 16384
    mask_token = 16384

    def __init__(self, upstream_root: Path, dropout: float = 0.1):
        super().__init__()
        Transformer = upstream_transformer(upstream_root)
        self.model = Transformer(input_size=16, hidden_dim=768, codebook_size=16384,
                                 depth=12, heads=12, mlp_dim=3072, dropout=dropout,
                                 nclass=1000, register=1, proj=1)

    def forward(self, tokens: torch.Tensor, labels: torch.Tensor,
                force_drop_ids: torch.Tensor | None = None,
                output_indices: torch.Tensor | None = None) -> torch.Tensor:
        if tokens.ndim != 2 or tokens.shape[1] != 256 or tokens.dtype != torch.long:
            raise ValueError("expected 256 flattened integer tokens")
        if labels.shape != (len(tokens),):
            raise ValueError("bad class labels")
        if force_drop_ids is None:
            force_drop_ids = torch.zeros_like(labels, dtype=torch.bool)
        logits = self.model(tokens.reshape(-1, 16, 16), labels, force_drop_ids.bool())
        # Keep the MASK output class in the training softmax, exactly as upstream.
        if output_indices is not None:
            if output_indices.ndim != 2 or len(output_indices) != len(tokens):
                raise ValueError("bad output indices")
            logits = logits.gather(1, output_indices.clamp(0, 255)[..., None]
                                   .expand(-1, -1, self.vocabulary + 1))
        return logits.float()


def load_pretrained(core: HaltonProxy, path: Path, transfer_tokens: bool,
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


class ProxyRouteDataset(E117SparseCodeDataset):
    def __init__(self, packed_root: Path, route_cache: Path, proxy_root: Path, split: str):
        super().__init__(packed_root, route_cache, split=split)
        proxy_root = proxy_root.resolve()
        config = json.loads((proxy_root / "config.json").read_text())
        summary = json.loads((proxy_root / "summary.json").read_text())
        if summary.get("status") != "complete" or not summary.get("complete_coverage"):
            raise RuntimeError("proxy cache is not complete")
        if config.get("format") not in (
            "mot199440_1d_base_rgb_reencoded_proxy_v1", "f1d_mapper_proxy_v1"
        ):
            raise RuntimeError("proxy cache has wrong semantics")
        if config["format"] == "f1d_mapper_proxy_v1":
            route_meta = json.loads((self.route_cache / "meta.json").read_text())
            if route_meta.get("router_mode") != "no-xbase" or (
                route_meta.get("router_proxy_sha256") != config.get("router_proxy_sha256")
                or route_meta.get("mapper_sha256") != config.get("mapper_checkpoint_sha256")
            ):
                raise RuntimeError("mapper proxy and no-xbase route cache identities differ")
        self.proxy = np.load(proxy_root / "proxy_codes.npy", mmap_mode="r", allow_pickle=False)
        expected_shape = ((2562334, 256) if config["format"] != "f1d_mapper_proxy_v1"
                          else tuple(config["shape"]))
        if self.proxy.shape != expected_shape or self.proxy.dtype != np.uint16:
            raise RuntimeError("proxy cache shape/dtype mismatch")
        if summary.get("proxy_codes_sha256") is None:
            raise RuntimeError("proxy cache lacks final SHA256")
        self.proxy_sha256 = summary["proxy_codes_sha256"]

    def __getitem__(self, index: int):
        row = super().__getitem__(index)
        flat = int(row["source_index"]) * 2 + int(row["augmentation"])
        row["proxy"] = torch.from_numpy(np.array(self.proxy[flat], dtype=np.int64, copy=True))
        return row


def collate_proxy_route(rows: list[dict]):
    batch = collate_e117_sparse(rows)
    batch["proxy"] = torch.stack([row["proxy"] for row in rows])
    return batch
