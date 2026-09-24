"""Frozen top-p assets for the faithful 1.907 mapper and Halton port.

Only the official TiTok tokenizer and the specified top-p EMA tokenizer
components are loaded.  No k-selector task checkpoint is read here.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from bert2d.paths import sha256
from h20_joint.assets import load_vendor
from h20_joint.mot_latent import TiTokToLlamaGenLatentDecoder

TITOK_SHA256 = "b8f0bf61e9ee1791d8b76fa723bdcb2c85a039a7d027e597f685db492935c31f"
TOP_P_SHA256 = "61e6e22acf5e4f7dcf44158518e4061a3fc9d6ca481ddc753e97f822546cb819"


class TopPFrozenAssets:
    def __init__(self, assets_root: Path, top_p_ckpt: Path, device="cpu", chunk=64):
        from omegaconf import OmegaConf

        load_vendor()
        from modeling.titok import TiTok
        from vq_model import VQ_16

        assets_root = Path(assets_root)
        top_p_ckpt = Path(top_p_ckpt)
        self.device = torch.device(device)
        self.chunk = int(chunk)
        if self.chunk < 1:
            raise ValueError("positive frozen chunk required")

        config_path = Path(__file__).resolve().parents[2] / (
            "third_party/TiTok/configs/infer/TiTok/titok_l32.yaml"
        )
        native = TiTok(OmegaConf.load(config_path))
        native_path = assets_root / "weights/tokenizer_titok_l32.bin"
        native_hash = sha256(native_path)
        top_p_hash = sha256(top_p_ckpt)
        if native_hash != TITOK_SHA256 or top_p_hash != TOP_P_SHA256:
            raise RuntimeError("frozen TiTok or top-p EMA checkpoint identity differs")
        state = torch.load(native_path, map_location="cpu", mmap=True, weights_only=True)
        state = state.get("model", state)
        native.load_state_dict(state, strict=True)
        native_tensors = len(state)
        del state

        latent_decoder = TiTokToLlamaGenLatentDecoder(native.decoder)
        vq = VQ_16(codebook_size=16384, codebook_embed_dim=8,
                   codebook_show_usage=False)
        checkpoint = torch.load(top_p_ckpt, map_location="cpu", mmap=True,
                                weights_only=True)
        if "model_ema" not in checkpoint:
            raise ValueError("top-p checkpoint has no model_ema")
        state = checkpoint["model_ema"]
        counts = {}
        for prefix, module in (("latent_decoder.", latent_decoder),
                               ("llamagen_vq.", vq)):
            subset = {key[len(prefix):]: value for key, value in state.items()
                      if key.startswith(prefix)}
            module.load_state_dict(subset, strict=True)
            counts[prefix[:-1]] = len(subset)
        step = int(checkpoint.get("step", -1))
        if step != 247482:
            raise RuntimeError("top-p EMA checkpoint step differs")
        del checkpoint, state, subset

        # Both training scripts consume packed 1D IDs; TiTok encoding is not
        # part of their online forward path.  Keep its frozen quantizer only.
        native.encoder = nn.Identity()
        native.latent_tokens = None
        self.native = native.to(self.device).eval().requires_grad_(False)
        shell = nn.Module()
        shell.latent_decoder = latent_decoder
        shell.llamagen_vq = vq
        self.shell = shell.to(self.device).eval().requires_grad_(False)
        self.audit = {
            "top_p_checkpoint": str(top_p_ckpt.resolve()),
            "top_p_state_key": "model_ema",
            "top_p_step": step,
            "top_p_checkpoint_sha256": top_p_hash,
            "loaded_tensors": counts,
            "native_titok_checkpoint": str(native_path.resolve()),
            "native_titok_sha256": native_hash,
            "native_titok_tensors": native_tensors,
            "k_selector_weights_loaded": False,
        }

    def assert_frozen(self):
        for module in (self.native, self.shell):
            if any(part.training for part in module.modules()):
                raise RuntimeError("frozen tokenizer entered train mode")
            if any(parameter.requires_grad or parameter.grad is not None
                   for parameter in module.parameters()):
                raise RuntimeError("frozen tokenizer received gradients")
