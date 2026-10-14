"""Frozen top-p assets for sparse 2D generation.

Only the official TiTok tokenizer and the specified top-p EMA tokenizer
components are loaded. No k-selector task checkpoint is read here.
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
TOP_P_INFERENCE_SHA256 = "e3806ee8011101a1f6bda002d18f2dd953f99b3dbfdf98d1be3126f1cb85547a"
PUBLIC_SHA256 = "457be008836e3f3702327795bb1953250f0032a0b35f74e26e6ffce9ef50e82e"
INFERENCE_PREFIXES = ("latent_decoder.", "llamagen_vq.")


def verified_top_p_artifact(path: Path) -> str:
    """Return the artifact hash after checking its known full/EMA-export identity."""
    digest = sha256(path)
    if digest not in (TOP_P_SHA256, TOP_P_INFERENCE_SHA256, PUBLIC_SHA256):
        raise RuntimeError("top-p EMA checkpoint identity differs")
    return digest


def verify_source_meta(meta: dict, root: Path) -> None:
    """Validate portable checkpoint and extraction-plan provenance."""
    keys = ("mot_source_sha256", "mot_checkpoint_sha256", "titok_checkpoint_sha256")
    if not any(key in meta for key in keys):
        return  # Historical caches predate content-hash metadata.
    if (meta.get("mot_source_sha256") != TOP_P_SHA256
            or meta.get("mot_checkpoint_sha256") not in (TOP_P_SHA256, TOP_P_INFERENCE_SHA256)
            or meta.get("titok_checkpoint_sha256") != TITOK_SHA256):
        raise RuntimeError("paired source cache checkpoint identity differs")
    if "run_config_sha256" in meta:
        config_path = Path(root) / "run_config.json"
        if sha256(config_path) != meta["run_config_sha256"]:
            raise RuntimeError("paired source run configuration hash differs")
        import json
        config = json.loads(config_path.read_text())
        if (config.get("batch_size") != meta.get("extraction_batch_size")
                or config.get("world_size") != meta.get("world_size")
                or config.get("mot_checkpoint_sha256") != meta["mot_checkpoint_sha256"]
                or config.get("titok_checkpoint_sha256") != TITOK_SHA256):
            raise RuntimeError("paired source run configuration differs from metadata")


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
        top_p_hash = verified_top_p_artifact(top_p_ckpt)
        if native_hash != TITOK_SHA256:
            raise RuntimeError("frozen TiTok checkpoint identity differs")
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
        if top_p_hash == TOP_P_INFERENCE_SHA256:
            if (checkpoint.get("format") != "top_p_ema_inference_v1"
                    or checkpoint.get("source_sha256") != TOP_P_SHA256
                    or len(state) != 661
                    or any(not key.startswith(INFERENCE_PREFIXES) for key in state)):
                raise RuntimeError("invalid inference-only top-p EMA export")
        if top_p_hash == PUBLIC_SHA256:
            inference = {k:v for k,v in state.items() if k.startswith(INFERENCE_PREFIXES)}
            if (set(checkpoint) != {"model_ema"} or len(state) != 692
                    or len(inference) != 661
                    or sum(v.numel() for v in inference.values()) != 377198731):
                raise RuntimeError("invalid public tokenizer export structure")
        counts = {}
        for prefix, module in (("latent_decoder.", latent_decoder),
                               ("llamagen_vq.", vq)):
            subset = {key[len(prefix):]: value for key, value in state.items()
                      if key.startswith(prefix)}
            module.load_state_dict(subset, strict=True)
            counts[prefix[:-1]] = len(subset)
        public_export = top_p_hash == PUBLIC_SHA256
        step = None if public_export else int(checkpoint.get("step", -1))
        if not public_export and step != 247482:
            raise RuntimeError("top-p EMA checkpoint step differs")
        del checkpoint, state, subset

        # Both training scripts consume packed 1D IDs; TiTok encoding is not
        # part of their online forward path. Keep its frozen quantizer only.
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
            "top_p_checkpoint_sha256": top_p_hash if public_export else TOP_P_SHA256,
            "training_source_identity_verified": not public_export,
            "top_p_artifact_sha256": top_p_hash,
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
