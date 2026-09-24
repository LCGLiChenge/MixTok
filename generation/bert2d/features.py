"""Portable frozen 1D feature provider for selected sparse 2D-only MaskGIT."""
from __future__ import annotations

import torch

from .paths import ASSETS
from h20_joint.assets import FrozenAssets


class PortableBaseFeatures:
    """Return frozen MoT EMA 16x16 base features from completed TiTok-L32 codes.

    The provider retains only the native TiTok quantizer and MoT EMA latent
    decoder on the training device.  No source image, 2D target, or router
    ground truth is used.
    """

    def __init__(self, device: str = "cuda", chunk: int = 16, *, assets_root=ASSETS):
        if chunk < 1:
            raise ValueError("positive frozen feature chunk required")
        loaded = FrozenAssets(assets_root, "cpu", chunk=chunk)
        if loaded.shell.latent_decoder.head_mode != "feature":
            raise ValueError("feature-mode decoder required")
        self.quantizer = loaded.native.quantize.to(device).eval().requires_grad_(False)
        self.decoder = loaded.shell.latent_decoder.to(device).eval().requires_grad_(False)
        self.audit = dict(
            loaded.audit,
            retained_modules=["native.quantize", "MoT_EMA.latent_decoder"],
            retained_parameters=sum(
                p.numel() for module in (self.quantizer, self.decoder) for p in module.parameters()
            ),
        )
        self.chunk = int(chunk)
        self._ids = None
        self._features = None
        self.calls = 0
        self.assert_frozen()

    @torch.no_grad()
    def __call__(self, z1: torch.Tensor) -> torch.Tensor:
        if z1.ndim != 2 or z1.shape[1] != 32 or z1.dtype != torch.long:
            raise ValueError("base condition requires completed [B,32] integer 1D codes")
        if bool(((z1 < 0) | (z1 >= 4096)).any()):
            raise ValueError("MASK tokens cannot enter the frozen decoder")
        if self._ids is not None and torch.equal(z1, self._ids):
            return self._features
        self.calls += 1
        outputs = []
        with torch.autocast(z1.device.type, enabled=False):
            for ids in z1.split(self.chunk):
                q = self.quantizer.get_codebook_entry(ids.reshape(-1))
                q = q.reshape(len(ids), 1, 32, -1).permute(0, 3, 1, 2).contiguous()
                outputs.append(self.decoder(q.float()).detach())
            features = torch.cat(outputs)
        if features.shape != (len(z1), 256, 16, 16) or not bool(torch.isfinite(features).all()):
            raise ValueError("invalid frozen spatial features")
        with torch.inference_mode(False):
            self._ids = z1.detach().clone()
            self._features = features.detach().clone()
        return self._features

    def assert_frozen(self) -> None:
        for module in (self.quantizer, self.decoder):
            if module.training or any(p.requires_grad or p.grad is not None for p in module.parameters()):
                raise RuntimeError("frozen feature dependency entered the training graph")
