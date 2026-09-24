import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.modules.module import _IncompatibleKeys
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from safetensors.torch import load_file as load_safetensors_file
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import models, transforms
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.datasets import ImageFolder

from models import TiTokLlamaGenStage2


DINO_V1_MODEL_ALIASES = {"dinov1_vits16", "dino_vits16", "dino_deitsmall16"}
DEFAULT_DINO_V1_CKPT = "../.cache/torch/hub/checkpoints/dino_deitsmall16_pretrain.pth"


def add_path(path):
    path = str(path)
    if path not in sys.path:
        sys.path.insert(0, path)


def load_titok(titok_root, config_path, ckpt_path, device):
    add_path(titok_root)
    from modeling.titok import TiTok

    config = OmegaConf.load(config_path)
    tokenizer = TiTok(config)
    if os.environ.get("MOT_SKIP_TITOK_INITIAL_WEIGHTS") == "1":
        tokenizer.to(device)
        tokenizer.eval()
        return tokenizer
    if str(ckpt_path).endswith(".safetensors"):
        state = load_safetensors_file(str(ckpt_path), device="cpu")
    else:
        state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
    tokenizer.load_state_dict(state, strict=True)
    tokenizer.to(device)
    tokenizer.eval()
    return tokenizer


def load_llamagen_vq(llamagen_root, ckpt_path, device, codebook_size=16384, codebook_embed_dim=8):
    add_path(llamagen_root)
    from tokenizer.tokenizer_image.vq_model import VQ_models

    vq = VQ_models["VQ-16"](codebook_size=codebook_size, codebook_embed_dim=codebook_embed_dim)
    if os.environ.get("MOT_SKIP_LLAMAGEN_INITIAL_WEIGHTS") == "1":
        vq.to(device)
        vq.eval().requires_grad_(False)
        return vq
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "ema" in ckpt:
        state = ckpt["ema"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt
    vq.load_state_dict(state, strict=True)
    vq.to(device)
    vq.eval().requires_grad_(False)
    return vq


class ConvNeXtPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.convnext_small(weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1).eval()
        self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])
        self.requires_grad_(False)

    def forward(self, input_01, target_01):
        input_01 = F.interpolate(input_01, size=224, mode="bilinear", align_corners=False, antialias=True)
        target_01 = F.interpolate(target_01, size=224, mode="bilinear", align_corners=False, antialias=True)
        pred = self.model((input_01 - self.imagenet_mean) / self.imagenet_std)
        target = self.model((target_01 - self.imagenet_mean) / self.imagenet_std)
        return F.mse_loss(pred, target, reduction="mean")


def build_perceptual_loss(args, device):
    if args.perceptual_loss == "none" or get_perceptual_weight(args) <= 0.0:
        return None
    if args.perceptual_loss == "lpips":
        add_path(args.llamagen_root)
        from tokenizer.tokenizer_image.lpips import LPIPS

        loss = LPIPS().to(device)
    elif args.perceptual_loss == "convnext_s":
        loss = ConvNeXtPerceptualLoss().to(device)
    else:
        raise ValueError(f"unsupported perceptual_loss {args.perceptual_loss}")
    loss.eval().requires_grad_(False)
    return loss


def get_perceptual_weight(args):
    return args.lambda_lpips if args.lambda_perceptual is None else args.lambda_perceptual


def compute_image_loss(pred, target, loss_type):
    if loss_type == "l1":
        return F.l1_loss(pred.float(), target.float())
    if loss_type == "l2":
        return F.mse_loss(pred.float(), target.float())
    raise ValueError(f"unsupported image_loss {loss_type}")


def compute_perceptual_loss(perceptual, perceptual_name, pred, target):
    if perceptual is None:
        return pred.new_zeros(())
    if perceptual_name == "lpips":
        return perceptual(pred.float(), target.float()).mean()
    if perceptual_name == "convnext_s":
        pred_01 = image_to_zero_one(pred, "minus1_1")
        target_01 = image_to_zero_one(target, "minus1_1")
        return perceptual(pred_01, target_01)
    raise ValueError(f"unsupported perceptual_loss {perceptual_name}")


def freeze_module(module):
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)


def set_requires_grad(module, requires_grad):
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad_(requires_grad)


class MultiScalePatchDiscriminator(nn.Module):
    def __init__(self, discriminator_cls, scales, loss_weights, hidden_channels=128, num_stages=3):
        super().__init__()
        if len(scales) != len(loss_weights):
            raise ValueError(f"disc_scales and disc_loss_weights length mismatch: {scales} vs {loss_weights}")
        if not scales:
            raise ValueError("disc_scales must not be empty")
        self.scales = tuple(float(scale) for scale in scales)
        self.loss_weights = tuple(float(weight) for weight in loss_weights)
        if any(scale <= 0.0 for scale in self.scales):
            raise ValueError(f"disc_scales must be positive, got {self.scales}")
        if any(weight <= 0.0 for weight in self.loss_weights):
            raise ValueError(f"disc_loss_weights must be positive, got {self.loss_weights}")
        self.discriminators = nn.ModuleList(
            discriminator_cls(hidden_channels=hidden_channels, num_stages=num_stages) for _ in self.scales
        )

    def forward(self, x):
        outputs = []
        for disc, scale, weight in zip(self.discriminators, self.scales, self.loss_weights):
            if abs(scale - 1.0) < 1e-8:
                x_scale = x
            else:
                size = (max(1, int(round(x.shape[-2] * scale))), max(1, int(round(x.shape[-1] * scale))))
                x_scale = F.interpolate(x, size=size, mode="bilinear", align_corners=False, antialias=True)
            outputs.append((disc(x_scale), weight))
        return outputs

    def load_primary_state_dict(self, state_dict, strict=True):
        return self.discriminators[0].load_state_dict(state_dict, strict=strict)


def parse_float_sequence(value, default):
    if value is None:
        return list(default)
    if isinstance(value, str):
        return [float(part.strip()) for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        return [float(part) for part in value]
    return [float(value)]


def spectral_linear(in_features, out_features):
    return nn.utils.spectral_norm(nn.Linear(in_features, out_features))


def load_dino_backbone(dino_repo, dino_model, dino_ckpt=None):
    dino_model = str(dino_model)
    if dino_model in DINO_V1_MODEL_ALIASES:
        import timm

        backbone = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=0)
        ckpt_path = dino_ckpt or DEFAULT_DINO_V1_CKPT
        state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict) and "teacher" in state:
            state = state["teacher"]
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        state = {
            key.replace("module.", "").replace("backbone.", ""): value
            for key, value in state.items()
            if not key.startswith("head.")
        }
        missing, unexpected = backbone.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"failed to load DINOv1-S checkpoint {ckpt_path}: "
                f"missing={missing[:8]}, unexpected={unexpected[:8]}"
            )
        return backbone, "dinov1", int(getattr(backbone, "num_features", 384))
    if not dino_repo:
        raise ValueError("dino_repo must point to a local torch hub DINOv2 repo")
    backbone = torch.hub.load(str(dino_repo), dino_model, source="local", pretrained=True)
    return backbone, "dinov2", int(getattr(backbone, "embed_dim", 384))


def extract_dino_tokens(backbone, family, x):
    features = backbone.forward_features(x)
    if family == "dinov1":
        if not torch.is_tensor(features) or features.ndim != 3 or features.shape[1] < 2:
            raise RuntimeError("DINOv1 backbone must return token tensor [B, N+1, C]")
        return features[:, 0], features[:, 1:]
    if not isinstance(features, dict) or "x_norm_clstoken" not in features:
        raise RuntimeError("DINOv2 backbone must return forward_features with x_norm_clstoken")
    return features["x_norm_clstoken"], features.get("x_norm_patchtokens")


class StyleGANDiscriminator(nn.Module):
    # Same architecture family as LlamaGen tokenizer_image/discriminator_stylegan.py,
    # with local blur implemented through conv2d to avoid an extra kornia dependency.
    def __init__(self, input_nc=3, channel_multiplier=1, image_size=256):
        super().__init__()
        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }
        if image_size not in channels:
            raise ValueError(f"StyleGANDiscriminator only supports power-of-two sizes in {sorted(channels)}, got {image_size}")
        log_size = int(math.log(image_size, 2))
        in_channel = channels[image_size]
        blocks = [nn.Conv2d(input_nc, in_channel, 3, padding=1), nn.LeakyReLU(0.2, inplace=True)]
        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            blocks.append(StyleGANDiscriminatorBlock(in_channel, out_channel))
            in_channel = out_channel
        self.blocks = nn.ModuleList(blocks)
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channel, channels[4], 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.final_linear = nn.Sequential(
            nn.Linear(channels[4] * 4 * 4, channels[4]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(channels[4], 1),
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.final_conv(x)
        x = x.view(x.shape[0], -1)
        return self.final_linear(x)


class StyleGANDiscriminatorBlock(nn.Module):
    def __init__(self, input_channels, filters, downsample=True):
        super().__init__()
        self.conv_res = nn.Conv2d(input_channels, filters, 1, stride=(2 if downsample else 1))
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, filters, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(filters, filters, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.downsample = nn.Sequential(
            StyleGANBlur(),
            nn.Conv2d(filters, filters, 3, padding=1, stride=2),
        ) if downsample else None

    def forward(self, x):
        res = self.conv_res(x)
        x = self.net(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return (x + res) * (1.0 / math.sqrt(2.0))


class StyleGANBlur(nn.Module):
    def __init__(self):
        super().__init__()
        filt = torch.tensor([1.0, 2.0, 1.0])
        filt = filt[:, None] * filt[None, :]
        filt = filt / filt.sum()
        self.register_buffer("filt", filt[None, None])

    def forward(self, x):
        filt = self.filt.to(dtype=x.dtype, device=x.device).repeat(x.shape[1], 1, 1, 1)
        return F.conv2d(x, filt, padding=1, groups=x.shape[1])


class DINOFeatureDiscriminator(nn.Module):
    def __init__(
        self,
        dino_repo,
        dino_model="dinov2_vits14",
        dino_ckpt=None,
        input_size=224,
        hidden_dim=256,
        use_patch_tokens=True,
    ):
        super().__init__()
        self.input_size = int(input_size)
        self.use_patch_tokens = bool(use_patch_tokens)
        backbone, self.dino_family, embed_dim = load_dino_backbone(dino_repo, dino_model, dino_ckpt=dino_ckpt)
        backbone.eval().requires_grad_(False)
        self.dino_proxy = (backbone,)
        self.cls_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            spectral_linear(embed_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_linear(hidden_dim, 1),
        )
        self.patch_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            spectral_linear(embed_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_linear(hidden_dim, 1),
        )
        self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])

    def _apply(self, fn, *args, **kwargs):
        super()._apply(fn, *args, **kwargs)
        self.dino_proxy[0]._apply(fn, *args, **kwargs)
        return self

    def train(self, mode=True):
        super().train(mode)
        self.dino_proxy[0].eval()
        return self

    def forward(self, x_01):
        x = x_01.float()
        if x.shape[-2:] != (self.input_size, self.input_size):
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False, antialias=True)
        x = (x - self.imagenet_mean) / self.imagenet_std
        cls_token, patch_tokens = extract_dino_tokens(self.dino_proxy[0], self.dino_family, x)
        logits = [self.cls_head(cls_token)]
        if self.use_patch_tokens and patch_tokens is not None:
            patch_logits = self.patch_head(patch_tokens).squeeze(-1)
            logits.append(patch_logits)
        return torch.cat(logits, dim=1)


class ProjectedPatchHead(nn.Module):
    def __init__(self, in_channels, hidden_channels=256, depth=2):
        super().__init__()
        depth = max(int(depth), 1)
        layers = [
            nn.utils.spectral_norm(nn.Conv2d(in_channels, hidden_channels, 1)),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        for _ in range(depth - 1):
            layers.extend([
                nn.utils.spectral_norm(nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1)),
                nn.LeakyReLU(0.2, inplace=True),
            ])
        layers.append(nn.utils.spectral_norm(nn.Conv2d(hidden_channels, 1, 1)))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.float())

    def features(self, x):
        h = x.float()
        outputs = []
        for layer in self.net[:-1]:
            h = layer(h)
            if isinstance(layer, nn.LeakyReLU):
                outputs.append(h)
        outputs.append(self.net[-1](h))
        return outputs


class ProjectedConvNeXtDiscriminator(nn.Module):
    # Projected-GAN-style D: frozen pretrained ConvNeXt features with trainable patch heads.
    # The backbone is kept outside registered submodules so optimizers/DDP only see the heads.
    def __init__(
        self,
        input_size=224,
        hidden_channels=256,
        backbone_name="convnext_small",
        feature_nodes=None,
        loss_weights=None,
        head_depth=2,
        pretrained=True,
    ):
        super().__init__()
        self.input_size = int(input_size)
        self.backbone_name = str(backbone_name)
        self.feature_nodes = tuple(
            feature_nodes
            or [
                "features.1.2.add",
                "features.3.2.add",
                "features.5.26.add",
                "features.7.2.add",
            ]
        )
        if self.backbone_name == "convnext_small":
            channels_by_node = {
                "features.1.2.add": 96,
                "features.3.2.add": 192,
                "features.5.26.add": 384,
                "features.7.2.add": 768,
            }
            weights = models.ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.convnext_small(weights=weights)
        elif self.backbone_name == "convnext_base":
            channels_by_node = {
                "features.1.2.add": 128,
                "features.3.2.add": 256,
                "features.5.26.add": 512,
                "features.7.2.add": 1024,
            }
            weights = models.ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.convnext_base(weights=weights)
        else:
            raise ValueError(f"unsupported projected ConvNeXt backbone {backbone_name}")
        missing = [node for node in self.feature_nodes if node not in channels_by_node]
        if missing:
            raise ValueError(f"unsupported projected ConvNeXt feature nodes for {self.backbone_name}: {missing}")
        backbone = backbone.eval().requires_grad_(False)
        return_nodes = {node: f"feat_{idx}" for idx, node in enumerate(self.feature_nodes)}
        self.backbone_proxy = (create_feature_extractor(backbone, return_nodes=return_nodes).eval().requires_grad_(False),)
        self.loss_weights = tuple(parse_float_sequence(loss_weights, [1.0] * len(self.feature_nodes)))
        if len(self.loss_weights) != len(self.feature_nodes):
            raise ValueError(f"projected loss weights length mismatch: {self.loss_weights} vs {self.feature_nodes}")
        self.heads = nn.ModuleList(
            ProjectedPatchHead(channels_by_node[node], hidden_channels=hidden_channels, depth=head_depth)
            for node in self.feature_nodes
        )
        self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])

    def _apply(self, fn, *args, **kwargs):
        super()._apply(fn, *args, **kwargs)
        self.backbone_proxy[0]._apply(fn, *args, **kwargs)
        return self

    def train(self, mode=True):
        super().train(mode)
        self.backbone_proxy[0].eval()
        return self

    def _extract(self, x_01):
        x = x_01.float()
        if x.shape[-2:] != (self.input_size, self.input_size):
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False, antialias=True)
        x = (x - self.imagenet_mean) / self.imagenet_std
        return list(self.backbone_proxy[0](x).values())

    def forward(self, x_01):
        features = self._extract(x_01)
        return [(head(feat), weight) for head, feat, weight in zip(self.heads, features, self.loss_weights)]

    def discriminator_features(self, x_01):
        features = self._extract(x_01)
        outputs = []
        for head, feat, weight in zip(self.heads, features, self.loss_weights):
            outputs.extend((item, weight) for item in head.features(feat))
        return outputs


def official_pg_conv2d(*args, **kwargs):
    return nn.utils.spectral_norm(nn.Conv2d(*args, **kwargs))


def official_pg_norm_layer(channels, mode="batch"):
    if mode == "group":
        return nn.GroupNorm(max(int(channels) // 2, 1), int(channels))
    if mode == "batch":
        return nn.BatchNorm2d(int(channels))
    raise ValueError(f"unsupported official Projected GAN norm mode {mode}")


class OfficialPGDownBlock(nn.Module):
    # Adapted from autonomousvision/projected_gan pg_modules.blocks.DownBlock.
    def __init__(self, in_planes, out_planes, separable=False):
        super().__init__()
        if separable:
            self.main = nn.Sequential(
                official_pg_conv2d(in_planes, in_planes, 3, groups=in_planes, bias=False, padding=1),
                official_pg_conv2d(in_planes, out_planes, 1, bias=False),
                official_pg_norm_layer(out_planes),
                nn.LeakyReLU(0.2, inplace=True),
                nn.AvgPool2d(2, 2),
            )
        else:
            self.main = nn.Sequential(
                official_pg_conv2d(in_planes, out_planes, 4, 2, 1),
                official_pg_norm_layer(out_planes),
                nn.LeakyReLU(0.2, inplace=True),
            )

    def forward(self, feat):
        return self.main(feat)


class OfficialPGDownBlockPatch(nn.Module):
    # Adapted from autonomousvision/projected_gan pg_modules.blocks.DownBlockPatch.
    def __init__(self, in_planes, out_planes, separable=False):
        super().__init__()
        self.main = nn.Sequential(
            OfficialPGDownBlock(in_planes, out_planes, separable=separable),
            official_pg_conv2d(out_planes, out_planes, 1, 1, 0, bias=False),
            official_pg_norm_layer(out_planes),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, feat):
        return self.main(feat)


class OfficialPGFeatureFusionBlock(nn.Module):
    # Adapted from autonomousvision/projected_gan pg_modules.blocks.FeatureFusionBlock.
    def __init__(self, features, expand=False, align_corners=True):
        super().__init__()
        out_features = int(features) // 2 if expand else int(features)
        self.expand = bool(expand)
        self.align_corners = bool(align_corners)
        self.out_conv = nn.Conv2d(int(features), out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

    def forward(self, *xs):
        output = xs[0]
        if len(xs) == 2:
            output = output + xs[1]
        output = F.interpolate(output, scale_factor=2, mode="bilinear", align_corners=self.align_corners)
        return self.out_conv(output)


class OfficialPGSingleDisc(nn.Module):
    # Adapted from autonomousvision/projected_gan pg_modules.discriminator.SingleDisc.
    def __init__(self, nc=None, ndf=None, start_sz=256, end_sz=8, head=False, separable=False, patch=False):
        super().__init__()
        channel_dict = {4: 512, 8: 512, 16: 256, 32: 128, 64: 64, 128: 64, 256: 32, 512: 16, 1024: 8}
        if start_sz not in channel_dict:
            start_sz = min(channel_dict.keys(), key=lambda size: abs(size - int(start_sz)))
        start_sz = int(start_sz)
        if ndf is None:
            nfc = dict(channel_dict)
        else:
            nfc = {key: int(ndf) for key in channel_dict}
        if nc is not None and not head:
            nfc[start_sz] = int(nc)
        layers = []
        if head:
            layers.extend([
                official_pg_conv2d(int(nc), nfc[256], 3, 1, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
            ])
        block_cls = OfficialPGDownBlockPatch if patch else OfficialPGDownBlock
        while start_sz > end_sz:
            layers.append(block_cls(nfc[start_sz], nfc[start_sz // 2], separable=separable))
            start_sz //= 2
        layers.append(official_pg_conv2d(nfc[end_sz], 1, 4, 1, 0, bias=False))
        self.main = nn.Sequential(*layers)

    def forward(self, x, c=None):
        return self.main(x)

    def features(self, x):
        h = x
        outputs = []
        for layer in self.main[:-1]:
            h = layer(h)
            outputs.append(h)
        outputs.append(self.main[-1](h))
        return outputs


class OfficialPGMultiScaleD(nn.Module):
    # Adapted from autonomousvision/projected_gan pg_modules.discriminator.MultiScaleD.
    def __init__(self, channels, resolutions, num_discs=4, ndf=None, separable=False, patch=False):
        super().__init__()
        num_discs = int(num_discs)
        if num_discs not in {1, 2, 3, 4}:
            raise ValueError(f"official Projected GAN num_discs must be 1..4, got {num_discs}")
        self.disc_in_channels = list(channels)[:num_discs]
        self.disc_in_res = list(resolutions)[:num_discs]
        self.mini_discs = nn.ModuleDict()
        for index, (cin, res) in enumerate(zip(self.disc_in_channels, self.disc_in_res)):
            start_sz = 16 if patch else int(res)
            self.mini_discs[str(index)] = OfficialPGSingleDisc(
                nc=int(cin),
                ndf=ndf,
                start_sz=start_sz,
                end_sz=8,
                separable=separable,
                patch=patch,
            )

    def forward(self, features, c=None):
        logits = []
        for key, disc in self.mini_discs.items():
            logits.append(disc(features[key], c).view(features[key].size(0), -1))
        return torch.cat(logits, dim=1)

    def features(self, features):
        outputs = []
        for key, disc in self.mini_discs.items():
            outputs.extend(disc.features(features[key]))
        return outputs


class OfficialPGFeatureNetwork(nn.Module):
    # Adapted from autonomousvision/projected_gan pg_modules.projector.F_RandomProj.
    def __init__(self, im_res=256, cout=64, expand=True, proj_type=2, pretrained=True):
        super().__init__()
        if proj_type not in {0, 1, 2}:
            raise ValueError(f"official Projected GAN proj_type must be 0, 1, or 2, got {proj_type}")
        import timm

        model = timm.create_model("tf_efficientnet_lite0", pretrained=bool(pretrained))
        stem_layers = [model.conv_stem, model.bn1]
        if hasattr(model, "act1"):
            stem_layers.append(model.act1)
        pretrained_layers = nn.Module()
        pretrained_layers.layer0 = nn.Sequential(*stem_layers, *model.blocks[0:2])
        pretrained_layers.layer1 = nn.Sequential(*model.blocks[2:3])
        pretrained_layers.layer2 = nn.Sequential(*model.blocks[3:5])
        pretrained_layers.layer3 = nn.Sequential(*model.blocks[5:9])
        pretrained_layers.eval().requires_grad_(False)
        self.pretrained_proxy = (pretrained_layers,)
        self.proj_type = int(proj_type)
        self.cout = int(cout)
        self.expand = bool(expand)
        self.RESOLUTIONS = [int(im_res) // 4, int(im_res) // 8, int(im_res) // 16, int(im_res) // 32]
        in_channels = self._calc_channels(int(im_res))
        if self.proj_type == 0:
            self.CHANNELS = in_channels
            self.scratch = None
            return
        scratch = nn.Module()
        out_channels = [self.cout, self.cout * 2, self.cout * 4, self.cout * 8] if self.expand else [self.cout] * 4
        scratch.layer0_ccm = nn.Conv2d(in_channels[0], out_channels[0], kernel_size=1, stride=1, padding=0, bias=True)
        scratch.layer1_ccm = nn.Conv2d(in_channels[1], out_channels[1], kernel_size=1, stride=1, padding=0, bias=True)
        scratch.layer2_ccm = nn.Conv2d(in_channels[2], out_channels[2], kernel_size=1, stride=1, padding=0, bias=True)
        scratch.layer3_ccm = nn.Conv2d(in_channels[3], out_channels[3], kernel_size=1, stride=1, padding=0, bias=True)
        if self.proj_type == 1:
            self.CHANNELS = out_channels
            self.scratch = scratch
            return
        scratch.layer3_csm = OfficialPGFeatureFusionBlock(out_channels[3], expand=self.expand)
        scratch.layer2_csm = OfficialPGFeatureFusionBlock(out_channels[2], expand=self.expand)
        scratch.layer1_csm = OfficialPGFeatureFusionBlock(out_channels[1], expand=self.expand)
        scratch.layer0_csm = OfficialPGFeatureFusionBlock(out_channels[0], expand=False)
        self.CHANNELS = [self.cout, self.cout, self.cout * 2, self.cout * 4] if self.expand else [self.cout] * 4
        self.RESOLUTIONS = [res * 2 for res in self.RESOLUTIONS]
        self.scratch = scratch

    def _apply(self, fn, *args, **kwargs):
        super()._apply(fn, *args, **kwargs)
        self.pretrained_proxy[0]._apply(fn, *args, **kwargs)
        return self

    def train(self, mode=True):
        super().train(mode)
        self.pretrained_proxy[0].eval()
        return self

    def _calc_channels(self, im_res):
        device = next(self.pretrained_proxy[0].parameters()).device
        with torch.no_grad():
            tmp = torch.zeros(1, 3, im_res, im_res, device=device)
            channels = []
            for layer_name in ["layer0", "layer1", "layer2", "layer3"]:
                tmp = getattr(self.pretrained_proxy[0], layer_name)(tmp)
                channels.append(tmp.shape[1])
        return channels

    def forward(self, x):
        out0 = self.pretrained_proxy[0].layer0(x)
        out1 = self.pretrained_proxy[0].layer1(out0)
        out2 = self.pretrained_proxy[0].layer2(out1)
        out3 = self.pretrained_proxy[0].layer3(out2)
        out = {"0": out0, "1": out1, "2": out2, "3": out3}
        if self.proj_type == 0:
            return out
        out0 = self.scratch.layer0_ccm(out["0"])
        out1 = self.scratch.layer1_ccm(out["1"])
        out2 = self.scratch.layer2_ccm(out["2"])
        out3 = self.scratch.layer3_ccm(out["3"])
        out = {"0": out0, "1": out1, "2": out2, "3": out3}
        if self.proj_type == 1:
            return out
        out3 = self.scratch.layer3_csm(out3)
        out2 = self.scratch.layer2_csm(out3, out2)
        out1 = self.scratch.layer1_csm(out2, out1)
        out0 = self.scratch.layer0_csm(out1, out0)
        return {"0": out0, "1": out1, "2": out2, "3": out3}


class OfficialPGDiffAugment(nn.Module):
    # Adapted from autonomousvision/projected_gan pg_modules.diffaug.
    def __init__(self, policy="color,translation,cutout"):
        super().__init__()
        self.policy = str(policy or "")

    def forward(self, x):
        if not self.policy:
            return x
        for policy in self.policy.split(','):
            policy = policy.strip()
            if not policy:
                continue
            if policy == "color":
                x = self.rand_brightness(x)
                x = self.rand_saturation(x)
                x = self.rand_contrast(x)
            elif policy == "translation":
                x = self.rand_translation(x)
            elif policy == "cutout":
                x = self.rand_cutout(x)
            else:
                raise ValueError(f"unsupported official Projected GAN diffaug policy {policy}")
        return x.contiguous()

    @staticmethod
    def rand_brightness(x):
        return x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)

    @staticmethod
    def rand_saturation(x):
        x_mean = x.mean(dim=1, keepdim=True)
        return (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2) + x_mean

    @staticmethod
    def rand_contrast(x):
        x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
        return (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5) + x_mean

    @staticmethod
    def rand_translation(x, ratio=0.125):
        shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
        translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
        translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
        grid_batch, grid_x, grid_y = torch.meshgrid(
            torch.arange(x.size(0), dtype=torch.long, device=x.device),
            torch.arange(x.size(2), dtype=torch.long, device=x.device),
            torch.arange(x.size(3), dtype=torch.long, device=x.device),
            indexing="ij",
        )
        grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
        grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
        x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
        return x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)

    @staticmethod
    def rand_cutout(x, ratio=0.2):
        cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
        offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
        offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
        grid_batch, grid_x, grid_y = torch.meshgrid(
            torch.arange(x.size(0), dtype=torch.long, device=x.device),
            torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
            torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
            indexing="ij",
        )
        grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
        grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
        mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
        mask[grid_batch, grid_x, grid_y] = 0
        return x * mask.unsqueeze(1)


class OfficialProjectedGANDiscriminator(nn.Module):
    # Adapted from autonomousvision/projected_gan pg_modules.discriminator.ProjectedDiscriminator.
    # The pretrained EfficientNet feature extractor is frozen; CCM/CSM and MultiScaleD are trainable.
    def __init__(
        self,
        input_size=224,
        im_res=256,
        cout=64,
        expand=True,
        proj_type=2,
        num_discs=4,
        separable=False,
        patch=False,
        pretrained=True,
        diffaug=True,
        diffaug_policy="color,translation,cutout",
    ):
        super().__init__()
        self.input_size = int(input_size)
        self.diffaug = OfficialPGDiffAugment(diffaug_policy) if diffaug else None
        self.feature_network = OfficialPGFeatureNetwork(
            im_res=int(im_res),
            cout=int(cout),
            expand=bool(expand),
            proj_type=int(proj_type),
            pretrained=bool(pretrained),
        )
        self.discriminator = OfficialPGMultiScaleD(
            channels=self.feature_network.CHANNELS,
            resolutions=self.feature_network.RESOLUTIONS,
            num_discs=int(num_discs),
            separable=bool(separable),
            patch=bool(patch),
        )

    def train(self, mode=True):
        super().train(mode)
        self.feature_network.train(False)
        return self

    def _prepare(self, x_01):
        x = x_01.float()
        if self.diffaug is not None:
            x = self.diffaug(x)
        if x.shape[-2:] != (self.input_size, self.input_size):
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        return x

    def forward(self, x_01):
        x = self._prepare(x_01)
        features = self.feature_network(x)
        return self.discriminator(features, None)

    def discriminator_features(self, x_01):
        x = self._prepare(x_01)
        features = self.feature_network(x)
        return [(feature, 1.0) for feature in self.discriminator.features(features)]


class FrozenChannelProjection(nn.Module):
    def __init__(self, in_channels, out_channels, normalize_input=True):
        super().__init__()
        self.normalize_input = bool(normalize_input)
        weight = torch.randn(int(out_channels), int(in_channels), 1, 1)
        weight = weight / math.sqrt(max(int(in_channels), 1))
        self.register_buffer("weight", weight)

    def forward(self, x):
        h = x.float()
        if self.normalize_input:
            h = F.normalize(h, dim=1)
        return F.conv2d(h, self.weight)


class ProjectedGANDiscriminator(nn.Module):
    # Closer to projected-GAN: frozen pretrained feature network, fixed random
    # channel projections, and trainable patch heads at multiple feature levels.
    def __init__(
        self,
        input_size=224,
        backbone_name="convnext_small",
        project_channels=256,
        hidden_channels=256,
        feature_nodes=None,
        loss_weights=None,
        head_depth=2,
        normalize_features=True,
        pretrained=True,
    ):
        super().__init__()
        self.input_size = int(input_size)
        self.backbone_name = str(backbone_name)
        self.feature_nodes = tuple(feature_nodes or [
            "features.1.2.add",
            "features.3.2.add",
            "features.5.26.add",
            "features.7.2.add",
        ])
        if self.backbone_name == "convnext_small":
            channels_by_node = {
                "features.1.2.add": 96,
                "features.3.2.add": 192,
                "features.5.26.add": 384,
                "features.7.2.add": 768,
            }
            weights = models.ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.convnext_small(weights=weights)
        elif self.backbone_name == "convnext_base":
            channels_by_node = {
                "features.1.2.add": 128,
                "features.3.2.add": 256,
                "features.5.26.add": 512,
                "features.7.2.add": 1024,
            }
            weights = models.ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.convnext_base(weights=weights)
        else:
            raise ValueError(f"unsupported projected GAN backbone {backbone_name}")
        missing = [node for node in self.feature_nodes if node not in channels_by_node]
        if missing:
            raise ValueError(f"unsupported projected GAN feature nodes for {self.backbone_name}: {missing}")
        return_nodes = {node: f"feat_{idx}" for idx, node in enumerate(self.feature_nodes)}
        self.backbone_proxy = (create_feature_extractor(backbone.eval().requires_grad_(False), return_nodes=return_nodes).eval().requires_grad_(False),)
        self.loss_weights = tuple(parse_float_sequence(loss_weights, [1.0] * len(self.feature_nodes)))
        if len(self.loss_weights) != len(self.feature_nodes):
            raise ValueError(f"projected GAN loss weights length mismatch: {self.loss_weights} vs {self.feature_nodes}")
        self.projections = nn.ModuleList(
            FrozenChannelProjection(
                channels_by_node[node],
                project_channels,
                normalize_input=normalize_features,
            )
            for node in self.feature_nodes
        )
        self.heads = nn.ModuleList(
            ProjectedPatchHead(project_channels, hidden_channels=hidden_channels, depth=head_depth)
            for _node in self.feature_nodes
        )
        self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])

    def _apply(self, fn, *args, **kwargs):
        super()._apply(fn, *args, **kwargs)
        self.backbone_proxy[0]._apply(fn, *args, **kwargs)
        return self

    def train(self, mode=True):
        super().train(mode)
        self.backbone_proxy[0].eval()
        return self

    def _extract(self, x_01):
        x = x_01.float()
        if x.shape[-2:] != (self.input_size, self.input_size):
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False, antialias=True)
        x = (x - self.imagenet_mean) / self.imagenet_std
        features = list(self.backbone_proxy[0](x).values())
        return [projection(feature) for projection, feature in zip(self.projections, features)]

    def forward(self, x_01):
        features = self._extract(x_01)
        return [(head(feat), weight) for head, feat, weight in zip(self.heads, features, self.loss_weights)]

    def discriminator_features(self, x_01):
        features = self._extract(x_01)
        outputs = []
        for head, feat, weight in zip(self.heads, features, self.loss_weights):
            outputs.extend((item, weight) for item in head.features(feat))
        return outputs


class FrozenDINOFeatureLoss(nn.Module):
    def __init__(
        self,
        dino_repo,
        dino_model="dinov2_vits14",
        dino_ckpt=None,
        input_size=224,
        use_patch_tokens=True,
        loss_type="l1",
        normalize_features=True,
    ):
        super().__init__()
        self.input_size = int(input_size)
        self.use_patch_tokens = bool(use_patch_tokens)
        self.loss_type = str(loss_type)
        self.normalize_features = bool(normalize_features)
        if self.loss_type not in {"l1", "l2"}:
            raise ValueError(f"unsupported dino feature loss_type {loss_type}")
        backbone, self.dino_family, _embed_dim = load_dino_backbone(dino_repo, dino_model, dino_ckpt=dino_ckpt)
        backbone.eval().requires_grad_(False)
        self.dino_proxy = (backbone,)
        self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])

    def _apply(self, fn, *args, **kwargs):
        super()._apply(fn, *args, **kwargs)
        self.dino_proxy[0]._apply(fn, *args, **kwargs)
        return self

    def train(self, mode=True):
        super().train(mode)
        self.dino_proxy[0].eval()
        return self

    def extract_features(self, x_01):
        x = x_01.float()
        if x.shape[-2:] != (self.input_size, self.input_size):
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False, antialias=True)
        x = (x - self.imagenet_mean) / self.imagenet_std
        cls_token, patch_tokens = extract_dino_tokens(self.dino_proxy[0], self.dino_family, x)
        outputs = [cls_token]
        if self.use_patch_tokens and patch_tokens is not None:
            outputs.append(patch_tokens)
        if self.normalize_features:
            outputs = [F.normalize(item.float(), dim=-1) for item in outputs]
        else:
            outputs = [item.float() for item in outputs]
        return outputs

    def forward(self, pred, target, pred_range="minus1_1", target_range="minus1_1"):
        pred_01 = image_to_zero_one(pred, pred_range)
        target_01 = image_to_zero_one(target, target_range)
        pred_features = self.extract_features(pred_01)
        with torch.no_grad():
            target_features = self.extract_features(target_01)
        losses = []
        for pred_feature, target_feature in zip(pred_features, target_features):
            if self.loss_type == "l1":
                losses.append(F.l1_loss(pred_feature, target_feature))
            else:
                losses.append(F.mse_loss(pred_feature, target_feature))
        if not losses:
            return pred.new_zeros(())
        return sum(losses) / len(losses)


def build_dino_feature_loss(args, device):
    if getattr(args, "lambda_dino_feat", 0.0) <= 0.0:
        return None
    loss = FrozenDINOFeatureLoss(
        dino_repo=getattr(args, "dino_repo", "../.cache/torch/hub/facebookresearch_dinov2_main"),
        dino_model=getattr(args, "dino_model", "dinov2_vits14"),
        dino_ckpt=getattr(args, "dino_ckpt", None),
        input_size=getattr(args, "dino_feat_input_size", getattr(args, "dino_input_size", 224)),
        use_patch_tokens=getattr(args, "dino_feat_use_patch_tokens", True),
        loss_type=getattr(args, "dino_feat_loss", "l1"),
        normalize_features=getattr(args, "dino_feat_normalize", True),
    ).to(device)
    loss.eval().requires_grad_(False)
    return loss


class FrozenCLIPImageFeatureLoss(nn.Module):
    def __init__(
        self,
        model_name="ViT-B-32",
        pretrained="openai",
        cache_dir=None,
        prefer_hf_hub=True,
        weights_only=True,
        input_size=None,
        loss_type="l1",
        normalize_features=True,
    ):
        super().__init__()
        self.loss_type = str(loss_type)
        self.normalize_features = bool(normalize_features)
        if self.loss_type not in {"l1", "l2"}:
            raise ValueError(f"unsupported CLIP feature loss_type {loss_type}")
        import open_clip

        kwargs = {}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        kwargs["weights_only"] = bool(weights_only)
        pretrained_arg = pretrained
        if pretrained and not prefer_hf_hub:
            pretrained_cfg = open_clip.get_pretrained_cfg(model_name, pretrained)
            if pretrained_cfg is None:
                raise ValueError(f"unknown open_clip pretrained tag {model_name}/{pretrained}")
            pretrained_arg = open_clip.pretrained.download_pretrained(
                pretrained_cfg,
                prefer_hf_hub=False,
                cache_dir=cache_dir,
            )
        model, _preprocess_train, _preprocess_val = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained_arg,
            **kwargs,
        )
        model.eval().requires_grad_(False)
        self.clip_proxy = (model,)
        preprocess_cfg = getattr(model.visual, "preprocess_cfg", {}) or {}
        image_size = input_size or preprocess_cfg.get("size", getattr(model.visual, "image_size", 224))
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        if isinstance(image_size, (tuple, list)) and len(image_size) == 1:
            image_size = (int(image_size[0]), int(image_size[0]))
        self.input_size = (int(image_size[0]), int(image_size[1]))
        mean = preprocess_cfg.get("mean", (0.48145466, 0.4578275, 0.40821073))
        std = preprocess_cfg.get("std", (0.26862954, 0.26130258, 0.27577711))
        self.register_buffer("clip_mean", torch.tensor(mean)[None, :, None, None])
        self.register_buffer("clip_std", torch.tensor(std)[None, :, None, None])

    def _apply(self, fn, *args, **kwargs):
        super()._apply(fn, *args, **kwargs)
        self.clip_proxy[0]._apply(fn, *args, **kwargs)
        return self

    def train(self, mode=True):
        super().train(mode)
        self.clip_proxy[0].eval()
        return self

    def extract_features(self, x_01):
        x = x_01.float()
        if x.shape[-2:] != self.input_size:
            x = F.interpolate(x, size=self.input_size, mode="bicubic", align_corners=False, antialias=True)
        x = (x - self.clip_mean) / self.clip_std
        features = self.clip_proxy[0].encode_image(x, normalize=self.normalize_features)
        return features.float()

    def forward(self, pred, target, pred_range="minus1_1", target_range="minus1_1"):
        pred_01 = image_to_zero_one(pred, pred_range)
        target_01 = image_to_zero_one(target, target_range)
        pred_features = self.extract_features(pred_01)
        with torch.no_grad():
            target_features = self.extract_features(target_01)
        if self.loss_type == "l1":
            return F.l1_loss(pred_features, target_features)
        return F.mse_loss(pred_features, target_features)


def build_clip_feature_loss(args, device):
    if getattr(args, "lambda_clip_feat", 0.0) <= 0.0:
        return None
    loss = FrozenCLIPImageFeatureLoss(
        model_name=getattr(args, "clip_model", "ViT-B-32"),
        pretrained=getattr(args, "clip_pretrained", "openai"),
        cache_dir=getattr(args, "clip_cache_dir", None),
        prefer_hf_hub=getattr(args, "clip_prefer_hf_hub", True),
        weights_only=getattr(args, "clip_weights_only", True),
        input_size=getattr(args, "clip_feat_input_size", None),
        loss_type=getattr(args, "clip_feat_loss", "l1"),
        normalize_features=getattr(args, "clip_feat_normalize", True),
    ).to(device)
    loss.eval().requires_grad_(False)
    return loss


class PatchAndDINOFeatureDiscriminator(nn.Module):
    primary_load_name = "primary_patch"

    def __init__(
        self,
        discriminator_cls,
        hidden_channels=128,
        num_stages=3,
        patch_loss_weight=1.0,
        dino_loss_weight=0.25,
        dino_repo="../.cache/torch/hub/facebookresearch_dinov2_main",
        dino_model="dinov2_vits14",
        dino_ckpt=None,
        dino_input_size=224,
        dino_head_hidden=256,
        dino_use_patch_tokens=True,
        patch_discriminator=None,
    ):
        super().__init__()
        if patch_loss_weight <= 0.0 or dino_loss_weight <= 0.0:
            raise ValueError("patch_loss_weight and dino_loss_weight must be positive")
        self.patch_loss_weight = float(patch_loss_weight)
        self.dino_loss_weight = float(dino_loss_weight)
        if patch_discriminator is None:
            patch_discriminator = discriminator_cls(hidden_channels=hidden_channels, num_stages=num_stages)
        self.patch_discriminator = patch_discriminator
        self.dino_discriminator = DINOFeatureDiscriminator(
            dino_repo=dino_repo,
            dino_model=dino_model,
            dino_ckpt=dino_ckpt,
            input_size=dino_input_size,
            hidden_dim=dino_head_hidden,
            use_patch_tokens=dino_use_patch_tokens,
        )

    def forward(self, x):
        return [
            (self.patch_discriminator(x), self.patch_loss_weight),
            (self.dino_discriminator(x), self.dino_loss_weight),
        ]

    def load_primary_state_dict(self, state_dict, strict=True):
        if hasattr(self.patch_discriminator, "load_primary_state_dict"):
            return self.patch_discriminator.load_primary_state_dict(state_dict, strict=strict)
        return self.patch_discriminator.load_state_dict(state_dict, strict=strict)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        patch_prefix = "patch_discriminator."
        dino_prefix = "dino_discriminator."
        has_single_patch = any(
            key.startswith(patch_prefix) and not key.startswith(patch_prefix + "discriminators.")
            for key in state_dict
        )
        if isinstance(self.patch_discriminator, MultiScalePatchDiscriminator) and has_single_patch:
            patch_state = {key[len(patch_prefix):]: value for key, value in state_dict.items() if key.startswith(patch_prefix)}
            dino_state = {key[len(dino_prefix):]: value for key, value in state_dict.items() if key.startswith(dino_prefix)}
            missing = []
            unexpected = []
            for idx, disc in enumerate(self.patch_discriminator.discriminators):
                patch_missing, patch_unexpected = disc.load_state_dict(patch_state, strict=strict, assign=assign)
                missing.extend(f"{patch_prefix}discriminators.{idx}.{key}" for key in patch_missing)
                unexpected.extend(f"{patch_prefix}{key}" for key in patch_unexpected)
            dino_missing, dino_unexpected = self.dino_discriminator.load_state_dict(dino_state, strict=strict, assign=assign)
            missing.extend(f"{dino_prefix}{key}" for key in dino_missing)
            unexpected.extend(f"{dino_prefix}{key}" for key in dino_unexpected)
            return _IncompatibleKeys(missing, unexpected)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)


def build_discriminator(args, device):
    if args.lambda_gan <= 0.0:
        return None
    add_path(args.titok_root)
    from modeling.modules.discriminator import NLayerDiscriminator

    discriminator_type = getattr(args, "discriminator_type", "patch")
    if discriminator_type == "patch":
        discriminator = NLayerDiscriminator(
            hidden_channels=args.disc_hidden_channels,
            num_stages=args.disc_num_stages,
        ).to(device)
    elif discriminator_type == "multiscale_patch":
        scales = parse_float_sequence(getattr(args, "disc_scales", [1.0, 0.5, 0.25]), [1.0, 0.5, 0.25])
        weights = parse_float_sequence(getattr(args, "disc_loss_weights", [1.0, 0.5, 0.25]), [1.0, 0.5, 0.25])
        discriminator = MultiScalePatchDiscriminator(
            NLayerDiscriminator,
            scales=scales,
            loss_weights=weights,
            hidden_channels=args.disc_hidden_channels,
            num_stages=args.disc_num_stages,
        ).to(device)
    elif discriminator_type in {"dino", "dinov1s"}:
        dino_model = getattr(args, "dino_model", "dinov2_vits14")
        if discriminator_type == "dinov1s":
            dino_model = "dinov1_vits16"
        discriminator = DINOFeatureDiscriminator(
            dino_repo=getattr(args, "dino_repo", "../.cache/torch/hub/facebookresearch_dinov2_main"),
            dino_model=dino_model,
            dino_ckpt=getattr(args, "dino_ckpt", None),
            input_size=getattr(args, "dino_input_size", 224),
            hidden_dim=getattr(args, "dino_head_hidden", 256),
            use_patch_tokens=getattr(args, "dino_use_patch_tokens", True),
        ).to(device)
    elif discriminator_type == "patch_dino":
        discriminator = PatchAndDINOFeatureDiscriminator(
            NLayerDiscriminator,
            hidden_channels=args.disc_hidden_channels,
            num_stages=args.disc_num_stages,
            patch_loss_weight=1.0,
            dino_loss_weight=getattr(args, "dino_loss_weight", 0.25),
            dino_repo=getattr(args, "dino_repo", "../.cache/torch/hub/facebookresearch_dinov2_main"),
            dino_model=getattr(args, "dino_model", "dinov2_vits14"),
            dino_ckpt=getattr(args, "dino_ckpt", None),
            dino_input_size=getattr(args, "dino_input_size", 224),
            dino_head_hidden=getattr(args, "dino_head_hidden", 256),
            dino_use_patch_tokens=getattr(args, "dino_use_patch_tokens", True),
        ).to(device)
    elif discriminator_type == "projected_convnext":
        discriminator = ProjectedConvNeXtDiscriminator(
            input_size=getattr(args, "projected_input_size", 224),
            hidden_channels=getattr(args, "projected_head_hidden", 256),
            backbone_name=getattr(args, "projected_backbone", "convnext_small"),
            head_depth=getattr(args, "projected_head_depth", 2),
            loss_weights=getattr(args, "projected_loss_weights", [1.0, 1.0, 1.0, 1.0]),
            pretrained=getattr(args, "projected_pretrained", True),
        ).to(device)
    elif discriminator_type == "projected_gan":
        discriminator = ProjectedGANDiscriminator(
            input_size=getattr(args, "projected_input_size", 224),
            backbone_name=getattr(args, "projected_backbone", "convnext_small"),
            project_channels=getattr(args, "projected_channels", 256),
            hidden_channels=getattr(args, "projected_head_hidden", 256),
            head_depth=getattr(args, "projected_head_depth", 2),
            loss_weights=getattr(args, "projected_loss_weights", [1.0, 1.0, 1.0, 1.0]),
            normalize_features=getattr(args, "projected_normalize_features", True),
            pretrained=getattr(args, "projected_pretrained", True),
        ).to(device)
    elif discriminator_type == "official_projected_gan":
        discriminator = OfficialProjectedGANDiscriminator(
            input_size=getattr(args, "official_pg_input_size", getattr(args, "projected_input_size", 224)),
            im_res=getattr(args, "official_pg_im_res", getattr(args, "image_size", 256)),
            cout=getattr(args, "official_pg_cout", 64),
            expand=getattr(args, "official_pg_expand", True),
            proj_type=getattr(args, "official_pg_proj_type", 2),
            num_discs=getattr(args, "official_pg_num_discs", 4),
            separable=getattr(args, "official_pg_separable", False),
            patch=getattr(args, "official_pg_patch", False),
            pretrained=getattr(args, "official_pg_pretrained", True),
            diffaug=getattr(args, "official_pg_diffaug", True),
            diffaug_policy=getattr(args, "official_pg_diffaug_policy", "color,translation,cutout"),
        ).to(device)
    elif discriminator_type == "stylegan":
        discriminator = StyleGANDiscriminator(
            input_nc=3,
            channel_multiplier=getattr(args, "stylegan_channel_multiplier", 1),
            image_size=args.image_size,
        ).to(device)
    elif discriminator_type == "multiscale_patch_dino":
        scales = parse_float_sequence(getattr(args, "disc_scales", [1.0, 0.5]), [1.0, 0.5])
        weights = parse_float_sequence(getattr(args, "disc_loss_weights", [1.0, 0.5]), [1.0, 0.5])
        patch_discriminator = MultiScalePatchDiscriminator(
            NLayerDiscriminator,
            scales=scales,
            loss_weights=weights,
            hidden_channels=args.disc_hidden_channels,
            num_stages=args.disc_num_stages,
        )
        discriminator = PatchAndDINOFeatureDiscriminator(
            NLayerDiscriminator,
            hidden_channels=args.disc_hidden_channels,
            num_stages=args.disc_num_stages,
            patch_loss_weight=1.0,
            dino_loss_weight=getattr(args, "dino_loss_weight", 0.25),
            dino_repo=getattr(args, "dino_repo", "../.cache/torch/hub/facebookresearch_dinov2_main"),
            dino_model=getattr(args, "dino_model", "dinov2_vits14"),
            dino_ckpt=getattr(args, "dino_ckpt", None),
            dino_input_size=getattr(args, "dino_input_size", 224),
            dino_head_hidden=getattr(args, "dino_head_hidden", 256),
            dino_use_patch_tokens=getattr(args, "dino_use_patch_tokens", True),
            patch_discriminator=patch_discriminator,
        ).to(device)
    else:
        raise ValueError(f"unsupported discriminator_type {discriminator_type}")
    return discriminator


def iter_weighted_logits(logits, base_weight=1.0):
    if isinstance(logits, (list, tuple)):
        if len(logits) == 0:
            raise ValueError("empty discriminator logits")
        for item in logits:
            if isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[1], (int, float)):
                yield from iter_weighted_logits(item[0], base_weight * float(item[1]))
            else:
                yield from iter_weighted_logits(item, base_weight)
    else:
        yield logits, base_weight


def _patch_discriminator_features(discriminator, x):
    h = discriminator.block_in(x)
    features = [h]
    for block in discriminator.blocks:
        h = block(h)
        features.append(h)
    h = discriminator.pool(h)
    features.append(h)
    return features


def _dino_head_hidden(head, tokens):
    h = head[0](tokens)
    h = head[1](h)
    h = head[2](h)
    return h


def _dino_discriminator_features(discriminator, x_01):
    x = x_01.float()
    if x.shape[-2:] != (discriminator.input_size, discriminator.input_size):
        x = F.interpolate(
            x,
            size=(discriminator.input_size, discriminator.input_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    x = (x - discriminator.imagenet_mean) / discriminator.imagenet_std
    cls_token, patch_tokens = extract_dino_tokens(discriminator.dino_proxy[0], discriminator.dino_family, x)
    outputs = [_dino_head_hidden(discriminator.cls_head, cls_token)]
    if discriminator.use_patch_tokens and patch_tokens is not None:
        outputs.append(_dino_head_hidden(discriminator.patch_head, patch_tokens))
    return outputs


def _weighted_discriminator_features(discriminator, x):
    discriminator = getattr(discriminator, "module", discriminator)
    if isinstance(discriminator, MultiScalePatchDiscriminator):
        outputs = []
        for disc, scale, weight in zip(discriminator.discriminators, discriminator.scales, discriminator.loss_weights):
            if abs(scale - 1.0) < 1e-8:
                x_scale = x
            else:
                size = (max(1, int(round(x.shape[-2] * scale))), max(1, int(round(x.shape[-1] * scale))))
                x_scale = F.interpolate(x, size=size, mode="bilinear", align_corners=False, antialias=True)
            outputs.extend((feature, float(weight)) for feature in _patch_discriminator_features(disc, x_scale))
        return outputs
    if isinstance(discriminator, PatchAndDINOFeatureDiscriminator):
        outputs = []
        for feature, weight in _weighted_discriminator_features(discriminator.patch_discriminator, x):
            outputs.append((feature, discriminator.patch_loss_weight * float(weight)))
        outputs.extend((feature, discriminator.dino_loss_weight) for feature in _dino_discriminator_features(discriminator.dino_discriminator, x))
        return outputs
    if isinstance(discriminator, DINOFeatureDiscriminator):
        return [(feature, 1.0) for feature in _dino_discriminator_features(discriminator, x)]
    if isinstance(discriminator, (ProjectedConvNeXtDiscriminator, ProjectedGANDiscriminator, OfficialProjectedGANDiscriminator)):
        return discriminator.discriminator_features(x)
    if hasattr(discriminator, "block_in") and hasattr(discriminator, "blocks") and hasattr(discriminator, "pool"):
        return [(feature, 1.0) for feature in _patch_discriminator_features(discriminator, x)]
    return []


def discriminator_feature_matching_loss(discriminator, fake, real):
    fake_features = _weighted_discriminator_features(discriminator, fake)
    if not fake_features:
        return fake.new_zeros(())
    with torch.no_grad():
        real_features = _weighted_discriminator_features(discriminator, real)
    losses = []
    weights = []
    for (fake_feature, fake_weight), (real_feature, real_weight) in zip(fake_features, real_features):
        weight = 0.5 * (float(fake_weight) + float(real_weight))
        losses.append(F.l1_loss(fake_feature.float(), real_feature.detach().float()) * weight)
        weights.append(weight)
    if not losses:
        return fake.new_zeros(())
    return sum(losses) / max(sum(weights), 1e-8)


def split_discriminator_logits(logits, chunks=2, dim=0):
    if isinstance(logits, (list, tuple)):
        split_parts = [[] for _ in range(chunks)]
        for logit, weight in iter_weighted_logits(logits):
            pieces = logit.chunk(chunks, dim=dim)
            for index, piece in enumerate(pieces):
                split_parts[index].append((piece, weight))
        return tuple(split_parts)
    return logits.chunk(chunks, dim=dim)


def weighted_logits_mean(logits):
    total = None
    denom = 0.0
    for logit, weight in iter_weighted_logits(logits):
        value = torch.mean(logit) * weight
        total = value if total is None else total + value
        denom += weight
    if total is None or denom <= 0.0:
        raise ValueError("empty discriminator logits")
    return total / denom


def gan_g_loss_from_logits(logits_fake):
    total = None
    denom = 0.0
    for logit, weight in iter_weighted_logits(logits_fake):
        value = -torch.mean(logit) * weight
        total = value if total is None else total + value
        denom += weight
    if total is None or denom <= 0.0:
        raise ValueError("empty discriminator logits")
    return total / denom


def hinge_d_loss(logits_real, logits_fake):
    total = None
    denom = 0.0
    for (real, weight_real), (fake, weight_fake) in zip(iter_weighted_logits(logits_real), iter_weighted_logits(logits_fake)):
        if abs(weight_real - weight_fake) > 1e-8:
            raise ValueError(f"real/fake discriminator weights mismatch: {weight_real} vs {weight_fake}")
        loss_real = torch.mean(F.relu(1.0 - real))
        loss_fake = torch.mean(F.relu(1.0 + fake))
        value = 0.5 * (loss_real + loss_fake) * weight_real
        total = value if total is None else total + value
        denom += weight_real
    if total is None or denom <= 0.0:
        raise ValueError("empty discriminator logits")
    return total / denom


def compute_lecam_loss(logits_real_mean, logits_fake_mean, ema_real_logits_mean, ema_fake_logits_mean):
    loss = torch.mean(torch.pow(F.relu(logits_real_mean - ema_fake_logits_mean), 2))
    loss = loss + torch.mean(torch.pow(F.relu(ema_real_logits_mean - logits_fake_mean), 2))
    return loss


def gan_factor_for_step(args, step):
    if args.lambda_gan <= 0.0 or step < args.gan_start_step:
        return 0.0
    if args.gan_ramp_steps <= 0:
        return args.discriminator_factor
    ramp = min(1.0, max(0.0, float(step - args.gan_start_step) / float(args.gan_ramp_steps)))
    return args.discriminator_factor * ramp


def image_to_zero_one(x, image_range):
    if image_range == "minus1_1":
        return (x.float() + 1.0) * 0.5
    if image_range == "zero_1":
        return x.float()
    raise ValueError(f"unsupported image range {image_range}")


def discriminator_input(x, image_range):
    return image_to_zero_one(x, image_range)


def llamagen_decoder_feature_from_quant(vq_model, quant, codebook_embed_dim=8):
    if quant.ndim != 4 or quant.shape[-2:] != (16, 16):
        raise ValueError(f"unexpected LlamaGen encode output shape: {tuple(quant.shape)}")
    if quant.shape[1] == codebook_embed_dim:
        return vq_model.post_quant_conv(quant)
    if quant.shape[1] == 256:
        return quant
    raise ValueError(
        f"cannot infer LlamaGen feature space from encode output {tuple(quant.shape)}; "
        f"expected channels {codebook_embed_dim} or 256"
    )


def target_llamagen_decoder_feature_and_codes(vq_model, x, codebook_embed_dim=8):
    quant, _, info = vq_model.encode(x)
    feature = llamagen_decoder_feature_from_quant(vq_model, quant, codebook_embed_dim)
    code_indices = None
    if info is not None and len(info) >= 3 and info[2] is not None:
        code_indices = info[2].view(x.shape[0], quant.shape[-2], quant.shape[-1]).long()
    return feature, code_indices


def target_llamagen_decoder_feature(vq_model, x, codebook_embed_dim=8):
    feature, _ = target_llamagen_decoder_feature_and_codes(vq_model, x, codebook_embed_dim)
    return feature


def make_transform(image_size, random_crop=False, random_flip=False):
    crop = transforms.RandomCrop(image_size) if random_crop else transforms.CenterCrop(image_size)
    ops = [
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        crop,
    ]
    if random_flip:
        ops.append(transforms.RandomHorizontalFlip())
    ops.append(transforms.ToTensor())
    return transforms.Compose(ops)


def convert_image_range(x_01, image_range):
    if image_range == "zero_1":
        return x_01
    if image_range == "minus1_1":
        return x_01 * 2.0 - 1.0
    raise ValueError(f"unsupported image range {image_range}")


def denorm_to_uint8(x, image_range="minus1_1"):
    x = x.detach().float().cpu()
    if image_range == "minus1_1":
        x = (x.clamp(-1.0, 1.0) + 1.0) * 0.5
    elif image_range == "zero_1":
        x = x.clamp(0.0, 1.0)
    else:
        raise ValueError(f"unsupported image range {image_range}")
    x = (x.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
    return x


def chw_to_pil(x):
    return Image.fromarray(x.permute(1, 2, 0).numpy())


def save_recon_grid(path, x, x_rec, x_mix=None, max_images=8, image_range="minus1_1"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    originals = denorm_to_uint8(x[:max_images], image_range)
    recs = denorm_to_uint8(x_rec[:max_images], image_range)
    tensors = [originals, recs]
    headers = ["target", "1d"]
    if x_mix is not None:
        tensors.append(denorm_to_uint8(x_mix[:max_images], image_range))
        headers.append("mix")

    n = originals.shape[0]
    cell = originals.shape[-1]
    label_h = 18
    canvas = Image.new("RGB", (len(tensors) * cell, n * (cell + label_h)), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for i in range(n):
        for j, batch in enumerate(tensors):
            x0 = j * cell
            y0 = i * (cell + label_h)
            draw.text((x0 + 4, y0 + 2), headers[j], fill=(0, 0, 0))
            canvas.paste(chw_to_pil(batch[i]), (x0, y0 + label_h))
    canvas.save(path)



def exclude_from_weight_decay(name, param):
    return (
        param.ndim < 2
        or "ln" in name
        or "bias" in name
        or "latent_tokens" in name
        or "mask_token" in name
        or "embedding" in name
        or "norm" in name
        or "gamma" in name
        or "embed" in name
    )


def adamw_param_groups(module, weight_decay):
    no_decay = []
    decay = []
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        if exclude_from_weight_decay(name, param):
            no_decay.append(param)
        else:
            decay.append(param)
    groups = []
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    return groups


class TrainableEMA:
    def __init__(self, module, decay=0.999):
        self.decay = decay
        self.shadow = {
            name: param.detach().clone()
            for name, param in module.named_parameters()
            if param.requires_grad
        }

    @torch.no_grad()
    def update(self, module):
        for name, param in module.named_parameters():
            if name not in self.shadow:
                continue
            self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def state_dict(self):
        return {name: value.detach().cpu().clone() for name, value in self.shadow.items()}

    def load_state_dict(self, state_dict, device=None):
        for name, value in state_dict.items():
            tensor = value.detach().clone()
            if device is not None:
                tensor = tensor.to(device)
            self.shadow[name] = tensor


def random_spatial_mask(batch_size, ratio_min, ratio_max, device):
    ratios = torch.empty(batch_size, device=device).uniform_(ratio_min, ratio_max)
    noise = torch.rand(batch_size, 1, 16, 16, device=device)
    masks = []
    for i, ratio in enumerate(ratios):
        k = int(round(float(ratio) * 256))
        flat = torch.zeros(256, device=device, dtype=noise.dtype)
        if k > 0:
            topk = torch.topk(noise[i, 0].flatten(), k).indices
            flat[topk] = 1.0
        masks.append(flat.reshape(1, 16, 16))
    return torch.stack(masks, dim=0)


def cycle(loader, sampler=None):
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


def compute_lr(args, step, base_lr=None, start_step=0, max_steps=None):
    base_lr = args.lr if base_lr is None else base_lr
    max_steps = args.max_steps if max_steps is None else max_steps
    rel_step = max(step - start_step, 0)
    total_steps = max(max_steps - start_step, 1)
    if args.lr_scheduler == "constant":
        return base_lr
    if args.lr_scheduler != "cosine":
        raise ValueError(f"unsupported lr_scheduler {args.lr_scheduler}")
    if args.warmup_steps > 0 and rel_step < args.warmup_steps:
        return base_lr * rel_step / args.warmup_steps
    denom = max(total_steps - args.warmup_steps, 1)
    progress = min(max((rel_step - args.warmup_steps) / denom, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return args.min_lr + (base_lr - args.min_lr) * cosine


def set_optimizer_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr


@torch.no_grad()
def codebook_usage_stats(code_probs):
    if code_probs is None:
        return ""
    probs = code_probs.detach().float()
    codebook_size = probs.shape[1]
    max_prob = probs.amax(dim=1).mean().item()
    probs_for_log = probs.clamp_min(1e-12)
    entropy = -(probs_for_log * probs_for_log.log()).sum(dim=1).mean().item()
    entropy = entropy / math.log(codebook_size)
    top1 = probs.argmax(dim=1)
    unique_denom = min(codebook_size, top1.numel())
    top1_use = top1.unique().numel() / max(unique_denom, 1)
    return f" code_ent {entropy:.4f} code_maxp {max_prob:.4f} code_top1_use {top1_use:.4f}"


def compute_codebook_entropy_loss(code_probs, entropy_target):
    if code_probs is None or entropy_target <= 0.0:
        return None
    probs = code_probs.float().clamp_min(1e-12)
    entropy = -(probs * probs.log()).sum(dim=1).mean() / math.log(probs.shape[1])
    return F.relu(entropy_target - entropy).square()


def compute_feature_moment_loss(pred, target):
    pred = pred.float()
    target = target.float()
    pred_mean = pred.mean(dim=(2, 3))
    target_mean = target.mean(dim=(2, 3))
    pred_std = pred.var(dim=(2, 3), unbiased=False).clamp_min(1e-12).sqrt()
    target_std = target.var(dim=(2, 3), unbiased=False).clamp_min(1e-12).sqrt()
    return F.l1_loss(pred_mean, target_mean) + F.l1_loss(pred_std, target_std)


def compute_codebook_ce_loss(code_logits, code_targets, label_smoothing=0.0):
    if code_logits is None or code_targets is None:
        return None
    if code_logits.ndim != 4 or code_targets.ndim != 3:
        raise ValueError(f"bad code CE shapes logits={tuple(code_logits.shape)} targets={tuple(code_targets.shape)}")
    if code_logits.shape[0] != code_targets.shape[0] or code_logits.shape[-2:] != code_targets.shape[-2:]:
        raise ValueError(f"code CE shape mismatch logits={tuple(code_logits.shape)} targets={tuple(code_targets.shape)}")
    return F.cross_entropy(code_logits.float(), code_targets.long(), label_smoothing=label_smoothing)


@torch.no_grad()
def compute_codebook_accuracy(code_logits, code_targets):
    if code_logits is None or code_targets is None:
        return code_logits.new_zeros(()) if code_logits is not None else torch.zeros(())
    pred = code_logits.detach().argmax(dim=1)
    return (pred == code_targets).float().mean()


def compute_codebook_temperature(args, step):
    final_temperature = args.codebook_temperature_final
    if final_temperature is None or args.codebook_temperature_warmup_steps <= 0:
        return args.codebook_temperature
    progress = min(max(step / args.codebook_temperature_warmup_steps, 0.0), 1.0)
    return args.codebook_temperature + (final_temperature - args.codebook_temperature) * progress


def set_codebook_temperature(model, temperature):
    core = model.module if isinstance(model, DDP) else model
    core.latent_decoder.codebook_temperature = temperature


def autocast_dtype(mixed_precision):
    if mixed_precision == "bf16":
        return torch.bfloat16
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "none":
        return torch.float32
    raise ValueError(f"unsupported mixed_precision {mixed_precision}")


def distributed_setup():
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rank = 0
        world_size = 1
        local_rank = 0
    return distributed, rank, world_size, local_rank, device


def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training")
    distributed, rank, world_size, local_rank, device = distributed_setup()
    is_main = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    dataset = ImageFolder(args.data_path, transform=make_transform(args.image_size, args.random_crop, args.random_flip))
    if args.limit_samples > 0:
        dataset = Subset(dataset, list(range(min(args.limit_samples, len(dataset)))))
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True) if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    data_iter = cycle(loader, sampler)
    loader_steps_per_epoch = max(1, len(loader))
    optimizer_steps_per_epoch = max(1, loader_steps_per_epoch // max(1, args.accum_steps))

    titok = load_titok(args.titok_root, args.titok_config, args.titok_ckpt, device)
    vq_model = load_llamagen_vq(args.llamagen_root, args.llamagen_ckpt, device, args.codebook_size, args.codebook_embed_dim)
    perceptual = build_perceptual_loss(args, device)

    model = TiTokLlamaGenStage2(
        titok,
        vq_model,
        lg_latent_channels=args.lg_latent_channels,
        head_channels=args.lg_head_channels,
        head_mode=args.latent_head_mode,
        codebook_size=args.codebook_size,
        codebook_temperature=args.codebook_temperature,
    ).to(device)

    params = [p for p in model.latent_decoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        adamw_param_groups(model.latent_decoder, args.weight_decay),
        lr=args.lr,
        betas=(0.9, 0.999),
    )
    ema = TrainableEMA(model.latent_decoder, decay=args.ema_decay) if args.use_ema else None
    discriminator = build_discriminator(args, device)
    disc_params = [p for p in discriminator.parameters() if p.requires_grad] if discriminator is not None else []
    optimizer_d = (
        torch.optim.AdamW(
            adamw_param_groups(discriminator, args.weight_decay),
            lr=args.lr_d,
            betas=(0.9, 0.999),
        )
        if discriminator is not None
        else None
    )
    lecam_ema_real = torch.zeros((), device=device)
    lecam_ema_fake = torch.zeros((), device=device)

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        state = ckpt.get("model", ckpt)
        if isinstance(state, dict) and any(key.startswith("latent_decoder.") for key in state):
            state = {key[len("latent_decoder."):]: value for key, value in state.items() if key.startswith("latent_decoder.")}
        model.load_trainable_state_dict(state, strict=True)
        if discriminator is not None and isinstance(ckpt, dict) and "discriminator" in ckpt:
            discriminator.load_state_dict(ckpt["discriminator"], strict=True)
        start_step = int(ckpt.get("step", 0)) if isinstance(ckpt, dict) else 0
        if isinstance(ckpt, dict) and "optimizer" in ckpt and not args.reset_optimizer:
            optimizer.load_state_dict(ckpt["optimizer"])
        if optimizer_d is not None and isinstance(ckpt, dict) and "optimizer_d" in ckpt and not args.reset_optimizer:
            optimizer_d.load_state_dict(ckpt["optimizer_d"])
        if isinstance(ckpt, dict):
            lecam_ema_real = ckpt.get("lecam_ema_real", lecam_ema_real.detach().cpu()).to(device)
            lecam_ema_fake = ckpt.get("lecam_ema_fake", lecam_ema_fake.detach().cpu()).to(device)
            if ema is not None and "model_ema" in ckpt:
                ema.load_state_dict(ckpt["model_ema"], device=device)
        if is_main:
            print(f"resumed trainable decoder from {args.resume} at step {start_step}", flush=True)

    if args.epochs > 0.0:
        train_steps = max(1, int(round(args.epochs * optimizer_steps_per_epoch)))
        args.max_steps = start_step + train_steps
    elif args.max_steps <= start_step:
        raise ValueError(
            f"max_steps ({args.max_steps}) must exceed the resumed step ({start_step}) "
            "when epochs is not set"
        )

    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        if discriminator is not None:
            discriminator = DDP(
                discriminator,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
                broadcast_buffers=False,
            )

    out_dir = Path(args.output_dir)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()
    log_path = out_dir / "log.txt"

    if is_main:
        core = model.module if distributed else model
        trainable_params = sum(p.numel() for p in core.latent_decoder.parameters() if p.requires_grad)
        effective_batch = args.batch_size * world_size * args.accum_steps
        msg = (
            f"dataset={len(dataset)} world_size={world_size} batch_size_per_gpu={args.batch_size} "
            f"accum_steps={args.accum_steps} global_batch={effective_batch} trainable_params={trainable_params} "
            f"loader_steps_per_epoch={loader_steps_per_epoch} "
            f"optimizer_steps_per_epoch={optimizer_steps_per_epoch} epochs={args.epochs} max_steps={args.max_steps} "
            f"loss_weights=image({args.image_loss}):{args.lambda_l1},perceptual({args.perceptual_loss}):{get_perceptual_weight(args)},"
            f"feat:{args.lambda_feat},feat_moment:{args.lambda_feat_moment},mix:{args.lambda_mix},"
            f"gan:{args.lambda_gan}@{args.gan_start_step}+ramp{args.gan_ramp_steps},"
            f"code_ent:{args.lambda_codebook_entropy}@{args.codebook_entropy_target},"
            f"code_ce:{args.lambda_codebook_ce},smooth:{args.codebook_ce_label_smoothing} "
            f"ranges=titok:{args.titok_input_range},llamagen:{args.llamagen_input_range} "
            f"head_mode:{args.latent_head_mode},codebook_temp:{args.codebook_temperature}->{args.codebook_temperature_final}"
            f"/{args.codebook_temperature_warmup_steps} "
            f"augment=random_crop:{args.random_crop},random_flip:{args.random_flip} lr_scheduler={args.lr_scheduler} "
            f"lr:{args.lr},min_lr:{args.min_lr},warmup:{args.warmup_steps},lr_d:{args.lr_d} "
            f"ema:{args.use_ema}@{args.ema_decay}"
        )
        print(msg, flush=True)
        log_path.write_text(msg + "\n")

    running = {
        "loss": 0.0, "recon": 0.0, "mse01": 0.0, "lpips": 0.0, "feat": 0.0, "feat_moment": 0.0, "mix": 0.0,
        "code_ent_loss": 0.0, "code_ce": 0.0, "code_acc": 0.0,
        "gan_g": 0.0, "d_loss": 0.0, "lecam": 0.0,
        "logits_real": 0.0, "logits_fake": 0.0, "grad": 0.0,
    }
    count = 0
    start_time = time.time()
    model.train()
    for step in range(start_step + 1, args.max_steps + 1):
        current_lr = compute_lr(args, step)
        current_lr_d = compute_lr(args, step, base_lr=args.lr_d, start_step=args.gan_start_step)
        current_codebook_temperature = compute_codebook_temperature(args, step)
        set_codebook_temperature(model, current_codebook_temperature)
        set_optimizer_lr(optimizer, current_lr)
        if optimizer_d is not None:
            set_optimizer_lr(optimizer_d, current_lr_d)
        gan_factor = gan_factor_for_step(args, step)
        train_discriminator = discriminator is not None and gan_factor > 0.0 and step % args.d_every == 0
        core_for_frozen = model.module if distributed else model

        optimizer.zero_grad(set_to_none=True)
        if optimizer_d is not None:
            optimizer_d.zero_grad(set_to_none=True)

        last_x_lg = None
        last_x_rec = None
        last_x_mix_rec = None
        last_f_1d_lg = None
        last_f_2d_lg = None
        last_code_probs = None
        log_due = is_main and (step == 1 or step % args.log_every == 0)
        metric_sums = {
            "loss": 0.0, "recon": 0.0, "mse01": 0.0, "lpips": 0.0, "feat": 0.0, "feat_moment": 0.0, "mix": 0.0,
            "code_ent_loss": 0.0, "code_ce": 0.0, "code_acc": 0.0,
            "gan_g": 0.0, "d_loss": 0.0, "lecam": 0.0,
            "logits_real": 0.0, "logits_fake": 0.0,
        }

        for _micro_step in range(args.accum_steps):
            x_01, _ = next(data_iter)
            x_01 = x_01.to(device, non_blocking=True)
            x_titok = convert_image_range(x_01, args.titok_input_range)
            x_lg = convert_image_range(x_01, args.llamagen_input_range)

            with torch.no_grad():
                f_2d_lg, code_targets = target_llamagen_decoder_feature_and_codes(
                    core_for_frozen.llamagen_vq, x_lg, args.codebook_embed_dim
                )

            with torch.autocast(device_type="cuda", dtype=autocast_dtype(args.mixed_precision), enabled=args.mixed_precision != "none"):
                x_rec, extra = model(x_titok)
                f_1d_lg = extra["f_1d_lg"]
                code_probs = extra.get("code_probs")
                code_logits = extra.get("code_logits")
                if log_due:
                    last_code_probs = code_probs
                if f_1d_lg.shape[1:] != (args.lg_latent_channels, 16, 16):
                    raise ValueError(f"bad f_1d_lg shape {tuple(f_1d_lg.shape)}")
                if f_2d_lg.shape[1:] != (args.lg_latent_channels, 16, 16):
                    raise ValueError(f"bad f_2d_lg shape {tuple(f_2d_lg.shape)}")
                if x_rec.shape != x_lg.shape:
                    raise ValueError(f"bad x_rec shape {tuple(x_rec.shape)} expected {tuple(x_lg.shape)}")

                x_rec_01 = image_to_zero_one(x_rec, args.llamagen_input_range)
                x_lg_01 = image_to_zero_one(x_lg, args.llamagen_input_range)
                recon_loss = compute_image_loss(x_rec, x_lg, args.image_loss)
                mse01_metric = F.mse_loss(x_rec_01.float(), x_lg_01.float())
                lpips_loss = compute_perceptual_loss(perceptual, args.perceptual_loss, x_rec, x_lg)
                feat_loss = F.l1_loss(f_1d_lg.float(), f_2d_lg.float())
                feat_moment_loss = compute_feature_moment_loss(f_1d_lg, f_2d_lg)
                codebook_entropy_loss = compute_codebook_entropy_loss(code_probs, args.codebook_entropy_target)
                if codebook_entropy_loss is None:
                    codebook_entropy_loss = x_rec.new_zeros(())
                codebook_ce_loss = compute_codebook_ce_loss(
                    code_logits, code_targets, args.codebook_ce_label_smoothing
                )
                if codebook_ce_loss is None:
                    codebook_ce_loss = x_rec.new_zeros(())
                codebook_acc = compute_codebook_accuracy(code_logits, code_targets).to(device=x_rec.device, dtype=x_rec.dtype)

                mix_loss = x_rec.new_zeros(())
                x_mix_rec = None
                if args.lambda_mix > 0.0:
                    mask = random_spatial_mask(x_lg.shape[0], args.mask_ratio_min, args.mask_ratio_max, x_lg.device).to(f_1d_lg.dtype)
                    f_mix = (1.0 - mask) * f_1d_lg + mask * f_2d_lg
                    x_mix_rec = core_for_frozen.llamagen_vq.decoder(f_mix)
                    mix_loss = compute_image_loss(x_mix_rec, x_lg, args.image_loss)

                gan_g_loss = x_rec.new_zeros(())
                if discriminator is not None and gan_factor > 0.0:
                    set_requires_grad(discriminator, False)
                    logits_fake_for_g = discriminator(discriminator_input(x_rec, args.llamagen_input_range))
                    gan_g_loss = gan_g_loss_from_logits(logits_fake_for_g)

                loss = (
                    args.lambda_l1 * recon_loss
                    + get_perceptual_weight(args) * lpips_loss
                    + args.lambda_feat * feat_loss
                    + args.lambda_feat_moment * feat_moment_loss
                    + args.lambda_mix * mix_loss
                    + args.lambda_codebook_entropy * codebook_entropy_loss
                    + args.lambda_codebook_ce * codebook_ce_loss
                    + args.lambda_gan * gan_factor * gan_g_loss
                )

            (loss / args.accum_steps).backward()

            d_loss = x_rec.new_zeros(())
            lecam_loss = x_rec.new_zeros(())
            logits_real_mean = x_rec.new_zeros(())
            logits_fake_mean = x_rec.new_zeros(())
            if train_discriminator:
                set_requires_grad(discriminator, True)
                real_for_d = discriminator_input(x_lg, args.llamagen_input_range).detach()
                fake_for_d = discriminator_input(x_rec.detach(), args.llamagen_input_range)
                logits_both = discriminator(torch.cat([real_for_d, fake_for_d], dim=0))
                logits_real, logits_fake = split_discriminator_logits(logits_both, chunks=2, dim=0)
                logits_real_mean = weighted_logits_mean(logits_real)
                logits_fake_mean = weighted_logits_mean(logits_fake)
                d_loss = gan_factor * hinge_d_loss(logits_real, logits_fake)
                if args.lecam_regularization_weight > 0.0:
                    lecam_loss = compute_lecam_loss(
                        logits_real_mean,
                        logits_fake_mean,
                        lecam_ema_real,
                        lecam_ema_fake,
                    ) * args.lecam_regularization_weight
                    d_loss = d_loss + lecam_loss
                    lecam_ema_real = lecam_ema_real * args.lecam_ema_decay + logits_real_mean.detach() * (1.0 - args.lecam_ema_decay)
                    lecam_ema_fake = lecam_ema_fake * args.lecam_ema_decay + logits_fake_mean.detach() * (1.0 - args.lecam_ema_decay)
                (d_loss / args.accum_steps).backward()

            metric_sums["loss"] += loss.detach().float().item()
            metric_sums["recon"] += recon_loss.detach().float().item()
            metric_sums["mse01"] += mse01_metric.detach().float().item()
            metric_sums["lpips"] += lpips_loss.detach().float().item()
            metric_sums["feat"] += feat_loss.detach().float().item()
            metric_sums["feat_moment"] += feat_moment_loss.detach().float().item()
            metric_sums["mix"] += mix_loss.detach().float().item()
            metric_sums["code_ent_loss"] += codebook_entropy_loss.detach().float().item()
            metric_sums["code_ce"] += codebook_ce_loss.detach().float().item()
            metric_sums["code_acc"] += codebook_acc.detach().float().item()
            metric_sums["gan_g"] += (args.lambda_gan * gan_factor * gan_g_loss).detach().float().item()
            metric_sums["d_loss"] += d_loss.detach().float().item()
            metric_sums["lecam"] += lecam_loss.detach().float().item()
            metric_sums["logits_real"] += logits_real_mean.detach().float().item()
            metric_sums["logits_fake"] += logits_fake_mean.detach().float().item()

            last_x_lg = x_lg
            last_x_rec = x_rec
            last_x_mix_rec = x_mix_rec
            last_f_1d_lg = f_1d_lg
            last_f_2d_lg = f_2d_lg

        grad_norm = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
        optimizer.step()
        if ema is not None:
            ema.update(model.module.latent_decoder if distributed else model.latent_decoder)
        if train_discriminator:
            torch.nn.utils.clip_grad_norm_(disc_params, args.max_grad_norm)
            optimizer_d.step()

        vals = torch.tensor(
            [
                metric_sums["loss"] / args.accum_steps,
                metric_sums["recon"] / args.accum_steps,
                metric_sums["mse01"] / args.accum_steps,
                metric_sums["lpips"] / args.accum_steps,
                metric_sums["feat"] / args.accum_steps,
                metric_sums["feat_moment"] / args.accum_steps,
                metric_sums["mix"] / args.accum_steps,
                metric_sums["code_ent_loss"] / args.accum_steps,
                metric_sums["code_ce"] / args.accum_steps,
                metric_sums["code_acc"] / args.accum_steps,
                metric_sums["gan_g"] / args.accum_steps,
                metric_sums["d_loss"] / args.accum_steps,
                metric_sums["lecam"] / args.accum_steps,
                metric_sums["logits_real"] / args.accum_steps,
                metric_sums["logits_fake"] / args.accum_steps,
                grad_norm.detach().float().item(),
            ],
            device=device,
            dtype=torch.float32,
        )
        if distributed:
            dist.all_reduce(vals, op=dist.ReduceOp.AVG)
        running["loss"] += vals[0].item()
        running["recon"] += vals[1].item()
        running["mse01"] += vals[2].item()
        running["lpips"] += vals[3].item()
        running["feat"] += vals[4].item()
        running["feat_moment"] += vals[5].item()
        running["mix"] += vals[6].item()
        running["code_ent_loss"] += vals[7].item()
        running["code_ce"] += vals[8].item()
        running["code_acc"] += vals[9].item()
        running["gan_g"] += vals[10].item()
        running["d_loss"] += vals[11].item()
        running["lecam"] += vals[12].item()
        running["logits_real"] += vals[13].item()
        running["logits_fake"] += vals[14].item()
        running["grad"] += vals[15].item()
        count += 1

        x_lg = last_x_lg
        x_rec = last_x_rec
        x_mix_rec = last_x_mix_rec
        f_1d_lg = last_f_1d_lg
        f_2d_lg = last_f_2d_lg

        if is_main and (step == 1 or step % args.log_every == 0):
            denom = max(count, 1)
            sec = (time.time() - start_time) / denom
            with torch.no_grad():
                stats = (
                    f"f1d mean/std {f_1d_lg.float().mean().item():.4f}/{f_1d_lg.float().std().item():.4f} "
                    f"f2d mean/std {f_2d_lg.float().mean().item():.4f}/{f_2d_lg.float().std().item():.4f} "
                    f"xrec mean/std {x_rec.float().mean().item():.4f}/{x_rec.float().std().item():.4f}"
                )
                stats += codebook_usage_stats(last_code_probs)
            mse01 = running["mse01"] / denom
            psnr01 = -10.0 * math.log10(max(mse01, 1e-12))
            msg = (
                f"Step {step:08d} | loss {running['loss']/denom:.5f} | recon {running['recon']/denom:.5f} | "
                f"mse01 {mse01:.5f} | psnr01 {psnr01:.2f} | "
                f"lpips {running['lpips']/denom:.5f} | feat {running['feat']/denom:.5f} | "
                f"feat_moment {running['feat_moment']/denom:.5f} | mix {running['mix']/denom:.5f} | code_ent_loss {running['code_ent_loss']/denom:.5f} | "
                f"code_ce {running['code_ce']/denom:.5f} | code_acc {running['code_acc']/denom:.4f} | "
                f"gan_g {running['gan_g']/denom:.5f} | d {running['d_loss']/denom:.5f} | "
                f"lecam {running['lecam']/denom:.5f} | "
                f"d_real {running['logits_real']/denom:.4f} | d_fake {running['logits_fake']/denom:.4f} | "
                f"grad {running['grad']/denom:.4f} | lr {current_lr:.6g} | lr_d {current_lr_d:.6g} | "
                f"cb_temp {current_codebook_temperature:.4f} | {sec:.3f}s/step | {stats}"
            )
            print(msg, flush=True)
            with log_path.open("a") as f:
                f.write(msg + "\n")
            running = {
                "loss": 0.0, "recon": 0.0, "mse01": 0.0, "lpips": 0.0, "feat": 0.0, "feat_moment": 0.0, "mix": 0.0,
                "code_ent_loss": 0.0, "code_ce": 0.0, "code_acc": 0.0,
                "gan_g": 0.0, "d_loss": 0.0, "lecam": 0.0,
                "logits_real": 0.0, "logits_fake": 0.0, "grad": 0.0,
            }
            count = 0
            start_time = time.time()

        if is_main and args.sample_every > 0 and (step == 1 or step % args.sample_every == 0):
            save_recon_grid(
                out_dir / "samples" / f"step_{step:08d}.png",
                x_lg,
                x_rec,
                x_mix_rec,
                args.sample_images,
                args.llamagen_input_range,
            )

        if is_main and args.save_every > 0 and step % args.save_every == 0:
            core = model.module if distributed else model
            core_d = discriminator.module if distributed and discriminator is not None else discriminator
            payload = {
                "model": core.trainable_state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
                "step": step,
                "lecam_ema_real": lecam_ema_real.detach().cpu(),
                "lecam_ema_fake": lecam_ema_fake.detach().cpu(),
            }
            if ema is not None:
                payload["model_ema"] = ema.state_dict()
            if core_d is not None:
                payload["discriminator"] = core_d.state_dict()
            if optimizer_d is not None:
                payload["optimizer_d"] = optimizer_d.state_dict()
            torch.save(payload, out_dir / "latest.pt")
            if args.save_step_checkpoints:
                torch.save(payload, out_dir / f"step_{step:08d}.pt")

    if is_main:
        core = model.module if distributed else model
        core_d = discriminator.module if distributed and discriminator is not None else discriminator
        payload = {
            "model": core.trainable_state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "step": args.max_steps,
            "lecam_ema_real": lecam_ema_real.detach().cpu(),
            "lecam_ema_fake": lecam_ema_fake.detach().cpu(),
        }
        if ema is not None:
            payload["model_ema"] = ema.state_dict()
        if core_d is not None:
            payload["discriminator"] = core_d.state_dict()
        if optimizer_d is not None:
            payload["optimizer_d"] = optimizer_d.state_dict()
        torch.save(payload, out_dir / "latest.pt")
        print(f"saved {out_dir / 'latest.pt'}", flush=True)
    if distributed:
        dist.destroy_process_group()


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--data-path", type=str, default="../ImageNet/train")
    parser.add_argument("--output-dir", type=str, default="results/titok_llamagen_recon_stage_a")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--reset-optimizer", action="store_true", default=False)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--random-crop", action="store_true", default=False)
    parser.add_argument("--random-flip", action="store_true", default=False)
    parser.add_argument("--titok-input-range", type=str, default="zero_1", choices=["zero_1", "minus1_1"])
    parser.add_argument("--llamagen-input-range", type=str, default="minus1_1", choices=["zero_1", "minus1_1"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--epochs", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--limit-samples", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-d", type=float, default=1e-4)
    parser.add_argument("--lr-scheduler", type=str, default="constant", choices=["constant", "cosine"])
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--min-lr", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--mixed-precision", type=str, default="bf16", choices=["bf16", "fp16", "none"])
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--image-loss", type=str, default="l1", choices=["l1", "l2"])
    parser.add_argument("--perceptual-loss", type=str, default="lpips", choices=["lpips", "convnext_s", "none"])
    parser.add_argument("--lambda-perceptual", type=float, default=None)
    parser.add_argument("--lambda-l1", type=float, default=1.0)
    parser.add_argument("--lambda-lpips", type=float, default=0.5)
    parser.add_argument("--lambda-feat", type=float, default=0.1)
    parser.add_argument("--lambda-feat-moment", type=float, default=0.0)
    parser.add_argument("--lambda-mix", type=float, default=0.0)
    parser.add_argument("--lambda-gan", type=float, default=0.0)
    parser.add_argument("--gan-start-step", type=int, default=20000)
    parser.add_argument("--gan-ramp-steps", type=int, default=0)
    parser.add_argument("--discriminator-factor", type=float, default=1.0)
    parser.add_argument("--d-every", type=int, default=1)
    parser.add_argument("--lecam-regularization-weight", type=float, default=0.001)
    parser.add_argument("--lecam-ema-decay", type=float, default=0.999)
    parser.add_argument("--discriminator-type", type=str, default="patch", choices=["patch", "multiscale_patch", "dino", "dinov1s", "patch_dino", "multiscale_patch_dino", "projected_convnext", "projected_gan", "official_projected_gan", "stylegan"])
    parser.add_argument("--stylegan-channel-multiplier", type=int, default=1)
    parser.add_argument("--projected-input-size", type=int, default=224)
    parser.add_argument("--projected-head-hidden", type=int, default=256)
    parser.add_argument("--projected-backbone", type=str, default="convnext_small", choices=["convnext_small", "convnext_base"])
    parser.add_argument("--projected-channels", type=int, default=256)
    parser.add_argument("--projected-normalize-features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--projected-loss-weights", type=float, nargs="*", default=[1.0, 1.0, 1.0, 1.0])
    parser.add_argument("--projected-pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--official-pg-input-size", type=int, default=224)
    parser.add_argument("--official-pg-im-res", type=int, default=256)
    parser.add_argument("--official-pg-cout", type=int, default=64)
    parser.add_argument("--official-pg-expand", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--official-pg-proj-type", type=int, default=2, choices=[0, 1, 2])
    parser.add_argument("--official-pg-num-discs", type=int, default=4, choices=[1, 2, 3, 4])
    parser.add_argument("--official-pg-separable", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--official-pg-patch", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--official-pg-pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--official-pg-diffaug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--official-pg-diffaug-policy", type=str, default="color,translation,cutout")
    parser.add_argument("--disc-scales", type=float, nargs="*", default=[1.0])
    parser.add_argument("--disc-loss-weights", type=float, nargs="*", default=[1.0])
    parser.add_argument("--disc-hidden-channels", type=int, default=128)
    parser.add_argument("--disc-num-stages", type=int, default=3)
    parser.add_argument("--dino-repo", type=str, default="../.cache/torch/hub/facebookresearch_dinov2_main")
    parser.add_argument("--dino-model", type=str, default="dinov2_vits14")
    parser.add_argument("--dino-ckpt", type=str, default=None)
    parser.add_argument("--dino-input-size", type=int, default=224)
    parser.add_argument("--dino-loss-weight", type=float, default=0.25)
    parser.add_argument("--dino-head-hidden", type=int, default=256)
    parser.add_argument("--dino-use-patch-tokens", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mask-ratio-min", type=float, default=0.1)
    parser.add_argument("--mask-ratio-max", type=float, default=0.9)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--save-step-checkpoints", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sample-every", type=int, default=200)
    parser.add_argument("--sample-images", type=int, default=8)
    parser.add_argument("--lg-latent-channels", type=int, default=256)
    parser.add_argument("--lg-head-channels", type=int, default=256)
    parser.add_argument("--latent-head-mode", type=str, default="feature", choices=["feature", "codebook"])
    parser.add_argument("--codebook-temperature", type=float, default=1.0)
    parser.add_argument("--codebook-temperature-final", type=float, default=None)
    parser.add_argument("--codebook-temperature-warmup-steps", type=int, default=0)
    parser.add_argument("--lambda-codebook-entropy", type=float, default=0.0)
    parser.add_argument("--codebook-entropy-target", type=float, default=0.0)
    parser.add_argument("--lambda-codebook-ce", type=float, default=0.0)
    parser.add_argument("--codebook-ce-label-smoothing", type=float, default=0.0)
    parser.add_argument("--titok-root", type=str, default="../1d-tokenizer")
    parser.add_argument("--titok-config", type=str, default="../1d-tokenizer/configs/infer/TiTok/titok_l32.yaml")
    parser.add_argument("--titok-ckpt", type=str, default="../1d-tokenizer/tokenizer_titok_l32.bin")
    parser.add_argument("--llamagen-root", type=str, default="../LlamaGen")
    parser.add_argument("--llamagen-ckpt", type=str, default="../LlamaGen/pretrained_models/vq_ds16_c2i.pt")
    parser.add_argument("--codebook-size", type=int, default=16384)
    parser.add_argument("--codebook-embed-dim", type=int, default=8)
    return parser


def parse_args():
    parser = build_parser()
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None)
    config_args, remaining = config_parser.parse_known_args()

    defaults = vars(parser.parse_args([]))
    config_values = {}
    if config_args.config is not None:
        config = OmegaConf.to_container(OmegaConf.load(config_args.config), resolve=True)
        if not isinstance(config, dict):
            raise ValueError(f"config must contain a mapping, got {type(config).__name__}")
        config_values = {key.replace("-", "_"): value for key, value in config.items()}
        unknown = sorted(set(config_values) - set(defaults))
        if unknown:
            raise ValueError(f"unknown config keys: {unknown}")

    parser.set_defaults(**config_values)
    args = parser.parse_args(remaining)
    args.config = config_args.config
    return args


if __name__ == "__main__":
    main(parse_args())
