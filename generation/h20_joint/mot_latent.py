import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class LlamaGenLatentResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        groups = 32 if channels % 32 == 0 else 1
        self.net = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)


class TiTokToLlamaGenLatentDecoder(nn.Module):
    """Expand TiTok 1D quantized latents into LlamaGen feature or codebook space."""

    def __init__(
        self,
        titok_decoder,
        lg_latent_channels=256,
        head_channels=256,
        head_mode="feature",
        codebook_size=16384,
        codebook_temperature=1.0,
    ):
        super().__init__()
        self.width = titok_decoder.width
        self.grid_size = titok_decoder.grid_size
        self.num_latent_tokens = titok_decoder.num_latent_tokens
        self.token_size = titok_decoder.token_size

        self.decoder_embed = copy.deepcopy(titok_decoder.decoder_embed)
        self.class_embedding = nn.Parameter(titok_decoder.class_embedding.detach().clone())
        self.positional_embedding = nn.Parameter(titok_decoder.positional_embedding.detach().clone())
        self.mask_token = nn.Parameter(titok_decoder.mask_token.detach().clone())
        self.latent_token_positional_embedding = nn.Parameter(
            titok_decoder.latent_token_positional_embedding.detach().clone()
        )
        self.ln_pre = copy.deepcopy(titok_decoder.ln_pre)
        self.transformer = copy.deepcopy(titok_decoder.transformer)
        self.ln_post = copy.deepcopy(titok_decoder.ln_post)

        if head_mode not in {"feature", "codebook"}:
            raise ValueError(f"unsupported LlamaGen latent head mode {head_mode}")
        self.head_mode = head_mode
        self.lg_latent_channels = lg_latent_channels
        self.codebook_size = codebook_size
        self.codebook_temperature = codebook_temperature
        out_channels = lg_latent_channels if head_mode == "feature" else codebook_size
        self.lg_latent_head = nn.Sequential(
            nn.Conv2d(self.width, head_channels, kernel_size=1),
            LlamaGenLatentResBlock(head_channels),
            LlamaGenLatentResBlock(head_channels),
            nn.Conv2d(head_channels, out_channels, kernel_size=1),
        )
        self._init_head()
        self.load_titok_decoder_backbone(titok_decoder.state_dict())

    def _init_head(self):
        for module in self.lg_latent_head.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GroupNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def load_titok_decoder_backbone(self, state_dict):
        backbone_state = {
            key: value
            for key, value in state_dict.items()
            if not key.startswith("ffn.") and not key.startswith("conv_out.")
        }
        missing, unexpected = self.load_state_dict(backbone_state, strict=False)
        self.titok_init_missing_keys = tuple(missing)
        self.titok_init_unexpected_keys = tuple(unexpected)

    @staticmethod
    def _expand_token(token, batch_size):
        return token.unsqueeze(0).expand(batch_size, -1, -1)

    def forward_backbone(self, z_quantized):
        n, c, h, w = z_quantized.shape
        if h != 1 or w != self.num_latent_tokens:
            raise ValueError(f"expected z_quantized [B,C,1,{self.num_latent_tokens}], got {tuple(z_quantized.shape)}")

        x = z_quantized.reshape(n, c * h, w).permute(0, 2, 1)
        x = self.decoder_embed(x)
        batch_size, seq_len, _ = x.shape

        mask_tokens = self.mask_token.repeat(batch_size, self.grid_size ** 2, 1).to(x.dtype)
        cls = self._expand_token(self.class_embedding, batch_size).to(mask_tokens.dtype)
        mask_tokens = torch.cat([cls, mask_tokens], dim=1)
        mask_tokens = mask_tokens + self.positional_embedding.to(mask_tokens.dtype)
        x = x + self.latent_token_positional_embedding[:seq_len]
        x = torch.cat([mask_tokens, x], dim=1)

        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        for block in self.transformer:
            x = block(x)
        x = x.permute(1, 0, 2)
        x = x[:, 1 : 1 + self.grid_size ** 2]
        x = self.ln_post(x)
        return x.permute(0, 2, 1).reshape(batch_size, self.width, self.grid_size, self.grid_size).contiguous()

    def forward(self, z_quantized):
        hidden = self.forward_backbone(z_quantized)
        out = self.lg_latent_head(hidden)
        expected_channels = self.lg_latent_channels if self.head_mode == "feature" else self.codebook_size
        expected = (z_quantized.shape[0], expected_channels, self.grid_size, self.grid_size)
        if out.shape != expected:
            raise ValueError(f"expected LlamaGen {self.head_mode} output {expected}, got {tuple(out.shape)}")
        return out



class TiTokLlamaGenStage2(nn.Module):
    """TiTok stage2-style wrapper with LlamaGen as the image decoder.

    This mirrors TiTok's stage2 ownership:
    frozen TiTok encoder/quantizer/latent tokens -> trainable TiTokDecoder
    expansion path -> frozen downstream image decoder. The downstream decoder is
    not run under torch.no_grad(), so gradients still flow back to the latent
    head and TiTok decoder backbone.
    """

    def __init__(
        self,
        titok,
        llamagen_vq,
        lg_latent_channels=256,
        head_channels=256,
        head_mode="feature",
        codebook_size=16384,
        codebook_temperature=1.0,
    ):
        super().__init__()
        self.titok = titok
        self.llamagen_vq = llamagen_vq
        self.latent_decoder = TiTokToLlamaGenLatentDecoder(
            titok.decoder,
            lg_latent_channels=lg_latent_channels,
            head_channels=head_channels,
            head_mode=head_mode,
            codebook_size=codebook_size,
            codebook_temperature=codebook_temperature,
        )
        self.freeze_titok_encoder_quantizer()
        self.freeze_llamagen_vq()

    def freeze_titok_encoder_quantizer(self):
        self.titok.eval().requires_grad_(False)
        self.titok.latent_tokens.requires_grad_(False)
        return self

    def freeze_llamagen_vq(self):
        self.llamagen_vq.eval().requires_grad_(False)
        return self

    def train(self, mode=True):
        super().train(mode)
        self.titok.eval()
        self.llamagen_vq.eval()
        return self

    @torch.no_grad()
    def encode_titok(self, x_titok):
        self.titok.encoder.eval()
        self.titok.quantize.eval()
        z_quantized, result_dict = self.titok.encode(x_titok)
        return z_quantized.detach(), result_dict

    def llamagen_codebook_embedding(self):
        embedding = self.llamagen_vq.quantize.embedding.weight
        if getattr(self.llamagen_vq.quantize, "l2_norm", False):
            embedding = F.normalize(embedding, p=2, dim=-1)
        return embedding

    def soft_codebook_quant(self, logits):
        probs = torch.softmax(logits / self.latent_decoder.codebook_temperature, dim=1)
        embedding = self.llamagen_codebook_embedding().to(dtype=probs.dtype, device=probs.device)
        quant = torch.einsum("bkhw,kd->bdhw", probs, embedding)
        return quant, probs

    def decode(self, z_quantized):
        latent_out = self.latent_decoder(z_quantized)
        if self.latent_decoder.head_mode == "feature":
            f_1d_lg = latent_out
            x_rec = self.llamagen_vq.decoder(f_1d_lg)
            return x_rec, f_1d_lg, None, None
        quant_1d_lg, code_probs = self.soft_codebook_quant(latent_out)
        f_1d_lg = self.llamagen_vq.post_quant_conv(quant_1d_lg)
        x_rec = self.llamagen_vq.decoder(f_1d_lg)
        return x_rec, f_1d_lg, code_probs, latent_out

    def forward(self, x_titok):
        z_quantized, result_dict = self.encode_titok(x_titok)
        x_rec, f_1d_lg, code_probs, code_logits = self.decode(z_quantized)
        return x_rec, {
            "z_1d": z_quantized,
            "f_1d_lg": f_1d_lg,
            "code_probs": code_probs,
            "code_logits": code_logits,
            "titok_result": result_dict,
        }

    def trainable_state_dict(self):
        return self.latent_decoder.state_dict()

    def load_trainable_state_dict(self, state_dict, strict=True):
        return self.latent_decoder.load_state_dict(state_dict, strict=strict)
