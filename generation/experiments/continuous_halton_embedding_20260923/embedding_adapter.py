"""Continuous f1d -> official Halton token-embedding regression."""
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F

FORMAT = 'top_p_continuous_halton_adapter_v1'
OFFICIAL_SHA256 = '7fe25cb80b05743e8b42dacc61d88792a9ab5217d6bd756bef28821e0a5fc68f'


class EmbeddingAdapter(nn.Module):
    def __init__(self, channels=256, hidden=128, embedding_dim=768):
        super().__init__()
        self.config = dict(channels=channels, hidden=hidden, embedding_dim=embedding_dim)
        self.residual = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1), nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.SiLU(),
            nn.Conv2d(hidden, channels, 1))
        self.project = nn.Conv2d(channels, embedding_dim, 1)
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, features):
        if features.ndim != 4 or tuple(features.shape[1:]) != (self.config['channels'], 16, 16):
            raise ValueError('expected f1d [B,C,16,16]')
        return self.project(features + self.residual(features)).flatten(2).transpose(1, 2).contiguous()


def official_embedding(path: Path):
    from bert2d.paths import sha256
    if sha256(path) != OFFICIAL_SHA256:
        raise RuntimeError('official Halton checkpoint identity mismatch')
    state = torch.load(path, map_location='cpu', mmap=True, weights_only=True)['model_state_dict']
    clean = {}
    for key, value in state.items():
        while key.startswith(('module.', '_orig_mod.')):
            key = key.split('.', 1)[1]
        clean[key] = value
    table = clean['tok_emb.weight']
    if table.shape != (16385, 768) or not torch.equal(table, clean['head.weight']):
        raise RuntimeError('official embedding dimensions or tied head differ')
    table = table[:16384].float().clone().detach()
    variance = table.var(unbiased=False).clamp_min(1e-12)
    return table, variance


def regression_loss(prediction, ids, table, variance):
    target = F.embedding(ids.reshape(len(ids), 256), table)
    return F.mse_loss(prediction.float(), target.float()) / variance


def load_adapter(path, device, expected_top_p=None, require_steps=20000):
    state = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
    if state.get('format') != FORMAT or int(state['step']) != require_steps:
        raise RuntimeError('wrong continuous adapter format or training step')
    cfg = state['train_config']
    if cfg['official_halton_sha256'] != OFFICIAL_SHA256:
        raise RuntimeError('adapter target was not official Halton embedding')
    if expected_top_p and cfg['top_p_checkpoint_sha256'] != expected_top_p:
        raise RuntimeError('adapter tokenizer differs')
    adapter = EmbeddingAdapter(**state['model_config'])
    adapter.load_state_dict(state['model'], strict=True)
    return adapter.to(device).eval().requires_grad_(False), cfg
