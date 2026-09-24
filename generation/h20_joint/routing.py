"""Unchanged sparse-mask and score-map semantics from local ablations."""
import hashlib
import numpy as np
import torch
MODES=('zero','aligned','shuffled')
SHUFFLE_SALT=b'e117-score-per-image-shuffle-v1'

def routed_grid(index, valid):
    if (index.ndim != 2 or index.shape != valid.shape or valid.dtype != torch.bool
            or index.dtype != torch.long or index.shape[1] != 128):
        raise ValueError('requires sparse long index/bool valid[B,128]')
    counts = valid.sum(1)
    if not bool(((counts == 64) | (counts == 128)).all()): raise ValueError('requiresK64/128')
    if bool(((index[valid] < 0) | (index[valid] >= 256)).any()): raise ValueError('invalid valid coordinate')
    grid = torch.zeros(len(index), 256, dtype=torch.long, device=index.device)
    grid.scatter_add_(1, index.masked_fill(~valid, 0), valid.long())
    if bool((grid > 1).any()): raise ValueError('duplicate route coordinate')
    return grid.bool()

def score_values(scores, completed_1d, mode):
    if mode not in MODES:
        raise ValueError('unknown score condition mode')
    if completed_1d.ndim != 2 or completed_1d.shape[1] != 32:
        raise ValueError('score condition requires completed32-code prefix')
    if scores.shape != (len(completed_1d), 1, 16, 16) or not torch.isfinite(scores).all():
        raise ValueError('score condition requires finite[B,1,16,16] E117 scores')
    if scores.device != completed_1d.device:
        raise ValueError('score/prefix device mismatch')
    flat = scores.detach().float().flatten(1)
    if mode == 'zero':
        return torch.zeros_like(flat)[..., None]
    if mode == 'shuffled':
        # Image-specific permutation from completed1D only. No labels, targets,
        # routes, batch position, epoch, or global Torch/NumPy RNG enter it.
        # Unlike one global permutation, the position embedding cannot learn a
        # single inverse map. This remains a diagnostic, not a proof of causality.
        ids = completed_1d.detach().cpu().numpy().astype('<i8', copy=False)
        permutations = []
        for row in ids:
            seed = int.from_bytes(hashlib.sha256(SHUFFLE_SALT+row.tobytes()).digest()[:8], 'little')
            permutations.append(np.random.Generator(np.random.PCG64(seed)).permutation(256))
        order = torch.as_tensor(np.stack(permutations), dtype=torch.long, device=scores.device)
        flat = flat.gather(1, order)
    return flat[..., None]
