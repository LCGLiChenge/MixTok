"""Zero-init spatial residual into 2D inputs; inherited 1D path is untouched."""
import torch
from torch import nn
from h20.model import TiTokSparseImageBert


class SpatialTiTokSparseImageBert(TiTokSparseImageBert):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.spatial_norm = nn.LayerNorm(256, elementwise_affine=False)
        self.spatial_projection = nn.Linear(256, self.dim, bias=False)
        nn.init.zeros_(self.spatial_projection.weight)
        object.__setattr__(self, '_feature_provider', None)

    def set_feature_provider(self, provider):
        # Not a child module: no accidental inclusion in DDP, EMA or optimizer.
        object.__setattr__(self, '_feature_provider', provider)

    def spatial_delta(self, features, indices, valid):
        if features.shape != (len(indices), 256, 16, 16):
            raise ValueError('expected [B,256,16,16] base features')
        flat = features.detach().flatten(2).transpose(1, 2)
        selected = flat.gather(1, indices.clamp(0, 255)[..., None].expand(-1, -1, 256))
        delta = self.spatial_projection(self.spatial_norm(selected))
        return delta * valid[..., None].to(delta.dtype)

    def forward_2d(self, completed_1d, input_tokens, route_indices, route_valid,
                   labels, force_drop_ids=None, base_features=None):
        self._validate_sparse_inputs(input_tokens, route_indices, route_valid)
        if base_features is None:
            if self._feature_provider is None:
                raise RuntimeError('spatial model requires the frozen 1D-only feature provider')
            base_features = self._feature_provider(completed_1d)
        batch, sparse_len = input_tokens.shape
        class_h = self._class_tokens(labels, force_drop_ids)
        one_d_h = self._one_d_embeddings(completed_1d)
        indices = route_indices.clamp(0, self.grid_tokens - 1)
        sparse_h = (self.embedding_2d(input_tokens) + self.pos_embedding_2d(indices)
                    + self.modality_embedding.weight[1][None, None])
        counts = route_valid.sum(1)
        if not bool(((counts == 64) | (counts == 128)).all()):
            raise ValueError('E117 requires K64 or K128')
        budget = torch.where(counts == 64, 1, 2)
        sparse_h = sparse_h + self.budget_embedding(budget)[:, None]
        # Zero-init preserves previous forward arithmetic and initial logits.
        sparse_h = sparse_h + self.spatial_delta(base_features, indices, route_valid).to(sparse_h.dtype)
        sparse_h = self.token_dropout(sparse_h)
        hidden = torch.cat((class_h, one_d_h, sparse_h), 1)
        prefix_valid = torch.ones((batch, 1 + self.titok_num_tokens), dtype=torch.bool, device=hidden.device)
        valid = torch.cat((prefix_valid, route_valid), 1)
        hidden = self._run_backbone(hidden, None, valid)
        return self.output_2d(hidden[:, 1 + self.titok_num_tokens:1 + self.titok_num_tokens + sparse_len]).float()


def create_model(conditioning, **kwargs):
    if conditioning == 'spatial':
        return SpatialTiTokSparseImageBert(**kwargs)
    if conditioning == 'control':
        return TiTokSparseImageBert(**kwargs)
    raise ValueError('unknown conditioning arm')
