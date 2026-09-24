"""Output-side1D-only conditioning candidates; no existing model mutation.

Each retains the trained input spatial branch and adds a zero-ended residual
before the2D prediction head. Initial source forwards remain exactly unchanged.
These implementations are NOT quality-validated or ready for remote long training.
"""
import torch
from torch import nn
from h20_joint.spatial import SpatialTiTokSparseImageBert

KINDS=('output_linear','output_mlp','output_cross')


def add_preserving_layout(hidden,delta):
    # F.linear may use a different biased GEMM for a contiguous tensor than for
    # the original prefix-sliced view, changing FP32 rounding even at delta=0.
    # Retain that original layout without writing into the saved backbone state.
    result=torch.empty_strided(hidden.shape,hidden.stride(),dtype=hidden.dtype,device=hidden.device)
    return result.copy_(hidden+delta.to(hidden.dtype))


class OutputCondition(nn.Module):
    def __init__(self,kind,dim,heads,memory_scope='full'):
        super().__init__()
        if kind not in KINDS:
            raise ValueError('unknown output condition')
        self.kind=kind
        if memory_scope not in ('full','selected'):
            raise ValueError('unknown spatial memory scope')
        self.memory_scope=memory_scope
        self.feature_norm=nn.LayerNorm(256,elementwise_affine=False)
        if kind=='output_linear':
            self.to_hidden=nn.Linear(256,dim,bias=False)
        elif kind=='output_mlp':
            self.trunk=nn.Sequential(nn.Linear(256,dim),nn.SiLU())
            self.to_hidden=nn.Linear(dim,dim,bias=False)
        else:
            self.memory_projection=nn.Linear(256,dim)
            self.memory_position=nn.Embedding(256,dim)
            self.query_norm=nn.LayerNorm(dim,elementwise_affine=False)
            self.attention=nn.MultiheadAttention(dim,heads,dropout=0.,batch_first=True)
            self.to_hidden=nn.Linear(dim,dim,bias=False)
        nn.init.zeros_(self.to_hidden.weight)

    def forward(self,hidden,features,index,valid):
        if features.shape!=(len(index),256,16,16) or hidden.shape[:2]!=valid.shape:
            raise ValueError('wrong1D-only feature/query shapes')
        memory=self.feature_norm(features.detach().flatten(2).transpose(1,2))
        if self.kind=='output_cross':
            positions=torch.arange(256,device=features.device)
            memory=self.memory_projection(memory)+self.memory_position(positions)[None]
            query=self.query_norm(hidden)
            blocked=None
            if self.memory_scope=='selected':
                selected=torch.zeros(len(index),256,device=index.device,dtype=torch.long)
                selected.scatter_add_(1,index.clamp(0,255),valid.long())
                blocked=selected==0
            delta=self.attention(query,memory,memory,key_padding_mask=blocked,need_weights=False)[0]
        else:
            delta=memory.gather(1,index.clamp(0,255)[...,None].expand(-1,-1,256))
            if self.kind=='output_mlp':
                delta=self.trunk(delta)
        return self.to_hidden(delta)*valid[...,None].to(hidden.dtype)


class OutputSpatialTiTokSparseImageBert(SpatialTiTokSparseImageBert):
    def __init__(self,*,enhancement_kind,memory_scope='full',**kwargs):
        super().__init__(**kwargs)
        self.enhancement_kind=enhancement_kind
        self.enhancement=OutputCondition(enhancement_kind,self.dim,self.layers.config.num_attention_heads,memory_scope)

    def load_spatial_state(self,state):
        if bool(self.enhancement.to_hidden.weight.detach().count_nonzero()):
            raise ValueError('source migration requires a zero-ended new branch')
        expected={n for n in self.state_dict() if n.startswith('enhancement.')}
        if set(self.state_dict())-set(state)!=expected or set(state)-set(self.state_dict()):
            raise ValueError('source is not the complete original spatial model')
        report=self.load_state_dict(state,strict=False)
        if set(report.missing_keys)!=expected or report.unexpected_keys:
            raise ValueError('unaccounted model migration')
        return dict(added_keys=sorted(expected),added_parameters=sum(p.numel() for p in self.enhancement.parameters()),
                    zero_ended=True,source_all_tensors_loaded=True)

    def forward_2d(self,completed_1d,input_tokens,route_indices,route_valid,
                   labels,force_drop_ids=None,base_features=None):
        # Preserve the existing spatial forward arithmetic up to the2D head.
        self._validate_sparse_inputs(input_tokens,route_indices,route_valid)
        if base_features is None:
            if self._feature_provider is None:
                raise RuntimeError('requires frozen1D-only feature provider')
            base_features=self._feature_provider(completed_1d)
        batch,sparse_len=input_tokens.shape
        class_h=self._class_tokens(labels,force_drop_ids)
        one_d_h=self._one_d_embeddings(completed_1d)
        indices=route_indices.clamp(0,self.grid_tokens-1)
        sparse_h=(self.embedding_2d(input_tokens)+self.pos_embedding_2d(indices)
                  +self.modality_embedding.weight[1][None,None])
        counts=route_valid.sum(1)
        if not bool(((counts==64)|(counts==128)).all()):
            raise ValueError('E117 requires K64 or K128')
        budget=torch.where(counts==64,1,2)
        sparse_h=sparse_h+self.budget_embedding(budget)[:,None]
        sparse_h=sparse_h+self.spatial_delta(base_features,indices,route_valid).to(sparse_h.dtype)
        sparse_h=self.token_dropout(sparse_h)
        hidden=torch.cat((class_h,one_d_h,sparse_h),1)
        prefix_valid=torch.ones((batch,1+self.titok_num_tokens),dtype=torch.bool,device=hidden.device)
        valid=torch.cat((prefix_valid,route_valid),1)
        hidden=self._run_backbone(hidden,None,valid)
        sparse=hidden[:,1+self.titok_num_tokens:1+self.titok_num_tokens+sparse_len]
        sparse=add_preserving_layout(sparse,self.enhancement(sparse,base_features,indices,route_valid))
        return self.output_2d(sparse).float()
