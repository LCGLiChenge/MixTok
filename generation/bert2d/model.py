"""TiTok-compatible BERT: [class | feature256 | sparse2dK], 2D-only."""
from __future__ import annotations
import os
from pathlib import Path
os.environ.setdefault("USE_TF", "0")
import torch
from torch import nn
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertEncoder
from .paths import ASSETS, sha256

class BertSparse2D(nn.Module):
    codebook_size = llamagen_vocab_size = 16384
    mask_token_2d = pad_token_2d = 16384
    titok_num_tokens = 32
    grid_size, grid_tokens, max_sparse_tokens = 16, 256, 128
    def _validate(self, completed_1d, input_tokens, route_indices, route_valid, labels):
        if completed_1d.shape != (len(input_tokens), self.titok_num_tokens):
            raise ValueError("completed_1d must be [B,32] for feature lookup")
        if input_tokens.shape != route_indices.shape or input_tokens.shape != route_valid.shape:
            raise ValueError("sparse token/index/valid shapes must match")
        if input_tokens.ndim != 2 or input_tokens.shape[1] > self.max_sparse_tokens:
            raise ValueError("expected [B,K<=128] sparse selected tokens")
        if route_indices.dtype != torch.long or route_valid.dtype != torch.bool:
            raise ValueError("route index/valid dtype mismatch")
        if labels.shape != (len(input_tokens),):
            raise ValueError("labels must be [B]")
        if bool(((input_tokens[route_valid] < 0) | (input_tokens[route_valid] > self.mask_token_2d)).any()):
            raise ValueError("2D token outside content+mask range")
        if bool(((route_indices[route_valid] < 0) | (route_indices[route_valid] >= self.grid_tokens)).any()):
            raise ValueError("valid route index outside 16x16 grid")
    def _flat_features(self, base_features: torch.Tensor, batch: int) -> torch.Tensor:
        if base_features.shape != (batch, 256, self.grid_size, self.grid_size):
            raise ValueError("expected base_features [B,256,16,16]")
        if base_features.dtype != torch.float32 or not bool(torch.isfinite(base_features).all()):
            raise ValueError("expected finite FP32 frozen 1D features")
        return base_features.detach().flatten(2).transpose(1, 2)
    def set_feature_provider(self, provider) -> None:
        object.__setattr__(self, "_feature_provider", provider)
    def _cfg_logits(self, cond: torch.Tensor, uncond: torch.Tensor, scale: float, mode: str = "standard") -> torch.Tensor:
        if mode != "standard" or cond.shape != uncond.shape:
            raise ValueError("Halton adapter supports standard CFG with matching logits")
        return uncond + float(scale) * (cond - uncond)

    def __init__(self, dim=768, depth=24, heads=16, dropout=0.1):
        super().__init__()
        self.dim = dim
        cfg = BertConfig(hidden_size=dim, num_hidden_layers=depth,
            num_attention_heads=heads, intermediate_size=4*dim,
            hidden_act="gelu", hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout, layer_norm_eps=1e-12, is_decoder=False)
        cfg._attn_implementation = "sdpa"
        self.encoder = BertEncoder(cfg)
        self.input_norm = nn.LayerNorm(dim, eps=1e-12)
        self.input_dropout = nn.Dropout(dropout)
        self.class_embedding = nn.Embedding(1001, dim)
        self.class_position = nn.Parameter(torch.zeros(1,1,dim))
        self.token_type = nn.Parameter(torch.zeros(1,1,dim))
        self.token_embedding = nn.Embedding(16385,dim)
        self.spatial_embedding = nn.Embedding(256,dim)
        self.modality_embedding = nn.Embedding(2,dim)
        self.feature_norm = nn.LayerNorm(256, elementwise_affine=False)
        self.feature_projection = nn.Linear(256,dim)
        self.head = nn.Linear(dim,16384)
        self.apply(self._initialize)
        object.__setattr__(self, "_feature_provider", None)

    @staticmethod
    def _initialize(m):
        if isinstance(m,(nn.Linear,nn.Embedding)):
            nn.init.normal_(m.weight,std=0.02)
            if isinstance(m,nn.Linear) and m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m,nn.LayerNorm) and m.weight is not None:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def enable_recompute(self, layers=None):
        from torch.utils.checkpoint import checkpoint
        self.encoder.gradient_checkpointing = True
        selected = set(self.encoder.layer[:layers]) if layers is not None else set(self.encoder.layer)
        def recompute(function, *args):
            if getattr(function, "__self__", None) in selected:
                return checkpoint(function, *args, use_reentrant=False)
            return function(*args)
        self.encoder._gradient_checkpointing_func = recompute

    def forward_2d(self,completed_1d,input_tokens,route_indices,route_valid,labels,
                   force_drop_ids=None,base_features=None):
        self._validate(completed_1d,input_tokens,route_indices,route_valid,labels)
        if base_features is None:
            if self._feature_provider is None: raise RuntimeError("feature provider required")
            base_features = self._feature_provider(completed_1d)
        flat = self._flat_features(base_features,len(labels))
        if force_drop_ids is None: force_drop_ids = torch.zeros_like(labels,dtype=torch.bool)
        y = torch.where(force_drop_ids.bool(),1000,labels)
        cls = self.class_embedding(y)[:,None] + self.class_position
        memory = (self.feature_projection(self.feature_norm(flat))
                  + self.spatial_embedding.weight[None] + self.modality_embedding.weight[0])
        sparse = (self.token_embedding(input_tokens.clamp(0,16384))
                  + self.spatial_embedding(route_indices.clamp(0,255)) + self.modality_embedding.weight[1])
        h = self.input_dropout(self.input_norm(torch.cat((cls,memory,sparse),1)+self.token_type))
        valid = torch.cat((torch.ones((len(labels),257),dtype=torch.bool,device=labels.device),route_valid),1)
        mask = torch.zeros((len(labels),1,1,h.shape[1]),device=h.device,dtype=h.dtype)
        mask.masked_fill_(~valid[:,None,None],torch.finfo(h.dtype).min)
        h = self.encoder(h,attention_mask=mask,return_dict=True).last_hidden_state
        return self.head(h[:,257:]).float()

    forward = forward_2d

def load_titok_backbone(core,path=ASSETS/"weights/generator_titok_l32.bin"):
    state = torch.load(path,map_location="cpu",mmap=True,weights_only=True)
    mapped = {"encoder."+n:state["model.encoder."+n] for n in core.encoder.state_dict()}
    for suffix in ("weight","bias"):
        mapped["input_norm."+suffix] = state["model.embeddings.LayerNorm."+suffix]
    mapped["class_embedding.weight"] = state["model.embeddings.word_embeddings.weight"][4097:5098]
    mapped["class_position"] = state["model.embeddings.position_embeddings.weight"][:1][None]
    mapped["token_type"] = state["model.embeddings.token_type_embeddings.weight"][:1][None]
    result = core.load_state_dict(mapped,strict=False)
    if result.unexpected_keys: raise RuntimeError(result)
    actual = core.state_dict()
    if not all(torch.equal(actual[k].cpu(),v) for k,v in mapped.items()):
        raise RuntimeError("official tensor copy not exact")
    return dict(source=str(Path(path).resolve()),sha256=sha256(path),
        loaded_tensors=len(mapped),loaded_parameters=sum(v.numel() for v in mapped.values()),
        fresh_parameters=result.missing_keys,backbone_exact=True,
        trainable_parameters=sum(p.numel() for p in core.parameters()),
        initialization="official TiTok L32 BERT; fresh 2D/feature interfaces")
