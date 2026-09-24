"""Auditable, in-memory official TiTok initialization; existing models unchanged."""
from __future__ import annotations

import copy
import os
from collections.abc import Mapping

os.environ.setdefault("USE_TF", "0")

import torch
from transformers.models.bert.modeling_bert import BertEncoder

from h20.base_model import E117SparseUnifiedMaskGIT


class TiTokSparseImageBert(E117SparseUnifiedMaskGIT):
    """One shared BERT, two stages; preserve upstream 1D embedding arithmetic."""

    def __init__(self, *, attention_implementation="eager", **kwargs):
        defaults = dict(
            dim=768, n_layer=24, n_head=16, titok_vocab_size=4096,
            titok_num_tokens=32, llamagen_vocab_size=16384, grid_size=16,
            max_sparse_tokens=128, num_classes=1000, class_dropout_prob=0.1,
            backbone_type="bert", bert_intermediate_size=3072,
            bert_layer_norm_eps=1e-12, token_dropout_p=0.0,
            attn_dropout_p=0.1, resid_dropout_p=0.1, grad_checkpointing=False,
            output_bias=True, zero_init_output=False,
        )
        defaults.update(kwargs)
        if defaults["backbone_type"] != "bert" or not defaults["output_bias"]:
            raise ValueError("official port requires BERT and prediction bias")
        if defaults["token_dropout_p"] != 0:
            raise ValueError("extra token dropout is not part of the official 1D path")
        if defaults["class_dropout_prob"] <= 0:
            raise ValueError("official null-class embedding must exist")
        super().__init__(**defaults)
        if attention_implementation not in ("eager", "sdpa"):
            raise ValueError("unsupported attention implementation")
        config = copy.deepcopy(self.layers.config)
        config._attn_implementation = attention_implementation
        self.layers = BertEncoder(config)
        self.layers.apply(self._init_weights)
        self.attention_implementation = attention_implementation

    def _one_d_embeddings(self, tokens):
        if tokens.ndim != 2 or tokens.shape[1] != self.titok_num_tokens:
            raise ValueError("wrong 1D sequence shape")
        if bool(torch.any(tokens < 0)) or bool(torch.any(tokens > self.mask_token_1d)):
            raise ValueError("1D token outside content+mask vocabulary")
        positions = torch.arange(self.titok_num_tokens, device=tokens.device)
        # Upstream BertEmbeddings: (word + token_type) + position, not vice versa.
        hidden = self.embedding_1d(tokens) + self.modality_embedding.weight[0][None, None]
        return hidden + self.pos_embedding_1d(positions)[None]

    def _class_tokens(self, labels, force_drop_ids):
        hidden = self.class_embedding(labels, train=self.training, force_drop_ids=force_drop_ids)
        hidden = hidden[:, :1] + self.modality_embedding.weight[0][None, None]
        return hidden + self.class_pos_embedding

    def _run_backbone(self, hidden, freqs, valid):
        if hidden.shape[:2] != valid.shape or freqs is not None:
            raise ValueError("invalid BERT hidden/valid/frequency inputs")
        # BertModel's SDPA mask converter returns None when every key is valid.
        # An explicit all-zero mask can choose a different BF16 CUDA kernel.
        if self.attention_implementation == "sdpa" and bool(valid.all()):
            hidden = self.bert_input_dropout(self.bert_input_norm(hidden))
            return self.layers(hidden, attention_mask=None, return_dict=False)[0]
        # Never remove the mask for mixed K64/K128 padded sparse batches.
        return super()._run_backbone(hidden, freqs, valid)


def load_official_state(model, state, source_config):
    """Validate all keys/shapes before copying. Never writes a checkpoint file."""
    if not isinstance(model, TiTokSparseImageBert):
        raise TypeError("use the exact-order TiTokSparseImageBert wrapper")
    if not isinstance(state, Mapping) or not all(isinstance(k, str) for k in state):
        raise TypeError("expected plain upstream state dict")
    target_config = model.layers.config
    for field in ("hidden_size", "num_hidden_layers", "num_attention_heads",
                  "intermediate_size", "hidden_act", "layer_norm_eps",
                  "position_embedding_type", "hidden_dropout_prob",
                  "attention_probs_dropout_prob"):
        if getattr(source_config, field) != getattr(target_config, field):
            raise ValueError("official architecture mismatch: " + field)
    if source_config._attn_implementation != model.attention_implementation:
        raise ValueError("official attention implementation mismatch")
    target = model.state_dict()
    mapped, used = {}, set()

    def source(name, shape):
        if name not in state:
            raise ValueError("missing official tensor: " + name)
        value = state[name]
        if not torch.is_tensor(value) or tuple(value.shape) != tuple(shape):
            raise ValueError("official shape/type mismatch: " + name)
        if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
            raise ValueError("official nonfinite/non-floating tensor: " + name)
        used.add(name)
        return value

    dim, vocab = model.dim, model.titok_vocab_size
    classes = model.class_embedding.num_classes
    words = source("model.embeddings.word_embeddings.weight", (vocab + classes + 2, dim))
    positions = source("model.embeddings.position_embeddings.weight", (model.titok_num_tokens + 1, dim))
    types = source("model.embeddings.token_type_embeddings.weight", (2, dim))
    mapped["embedding_1d.weight"] = words[:vocab + 1]
    mapped["class_embedding.embedding_table.weight"] = words[vocab + 1:]
    mapped["pos_embedding_1d.weight"] = positions[1:]
    mapped["class_pos_embedding"] = positions[:1].unsqueeze(0)
    mapped["modality_embedding.weight"] = types
    for name, tensor in target.items():
        if name.startswith("layers."):
            original = "model.encoder." + name.removeprefix("layers.")
        elif name.startswith("bert_input_norm."):
            original = "model.embeddings.LayerNorm." + name.removeprefix("bert_input_norm.")
        elif name.startswith("output_1d."):
            original = "model.lm_head." + name.removeprefix("output_1d.")
        else:
            continue
        mapped[name] = source(original, tensor.shape)
    new_prefixes = ("embedding_2d.", "pos_embedding_2d.", "budget_embedding.", "output_2d.")
    expected_new = {name for name in target if name.startswith(new_prefixes)}
    if set(state) != used:
        raise ValueError("unexpected official tensors: " + repr(sorted(set(state) - used)))
    if set(target) - set(mapped) != expected_new:
        raise ValueError("unaccounted target parameters")
    for name, value in mapped.items():
        if value.shape != target[name].shape:
            raise ValueError("target shape mismatch: " + name)
    result = model.load_state_dict(mapped, strict=False)
    if set(result.missing_keys) != expected_new or result.unexpected_keys:
        raise RuntimeError("load accounting mismatch")
    return dict(
        source_tensors=len(used), loaded_target_tensors=len(mapped),
        loaded_target_parameters=sum(v.numel() for v in mapped.values()),
        new_target_tensors=sorted(expected_new),
        new_target_parameters=sum(target[k].numel() for k in expected_new),
        total_parameters=model.parameter_count(),
        attention_implementation=model.attention_implementation,
        type1_note="source row is unused by upstream 1D; not pretrained for 2D",
    )


class OfficialSamplingView(torch.nn.Module):
    """Use the actual upstream ImageBert.generate with this forward adapter.

    Upstream draws class-drop uniforms even at probabilities zero and one.
    Preserve these draws so sample-by-sample RNG replay is meaningful.
    """

    def __init__(self, core):
        super().__init__()
        self.core = core
        self.image_seq_len = core.titok_num_tokens
        self.mask_token_id = core.mask_token_1d

    def forward(self, input_ids=None, condition=None, cond_drop_prob=0.1):
        force = (torch.rand_like(condition, dtype=torch.float) < cond_drop_prob).long()
        return self.core.forward_1d(input_ids, condition, force_drop_ids=force)
