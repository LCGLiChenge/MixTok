"""Small CPU tests; they do not establish real checkpoint or image quality."""
import copy

import unittest
import torch
from transformers import BertConfig, BertModel

from h20.model import (
    TiTokSparseImageBert, load_official_state, OfficialSamplingView,
)


def pair(attention="eager"):
    torch.manual_seed(71)
    config = BertConfig(vocab_size=26, hidden_size=32, num_hidden_layers=2,
                        num_attention_heads=4, intermediate_size=64,
                        max_position_embeddings=33, layer_norm_eps=1e-12,
                        pad_token_id=None, attn_implementation=attention)
    reference = BertModel(config, add_pooling_layer=False)
    reference.lm_head = torch.nn.Linear(32, 17, bias=True)
    state = {"model." + k: v for k, v in reference.state_dict().items()}
    target = TiTokSparseImageBert(dim=32, n_layer=2, n_head=4,
                                 bert_intermediate_size=64, num_classes=7,
                                 titok_vocab_size=17, llamagen_vocab_size=23,
                                 attention_implementation=attention)
    return reference.eval(), target.eval(), state


def test_exact_logits_and_load_accounting(attention, drop):
    reference, target, state = pair(attention)
    before = target.embedding_2d.weight.detach().clone()
    report = load_official_state(target, state, reference.config)
    assert report["source_tensors"] == len(state)
    assert torch.equal(before, target.embedding_2d.weight)
    ids = torch.randint(0, 18, (4, 32)); ids[0].fill_(17)
    labels = torch.tensor([0, 1, 5, 6])
    class_ids = (torch.full_like(labels, 7) if drop else labels) + 18
    with torch.no_grad():
        wanted = reference.lm_head(reference(torch.cat([class_ids[:, None], ids], 1))[0][:, 1:])
        got = target.forward_1d(ids, labels, torch.full_like(labels, int(drop)))
    torch.testing.assert_close(got, wanted, atol=1e-6, rtol=1e-6)
    assert torch.equal(got.argmax(-1), wanted.argmax(-1))


def test_reject_without_mutating_target(defect):
    reference, target, state = pair()
    before = {k: v.clone() for k, v in target.state_dict().items()}
    key = "model.lm_head.weight"
    state = dict(state)
    if defect == "missing": del state[key]
    elif defect == "extra": state["extra"] = torch.ones(1)
    elif defect == "shape": state[key] = state[key][:1]
    else:
        state[key] = state[key].clone(); state[key][0, 0] = float("nan")
    with unittest.TestCase().assertRaises(ValueError):
        load_official_state(target, state, reference.config)
    assert all(torch.equal(v, target.state_dict()[k]) for k, v in before.items())


def test_sampling_rng_adapter():
    reference, target, state = pair(); load_official_state(target, state, reference.config)
    ids = torch.full((3, 32), 17); labels = torch.tensor([0, 3, 6])
    view = OfficialSamplingView(target)
    for probability in (0.0, 0.4, 1.0):
        torch.manual_seed(123)
        dropped = torch.rand_like(labels, dtype=torch.float) < probability
        shifted = labels + 18; shifted[dropped] = 25
        expected = reference.lm_head(reference(torch.cat([shifted[:, None], ids], 1))[0][:, 1:])
        rng = torch.get_rng_state()
        torch.manual_seed(123)
        got = view(ids, labels, probability)
        assert torch.equal(torch.get_rng_state(), rng)
        torch.testing.assert_close(got, expected, atol=1e-6, rtol=1e-6)


def test_sparse_padding_and_1d_isolation():
    reference, target, state = pair(); load_official_state(target, state, reference.config)
    z1 = torch.randint(0, 17, (2, 32)); labels = torch.tensor([0, 6])
    index = torch.arange(128)[None].expand(2, -1).clone()
    valid = torch.ones_like(index, dtype=torch.bool); valid[0, 64:] = False
    z2 = torch.full_like(index, target.mask_token_2d)
    z2[~valid] = target.pad_token_2d
    before = target.forward_1d(z1, labels).detach().clone()
    logits = target.forward_2d(z1, z2, index, valid, labels)
    index[~valid] = 255
    other = target.forward_2d(z1, z2, index, valid, labels)
    torch.testing.assert_close(logits[valid], other[valid], atol=1e-6, rtol=1e-6)
    logits[valid].square().mean().backward()
    assert target.output_2d.weight.grad.abs().sum() > 0
    assert target.layers.layer[0].attention.self.query.weight.grad is not None
    assert target.output_1d.weight.grad is None
    assert torch.equal(before, target.forward_1d(z1, labels))
    with torch.no_grad(): target.embedding_2d.weight.add_(1)
    assert torch.equal(before, target.forward_1d(z1, labels))


def test_bad_architecture():
    with unittest.TestCase().assertRaises(ValueError): TiTokSparseImageBert(backbone_type="llama")
    with unittest.TestCase().assertRaises(ValueError): TiTokSparseImageBert(token_dropout_p=0.1)
    reference, target, state = pair()
    bad = copy.deepcopy(reference.config)
    bad.num_attention_heads = 8
    with unittest.TestCase().assertRaises(ValueError): load_official_state(target, state, bad)


class MigrationTests(unittest.TestCase):
    def test_logits(self):
        for attention in ("eager", "sdpa"):
            for drop in (False, True):
                with self.subTest(attention=attention, drop=drop):
                    test_exact_logits_and_load_accounting(attention, drop)

    def test_rejections(self):
        for defect in ("missing", "extra", "shape", "nan"):
            with self.subTest(defect=defect): test_reject_without_mutating_target(defect)

    def test_sampling_rng(self): test_sampling_rng_adapter()

    def test_sparse_stage(self): test_sparse_padding_and_1d_isolation()

    def test_architecture(self): test_bad_architecture()


if __name__ == "__main__": unittest.main()
