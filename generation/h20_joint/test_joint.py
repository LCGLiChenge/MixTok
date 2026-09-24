"""Small CPU correctness tests, NOT evidence of image quality or H20 capacity."""
from copy import deepcopy
from collections import namedtuple
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from h20.model import TiTokSparseImageBert
from h20_joint.spatial import SpatialTiTokSparseImageBert
from h20_joint.structure import OutputSpatialTiTokSparseImageBert
from h20_joint.model import JointSystem
from h20_joint.objective import TrainingForward,optimizer_for,enter_joint,set_lrs
from h20_joint.checkpoint import save_latest,metadata,restore
from h20_joint.sampling import generate,MARGIN4
from h20_joint.routing import routed_grid


def kwargs():
    return dict(dim=32,n_layer=2,n_head=4,bert_intermediate_size=64,num_classes=7,
                titok_vocab_size=17,llamagen_vocab_size=23)


def tiny():return JointSystem(SpatialTiTokSparseImageBert(**kwargs())).eval()


def packet(b=2):
    index=torch.arange(128).expand(b,-1).clone();valid=torch.ones_like(index,dtype=torch.bool)
    valid[0,64:]=False;index[~valid]=-1
    selected=routed_grid(index,valid).view(b,1,16,16)
    base=torch.randn(b,256,16,16)
    return dict(base=base,mixed=base+torch.randn_like(base)*selected,ids=torch.zeros(b,32,dtype=torch.long),
                index=index,valid=valid,scores=torch.zeros(b,1,16,16),target=torch.randn(b,3,16,16))


class Provider:
    def features(self,ids):return torch.ones(len(ids),256,16,16)
    def render(self,features):return features[:,:3].sigmoid()


class JointTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(7);torch.set_num_threads(2)

    def test_zero_spatial_and_1d_isolation(self):
        baseline=TiTokSparseImageBert(**kwargs()).eval()
        z1=torch.randint(17,(2,32));labels=torch.tensor([0,6]);row=packet()
        tokens=torch.full_like(row['index'],baseline.mask_token_2d).masked_fill(~row['valid'],baseline.pad_token_2d)
        wanted1=baseline.forward_1d(z1,labels)
        wanted2=baseline.forward_2d(z1,tokens,row['index'],row['valid'],labels)
        for model in (SpatialTiTokSparseImageBert(**kwargs()),
                      OutputSpatialTiTokSparseImageBert(enhancement_kind='output_cross',**kwargs())):
            model.eval();model.load_state_dict(baseline.state_dict(),strict=False)
            got=model.forward_2d(z1,tokens,row['index'],row['valid'],labels,base_features=row['base'])
            torch.testing.assert_close(got,wanted2,atol=1e-6,rtol=1e-6)
            got[row['valid']].square().mean().backward()
            self.assertGreater(float(model.spatial_projection.weight.grad.abs().sum()),0)
            with torch.no_grad():model.spatial_projection.weight.normal_();model.embedding_2d.weight.normal_()
            #1D has no provider or2D dependency even after spatial parameters change.
            model.set_feature_provider(lambda _:(_ for _ in ()).throw(AssertionError('1D called spatial provider')))
            torch.testing.assert_close(model.forward_1d(z1,labels),wanted1,atol=0,rtol=0)

    def test_fusion_isolation_and_half_initialization(self):
        system=tiny();row=packet();row['base'].requires_grad_();row['mixed'].requires_grad_()
        fused,alpha=system.fusion(row['base'],row['mixed'],row['scores'],row['ids'],row['index'],row['valid'])
        torch.testing.assert_close(fused,row['base']+.5*(row['mixed']-row['base']),atol=0,rtol=0)
        self.assertEqual(int(torch.count_nonzero(alpha)),int(row['valid'].sum()))
        loss,metrics=TrainingForward(system,Provider())(None,True,row);loss.backward()
        self.assertTrue(all(p.grad is not None and not p.grad.count_nonzero() for p in system.generator.parameters()))
        self.assertGreater(float(system.fusion.output.weight.grad.abs().sum()),0)
        self.assertIsNone(row['base'].grad);self.assertIsNone(row['mixed'].grad)

    def test_separate_fusion_accumulation(self):
        first=tiny();second=deepcopy(first);row=packet(4)
        TrainingForward(first,Provider())(None,True,row)[0].backward()
        for start in (0,2):
            sub={k:v[start:start+2] for k,v in row.items()}
            (TrainingForward(second,Provider())(None,True,sub)[0]/2).backward()
        for a,b in zip(first.fusion.parameters(),second.fusion.parameters()):
            torch.testing.assert_close(a.grad,b.grad,atol=1e-7,rtol=1e-5)

    def test_ce_not_gated_and_warmup_groups(self):
        system=tiny();wrapper=TrainingForward(system,Provider());p=packet()
        batch=dict(z1d=torch.randint(17,(2,32)),z2d=torch.randint(23,(2,128)),
            route_indices=p['index'],route_valid=p['valid'],label=torch.tensor([0,6]))
        loss,_=wrapper(batch,True);loss.backward()
        self.assertGreater(float(system.generator.output_2d.weight.grad.abs().sum()),0)
        self.assertTrue(all(not p.grad.count_nonzero() for p in system.fusion.parameters()))
        opt=optimizer_for(system);set_lrs(opt,600,600)
        self.assertEqual([g['name'] for g in opt.param_groups],['new2d','fusion'])
        self.assertEqual(opt.param_groups[1]['lr'],0)
        enter_joint(opt,system);set_lrs(opt,601,600)
        self.assertGreater(opt.param_groups[1]['lr'],0)
        self.assertEqual(len({id(p) for g in opt.param_groups for p in g['params']}),len(list(system.parameters())))

    def test_roundtrip_next_adam_and_format_rejection(self):
        system=tiny();ema=deepcopy(system);opt=optimizer_for(system);enter_joint(opt,system)
        for p in system.parameters():p.grad=torch.full_like(p,.001)
        opt.step()
        with tempfile.TemporaryDirectory(prefix='joint-test-',dir='.') as directory:
            path=Path(directory);Usage=namedtuple('Usage','total used free')
            with patch('h20_joint.checkpoint.shutil.disk_usage',return_value=Usage(10**12,0,10**12)):
                save_latest(path,system,ema,opt,[dict(cpu=torch.get_rng_state(),cuda=torch.get_rng_state())],
                    dict(step=1,config=dict(world=1),cursor=dict(packed_pass=0,next_microbatch_offset=1)))
            other=tiny();e2=deepcopy(other);o2=optimizer_for(other)
            restore(path,other,e2,o2,metadata(path),rank=0,world=1)
            for p,q in zip(system.parameters(),other.parameters()):
                self.assertTrue(torch.equal(p,q));p.grad=torch.full_like(p,.002);q.grad=p.grad.clone()
            opt.step();o2.step()
            self.assertTrue(all(torch.equal(p,q) for p,q in zip(system.parameters(),other.parameters())))
            from h20.checkpoint import metadata as baseline_metadata
            with self.assertRaises(ValueError):baseline_metadata(path)
            with (path/'latest.safetensors').open('ab') as f:f.write(b'x')
            with self.assertRaises(ValueError):metadata(path)

    def test_margin_sampler_reproducible_padding(self):
        system=tiny();system.generator.set_feature_provider(Provider().features)
        row=packet();labels=torch.tensor([0,6]);trace=[]
        torch.manual_seed(321);out=generate(system.generator,row['ids'],row['index'],row['valid'],labels,MARGIN4,trace)
        rng=torch.get_rng_state();torch.manual_seed(321)
        repeat=generate(system.generator,row['ids'],row['index'],row['valid'],labels,MARGIN4)
        self.assertTrue(torch.equal(out,repeat));self.assertTrue(torch.equal(torch.get_rng_state(),rng))
        self.assertTrue((out[~row['valid']]==system.generator.pad_token_2d).all())
        self.assertEqual(trace[-1]['remaining'],0)
        self.assertTrue(all(t['previously_known_remasked']==0 for t in trace))


if __name__=='__main__':unittest.main()
