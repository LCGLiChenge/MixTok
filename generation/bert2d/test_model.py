"""Regression tests for the sparse-only BERT interface."""
import unittest
import torch
from .model import BertSparse2D

class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(0)
        self.m=BertSparse2D(dim=32,depth=2,heads=4,dropout=0).eval()
        self.z=torch.randint(0,4096,(2,32))
        self.t=torch.tensor([[16384,2,3,0],[16384,6,7,8]])
        self.idx=torch.tensor([[1,9,45,0],[2,15,99,255]])
        self.valid=torch.tensor([[True,True,True,False],[True,True,True,True]])
        self.y=torch.tensor([3,5])
        self.f=torch.randn(2,256,16,16)

    def run_model(self,**kwargs):
        a=dict(completed_1d=self.z,input_tokens=self.t,route_indices=self.idx,
               route_valid=self.valid,labels=self.y,base_features=self.f)
        a.update(kwargs)
        return self.m(**a)

    def test_padding(self):
        a=self.run_model()
        t=self.t.clone(); t[0,3]=888
        idx=self.idx.clone(); idx[0,3]=200
        b=self.run_model(input_tokens=t,route_indices=idx)
        torch.testing.assert_close(a[self.valid],b[self.valid],atol=1e-6,rtol=1e-5)

    def test_sparse_permutation(self):
        p=torch.tensor([2,0,3,1])
        a=self.run_model()[:,p]
        b=self.run_model(input_tokens=self.t[:,p],route_indices=self.idx[:,p],route_valid=self.valid[:,p])
        torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)

    def test_global_features_matter(self):
        f=self.f.clone()
        f[:,:,8,8]+=torch.arange(256).float()[None,:]
        self.assertGreater((self.run_model()-self.run_model(base_features=f)).abs().max().item(),1e-7)

    def test_cfg_and_gradients(self):
        out=self.run_model(force_drop_ids=torch.ones(2,dtype=torch.bool))
        out[self.valid].square().mean().backward()
        for p in (self.m.class_embedding.weight,self.m.feature_projection.weight,
                  self.m.encoder.layer[0].attention.self.query.weight,self.m.head.weight):
            self.assertIsNotNone(p.grad)
            self.assertGreater(p.grad.abs().sum().item(),0)
        self.assertGreater(self.m.class_embedding.weight.grad[1000].abs().sum().item(),0)
        self.assertEqual(self.m.class_embedding.weight.grad[:1000].abs().sum().item(),0)

    def test_recompute(self):
        self.m.train()
        a=self.run_model()
        self.m.enable_recompute()
        b=self.run_model()
        torch.testing.assert_close(a,b)
        b.mean().backward()
        self.assertTrue(torch.isfinite(self.m.head.weight.grad).all())

if __name__=="__main__": unittest.main()
