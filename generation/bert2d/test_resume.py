"""CPU resume regression: uninterrupted Adam update and data cursor."""
import copy
import unittest
import torch
from torch import nn
from torch.utils.data import DataLoader
from .resume import restore_payload, restore_rng, resumed_stream
from .runtime import OffsetSampler
from .runtime import optimizer_description

class ResumeTest(unittest.TestCase):
    def test_next_update_matches_uninterrupted(self):
        torch.set_num_threads(2)
        torch.manual_seed(321)
        core=nn.Sequential(nn.Linear(3,8),nn.Dropout(.2),nn.Linear(8,2))
        opt=torch.optim.AdamW(core.parameters(),lr=1e-4,betas=(.9,.96),weight_decay=.03)
        x=torch.randn(4,3)
        def update(model,optim):
            optim.zero_grad()
            loss=model(x).square().mean()
            loss.backward()
            optim.step()
            return loss.detach().clone()
        for _ in range(4): update(core,opt)
        tensors={"raw/"+n:v.detach().clone() for n,v in core.state_dict().items()}
        for n,p in core.named_parameters():
            for k,v in opt.state[p].items(): tensors["adam/"+n+"/"+k]=v.clone()
        tensors["rng/0/cpu"]=torch.get_rng_state()
        meta=dict(step=4,optimizer=optimizer_description(opt,core),cursor=dict(packed_pass=0,next_microbatch_offset=4))
        expected=update(core,opt)
        clone=copy.deepcopy(core)
        resumed=torch.optim.AdamW(clone.parameters(),lr=.8)
        state=restore_payload(clone,resumed,tensors,meta,0,restore_cuda=False)
        restore_rng(state)
        actual=update(clone,resumed)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        for a,b in zip(core.parameters(),clone.parameters()):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
            for key in ("step","exp_avg","exp_avg_sq"):
                torch.testing.assert_close(opt.state[a][key],resumed.state[b][key],rtol=0,atol=0)
        broken=dict(tensors); del broken["adam/0.weight/exp_avg"]
        with self.assertRaisesRegex(ValueError,"incomplete Adam"):
            restore_payload(clone,resumed,broken,meta,0,restore_cuda=False)

    def test_cursor_does_not_replay_or_skip(self):
        class Base:
            def set_epoch(self,e): self.epoch=e
            def __len__(self): return 5
            def __iter__(self):
                for i in range(5): yield [self.epoch*5+i]
        sampler=OffsetSampler(Base())
        loader=DataLoader(list(range(100)),batch_sampler=sampler)
        stream=resumed_stream(loader,sampler,dict(packed_pass=2,next_microbatch_offset=3))
        rows=[next(stream) for _ in range(3)]
        self.assertEqual([int(x[0].item()) for x in rows],[13,14,15])
        self.assertEqual(rows[0][1],dict(packed_pass=2,next_microbatch_offset=4))
        self.assertEqual(rows[2][1],dict(packed_pass=3,next_microbatch_offset=1))

if __name__=="__main__": unittest.main()
