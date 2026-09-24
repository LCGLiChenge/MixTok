"""CPU checks for migration semantics, parameter isolation and AdamW restore."""
import tempfile
import unittest
from unittest.mock import patch
from collections import namedtuple
from pathlib import Path
from copy import deepcopy
import torch
from h20.model import TiTokSparseImageBert
from h20.training import optimizer_for,enter_joint,save_latest,update_ema
from h20.checkpoint import metadata,restore,stream_origin
from h20.launch import candidates
from h20.assets import materialize,digest

def tiny():
    return TiTokSparseImageBert(dim=32,n_layer=2,n_head=4,bert_intermediate_size=64,
        num_classes=7,titok_vocab_size=17,llamagen_vocab_size=23)

class PortabilityTests(unittest.TestCase):
    def test_transport_shards(self):
        import hashlib
        with tempfile.TemporaryDirectory(prefix='h20-parts-',dir='.') as directory:
            root=Path(directory);parts=[];payload=b''
            for i,block in enumerate([b'first bytes',b'next bytes']):
                path=root/f'part{i}';path.write_bytes(block);payload+=block
                parts.append(dict(filename=path.name,sha256=digest(path),size=len(block)))
            spec=dict(local_path='latest.safetensors',repo_id='fixture',revision='pinned',parts=parts,
                      sha256=hashlib.sha256(payload).hexdigest(),size=len(payload))
            materialize(spec,root/'result',lambda **kw:root/kw['filename'])
            self.assertEqual((root/'result').read_bytes(),payload)
            spec['sha256']='0'*64
            with self.assertRaises(RuntimeError):materialize(spec,root/'bad',lambda **kw:root/kw['filename'])
            self.assertFalse((root/'bad').exists())

    def test_capacity_global_batch(self):
        for world in [1,2,4,8]:
            c=candidates(2048,world)
            self.assertTrue(c and c==sorted(c,reverse=True))
            self.assertTrue(all(2048%(b*world)==0 for b in c))

    def test_stream_layout(self):
        meta={'config':dict(micro=128,world=4,accumulation=4),
              'cursor':dict(packed_pass=8,next_microbatch_offset=3728)}
        self.assertEqual(stream_origin(meta,micro=128,world=4,accumulation=4),(8,3728,True))
        self.assertEqual(stream_origin(meta,micro=256,world=8,accumulation=1),(9,0,False))

    def test_checkpoint_round_trip_and_next_adam_step(self):
        torch.manual_seed(21);m=tiny();ema=deepcopy(m);opt=optimizer_for(m);enter_joint(opt,m)
        for p in m.parameters():p.grad=torch.ones_like(p)*.001
        opt.step();update_ema(ema,m)
        with tempfile.TemporaryDirectory(prefix='h20-unit-',dir='.') as directory:
            d=Path(directory)
            # Tiny CPU fixture needs <1MB, not the production 10GiB reserve.
            Usage=namedtuple('Usage','total used free')
            with patch('h20.training.shutil.disk_usage',return_value=Usage(10**12,0,10**12)):
                save_latest(d,m,ema,opt,[dict(cpu=torch.get_rng_state(),cuda=torch.get_rng_state())],
                    dict(step=1,config=dict(world=1),cursor=dict(packed_pass=0,next_microbatch_offset=1)))
            m2=tiny();e2=deepcopy(m2);o2=optimizer_for(m2)
            restore(d,m2,e2,o2,metadata(d),rank=0,world=1)
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(m.parameters(),m2.parameters())))
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(ema.parameters(),e2.parameters())))
            for first,second in zip(m.parameters(),m2.parameters()):
                for key in ['step','exp_avg','exp_avg_sq']:
                    self.assertTrue(torch.equal(opt.state[first][key],o2.state[second][key]))
                first.grad=torch.ones_like(first)*.002;second.grad=torch.ones_like(second)*.002
            opt.step();o2.step()
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(m.parameters(),m2.parameters())))

if __name__=='__main__':unittest.main()
