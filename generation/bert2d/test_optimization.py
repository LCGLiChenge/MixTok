"""Theory arithmetic, baseline parity, strict resume, and CLI regression tests."""
import copy
import math
import unittest
from argparse import ArgumentParser
import torch
from torch import nn
from .optimization import (add_arguments, cli_arguments, factor, from_args, make_optimizer,
                           resolve, set_learning_rates, validate_saved_config, validate_saved_groups)
from .resume import restore_payload, restore_rng
from .runtime import optimizer_description


class OptimizationTest(unittest.TestCase):
    def test_3200_formula(self):
        r=resolve(3200); k=3200/448
        self.assertAlmostEqual(r['lr_new'],1e-4*math.sqrt(k))
        self.assertAlmostEqual(r['lr_pretrained'],1e-5*math.sqrt(k))
        self.assertAlmostEqual(r['betas'][0],2/7)
        self.assertAlmostEqual(r['betas'][1],5/7)
        self.assertAlmostEqual(r['eps'],1e-8/math.sqrt(k))
        self.assertAlmostEqual(r['weight_decay'],.03*math.sqrt(k))
        # Same first-order decay per sample, both LR groups.
        for name,lr in [('lr_new',1e-4),('lr_pretrained',1e-5)]:
            self.assertAlmostEqual(r[name]*r['weight_decay']/3200,lr*.03/448)

    def test_sample_clock(self):
        big,small=resolve(3200),resolve(448)
        for large_step,small_step in [(7,50),(14,100),(21,150)]:
            for group in ('fresh','pretrained'):
                self.assertAlmostEqual(factor(big,group,large_step),factor(small,group,small_step))
        self.assertEqual(factor(big,'pretrained',2),0.)
        self.assertGreater(factor(big,'pretrained',3),0.)
        self.assertEqual(factor(big,'pretrained',17),1.)

    def test_legacy_is_not_scaled(self):
        r=resolve(3200,'legacy')
        self.assertEqual(r['lr_new'],1e-4)
        self.assertEqual(r['betas'],[.9,.96])
        self.assertEqual(r['weight_decay'],.03)
        self.assertEqual(factor(r,'fresh',25),.5)
        self.assertEqual(factor(r,'pretrained',70),.5)

    def test_invalid_inputs_fail_closed(self):
        for batch in (0,-1,1.5,4800,6400):
            with self.assertRaises(ValueError):resolve(batch)
        for value in (0,-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):resolve(3200,lr_new=value)
        with self.assertRaises(ValueError):resolve(448,'unknown')

    def test_cli_round_trip_no_double_scaling(self):
        p=ArgumentParser();p.add_argument('--global-batch',type=int);add_arguments(p)
        a=p.parse_args(['--global-batch','3200'])
        b=p.parse_args(['--global-batch','3200']+cli_arguments(a))
        self.assertEqual(from_args(a),from_args(b))
        self.assertEqual(a.lr_new,1e-4)

    def test_resume_guard(self):
        old=dict(global_batch=3200,lr_new=1e-4,lr_pretrained=1e-5)
        validate_saved_config(old,resolve(3200,'legacy'))
        with self.assertRaisesRegex(ValueError,'old checkpoint'):
            validate_saved_config(old,resolve(3200))
        current=dict(old,optimization=resolve(3200))
        validate_saved_config(current,resolve(3200))
        with self.assertRaisesRegex(ValueError,'recipe mismatch'):
            validate_saved_config(current,resolve(448))

    def test_reference_batch_bit_exact_old_adamw(self):
        torch.manual_seed(3);torch.set_num_threads(2)
        core=nn.Sequential(nn.Linear(3,5),nn.Linear(5,2));clone=copy.deepcopy(core)
        fresh=['1.weight','1.bias'];audit=dict(fresh_parameters=fresh)
        recipe=resolve(448)
        opt=make_optimizer(core,audit,recipe)
        old=torch.optim.AdamW([
            dict(params=list(clone[1].parameters()),name='fresh',lr=0.,base_lr=1e-4),
            dict(params=list(clone[0].parameters()),name='pretrained',lr=0.,base_lr=1e-5)
        ],betas=(.9,.96),weight_decay=.03)
        x=torch.randn(8,3)
        for step in range(1,125):
            set_learning_rates(opt,recipe,step)
            for g in old.param_groups:
                f=min(1.,step/50) if g['name']=='fresh' else max(0.,min(1.,(step-20)/100))
                g['lr']=g['base_lr']*f
            for m,o in ((core,opt),(clone,old)):
                o.zero_grad();m(x).square().mean().backward();o.step()
        for a,b in zip(core.parameters(),clone.parameters()):
            torch.testing.assert_close(a,b,atol=0,rtol=0)
            for key in ('step','exp_avg','exp_avg_sq'):
                torch.testing.assert_close(opt.state[a][key],old.state[b][key],atol=0,rtol=0)

    def test_scaled_resume_exact_and_saved_groups_checked(self):
        torch.manual_seed(7)
        core=nn.Sequential(nn.Linear(3,5),nn.Dropout(.2),nn.Linear(5,2))
        audit=dict(fresh_parameters=['2.weight','2.bias']);recipe=resolve(3200)
        opt=make_optimizer(core,audit,recipe);x=torch.randn(8,3)
        def update(m,o,step):
            set_learning_rates(o,recipe,step);o.zero_grad();loss=m(x).square().mean()
            loss.backward();o.step();return loss.detach()
        for step in range(1,5):update(core,opt,step)
        tensors={'raw/'+n:v.clone() for n,v in core.state_dict().items()}
        for n,p in core.named_parameters():
            for k,v in opt.state[p].items():tensors['adam/'+n+'/'+k]=v.clone()
        tensors['rng/0/cpu']=torch.get_rng_state()
        meta=dict(step=4,optimizer=optimizer_description(opt,core),cursor={})
        validate_saved_groups(meta['optimizer'],recipe)
        broken=copy.deepcopy(meta['optimizer']);broken[0]['betas']=[.9,.96]
        with self.assertRaisesRegex(ValueError,'hyperparameters'):
            validate_saved_groups(broken,recipe)
        expected=update(core,opt,5)
        clone=copy.deepcopy(core);resumed=make_optimizer(clone,audit,recipe)
        state=restore_payload(clone,resumed,tensors,meta,0,restore_cuda=False)
        restore_rng(state);actual=update(clone,resumed,5)
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)
        for a,b in zip(core.parameters(),clone.parameters()):
            torch.testing.assert_close(a,b,atol=0,rtol=0)
            for key in ('step','exp_avg','exp_avg_sq'):
                torch.testing.assert_close(opt.state[a][key],resumed.state[b][key],atol=0,rtol=0)


if __name__=='__main__':unittest.main()
