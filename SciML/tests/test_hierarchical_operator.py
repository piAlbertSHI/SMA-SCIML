import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from SciML.models.v12.learning_historical_rollout import HistoricalModel, mask
from SciML.models.v13.hierarchical_operator import HierarchicalStateOperator, OperatorConfig, joint_loss

ROOT=Path(__file__).resolve().parents[2]


class HierarchicalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(1)
        base=HistoricalModel.load(ROOT/'checkpoints/historical-rollout-r1/s20260910/model.pt')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(13)
            cls.model=HierarchicalStateOperator(base,OperatorConfig(width=32,attention_heads=4,latent_tokens=4,attention_layers=1,memory_size=48,query_chunk=7))
            cls.model.eval()
        with np.load(ROOT/'data/historical12_state.npz',allow_pickle=False) as d:
            ci=int(np.flatnonzero(d['split']=='train')[0]);use=d['case_index']==ci
            cls.initial=torch.tensor(d['initial'][ci:ci+1]);cls.control=torch.tensor(d['control'][ci:ci+1])
            cls.before=torch.tensor(d['before'][use]);cls.truth=torch.tensor(d['after'][use])[None]
        cls.times=torch.tensor([0.,.3,1.],dtype=torch.float64)
        cls.positions=torch.tensor([0.,.1,.4,.8,1.],dtype=torch.float64)

    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)

    def test_all_stage_masks_and_projection(self):
        for stage in range(1,6):
            x=self.before[stage-1:stage]
            out=self.model.step(x,self.control,stage)['state']
            np.testing.assert_allclose(out.detach().numpy()[:,~mask(stage)],x.numpy()[:,~mask(stage)],atol=1e-12,rtol=0)
            self.assertTrue(torch.equal(out[:,[0,50,101]],torch.zeros(1,3,dtype=out.dtype)))
            self.assertTrue(torch.all((out[:,153:204]>=0)&(out[:,153:204]<=1)))
            torch.testing.assert_close(out[:,414:],-out[:,414:].flip(-1),rtol=0,atol=0)

    def test_hard_initial_value_and_chunk_equivalence(self):
        x=self.before[1:2]
        out=self.model.step(x,self.control,2,times=self.times,positions=self.positions)
        from SciML.models.v12.learning_coordinate_operator import interpolate
        initial=interpolate(self.model.initial_fields(x,2),self.model.decoder.grid,self.positions)
        torch.testing.assert_close(out['field'][:,:,0],initial,rtol=0,atol=0)
        single=self.model.decoder(out['latent'],self.model.initial_fields(x,2),self.times[1:2],self.positions)
        torch.testing.assert_close(out['field'][:,:,1:2],single,rtol=1e-12,atol=1e-12)

    def test_cross_step_field_gradient_reaches_earlier_state_and_memory(self):
        first=self.model.step(self.initial,self.control,1)
        later=self.model.step(first['state'],self.control,2,first['memory'],times=self.times,positions=self.positions)
        loss=later['field'][:,:,1:].square().mean()
        gradients=torch.autograd.grad(loss,(first['state'],first['memory']))
        for g in gradients:
            self.assertTrue(torch.isfinite(g).all());self.assertGreater(float(g.abs().max()),1e-10)

    def test_causal_prefix_and_learned_restart(self):
        with torch.no_grad():
            one=self.model(self.initial,self.control,1)
            two=self.model(self.initial,self.control,2)
        torch.testing.assert_close(one['states'],two['states'][:,:4],rtol=0,atol=0)
        self.assertEqual(two['keys'],[(1,1),(1,2),(1,3),(1,4),(1,5),(2,1),(2,2),(2,3),(2,4)])
        fresh=self.model.step(two['states'][:,3],self.control,5)['state']
        self.assertGreater(float((fresh-two['states'][:,4]).abs().max().detach()),1e-10)

    def test_checkpoint_is_self_contained_and_exact(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'model.pt';self.model.save(path,metadata={'status':'unit_contract_only'})
            loaded=HierarchicalStateOperator.load(path)
            with torch.no_grad():
                a=self.model(self.initial,self.control,1,times=self.times,positions=self.positions)
                b=loaded(self.initial,self.control,1,times=self.times,positions=self.positions)
            for k in ('states','fields','final_memory'):torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)
            with self.assertRaises(FileExistsError):self.model.save(path)

    def test_rejects_future_label_inputs_bad_queries_and_partial_coordinates(self):
        with self.assertRaises(TypeError):self.model(self.initial,self.control,2,future_truth=self.truth)
        with self.assertRaises(ValueError):self.model(self.initial,self.control,2,times=self.times)
        with self.assertRaises(ValueError):self.model.step(self.initial,self.control,2,times=torch.tensor([-1.]),positions=self.positions)
        with self.assertRaises(ValueError):self.model.step(self.initial,self.control,1,times=self.times,positions=self.positions)

    def test_joint_loss_backpropagates_into_all_major_modules(self):
        self.model.zero_grad(set_to_none=True)
        out=self.model(self.initial,self.control,2,times=self.times,positions=self.positions)
        # Synthetic perturbation tests gradients only, not scientific accuracy.
        truth=out['fields'].detach()+.5
        loss=joint_loss(out,self.truth,self.model.feature_scale,field_truth=truth,
                        field_scale=torch.ones(2),positions=self.positions)['total']
        loss.backward()
        for name,module in self.model.named_children():
            self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().max()>0 for p in module.parameters()),name)
        self.assertFalse(any(b.requires_grad for b in self.model.buffers()))

    def test_curvature_loss_is_optional_and_supervised(self):
        out=self.model(self.initial,self.control,2,times=self.times,positions=self.positions)
        perturbation=.25*self.positions.view(1,1,1,1,-1).square()
        truth=out['fields'].detach()+perturbation
        base=joint_loss(out,self.truth,self.model.feature_scale,field_truth=truth,
                        field_scale=torch.ones(2),positions=self.positions)
        curved=joint_loss(out,self.truth,self.model.feature_scale,field_truth=truth,
                          field_scale=torch.ones(2),positions=self.positions,curvature_weight=.001,
                          curvature_channel_weights=[1.,0.],field_channel_weights=[1.,2.],
                          times=self.times,entry_time_weight=1.)
        self.assertNotIn('spatial_curvature',base)
        self.assertIn('spatial_curvature',curved)
        self.assertTrue(torch.isfinite(curved['spatial_curvature']))
        self.assertGreater(float(curved['spatial_curvature'].detach()),0.)

    def test_entry_field_loss_is_optional_and_supervised(self):
        out=self.model(self.initial,self.control,2,times=self.times,positions=self.positions)
        truth=out['fields'].detach()+.25
        base=joint_loss(out,self.truth,self.model.feature_scale,field_truth=truth,
                        field_scale=torch.ones(2),positions=self.positions)
        weighted=joint_loss(out,self.truth,self.model.feature_scale,field_truth=truth,
                            field_scale=torch.ones(2),positions=self.positions,
                            field_channel_weights=[1.,2.],entry_field_weight=1.)
        self.assertNotIn('entry_field',base)
        self.assertIn('entry_field',weighted)
        self.assertTrue(torch.isfinite(weighted['entry_field']))
        self.assertGreater(float(weighted['entry_field'].detach()),0.)
        self.assertGreater(float(weighted['total'].detach()),float(base['total'].detach()))

    def test_state_block_and_stage_weights_are_optional(self):
        out=self.model(self.initial,self.control,2,times=self.times,positions=self.positions)
        base=joint_loss(out,self.truth,self.model.feature_scale)
        weighted=joint_loss(out,self.truth,self.model.feature_scale,
                            state_block_weights=[1.,1.,1.,1.,1.,3.,3.,1.],
                            state_stage_weights=[1.,1.,1.,1.,1.,1.,1.25,1.5,1.75])
        self.assertTrue(torch.isfinite(weighted['state']))
        self.assertGreater(float((weighted['state']-base['state']).abs().detach()),0.)

    def test_sample_material_field_weights_are_optional(self):
        out=self.model(self.initial,self.control,2,times=self.times,positions=self.positions)
        truth=out['fields'].detach()+.25
        base=joint_loss(out,self.truth,self.model.feature_scale,field_truth=truth,
                        field_scale=torch.ones(2),positions=self.positions)
        weighted=joint_loss(out,self.truth,self.model.feature_scale,field_truth=truth,
                            field_scale=torch.ones(2),positions=self.positions,
                            field_sample_channel_weights=[[1.,2.]])
        self.assertTrue(torch.isfinite(weighted['field']))
        self.assertGreater(float((weighted['field']-base['field']).abs().detach()),0.)


if __name__=='__main__':unittest.main()
