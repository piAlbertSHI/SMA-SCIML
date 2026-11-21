import unittest
import numpy as np
import torch
import tempfile
from pathlib import Path

from SciML.models.v13.relaxation_readout import RelaxationMeanReadout,VolumeRelaxationMeanReadout,relaxation_numpy,RelaxationReadoutOperator
from SciML.models.v13.stage_thermal_readout import StageThermalReadoutOperator
from examples.multicycle_multistep import ROOT
from examples.train_hierarchical import training_data
from examples.mean_readout_study import cache
from examples.relaxation_readout_study import fit_head,predict


class RelaxationTests(unittest.TestCase):
    def setUp(self):
        self.control=torch.tensor([[.04,.0002,1.e6,1.,2.4,1.,0.,.1,.1]],dtype=torch.float64)

    def test_exact_initial_equilibrium_and_temperature_bounds(self):
        model=VolumeRelaxationMeanReadout()
        initial=torch.tensor([303.],dtype=torch.float64);fluid=torch.full((1,50),293.,dtype=torch.float64)
        value=model(initial,fluid,self.control)
        torch.testing.assert_close(value[:,0],initial,rtol=0,atol=0)
        expected=293.+10*torch.exp(-model.time_grid)
        torch.testing.assert_close(value[0],expected,rtol=0,atol=1e-11)
        self.assertTrue(((value>=293.)&(value<=303.)).all())
        torch.testing.assert_close(model(initial,fluid+10,self.control),fluid+10,rtol=0,atol=1e-11)

    def test_synthetic_rate_recovery_and_numpy_parity(self):
        truth=VolumeRelaxationMeanReadout();truth.rate.fill_(2.7)
        control=self.control.repeat(3,1);control[:,0]=torch.tensor([.032,.038,.042])
        initial=torch.tensor([300.,306.,310.],dtype=torch.float64)
        fluid=torch.linspace(293.,296.,50,dtype=torch.float64)[None].repeat(3,1)
        target=truth(initial,fluid,control);model=VolumeRelaxationMeanReadout()
        fit=model.fit(initial,fluid,control,target)
        self.assertLess(fit['fit_mean_rmse_K'],1e-6);self.assertAlmostEqual(float(model.rate),2.7,places=5)
        expected=relaxation_numpy(initial.numpy(),fluid.numpy(),(truth.rate*truth.geometry(control)).numpy())
        np.testing.assert_allclose(target.numpy(),expected,rtol=0,atol=1e-11)

    def test_volume_scaling_and_invalid_inputs(self):
        control=self.control.repeat(2,1);control[1,0]*=2
        scale=VolumeRelaxationMeanReadout.geometry(control)
        self.assertEqual(float(scale[0]/scale[1]),4.)
        model=RelaxationMeanReadout();initial=torch.tensor([300.],dtype=torch.float64)
        with self.assertRaises(ValueError):model(initial,torch.zeros(1,49),self.control)
        control=self.control.clone();control[:,1]=0
        with self.assertRaises(ValueError):model(initial,torch.zeros(1,50),control)

    def test_autograd_has_initial_fluid_and_rate_sensitivity(self):
        model=VolumeRelaxationMeanReadout();initial=torch.tensor([303.],dtype=torch.float64,requires_grad=True)
        fluid=torch.full((1,50),293.,dtype=torch.float64,requires_grad=True)
        model.rate.requires_grad_(True);model(initial,fluid,self.control)[:,-1].sum().backward()
        self.assertGreater(float(initial.grad.abs().sum()),0.)
        self.assertGreater(float(fluid.grad.abs().sum()),0.)
        self.assertGreater(float(model.rate.grad.abs()),0.)


class RelaxationOperatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(1)
        cls.arrays,_,_=training_data()
        cls.reference=StageThermalReadoutOperator.load(ROOT/'evidence/local_studies/sma_stage_readout/formal/s20261121/model.pt')
        cls.fit=cache(cls.reference.reference.backbone,cls.arrays,[0,1,2,4])
        cls.response,_=fit_head(cls.fit,'autonomous_thermal_response',True)
        cls.model=RelaxationReadoutOperator(cls.reference,cls.response,.5)

    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)

    def test_state_fluid_initial_stage4_and_spatial_shape_preserved(self):
        with torch.no_grad():
            q={'times':torch.tensor([0.,.3,1.],dtype=torch.float64),'positions':torch.tensor([0.,.17,1.],dtype=torch.float64)}
            x=self.arrays['initial'][:1];u=self.arrays['control'][:1]
            a=self.reference(x,u,3,**q);b=self.model(x,u,3,**q)
            for left,right in [(a['states'],b['states']),(a['fields'][:,1::2],b['fields'][:,1::2]),
                (a['fields'][:,:,1],b['fields'][:,:,1]),(a['fields'][:,:,:,0],b['fields'][:,:,:,0])]:
                torch.testing.assert_close(left,right,rtol=0,atol=0)
            torch.testing.assert_close(a['fields'][:,:,0].diff(dim=-1),b['fields'][:,:,0].diff(dim=-1),rtol=0,atol=1e-12)
            self.assertFalse(torch.equal(a['fields'][:,::2,0,1:],b['fields'][:,::2,0,1:]))

    def test_checkpoint_query_subset_and_causal_prefix(self):
        with tempfile.TemporaryDirectory() as temp,torch.no_grad():
            x=self.arrays['initial'][:1];u=self.arrays['control'][:1]
            t=self.response.time_grid[[0,15,49]];space=self.reference.stage2_readout.space_grid
            before=self.model(x,u,3,times=t,positions=space)
            path=Path(temp)/'model.pt';self.model.save(path);loaded=RelaxationReadoutOperator.load(path)
            after=loaded(x,u,3,times=t,positions=space)
            torch.testing.assert_close(before['fields'],after['fields'],rtol=0,atol=0)
            subset=loaded(x,u,1,times=t,positions=space[[0,17,100]])
            torch.testing.assert_close(subset['fields'],before['fields'][:,:2,:,:,[0,17,100]],rtol=0,atol=1e-12)
            with self.assertRaises(FileExistsError):loaded.save(path)
            with self.assertRaises(TypeError):loaded(x,u,1,truth=torch.zeros(1))

    def test_cached_inference_is_label_free_and_fit_excludes_stage4(self):
        changed={k:v.clone() for k,v in self.fit.items()};changed['truth'][changed['stage']==4]+=1000.
        other,_=fit_head(changed,'autonomous_thermal_response',True)
        torch.testing.assert_close(other.rate,self.response.rate,rtol=0,atol=0)
        before=predict(self.reference,self.response,self.fit,.5)
        changed['truth'][:]=float('nan');changed['observed_entry_state'][:]=float('nan')
        after=predict(self.reference,self.response,changed,.5)
        torch.testing.assert_close(before,after,rtol=0,atol=0)
