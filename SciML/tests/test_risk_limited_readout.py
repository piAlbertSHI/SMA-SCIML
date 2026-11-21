import tempfile
import unittest
from pathlib import Path

import torch

from SciML.models.v13.risk_limited_readout import RiskLimitedRelaxationOperator,shrink_correction
from SciML.models.v13.relaxation_readout import RelaxationReadoutOperator
from SciML.models.v13.stage_thermal_readout import StageThermalReadoutOperator
from examples.multicycle_multistep import ROOT
from examples.train_hierarchical import training_data
from examples.mean_readout_study import cache
from examples.risk_limited_readout_study import fit_response,predict
from examples.fit_risk_limited_readout import fit_arrays


class RiskLimitedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(1)
        cls.arrays,_,_=training_data()
        cls.reference=StageThermalReadoutOperator.load(ROOT/'evidence/local_studies/sma_stage_readout/formal/s20261121/model.pt')
        cls.fit=cache(cls.reference.reference.backbone,cls.arrays,[0,1,2,4])
        cls.response,cls.scale,cls.info,_=fit_response(cls.fit)
        cls.model=RiskLimitedRelaxationOperator(cls.reference,cls.response,error_scale=cls.scale)

    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)

    def test_projection_never_amplifies_or_reverses(self):
        delta=torch.tensor([[-2.,-.1,0.,.4,3.]],dtype=torch.float64);scale=torch.full((5,),.5,dtype=torch.float64)
        result=shrink_correction(delta,scale)
        torch.testing.assert_close(result,torch.tensor([[-1.5,0.,0.,0.,2.5]],dtype=torch.float64))
        # Conditional mathematical property only: if truth is within the
        # response interval, projection cannot worsen squared scalar error.
        base=torch.zeros_like(delta)
        for offset in (-.5,0.,.5):
            truth=delta+offset
            self.assertTrue(torch.all((base+result-truth).square()<=(base-truth).square()))
        with self.assertRaises(ValueError):shrink_correction(delta,-scale)

    def test_checkpoint_queries_and_frozen_outputs(self):
        with tempfile.TemporaryDirectory() as temp,torch.no_grad():
            x=self.arrays['initial'][:1];u=self.arrays['control'][:1]
            t=self.response.time_grid[[0,15,49]];space=self.reference.stage2_readout.space_grid
            before=self.reference(x,u,3,times=t,positions=space)
            actual=self.model(x,u,3,times=t,positions=space)
            for a,b in ((actual['states'],before['states']),(actual['fields'][:,1::2],before['fields'][:,1::2]),
                (actual['fields'][:,:,1],before['fields'][:,:,1]),(actual['fields'][:,:,:,0],before['fields'][:,:,:,0])):
                torch.testing.assert_close(a,b,rtol=0,atol=0)
            torch.testing.assert_close(actual['fields'][:,:,0].diff(dim=-1),before['fields'][:,:,0].diff(dim=-1),rtol=0,atol=1e-12)
            path=Path(temp)/'model.pt';self.model.save(path);loaded=RiskLimitedRelaxationOperator.load(path)
            torch.testing.assert_close(loaded.error_scale,self.scale,rtol=0,atol=0)
            replay=loaded(x,u,3,times=t,positions=space)
            torch.testing.assert_close(actual['fields'],replay['fields'],rtol=0,atol=0)
            subset=loaded(x,u,1,times=t,positions=space[[0,17,100]])['fields']
            torch.testing.assert_close(subset,actual['fields'][:,:2,:,:,[0,17,100]],rtol=0,atol=1e-12)
            with self.assertRaises(ValueError):RelaxationReadoutOperator.load(path)
            with self.assertRaises(FileExistsError):loaded.save(path)

    def test_fit_ignores_stage4_and_inference_ignores_all_labels(self):
        modified={k:v.clone() for k,v in self.fit.items()};modified['truth'][modified['stage']==4]+=1000.
        head,scale,_,_=fit_response(modified)
        torch.testing.assert_close(head.rate,self.response.rate,rtol=0,atol=0)
        torch.testing.assert_close(scale,self.scale,rtol=0,atol=0)
        before=predict(self.reference,head,scale,modified)
        modified['truth'][:]=float('nan');modified['observed_entry_state'][:]=float('nan')
        torch.testing.assert_close(predict(self.reference,head,scale,modified),before,rtol=0,atol=0)

    def test_case_deletions_and_scale_contract(self):
        self.assertEqual([r['deleted_case_index'] for r in self.info['case_deletions']],[0,1,2,4])
        self.assertFalse(self.info['coverage_calibrated'])
        self.assertTrue(torch.all(self.scale>=0))
        for invalid in (torch.zeros(49),torch.full((50,),float('nan')),torch.full((50,),-1.)):
            with self.assertRaises(ValueError):RiskLimitedRelaxationOperator(self.reference,self.response,error_scale=invalid)

    def test_expansion_interface_and_unselected_label_isolation(self):
        # Synthetic three-cycle tensors exercise the data contract only. They
        # are not new physical data or an accuracy/generalization experiment.
        arrays={k:v[:4].clone() for k,v in self.arrays.items()}
        arrays['fields']=torch.cat((arrays['fields'],arrays['fields'][:,:2]),dim=1)
        arrays['states']=torch.cat((arrays['states'],arrays['states'][:,:5]),dim=1)
        ids=['contract_a','contract_b','contract_c','contract_held_out']
        model,info,_=fit_arrays(self.reference,arrays,ids,[0,1,2])
        self.assertEqual(info['cycles_per_case'],3)
        self.assertEqual(info['fit_case_ids'],ids[:3])
        self.assertFalse(info['candidate_promoted'])
        arrays['fields'][3]=float('nan');arrays['states'][3]=float('nan')
        replay,_,_=fit_arrays(self.reference,arrays,ids,[0,1,2])
        torch.testing.assert_close(model.response.rate,replay.response.rate,rtol=0,atol=0)
        torch.testing.assert_close(model.error_scale,replay.error_scale,rtol=0,atol=0)
        with self.assertRaises(ValueError):fit_arrays(self.reference,arrays,ids,[0,0,2])
        with self.assertRaises(ValueError):fit_arrays(self.reference,arrays,ids,[0,1,3])
        arrays['states']=arrays['states'][:,:-1]
        with self.assertRaises(ValueError):fit_arrays(self.reference,arrays,ids,[0,1,2])
