import copy
import tempfile
import unittest
from pathlib import Path

import torch
from threadpoolctl import threadpool_limits

from examples.train_hierarchical import training_data
from examples.fit_thermal_readout import fit_readout
from examples.stage_readout_study import cache, fit_head, gate
from SciML.models.v12.learning_historical_rollout import HistoricalModel
from SciML.models.v13.modal_operator import ModalStateOperator, ModalConfig
from SciML.models.v13.stage_thermal_readout import StageThermalReadoutOperator

ROOT=Path(__file__).resolve().parents[2]


class StageReadoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(1)
        cls.arrays,_,_=training_data();cls.selected=[0,1,2,4]
        base=HistoricalModel.load(ROOT/'checkpoints/historical-rollout-r1/s20260910/model.pt')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(31)
            source=ModalStateOperator(base,ModalConfig(width=32,attention_heads=4,latent_tokens=4,
                attention_layers=1,memory_size=48,spatial_modes=12)).eval()
        cls.ref,_=fit_readout(source,cls.arrays,cls.selected,'residual')
        cls.fit=cache(source,cls.arrays,cls.selected)
        cls.candidate={'kind':'separate','rank':4,'alpha':10.}
        with threadpool_limits(limits=1):cls.head,_=fit_head(cls.ref,cls.fit,cls.candidate)
        cls.model=StageThermalReadoutOperator(cls.ref,cls.head)

    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)

    def test_stage4_and_propagated_state_are_exact_for_three_cycles(self):
        x=self.arrays['initial'][:1];u=self.arrays['control'][:1]
        with torch.no_grad():
            q={'times':torch.tensor([0.,.3,1.],dtype=torch.float64),'positions':torch.tensor([0.,.17,1.],dtype=torch.float64)}
            a=self.ref(x,u,3,**q);b=self.model(x,u,3,**q)
            for left,right in [(a['states'],b['states']),(a['fields'][:,1::2],b['fields'][:,1::2]),
                (a['fields'][:,:,1],b['fields'][:,:,1]),(a['fields'][:,:,:,0],b['fields'][:,:,:,0])]:
                torch.testing.assert_close(left,right,rtol=0,atol=0)
            self.assertFalse(torch.equal(a['fields'][:,::2,0,1:],b['fields'][:,::2,0,1:]))

    def test_stage2_fitting_cannot_use_stage4_targets(self):
        modified={k:v.clone() for k,v in self.fit.items()}
        modified['truth'][modified['stage']==4]+=1234.
        with threadpool_limits(limits=1):other,_=fit_head(self.ref,modified,self.candidate)
        for key,value in self.head.state_dict().items():
            torch.testing.assert_close(value,other.state_dict()[key],rtol=0,atol=0)

    def test_cache_excludes_withheld_cases(self):
        changed={k:v.clone() for k,v in self.arrays.items()}
        for value in changed.values():value[[3,5,6,7]]+=1000.
        other=cache(self.ref.backbone,changed,self.selected)
        for key,value in self.fit.items():torch.testing.assert_close(value,other[key],rtol=0,atol=0)

    def test_self_contained_checkpoint_and_scale_contract(self):
        with tempfile.TemporaryDirectory() as temp,torch.no_grad():
            path=Path(temp)/'model.pt';self.model.save(path);loaded=StageThermalReadoutOperator.load(path)
            for key,value in self.model.state_dict().items():torch.testing.assert_close(value,loaded.state_dict()[key],rtol=0,atol=0)
            with self.assertRaises(FileExistsError):self.model.save(path)
        with self.assertRaises(ValueError):StageThermalReadoutOperator(self.ref,self.head,-.1)

    def test_gate_rejects_segment_tradeoff_and_no_improvement(self):
        ref={'stage2_mean_rmse_K':1.,'segments':[{'case_index':0,'cycle':1,'stage':2,'rmse_K':.5},
                                              {'case_index':0,'cycle':2,'stage':2,'rmse_K':1.5}]}
        self.assertFalse(gate(ref,ref))
        changed=copy.deepcopy(ref);changed['stage2_mean_rmse_K']=.9;changed['segments'][0]['rmse_K']=.6
        changed['segments'][1]['rmse_K']=1.2
        self.assertFalse(gate(changed,ref))

    def test_spatial_mean_is_preserved_independently_of_query_set(self):
        model=StageThermalReadoutOperator(self.ref,self.head,preserve_spatial_mean=True)
        x=self.arrays['initial'][:1];u=self.arrays['control'][:1]
        times=torch.tensor([0.,.3,1.],dtype=torch.float64);grid=self.head.space_grid
        with torch.no_grad(),tempfile.TemporaryDirectory() as temp:
            a=self.ref(x,u,3,times=times,positions=grid)['fields']
            b=model(x,u,3,times=times,positions=grid)['fields']
            torch.testing.assert_close(a[:,:,0].mean(-1),b[:,:,0].mean(-1),rtol=0,atol=1e-12)
            query=model(x,u,3,times=times,positions=grid[[0,17,100]])['fields']
            torch.testing.assert_close(query,b[:,:,:,:,[0,17,100]],rtol=0,atol=1e-12)
            path=Path(temp)/'model.pt';model.save(path);loaded=StageThermalReadoutOperator.load(path)
            self.assertTrue(loaded.preserve_spatial_mean)
            torch.testing.assert_close(loaded(x,u,3,times=times,positions=grid)['fields'],b,rtol=0,atol=0)


if __name__=='__main__':unittest.main()
