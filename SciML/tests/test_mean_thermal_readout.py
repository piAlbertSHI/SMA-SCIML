import tempfile
import unittest
import json
import copy
from pathlib import Path

import torch

from examples.train_hierarchical import training_data
from examples.mean_readout_study import current_from_source, fit_mean, mean_prediction, formal, cache
from examples.stage_readout_study import gate
from SciML.models.v12.learning_historical_rollout import HistoricalModel
from SciML.models.v13.modal_operator import ModalStateOperator, ModalConfig
from SciML.models.v13.mean_thermal_readout import MeanThermalReadout, MeanThermalReadoutOperator, mean_features, shrink_by_sensitivity

ROOT=Path(__file__).resolve().parents[2]


class MeanReadoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(1)
        cls.arrays,_,_=training_data();cls.selected=[0,1,2,4,5,6]
        base=HistoricalModel.load(ROOT/'checkpoints/historical-rollout-r1/s20260910/model.pt')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(47)
            source=ModalStateOperator(base,ModalConfig(width=32,attention_heads=4,latent_tokens=4,
                attention_layers=1,memory_size=48,spatial_modes=12)).eval()
        cls.ref,cls.fit=current_from_source(source,cls.arrays,cls.selected)
        cls.candidate={'kind':'compact','rank':3,'alpha':1.}
        cls.head,_=fit_mean(cls.ref,cls.fit,cls.candidate)
        cls.model=MeanThermalReadoutOperator(cls.ref,cls.head)

    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)

    def test_frozen_paths_and_spatial_differences(self):
        x=self.arrays['initial'][:1];u=self.arrays['control'][:1]
        q={'times':torch.tensor([0.,.3,1.],dtype=torch.float64),'positions':torch.tensor([0.,.23,1.],dtype=torch.float64)}
        with torch.no_grad():
            a=self.ref(x,u,3,**q);b=self.model(x,u,3,**q)
            for left,right in [(a['states'],b['states']),(a['fields'][:,1::2],b['fields'][:,1::2]),
                (a['fields'][:,:,1],b['fields'][:,:,1]),(a['fields'][:,:,:,0],b['fields'][:,:,:,0])]:
                torch.testing.assert_close(left,right,rtol=0,atol=0)
            torch.testing.assert_close(torch.diff(a['fields'][:,:,0],dim=-1),torch.diff(b['fields'][:,:,0],dim=-1),rtol=0,atol=1e-12)

    def test_fit_excludes_stage4_targets(self):
        modified={k:v.clone() for k,v in self.fit.items()};modified['truth'][modified['stage']==4]+=1000.
        other,_=fit_mean(self.ref,modified,self.candidate)
        for k,v in self.head.state_dict().items():torch.testing.assert_close(v,other.state_dict()[k],rtol=0,atol=0)

    def test_checkpoint_and_query_subset(self):
        x=self.arrays['initial'][:1];u=self.arrays['control'][:1]
        t=torch.tensor([0.,.3,1.],dtype=torch.float64);grid=self.ref.stage2_readout.space_grid
        with tempfile.TemporaryDirectory() as temp,torch.no_grad():
            path=Path(temp)/'model.pt';self.model.save(path);loaded=MeanThermalReadoutOperator.load(path)
            a=self.model(x,u,2,times=t,positions=grid)['fields']
            b=loaded(x,u,2,times=t,positions=grid)['fields'];torch.testing.assert_close(a,b,rtol=0,atol=0)
            c=loaded(x,u,2,times=t,positions=grid[[0,23,100]])['fields']
            torch.testing.assert_close(c,b[:,:,:,:,[0,23,100]],rtol=0,atol=1e-12)
            with self.assertRaises(FileExistsError):self.model.save(path)

    def test_no_mechanical_material_or_cycle_feature(self):
        initial=self.fit['initial'][:1];u=self.fit['control'][:1].clone();a=mean_features(initial,u,'compact')
        u[:,[2,5,6,7,8]]+=100
        torch.testing.assert_close(a,mean_features(initial,u,'compact'),rtol=0,atol=0)

    def test_nonfinite_supervision_and_invalid_query_rejected(self):
        use=self.fit['stage']==2;head=MeanThermalReadout();target=torch.zeros((int(use.sum()),50),dtype=torch.float64);target[0,1]=float('nan')
        with self.assertRaises(ValueError):head.fit(self.fit['initial'][use],self.fit['control'][use],target)
        with self.assertRaises(ValueError):self.head(self.fit['initial'][:1],self.fit['control'][:1],torch.tensor([-1.]))

    def test_cached_inference_never_reads_true_fields_or_true_entry_states(self):
        changed={k:v.clone() for k,v in self.fit.items()}
        changed['truth']+=1000.;changed['observed_entry_state']+=1000.
        a=mean_prediction(self.ref,self.head,self.fit,self.candidate)
        b=mean_prediction(self.ref,self.head,changed,self.candidate)
        torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_context_and_latent_increment_checkpoint(self):
        x=self.arrays['initial'][:1];u=self.arrays['control'][:1]
        q={'times':torch.tensor([0.,.27,1.],dtype=torch.float64),'positions':torch.tensor([0.,.3,1.],dtype=torch.float64)}
        with tempfile.TemporaryDirectory() as temp,torch.no_grad():
            for kind in ('full_context','latent'):
                candidate={'kind':kind,'rank':3,'alpha':1.,'mode':'increment','entry':'autonomous'}
                head,_=fit_mean(self.ref,self.fit,candidate)
                model=MeanThermalReadoutOperator(self.ref,head,'increment');path=Path(temp)/(kind+'.pt');model.save(path)
                a=model(x,u,3,**q);b=MeanThermalReadoutOperator.load(path)(x,u,3,**q);ref=self.ref(x,u,3,**q)
                torch.testing.assert_close(a['fields'],b['fields'],rtol=0,atol=0)
                torch.testing.assert_close(a['fields'][:,:,:,0],ref['fields'][:,:,:,0],rtol=0,atol=0)
                torch.testing.assert_close(a['fields'][:,1::2],ref['fields'][:,1::2],rtol=0,atol=0)

    def test_kernel_equilibrium_and_checkpoint(self):
        candidate={'kind':'thermal_kernel','rank':3,'alpha':.1,'mode':'increment','entry':'autonomous'}
        head,_=fit_mean(self.ref,self.fit,candidate);use=self.fit['stage']==2
        initial=torch.full_like(self.fit['initial'][use],293.);state=self.fit['entry_state'][use].clone();state[:,406:414]=293.
        with torch.no_grad(),tempfile.TemporaryDirectory() as temp:
            torch.testing.assert_close(head(initial,self.fit['control'][use],head.time_grid,state),torch.zeros((len(initial),50),dtype=torch.float64),rtol=0,atol=0)
            model=MeanThermalReadoutOperator(self.ref,head,'increment');path=Path(temp)/'kernel.pt';model.save(path)
            loaded=MeanThermalReadoutOperator.load(path)
            for key,value in model.state_dict().items():torch.testing.assert_close(value,loaded.state_dict()[key],rtol=0,atol=0)

    def test_sensitivity_shrink_never_amplifies_or_reverses_correction(self):
        delta=torch.tensor([[0.,1.,-2.,.1]],dtype=torch.float64);central=torch.zeros_like(delta)
        members=torch.tensor([[[0.,.1,.1,1.]],[[0.,-.1,-.1,-1.]]],dtype=torch.float64)
        for mode in ('trajectory','pointwise'):
            out=shrink_by_sensitivity(delta,central,members,2.,mode)
            self.assertTrue(torch.all(out.abs()<=delta.abs()));self.assertTrue(torch.all(out*delta>=0))
        self.assertEqual(float(shrink_by_sensitivity(delta,central,members,2.,'pointwise')[0,-1]),0.)

    def test_sensitive_model_checkpoint_and_query_subset(self):
        members=[copy.deepcopy(self.head),copy.deepcopy(self.head)]
        members[0].mean_change[1:]+=.05;members[1].mean_change[1:]-=.05
        model=MeanThermalReadoutOperator(self.ref,self.head,'increment',members=members,
            sensitivity_strength=2.,sensitivity_mode='pointwise')
        x=self.arrays['initial'][:1];u=self.arrays['control'][:1]
        times=torch.tensor([0.,.17,.73,1.],dtype=torch.float64);grid=self.ref.stage2_readout.space_grid
        with torch.no_grad(),tempfile.TemporaryDirectory() as temp:
            a=model(x,u,3,times=times,positions=grid)['fields'];path=Path(temp)/'sensitive.pt';model.save(path)
            loaded=MeanThermalReadoutOperator.load(path)
            b=loaded(x,u,3,times=times,positions=grid[[0,20,100]])['fields']
            torch.testing.assert_close(b,a[:,:,:,:,[0,20,100]],rtol=0,atol=1e-12)
            ref=self.ref(x,u,3,times=times,positions=grid)['fields']
            torch.testing.assert_close(a[:,:,:,0],ref[:,:,:,0],rtol=0,atol=0)
            torch.testing.assert_close(a[:,1::2],ref[:,1::2],rtol=0,atol=0)

    def test_gate_rejects_missing_duplicate_and_nonfinite_records(self):
        ref={'stage2_mean_rmse_K':1.,'segments':[{'case_index':0,'cycle':1,'stage':2,'rmse_K':1.}]}
        for records in ([],ref['segments']*2,[{**ref['segments'][0],'rmse_K':float('nan')}],
                        [{**ref['segments'][0],'case_index':99}]):
            with self.assertRaises(ValueError):gate({**ref,'segments':records},ref)

    def test_failed_development_cannot_create_formal_run(self):
        with tempfile.TemporaryDirectory() as temp:
            development=Path(temp)/'dev';development.mkdir()
            (development/'summary.json').write_text(json.dumps({'passed_both_folds':False}))
            output=Path(temp)/'formal'
            with self.assertRaises(ValueError):formal(output,development)
            self.assertFalse(output.exists())

    def test_cache_accepts_extended_horizon_without_ignoring_exchanges(self):
        # Synthetic shape fixture only; it never enters scientific evidence.
        arrays={k:v[:1].clone() for k,v in self.arrays.items()}
        arrays['states']=torch.cat((arrays['states'],arrays['states'][:,-5:]),dim=1)
        arrays['fields']=torch.cat((arrays['fields'],arrays['fields'][:,-2:]),dim=1)
        value=cache(self.ref.reference.backbone,arrays,[0])
        self.assertEqual(value['cycle'].tolist(),[1,1,2,2,3,3])
        self.assertEqual(value['stage'].tolist(),[2,4,2,4,2,4])
        arrays['states']=arrays['states'][:,:-1]
        with self.assertRaises(ValueError):cache(self.ref.reference.backbone,arrays,[0])


if __name__=='__main__':unittest.main()
