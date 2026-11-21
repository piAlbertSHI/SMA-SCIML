import tempfile
import unittest
from pathlib import Path

import torch

from examples.train_hierarchical import training_data
from examples.fit_thermal_readout import fit_readout
from SciML.models.v12.learning_historical_rollout import HistoricalModel
from SciML.models.v13.modal_operator import ModalStateOperator, ModalConfig
from SciML.models.v13.thermal_readout import ThermalReadoutOperator, thermal_features

ROOT=Path(__file__).resolve().parents[2]


class ThermalReadoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(1)
        cls.arrays,_,_=training_data()
        base=HistoricalModel.load(ROOT/'checkpoints/historical-rollout-r1/s20260910/model.pt')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(29)
            cls.source=ModalStateOperator(base,ModalConfig(width=32,attention_heads=4,latent_tokens=4,
                attention_layers=1,memory_size=48,spatial_modes=12)).eval()
        cls.model,_=fit_readout(cls.source,cls.arrays,[0,1,2,4],'residual')

    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)

    def test_exact_initial_state_fluid_and_checkpoint(self):
        x=self.arrays['initial'][:1];u=self.arrays['control'][:1]
        q={'times':torch.tensor([0.,.27,1.],dtype=torch.float64),'positions':torch.tensor([0.,.14,.8,1.],dtype=torch.float64)}
        with tempfile.TemporaryDirectory() as temp,torch.no_grad():
            old=self.source(x,u,2,**q);pred=self.model(x,u,2,**q)
            torch.testing.assert_close(pred['states'],old['states'],rtol=0,atol=0)
            torch.testing.assert_close(pred['fields'][:,:,1],old['fields'][:,:,1],rtol=0,atol=0)
            torch.testing.assert_close(pred['fields'][:,:,:,0],old['fields'][:,:,:,0],rtol=0,atol=0)
            path=Path(temp)/'m.pt';self.model.save(path);replay=ThermalReadoutOperator.load(path)(x,u,2,**q)
            torch.testing.assert_close(pred['fields'],replay['fields'],rtol=0,atol=0)
            with self.assertRaises(FileExistsError):self.model.save(path)

    def test_fitting_excludes_withheld_arrays(self):
        arrays={k:v.clone() for k,v in self.arrays.items()}
        for value in arrays.values():value[[3,5,6,7]]+=1000.
        other,_=fit_readout(self.source,arrays,[0,1,2,4],'residual')
        for name,value in self.model.readout.state_dict().items():
            torch.testing.assert_close(value,other.readout.state_dict()[name],rtol=0,atol=0)

    def test_material_label_and_mechanical_pressure_excluded(self):
        initial=self.source.initial_fields(self.arrays['states'][:1,0],2)
        controls=self.arrays['control'][:1].clone();stages=torch.tensor([2])
        a=thermal_features(initial,controls,stages)
        controls[:,[2,5,6,7,8]]+=100.
        torch.testing.assert_close(a,thermal_features(initial,controls,stages),rtol=0,atol=0)
        controls[:,3]+=.1
        self.assertFalse(torch.equal(a,thermal_features(initial,controls,stages)))

    def test_bad_queries_and_future_labels_rejected(self):
        with self.assertRaises(TypeError):self.model(self.arrays['initial'][:1],self.arrays['control'][:1],1,future_truth=self.arrays['fields'][:1])
        with self.assertRaises(ValueError):self.model(self.arrays['initial'][:1],self.arrays['control'][:1],1,
            times=torch.tensor([-.1]),positions=torch.tensor([0.,1.]))


if __name__=='__main__':unittest.main()
