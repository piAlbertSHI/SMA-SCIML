import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from examples.train_hierarchical import training_data
from examples.adapt_sma import observed_cache
from examples.diagnose_sma_rollout import decompose
from SciML.models.v12.learning_historical_rollout import HistoricalModel
from SciML.models.v13.modal_operator import ModalConfig, ModalStateOperator
from SciML.models.v13.selective_sma import SelectiveSmaOperator

ROOT=Path(__file__).resolve().parents[2]


class SelectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(1)
        cls.arrays,cls.ids,_=training_data()
        base=HistoricalModel.load(ROOT/'checkpoints/historical-rollout-r1/s20260910/model.pt')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(19)
            cls.source=ModalStateOperator(base,ModalConfig(width=32,attention_heads=4,latent_tokens=4,
                attention_layers=1,memory_size=48,spatial_modes=12)).eval()
        cls.query={'times':torch.tensor([0.,.3,1.],dtype=torch.float64),
                   'positions':torch.tensor([0.,.12,.6,1.],dtype=torch.float64)}

    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)

    def test_source_identity_then_exact_fluid_and_state_preservation(self):
        model=SelectiveSmaOperator.from_model(self.source)
        x=self.arrays['initial'][:1];u=self.arrays['control'][:1]
        with torch.no_grad():
            old=self.source(x,u,2,**self.query);same=model(x,u,2,**self.query)
            for k in ('states','fields','final_memory'):torch.testing.assert_close(old[k],same[k],rtol=0,atol=0)
        loss=model(x,u,2,**self.query)['fields'][:,:,0,1:].square().mean();loss.backward()
        for name,p in model.named_parameters():
            if not name.startswith('decoder.sma.'):self.assertIsNone(p.grad,name)
        optimizer=torch.optim.AdamW(model.decoder.sma.parameters(),lr=.001);optimizer.step()
        with torch.no_grad():after=model(x,u,2,**self.query)
        torch.testing.assert_close(old['states'],after['states'],rtol=0,atol=0)
        torch.testing.assert_close(old['fields'][:,:,1],after['fields'][:,:,1],rtol=0,atol=0)
        torch.testing.assert_close(old['fields'][:,:,:,0],after['fields'][:,:,:,0],rtol=0,atol=0)
        self.assertGreater(float((old['fields'][:,:,0,1:]-after['fields'][:,:,0,1:]).abs().max()),0.)

    def test_checkpoint_self_containment(self):
        model=SelectiveSmaOperator.from_model(self.source)
        with tempfile.TemporaryDirectory() as temp,torch.no_grad():
            path=Path(temp)/'model.pt';model.save(path);loaded=SelectiveSmaOperator.load(path)
            a=model(self.arrays['initial'][:1],self.arrays['control'][:1],1,**self.query)
            b=loaded(self.arrays['initial'][:1],self.arrays['control'][:1],1,**self.query)
            torch.testing.assert_close(a['fields'],b['fields'],rtol=0,atol=0)
            self.assertTrue(all(not p.requires_grad for n,p in loaded.named_parameters() if not n.startswith('decoder.sma.')))
            with self.assertRaises(FileExistsError):loaded.save(path)

    def test_observed_training_cache_ignores_unselected_cases(self):
        a=observed_cache(self.source,self.arrays,[0])
        changed={k:v.clone() for k,v in self.arrays.items()}
        for k in changed:changed[k][1:]+=1000
        b=observed_cache(self.source,changed,[0])
        for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=0,atol=0)

    def test_signed_error_cancellation_is_not_misreported(self):
        truth=np.zeros((2,50,101));p=np.ones_like(truth)*2
        p[:,1:]=1.
        d=decompose(p,truth)
        np.testing.assert_array_equal(d['entry_mse_K2'],[4.,4.])
        np.testing.assert_array_equal(d['evolution_mse_K2'],[1.,1.])
        np.testing.assert_array_equal(d['cross_mse_K2'],[-4.,-4.])
        np.testing.assert_array_equal(d['rmse_K'],[1.,1.])


if __name__=='__main__':unittest.main()
