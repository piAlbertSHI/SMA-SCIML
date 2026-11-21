import unittest
from pathlib import Path
import numpy as np
import torch

from SciML.models.v12.learning_historical_rollout import HistoricalModel, project
from SciML.models.v12.multistep_historical import DifferentiableHistorical, project_tensor, balanced_state_loss

ROOT=Path(__file__).resolve().parents[2]


class MultistepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(1)
        cls.base=HistoricalModel.load(ROOT/'checkpoints/historical-rollout-r1/s20260910/model.pt')
        cls.model=DifferentiableHistorical(cls.base)
        with np.load(ROOT/'data/historical12_state.npz',allow_pickle=False) as d:
            cls.data={k:d[k].copy() for k in d.files if k!='field_truth'}

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_torch_transition_matches_frozen_numpy_model(self):
        for stage in range(1,6):
            i=int(np.flatnonzero(self.data['stage']==stage)[0]);c=self.data['control'][self.data['case_index'][i]]
            actual=self.model.predict(self.data['before'][i],c,stage)
            np.testing.assert_allclose(actual,self.base.predict(self.data['before'][i],c,stage),rtol=0,atol=1e-9)

    def test_last_step_loss_reaches_first_predicted_state(self):
        i=int(np.flatnonzero(self.data['split']=='train')[0])
        c=torch.tensor(self.data['control'][i:i+1]);state=torch.tensor(self.data['initial'][i:i+1])
        first=self.model.step(state,c,1);last=first
        for stage in [2,3,4,5,1,2,3,4]:last=self.model.step(last,c,stage)
        grad=torch.autograd.grad(last[:,102:153].square().mean(),first)[0]
        self.assertTrue(torch.isfinite(grad).all());self.assertGreater(float(grad.abs().max()),1e-8)

    def test_projection_preserves_temperature_and_has_gradients(self):
        state=np.arange(515,dtype=float)+500
        t=torch.tensor(state[None],requires_grad=True);p=project_tensor(t)
        np.testing.assert_array_equal(p.detach().numpy()[0],project(state))
        p[:,102:153].sum().backward();np.testing.assert_array_equal(t.grad[0,102:153],np.ones(51))
        self.assertEqual(float(p[0,102].detach()),602.)

    def test_block_balancing_and_inference_input_contract(self):
        prediction=torch.zeros(1,1,515);target=prediction.clone();target[:,:,102:153]=1
        loss=balanced_state_loss(prediction,target,torch.ones(529))
        self.assertAlmostEqual(float(loss),1/8)
        with self.assertRaises(TypeError):self.model.predict(np.zeros(515),np.zeros(9),1,future_truth=np.zeros(515))


if __name__=='__main__':unittest.main()
