from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
import torch
from SciML.models.v12.learning_pinn_inverse import (
    HeatInversePINN, analytic_sensitivity, exact_temperature, fit, predict, run,
)


class PINNInverseTests(unittest.TestCase):
    def test_hard_initial_and_boundary_conditions(self):
        model = HeatInversePINN()
        x = np.linspace(0,1,31)
        points = np.column_stack((x, np.zeros_like(x)))
        np.testing.assert_allclose(predict(model, points)[:,0], exact_temperature(x, 0, .12), atol=1e-14)
        for boundary in (0.,1.):
            np.testing.assert_allclose(predict(model, np.column_stack((np.full_like(x,boundary),x))), 0., atol=1e-14)

    def test_residual_has_parameter_and_second_derivative_gradient(self):
        model = HeatInversePINN()
        points = torch.tensor([[.2,.3],[.7,.8]], dtype=torch.float64, requires_grad=True)
        model.residual(points).square().mean().backward()
        self.assertTrue(torch.isfinite(model.log_diffusivity.grad))
        self.assertGreater(abs(float(model.log_diffusivity.grad)), 0.)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.network.parameters()))
        with self.assertRaises(ValueError):
            model.residual(points.detach())

    def test_analytic_sensitivity_and_nonidentifiability(self):
        x,t,a = np.array([.2,.4,.7]), np.array([.1,.5,.8]), .12
        derivative = (exact_temperature(x,t,a+1e-6)-exact_temperature(x,t,a-1e-6))/2e-6
        np.testing.assert_allclose(analytic_sensitivity(x,t,a), derivative, rtol=1e-8)
        np.testing.assert_array_equal(analytic_sensitivity(np.zeros(3),t,a), 0.)

    def test_fit_is_deterministic_restores_rng_and_data_only_cannot_fit_coefficient(self):
        x = np.array([[.2,.3],[.4,.5],[.8,.7]])
        y = exact_temperature(x[:,0],x[:,1],.12)[:,None]
        before = torch.random.get_rng_state().clone()
        threads = torch.get_num_threads()
        a,_ = fit(x,y,seed=4,epochs=3,lbfgs_steps=0)
        b,_ = fit(x,y,seed=4,epochs=3,lbfgs_steps=0)
        np.testing.assert_array_equal(predict(a,x),predict(b,x))
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
        self.assertEqual(threads,torch.get_num_threads())
        no_physics,_ = fit(x,y,seed=4,epochs=3,lbfgs_steps=0,physics=False)
        self.assertAlmostEqual(float(no_physics.diffusivity.detach()), .25)

    def test_bad_observations_and_existing_output_rejected(self):
        with self.assertRaises(ValueError):
            fit(np.zeros((3,2)),np.full((3,1),np.nan),seed=4)
        with self.assertRaises(ValueError):
            HeatInversePINN(-1.)
        with TemporaryDirectory() as temp:
            with self.assertRaises(FileExistsError):
                run(Path(temp))


if __name__ == "__main__":
    unittest.main()
