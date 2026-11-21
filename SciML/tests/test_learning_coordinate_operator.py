import unittest
import numpy as np
import torch
from SciML.models.v12.learning_coordinate_operator import CoordinateOperator, interpolate, SpectralLayer


class CoordinateOperatorTests(unittest.TestCase):
    def test_initial_value_exact_and_nonuniform_coordinate_query(self):
        torch.manual_seed(2)
        model = CoordinateOperator().double()
        initial = torch.linspace(290., 295., 101, dtype=torch.float64).expand(2, 2, 101).clone()
        branch = torch.zeros((2, 221), dtype=torch.float64)
        positions = torch.tensor([0., .125, .511, 1.], dtype=torch.float64)
        times = torch.tensor([0., .41, 1.], dtype=torch.float64)
        prediction = model(branch, initial, times, positions)
        self.assertEqual(prediction.shape, (2, 2, 3, 4))
        np.testing.assert_allclose(prediction[:, :, 0].detach(), interpolate(initial, model.initial_grid, positions).detach(), atol=0, rtol=0)
        prediction[:, :, 1:].square().mean().backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.core.trunk.parameters()), 0)
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.core.field_branch.parameters()), 0)

    def test_spectral_weights_receive_gradients_and_resolution_changes(self):
        torch.manual_seed(3)
        layer = SpectralLayer(width=4, modes=3).double()
        value = torch.randn(2, 4, 17, dtype=torch.float64)
        out = layer(value)
        self.assertEqual(out.shape, value.shape)
        out.square().mean().backward()
        self.assertGreater(float(layer.weight.grad.abs().sum()), 0)
        self.assertEqual(layer(torch.randn(2, 4, 33, dtype=torch.float64)).shape, (2, 4, 33))

    def test_fourier_initial_condition_and_no_stage_ablation(self):
        branch = torch.zeros((1, 221), dtype=torch.float64)
        initial = torch.ones((1, 2, 101), dtype=torch.float64)*293
        x = torch.linspace(0, 1, 51, dtype=torch.float64)
        t = torch.tensor([0., 1.], dtype=torch.float64)
        model = CoordinateOperator("fourier").double()
        output = model(branch, initial, t, x)
        np.testing.assert_array_equal(output[:, :, 0].detach(), np.full((1, 2, 51), 293.))
        ablated = CoordinateOperator("deeponet_no_stage").double()
        changed = branch.clone(); changed[:, 204:206] = 100.
        np.testing.assert_array_equal(ablated(branch, initial, t, x).detach(), ablated(changed, initial, t, x).detach())
        with self.assertRaises(ValueError):
            model(branch, initial, torch.tensor([1.1], dtype=torch.float64), x)


if __name__ == "__main__":
    unittest.main()
