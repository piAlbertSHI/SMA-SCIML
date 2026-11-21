import unittest
import numpy as np
import torch

from SciML.models.v12.learning_staged_physics import (FO, StagedHeat, pde_residual, energy_residual,
    interface_residuals, fit, exact)


class Analytic(torch.nn.Module):
    def forward(self, coords, blocks):
        x, tau = coords[:, :1], coords[:, 1:]
        previous = torch.tensor([FO[np.arange(b) % 4].sum() for b in blocks.tolist()], dtype=torch.float64)[:, None]
        fo = torch.tensor(FO, dtype=torch.float64)[blocks % 4, None]
        return torch.sin(torch.pi*x)*torch.exp(-torch.pi**2*(previous+fo*tau))


class StagedPhysicsTests(unittest.TestCase):
    def test_analytic_pde_energy_and_interfaces(self):
        model = Analytic()
        points = torch.tensor([[.2, .3], [.6, .8]]*8, dtype=torch.float64, requires_grad=True)
        blocks = torch.arange(8).repeat_interleave(2)
        np.testing.assert_allclose(pde_residual(model, points, blocks).detach(), 0., atol=1e-15)
        np.testing.assert_allclose(energy_residual(model, torch.tensor([.2, .6], dtype=torch.float64), 24).detach(), 0., atol=1e-15)
        for r in interface_residuals(model, torch.linspace(0, 1, 13, dtype=torch.float64)):
            np.testing.assert_allclose(r.detach(), 0., atol=2e-16)
        np.testing.assert_allclose(model(points, blocks).detach(), exact(points.detach().numpy(), blocks.numpy()), atol=1e-15)

    def test_hard_conditions_only_first_initial_and_every_boundary(self):
        model = StagedHeat()
        x = torch.linspace(0, 1, 11, dtype=torch.float64)
        coords = torch.stack((x, torch.zeros_like(x)), dim=1)
        np.testing.assert_array_equal(model(coords, torch.zeros(11, dtype=torch.long)).detach(), torch.sin(torch.pi*x)[:, None])
        self.assertFalse(torch.allclose(model(coords, torch.ones(11, dtype=torch.long)), torch.sin(torch.pi*x)[:, None]))
        edge = torch.tensor([[0., .3], [1., .7]]*8, dtype=torch.float64)
        np.testing.assert_allclose(model(edge, torch.arange(8).repeat_interleave(2)).detach(), 0, atol=2e-16)

    def test_pde_energy_and_interface_gradients_reach_parameters(self):
        model = StagedHeat()
        points = torch.tensor([[.2, .3], [.6, .8]]*8, dtype=torch.float64, requires_grad=True)
        blocks = torch.arange(8).repeat_interleave(2)
        residuals = [pde_residual(model, points, blocks), energy_residual(model, torch.tensor([.4], dtype=torch.float64))]
        residuals += list(interface_residuals(model, torch.linspace(.1, .9, 5, dtype=torch.float64)))
        for residual in residuals:
            grads = torch.autograd.grad((residual**2).mean(), tuple(model.parameters()), retain_graph=True)
            self.assertTrue(all(torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0.)

    def test_fit_replays_without_dense_truth_and_preserves_rng(self):
        coords = np.array([[.2, .3], [.7, .8]]*8)
        blocks = np.arange(8).repeat(2)
        labels = exact(coords, blocks)
        state = torch.random.get_rng_state().clone()
        a, ah = fit(coords, blocks, labels, seed=81, variant="full", steps=2)
        b, bh = fit(coords, blocks, labels, seed=81, variant="full", steps=2)
        self.assertEqual(ah, bh)
        np.testing.assert_array_equal(state, torch.random.get_rng_state())
        for k, value in a.state_dict().items():
            np.testing.assert_array_equal(value, b.state_dict()[k])


if __name__ == "__main__":
    unittest.main()
