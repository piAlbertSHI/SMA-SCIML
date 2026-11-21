import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from examples.train_hierarchical import training_data
from examples.modal_hierarchical import fit_prior
from SciML.models.v12.learning_historical_rollout import HistoricalModel
from SciML.models.v13.modal_operator import ModalConfig, ModalStateOperator

ROOT = Path(__file__).resolve().parents[2]


class ModalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1)
        cls.arrays, cls.ids, _ = training_data()
        base = HistoricalModel.load(ROOT/'checkpoints/historical-rollout-r1/s20260910/model.pt')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            cls.model = ModalStateOperator(base, ModalConfig(width=32, attention_heads=4, latent_tokens=4,
                attention_layers=1, memory_size=48, query_chunk=2, spatial_modes=12)).eval()

    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.threads)

    def test_query_restriction_and_initial_constraint(self):
        state = self.arrays['states'][:1, 0]
        times = torch.tensor([0., .33, 1.], dtype=torch.float64)
        positions = torch.linspace(0, 1, 101, dtype=torch.float64)
        with torch.no_grad():
            result = self.model.step(state, self.arrays['control'][:1], 2, times=times, positions=positions)
            sparse = self.model.decoder(result['latent'], self.model.initial_fields(state, 2), times, positions[::4])
        torch.testing.assert_close(result['field'][:, :, 0], self.model.initial_fields(state, 2), rtol=0, atol=0)
        torch.testing.assert_close(sparse, result['field'][..., ::4], rtol=0, atol=1e-12)
        with self.assertRaises(ValueError):
            self.model.decoder(result['latent'], self.model.initial_fields(state, 2), times, torch.tensor([float('nan')]))

    def test_field_gradient_crosses_state_and_memory(self):
        first = self.model.step(self.arrays['initial'][:1], self.arrays['control'][:1], 1)
        result = self.model.step(first['state'], self.arrays['control'][:1], 2, first['memory'],
            times=torch.tensor([0., .5, 1.], dtype=torch.float64), positions=torch.tensor([.13, .8], dtype=torch.float64))
        grads = torch.autograd.grad(result['field'][:, :, 1:].square().mean(), (first['state'], first['memory']))
        for grad in grads:
            self.assertTrue(torch.isfinite(grad).all()); self.assertGreater(float(grad.abs().max()), 1e-10)

    def test_checkpoint_and_causal_prefix(self):
        kwargs = dict(times=torch.tensor([0., .3, 1.], dtype=torch.float64), positions=torch.tensor([0., .17, 1.], dtype=torch.float64))
        with tempfile.TemporaryDirectory() as temp, torch.no_grad():
            path = Path(temp)/'model.pt'; self.model.save(path)
            loaded = ModalStateOperator.load(path)
            one = self.model(self.arrays['initial'][:1], self.arrays['control'][:1], 1, **kwargs)
            two = loaded(self.arrays['initial'][:1], self.arrays['control'][:1], 2, **kwargs)
            torch.testing.assert_close(one['states'], two['states'][:, :4], rtol=0, atol=0)
            torch.testing.assert_close(one['fields'], two['fields'][:, :2], rtol=0, atol=0)
            with self.assertRaises(FileExistsError): loaded.save(path)

    def test_inner_prior_cannot_see_validation_targets(self):
        selected = [0, 1, 2, 4, 5, 6]
        a = fit_prior(self.arrays, self.ids, selected)
        changed = {k: v.clone() for k, v in self.arrays.items()}
        changed['states'][[3, 7]] += 1000
        changed['initial'][[3, 7]] += 1000
        changed['control'][[3, 7]] += 1000
        b = fit_prior(changed, self.ids, selected)
        np.testing.assert_array_equal(a.mean, b.mean); np.testing.assert_array_equal(a.scale, b.scale)
        for stage in range(1, 6): np.testing.assert_array_equal(a.linear[stage].coef_, b.linear[stage].coef_)
        self.assertFalse(any(self.ids[i] in s for i in [3, 7] for s in a.training_ids))

    def test_refitted_full_prior_matches_archived_prior(self):
        actual = fit_prior(self.arrays, self.ids, list(range(8)))
        archived = HistoricalModel.load(ROOT/'checkpoints/historical-rollout-r1/s20260910/model.pt')
        np.testing.assert_allclose(actual.mean, archived.mean, rtol=0, atol=1e-12)
        np.testing.assert_allclose(actual.scale, archived.scale, rtol=0, atol=1e-12)
        for stage in range(1, 6):
            np.testing.assert_allclose(actual.linear[stage].coef_, archived.linear[stage].coef_, rtol=0, atol=1e-12)

    def test_decoder_cache_has_no_target_input_path(self):
        from examples.refine_modal import cache
        a = cache(self.model, self.arrays, [0])
        modified = {k: v.clone() for k, v in self.arrays.items()}
        modified['states'] += 1000.; modified['fields'] += 1000.
        b = cache(self.model, modified, [0])
        for index in (0, 1): torch.testing.assert_close(a[index], b[index], rtol=0, atol=0)
        self.assertFalse(torch.equal(a[2], b[2]))


if __name__ == '__main__': unittest.main()
