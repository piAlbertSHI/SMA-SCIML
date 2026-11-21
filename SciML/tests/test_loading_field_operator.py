import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from SciML.models.v13.loading_field_operator import INITIAL, NEURAL_METHODS, NeuralLoadingOperator, PODRidgeLoading
from examples.loading_field_benchmark import stratified_split, training_arrays


class LoadingFieldContractTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def test_learned_forecasts_roundtrip_and_preserve_known_initial_frame(self):
        x = torch.tensor([[.038, .00019, 1e6, 0.], [.042, .00021, 1.2e6, 1.]])
        t, r = torch.tensor([0., .5, 1.]), torch.linspace(0, 1, 11)
        tolerance = {"rtol": 0, "atol": 0}
        for kind in NEURAL_METHODS:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                model = NeuralLoadingOperator(kind)
                model.set_scaling([.04, .0002, 1.1e6, .5], [.01, .0001, 1e5, .5], [1., 2., 3., .5])
                pred = model(x, t, r)
                self.assertEqual(pred.shape, (2, 4, 3, 11))
                initial = torch.tensor(INITIAL)[None, :, None].expand(2, 4, 11)
                torch.testing.assert_close(pred[:, :, 0], initial, **tolerance)
                model.normalized_change(x, t, r)[:, :, 1:].square().mean().backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
                checkpoint = Path(tmp) / "model.pt"
                model.save(checkpoint)
                torch.testing.assert_close(NeuralLoadingOperator.load(checkpoint)(x, t, r), pred, **tolerance)
                if kind == "fourier":
                    with self.assertRaises(ValueError):
                        model(x, t, torch.linspace(0, 1, 11).square())

    def test_pod_roundtrip_on_low_rank_fields(self):
        rng = np.random.default_rng(310)
        x = rng.normal(size=(32, 4))
        coords = np.linspace(0, 1, 5)
        later = np.linspace(0, 1, 9)
        def fields(controls, positions):
            return (controls[:, :, None, None] * positions[None, None, :, None]
                    * (1 - positions[None, None, None, :]))
        with threadpool_limits(limits=1), tempfile.TemporaryDirectory() as tmp:
            pod = PODRidgeLoading.fit(x, fields(x, coords), coords, coords, x.mean(0), x.std(0), np.ones(4), rank=2)
            y = fields(x[:4], later) + np.asarray(INITIAL)[None, :, None, None]
            result = pod.predict(x[:4], later, later)
            self.assertLess(np.sqrt(np.mean((result - y)**2)), .04)
            np.testing.assert_array_equal(result[:, :, 0], y[:, :, 0])
            path = Path(tmp) / "pod.npz"
            pod.save(path)
            np.testing.assert_array_equal(PODRidgeLoading.load(path).predict(x[:4], later, later), result)

    def test_splits_and_training_preprocessing_exclude_held_out_labels(self):
        controls = np.array([[.032 + .002 * i, (.032 + .002 * i) / 200, 1e6 + 100 * j, material]
                             for i in range(5) for material in (0, 1) for j in range(8)])
        ids = [f"case_{i}" for i in range(len(controls))]
        split = stratified_split(ids, controls)
        self.assertEqual({s: int(np.sum(split == s)) for s in np.unique(split)},
                         {"test": 30, "train": 20, "validation": 30})
        fields = np.random.default_rng(12).normal(size=(len(controls), 4, 50, 51))
        corpus = {"controls": controls, "fields": fields, "split": split}
        original = training_arrays(corpus)
        corpus["fields"][split != "train"] = np.nan
        for a, b in zip(original, training_arrays(corpus)):
            np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
