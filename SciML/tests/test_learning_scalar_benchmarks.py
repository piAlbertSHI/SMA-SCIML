import unittest
import numpy as np
import torch
from tempfile import TemporaryDirectory
from pathlib import Path

from SciML.models.v12.learning_scalar_benchmarks import fit_tracked, classical_models, score_predictions
from SciML.models.v12.learning_ensemble import LearningEnsemble


class ScalarBenchmarkTests(unittest.TestCase):
    def data(self):
        x = np.arange(21, dtype=float).reshape(7, 3)/7
        y = np.column_stack((x[:, 0]**2, 100*np.sin(x[:, 1])))
        return x, y, [str(i) for i in range(7)]

    def test_diagnostic_labels_cannot_affect_fit_or_scaling(self):
        x, y, ids = self.data()
        rng = torch.random.get_rng_state().clone()
        threads = torch.get_num_threads()
        a = fit_tracked(x, y, ids, x+10, y+10, seed=9, epochs=5, members=2)
        b = fit_tracked(x, y, ids, x+10, y+1000, seed=9, epochs=5, members=2)
        np.testing.assert_array_equal(a.predict_members(x), b.predict_members(x))
        np.testing.assert_array_equal(a.x_scaler.mean_, x.mean(axis=0))
        np.testing.assert_array_equal(a.y_scaler.mean_, y.mean(axis=0))
        self.assertNotEqual(a.history[0]["curve"][-1]["calibration_standardized_mse"],
                            b.history[0]["curve"][-1]["calibration_standardized_mse"])
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertEqual(threads, torch.get_num_threads())

    def test_tracked_matches_base_and_raw_identity_checkpoint(self):
        x, y, ids = self.data()
        a = fit_tracked(x, y, ids, x, y, seed=9, epochs=5, members=2)
        b = LearningEnsemble(members=2, epochs=5, seed=9).fit(x, y, ids)
        np.testing.assert_array_equal(a.predict_members(x), b.predict_members(x))
        raw = fit_tracked(x, y, ids, x, y, seed=9, epochs=5, members=2, normalize_targets=False)
        np.testing.assert_array_equal(raw.y_scaler.transform(y), y)
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/"raw.pt"
            raw.save(path)
            np.testing.assert_array_equal(raw.predict_members(x), LearningEnsemble.load(path).predict_members(x))
        self.assertEqual(a.history[0]["curve"][0]["updates_completed"], 0)
        self.assertEqual(a.history[0]["curve"][-1]["updates_completed"], 5)

    def test_metric_direction_and_small_sample_tree_configuration(self):
        target = np.array([[1., 10.], [3., 20.]])
        baseline = np.array([[2., 15.], [2., 15.]])
        perfect = score_predictions(target, target, baseline)
        self.assertEqual(perfect["r2"], [1., 1.])
        self.assertEqual(perfect["rmse_skill_vs_train_mean"], [1., 1.])
        bad = score_predictions(target+100, target, baseline)
        self.assertTrue(all(x < 0 for x in bad["r2"]))
        boost = classical_models(1)["hist_gradient_boosting"].estimator
        self.assertEqual(boost.min_samples_leaf, 3)
        self.assertFalse(boost.early_stopping)
        with self.assertRaises(ValueError):
            score_predictions(target*np.nan, target, baseline)


if __name__ == "__main__":
    unittest.main()
