"""Learning scope does not bypass scientific gates or leak physical conditions."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from SciML.models.v12.run_learning_existing import (
    comparison, conformal_radius, metrics, propagate, run, split_conditions, write_json,
)


class LearningExistingTests(unittest.TestCase):
    def test_split_is_unique_deterministic_disjoint_and_order_invariant(self):
        ids = [f"condition:{i}" for i in range(63)]
        splits = split_conditions(ids)
        self.assertEqual({k: len(v) for k, v in splits.items()},
                         {"train": 37, "calibration": 13, "test": 13})
        self.assertEqual(splits, split_conditions(list(reversed(ids))+ids))
        flat = sum(splits.values(), [])
        self.assertEqual(len(flat), len(set(flat)))
        self.assertEqual(set(flat), set(ids))
        with self.assertRaises(ValueError):
            split_conditions(ids[:10])

    def test_metric_rejects_nonfinite_and_does_not_hide_negative_skill(self):
        for bad in (np.nan, np.inf):
            with self.assertRaises(ValueError):
                metrics([bad], [1.])
        with self.assertRaises(ValueError):
            metrics([], [])
        self.assertEqual(comparison([3.], [2.], [1.])["rmse_skill"], -1.)
        self.assertIsNone(comparison([3.], [1.], [1.])["rmse_skill"])

    def test_conformal_uses_calibration_order_statistic_across_all_targets(self):
        residual = np.column_stack((np.arange(1,14), np.arange(1,14)*2))
        q, rank = conformal_radius(residual, np.array([1., 2.]))
        self.assertEqual((q, rank), (13., 13))
        for scale in ([0.,1.], [np.inf,1.], [1.]):
            with self.assertRaises(ValueError):
                conformal_radius(residual, scale)
        with self.assertRaises(ValueError):
            conformal_radius(residual[:1], [1.,2.])

    def test_propagation_uses_previous_prediction_not_future_observed_state(self):
        records = [SimpleNamespace(case_id=c, stage_index=s,
            state_in=np.array([0. if s == 1 else 999.]), controls=np.array([s]))
            for c in ("a", "b") for s in range(1,5)]
        model = SimpleNamespace(predict=lambda x, u, s: x+1,
                                predict_stage_mean=lambda x, s: x+2)
        np.testing.assert_array_equal(propagate(model, records).ravel(), [1,2,3,4]*2)
        np.testing.assert_array_equal(propagate(model, records, True).ravel(), [2,4,6,8]*2)
        with self.assertRaises(ValueError):
            propagate(model, records[1:])

    def test_outputs_are_append_only_and_existing_output_refused_before_fit(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            write_json(root/"evidence.json", {"original": True})
            with self.assertRaises(FileExistsError):
                write_json(root/"evidence.json", {"original": False})
            with patch("SciML.models.v12.run_learning_existing.validate_dataset") as validate:
                with self.assertRaises(FileExistsError):
                    run(root/"missing", root)
                validate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
