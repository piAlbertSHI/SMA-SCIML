import unittest

import numpy as np

from SciML.evaluation.audit_learning_inverse_decision_chapters import conformal_order, rmse


class InverseDecisionReviewTests(unittest.TestCase):
    def test_rmse_preserves_target_axis(self):
        np.testing.assert_array_equal(rmse([[3., 4.], [3., 4.]], np.zeros((2, 2))), [3., 4.])

    def test_error_rejects_broadcast_empty_and_nonfinite(self):
        for pred, truth in ((np.zeros((2, 1)), np.zeros(2)), ([], []), ([np.inf], [1.])):
            with self.subTest(pred=pred), self.assertRaises(ValueError):
                rmse(pred, truth)

    def test_calibration_order_does_not_clip_missing_rank(self):
        scores = np.arange(13., 0., -1.)
        self.assertEqual(conformal_order(scores, .9), (13, 13.))
        self.assertEqual(conformal_order(scores, .95), (14, None))
        self.assertEqual(conformal_order(np.arange(1., 20.), .95), (19, 19.))

    def test_calibration_rejects_invalid_inputs(self):
        for scores, coverage in (([], .9), ([[1.]], .9), ([np.nan], .9), ([1.], 1.)):
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                conformal_order(scores, coverage)


if __name__ == "__main__":
    unittest.main()
