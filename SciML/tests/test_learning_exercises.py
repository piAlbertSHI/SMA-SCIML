import unittest
import numpy as np
from SciML.evaluation.learning_exercises import reference_checks, strict_rmse, training_statistics, split_directional_memory


class LearningExerciseTests(unittest.TestCase):
    def test_all_eight_reference_answers(self):
        answers = reference_checks()
        self.assertEqual(set(answers), {f"E{i:02d}" for i in range(1, 9)})
        self.assertEqual(answers["E03"]["gradient"], -10.)
        self.assertTrue(answers["E08"]["rejected_queries_do_not_mutate_history"])

    def test_rmse_shape_empty_and_nonfinite(self):
        for prediction, truth in ((np.zeros((3, 1)), np.zeros(3)), (np.zeros((0, 2)), np.zeros((0, 2))), ([[np.nan]], [[0.]])):
            with self.subTest(shape=np.shape(prediction)), self.assertRaises(ValueError):
                strict_rmse(prediction, truth)
        np.testing.assert_array_equal(strict_rmse([[1, 2], [3, 4]], [[0, 0], [2, 2]]), [1, 2])

    def test_statistics_training_permission_and_constant_column(self):
        x = np.array([[1., 3.], [3., 3.], [np.nan, np.nan]])
        mean, scale = training_statistics(x, [True, True, False])
        np.testing.assert_array_equal(mean, [2, 3])
        np.testing.assert_array_equal(scale, [1, 1])
        for mask in ([0, 1, 0], [False]*3, [True], [True, True, True]):
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                training_statistics(x, mask)

    def test_memory_shape_and_finiteness(self):
        for x in (np.zeros(51), np.full(101, np.nan), 1.):
            with self.assertRaises(ValueError):
                split_directional_memory(x)
        x = np.arange(202.).reshape(2, 101)
        even, odd = split_directional_memory(x)
        np.testing.assert_array_equal(even+odd, x)
        np.testing.assert_array_equal(odd[:, 50], [0, 0])


if __name__ == "__main__":
    unittest.main()
