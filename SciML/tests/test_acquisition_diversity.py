import unittest
import numpy as np
from examples.active_diversity import diverse_batch


class DiversityTests(unittest.TestCase):
    def test_equal_scores_cover_distinct_regions(self):
        ids = ['near_a', 'near_b', 'far']
        chosen, _ = diverse_batch(ids, [[1], [1.01], [10]], [[0]], np.ones(3) * .5, 2)
        self.assertEqual(chosen, ['far', 'near_b'])
        self.assertEqual(len(set(chosen)), 2)

    def test_permutation_does_not_change_tie_breaking(self):
        a, _ = diverse_batch(['b', 'a', 'c'], [[1], [-1], [3]], [[0]], [.5] * 3, 2)
        b, _ = diverse_batch(['c', 'a', 'b'], [[3], [-1], [1]], [[0]], [.5] * 3, 2)
        self.assertEqual(a, b)

    def test_rejects_nonfinite_or_misaligned_features(self):
        with self.assertRaises(ValueError):
            diverse_batch(['a'], [[np.nan]], [[0]], [.5], 1)
        with self.assertRaises(ValueError):
            diverse_batch(['a'], [[1, 2]], [[0]], [.5], 1)
        with self.assertRaises(ValueError):
            diverse_batch(['a', 'a'], [[1], [2]], [[0]], [.5, .5], 1)


if __name__ == '__main__':
    unittest.main()
