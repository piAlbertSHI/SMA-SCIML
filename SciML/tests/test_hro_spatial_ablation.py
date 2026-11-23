import unittest

import numpy as np

from examples.hro_spatial_ablation import VARIANTS, _coordinates, _query_positions


class SpatialAblationTests(unittest.TestCase):
    def test_query_jitter_preserves_sorted_endpoints(self):
        rng = np.random.default_rng(20261121)
        _, indices = _coordinates(51, np.array([0, 1, 2, 3, 4, 49]))
        positions = _query_positions(indices, .5, rng).numpy()
        self.assertEqual(positions[0], 0.)
        self.assertEqual(positions[-1], 1.)
        self.assertTrue(np.all(np.diff(positions) > 0))
        self.assertTrue(np.max(np.abs(positions - indices / 100.)) <= .005 + 1e-15)

    def test_fixed_query_path_has_native_coordinates(self):
        rng = np.random.default_rng(3)
        _, indices = _coordinates(51, np.array([0, 1, 2, 3, 4, 49]))
        positions = _query_positions(indices, 0., rng).numpy()
        np.testing.assert_array_equal(positions, indices / 100.)
        self.assertEqual(VARIANTS['dense_jitter']['spatial_jitter_fraction'], .5)


if __name__ == '__main__':
    unittest.main()
