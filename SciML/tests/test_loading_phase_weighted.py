import tempfile
import unittest
from pathlib import Path

import numpy as np

from SciML.models.v13.loading_field_operator import PODRidgeLoading
from SciML.models.v13.loading_phase_reconstruction import MaterialPressurePhaseRidge
from SciML.models.v13.loading_phase_weighted import SaturationWeightedPhaseRidge


class SaturationWeightedPhaseTests(unittest.TestCase):
    def test_weighted_head_preserves_other_fields_and_bounds_phase(self):
        rng = np.random.default_rng(31)
        controls = np.asarray([[r, r / 200, p, material]
                               for r in (.034, .040)
                               for material in (0., 1.)
                               for p in (.8e6, 1.2e6, 1.6e6, 1.9e6)])
        times = np.linspace(0, 1, 7)
        radii = np.linspace(0, 1, 6)
        mean, scale = controls.mean(0), controls.std(0)
        normalized = rng.normal(size=(len(controls), len(times), len(radii), 4)).transpose(0, 3, 1, 2) * .01
        normalized[:, 3] = np.clip(
            (controls[:, 2, None, None] / 2e6 + .25 * controls[:, 3, None, None]) * times[None, None, :, None]
            * (1 - .2 * radii[None, None, None, :]), 0, 1,
        )
        with tempfile.TemporaryDirectory() as tmp:
            pod = PODRidgeLoading.fit(controls, normalized, times, radii, mean, scale, np.ones(4), rank=2)
            baseline = MaterialPressurePhaseRidge.fit(pod, controls, normalized[:, 3])
            weighted = SaturationWeightedPhaseRidge.fit(
                baseline, controls, normalized[:, 3], .8, 40., .8, .9,
            )
            query_t, query_r = np.linspace(0, 1, 13), np.linspace(0, 1, 9)
            base = baseline.predict(controls, query_t, query_r)
            result = weighted.predict(controls, query_t, query_r)
            np.testing.assert_array_equal(result[:, :3], base[:, :3])
            self.assertTrue(((result[:, 3] >= 0) & (result[:, 3] <= 1)).all())
            self.assertTrue((np.diff(result[:, 3], axis=1) >= -1e-12).all())
            np.testing.assert_array_equal(result[:, :, 0], base[:, :, 0])
            path = Path(tmp) / "weighted.npz"
            weighted.save(path)
            replayed = SaturationWeightedPhaseRidge.load(path, baseline)
            np.testing.assert_array_equal(replayed.predict(controls, query_t, query_r), result)

    def test_invalid_blend_and_grid_are_rejected(self):
        controls = np.asarray([[r, r / 200, p, material]
                               for r in (.038, .042) for material in (0., 1.)
                               for p in (.8e6, 1.6e6)])
        times, radii = np.linspace(0, 1, 4), np.linspace(0, 1, 4)
        values = np.zeros((len(controls), 4, 4))
        values[:, -1, -1] = np.linspace(0, 1, len(controls))
        pod = PODRidgeLoading.fit(controls, values[:, None] + np.zeros((len(controls), 4, 4, 4)), times, radii,
                                  controls.mean(0), controls.std(0), np.ones(4), rank=1)
        baseline = MaterialPressurePhaseRidge.fit(pod, controls, values)
        with self.assertRaises(ValueError):
            SaturationWeightedPhaseRidge.fit(baseline, controls, values, .8, 1., .9, .8)
        with self.assertRaises(ValueError):
            SaturationWeightedPhaseRidge.fit(baseline, controls, values, .8, 1., .8, 1.1)


if __name__ == "__main__":
    unittest.main()
