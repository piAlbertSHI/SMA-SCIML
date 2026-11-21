import tempfile
import unittest
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from examples.loading_phase_study import TIME_MS, arrival_times
from SciML.models.v13.loading_field_operator import PODRidgeLoading
from SciML.models.v13.loading_phase_reconstruction import (
    MaterialPressurePhaseRidge, constrained_phase_fields, project_phase_anchors,
)


class LoadingPhaseTests(unittest.TestCase):
    def test_bounded_isotonic_anchors_and_reconstruction(self):
        phase = np.array([[[0, 0], [-.2, .1], [.4, .3], [.2, .9], [1.2, .8]]])
        wanted = np.array([[[0, 0], [0, .1], [.3, .3], [.3, .85], [1., .85]]])
        np.testing.assert_allclose(project_phase_anchors(phase), wanted, rtol=0, atol=1e-15)
        anchor_t = np.array([0., .2, .4, .6, 1.])
        anchor_r = np.array([0., 1.])
        times = np.sort(np.r_[anchor_t, .1, .3, .5, .8])
        radii = np.array([0., .5, 1.])
        for name, result in constrained_phase_fields(phase, anchor_t, anchor_r, times, radii).items():
            with self.subTest(name=name):
                self.assertTrue(((result >= 0) & (result <= 1)).all())
                self.assertTrue((np.diff(result, axis=1) >= -1e-12).all())
                np.testing.assert_allclose(result[:, ::2, ::2], wanted, rtol=0, atol=1e-14)

    def test_arrival_interpolation_first_crossing_and_missing(self):
        phase = np.zeros((1, len(TIME_MS), 3))
        phase[:, 1, 0], phase[:, 3:, 0] = 1, 1
        phase[:, :, 1] = .3
        phase[:, 2:, 2] = .5
        got = arrival_times(phase, .5)[0]
        self.assertAlmostEqual(got[0], .5 * TIME_MS[1])
        self.assertTrue(np.isnan(got[1]))
        self.assertEqual(got[2], TIME_MS[2])

    def test_material_pressure_checkpoint_preserves_nonphase_fields(self):
        controls = np.asarray([[r, r / 200, pressure, material]
                               for r in (.034, .040) for material in (0., 1.)
                               for pressure in (.75e6 + .5e6 * material, .85e6 + .5e6 * material,
                                                .95e6 + .5e6 * material)])
        times, radii = np.linspace(0, 1, 5), np.linspace(0, 1, 4)
        mean, scale = controls.mean(axis=0), controls.std(axis=0)
        amplitude = (controls - mean) / scale
        fields = amplitude[:, :, None, None] * times[None, None, :, None] * (1 - radii[None, None, None, :])
        fields[:, 3] = (.2 + .05 * amplitude[:, 2])[:, None, None] * times[None, :, None]
        with threadpool_limits(limits=1), tempfile.TemporaryDirectory() as tmp:
            pod = PODRidgeLoading.fit(controls, fields, times, radii, mean, scale, np.ones(4), rank=2)
            fitted = MaterialPressurePhaseRidge.fit(pod, controls, fields[:, 3])
            target_times = np.linspace(0, 1, 11)
            pred = fitted.predict(controls, target_times, radii)
            reference = pod.predict(controls, target_times, radii)
            np.testing.assert_array_equal(pred[:, :3], reference[:, :3])
            np.testing.assert_array_equal(pred[:, 3, 0], 0)
            self.assertTrue(((pred[:, 3] >= 0) & (pred[:, 3] <= 1)).all())
            self.assertTrue((np.diff(pred[:, 3], axis=1) >= -1e-12).all())
            path = Path(tmp) / "phase.npz"
            fitted.save(path)
            np.testing.assert_array_equal(MaterialPressurePhaseRidge.load(path, pod).predict(controls, target_times, radii), pred)
            invalid = controls.copy()
            invalid[:, 1] *= 1.25
            with self.assertRaises(ValueError):
                fitted.predict(invalid, target_times, radii)


if __name__ == "__main__":
    unittest.main()
