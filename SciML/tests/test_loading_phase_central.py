import tempfile
import unittest
from pathlib import Path

import numpy as np

from SciML.models.v13.loading_field_operator import PODRidgeLoading
from SciML.models.v13.loading_phase_reconstruction import MaterialPressurePhaseRidge
from SciML.models.v13.loading_phase_weighted import SaturationWeightedPhaseRidge
from SciML.models.v13.loading_phase_central import CentralSaturationPhaseRidge


class CentralSaturationPhaseTests(unittest.TestCase):
    def test_central_head_preserves_channels_and_constraints(self):
        controls = np.asarray([[r, r / 200, p, material]
                               for r in (.034, .040)
                               for material in (0., 1.)
                               for p in (.8e6, 1.2e6, 1.6e6, 1.9e6)])
        times, radii = np.linspace(0, 1, 7), np.linspace(0, 1, 6)
        mean, scale = controls.mean(0), controls.std(0)
        phase = ((controls[:, 2, None, None] / 2e6 + .2 * controls[:, 3, None, None])
                 * times[None, :, None] * (1 - .2 * radii[None, None, :]))
        normalized = np.zeros((len(controls), 4, len(times), len(radii)))
        normalized[:, 0] = .001 * times[None, :, None]
        normalized[:, 1] = .002 * times[None, :, None] * (1 - radii[None, None, :])
        normalized[:, 2] = .01 * times[None, :, None]
        normalized[:, 3] = np.clip(phase, 0, 1)
        pod = PODRidgeLoading.fit(controls, normalized, times, radii, mean, scale, np.ones(4), rank=2)
        baseline_phase = MaterialPressurePhaseRidge.fit(pod, controls, normalized[:, 3])
        baseline = SaturationWeightedPhaseRidge.fit(baseline_phase, controls, normalized[:, 3])
        central = CentralSaturationPhaseRidge.fit(baseline, controls, normalized[:, 3], .8, 100., 40., .4, .8, .95, .04)
        with tempfile.TemporaryDirectory() as tmp:
            query_t, query_r = np.linspace(0, 1, 13), np.linspace(0, 1, 9)
            reference = baseline.predict(controls, query_t, query_r)
            result = central.predict(controls, query_t, query_r)
            np.testing.assert_array_equal(result[:, :3], reference[:, :3])
            self.assertTrue(((result[:, 3] >= 0) & (result[:, 3] <= 1)).all())
            self.assertTrue((np.diff(result[:, 3], axis=1) >= -1e-12).all())
            path = Path(tmp) / "central.npz"
            central.save(path)
            np.testing.assert_array_equal(CentralSaturationPhaseRidge.load(path, baseline).predict(controls, query_t, query_r), result)


if __name__ == "__main__":
    unittest.main()
