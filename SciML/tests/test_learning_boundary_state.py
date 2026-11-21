import copy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from SciML.models.v11.state_space import TransitionRecord, LAYOUT
from SciML.models.v12.learning_boundary_state import BoundaryStateModel, boundary_mask, autonomous
from SciML.evaluation.inspect_learning_state_contract import request_initial_state, requested_controls


def request():
    return {"case_spec": {"material_id": "recA", "radius_m": .04, "thickness_m": .0002,
                          "pressure_Pa": 1e6, "flush_volume_target_ratio": 1., "exchange_duration_s": 2.4},
            "run_options": {"solver_options": {"Tr": 293., "T_load": .1, "T_unload": .1}, "exchange_options": {}}}


class BoundaryTests(unittest.TestCase):
    def records(self):
        records = []
        for case in range(5):
            req = request()
            req["case_spec"]["pressure_Pa"] += case*1e4
            state = request_initial_state(req)
            for stage in (1, 2, 3, 4):
                end = state.copy()
                end[LAYOUT.sma_T] += stage*.2+case*.01
                records.append(TransitionRecord(f"c{case}:s{stage}", f"c{case}", f"cond{case}", "recA", "g", 1, stage,
                                                state, end, requested_controls(req, stage)))
                state = end
        return records

    def test_zero_residual_channels_cannot_generate_one_meter_scale_noise(self):
        model = BoundaryStateModel(seed=3).fit(self.records(), epochs=5)
        for stage in (1, 2, 3):
            active = np.flatnonzero(boundary_mask(stage))
            index = int(np.flatnonzero(active == LAYOUT.u.start)[0])
            self.assertEqual(model.residual_scale[stage][index], 0.)
        req = request()
        predictions = autonomous(model, {"c": req}, ["c"])
        np.testing.assert_array_equal(predictions[:, LAYOUT.u], np.zeros((4, 51)))
        np.testing.assert_array_equal(predictions[:, LAYOUT.w], np.zeros((4, 51)))
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/"model.pt"
            model.save(path)
            np.testing.assert_array_equal(predictions, autonomous(BoundaryStateModel.load(path), {"c": req}, ["c"]))

    def test_initialization_and_inference_ignore_injected_future_data(self):
        model = BoundaryStateModel(seed=3).fit(self.records(), epochs=5)
        req, contaminated = request(), request()
        contaminated["future_stage4_temperature"] = [350.]*101
        contaminated["test_target"] = [1000.]*414
        np.testing.assert_array_equal(request_initial_state(req), request_initial_state(contaminated))
        np.testing.assert_array_equal(autonomous(model, {"c": req}, ["c"]), autonomous(model, {"c": contaminated}, ["c"]))
        contaminated["run_options"]["solver_options"]["initial_checkpoint"] = "unsupported_restart.mat"
        with self.assertRaises(ValueError):
            request_initial_state(contaminated)

    def test_composite_masks_express_restart_ownership(self):
        self.assertFalse(boundary_mask(1)[LAYOUT.side].any())
        self.assertTrue(boundary_mask(2)[LAYOUT.xi].all())  # restart re-equilibration
        self.assertFalse(boundary_mask(2)[LAYOUT.htf_stage4_T].any())
        self.assertTrue(boundary_mask(3)[LAYOUT.side].all())  # latent/setup transfer
        self.assertTrue(boundary_mask(3)[LAYOUT.htf_stage4_T].all())
        self.assertFalse(boundary_mask(4)[LAYOUT.u].any())


if __name__ == "__main__":
    unittest.main()
