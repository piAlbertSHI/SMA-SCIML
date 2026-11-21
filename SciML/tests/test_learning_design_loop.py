import unittest
import numpy as np
from SciML.active_learning.learning_design_loop import propose, adjudicate


class DesignTests(unittest.TestCase):
    def test_proposal_uses_calibration_and_lower_pareto_not_candidate_truth(self):
        means = np.array([[10, 1, 4, 6, 1, 3], [10, 1, 6, 4, 1, 3], [10, 1, 3, 3, 1, 3]], float)
        cal = np.ones((13, 6))*5
        result = propose(["b", "a", "c"], means, np.ones_like(means)*.1,
                         cal+.1, cal, np.ones_like(cal)*.1, np.ones(6), cal, 2)
        self.assertEqual(result["selected_ids"], ["a", "b"])
        self.assertEqual(result["calibration_rank"], 13)
        self.assertFalse(result["post_selection_coverage_guaranteed"])
        self.assertEqual(result["lower_pareto_mask"], [True, True, False])
        with self.assertRaises(ValueError):
            propose(["b", "a", "c"], means, np.ones_like(means), cal[:3], cal[:3],
                    np.ones_like(cal[:3]), np.ones(6), cal)

    def test_no_candidate_is_fabricated_when_all_filtered(self):
        cal = np.ones((13, 6))*5
        result = propose(["a"], -np.ones((1, 6)), np.ones((1, 6)), cal, cal,
                         np.ones_like(cal), np.ones(6), cal)
        self.assertEqual(result["selected_ids"], [])

    def test_unknown_limits_are_not_safe_and_failures_are_rejected(self):
        observation = dict(existing_physics_gate_ok=True, exchange_system_energy_audits_ok=True,
                           mechanical_phase_min=0., mechanical_phase_max=1.,
                           sigma_eq_peak_MPa=500., center_deflection_peak_over_radius=.2)
        reference = dict(sigma_eq_peak_MPa=400., center_deflection_peak_over_radius=.3)
        result = adjudicate(observation, reference)
        self.assertEqual(result["engineering_safety"], "unknown")
        self.assertFalse(result["within_revealed_reference_envelope"]["sigma_eq_peak_MPa"])
        self.assertEqual(adjudicate(observation, reference, {"sigma_eq_peak_MPa": 450.})["engineering_safety"], "rejected")
        observation["mechanical_phase_max"] = 1.01
        self.assertEqual(adjudicate(observation, reference)["engineering_safety"], "rejected")


if __name__ == "__main__":
    unittest.main()
