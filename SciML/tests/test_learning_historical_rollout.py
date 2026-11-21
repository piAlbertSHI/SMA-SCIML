import unittest
import numpy as np
import pandas as pd

from SciML.models.v12.learning_historical_rollout import SIZE, MEMORY, mask, project, rollout, split_cases


class CountingModel:
    def __init__(self):
        self.calls = []

    def predict(self, state, control, stage, neural=True):
        self.calls.append((stage, float(state[102])))
        out = state.copy()
        out[102] += stage
        out[MEMORY] = np.linspace(-1, 1, 101)
        return out


class HistoricalRolloutTests(unittest.TestCase):
    def test_rollout_uses_previous_prediction_and_explicit_restart(self):
        model = CountingModel()
        initial = np.zeros(SIZE)
        initial[102] = 293.
        out = rollout(model, initial, np.zeros(9), 2)
        self.assertEqual(out.shape, (9, SIZE))
        self.assertEqual([v[0] for v in model.calls], [1, 2, 3, 4, 5, 1, 2, 3, 4])
        np.testing.assert_array_equal(out[:, 102], 293+np.cumsum([1, 2, 3, 4, 5, 1, 2, 3, 4]))
        self.assertEqual(initial[102], 293.)
        with self.assertRaises(TypeError):
            rollout(model, initial, np.zeros(9), 2, future_truth=out)

    def test_skip_restart_and_erase_memory_are_explicit_interventions(self):
        model = CountingModel()
        out = rollout(model, np.zeros(SIZE), np.zeros(9), 2, skip_restart=True, erase_memory=True)
        self.assertNotIn(5, [v[0] for v in model.calls])
        self.assertEqual(out[4, 102], out[3, 102])
        np.testing.assert_array_equal(out[:, MEMORY], np.zeros((9, 101)))

    def test_projection_does_not_hide_temperature_drift(self):
        state = np.full(SIZE, 500.)
        projected = project(state)
        self.assertEqual(projected[102], 500.)
        np.testing.assert_array_equal(projected[153:204], np.ones(51))
        np.testing.assert_array_equal(projected[[0, 50, 101]], np.zeros(3))
        np.testing.assert_array_equal(projected[MEMORY], np.zeros(101))
        self.assertTrue(mask(5)[:204].all())
        self.assertFalse(mask(5)[204:].any())

    def test_case_split_holds_all_longer_horizons_without_label_selection(self):
        table = pd.DataFrame({"case_id": [f"c{i}" for i in range(12)],
            "cycle_count": [2]*5+[3]+[2]*5+[3], "material_id": ["recA"]*6+["recAs"]*6})
        a = split_cases(table)
        b = split_cases(table.sample(frac=1, random_state=7).assign(COP=np.nan))
        self.assertEqual(a, b)
        self.assertEqual(len(a["train"]), 8)
        self.assertEqual(len(a["test"]), 4)
        self.assertTrue({"c5", "c11"}.issubset(a["test"]))

    def test_field_decoder_uses_predicted_memory_and_correct_inactive_path(self):
        from SciML.models.v12.learning_historical_fields import field_inputs
        state = np.full(SIZE, 293.)
        state[MEMORY] = np.linspace(-2, 2, 101)
        state[412] = np.nan  # inactive stage4 path at stage2
        c = np.array([.04, .0002, 1e6, 1., 2.4, 1., 0., .1, .1])
        branch, initial = field_inputs(state, c, 2)
        self.assertEqual(branch.shape, (221,))
        np.testing.assert_allclose(initial[0], 293+state[MEMORY])
        self.assertEqual(branch[219], 293.)
        with self.assertRaises(ValueError):
            field_inputs(state, c, 4)  # that path is active now, NaN must fail


if __name__ == "__main__":
    unittest.main()
