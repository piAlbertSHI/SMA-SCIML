import unittest
import numpy as np
from SciML.evaluation.inspect_learning_data_tutorial import grouping_counts, scaling_demo


class DataTutorialTests(unittest.TestCase):
    def test_case_leakage_even_when_row_masks_disjoint(self):
        with self.assertRaises(ValueError):
            grouping_counts(["a", "a", "b"], {"train": np.array([True, False, False]), "test": np.array([False, True, True])})

    def test_train_only_scaler_and_input_copy(self):
        x = np.arange(28.).reshape(7, 4)
        backup = x.copy()
        result = scaling_demo(x, [True, True, True, False, False, False, False])
        self.assertTrue(result["train_only_mean_unchanged_exact"])
        self.assertGreater(result["wrong_global_mean_max_change"], 1e5)
        np.testing.assert_array_equal(x, backup)


if __name__ == "__main__":
    unittest.main()
