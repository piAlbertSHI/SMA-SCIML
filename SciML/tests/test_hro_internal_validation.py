import unittest

from examples.hro_internal_validation import (
    CANDIDATES,
    FIT_CASE_IDS,
    SEALED_TEST_CASE_IDS,
    VALIDATION_CASE_IDS,
)
from examples.hro_spatial_ablation import VARIANTS


class HROInternalValidationTests(unittest.TestCase):
    def test_fixed_split_is_disjoint_and_material_stratified(self):
        self.assertEqual(len(FIT_CASE_IDS), 6)
        self.assertEqual(len(VALIDATION_CASE_IDS), 2)
        self.assertEqual(len(SEALED_TEST_CASE_IDS), 4)
        self.assertEqual(set(FIT_CASE_IDS) & set(VALIDATION_CASE_IDS), set())
        self.assertEqual(set(FIT_CASE_IDS) & set(SEALED_TEST_CASE_IDS), set())
        self.assertEqual(set(VALIDATION_CASE_IDS) & set(SEALED_TEST_CASE_IDS), set())
        self.assertEqual(sum('_reca_' in case_id for case_id in VALIDATION_CASE_IDS), 1)
        self.assertEqual(sum('_recas_' in case_id for case_id in VALIDATION_CASE_IDS), 1)

    def test_candidate_matrix_is_available(self):
        self.assertEqual(CANDIDATES, ('coordinate_reference', 'jitter_fluid2', 'jitter_fluid2_entry', 'jitter_fluid3_balance'))
        for candidate in CANDIDATES:
            self.assertIn(candidate, VARIANTS)
        self.assertEqual(VARIANTS['coordinate_reference']['spatial_points'], 26)
        self.assertEqual(VARIANTS['jitter_fluid3_balance']['field_channel_weights'], [1.0, 3.0])


if __name__ == '__main__':
    unittest.main()
