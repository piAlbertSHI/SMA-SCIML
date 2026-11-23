import unittest

from examples.hro_material_conditioned_refined import REFINED_CANDIDATES


class HROMaterialRefinedTests(unittest.TestCase):
    def test_refined_weights_are_smaller_than_screened_weights(self):
        self.assertEqual(REFINED_CANDIDATES['recAS_htf_light']['material_field_channel_weights']['recAS'], [1., 1.10])
        self.assertEqual(REFINED_CANDIDATES['recAS_htf_moderate']['material_field_channel_weights']['recAS'], [1., 1.25])


if __name__ == '__main__':
    unittest.main()
