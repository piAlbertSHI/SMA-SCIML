import unittest

from examples.hro_material_conditioned_ablation import CANDIDATES


class HROMaterialConditionedTests(unittest.TestCase):
    def test_material_weights_are_declared_before_screening(self):
        self.assertEqual(set(CANDIDATES), {'global_pareto_control', 'recAS_htf_guard', 'material_htf_balanced'})
        self.assertNotIn('material_field_channel_weights', CANDIDATES['global_pareto_control'])
        self.assertEqual(CANDIDATES['recAS_htf_guard']['material_field_channel_weights']['recAS'], [1., 1.75])
        self.assertEqual(CANDIDATES['material_htf_balanced']['material_field_channel_weights']['recA'], [1., 1.5])


if __name__ == '__main__':
    unittest.main()
