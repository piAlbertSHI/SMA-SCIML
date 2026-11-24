import unittest

from examples.hro_temporal_profile_ablation import CANDIDATES


class HROTemporalProfileTests(unittest.TestCase):
    def test_protocol_contains_control_light_and_moderate_profiles(self):
        self.assertEqual(set(CANDIDATES), {
            'protected_state_control',
            'protected_state_late_light',
            'protected_state_late_moderate',
        })
        self.assertEqual(CANDIDATES['protected_state_control']['late_time_weight'], 0.0)
        self.assertEqual(CANDIDATES['protected_state_late_light']['late_time_weight'], 0.25)
        self.assertEqual(CANDIDATES['protected_state_late_moderate']['late_time_weight'], 0.5)


if __name__ == '__main__':
    unittest.main()
