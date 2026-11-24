import unittest

from examples.hro_temporal_channel_refined import CANDIDATES


class HROTemporalChannelRefinedTests(unittest.TestCase):
    def test_micro_and_small_profiles_are_predeclared(self):
        self.assertEqual(CANDIDATES['protected_state_sma_late_micro']['late_time_weight'], 0.10)
        self.assertEqual(CANDIDATES['protected_state_sma_late_small']['late_time_weight'], 0.15)


if __name__ == '__main__':
    unittest.main()
