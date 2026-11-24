import unittest

from examples.hro_temporal_channel_profile_ablation import CANDIDATES


class HROTemporalChannelProfileTests(unittest.TestCase):
    def test_late_profile_is_restricted_to_sma_channel(self):
        self.assertEqual(CANDIDATES['protected_state_sma_late_light']['late_time_channel_weights'], [1., 0.])
        self.assertEqual(CANDIDATES['protected_state_sma_late_moderate']['late_time_channel_weights'], [1., 0.])


if __name__ == '__main__':
    unittest.main()
