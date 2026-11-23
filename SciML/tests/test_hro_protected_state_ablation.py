import unittest

from examples.hro_protected_state_ablation import CANDIDATES, LATE_STAGE_GUARD, STATE_GUARD


class HROProtectedStateTests(unittest.TestCase):
    def test_candidate_matrix_contains_state_and_stage_protection(self):
        self.assertEqual(len(CANDIDATES), 5)
        self.assertEqual(len(STATE_GUARD), 8)
        self.assertEqual(len(LATE_STAGE_GUARD), 9)
        self.assertIn('jitter_fluid2_stateguard', CANDIDATES)
        self.assertIn('jitter_fluid2_stageguard', CANDIDATES)
        self.assertEqual(CANDIDATES['jitter_fluid2_stageguard']['state_stage_weights'], LATE_STAGE_GUARD)


if __name__ == '__main__':
    unittest.main()
