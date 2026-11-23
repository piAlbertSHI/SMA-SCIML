import unittest

from examples.audit_hro_temporal_error import SEEDS


class HROTemporalErrorTests(unittest.TestCase):
    def test_temporal_audit_uses_confirmed_seed_set(self):
        self.assertEqual(SEEDS, (20261121, 20261122, 20261123))


if __name__ == '__main__':
    unittest.main()
