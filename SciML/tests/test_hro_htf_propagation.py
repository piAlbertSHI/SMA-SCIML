import unittest

from examples.audit_hro_htf_propagation import _corr


class HROHTFPropagationTests(unittest.TestCase):
    def test_constant_series_has_no_correlation(self):
        self.assertIsNone(_corr([{'x': 1., 'htf_field_K': 1.}, {'x': 1., 'htf_field_K': 2.}], 'x'))


if __name__ == '__main__':
    unittest.main()
