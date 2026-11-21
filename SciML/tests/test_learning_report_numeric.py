import unittest
from SciML.evaluation.audit_learning_scalar_state_chapters import rounded_value
from SciML.evaluation.audit_learning_operator_chapter import table_cells


class ReportNumericTests(unittest.TestCase):
    def test_rounding_uses_display_precision_not_global_tolerance(self):
        rounded_value("1.23", 1.2349)
        rounded_value("1.2300e-4", .0001230049)
        with self.assertRaises(ValueError):
            rounded_value("1.23", 1.2351)
        with self.assertRaises(ValueError):
            rounded_value("1.2300e-4", .000123006)

    def test_nonfinite_or_non_numeric_cells_rejected(self):
        for token, expected in (("nan", 1.), ("inf", 1.), ("1.0", float("nan")), ("approximately 1", 1.)):
            with self.subTest(token=token), self.assertRaises(ValueError):
                rounded_value(token, expected)

    def test_ambiguous_rows_require_section_scope(self):
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            table_cells("| B1 | 204 |\n| B1 | 0.3 |", "B1")
        self.assertEqual(table_cells("| B1 | 204 |", "B1"), ["204"])


if __name__ == "__main__":
    unittest.main()
