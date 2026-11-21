import unittest
import numpy as np
import pandas as pd

from SciML.models.v12.learning_structured_ood import structured_split, SCHEMES


def metadata():
    return pd.DataFrame({"condition_id": [f"c{i:03}" for i in range(63)],
        "radius_m": [.032+.002*(i % 5) for i in range(63)],
        "thickness_m": [(.032+.002*(i % 5))/200 for i in range(63)],
        "material_id": ["recAs" if i % 9 == 0 else "recA" for i in range(63)],
        "exchange_duration_s": [2.8 if i % 6 == 0 else 2.4 for i in range(63)],
        "Qcool_J": np.linspace(1, 10, 63)})


class StructuredSplitTests(unittest.TestCase):
    def test_partition_ignores_labels_and_row_order(self):
        table = metadata()
        changed = table.sample(frac=1, random_state=5).copy()
        changed["Qcool_J"] = np.nan
        for scheme in SCHEMES:
            a, b = structured_split(table, scheme), structured_split(changed, scheme)
            self.assertEqual(a, b)
            flat = sum(a.values(), [])
            self.assertEqual(len(flat), len(set(flat)))
            self.assertEqual(set(flat), set(table.condition_id))
            self.assertEqual(len(a["calibration"]), 13)

    def test_combined_group_is_unseen_along_both_axes(self):
        table = metadata().set_index("condition_id", drop=False)
        groups = structured_split(table, "combined")
        for part in ("train", "calibration", "id_test"):
            rows = table.loc[groups[part]]
            self.assertTrue((rows.radius_m < table.radius_m.max()).all())
            self.assertTrue((rows.material_id == "recA").all())
        joint = table.loc[groups["ood_combined"]]
        self.assertGreater(len(joint), 0)
        self.assertTrue((joint.radius_m == table.radius_m.max()).all())
        self.assertTrue((joint.material_id == "recAs").all())

    def test_duplicate_and_insufficient_groups_rejected(self):
        table = metadata()
        with self.assertRaises(ValueError):
            structured_split(pd.concat((table, table.iloc[:1])), "geometry")
        with self.assertRaises(ValueError):
            structured_split(table.iloc[:10], "material")

    def test_unseen_material_abstention_does_not_certify_seen(self):
        from SciML.evaluation.diagnose_learning_material_ood import material_support
        decisions = material_support(["recA", "recA"], ["recA", "recAs"])
        self.assertEqual(decisions[0]["action"], "eligible_for_further_checks")
        self.assertEqual(decisions[1]["action"], "abstain_unseen_material")
        with self.assertRaises(ValueError):
            material_support([], ["recA"])


if __name__ == "__main__":
    unittest.main()
