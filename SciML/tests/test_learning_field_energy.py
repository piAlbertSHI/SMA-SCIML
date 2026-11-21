import unittest
import numpy as np

from SciML.evaluation.audit_learning_field_energy import port_heat, pump_work, counterexamples


class FieldEnergyTests(unittest.TestCase):
    def test_signed_heat_and_pressure_units(self):
        t = np.array([0., .2, 1.])
        m = np.full(3, .001)
        tin, tout = np.full(3, 300.), np.full(3, 299.)
        self.assertAlmostEqual(port_heat(t, m, tin, tout, 4180.), 4.18)
        self.assertAlmostEqual(port_heat(t, -m, tin, tout, 4180.), -4.18)
        self.assertAlmostEqual(pump_work(t, -m, np.ones(3), 1000.), 1.)

    def test_missing_misaligned_or_nonmonotone_data_cannot_be_imputed(self):
        for time, m in (([0., 0.], [1., 1.]), ([1., 0.], [1., 1.]), ([0., 1.], [1.]), ([0., 1.], [1., np.nan])):
            with self.assertRaises(ValueError):
                port_heat(time, m, [300., 300.], [299., 299.], 4180.)
        with self.assertRaises(ValueError):
            pump_work([0., 1.], [1., 1.], [1., 1.], 0.)

    def test_same_temperatures_do_not_identify_flux_work_or_storage(self):
        data = counterexamples()
        self.assertAlmostEqual(data["heat_J"][1], 2*data["heat_J"][0])
        self.assertNotEqual(*data["same_heat_different_work_COP"]["COP"])
        self.assertNotEqual(*data["same_temperature_change_different_capacity_storage"]["delta_U_J"])


if __name__ == "__main__":
    unittest.main()
