import unittest
import numpy as np
from SciML.models.v12.learning_operator_cascade import antisymmetric, field_inputs_with_memory
from SciML.tests.test_learning_boundary_state import request
from SciML.evaluation.inspect_learning_state_contract import request_initial_state


class MemoryTests(unittest.TestCase):
    def test_symmetric_projection_loses_antisymmetric_memory(self):
        x = np.linspace(-1, 1, 101)
        field = 293+2*x+x*x
        memory = antisymmetric(field)
        np.testing.assert_allclose(memory, 2*x, atol=1e-12)
        np.testing.assert_allclose(memory+memory[::-1], 0, atol=0)
        np.testing.assert_allclose(.5*(field+field[::-1])+memory, field, atol=1e-12)

    def test_causal_input_reconstruction_and_invalid_memory(self):
        req = request()
        state = request_initial_state(req)
        memory = np.linspace(-2, 2, 101)
        branch, initial = field_inputs_with_memory(state, req, memory)
        np.testing.assert_allclose(initial[0], 293+memory)
        np.testing.assert_allclose(branch[:101], memory, atol=1e-12)
        np.testing.assert_array_equal(initial[1], np.full(101, 293.))
        self.assertEqual(initial[0, 50], 293.)
        with self.assertRaises(ValueError):
            field_inputs_with_memory(state, req, memory+1.)


class CrossBatchRoundoffTests(unittest.TestCase):
    def test_roundoff_is_not_a_physical_tolerance(self):
        from SciML.models.v12.learning_operator_cascade import assert_roundoff_only
        expected = np.full((2, 101), 300.)
        actual = np.nextafter(expected, np.inf)
        self.assertFalse(assert_roundoff_only(actual, expected)["exact_equal"])
        with self.assertRaises(AssertionError):
            assert_roundoff_only(expected+1e-8, expected)


if __name__ == "__main__":
    unittest.main()
