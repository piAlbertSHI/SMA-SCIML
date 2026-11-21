import unittest
import numpy as np
from SciML.active_learning.learning_pool_loop import (
    ArchivedLabelOracle,pareto_mask,hypervolume,acquisition_components,choose_batch,
)


class PoolLoopTests(unittest.TestCase):
    def test_oracle_refuses_test_and_duplicate_queries(self):
        oracle=ArchivedLabelOracle(["a","b"],np.array([[1,2],[3,4]]))
        np.testing.assert_array_equal(oracle.reveal(["a"],"initial"),[[1,2]])
        for ids in (["a"],["test"],["b","b"]):
            with self.assertRaises(ValueError):
                oracle.reveal(ids,"forbidden")
        self.assertEqual(oracle.revealed,{"a"})

    def test_pareto_and_hypervolume_have_correct_maximization_direction(self):
        points=np.array([[1,3],[2,2],[1,1]])
        np.testing.assert_array_equal(pareto_mask(points),[True,True,False])
        self.assertEqual(hypervolume(points,[0,0]),5.)
        self.assertEqual(hypervolume(np.vstack((points,[3,1])),[0,0]),6.)
        self.assertEqual(hypervolume([[0,0]],[1,1]),0.)

    def test_selector_has_no_unrevealed_label_input_and_deterministic_ties(self):
        mean=np.array([[10,1,2,30,1,5],[12,1,3,40,1,4],[11,1,2,25,1,5.5]],float)
        std=np.ones_like(mean)*.1
        known=np.array([[9,1,1.5,20,1,6],[10,1,2,25,1,5]],float)
        scores=acquisition_components(mean,std,known,np.zeros(6),np.ones(6))
        np.testing.assert_allclose(scores["performance_identity_residual"],0.)
        self.assertEqual(choose_batch(["b","a","c"],scores,"uncertainty",1,3),["a"])
        self.assertEqual(choose_batch(["b","a","c"],scores,"random",2,3),
                         choose_batch(["b","a","c"],scores,"random",2,3))
        with self.assertRaises(ValueError):
            choose_batch(["a"],{"uncertainty":np.array([np.nan])},"uncertainty",1,3)


if __name__=="__main__":
    unittest.main()
