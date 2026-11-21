from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
import torch
from SciML.models.v12.learning_ensemble import LearningEnsemble,calibrated_intervals,rank_correlation


class EnsembleTests(unittest.TestCase):
    def test_real_independent_members_train_only_scaling_replay_and_rng(self):
        x=np.arange(21,dtype=float).reshape(7,3)/7
        y=np.column_stack((x[:,0]**2,np.sin(x[:,1])))
        state=torch.random.get_rng_state().clone()
        a=LearningEnsemble(members=3,epochs=10,seed=5).fit(x,y,list(map(str,range(7))))
        p=a.predict_members(x)
        self.assertEqual(p.shape,(3,7,2))
        self.assertFalse(np.array_equal(p[0],p[1]))
        np.testing.assert_array_equal(a.x_scaler.mean_,x.mean(axis=0))
        self.assertTrue(torch.equal(state,torch.random.get_rng_state()))
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/"ensemble.pt"
            a.save(path)
            b=LearningEnsemble.load(path)
            np.testing.assert_array_equal(p,b.predict_members(x))
            with self.assertRaises(FileExistsError):
                a.save(path)

    def test_95_percent_interval_cannot_be_finite_with_13_calibration_cases(self):
        y=np.ones((13,2));m=np.zeros_like(y);s=np.ones_like(y)
        info,width=calibrated_intervals(y,m,s,y,m,s,np.ones(2),.95)
        self.assertFalse(info["finite_interval_available"])
        self.assertEqual(info["order_statistic_rank"],14)
        self.assertIsNone(width)
        info,width=calibrated_intervals(y,m,s,y,m,s,np.ones(2),.9)
        self.assertEqual(info["order_statistic_rank"],13)
        self.assertEqual(info["test_simultaneous_coverage"],1.)
        np.testing.assert_array_equal(width,np.ones_like(y))

    def test_nonfinite_and_missing_fit_rejected(self):
        a=LearningEnsemble()
        with self.assertRaises(RuntimeError):
            a.predict(np.zeros((3,2)))
        with self.assertRaises(ValueError):
            a.fit(np.zeros((3,2)),np.full((3,2),np.nan),["a","b","c"])
        with self.assertRaises(ValueError):
            LearningEnsemble(members=1)
        self.assertIsNone(rank_correlation(np.ones(4),np.arange(4)))


if __name__=="__main__":
    unittest.main()
