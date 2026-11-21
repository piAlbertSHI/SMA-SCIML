import unittest
import numpy as np
import torch
from SciML.models.v13.thermal_geometry import strip_weights, radial_weights, weighted_mean
from SciML.models.v13.increment_consistency import project_heat_increment


class ThermalGeometryTests(unittest.TestCase):
    def test_constant_and_symmetric_capacity(self):
        for w in (strip_weights(),radial_weights()):
            self.assertAlmostEqual(w.sum(),1.,places=14)
            self.assertTrue((w>=0).all())
            np.testing.assert_allclose(weighted_mean(np.full((4,len(w)),293.),w),293.)
        np.testing.assert_allclose(strip_weights(),strip_weights()[::-1],rtol=0,atol=1e-16)
        self.assertEqual(strip_weights()[0],0.)

    def test_mirroring_does_not_preserve_disc_mean(self):
        r=np.linspace(0.,1.,51);radial=r*r
        mirrored=np.r_[radial[1:][::-1],radial]
        self.assertAlmostEqual(weighted_mean(radial,radial_weights()),.5,places=3)
        self.assertAlmostEqual(weighted_mean(mirrored,strip_weights()),.25,places=2)
        self.assertGreater(abs(mirrored.mean()-weighted_mean(mirrored,strip_weights())),.08)

    def test_invalid_geometry_and_values_rejected(self):
        for count in (0,1,2,100,3.5):
            with self.assertRaises(ValueError):strip_weights(count)
        for w in ([1.,1.],[-1.,2.],[np.nan,1.]):
            with self.assertRaises(ValueError):weighted_mean([293.,293.],w)
        with self.assertRaises(ValueError):weighted_mean([293.,np.nan],[.5,.5])

    def test_increment_projection_keeps_initial_and_spatial_differences(self):
        g=torch.Generator().manual_seed(47)
        field=torch.randn(2,50,101,generator=g,dtype=torch.float64)+293.
        before=torch.zeros(2,515,dtype=torch.float64);after=before.clone();after[:,102:153]=2.
        result,delta=project_heat_increment(field,before,after)
        torch.testing.assert_close(result[:,0],field[:,0],rtol=0,atol=0)
        torch.testing.assert_close(result.diff(dim=-1),field.diff(dim=-1),rtol=0,atol=1e-12)
        w=result.new_tensor(strip_weights())
        torch.testing.assert_close(((result[:,-1]-result[:,0])*w).sum(-1),torch.full_like(delta,2.),rtol=0,atol=1e-12)
        with self.assertRaises(ValueError):project_heat_increment(field[:,:20],before,after)
