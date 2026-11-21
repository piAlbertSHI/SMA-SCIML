import unittest
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from SciML.models.v12.learning_local_thermal_inverse import ode_prediction


class LocalThermalTests(unittest.TestCase):
    def test_constant_and_linear_forcing(self):
        t=np.linspace(0,2.4,50)
        for k in (.0000001,.7,10.):
            p=ode_prediction(t,np.full_like(t,300.),290.,k)
            np.testing.assert_allclose(p,300.-10*np.exp(-k*t),rtol=0,atol=2e-12)
        k=.7
        exact=300.-10*np.exp(-k*t)+2*(t-(1-np.exp(-k*t))/k)
        np.testing.assert_allclose(ode_prediction(t,300.+2*t,290.,k),exact,rtol=0,atol=2e-12)

    def test_nonuniform_piecewise_forcing_against_tight_independent_integrator(self):
        t=np.array([0.,.13,.4,.75,1.2,2.4])
        f=np.array([293.,295.,291.,297.,294.,295.])
        k=1.25
        expected=[290.]
        for j in range(len(t)-1):
            q=solve_ivp(lambda x,y:k*(np.interp(x,t,f)-y),(t[j],t[j+1]),[expected[-1]],
                        rtol=1e-12,atol=1e-12)
            expected.append(q.y[0,-1])
        np.testing.assert_allclose(ode_prediction(t,f,290.,k),expected,atol=1e-8,rtol=0)

    def test_optimizer_recovers_rate_far_from_initial_without_noisy_numerical_gradient(self):
        t=np.linspace(0,2.4,50)
        f=293.+.4*t+.2*np.sin(2*t)
        truth=ode_prediction(t,f,303.,1.25)
        ix=np.array([0,5,11,17,23,29,35,39])
        result=least_squares(lambda p:ode_prediction(t,f,303.,np.exp(p[0]))[ix]-truth[ix],
                             [np.log(.5)],bounds=([-10.],[5.]))
        self.assertTrue(result.success)
        self.assertAlmostEqual(np.exp(result.x[0]),1.25,places=7)

    def test_bad_inputs_rejected(self):
        for t,f,k in (([0,0],[290,300],.5),([0,1],[290,float('nan')],.5),([0,1],[290,300],-1)):
            with self.assertRaises(ValueError):
                ode_prediction(t,f,293.,k)


if __name__=='__main__':
    unittest.main()
