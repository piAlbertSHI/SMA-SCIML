"""Stable thermal response driven by a frozen operator's fluid trajectory.

The positive effective rate is learned from the sealed training trajectories.
It is not identified as a physical heat-transfer coefficient. Linear fluid
forcing is integrated exactly between native time samples; the response is a
convex combination of its previous temperature and the two forcing endpoints.
"""
import numpy as np
import torch
from torch import nn
from scipy.optimize import minimize_scalar


def relaxation_numpy(initial, fluid, rates):
    z=np.asarray(rates)/(fluid.shape[-1]-1)
    decay=np.exp(-z);response=-np.expm1(-z)/z
    left=response-decay;right=1-response
    values=[np.asarray(initial)]
    for j in range(1,fluid.shape[-1]):
        values.append(decay*values[-1]+left*fluid[:,j-1]+right*fluid[:,j])
    return np.stack(values,axis=1)


class RelaxationMeanReadout(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('rate',torch.tensor(1.,dtype=torch.float64))
        self.register_buffer('time_grid',torch.linspace(0,1,50,dtype=torch.float64))

    @staticmethod
    def validate(initial,fluid,control):
        if initial.ndim!=1 or fluid.shape!=(len(initial),50) or control.shape!=(len(initial),9):
            raise ValueError('Expected initial mean B, fluid mean Bx50, controls Bx9')
        if not all(torch.isfinite(x).all() for x in (initial,fluid,control)):
            raise ValueError('Nonfinite thermal-response input')
        if torch.any(control[:,1]<=0) or torch.any(control[:,4]<=0):
            raise ValueError('Thickness and exchange duration must be positive')

    def fit(self,initial,fluid,control,target):
        self.validate(initial,fluid,control)
        if target.shape!=fluid.shape or not torch.isfinite(target).all():
            raise ValueError('Invalid SMA mean targets')
        s,f,c,y=(v.detach().cpu().numpy() for v in (initial,fluid,control,target))
        geometry=self.geometry(control).detach().cpu().numpy()
        def objective(log_rate):
            pred=relaxation_numpy(s,f,np.exp(log_rate)*geometry)
            return float(np.mean((pred[:,1:]-y[:,1:])**2))
        result=minimize_scalar(objective,bounds=(np.log(.001),np.log(1000.)),method='bounded',
            options={'xatol':1e-9,'maxiter':100})
        if not result.success or not np.isfinite(result.fun):raise RuntimeError('Rate identification failed')
        self.rate.copy_(torch.tensor(np.exp(result.x),dtype=torch.float64))
        return {'rate':float(self.rate),'fit_mean_rmse_K':float(np.sqrt(result.fun)),
                'objective_evaluations':int(result.nfev),'samples':len(initial),
                'rate_at_numerical_bound':bool(float(self.rate)<.00101 or float(self.rate)>999.),
                'rate_interpretation':'effective normalized relaxation, not an identified physical heat-transfer coefficient'}

    @staticmethod
    def geometry(control):
        return (control[:,4]/2.4)*(.0002/control[:,1])

    def forward(self,initial,fluid,control):
        self.validate(initial,fluid,control)
        z=self.rate*self.geometry(control)/49.
        decay=torch.exp(-z);response=-torch.expm1(-z)/z
        left=response-decay;right=1-response;values=[initial]
        for j in range(1,50):
            values.append(decay*values[-1]+left*fluid[:,j-1]+right*fluid[:,j])
        return torch.stack(values,dim=1)


class VolumeRelaxationMeanReadout(RelaxationMeanReadout):
    """Mass-scaled response motivated by equal masses in the sealed solver.

    The pipe storage geometry is shared across cases; an inverse plate-volume
    factor supplies a nominal rate dependence. One coefficient is still fitted
    and can absorb unresolved effective conductance, not certify its value.
    """
    @staticmethod
    def geometry(control):
        if torch.any(control[:,0]<=0):raise ValueError('Radius must be positive')
        return (control[:,4]/2.4)*(.0002/control[:,1])*(.04/control[:,0]).square()
