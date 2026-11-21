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
from dataclasses import asdict
from pathlib import Path

from .hierarchical_operator import HierarchicalStateOperator
from .modal_operator import ModalStateOperator,ModalConfig
from .selective_sma import base_from_weights
from .thermal_readout import ThermalReadout,ThermalReadoutOperator
from .stage_thermal_readout import StageThermalReadoutOperator
from SciML.models.v12.learning_coordinate_operator import interpolate


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
        if not torch.isfinite(self.rate) or self.rate<=0:raise ValueError('Positive finite response rate required')
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


class RelaxationReadoutOperator(nn.Module):
    """SMA mean correction driven by predicted fluid, with frozen recurrence.

    Evaluate the correction once on the native 50x101 grid and interpolate it
    to requested times. Its definition cannot depend on which positions are
    requested. The corrected output is not fed back into the state recurrence.
    """
    SCHEMA='hro_relaxation_readout_v1'

    def __init__(self,reference,response,scale=.5):
        super().__init__()
        if not 0<=scale<=1:raise ValueError('Blend scale must lie in [0,1]')
        self.reference=reference;self.response=response
        self.register_buffer('mean_scale',torch.tensor(scale,dtype=torch.float64))
        self.requires_grad_(False)

    @property
    def training_ids(self):return self.reference.training_ids

    def native_correction(self,mean,fluid,control):
        response=self.response(mean[:,0],fluid,control)
        delta=response-mean;delta[:,0]=0.
        return delta

    def step(self,state,control,stage,memory=None,*,times=None,positions=None):
        result=self.reference.step(state,control,stage,memory,times=times,positions=positions)
        if stage==2 and times is not None:
            grid=self.response.time_grid;space=self.reference.stage2_readout.space_grid
            native=result if torch.equal(times,grid) and torch.equal(positions,space) else self.reference.step(
                state,control,stage,memory,times=grid,positions=space)
            field=native['field'];mean=field[:,0].mean(-1)
            delta=self.native_correction(mean,field[:,1].mean(-1),control)
            query=interpolate(delta,grid,times)*self.mean_scale
            result['field']=torch.cat(((result['field'][:,0]+query[:,:,None])[:,None],result['field'][:,1:2]),dim=1)
        return result

    def forward(self,*args,**kwargs):return HierarchicalStateOperator.forward(self,*args,**kwargs)

    def save(self,path,*,metadata=None):
        stage=self.reference;ref=stage.reference
        with Path(path).open('xb') as stream:
            torch.save({'schema':self.SCHEMA,'config':asdict(ref.backbone.config),
                'training_ids':self.training_ids,'reference_rank':ref.readout.rank,
                'stage2_rank':stage.stage2_readout.rank,'preserve_spatial_mean':stage.preserve_spatial_mean,
                'volume_scaling':isinstance(self.response,VolumeRelaxationMeanReadout),
                'weights':self.state_dict(),'metadata':metadata or {}},stream)

    @classmethod
    def load(cls,path):
        value=torch.load(path,weights_only=True)
        if value['schema']!=cls.SCHEMA:raise ValueError('Unknown response checkpoint')
        prefix='reference.reference.backbone.'
        base={k[len(prefix):]:w for k,w in value['weights'].items() if k.startswith(prefix)}
        with torch.random.fork_rng(devices=[]):
            backbone=ModalStateOperator(base_from_weights(base,value['training_ids']),ModalConfig(**value['config']))
        shared=ThermalReadoutOperator(backbone,ThermalReadout(value['reference_rank'],'residual'))
        stage=StageThermalReadoutOperator(shared,ThermalReadout(value['stage2_rank'],'residual'),
            preserve_spatial_mean=value['preserve_spatial_mean'])
        response=VolumeRelaxationMeanReadout() if value['volume_scaling'] else RelaxationMeanReadout()
        model=cls(stage,response);model.load_state_dict(value['weights']);return model.eval()
