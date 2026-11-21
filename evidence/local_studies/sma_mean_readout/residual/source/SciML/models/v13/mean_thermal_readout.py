"""Conditional mean-temperature correction separated from spatial field shape."""
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from .hierarchical_operator import HierarchicalStateOperator
from .modal_operator import ModalStateOperator, ModalConfig
from .selective_sma import base_from_weights
from .thermal_readout import ThermalReadout, ThermalReadoutOperator, thermal_features
from .stage_thermal_readout import StageThermalReadoutOperator
from SciML.models.v12.learning_coordinate_operator import interpolate


def mean_features(initial,control,kind):
    stage=torch.full((len(initial),),2,dtype=torch.long,device=initial.device)
    full=thermal_features(initial,control,stage)
    if kind=='full':return full
    if kind!='compact':raise ValueError('Unknown mean feature set')
    return torch.cat((initial.mean(-1),initial.std(-1,unbiased=False),control[:,[0,1,3,4]]),dim=1)


class MeanThermalReadout(nn.Module):
    def __init__(self,kind='compact',rank=3):
        super().__init__()
        if kind not in ('compact','full') or rank<1:raise ValueError('Invalid mean readout configuration')
        self.kind=kind;self.rank=rank;count=8 if kind=='compact' else 208
        for name,value in {'feature_mean':torch.zeros(count),'feature_scale':torch.ones(count),
            'coef':torch.zeros(rank,count),'intercept':torch.zeros(rank),
            'basis':torch.zeros(rank,50),'mean_change':torch.zeros(50)}.items():self.register_buffer(name,value.double())
        self.register_buffer('time_grid',torch.linspace(0,1,50,dtype=torch.float64))

    @threadpool_limits.wrap(limits=1)
    def fit(self,initial,control,target,alpha=1.):
        x=mean_features(initial,control,self.kind).detach().cpu().numpy()
        y=target.detach().cpu().numpy().copy()
        if y.shape!=(len(x),50) or len(x)<=self.rank or not np.isfinite(y).all() or alpha<=0:
            raise ValueError('Invalid mean-temperature supervision or regularization')
        y[:,0]=0.
        floor=np.r_[np.full(4,.1),.003,1.5e-5,.1,.4] if self.kind=='compact' else np.r_[np.full(202,.1),.003,1.5e-5,.1,.4,.5,.5]
        center=x.mean(0);scale=np.maximum(x.std(0),floor);mean=y.mean(0)
        _,_,vh=np.linalg.svd(y-mean,full_matrices=False);basis=vh[:self.rank];codes=(y-mean)@basis.T
        reg=Ridge(alpha=alpha).fit((x-center)/scale,codes)
        for name,value in {'feature_mean':center,'feature_scale':scale,'coef':reg.coef_,
            'intercept':reg.intercept_,'basis':basis,'mean_change':mean}.items():getattr(self,name).copy_(torch.as_tensor(value))
        return {'samples':len(x),'features':len(center),'rank':self.rank,'alpha':alpha,
                'training_basis_rmse_K':float(np.sqrt(np.mean((codes@basis+mean-y)[:,1:]**2)))}

    def forward(self,initial,control,times):
        if times.ndim!=1 or not len(times) or not torch.isfinite(times).all() or torch.any((times<0)|(times>1)):
            raise ValueError('Time queries must be finite within [0,1]')
        x=(mean_features(initial,control,self.kind)-self.feature_mean)/self.feature_scale
        profile=nn.functional.linear(x,self.coef,self.intercept)@self.basis+self.mean_change
        profile=torch.cat((torch.zeros_like(profile[:,:1]),profile[:,1:]),dim=1)
        return interpolate(profile,self.time_grid,times)


class MeanThermalReadoutOperator(nn.Module):
    def __init__(self,reference,readout):
        super().__init__();self.reference=reference;self.mean_readout=readout;self.requires_grad_(False)

    @property
    def training_ids(self):return self.reference.training_ids

    def step(self,state,control,stage,memory=None,*,times=None,positions=None):
        result=self.reference.step(state,control,stage,memory,times=times,positions=positions)
        if stage==2 and times is not None:
            initial=self.reference.reference.backbone.initial_fields(state,stage)
            delta=self.mean_readout(initial,control,times)
            result['field']=torch.cat(((result['field'][:,0]+delta[:,:,None])[:,None],result['field'][:,1:2]),dim=1)
        return result

    def forward(self,*args,**kwargs):return HierarchicalStateOperator.forward(self,*args,**kwargs)

    def save(self,path,*,metadata=None):
        stage=self.reference;ref=stage.reference
        with Path(path).open('xb') as stream:
            torch.save({'schema':'hro_mean_thermal_readout_v1','config':asdict(ref.backbone.config),
                'training_ids':self.training_ids,'reference_rank':ref.readout.rank,'stage2_rank':stage.stage2_readout.rank,
                'preserve_spatial_mean':stage.preserve_spatial_mean,'mean_kind':self.mean_readout.kind,
                'mean_rank':self.mean_readout.rank,'weights':self.state_dict(),'metadata':metadata or {}},stream)

    @classmethod
    def load(cls,path):
        v=torch.load(path,weights_only=True)
        if v['schema']!='hro_mean_thermal_readout_v1':raise ValueError('Unknown checkpoint')
        prefix='reference.reference.backbone.'
        base={k[len(prefix):]:w for k,w in v['weights'].items() if k.startswith(prefix)}
        with torch.random.fork_rng(devices=[]):
            backbone=ModalStateOperator(base_from_weights(base,v['training_ids']),ModalConfig(**v['config']))
        shared=ThermalReadoutOperator(backbone,ThermalReadout(v['reference_rank'],'residual'))
        stage=StageThermalReadoutOperator(shared,ThermalReadout(v['stage2_rank'],'residual'),preserve_spatial_mean=v['preserve_spatial_mean'])
        model=cls(stage,MeanThermalReadout(v['mean_kind'],v['mean_rank']))
        model.load_state_dict(v['weights']);return model.eval()
