"""Physical-input POD/Ridge SMA readout attached to a frozen recurrent operator."""
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from .hierarchical_operator import HierarchicalStateOperator
from .modal_operator import ModalStateOperator, ModalConfig
from .selective_sma import base_from_weights
from SciML.models.v12.learning_coordinate_operator import interpolate


def thermal_features(initial, control, stages):
    """Explicit entry fields, geometry, flush ratio, duration and stage identity.

    Material identity and cycle counter are excluded. The sealed historical
    profiles share thermal properties; this assumption does not authorize use
    with a changed material/physics profile.
    """
    if initial.ndim!=3 or initial.shape[1:]!=(2,101) or control.shape!=(len(initial),9):
        raise ValueError('Expected initial Bx2x101 and control Bx9')
    if stages.shape!=(len(initial),) or not torch.all((stages==2)|(stages==4)):
        raise ValueError('Exchange stage must be 2 or 4')
    out=torch.cat((initial.flatten(1),control[:,[0,1,3,4]],(stages==2)[:,None],(stages==4)[:,None]),dim=1)
    if not torch.isfinite(out).all():raise ValueError('Nonfinite thermal inputs')
    return out


class ThermalReadout(nn.Module):
    def __init__(self, rank=12, mode='observed'):
        super().__init__()
        if mode not in ('observed','autonomous','residual'):raise ValueError('Unknown readout mode')
        self.rank=rank;self.mode=mode
        for name,value in {
            'feature_mean':torch.zeros(208),'feature_scale':torch.ones(208),
            'coef':torch.zeros(rank,208),'intercept':torch.zeros(rank),
            'basis':torch.zeros(rank,50,101),'mean_change':torch.zeros(50,101),
            'time_grid':torch.linspace(0,1,50),'space_grid':torch.linspace(0,1,101)}.items():
            self.register_buffer(name,value.double())
        # Construct grids directly in double precision, independent of defaults.
        self.time_grid=torch.linspace(0,1,50,dtype=torch.float64)
        self.space_grid=torch.linspace(0,1,101,dtype=torch.float64)

    def fit(self, initial, control, stages, change, alpha=1.):
        x=thermal_features(initial,control,stages).detach().cpu().numpy()
        y=change.detach().cpu().numpy().copy()
        if y.shape!=(len(x),50,101) or len(x)<=self.rank or not np.isfinite(y).all():
            raise ValueError('Insufficient or invalid readout supervision')
        y[:,0]=0.
        floor=np.r_[np.full(202,.1),.003,1.5e-5,.1,.4,.5,.5]
        mean=x.mean(0);scale=np.maximum(x.std(0),floor)
        flat=y.reshape(len(x),-1);center=flat.mean(0)
        # Match experiment precision/threading even when called from a library
        # or unit test; avoid multithreaded BLAS reduction-order variation.
        with threadpool_limits(limits=1):
            _,_,vh=np.linalg.svd(flat-center,full_matrices=False)
            basis=vh[:self.rank];codes=(flat-center)@basis.T
            reg=Ridge(alpha=alpha).fit((x-mean)/scale,codes)
        values={'feature_mean':mean,'feature_scale':scale,'coef':reg.coef_,'intercept':reg.intercept_,
                'basis':basis.reshape(self.rank,50,101),'mean_change':center.reshape(50,101)}
        for name,value in values.items():getattr(self,name).copy_(torch.as_tensor(value))
        oracle=(codes@basis+center).reshape(y.shape)
        return {'rank':self.rank,'alpha':alpha,'samples':len(x),
                'training_basis_rmse_K':float(np.sqrt(np.mean((oracle[:,1:]-y[:,1:])**2)))}

    def forward(self, initial, control, stages, times, positions):
        for q in (times,positions):
            if q.ndim!=1 or not len(q) or not torch.isfinite(q).all() or torch.any((q<0)|(q>1)):
                raise ValueError('Queries must be nonempty vectors within [0,1]')
        x=(thermal_features(initial,control,stages)-self.feature_mean)/self.feature_scale
        codes=nn.functional.linear(x,self.coef,self.intercept)
        change=torch.einsum('br,rtx->btx',codes,self.basis)+self.mean_change
        change=torch.cat((torch.zeros_like(change[:,:1]),change[:,1:]),dim=1)
        spatial=interpolate(change,self.space_grid,positions)
        query=interpolate(spatial.transpose(1,2),self.time_grid,times).transpose(1,2)
        return query


class ThermalReadoutOperator(nn.Module):
    def __init__(self, backbone, readout):
        super().__init__();self.backbone=backbone;self.readout=readout
        self.requires_grad_(False)

    @property
    def training_ids(self):return self.backbone.training_ids

    def step(self,state,control,stage,memory=None,*,times=None,positions=None):
        result=self.backbone.step(state,control,stage,memory,times=times,positions=positions)
        if times is not None:
            initial=self.backbone.initial_fields(state,stage)
            stages=torch.full((len(state),),stage,dtype=torch.long,device=state.device)
            delta=self.readout(initial,control,stages,times,positions)
            if self.readout.mode=='residual':sma=result['field'][:,0]+delta
            else:sma=interpolate(initial[:,0],self.readout.space_grid,positions)[:,None]+delta
            result['field']=torch.cat((sma[:,None],result['field'][:,1:2]),dim=1)
        return result

    def forward(self,*args,**kwargs):return HierarchicalStateOperator.forward(self,*args,**kwargs)

    def save(self,path,*,metadata=None):
        from dataclasses import asdict
        with Path(path).open('xb') as stream:
            torch.save({'schema':'hro_thermal_readout_v1','config':asdict(self.backbone.config),
                        'training_ids':self.training_ids,'rank':self.readout.rank,'mode':self.readout.mode,
                        'backbone':self.backbone.state_dict(),'readout':self.readout.state_dict(),'metadata':metadata or {}},stream)

    @classmethod
    def load(cls,path):
        value=torch.load(path,weights_only=True)
        if value['schema']!='hro_thermal_readout_v1':raise ValueError('Unknown checkpoint')
        with torch.random.fork_rng(devices=[]):
            backbone=ModalStateOperator(base_from_weights(value['backbone'],value['training_ids']),ModalConfig(**value['config']))
        backbone.load_state_dict(value['backbone'])
        readout=ThermalReadout(value['rank'],value['mode']);readout.load_state_dict(value['readout'])
        return cls(backbone,readout).eval()
