"""Stage-specific SMA residual readout with an immutable stage-four path."""
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from .hierarchical_operator import HierarchicalStateOperator
from .modal_operator import ModalStateOperator, ModalConfig
from .selective_sma import base_from_weights
from .thermal_readout import ThermalReadout, ThermalReadoutOperator


class StageThermalReadoutOperator(nn.Module):
    def __init__(self, reference, stage2_readout, scale=1.):
        super().__init__()
        if reference.readout.mode != 'residual' or stage2_readout.mode != 'residual':
            raise ValueError('Only additive residual readouts are supported')
        if not 0 <= scale <= 1:raise ValueError('Scale must be within [0,1]')
        self.reference=reference;self.stage2_readout=stage2_readout
        self.register_buffer('scale',torch.tensor(scale,dtype=torch.float64))
        self.requires_grad_(False)

    @property
    def training_ids(self):return self.reference.training_ids

    def step(self,state,control,stage,memory=None,*,times=None,positions=None):
        if stage != 2 or times is None:
            return self.reference.step(state,control,stage,memory,times=times,positions=positions)
        backbone=self.reference.backbone
        result=backbone.step(state,control,stage,memory,times=times,positions=positions)
        initial=backbone.initial_fields(state,stage)
        stages=torch.full((len(state),),stage,dtype=torch.long,device=state.device)
        delta=self.stage2_readout(initial,control,stages,times,positions)*self.scale
        result['field']=torch.cat(((result['field'][:,0]+delta)[:,None],result['field'][:,1:2]),dim=1)
        return result

    def forward(self,*args,**kwargs):return HierarchicalStateOperator.forward(self,*args,**kwargs)

    def save(self,path,*,metadata=None):
        ref=self.reference
        with Path(path).open('xb') as stream:
            torch.save({'schema':'hro_stage_thermal_readout_v1','config':asdict(ref.backbone.config),
                'training_ids':self.training_ids,'reference_rank':ref.readout.rank,
                'stage2_rank':self.stage2_readout.rank,'weights':self.state_dict(),
                'metadata':metadata or {}},stream)

    @classmethod
    def load(cls,path):
        value=torch.load(path,weights_only=True)
        if value['schema']!='hro_stage_thermal_readout_v1':raise ValueError('Unknown checkpoint')
        weights=value['weights'];prefix='reference.backbone.'
        base_weights={k[len(prefix):]:v for k,v in weights.items() if k.startswith(prefix)}
        with torch.random.fork_rng(devices=[]):
            backbone=ModalStateOperator(base_from_weights(base_weights,value['training_ids']),ModalConfig(**value['config']))
        ref=ThermalReadoutOperator(backbone,ThermalReadout(value['reference_rank'],'residual'))
        model=cls(ref,ThermalReadout(value['stage2_rank'],'residual'))
        model.load_state_dict(weights)
        return model.eval()
