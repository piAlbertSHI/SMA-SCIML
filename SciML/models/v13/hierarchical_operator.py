"""Spatial encoders, latent attention, recurrent dynamics and a coordinate decoder.

Inference accepts initial state, known controls and requested horizon only.
This architecture is implemented and differentiable; no accuracy claim is made.
"""
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from SciML.models.v12.learning_historical_rollout import HistoricalModel, mask, BLOCKS
from SciML.models.v12.multistep_historical import project_tensor
from SciML.models.v12.learning_coordinate_operator import interpolate


@dataclass(frozen=True)
class OperatorConfig:
    width: int = 128
    attention_heads: int = 8
    latent_tokens: int = 16
    attention_layers: int = 3
    memory_size: int = 256
    conv_dilations: tuple = (1, 2, 4)
    fourier_bands: int = 4
    query_chunk: int = 1024
    correction_scale: float = 0.1

    def __post_init__(self):
        if self.width < 8 or self.width % self.attention_heads or self.width % 8:
            raise ValueError('width must be divisible by heads and group-norm groups')
        if min(self.latent_tokens,self.attention_layers,self.memory_size,self.fourier_bands,self.query_chunk)<1:
            raise ValueError('Model dimensions must be positive')
        if not self.conv_dilations or any(d<1 for d in self.conv_dilations) or self.correction_scale<=0:
            raise ValueError('Invalid dilation or residual scale')


def fourier_features(coordinates, bands):
    frequency=2.**torch.arange(bands,device=coordinates.device,dtype=coordinates.dtype)*torch.pi
    phase=coordinates[...,None]*frequency
    return torch.cat((coordinates,phase.sin().flatten(-2),phase.cos().flatten(-2)),dim=-1)


class SpatialResidual(nn.Module):
    def __init__(self,width,dilation):
        super().__init__()
        self.block=nn.Sequential(nn.GroupNorm(8,width),nn.GELU(),
            nn.Conv1d(width,width,3,padding=dilation,dilation=dilation),
            nn.GroupNorm(8,width),nn.GELU(),nn.Conv1d(width,width,1))

    def forward(self,x):
        return x+self.block(x)


class SpatialEncoder(nn.Module):
    def __init__(self,channels,config):
        super().__init__()
        self.lift=nn.Conv1d(channels,config.width,1)
        self.blocks=nn.Sequential(*[SpatialResidual(config.width,d) for d in config.conv_dilations])
        self.position=nn.Linear(1+2*config.fourier_bands,config.width)
        self.kind=nn.Parameter(torch.zeros(1,1,config.width))
        self.bands=config.fourier_bands

    def forward(self,x):
        tokens=self.blocks(self.lift(x)).transpose(1,2)
        coordinate=torch.linspace(0,1,x.shape[-1],device=x.device,dtype=x.dtype)[:,None]
        return tokens+self.position(fourier_features(coordinate,self.bands))[None]+self.kind


class LatentEncoder(nn.Module):
    def __init__(self,config):
        super().__init__()
        w=config.width
        self.queries=nn.Parameter(torch.randn(config.latent_tokens,w)*.02)
        self.cross=nn.MultiheadAttention(w,config.attention_heads,batch_first=True,dropout=0.)
        self.cross_norm=nn.LayerNorm(w)
        self.blocks=nn.ModuleList([nn.TransformerEncoderLayer(w,config.attention_heads,4*w,
            dropout=0.,activation='gelu',batch_first=True,norm_first=True) for _ in range(config.attention_layers)])
        self.norm=nn.LayerNorm(w)

    def forward(self,tokens,condition):
        query=self.queries[None].expand(len(tokens),-1,-1)+condition[:,None]
        attended,_=self.cross(query,tokens,tokens,need_weights=False)
        latent=self.cross_norm(query+attended)
        for block in self.blocks:latent=block(latent)
        return self.norm(latent)


class TransitionHead(nn.Module):
    def __init__(self,width,output):
        super().__init__()
        self.input=nn.Linear(2*width,2*width)
        self.film=nn.Linear(width,4*width)
        self.refine=nn.Sequential(nn.GELU(),nn.Linear(2*width,2*width),nn.GELU())
        self.output=nn.Linear(2*width,output)
        nn.init.normal_(self.output.weight,std=1e-3);nn.init.zeros_(self.output.bias)

    def forward(self,context,condition):
        scale,shift=self.film(condition).chunk(2,dim=-1)
        return self.output(self.refine(self.input(context)*(1+scale)+shift))


class CoordinateDecoder(nn.Module):
    """Coordinate cross-attention followed by a hard initial-value construction."""
    def __init__(self,config):
        super().__init__()
        w=config.width;self.config=config
        self.query=nn.Sequential(nn.Linear(2+4*config.fourier_bands,w),nn.GELU(),nn.Linear(w,w))
        self.attention=nn.MultiheadAttention(w,config.attention_heads,batch_first=True,dropout=0.)
        self.norm=nn.LayerNorm(w)
        self.output=nn.Sequential(nn.Linear(w,w),nn.GELU(),nn.Linear(w,2))
        self.register_buffer('grid',torch.linspace(0,1,101,dtype=torch.float64))
        self.register_buffer('field_scale',torch.ones(2,dtype=torch.float64))

    def forward(self,latent,initial,times,positions):
        for q in (times,positions):
            if q.ndim!=1 or len(q)==0 or not torch.isfinite(q).all() or torch.any((q<0)|(q>1)):
                raise ValueError('Coordinates must be nonempty one-dimensional values in [0,1]')
        if initial.shape!=(len(latent),2,101):raise ValueError('Invalid initial field')
        tt,xx=torch.meshgrid(times,positions,indexing='ij')
        coords=torch.stack((xx,tt),dim=-1).reshape(-1,2)
        outputs=[]
        for chunk in coords.split(self.config.query_chunk):
            query=self.query(fourier_features(chunk,self.config.fourier_bands))[None].expand(len(latent),-1,-1)
            attended,_=self.attention(query,latent,latent,need_weights=False)
            outputs.append(self.output(self.norm(query+attended)))
        residual=torch.cat(outputs,dim=1).transpose(1,2).reshape(len(latent),2,len(times),len(positions))
        start=interpolate(initial,self.grid,positions)
        gate=-torch.expm1(-4*times)
        return start[:,:,None]+self.field_scale[None,:,None,None]*gate[None,None,:,None]*residual


class HierarchicalStateOperator(nn.Module):
    """One causal step updates both the explicit 515-state and recurrent memory."""
    def __init__(self,base,config=None):
        super().__init__()
        self.config=config or OperatorConfig();c=self.config;w=c.width
        self.training_ids=list(base.training_ids)
        self.register_buffer('feature_mean',torch.tensor(base.mean))
        self.register_buffer('feature_scale',torch.tensor(base.scale))
        for s in range(1,6):
            self.register_buffer(f'indices_{s}',torch.tensor(np.flatnonzero(mask(s))))
            self.register_buffer(f'coef_{s}',torch.tensor(base.linear[s].coef_))
            self.register_buffer(f'intercept_{s}',torch.tensor(base.linear[s].intercept_))
        self.mechanical=SpatialEncoder(4,c)
        self.transport=SpatialEncoder(3,c)
        self.condition=nn.Sequential(nn.Linear(22,w),nn.GELU(),nn.Linear(w,w))
        self.encoder=LatentEncoder(c)
        self.recurrence=nn.GRUCell(2*w,c.memory_size)
        self.memory_projection=nn.Linear(c.memory_size,w)
        self.memory_film=nn.Linear(c.memory_size,2*w)
        self.transitions=nn.ModuleDict({str(s):TransitionHead(w,int(mask(s).sum())) for s in range(1,6)})
        self.decoder=CoordinateDecoder(c)
        self.double()

    def _validate(self,state,control,stage,memory):
        if state.ndim!=2 or state.shape[1]!=515 or control.shape!=(len(state),9) or stage not in range(1,6):
            raise ValueError('Expected state Bx515, control Bx9, stage in 1..5')
        if memory is not None and memory.shape!=(len(state),self.config.memory_size):raise ValueError('Memory shape mismatch')
        if not all(torch.isfinite(x).all() for x in (state,control) if x is not None):raise ValueError('Nonfinite inputs')
        if memory is not None and not torch.isfinite(memory).all():raise ValueError('Nonfinite memory')

    @staticmethod
    def initial_fields(state,stage):
        if stage not in (2,4):raise ValueError('Field decoder is defined for exchange stages 2 and 4')
        radial=state[:,102:153]
        solid=torch.cat((radial[:,1:].flip(-1),radial),dim=-1)+state[:,414:]
        fluid=state[:,204:305] if stage==2 else state[:,305:406]
        return torch.stack((solid,fluid),dim=1)

    def step(self,state,control,stage,memory=None,*,times=None,positions=None):
        self._validate(state,control,stage,memory)
        if (times is None)!=(positions is None):raise ValueError('Supply both time and position coordinates')
        if times is not None and stage not in (2,4):raise ValueError('Only exchange stages have field outputs')
        if memory is None:memory=state.new_zeros(len(state),self.config.memory_size)
        onehot=F.one_hot(torch.full((len(state),),stage-1,device=state.device),5).to(state.dtype)
        z=(torch.cat((state,control,onehot),dim=-1)-self.feature_mean)/self.feature_scale
        mechanical=torch.stack([z[:,i:i+51] for i in (0,51,102,153)],dim=1)
        transport=torch.stack((z[:,204:305],z[:,305:406],z[:,414:515]),dim=1)
        condition=self.condition(torch.cat((z[:,406:414],z[:,515:]),dim=-1))
        tokens=torch.cat((self.mechanical(mechanical),self.transport(transport)),dim=1)
        latent=self.encoder(tokens,condition)
        hidden=self.recurrence(torch.cat((latent.mean(1),condition),dim=-1),memory)
        gain,shift=self.memory_film(hidden).chunk(2,dim=-1)
        conditioned=latent*(1+gain[:,None])+shift[:,None]
        context=torch.cat((conditioned.mean(1),self.memory_projection(hidden)),dim=-1)
        active=getattr(self,f'indices_{stage}')
        prior=F.linear(z,getattr(self,f'coef_{stage}'),getattr(self,f'intercept_{stage}'))
        correction=self.transitions[str(stage)](context,condition)*self.feature_scale[active]*self.config.correction_scale
        predicted=project_tensor(state.index_add(-1,active,prior+correction))
        if not torch.isfinite(predicted).all():raise RuntimeError('Nonfinite forecast; no clipping or fallback')
        result={'state':predicted,'memory':hidden,'latent':conditioned}
        if times is not None:
            result['field']=self.decoder(conditioned,self.initial_fields(state,stage),times,positions)
        return result

    def forward(self,initial,control,cycles,*,memory=None,times=None,positions=None):
        if type(cycles) is not int or cycles<1:raise ValueError('cycles must be a positive integer')
        if (times is None)!=(positions is None):raise ValueError('Supply both query coordinate vectors')
        state=initial;states=[];memories=[];fields=[];keys=[];field_keys=[]
        for cycle in range(1,cycles+1):
            stages=([5] if cycle>1 else [])+[1,2,3,4]
            for stage in stages:
                decode=times is not None and stage in (2,4)
                out=self.step(state,control,stage,memory,times=times if decode else None,positions=positions if decode else None)
                state,memory=out['state'],out['memory'];states.append(state);memories.append(memory)
                keys.append((cycle-1 if stage==5 else cycle,stage))
                if decode:fields.append(out['field']);field_keys.append((cycle,stage))
        result={'states':torch.stack(states,1),'memories':torch.stack(memories,1),'keys':keys,'final_memory':memory}
        if fields:result.update(fields=torch.stack(fields,1),field_keys=field_keys)
        return result

    def parameter_report(self):
        modules={name:sum(p.numel() for p in module.parameters()) for name,module in self.named_children()}
        return {'trainable':sum(p.numel() for p in self.parameters()),'by_module':modules,
                'frozen_buffer_values':sum(v.numel() for v in self.buffers())}

    def save(self,path,*,metadata=None):
        with Path(path).open('xb') as f:
            torch.save({'schema':'hierarchical_state_operator_v1','config':asdict(self.config),'training_ids':self.training_ids,
                        'weights':self.state_dict(),'metadata':metadata or {}},f)

    @classmethod
    def load(cls,path):
        p=torch.load(path,weights_only=True)
        if p['schema']!='hierarchical_state_operator_v1':raise ValueError('Unknown checkpoint schema')
        # Allocate the frozen linear prior from stored tensors; no external file dependency.
        from sklearn.linear_model import Ridge
        base=HistoricalModel();base.training_ids=p['training_ids']
        base.mean=p['weights']['feature_mean'].numpy();base.scale=p['weights']['feature_scale'].numpy()
        for stage in range(1,6):
            reg=Ridge();reg.coef_=p['weights'][f'coef_{stage}'].numpy();reg.intercept_=p['weights'][f'intercept_{stage}'].numpy()
            base.linear[stage]=reg
        with torch.random.fork_rng(devices=[]):model=cls(base,OperatorConfig(**p['config']))
        model.load_state_dict(p['weights']);model.eval()
        return model


def joint_loss(prediction,state_truth,state_scale,*,field_truth=None,field_scale=None,positions=None,
               field_weight=1.,gradient_weight=.01,curvature_weight=0.,curvature_scale=None,
               curvature_channel_weights=None,field_channel_weights=None,times=None,
               entry_time_weight=0.):
    """Supervised state + field + spatial derivative losses; no claimed PDE residual.

    ``curvature_weight`` is an optional supervised second-spatial-derivative term.
    It compares the predicted and observed curvature instead of penalizing field
    curvature itself, so a sharp transformation front remains representable. The
    error is normalized by the training-batch truth-curvature RMS unless an
    explicit ``curvature_scale`` is supplied. ``curvature_channel_weights`` can
    focus the term on the solid or fluid field while preserving a normalized
    two-channel mean. ``field_channel_weights`` applies the same normalized
    weighting to field-value and spatial-gradient supervision. ``entry_time_weight``
    emphasizes early nonzero field times with a normalized exponential weight.
    The default is zero to preserve the original HRO objective exactly.
    """
    if field_weight < 0 or gradient_weight < 0 or curvature_weight < 0:
        raise ValueError('Loss weights must be nonnegative')
    if entry_time_weight < 0:
        raise ValueError('Entry time weight must be nonnegative')
    if curvature_scale is not None and curvature_scale <= 0:
        raise ValueError('Curvature scale must be positive')
    if curvature_channel_weights is not None:
        channel_weights=torch.as_tensor(curvature_channel_weights,dtype=torch.float64)
        if channel_weights.shape!=(2,) or not torch.isfinite(channel_weights).all() or torch.any(channel_weights<0) or not torch.any(channel_weights>0):
            raise ValueError('Curvature channel weights must be two finite nonnegative values with one positive value')
    if field_channel_weights is not None:
        field_weights=torch.as_tensor(field_channel_weights,dtype=torch.float64)
        if field_weights.shape!=(2,) or not torch.isfinite(field_weights).all() or torch.any(field_weights<0) or not torch.any(field_weights>0):
            raise ValueError('Field channel weights must be two finite nonnegative values with one positive value')
    if prediction['states'].shape!=state_truth.shape:raise ValueError('State supervision shape mismatch')
    error=(prediction['states']-state_truth)/state_scale[:515]
    state_loss=torch.stack([error[...,sl].square().mean() for sl in BLOCKS.values()]).mean()
    result={'state':state_loss}
    if field_truth is not None:
        if 'fields' not in prediction or prediction['fields'].shape!=field_truth.shape:raise ValueError('Field supervision shape mismatch')
        if field_scale is None or field_scale.shape!=(2,) or torch.any(field_scale<=0):raise ValueError('Train-only field scale required')
        scaled=(prediction['fields']-field_truth)/field_scale[None,None,:,None,None]
        if field_channel_weights is not None:
            normalized_field_weights=field_weights.to(scaled)*2/field_weights.sum()
            field_weight_tensor=normalized_field_weights.view(1,1,2,1,1)
        else:
            field_weight_tensor=1.
        if entry_time_weight:
            if times is None or times.ndim != 1 or len(times) != scaled.shape[3] or not torch.isfinite(times).all():
                raise ValueError('Increasing field times are required for entry weighting')
            if torch.any(times < 0) or torch.any(times > 1) or not torch.all(torch.diff(times) > 0):
                raise ValueError('Field times must increase within [0,1]')
            temporal_weights=1.+entry_time_weight*torch.exp(-4.*times[1:])
            temporal_weights=temporal_weights*len(temporal_weights)/temporal_weights.sum()
            temporal_weight_tensor=temporal_weights.to(scaled).view(1,1,1,-1,1)
        else:
            temporal_weight_tensor=1.
        # Time zero is an input constraint; omit it from supervised future error.
        result['field']=(scaled[:,:,:,1:].square()*field_weight_tensor*temporal_weight_tensor).mean()
        if positions is None or len(positions)<2 or not torch.all(torch.diff(positions)>0):raise ValueError('Increasing spatial grid required')
        result['spatial_gradient']=(torch.diff(scaled[:,:,:,1:],dim=-1).div(torch.diff(positions)).square()*field_weight_tensor*temporal_weight_tensor).mean()
        if curvature_weight:
            if len(positions)<3:
                raise ValueError('At least three spatial points are required for curvature loss')
            spacing=torch.diff(positions)
            predicted_gradient=torch.diff(scaled[:,:,:,1:],dim=-1)/spacing
            truth_scaled=field_truth/field_scale[None,None,:,None,None]
            truth_gradient=torch.diff(truth_scaled[:,:,:,1:],dim=-1)/spacing
            denominator=(spacing[:-1]+spacing[1:])/2
            predicted_curvature=torch.diff(predicted_gradient,dim=-1)/denominator
            truth_curvature=torch.diff(truth_gradient,dim=-1)/denominator
            scale=(truth_curvature.detach().square().mean().sqrt().clamp_min(1.)
                   if curvature_scale is None else
                   torch.as_tensor(curvature_scale,dtype=truth_curvature.dtype,device=truth_curvature.device))
            curvature_error=((predicted_curvature-truth_curvature)/scale).square()
            if curvature_channel_weights is not None:
                weights=channel_weights.to(curvature_error).view(1,1,2,1,1)
                weights=weights*2/weights.sum()
                curvature_error=curvature_error*weights
            curvature_error=curvature_error*temporal_weight_tensor
            result['spatial_curvature']=curvature_error.mean()
    total=(state_loss+field_weight*result.get('field',0.)+
           gradient_weight*result.get('spatial_gradient',0.)+
           curvature_weight*result.get('spatial_curvature',0.))
    if not torch.isfinite(total):raise RuntimeError('Nonfinite joint loss')
    result['total']=total
    return result
