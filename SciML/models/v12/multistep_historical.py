"""Differentiable continuation of the existing 515-state historical model.

Affine maps, normalization, residual scales and projection rules stay frozen.
Only the five neural residual maps are trainable. No temperature clipping.
"""
from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .learning_historical_rollout import HistoricalModel, BLOCKS, SIZE, mask


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def project_tensor(state):
    endpoint = torch.ones(SIZE, dtype=state.dtype, device=state.device)
    endpoint[[0, 50, 101]] = 0
    value = state * endpoint
    return torch.cat((value[..., :153], value[..., 153:204].clamp(0, 1), value[..., 204:414],
                      .5 * (value[..., 414:] - value[..., 414:].flip(-1))), dim=-1)


class DifferentiableHistorical(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.seed = base.seed
        self.training_ids = list(base.training_ids)
        self.networks = nn.ModuleDict({str(s): deepcopy(base.networks[s]) for s in range(1, 6)})
        self.register_buffer('feature_mean', torch.tensor(base.mean))
        self.register_buffer('feature_scale', torch.tensor(base.scale))
        for s in range(1, 6):
            self.register_buffer(f'indices_{s}', torch.tensor(np.flatnonzero(mask(s))))
            self.register_buffer(f'coef_{s}', torch.tensor(base.linear[s].coef_))
            self.register_buffer(f'intercept_{s}', torch.tensor(base.linear[s].intercept_))
            self.register_buffer(f'residual_scale_{s}', torch.tensor(base.scales[s]))

    def step(self, state, control, stage):
        if state.ndim != 2 or state.shape[1] != SIZE or control.shape != (len(state), 9) or stage not in range(1, 6):
            raise ValueError('Invalid state/control/stage')
        onehot = torch.eye(5, dtype=state.dtype, device=state.device)[stage-1].expand(len(state), -1)
        x = (torch.cat((state, control, onehot), dim=-1) - self.feature_mean) / self.feature_scale
        delta = F.linear(x, getattr(self, f'coef_{stage}'), getattr(self, f'intercept_{stage}'))
        delta = delta + self.networks[str(stage)](x) * getattr(self, f'residual_scale_{stage}')
        result = project_tensor(state.index_add(-1, getattr(self, f'indices_{stage}'), delta))
        if not torch.isfinite(result).all():
            raise RuntimeError('Nonfinite multi-step state; no clipping, fallback or retry')
        return result

    def trajectory(self, initial, control, stages, teacher_inputs=None):
        state = initial
        outputs = []
        for i, stage in enumerate(stages):
            entry = state if teacher_inputs is None else teacher_inputs[:, i]
            state = self.step(entry, control, int(stage))
            outputs.append(state)
        return torch.stack(outputs, dim=1)

    def predict(self, state, control, stage, neural=True):
        if not neural:
            raise ValueError('Use the archived HistoricalModel for affine-only evaluation')
        with torch.no_grad():
            return self.step(torch.as_tensor(state[None], dtype=torch.float64),
                             torch.as_tensor(control[None], dtype=torch.float64), stage)[0].numpy()

    def save_residual(self, path, base_relative, base_hash, mode, history):
        with Path(path).open('xb') as f:
            torch.save({'schema':'historical515_multistep_residual_v1', 'seed':self.seed,
                        'base_relative':base_relative, 'base_sha256':base_hash, 'mode':mode,
                        'networks':self.networks.state_dict(), 'history':history,
                        'training_ids':self.training_ids}, f)

    @classmethod
    def load_residual(cls, path, repository):
        p = torch.load(path, weights_only=True)
        if p['schema'] != 'historical515_multistep_residual_v1':
            raise ValueError('Unknown residual checkpoint schema')
        base_path = Path(repository)/p['base_relative']
        if sha(base_path) != p['base_sha256']:
            raise ValueError('Frozen base checkpoint changed')
        model = cls(HistoricalModel.load(base_path))
        if model.training_ids != p['training_ids']:
            raise ValueError('Training ownership changed')
        model.networks.load_state_dict(p['networks']); model.eval()
        return model


def balanced_state_loss(prediction, target, feature_scale):
    """Equal weighting of the eight state blocks, then cases and time steps."""
    error = (prediction-target)/feature_scale[:SIZE]
    return torch.stack([error[..., block].square().mean() for block in BLOCKS.values()]).mean()


def continuation(model, initial, control, before, after, stages, mode, updates=300, lr=.0005):
    if mode not in ('single_step_continue', 'multistep'):
        raise ValueError('Unknown continuation mode')
    tensors=[torch.as_tensor(a, dtype=torch.float64) for a in (initial, control, before, after)]
    initial_t, control_t, before_t, after_t = tensors
    if list(stages) != [1,2,3,4,5,1,2,3,4] or before_t.shape != (8,9,SIZE):
        raise ValueError('This protocol requires eight complete two-cycle training cases')
    optimizer=torch.optim.AdamW(model.networks.parameters(),lr=lr,weight_decay=1e-4)
    history=[];model.train()
    for update in range(updates):
        optimizer.zero_grad(set_to_none=True)
        predicted=model.trajectory(initial_t,control_t,stages,before_t if mode=='single_step_continue' else None)
        loss=balanced_state_loss(predicted,after_t,model.feature_scale)
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite continuation loss')
        loss.backward()
        grad_norm=nn.utils.clip_grad_norm_(model.networks.parameters(),10.,error_if_nonfinite=True)
        optimizer.step()
        if update%25==0 or update==updates-1:
            history.append({'update':update+1,'training_loss_before_update':float(loss.detach()),'gradient_norm_before_clip':float(grad_norm)})
    model.eval()
    with torch.no_grad():
        diagnostics={label:float(balanced_state_loss(model.trajectory(initial_t,control_t,stages,teacher),after_t,model.feature_scale))
                     for label,teacher in [('teacher_forced',before_t),('autonomous',None)]}
    return history,diagnostics
