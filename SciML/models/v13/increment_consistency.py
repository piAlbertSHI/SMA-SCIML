"""Experimental native-grid temperature projection for a frozen state model.

The constant-cp heat increment of the radial state constrains the strip-field
endpoint. A linear-in-time uniform correction minimizes its temporal-gradient
squared integral subject to zero initial correction and that endpoint. This
does not ensure the predicted state increment is accurate or close energy fluxes.
"""
import torch
from .thermal_geometry import strip_weights, radial_weights


def project_heat_increment(field, entry_state, exit_state):
    if (field.ndim!=3 or field.shape[1:]!=(50,101)
            or entry_state.shape!=(len(field),515) or exit_state.shape!=entry_state.shape):
        raise ValueError('Expected native SMA field Bx50x101 and paired Bx515 states')
    if not all(torch.isfinite(v).all() for v in (field,entry_state,exit_state)):
        raise ValueError('Nonfinite field or state')
    ws=field.new_tensor(strip_weights());wr=field.new_tensor(radial_weights())
    state_delta=((exit_state[:,102:153]-entry_state[:,102:153])*wr).sum(-1)
    field_delta=((field[:,-1]-field[:,0])*ws).sum(-1)
    mismatch=state_delta-field_delta
    ramp=torch.linspace(0,1,50,dtype=field.dtype,device=field.device)
    return field+mismatch[:,None,None]*ramp[None,:,None], mismatch
