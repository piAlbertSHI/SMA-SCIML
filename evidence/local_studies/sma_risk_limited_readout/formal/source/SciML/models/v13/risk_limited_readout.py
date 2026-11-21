"""Shrink thermal-response corrections by a training-case error scale.

The error scale is empirical RMS, not a calibrated prediction interval. The
projection is nonexpansive and cannot reverse a proposed correction. It only
guarantees squared-error improvement conditionally on the unknown truth lying
inside the proposed interval; this condition is NOT guaranteed by the fit.
"""
import torch

from .relaxation_readout import RelaxationReadoutOperator


def shrink_correction(delta,error_scale):
    if delta.ndim!=2 or error_scale.shape!=(delta.shape[1],):
        raise ValueError('Expected batch/time corrections and a time error scale')
    if not torch.isfinite(delta).all() or not torch.isfinite(error_scale).all() or torch.any(error_scale<0):
        raise ValueError('Finite corrections and nonnegative error scales required')
    return delta.sign()*(delta.abs()-error_scale[None]).clamp_min(0.)


class RiskLimitedRelaxationOperator(RelaxationReadoutOperator):
    SCHEMA='hro_risk_limited_relaxation_v1'

    def __init__(self,reference,response,scale=.5,error_scale=None):
        super().__init__(reference,response,scale)
        value=torch.zeros(50,dtype=torch.float64) if error_scale is None else torch.as_tensor(error_scale,dtype=torch.float64)
        if value.shape!=(50,) or not torch.isfinite(value).all() or torch.any(value<0):
            raise ValueError('Expected 50 nonnegative finite error scales')
        self.register_buffer('error_scale',value.detach().clone())

    def native_correction(self,mean,fluid,control):
        return shrink_correction(super().native_correction(mean,fluid,control),self.error_scale)
