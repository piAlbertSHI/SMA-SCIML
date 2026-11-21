"""Normalized heat-capacity weights for the archived constant-cp disc proxy.

These are diagnostic weights, not replacement benchmark metrics. A mirrored
radial profile and a strip field are different quadratures; their weighted
absolute temperatures must not be equated without the physical bridge.
"""
import numpy as np


def strip_weights(count=101):
    if not isinstance(count, int) or count < 3 or count % 2 != 1:
        raise ValueError('Expected an odd strip count of at least three')
    x = np.linspace(-1., 1., count)
    width = np.full(count, 2. / (count - 1))
    width[[0, -1]] *= .5
    area = 2. * np.sqrt(np.maximum(1. - x*x, 0.)) * width
    return area / area.sum()


def radial_weights(count=51):
    if not isinstance(count, int) or count < 2:
        raise ValueError('Expected at least two radial nodes')
    r = np.linspace(0., 1., count)
    edges = np.r_[0., (r[:-1] + r[1:]) / 2., 1.]
    return np.diff(edges**2)


def weighted_mean(field, weights):
    field, weights = np.asarray(field), np.asarray(weights)
    if (field.ndim < 1 or weights.ndim != 1 or field.shape[-1] != len(weights)
            or not np.isfinite(field).all() or not np.isfinite(weights).all()
            or np.any(weights < 0) or not np.isclose(weights.sum(), 1., rtol=0, atol=1e-12)):
        raise ValueError('Finite fields and normalized nonnegative weights required')
    return np.sum(field * weights, axis=-1)
