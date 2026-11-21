"""Dependency-light neural-operator prototypes for the audited WP4 corpus."""

from .data import OperatorBatch, load_protocol_holdout, manifest_coverage
from .deeponet import DeepONetStyleRandomFeatureRegressor
from .metrics import operator_metrics, training_temperature_range
from .spectral import SpectralResidualBaseline

__all__ = [
    "OperatorBatch",
    "SpectralResidualBaseline",
    "DeepONetStyleRandomFeatureRegressor",
    "load_protocol_holdout",
    "manifest_coverage",
    "operator_metrics",
    "training_temperature_range",
]
