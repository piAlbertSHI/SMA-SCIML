"""WP3 stage-level physics--neural state-space prototype.

The dataset-v1 operator corpus contains only 20 audit checkpoints from six
physical runs.  This package therefore exposes research/prototyping APIs and
does not claim a generally trained model.
"""

from .corpus import CorpusAudit, OperatorCorpus, Transition
from .hybrid import (
    Ablation,
    HeuristicExchangeBaseline,
    HybridStateSpaceModel,
    RidgeResidual,
    RolloutResult,
    StageControl,
    ZeroResidual,
)
from .layout import StateLayout
from .projection import PhysicalProjector, ProjectionConfig, ProjectionReport
from .training import (
    HeldOutMetrics,
    NeuralTrainingResult,
    TrainingAudit,
    evaluate_model_on_role,
    fit_ridge_from_corpus,
    fit_torch_from_corpus,
)

__all__ = [
    "Ablation",
    "CorpusAudit",
    "HeuristicExchangeBaseline",
    "HeldOutMetrics",
    "HybridStateSpaceModel",
    "OperatorCorpus",
    "PhysicalProjector",
    "ProjectionConfig",
    "NeuralTrainingResult",
    "ProjectionReport",
    "RidgeResidual",
    "RolloutResult",
    "StageControl",
    "StateLayout",
    "Transition",
    "TrainingAudit",
    "ZeroResidual",
    "fit_ridge_from_corpus",
    "fit_torch_from_corpus",
    "evaluate_model_on_role",
]
