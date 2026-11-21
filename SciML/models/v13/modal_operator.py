"""HRO with time-conditioned attention and a finite spatial expansion.

The correction is represented explicitly in space instead of independently at
each coordinate. This is a representation constraint, not a PDE boundary law.
"""
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from sklearn.linear_model import Ridge

from SciML.models.v12.learning_historical_rollout import HistoricalModel
from SciML.models.v12.learning_coordinate_operator import interpolate
from .hierarchical_operator import HierarchicalStateOperator, OperatorConfig, fourier_features


@dataclass(frozen=True)
class ModalConfig(OperatorConfig):
    spatial_modes: int = 24

    def __post_init__(self):
        super().__post_init__()
        if type(self.spatial_modes) is not int or not 2 <= self.spatial_modes <= 100:
            raise ValueError('spatial_modes must be an integer in 2..100')


class ModalDecoder(nn.Module):
    """Time-query cross-attention produces coefficients shared by all positions.

    Basis: 1, x, cos(pi*x), ..., cos((M-2)*pi*x). The linear term permits
    nonzero boundary slopes. The hard initial value uses the inherited field
    interpolation. Neither conservation nor a stage-end identity is assumed.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        w = config.width
        self.query = nn.Sequential(nn.Linear(1+2*config.fourier_bands, w), nn.GELU(), nn.Linear(w, w))
        self.attention = nn.MultiheadAttention(w, config.attention_heads, batch_first=True, dropout=0.)
        self.norm = nn.LayerNorm(w)
        self.output = nn.Sequential(nn.Linear(w, w), nn.GELU(), nn.Linear(w, 2*config.spatial_modes))
        nn.init.normal_(self.output[-1].weight, std=1e-3)
        nn.init.zeros_(self.output[-1].bias)
        self.register_buffer('grid', torch.linspace(0, 1, 101, dtype=torch.float64))
        self.register_buffer('field_scale', torch.ones(2, dtype=torch.float64))

    def basis(self, positions):
        k = torch.arange(1, self.config.spatial_modes-1, device=positions.device, dtype=positions.dtype)
        return torch.cat((torch.ones_like(positions)[:, None], positions[:, None],
                          torch.cos(torch.pi*positions[:, None]*k[None])), dim=-1)

    def forward(self, latent, initial, times, positions):
        for query in (times, positions):
            if query.ndim != 1 or len(query) == 0 or not torch.isfinite(query).all() or torch.any((query < 0) | (query > 1)):
                raise ValueError('Coordinates must be nonempty one-dimensional values in [0,1]')
        if initial.shape != (len(latent), 2, 101):
            raise ValueError('Invalid initial field')
        coefficients = []
        for t in times.split(self.config.query_chunk):
            query = self.query(fourier_features(t[:, None], self.config.fourier_bands))[None].expand(len(latent), -1, -1)
            attended, _ = self.attention(query, latent, latent, need_weights=False)
            coefficients.append(self.output(self.norm(query+attended)).reshape(len(latent), len(t), 2, -1))
        coefficients = torch.cat(coefficients, dim=1)
        residual = torch.einsum('btcm,xm->bctx', coefficients, self.basis(positions))
        start = interpolate(initial, self.grid, positions)
        gate = -torch.expm1(-4*times)
        return start[:, :, None]+self.field_scale[None, :, None, None]*gate[None, None, :, None]*residual


class ModalStateOperator(HierarchicalStateOperator):
    """Same causal state dynamics as HRO, with an explicit spatial field basis."""
    def __init__(self, base, config=None):
        config = config or ModalConfig()
        super().__init__(base, config)
        self.decoder = ModalDecoder(config).double()

    def save(self, path, *, metadata=None):
        with Path(path).open('xb') as stream:
            torch.save({'schema': 'hro_modal_operator_v1', 'config': asdict(self.config),
                        'training_ids': self.training_ids, 'weights': self.state_dict(),
                        'metadata': metadata or {}}, stream)

    @classmethod
    def load(cls, path):
        payload = torch.load(path, weights_only=True)
        if payload['schema'] != 'hro_modal_operator_v1':
            raise ValueError('Unknown modal checkpoint schema')
        weights = payload['weights']
        base = HistoricalModel()
        base.training_ids = payload['training_ids']
        base.mean = weights['feature_mean'].numpy()
        base.scale = weights['feature_scale'].numpy()
        for stage in range(1, 6):
            reg = Ridge()
            reg.coef_ = weights[f'coef_{stage}'].numpy()
            reg.intercept_ = weights[f'intercept_{stage}'].numpy()
            base.linear[stage] = reg
        with torch.random.fork_rng(devices=[]):
            model = cls(base, ModalConfig(**payload['config']))
        model.load_state_dict(weights)
        return model.eval()
