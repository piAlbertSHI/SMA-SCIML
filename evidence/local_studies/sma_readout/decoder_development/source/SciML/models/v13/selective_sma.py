"""Field-specific adaptation with exactly preserved state and fluid predictions."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from sklearn.linear_model import Ridge

from SciML.models.v12.learning_historical_rollout import HistoricalModel
from .modal_operator import ModalStateOperator, ModalConfig


class SelectiveDecoder(nn.Module):
    """Keep the reference fluid channel; adapt the independent SMA readout."""
    def __init__(self, decoder):
        super().__init__()
        self.reference = decoder
        self.sma = deepcopy(decoder)
        self.reference.requires_grad_(False)

    @property
    def field_scale(self): return self.reference.field_scale

    @property
    def grid(self): return self.reference.grid

    def forward(self, latent, initial, times, positions):
        reference = self.reference(latent, initial, times, positions)
        adapted = self.sma(latent, initial, times, positions)
        return torch.cat((adapted[:, :1], reference[:, 1:2]), dim=1)


def base_from_weights(weights, training_ids):
    base = HistoricalModel(); base.training_ids = list(training_ids)
    base.mean = weights['feature_mean'].cpu().numpy()
    base.scale = weights['feature_scale'].cpu().numpy()
    for stage in range(1, 6):
        reg = Ridge()
        reg.coef_ = weights[f'coef_{stage}'].cpu().numpy()
        reg.intercept_ = weights[f'intercept_{stage}'].cpu().numpy()
        base.linear[stage] = reg
    return base


class SelectiveSmaOperator(ModalStateOperator):
    def __init__(self, base, config=None):
        super().__init__(base, config)
        self.decoder = SelectiveDecoder(self.decoder)
        self.requires_grad_(False)
        self.decoder.sma.requires_grad_(True)

    @classmethod
    def from_model(cls, source):
        original = source.state_dict()
        with torch.random.fork_rng(devices=[]):
            model = cls(base_from_weights(original, source.training_ids), source.config)
        weights = {}
        for name, tensor in original.items():
            if name.startswith('decoder.'):
                suffix = name[len('decoder.'):]
                weights['decoder.reference.'+suffix] = tensor
                weights['decoder.sma.'+suffix] = tensor
            else: weights[name] = tensor
        model.load_state_dict(weights)
        return model.eval()

    def save(self, path, *, metadata=None):
        with Path(path).open('xb') as stream:
            torch.save({'schema':'hro_selective_sma_v1','config':asdict(self.config),
                        'training_ids':self.training_ids,'weights':self.state_dict(),
                        'metadata':metadata or {}}, stream)

    @classmethod
    def load(cls, path):
        value = torch.load(path, weights_only=True)
        if value['schema'] != 'hro_selective_sma_v1': raise ValueError('Unexpected checkpoint schema')
        with torch.random.fork_rng(devices=[]):
            model = cls(base_from_weights(value['weights'], value['training_ids']), ModalConfig(**value['config']))
        model.load_state_dict(value['weights'])
        return model.eval()

    def parameter_report(self):
        result = super().parameter_report()
        result['total_parameters'] = sum(p.numel() for p in self.parameters())
        result['trainable'] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        result['frozen_parameters'] = result['total_parameters']-result['trainable']
        return result
