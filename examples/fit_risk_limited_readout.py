"""Fit the shared response correction to explicitly selected, aligned case arrays.

This entry point fits the response head only; it does not retrain the backbone,
choose hyperparameters, score evaluation cases or promote a model automatically.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .multicycle_multistep import ROOT, write
from .stage_readout_study import cache
from .risk_limited_readout_study import fit_response
from SciML.models.v12.multistep_historical import sha
from SciML.models.v13.stage_thermal_readout import StageThermalReadoutOperator
from SciML.models.v13.risk_limited_readout import RiskLimitedRelaxationOperator


INPUT_ARRAYS = ('initial', 'control', 'states', 'fields')


def fit_arrays(reference, arrays, case_ids, selected):
    """Fit on whole cases; all cases in a batch must share a cycle horizon."""
    n = len(case_ids)
    if any(not isinstance(value, str) or not value for value in case_ids) or n != len(set(case_ids)):
        raise ValueError('Unique nonempty string case IDs required')
    if any(type(i) is not int or not 0 <= i < n for i in selected):
        raise ValueError('Fit case indices must be integers within the dataset')
    if len(selected) < 3 or len(selected) != len(set(selected)):
        raise ValueError('Select at least three distinct valid case indices')
    if set(arrays) != set(INPUT_ARRAYS):
        raise ValueError('Expected initial, control, states and fields arrays')
    fields = arrays['fields']
    if fields.ndim != 5 or fields.shape[0] != n or fields.shape[2:] != (2, 50, 101):
        raise ValueError('Expected fields with shape N x 2C x 2 x 50 x 101')
    exchanges = fields.shape[1]
    if exchanges < 2 or exchanges % 2:
        raise ValueError('Paired exchange stages required')
    cycles = exchanges // 2
    shapes = {'initial': (n, 515), 'control': (n, 9), 'states': (n, 5 * cycles - 1, 515)}
    for key, shape in shapes.items():
        if arrays[key].shape != shape:
            raise ValueError(f'Invalid {key} shape, expected {shape}')
    for key, value in arrays.items():
        if value.dtype != torch.float64 or value.device.type != 'cpu' or not torch.isfinite(value[selected]).all():
            raise ValueError(f'{key}: selected training arrays must be finite CPU float64 tensors')
    values = cache(reference.reference.backbone, arrays, selected)
    head, scale, info, members = fit_response(values)
    if info['central']['rate_at_numerical_bound'] or any(r['fit']['rate_at_numerical_bound'] for r in info['case_deletions']):
        raise ValueError('Fitted response rate reached its numerical bound')
    info.update(
        fit_case_ids=[case_ids[i] for i in selected],
        cycles_per_case=cycles,
        fit_scope='response rate and empirical error scale only, baseline weights frozen',
        mean_scale=.5,
        error_scale_multiplier=1.,
        candidate_promoted=False,
    )
    return RiskLimitedRelaxationOperator(reference, head, error_scale=scale), info, members


def run(dataset, fit_ids, checkpoint, output):
    if output.exists():
        raise FileExistsError(output)
    with np.load(dataset, allow_pickle=False) as archive:
        if set(archive.files) != {'case_ids', *INPUT_ARRAYS}:
            raise ValueError('Unexpected dataset schema')
        case_array = archive['case_ids']
        if case_array.ndim != 1 or case_array.dtype.kind != 'U':
            raise ValueError('case_ids must be a one-dimensional Unicode array')
        ids = case_array.tolist()
        arrays = {}
        for key in INPUT_ARRAYS:
            value = archive[key]
            if value.dtype.kind not in 'fiu':
                raise ValueError(f'{key} must contain real numeric values')
            arrays[key] = torch.as_tensor(value, dtype=torch.float64)
    chosen = json.loads(fit_ids.read_text(encoding='utf-8'))
    if not isinstance(chosen, list) or not all(isinstance(value, str) for value in chosen):
        raise ValueError('Fit IDs must be a JSON list of strings')
    index_by_id = {case_id: index for index, case_id in enumerate(ids)}
    if any(case_id not in index_by_id for case_id in chosen):
        raise ValueError('Unknown fit case ID')
    reference = StageThermalReadoutOperator.load(checkpoint)
    start = time.perf_counter()
    model, info, members = fit_arrays(reference, arrays, ids, [index_by_id[case_id] for case_id in chosen])
    sources = (
        'examples/fit_risk_limited_readout.py',
        'examples/risk_limited_readout_study.py',
        'examples/relaxation_readout_study.py',
        'examples/stage_readout_study.py',
        'SciML/models/v13/risk_limited_readout.py',
        'SciML/models/v13/relaxation_readout.py',
    )
    protocol = {
        'dataset_sha256': sha(dataset), 'selection_sha256': sha(fit_ids), 'reference_sha256': sha(checkpoint),
        'reference_training_ids': reference.training_ids, 'fit': info,
        'fit_seconds': time.perf_counter() - start,
        'source_sha256': {name: sha(ROOT / name) for name in sources},
        'evaluation_cases_scored': False, 'fresh_solver_calls': 0, 'neural_training_updates': 0,
    }
    output.mkdir(parents=True)
    model.save(output / 'model.pt', metadata=protocol)
    torch.save({'members': [member.state_dict() for member in members]}, output / 'calibration.pt')
    write(output / 'protocol.json', protocol)
    write(output / 'manifest.json', {'artifacts': {p.name: sha(p) for p in output.iterdir() if p.is_file()}})
    return {
        'status': 'fitted_response_head_pending_validation', 'fit_cases': len(chosen),
        'cycles_per_case': info['cycles_per_case'], 'fit_seconds': protocol['fit_seconds'],
        'candidate_promoted': False,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'fit-ids', 'reference', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    with torch.no_grad(), threadpool_limits(limits=1):
        result = run(args.dataset, args.fit_ids, args.reference, args.output)
    print(json.dumps(result, indent=2))
