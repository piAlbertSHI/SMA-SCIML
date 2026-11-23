"""Decompose third-cycle stage-4 field error by time and material."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .formal_hierarchical import infer
from .multicycle_multistep import ROOT, data, write
from SciML.models.v12.multistep_historical import sha
from SciML.models.v13.hierarchical_operator import HierarchicalStateOperator


DEFAULT_SELECTED = ROOT / 'evidence/experiments/hro_protected_state_confirmation_20261123'
DEFAULT_FORMAL = ROOT / 'evidence/experiments/v13_formal_v1'
DEFAULT_OUTPUT = ROOT / 'evidence/local_studies/hro_protected_state_confirmation_20261123/temporal_error.json'
SEEDS = (20261121, 20261122, 20261123)


def _case_indices(d):
    test = np.flatnonzero(d['split'] == 'test')
    fixed = test[d['cycles'][test] == 3]
    return {
        material: np.asarray([i for i in fixed if ('_recas_' in str(d['case_ids'][i]).lower()) == (material == 'recAS')], dtype=int)
        for material in ('recA', 'recAS')
    }


def _time_errors(prediction, d, cases):
    rows = []
    for case_index in cases:
        use = ((d['field_case_index'] == case_index) & (d['field_cycle'] == 3) & (d['field_stage'] == 4))
        if use.sum() != 1:
            raise ValueError('Expected one c3s4 field row per case')
        position = int(np.flatnonzero(use)[0])
        error = prediction['field'][position] - d['field_truth'][position]
        rows.append({
            'case_id': str(d['case_ids'][case_index]),
            'sma_K': np.sqrt(np.mean(error[0] ** 2, axis=1)).tolist(),
            'htf_K': np.sqrt(np.mean(error[1] ** 2, axis=1)).tolist(),
        })
    return rows


def _aggregate(rows):
    result = {}
    for model_name, model_rows in rows.items():
        result[model_name] = {}
        for material, seed_rows in model_rows.items():
            result[model_name][material] = {}
            for channel in ('sma_K', 'htf_K'):
                values = np.asarray([[case[channel] for case in seed_row] for seed_row in seed_rows], dtype=float)
                result[model_name][material][channel] = {
                    'mean': values.mean(axis=(0, 1)).tolist(),
                    'sd_across_seed_case_means': values.mean(axis=1).std(axis=0, ddof=1).tolist(),
                    'case_count': len(seed_rows[0]),
                    'seed_count': len(seed_rows),
                }
    return result


def run(selected=DEFAULT_SELECTED, formal=DEFAULT_FORMAL, output=DEFAULT_OUTPUT):
    selected = Path(selected).resolve()
    formal = Path(formal).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Use a new temporal-error audit path')
    d, provenance = data()
    cases = _case_indices(d)
    rows = {'protected_state': {material: [] for material in cases}, 'formal_reference': {material: [] for material in cases}}
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        for seed in SEEDS:
            protected = infer(HierarchicalStateOperator.load(selected / f's{seed}' / 'model.pt'), d)
            reference = infer(HierarchicalStateOperator.load(formal / f's{seed}' / 'model.pt'), d)
            for model_name, prediction in (('protected_state', protected), ('formal_reference', reference)):
                for material, material_cases in cases.items():
                    rows[model_name][material].append(_time_errors(prediction, d, material_cases))
    result = {
        'status': 'completed_third_cycle_temporal_error_audit',
        'time_fraction': (np.arange(50) / 49.).tolist(),
        'material_case_ids': {material: d['case_ids'][indices].tolist() for material, indices in cases.items()},
        'data_snapshot_sha256': provenance['snapshot_sha256'],
        'aggregate': _aggregate(rows),
        'fresh_training_calls': 0,
        'fresh_solver_calls': 0,
        'source_sha256': sha(ROOT / 'examples/audit_hro_temporal_error.py'),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write(output, result)
    return {'status': result['status'], 'fresh_training_calls': 0, 'fresh_solver_calls': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selected', type=Path, default=DEFAULT_SELECTED)
    parser.add_argument('--formal', type=Path, default=DEFAULT_FORMAL)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.selected, args.formal, args.output), indent=2))
