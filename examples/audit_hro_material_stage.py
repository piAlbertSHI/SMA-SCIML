"""Compare selected HRO and formal reference by material and stage.

This is a post-selection replay over the existing historical test cohort. The
comparison is contextual because the selected checkpoint uses six fit cases,
whereas the formal reference uses its archived eight-case protocol.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .formal_hierarchical import infer
from .multicycle_multistep import ROOT, data, score, write
from SciML.models.v12.multistep_historical import sha
from SciML.models.v13.hierarchical_operator import HierarchicalStateOperator


DEFAULT_SELECTED = ROOT / 'evidence/experiments/hro_protected_state_confirmation_20261123'
DEFAULT_FORMAL = ROOT / 'evidence/experiments/v13_formal_v1'
DEFAULT_OUTPUT = ROOT / 'evidence/local_studies/hro_protected_state_confirmation_20261123/material_stage_diagnostics.json'
SEEDS = (20261121, 20261122, 20261123)


def _material_cases(d):
    test = np.flatnonzero(d['split'] == 'test')
    return {
        material: np.asarray([i for i in test if ('_recas_' in str(d['case_ids'][i]).lower()) == (material == 'recAS')], dtype=int)
        for material in ('recA', 'recAS')
    }


def _aggregate(rows):
    result = {}
    for model_name, model_rows in rows.items():
        result[model_name] = {}
        for material in ('recA', 'recAS'):
            result[model_name][material] = {}
            for category in ('state', 'field'):
                result[model_name][material][category] = {}
                first = model_rows[0][material][category]
                for stage_key, metrics in first.items():
                    result[model_name][material][category][stage_key] = {'case_count': metrics['case_count']}
                    for metric in metrics:
                        if metric == 'case_count':
                            continue
                        values = [row[material][category][stage_key][metric] for row in model_rows]
                        result[model_name][material][category][stage_key][metric] = {
                            'mean': float(np.mean(values)),
                            'sd': float(np.std(values, ddof=1)),
                            'seeds': list(SEEDS),
                            'values': values,
                        }
    return result


def run(selected=DEFAULT_SELECTED, formal=DEFAULT_FORMAL, output=DEFAULT_OUTPUT):
    selected = Path(selected).resolve()
    formal = Path(formal).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Use a new material-stage audit path')
    d, provenance = data()
    material_cases = _material_cases(d)
    rows = {'protected_state': [], 'formal_reference': []}
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        for seed in SEEDS:
            protected = HierarchicalStateOperator.load(selected / f's{seed}' / 'model.pt')
            reference = HierarchicalStateOperator.load(formal / f's{seed}' / 'model.pt')
            protected_prediction = infer(protected, d)
            reference_prediction = infer(reference, d)
            for name, prediction, destination in (
                ('protected_state', protected_prediction, rows['protected_state']),
                ('formal_reference', reference_prediction, rows['formal_reference']),
            ):
                destination.append({
                    material: score(prediction, d, cases)
                    for material, cases in material_cases.items()
                })
    result = {
        'status': 'completed_post_selection_material_stage_diagnostic',
        'context': 'Protected-state checkpoint uses six fit cases. Formal reference uses its archived eight-case protocol. The comparison is contextual and not an architecture-only ablation.',
        'selected_experiment': str(selected.relative_to(ROOT)).replace('\\', '/'),
        'formal_experiment': str(formal.relative_to(ROOT)).replace('\\', '/'),
        'material_case_ids': {key: d['case_ids'][value].tolist() for key, value in material_cases.items()},
        'data_snapshot_sha256': provenance['snapshot_sha256'],
        'aggregate': _aggregate(rows),
        'fresh_training_calls': 0,
        'fresh_solver_calls': 0,
        'source_sha256': sha(ROOT / 'examples/audit_hro_material_stage.py'),
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
