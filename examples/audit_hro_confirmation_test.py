"""Post-selection third-cycle audit for the confirmed protected-state HRO.

This read-only audit uses the existing historical test cohort after candidate
selection. Its scores do not alter the selection decision and no solver is
called.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .formal_hierarchical import cohorts, infer
from .multicycle_multistep import ROOT, data, score, write
from SciML.models.v12.multistep_historical import sha
from SciML.models.v13.hierarchical_operator import HierarchicalStateOperator


DEFAULT_EXPERIMENT = ROOT / 'evidence/experiments/hro_protected_state_confirmation_20261123'
DEFAULT_OUTPUT = ROOT / 'evidence/local_studies/hro_protected_state_confirmation_20261123/test_audit.json'
SEEDS = (20261121, 20261122, 20261123)


def run(experiment=DEFAULT_EXPERIMENT, output=DEFAULT_OUTPUT):
    experiment = Path(experiment).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Use a new post-selection audit path')
    d, provenance = data()
    test_cases = np.flatnonzero(d['split'] == 'test')
    fixed = test_cases[d['cycles'][test_cases] == 3]
    material_cases = {
        material: np.asarray([i for i in test_cases if ('_recas_' in str(d['case_ids'][i]).lower()) == (material == 'recAS')], dtype=int)
        for material in ('recA', 'recAS')
    }
    reports = []
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        for seed in SEEDS:
            model = HierarchicalStateOperator.load(experiment / f's{seed}' / 'model.pt')
            prediction = infer(model, d)
            row = {
                'seed': seed,
                'all_test': cohorts(prediction, d)['all_test'],
                'fixed_three_cycle': cohorts(prediction, d)['fixed_three_cycle'],
                'per_material': {
                    material: score(prediction, d, selected)
                    for material, selected in material_cases.items()
                },
                'per_three_cycle_case': cohorts(prediction, d)['per_three_cycle_case'],
            }
            reports.append(row)
    aggregate = {}
    first = reports[0]['fixed_three_cycle']
    for category, groups in first.items():
        aggregate[category] = {}
        for key, metrics in groups.items():
            aggregate[category][key] = {'case_count': metrics['case_count']}
            for metric in metrics:
                if metric == 'case_count':
                    continue
                values = [row['fixed_three_cycle'][category][key][metric] for row in reports]
                aggregate[category][key][metric] = {
                    'mean': float(np.mean(values)),
                    'sd': float(np.std(values, ddof=1)),
                    'seeds': list(SEEDS),
                    'values': values,
                }
    summary = {
        'status': 'completed_post_selection_third_cycle_audit',
        'selection_boundary': 'Candidate was selected before this test-cohort audit. Scores were not used to change the candidate.',
        'experiment': str(experiment.relative_to(ROOT)).replace('\\', '/'),
        'test_case_ids': d['case_ids'][test_cases].tolist(),
        'fixed_three_cycle_case_ids': d['case_ids'][fixed].tolist(),
        'material_case_ids': {key: d['case_ids'][value].tolist() for key, value in material_cases.items()},
        'data_snapshot_sha256': provenance['snapshot_sha256'],
        'reports': reports,
        'fixed_three_cycle_aggregate': aggregate,
        'fresh_training_calls': 0,
        'fresh_solver_calls': 0,
        'source_sha256': sha(ROOT / 'examples/audit_hro_confirmation_test.py'),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write(output, summary)
    return {
        'status': summary['status'],
        'fixed_three_cycle_cases': len(fixed),
        'fresh_training_calls': 0,
        'fresh_solver_calls': 0,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.experiment, args.output), indent=2))
