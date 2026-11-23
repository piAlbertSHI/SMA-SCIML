"""Audit stage-resolved metrics from an existing HRO internal screen.

This is a read-only replay of saved checkpoints. It performs no fitting,
does not open the sealed test cases and does not call a physical solver.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .hro_internal_validation import (
    VALIDATION_CASE_IDS,
    _case_subset,
    _rmse,
    _predict,
)
from .multicycle_multistep import ROOT
from .train_hierarchical import training_data
from SciML.models.v12.learning_historical_rollout import BLOCKS
from SciML.models.v12.multistep_historical import sha
from SciML.models.v13.hierarchical_operator import HierarchicalStateOperator


DEFAULT_EXPERIMENT = ROOT / 'evidence/experiments/hro_internal_validation_20261123_r3'
DEFAULT_OUTPUT = ROOT / 'evidence/local_studies/hro_internal_validation_20261123_r3/stage_metrics.json'
STATE_KEYS = ('c1s1', 'c1s2', 'c1s3', 'c1s4', 'c1s5', 'c2s1', 'c2s2', 'c2s3', 'c2s4')
FIELD_KEYS = ('c1s2', 'c1s4', 'c2s2', 'c2s4')


def _stage_score(model, arrays):
    prediction = _predict(model, arrays)
    state_truth = arrays['states'].numpy()
    field_truth = arrays['fields'].numpy()
    state = {}
    for index, key in enumerate(STATE_KEYS):
        state[key] = {name: _rmse(prediction['states'][:, index, block], state_truth[:, index, block])
                      for name, block in BLOCKS.items()}
    field = {}
    for index, key in enumerate(FIELD_KEYS):
        field[key] = {
            'sma_entry_K': _rmse(prediction['fields'][:, index, 0, 0, :], field_truth[:, index, 0, 0, :]),
            'htf_entry_K': _rmse(prediction['fields'][:, index, 1, 0, :], field_truth[:, index, 1, 0, :]),
            'sma_future_K': _rmse(prediction['fields'][:, index, 0, 1:, :], field_truth[:, index, 0, 1:, :]),
            'htf_future_K': _rmse(prediction['fields'][:, index, 1, 1:, :], field_truth[:, index, 1, 1:, :]),
        }
        field[key]['combined_future_K'] = float(np.mean([
            field[key]['sma_future_K'], field[key]['htf_future_K']]))
    return {'state_by_stage': state, 'field_by_stage': field}


def _aggregate(rows, seeds):
    result = {}
    for candidate, candidate_rows in rows.items():
        result[candidate] = {}
        for category in ('state_by_stage', 'field_by_stage'):
            result[candidate][category] = {}
            for key in candidate_rows[0][category]:
                result[candidate][category][key] = {}
                for metric in candidate_rows[0][category][key]:
                    values = [row[category][key][metric] for row in candidate_rows]
                    result[candidate][category][key][metric] = {
                        'mean': float(np.mean(values)),
                        'sd': float(np.std(values, ddof=1)),
                        'seeds': list(seeds),
                        'values': values,
                    }
    return result


def run(experiment=DEFAULT_EXPERIMENT, output=DEFAULT_OUTPUT):
    experiment = Path(experiment).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Use a new stage-metrics output path')
    protocol = json.loads((experiment / 'protocol.json').read_text())
    arrays, all_ids, hashes = training_data()
    validation = _case_subset(arrays, all_ids, VALIDATION_CASE_IDS)
    if 'candidates' in protocol:
        candidates = tuple(protocol['candidates'])
    else:
        candidates = (protocol['selected_candidate'],)
    rows = {candidate: [] for candidate in candidates}
    with threadpool_limits(limits=1):
        for candidate in candidates:
            for seed in protocol['seeds']:
                model_dir = experiment / f's{seed}' if 'selected_candidate' in protocol else experiment / candidate / f's{seed}'
                model = HierarchicalStateOperator.load(model_dir / 'model.pt')
                rows[candidate].append(_stage_score(model, validation))
    summary = {
        'status': 'completed_stage_resolved_readonly_audit',
        'experiment': str(experiment.relative_to(ROOT)).replace('\\', '/'),
        'validation_case_ids': list(VALIDATION_CASE_IDS),
        'seeds': protocol['seeds'],
        'data_sha256': hashes,
        'aggregate': _aggregate(rows, protocol['seeds']),
        'fresh_training_calls': 0,
        'fresh_solver_calls': 0,
        'sealed_test_cases_opened': False,
        'source_sha256': sha(ROOT / 'examples/audit_hro_internal_validation_stages.py'),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding='utf-8', newline='\n')
    return {'status': summary['status'], 'fresh_training_calls': 0, 'fresh_solver_calls': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.experiment, args.output), indent=2))
