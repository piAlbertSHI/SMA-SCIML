"""Fixed-budget HRO ablation with explicit state and stage protection.

This study reuses the six fit cases and two material-stratified validation
cases from the preceding internal screen. It adds normalized state-block and
transition-stage weights to the loss while preserving the HRO architecture,
frozen prior and saved historical snapshots. No solver is called.
"""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .hro_internal_validation import (
    FIT_CASE_IDS,
    VALIDATION_CASE_IDS,
    _aggregate,
    _case_subset,
    _gate,
    _score,
)
from .hro_spatial_ablation import _train_one
from .multicycle_multistep import ROOT, write
from .train_hierarchical import training_data
from SciML.models.v12.multistep_historical import sha


DEFAULT = ROOT / 'evidence/experiments/hro_protected_state_ablation_20261123'
SEEDS = (20261121, 20261122, 20261123)
UPDATES = 160
BATCH_CASES = 2
LR = .0003

STATE_GUARD = [1., 1., 1., 1., 1., 2., 2., 1.]
LATE_STAGE_GUARD = [1., 1., 1., 1., 1., 1., 1.25, 1.5, 1.75]
STRONG_STATE_GUARD = [1., 1., 1., 1., 1., 3., 3., 1.]

CANDIDATES = {
    'coordinate_reference': {
        'spatial_points': 26, 'gradient_weight': .01, 'curvature_weight': 0.,
        'curvature_channel_weights': None, 'field_channel_weights': [1., 1.],
        'curvature_start_fraction': 1., 'spatial_jitter_fraction': 0.,
    },
    'jitter_fluid2_stateguard': {
        'spatial_points': 51, 'gradient_weight': .01, 'curvature_weight': 0.,
        'curvature_channel_weights': None, 'field_channel_weights': [1., 2.],
        'curvature_start_fraction': 1., 'spatial_jitter_fraction': .5,
        'entry_time_weight': 1., 'entry_field_weight': 1.,
        'state_block_weights': STATE_GUARD,
    },
    'jitter_fluid2_stageguard': {
        'spatial_points': 51, 'gradient_weight': .01, 'curvature_weight': 0.,
        'curvature_channel_weights': None, 'field_channel_weights': [1., 2.],
        'curvature_start_fraction': 1., 'spatial_jitter_fraction': .5,
        'entry_time_weight': 1., 'entry_field_weight': 1.,
        'state_block_weights': STATE_GUARD, 'state_stage_weights': LATE_STAGE_GUARD,
    },
    'jitter_fluid2_pareto': {
        'spatial_points': 51, 'gradient_weight': .01, 'curvature_weight': 0.,
        'curvature_channel_weights': None, 'field_channel_weights': [1., 1.5],
        'curvature_start_fraction': 1., 'spatial_jitter_fraction': .5,
        'entry_time_weight': .5, 'entry_field_weight': .75,
        'state_block_weights': [1., 1., 1., 1., 1., 2.5, 2.5, 1.25],
        'state_stage_weights': LATE_STAGE_GUARD,
    },
    'jitter_fluid2_strong_stateguard': {
        'spatial_points': 51, 'gradient_weight': .01, 'curvature_weight': 0.,
        'curvature_channel_weights': None, 'field_channel_weights': [1., 2.],
        'curvature_start_fraction': 1., 'spatial_jitter_fraction': .5,
        'entry_time_weight': 1., 'entry_field_weight': 1.,
        'state_block_weights': STRONG_STATE_GUARD, 'state_stage_weights': LATE_STAGE_GUARD,
    },
}


def run(output=DEFAULT, updates=UPDATES, batch_cases=BATCH_CASES, lr=LR):
    if output.exists():
        raise FileExistsError('Use a new experiment directory')
    arrays, all_ids, hashes = training_data()
    if set(all_ids) != set(FIT_CASE_IDS + VALIDATION_CASE_IDS):
        raise ValueError('Unexpected historical training split')
    fit_arrays = _case_subset(arrays, all_ids, FIT_CASE_IDS)
    validation_arrays = _case_subset(arrays, all_ids, VALIDATION_CASE_IDS)
    output.mkdir(parents=True)
    protocol = {
        'status': 'written_before_fitting',
        'experiment': 'hro_protected_state_ablation_20261123',
        'seeds': list(SEEDS),
        'updates_per_seed': updates,
        'batch_cases': batch_cases,
        'learning_rate': lr,
        'fit_case_ids': list(FIT_CASE_IDS),
        'validation_case_ids': list(VALIDATION_CASE_IDS),
        'candidates': CANDIDATES,
        'data_sha256': hashes,
        'source_sha256': {
            'examples/hro_protected_state_ablation.py': sha(ROOT / 'examples/hro_protected_state_ablation.py'),
            'examples/hro_spatial_ablation.py': sha(ROOT / 'examples/hro_spatial_ablation.py'),
            'SciML/models/v13/hierarchical_operator.py': sha(ROOT / 'SciML/models/v13/hierarchical_operator.py'),
        },
        'fresh_solver_calls': 0,
        'sealed_test_cases_opened': False,
        'environment': {'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__},
    }
    write(output / 'protocol.json', protocol)
    rows = {candidate: [] for candidate in CANDIDATES}
    fit_seconds = {}
    started = time.perf_counter()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        for candidate, settings in CANDIDATES.items():
            for seed in SEEDS:
                destination = output / candidate / f's{seed}'
                fit = _train_one(destination, fit_arrays, list(FIT_CASE_IDS), hashes,
                                 candidate, seed, updates, batch_cases, lr,
                                 settings_override=settings)
                from SciML.models.v13.hierarchical_operator import HierarchicalStateOperator
                model = HierarchicalStateOperator.load(destination / 'model.pt')
                validation = _score(model, validation_arrays, VALIDATION_CASE_IDS)
                write(destination / 'validation.json', validation)
                rows[candidate].append(validation)
                fit_seconds[f'{candidate}/s{seed}'] = fit['training_seconds']
    aggregate = _aggregate(rows, SEEDS)
    gate = _gate(aggregate)
    summary = {
        'status': 'completed_hro_protected_state_screening',
        'fresh_solver_calls': 0,
        'sealed_test_cases_opened': False,
        'aggregate': aggregate,
        'gates': gate,
        'fit_seconds_by_run': fit_seconds,
        'total_fit_seconds': float(sum(fit_seconds.values())),
        'total_seconds': float(time.perf_counter() - started),
        'artifacts': {
            path.relative_to(output).as_posix(): sha(path)
            for path in sorted(output.rglob('*'))
            if path.is_file() and path.name != 'summary.json'
        },
    }
    write(output / 'summary.json', summary)
    return {
        'status': summary['status'],
        'selected_candidate': gate['selected_candidate'],
        'total_fit_seconds': summary['total_fit_seconds'],
        'total_seconds': summary['total_seconds'],
        'fresh_solver_calls': 0,
        'sealed_test_cases_opened': False,
    }


def verify(output=DEFAULT):
    from SciML.models.v13.hierarchical_operator import HierarchicalStateOperator
    protocol = json.loads((output / 'protocol.json').read_text())
    summary = json.loads((output / 'summary.json').read_text())
    for relative, expected in summary['artifacts'].items():
        if sha(output / relative) != expected:
            raise ValueError(f'Artifact hash mismatch: {relative}')
    arrays, all_ids, hashes = training_data()
    if protocol['data_sha256'] != hashes:
        raise ValueError('Training snapshot checksum changed')
    validation_arrays = _case_subset(arrays, all_ids, protocol['validation_case_ids'])
    rows = {candidate: [] for candidate in protocol['candidates']}
    with threadpool_limits(limits=1):
        for candidate in protocol['candidates']:
            for seed in protocol['seeds']:
                destination = output / candidate / f's{seed}'
                model = HierarchicalStateOperator.load(destination / 'model.pt')
                saved = json.loads((destination / 'validation.json').read_text())
                replay = _score(model, validation_arrays, protocol['validation_case_ids'])
                assert replay == saved
                rows[candidate].append(replay)
                training_protocol = json.loads((destination / 'protocol.json').read_text())
                assert training_protocol['training_case_ids'] == protocol['fit_case_ids']
                assert training_protocol['updates'] == protocol['updates_per_seed']
                history = json.loads((destination / 'history.json').read_text())
                assert len(history) == protocol['updates_per_seed']
                assert all(set(row['train_case_ids']) <= set(protocol['fit_case_ids']) for row in history)
    aggregate = _aggregate(rows, protocol['seeds'])
    assert aggregate == summary['aggregate']
    assert _gate(aggregate) == summary['gates']
    return {
        'status': 'verified_hro_protected_state_screening',
        'checkpoint_replays': len(protocol['candidates']) * len(protocol['seeds']),
        'scores_recomputed': True,
        'fresh_solver_calls': 0,
        'sealed_test_cases_opened': False,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    parser.add_argument('--updates', type=int, default=UPDATES)
    parser.add_argument('--batch-cases', type=int, default=BATCH_CASES)
    parser.add_argument('--lr', type=float, default=LR)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    result = verify(args.output) if args.verify_only else run(args.output, args.updates, args.batch_cases, args.lr)
    print(json.dumps(result, indent=2))
