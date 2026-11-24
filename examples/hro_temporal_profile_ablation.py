"""Fixed-budget late-time field-profile screen under protected-state gates."""
import argparse
import copy
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
    _predict,
    _rmse,
    _score,
)
from .hro_protected_state_ablation import SEEDS
from .hro_spatial_ablation import _train_one
from .multicycle_multistep import ROOT, write
from .train_hierarchical import training_data
from SciML.models.v12.multistep_historical import sha
from SciML.models.v13.hierarchical_operator import HierarchicalStateOperator


DEFAULT = ROOT / 'evidence/experiments/hro_temporal_profile_ablation_20261123'
UPDATES = 120
BATCH_CASES = 2
LR = .0003
BASE = {
    'spatial_points': 51,
    'gradient_weight': .01,
    'curvature_weight': 0.,
    'curvature_channel_weights': None,
    'field_channel_weights': [1., 1.5],
    'curvature_start_fraction': 1.,
    'spatial_jitter_fraction': .5,
    'entry_time_weight': .5,
    'entry_field_weight': .75,
    'state_block_weights': [1., 1., 1., 1., 1., 2.5, 2.5, 1.25],
    'state_stage_weights': [1., 1., 1., 1., 1., 1., 1.25, 1.5, 1.75],
}
CANDIDATES = {
    'protected_state_control': {**copy.deepcopy(BASE), 'late_time_weight': 0.0},
    'protected_state_late_light': {**copy.deepcopy(BASE), 'late_time_weight': 0.25},
    'protected_state_late_moderate': {**copy.deepcopy(BASE), 'late_time_weight': 0.5},
}


def _temporal_score(model, arrays):
    prediction = _predict(model, arrays)
    truth = arrays['fields'].numpy()
    times = np.linspace(0., 1., truth.shape[3])
    late = times >= .7
    midlate = times >= .4
    return {
        'sma_late_K': _rmse(prediction['fields'][:, :, 0, late, :], truth[:, :, 0, late, :]),
        'sma_midlate_K': _rmse(prediction['fields'][:, :, 0, midlate, :], truth[:, :, 0, midlate, :]),
        'htf_late_K': _rmse(prediction['fields'][:, :, 1, late, :], truth[:, :, 1, late, :]),
    }


def _aggregate_temporal(rows, seeds):
    base_rows = {candidate: [row['base'] for row in candidate_rows] for candidate, candidate_rows in rows.items()}
    result = _aggregate(base_rows, seeds)
    for candidate, candidate_rows in rows.items():
        result[candidate]['temporal'] = {}
        for key in candidate_rows[0]['temporal']:
            values = [row['temporal'][key] for row in candidate_rows]
            result[candidate]['temporal'][key] = {
                'mean': float(np.mean(values)),
                'sd': float(np.std(values, ddof=1)),
                'seeds': list(seeds),
                'values': values,
            }
    return result


def _gate(aggregate):
    reference = aggregate['protected_state_control']
    gates = {}
    for candidate, metrics in aggregate.items():
        state_ratios = {
            key: metrics['state'][key]['mean'] / reference['state'][key]['mean'] - 1.
            for key in metrics['state']
        }
        material_safe = all(
            metrics['per_material'][material]['combined_future_K']['mean'] <= reference['per_material'][material]['combined_future_K']['mean']
            for material in ('recA', 'recAS'))
        entry_safe = all(
            metrics['field'][key]['mean'] <= reference['field'][key]['mean']
            for key in ('sma_entry_K', 'htf_entry_K'))
        late_sma_improved = metrics['temporal']['sma_late_K']['mean'] <= reference['temporal']['sma_late_K']['mean']
        seed_late_safe = all(
            a <= b for a, b in zip(
                metrics['temporal']['sma_late_K']['values'],
                reference['temporal']['sma_late_K']['values']))
        gates[candidate] = {
            'state_safe': bool(max(state_ratios.values()) <= .05),
            'entry_safe': bool(entry_safe),
            'material_field_safe': bool(material_safe),
            'late_sma_improved_or_preserved': bool(late_sma_improved),
            'three_seed_late_sma_non_regression': bool(seed_late_safe),
            'max_state_regression_fraction': float(max(state_ratios.values())),
            'late_sma_change_fraction': float(metrics['temporal']['sma_late_K']['mean'] / reference['temporal']['sma_late_K']['mean'] - 1.),
            'passed': bool(candidate != 'protected_state_control' and max(state_ratios.values()) <= .05 and entry_safe and material_safe and late_sma_improved and seed_late_safe),
        }
    passed = [candidate for candidate, gate in gates.items() if gate['passed']]
    selected = min(passed, key=lambda name: aggregate[name]['temporal']['sma_late_K']['mean']) if passed else None
    return {'candidates': gates, 'selected_candidate': selected}


def run(output=DEFAULT, updates=UPDATES, batch_cases=BATCH_CASES, lr=LR, candidate_settings=None, experiment_name='hro_temporal_profile_ablation_20261123'):
    if output.exists():
        raise FileExistsError('Use a new temporal-profile experiment directory')
    arrays, all_ids, hashes = training_data()
    fit_arrays = _case_subset(arrays, all_ids, FIT_CASE_IDS)
    validation_arrays = _case_subset(arrays, all_ids, VALIDATION_CASE_IDS)
    output.mkdir(parents=True)
    candidate_settings = CANDIDATES if candidate_settings is None else candidate_settings
    protocol = {
        'status': 'written_before_fitting',
        'experiment': experiment_name,
        'seeds': list(SEEDS),
        'updates_per_seed': updates,
        'batch_cases': batch_cases,
        'learning_rate': lr,
        'fit_case_ids': list(FIT_CASE_IDS),
        'validation_case_ids': list(VALIDATION_CASE_IDS),
        'candidates': candidate_settings,
        'data_sha256': hashes,
        'fresh_solver_calls': 0,
        'sealed_test_cases_opened': False,
        'source_sha256': {
            'examples/hro_temporal_profile_ablation.py': sha(ROOT / 'examples/hro_temporal_profile_ablation.py'),
            'examples/hro_spatial_ablation.py': sha(ROOT / 'examples/hro_spatial_ablation.py'),
            'SciML/models/v13/hierarchical_operator.py': sha(ROOT / 'SciML/models/v13/hierarchical_operator.py'),
        },
        'environment': {'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__},
    }
    write(output / 'protocol.json', protocol)
    rows = {candidate: [] for candidate in candidate_settings}
    fit_seconds = {}
    started = time.perf_counter()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        for candidate, settings in candidate_settings.items():
            for seed in SEEDS:
                destination = output / candidate / f's{seed}'
                fit = _train_one(destination, fit_arrays, list(FIT_CASE_IDS), hashes, candidate, seed,
                                 updates, batch_cases, lr, settings_override=settings)
                model = HierarchicalStateOperator.load(destination / 'model.pt')
                validation = {'base': _score(model, validation_arrays, VALIDATION_CASE_IDS),
                              'temporal': _temporal_score(model, validation_arrays)}
                write(destination / 'validation.json', validation)
                rows[candidate].append(validation)
                fit_seconds[f'{candidate}/s{seed}'] = fit['training_seconds']
    aggregate = _aggregate_temporal(rows, SEEDS)
    gates = _gate(aggregate)
    summary = {
        'status': 'completed_hro_temporal_profile_screening',
        'fresh_solver_calls': 0,
        'sealed_test_cases_opened': False,
        'aggregate': aggregate,
        'gates': gates,
        'fit_seconds_by_run': fit_seconds,
        'total_fit_seconds': float(sum(fit_seconds.values())),
        'total_seconds': float(time.perf_counter() - started),
        'artifacts': {
            path.relative_to(output).as_posix(): sha(path)
            for path in sorted(output.rglob('*')) if path.is_file() and path.name != 'summary.json'
        },
    }
    write(output / 'summary.json', summary)
    return {'status': summary['status'], 'selected_candidate': gates['selected_candidate'],
            'total_fit_seconds': summary['total_fit_seconds'], 'total_seconds': summary['total_seconds'],
            'fresh_solver_calls': 0}


def verify(output=DEFAULT):
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
                replay = {'base': _score(model, validation_arrays, protocol['validation_case_ids']),
                          'temporal': _temporal_score(model, validation_arrays)}
                assert replay == saved
                rows[candidate].append(replay)
                training_protocol = json.loads((destination / 'protocol.json').read_text())
                assert training_protocol['training_case_ids'] == protocol['fit_case_ids']
                assert training_protocol['updates'] == protocol['updates_per_seed']
                assert len(json.loads((destination / 'history.json').read_text())) == protocol['updates_per_seed']
    aggregate = _aggregate_temporal(rows, protocol['seeds'])
    assert aggregate == summary['aggregate']
    assert _gate(aggregate) == summary['gates']
    return {'status': 'verified_hro_temporal_profile_screening',
            'checkpoint_replays': len(protocol['candidates']) * len(protocol['seeds']),
            'scores_recomputed': True, 'fresh_solver_calls': 0,
            'sealed_test_cases_opened': False}


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
