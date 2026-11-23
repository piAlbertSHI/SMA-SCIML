"""Low-cost material-conditioned HTF weighting screen for the HRO checkpoint."""
import argparse
import copy
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .hro_internal_validation import FIT_CASE_IDS, VALIDATION_CASE_IDS, _aggregate, _case_subset, _score
from .hro_protected_state_ablation import SEEDS
from .hro_spatial_ablation import _train_one
from .multicycle_multistep import ROOT, write
from .train_hierarchical import training_data
from SciML.models.v12.multistep_historical import sha
from SciML.models.v13.hierarchical_operator import HierarchicalStateOperator


DEFAULT = ROOT / 'evidence/experiments/hro_material_conditioned_ablation_20261123'
BASE = {
    'spatial_points': 51, 'gradient_weight': .01, 'curvature_weight': 0.,
    'curvature_channel_weights': None, 'field_channel_weights': [1., 1.5],
    'curvature_start_fraction': 1., 'spatial_jitter_fraction': .5,
    'entry_time_weight': .5, 'entry_field_weight': .75,
    'state_block_weights': [1., 1., 1., 1., 1., 2.5, 2.5, 1.25],
    'state_stage_weights': [1., 1., 1., 1., 1., 1., 1.25, 1.5, 1.75],
}
CANDIDATES = {
    'global_pareto_control': copy.deepcopy(BASE),
    'recAS_htf_guard': {
        **copy.deepcopy(BASE),
        'material_field_channel_weights': {'recA': [1., 1.25], 'recAS': [1., 1.75]},
    },
    'material_htf_balanced': {
        **copy.deepcopy(BASE),
        'material_field_channel_weights': {'recA': [1., 1.5], 'recAS': [1., 2.]},
    },
}
UPDATES = 120
BATCH_CASES = 2
LR = .0003


def _material_gate(aggregate):
    reference = aggregate['global_pareto_control']
    gates = {}
    for candidate, metrics in aggregate.items():
        state_ratios = {key: metrics['state'][key]['mean'] / reference['state'][key]['mean'] - 1.
                        for key in metrics['state']}
        material_ratios = {material: {
            'recAS_htf': metrics['per_material'][material]['htf_future_K']['mean'] /
            reference['per_material'][material]['htf_future_K']['mean'] - 1.,
            'combined': metrics['per_material'][material]['combined_future_K']['mean'] /
            reference['per_material'][material]['combined_future_K']['mean'] - 1.,
        } for material in ('recA', 'recAS')}
        state_safe = max(state_ratios.values()) <= .05
        future_safe = metrics['field']['combined_future_K']['mean'] <= reference['field']['combined_future_K']['mean']
        entry_safe = all(metrics['field'][key]['mean'] <= reference['field'][key]['mean']
                         for key in ('sma_entry_K', 'htf_entry_K'))
        recas_htf_improved = material_ratios['recAS']['recAS_htf'] <= 0.
        reca_htf_safe = material_ratios['recA']['recAS_htf'] <= .05
        seed_safe = all(a <= b for a, b in zip(
            metrics['per_material']['recAS']['htf_future_K']['values'],
            reference['per_material']['recAS']['htf_future_K']['values']))
        gates[candidate] = {
            'state_safe': bool(state_safe),
            'future_field_safe': bool(future_safe),
            'entry_safe': bool(entry_safe),
            'recAS_htf_improved': bool(recas_htf_improved),
            'recA_htf_safe': bool(reca_htf_safe),
            'three_seed_recAS_non_regression': bool(seed_safe),
            'max_state_regression_fraction': float(max(state_ratios.values())),
            'recAS_htf_change_fraction': float(material_ratios['recAS']['recAS_htf']),
            'passed': bool(candidate != 'global_pareto_control' and state_safe and future_safe and entry_safe and recas_htf_improved and reca_htf_safe and seed_safe),
        }
    passed = [candidate for candidate, value in gates.items() if value['passed']]
    selected = min(passed, key=lambda name: aggregate[name]['per_material']['recAS']['htf_future_K']['mean']) if passed else None
    return {'candidates': gates, 'selected_candidate': selected}


def run(output=DEFAULT, updates=UPDATES, batch_cases=BATCH_CASES, lr=LR, candidate_settings=None, experiment_name='hro_material_conditioned_ablation_20261123'):
    if output.exists():
        raise FileExistsError('Use a new material-conditioned experiment directory')
    arrays, all_ids, hashes = training_data()
    fit_arrays = _case_subset(arrays, all_ids, FIT_CASE_IDS)
    validation_arrays = _case_subset(arrays, all_ids, VALIDATION_CASE_IDS)
    output.mkdir(parents=True)
    candidate_settings = CANDIDATES if candidate_settings is None else candidate_settings
    protocol = {
        'status': 'written_before_fitting',
        'experiment': experiment_name,
        'seeds': list(SEEDS), 'updates_per_seed': updates, 'batch_cases': batch_cases, 'learning_rate': lr,
        'fit_case_ids': list(FIT_CASE_IDS), 'validation_case_ids': list(VALIDATION_CASE_IDS),
        'candidates': candidate_settings, 'data_sha256': hashes, 'fresh_solver_calls': 0,
        'sealed_test_cases_opened': False,
        'source_sha256': {
            'examples/hro_material_conditioned_ablation.py': sha(ROOT / 'examples/hro_material_conditioned_ablation.py'),
            'SciML/models/v13/hierarchical_operator.py': sha(ROOT / 'SciML/models/v13/hierarchical_operator.py'),
            'examples/hro_spatial_ablation.py': sha(ROOT / 'examples/hro_spatial_ablation.py'),
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
                validation = _score(model, validation_arrays, VALIDATION_CASE_IDS)
                write(destination / 'validation.json', validation)
                rows[candidate].append(validation)
                fit_seconds[f'{candidate}/s{seed}'] = fit['training_seconds']
    aggregate = _aggregate(rows, SEEDS)
    gates = _material_gate(aggregate)
    summary = {
        'status': 'completed_hro_material_conditioned_screening', 'fresh_solver_calls': 0,
        'sealed_test_cases_opened': False, 'aggregate': aggregate, 'gates': gates,
        'fit_seconds_by_run': fit_seconds, 'total_fit_seconds': float(sum(fit_seconds.values())),
        'total_seconds': float(time.perf_counter() - started),
        'artifacts': {path.relative_to(output).as_posix(): sha(path) for path in sorted(output.rglob('*')) if path.is_file() and path.name != 'summary.json'},
    }
    write(output / 'summary.json', summary)
    return {'status': summary['status'], 'selected_candidate': gates['selected_candidate'],
            'total_fit_seconds': summary['total_fit_seconds'], 'total_seconds': summary['total_seconds'], 'fresh_solver_calls': 0}


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
                replay = _score(model, validation_arrays, protocol['validation_case_ids'])
                assert replay == saved
                rows[candidate].append(replay)
                training_protocol = json.loads((destination / 'protocol.json').read_text())
                assert training_protocol['training_case_ids'] == protocol['fit_case_ids']
                assert training_protocol['updates'] == protocol['updates_per_seed']
                assert len(json.loads((destination / 'history.json').read_text())) == protocol['updates_per_seed']
    aggregate = _aggregate(rows, protocol['seeds'])
    assert aggregate == summary['aggregate']
    assert _material_gate(aggregate) == summary['gates']
    return {'status': 'verified_hro_material_conditioned_screening', 'checkpoint_replays': len(protocol['candidates']) * len(protocol['seeds']), 'scores_recomputed': True, 'fresh_solver_calls': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    parser.add_argument('--updates', type=int, default=UPDATES)
    parser.add_argument('--batch-cases', type=int, default=BATCH_CASES)
    parser.add_argument('--lr', type=float, default=LR)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(verify(args.output) if args.verify_only else run(args.output, args.updates, args.batch_cases, args.lr), indent=2))
