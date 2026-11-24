"""Three-seed confirmation for the selected SMA-only late-time profile."""
import argparse
import json
import platform
import time
from pathlib import Path

import torch
from threadpoolctl import threadpool_limits

from .hro_internal_validation import FIT_CASE_IDS, VALIDATION_CASE_IDS, _aggregate, _case_subset, _score
from .hro_protected_state_ablation import SEEDS
from .hro_spatial_ablation import _train_one
from .hro_temporal_channel_refined import CANDIDATES
from .hro_temporal_profile_ablation import _temporal_score
from .multicycle_multistep import ROOT, write
from .train_hierarchical import training_data
from SciML.models.v12.multistep_historical import sha
from SciML.models.v13.hierarchical_operator import HierarchicalStateOperator


SELECTED = 'protected_state_sma_late_small'
DEFAULT = ROOT / 'evidence/experiments/hro_temporal_channel_confirmation_20261123'


def run(output=DEFAULT, updates=200, batch_cases=2, lr=.0003):
    if output.exists():
        raise FileExistsError('Use a new temporal-channel confirmation directory')
    arrays, all_ids, hashes = training_data()
    fit_arrays = _case_subset(arrays, all_ids, FIT_CASE_IDS)
    validation_arrays = _case_subset(arrays, all_ids, VALIDATION_CASE_IDS)
    output.mkdir(parents=True)
    protocol = {
        'status': 'written_before_confirmation',
        'experiment': 'hro_temporal_channel_confirmation_20261123',
        'selected_candidate': SELECTED,
        'selection_source': 'evidence/experiments/hro_temporal_channel_refined_20261123/summary.json',
        'seeds': list(SEEDS),
        'updates_per_seed': updates,
        'batch_cases': batch_cases,
        'learning_rate': lr,
        'fit_case_ids': list(FIT_CASE_IDS),
        'validation_case_ids': list(VALIDATION_CASE_IDS),
        'candidate_settings': CANDIDATES[SELECTED],
        'data_sha256': hashes,
        'source_sha256': {
            'examples/confirm_hro_temporal_channel.py': sha(ROOT / 'examples/confirm_hro_temporal_channel.py'),
            'examples/hro_temporal_profile_ablation.py': sha(ROOT / 'examples/hro_temporal_profile_ablation.py'),
            'SciML/models/v13/hierarchical_operator.py': sha(ROOT / 'SciML/models/v13/hierarchical_operator.py'),
        },
        'fresh_solver_calls': 0,
        'sealed_test_cases_opened': False,
        'test_scores_computed': False,
        'environment': {'python': platform.python_version(), 'torch': torch.__version__},
    }
    write(output / 'protocol.json', protocol)
    rows = []
    training_seconds = []
    started = time.perf_counter()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        for seed in SEEDS:
            destination = output / f's{seed}'
            fit = _train_one(destination, fit_arrays, list(FIT_CASE_IDS), hashes, SELECTED, seed,
                             updates, batch_cases, lr, settings_override=CANDIDATES[SELECTED])
            model = HierarchicalStateOperator.load(destination / 'model.pt')
            validation = {'base': _score(model, validation_arrays, VALIDATION_CASE_IDS),
                          'temporal': _temporal_score(model, validation_arrays)}
            write(destination / 'validation.json', validation)
            rows.append(validation)
            training_seconds.append(fit['training_seconds'])
    base_rows = {SELECTED: [row['base'] for row in rows]}
    aggregate = _aggregate(base_rows, SEEDS)[SELECTED]
    aggregate['temporal'] = {}
    for key in rows[0]['temporal']:
        values = [row['temporal'][key] for row in rows]
        aggregate['temporal'][key] = {
            'mean': float(sum(values) / len(values)),
            'sd': float(torch.tensor(values, dtype=torch.float64).std(unbiased=True)),
            'seeds': list(SEEDS),
            'values': values,
        }
    summary = {
        'status': 'completed_hro_temporal_channel_confirmation',
        'selected_candidate': SELECTED,
        'fresh_solver_calls': 0,
        'sealed_test_cases_opened': False,
        'test_scores_computed': False,
        'aggregate': aggregate,
        'training_seconds_by_seed': training_seconds,
        'total_training_seconds': float(sum(training_seconds)),
        'total_seconds': float(time.perf_counter() - started),
        'artifacts': {
            path.relative_to(output).as_posix(): sha(path)
            for path in sorted(output.rglob('*')) if path.is_file() and path.name != 'summary.json'
        },
    }
    write(output / 'summary.json', summary)
    return {'status': summary['status'], 'selected_candidate': SELECTED,
            'total_training_seconds': summary['total_training_seconds'],
            'total_seconds': summary['total_seconds'], 'fresh_solver_calls': 0}


def verify(output=DEFAULT):
    protocol = json.loads((output / 'protocol.json').read_text())
    summary = json.loads((output / 'summary.json').read_text())
    for relative, expected in summary['artifacts'].items():
        if sha(output / relative) != expected:
            raise ValueError(f'Artifact hash mismatch: {relative}')
    arrays, all_ids, hashes = training_data()
    if protocol['data_sha256'] != hashes:
        raise ValueError('Training snapshot checksum changed')
    validation_arrays = _case_subset(arrays, all_ids, VALIDATION_CASE_IDS)
    rows = []
    with threadpool_limits(limits=1):
        for seed in protocol['seeds']:
            destination = output / f's{seed}'
            model = HierarchicalStateOperator.load(destination / 'model.pt')
            saved = json.loads((destination / 'validation.json').read_text())
            replay = {'base': _score(model, validation_arrays, VALIDATION_CASE_IDS),
                      'temporal': _temporal_score(model, validation_arrays)}
            assert replay == saved
            rows.append(replay)
            training_protocol = json.loads((destination / 'protocol.json').read_text())
            assert training_protocol['training_case_ids'] == protocol['fit_case_ids']
            assert training_protocol['updates'] == protocol['updates_per_seed']
            assert len(json.loads((destination / 'history.json').read_text())) == protocol['updates_per_seed']
    base_rows = {SELECTED: [row['base'] for row in rows]}
    aggregate = _aggregate(base_rows, protocol['seeds'])[SELECTED]
    aggregate['temporal'] = {}
    for key in rows[0]['temporal']:
        values = [row['temporal'][key] for row in rows]
        aggregate['temporal'][key] = {
            'mean': float(sum(values) / len(values)),
            'sd': float(torch.tensor(values, dtype=torch.float64).std(unbiased=True)),
            'seeds': list(protocol['seeds']),
            'values': values,
        }
    assert aggregate == summary['aggregate']
    return {'status': 'verified_hro_temporal_channel_confirmation',
            'checkpoint_replays': len(protocol['seeds']),
            'scores_recomputed': True,
            'max_replay_absolute_difference': 0.0,
            'fresh_solver_calls': 0,
            'sealed_test_cases_opened': False,
            'test_scores_computed': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    parser.add_argument('--updates', type=int, default=200)
    parser.add_argument('--batch-cases', type=int, default=2)
    parser.add_argument('--lr', type=float, default=.0003)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    result = verify(args.output) if args.verify_only else run(args.output, args.updates, args.batch_cases, args.lr)
    print(json.dumps(result, indent=2))
