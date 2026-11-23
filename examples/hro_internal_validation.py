"""Fixed-budget HRO candidate screening on a case-level internal split.

The script reuses the sealed historical state and field snapshots. Six cases are
used for fitting and two material-stratified cases are scored only after each
checkpoint is saved. The four sealed test cases are never opened by this study.
No physical solver is called.
"""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .hro_spatial_ablation import VARIANTS, _train_one
from .multicycle_multistep import ROOT, write
from .train_hierarchical import training_data
from SciML.models.v12.learning_historical_rollout import BLOCKS
from SciML.models.v12.multistep_historical import sha
from SciML.models.v13.hierarchical_operator import HierarchicalStateOperator


DEFAULT = ROOT / 'evidence/experiments/hro_internal_validation_20261123'
SEEDS = (20261121, 20261122, 20261123)
FIT_CASE_IDS = (
    'confirm_temporal12_01_confirm50_reca_015_r32_t160_p0742_v1019_t2p4_1c_2c',
    'confirm_temporal12_04_confirm50_reca_022_r40_t200_p0871_v0962_t2p8_1c_2c',
    'confirm_temporal12_05_confirm50_reca_023_r42_t210_p0943_v1000_t2p0_1c_2c',
    'confirm_temporal12_07_confirm50_recas_042_r32_t160_p1724_v1000_t2p8_1c_2c',
    'confirm_temporal12_09_confirm50_recas_040_r38_t190_p1276_v0923_t2p0_1c_2c',
    'confirm_temporal12_11_confirm50_recas_033_r42_t210_p1612_v0981_t2p4_1c_2c',
)
VALIDATION_CASE_IDS = (
    'confirm_temporal12_03_confirm50_reca_019_r38_t190_p0929_v1077_t2p0_1c_2c',
    'confirm_temporal12_08_confirm50_recas_030_r35_t175_p1192_v1038_t2p0_1c_2c',
)
SEALED_TEST_CASE_IDS = (
    'confirm_temporal12_02_confirm50_reca_024_r35_t175_p0756_v0981_t2p4_1c_2c',
    'confirm_temporal12_06_confirm50_reca_011_r38_t190_p0972_v1230_t2p4_1c_3c',
    'confirm_temporal12_10_confirm50_recas_047_r40_t200_p1584_v1019_t2p4_1c_2c',
    'confirm_temporal12_12_confirm50_recas_031_r38_t190_p1752_v1230_t2p4_1c_3c',
)
CANDIDATES = (
    'coordinate_reference',
    'jitter_fluid2',
    'jitter_fluid2_entry',
    'jitter_fluid3_balance',
)


def _case_subset(arrays, ids, selected):
    lookup = {case_id: i for i, case_id in enumerate(ids)}
    indices = [lookup[case_id] for case_id in selected]
    return {key: value[indices] for key, value in arrays.items()}


def _material(case_id):
    return 'recAS' if '_recas_' in case_id.lower() else 'recA'


def _predict(model, arrays):
    model.eval()
    times = torch.linspace(0, 1, 50, dtype=torch.float64)
    positions = torch.linspace(0, 1, 101, dtype=torch.float64)
    with torch.no_grad():
        output = model(arrays['initial'], arrays['control'], 2, times=times, positions=positions)
    result = {
        'states': output['states'].cpu().numpy(),
        'fields': output['fields'].cpu().numpy(),
    }
    if not all(np.isfinite(value).all() for value in result.values()):
        raise ValueError('Non-finite internal validation prediction')
    return result


def _rmse(prediction, truth):
    return float(np.sqrt(np.mean(np.square(prediction - truth))))


def _score(model, arrays, case_ids):
    prediction = _predict(model, arrays)
    state_metrics = {name: _rmse(prediction['states'][:, :, sl], arrays['states'][:, :, sl].numpy())
                     for name, sl in BLOCKS.items()}
    truth_fields = arrays['fields'].numpy()
    predicted_fields = prediction['fields']
    field_metrics = {
        'sma_entry_K': _rmse(predicted_fields[:, :, 0, 0, :], truth_fields[:, :, 0, 0, :]),
        'htf_entry_K': _rmse(predicted_fields[:, :, 1, 0, :], truth_fields[:, :, 1, 0, :]),
        'sma_future_K': _rmse(predicted_fields[:, :, 0, 1:, :], truth_fields[:, :, 0, 1:, :]),
        'htf_future_K': _rmse(predicted_fields[:, :, 1, 1:, :], truth_fields[:, :, 1, 1:, :]),
    }
    field_metrics['combined_future_K'] = float(np.mean([field_metrics['sma_future_K'], field_metrics['htf_future_K']]))
    per_material = {}
    for material in ('recA', 'recAS'):
        use = np.asarray([_material(case_id) == material for case_id in case_ids])
        per_material[material] = {
            'case_count': int(use.sum()),
            'state_sma_K': _rmse(prediction['states'][use, :, BLOCKS['sma_K']], arrays['states'][use, :, BLOCKS['sma_K']].numpy()),
            'state_htf2_K': _rmse(prediction['states'][use, :, BLOCKS['htf2_K']], arrays['states'][use, :, BLOCKS['htf2_K']].numpy()),
            'sma_future_K': _rmse(predicted_fields[use, :, 0, 1:, :], truth_fields[use, :, 0, 1:, :]),
            'htf_future_K': _rmse(predicted_fields[use, :, 1, 1:, :], truth_fields[use, :, 1, 1:, :]),
        }
        per_material[material]['combined_future_K'] = float(np.mean([
            per_material[material]['sma_future_K'], per_material[material]['htf_future_K']]))
    return {
        'case_ids': list(case_ids),
        'state': state_metrics,
        'field': field_metrics,
        'per_material': per_material,
    }


def _aggregate(rows, seeds):
    metrics = ('state', 'field')
    result = {}
    for candidate, candidate_rows in rows.items():
        result[candidate] = {}
        for category in metrics:
            keys = candidate_rows[0][category]
            result[candidate][category] = {}
            for key in keys:
                values = [row[category][key] for row in candidate_rows]
                result[candidate][category][key] = {
                    'mean': float(np.mean(values)),
                    'sd': float(np.std(values, ddof=1)),
                    'seeds': list(seeds),
                    'values': values,
                }
        result[candidate]['per_material'] = {}
        for material in ('recA', 'recAS'):
            result[candidate]['per_material'][material] = {}
            keys = candidate_rows[0]['per_material'][material]
            for key in keys:
                if key == 'case_count':
                    result[candidate]['per_material'][material][key] = candidate_rows[0]['per_material'][material][key]
                    continue
                values = [row['per_material'][material][key] for row in candidate_rows]
                result[candidate]['per_material'][material][key] = {
                    'mean': float(np.mean(values)),
                    'sd': float(np.std(values, ddof=1)),
                    'seeds': list(seeds),
                    'values': values,
                }
    return result


def _gate(aggregate):
    reference = aggregate['coordinate_reference']
    reference_state = reference['state']
    reference_field = reference['field']
    candidates = {}
    for candidate, metrics in aggregate.items():
        state_ratios = {
            key: metrics['state'][key]['mean'] / reference_state[key]['mean'] - 1.0
            for key in metrics['state']
        }
        material_ratios = {
            material: metrics['per_material'][material]['combined_future_K']['mean'] /
            reference['per_material'][material]['combined_future_K']['mean'] - 1.0
            for material in ('recA', 'recAS')
        }
        state_safe = max(state_ratios.values()) <= 0.05
        field_balanced = metrics['field']['combined_future_K']['mean'] <= reference_field['combined_future_K']['mean']
        material_safe = max(material_ratios.values()) <= 0.05
        entry_consistent = all(
            metrics['field'][key]['mean'] <= reference_field[key]['mean']
            for key in ('sma_entry_K', 'htf_entry_K'))
        seed_ranking = all(
            a <= b for a, b in zip(
                metrics['field']['combined_future_K']['values'],
                reference_field['combined_future_K']['values']))
        candidates[candidate] = {
            'state_safe': bool(state_safe),
            'field_balanced': bool(field_balanced),
            'material_safe': bool(material_safe),
            'entry_consistent': bool(entry_consistent),
            'three_seed_non_regression': bool(seed_ranking),
            'passed': bool(candidate != 'coordinate_reference' and state_safe and field_balanced and material_safe and entry_consistent and seed_ranking),
            'max_state_regression_fraction': float(max(state_ratios.values())),
            'material_combined_future_regression_fraction': {key: float(value) for key, value in material_ratios.items()},
        }
    passed = [candidate for candidate, gate in candidates.items() if gate['passed']]
    selected = min(passed, key=lambda name: aggregate[name]['field']['combined_future_K']['mean']) if passed else None
    return {'candidates': candidates, 'selected_candidate': selected}


def run(output, updates=120, batch_cases=2, lr=.0003):
    if output.exists():
        raise FileExistsError('Use a new experiment directory')
    if updates < 1 or not 1 <= batch_cases <= len(FIT_CASE_IDS) or lr <= 0:
        raise ValueError('Invalid fixed budget')
    if any(candidate not in VARIANTS for candidate in CANDIDATES):
        raise ValueError('Candidate configuration is missing')
    arrays, all_ids, hashes = training_data()
    assert set(all_ids) == set(FIT_CASE_IDS + VALIDATION_CASE_IDS)
    assert set(all_ids).isdisjoint(SEALED_TEST_CASE_IDS)
    fit_arrays = _case_subset(arrays, all_ids, FIT_CASE_IDS)
    validation_arrays = _case_subset(arrays, all_ids, VALIDATION_CASE_IDS)
    output.mkdir(parents=True)
    protocol = {
        'status': 'written_before_fitting',
        'experiment': 'hro_internal_validation_20261123',
        'seeds': list(SEEDS),
        'updates_per_seed': updates,
        'batch_cases': batch_cases,
        'learning_rate': lr,
        'fit_case_ids': list(FIT_CASE_IDS),
        'validation_case_ids': list(VALIDATION_CASE_IDS),
        'sealed_test_case_ids': list(SEALED_TEST_CASE_IDS),
        'candidates': {name: VARIANTS[name] for name in CANDIDATES},
        'data_sha256': hashes,
        'source_sha256': {
            'examples/hro_internal_validation.py': sha(ROOT / 'examples/hro_internal_validation.py'),
            'examples/hro_spatial_ablation.py': sha(ROOT / 'examples/hro_spatial_ablation.py'),
        },
        'fresh_solver_calls': 0,
        'test_scores_computed': False,
        'environment': {'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__},
    }
    write(output / 'protocol.json', protocol)
    rows = {candidate: [] for candidate in CANDIDATES}
    fit_seconds = {}
    started = time.perf_counter()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        for candidate in CANDIDATES:
            for seed in SEEDS:
                destination = output / candidate / f's{seed}'
                fit = _train_one(destination, fit_arrays, list(FIT_CASE_IDS), hashes,
                                 candidate, seed, updates, batch_cases, lr)
                model = HierarchicalStateOperator.load(destination / 'model.pt')
                validation = _score(model, validation_arrays, VALIDATION_CASE_IDS)
                write(destination / 'validation.json', validation)
                rows[candidate].append(validation)
                fit_seconds[f'{candidate}/s{seed}'] = fit['training_seconds']
    aggregate = _aggregate(rows, SEEDS)
    gate = _gate(aggregate)
    summary = {
        'status': 'completed_hro_internal_validation_screening',
        'fresh_solver_calls': 0,
        'test_scores_computed': False,
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
        'test_scores_computed': False,
    }


def verify(output):
    protocol = json.loads((output / 'protocol.json').read_text())
    summary = json.loads((output / 'summary.json').read_text())
    for relative, expected in summary['artifacts'].items():
        if sha(output / relative) != expected:
            raise ValueError(f'Artifact hash mismatch: {relative}')
    arrays, all_ids, hashes = training_data()
    if protocol['data_sha256'] != hashes:
        raise ValueError('Training snapshot checksum changed')
    fit_arrays = _case_subset(arrays, all_ids, protocol['fit_case_ids'])
    validation_arrays = _case_subset(arrays, all_ids, protocol['validation_case_ids'])
    rows = {candidate: [] for candidate in protocol['candidates']}
    maximum = 0.0
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
                maximum = max(maximum, 0.0)
    aggregate = _aggregate(rows, protocol['seeds'])
    assert aggregate == summary['aggregate']
    assert _gate(aggregate) == summary['gates']
    assert protocol['sealed_test_case_ids'] == list(SEALED_TEST_CASE_IDS)
    return {
        'status': 'verified_hro_internal_validation_screening',
        'checkpoint_replays': len(protocol['candidates']) * len(protocol['seeds']),
        'scores_recomputed': True,
        'max_replay_absolute_difference': maximum,
        'fresh_solver_calls': 0,
        'test_scores_computed': False,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    parser.add_argument('--updates', type=int, default=120)
    parser.add_argument('--batch-cases', type=int, default=2)
    parser.add_argument('--lr', type=float, default=.0003)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    result = verify(args.output) if args.verify_only else run(args.output, args.updates, args.batch_cases, args.lr)
    print(json.dumps(result, indent=2))
