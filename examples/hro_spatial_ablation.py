"""Fixed-budget spatial-supervision ablation for the hierarchical operator.

The study reuses the historical 12-case state and field snapshots. It changes
only the field-query density and the optional supervised spatial-curvature term.
No solver is called and no test result is used to update a checkpoint.
"""
import argparse
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .formal_hierarchical import cohorts, infer
from .multicycle_multistep import ROOT, write
from .train_hierarchical import training_data
from SciML.models.v12.learning_historical_rollout import HistoricalModel
from SciML.models.v12.learning_coordinate_operator import interpolate
from SciML.models.v12.multistep_historical import sha
from SciML.models.v13.hierarchical_operator import (
    HierarchicalStateOperator,
    OperatorConfig,
    joint_loss,
)


DEFAULT = ROOT / 'evidence/experiments/hro_spatial_ablation_20261123'
FORMAL = ROOT / 'evidence/experiments/v13_formal_v1'
DEFAULT_SEEDS = (20261121, 20261122, 20261123)
VARIANTS = {
    'coordinate_reference': {'spatial_points': 26, 'gradient_weight': 0.01, 'curvature_weight': 0.0, 'curvature_channel_weights': None, 'field_channel_weights': [1.0, 1.0], 'curvature_start_fraction': 1.0},
    'dense_field': {'spatial_points': 51, 'gradient_weight': 0.01, 'curvature_weight': 0.0, 'curvature_channel_weights': None, 'field_channel_weights': None, 'curvature_start_fraction': 1.0},
    'dense_jitter': {'spatial_points': 51, 'gradient_weight': 0.01, 'curvature_weight': 0.0, 'curvature_channel_weights': None, 'field_channel_weights': None, 'curvature_start_fraction': 1.0, 'spatial_jitter_fraction': 0.5},
    'jitter_fluid4': {'spatial_points': 51, 'gradient_weight': 0.01, 'curvature_weight': 0.0, 'curvature_channel_weights': None, 'field_channel_weights': [1.0, 4.0], 'curvature_start_fraction': 1.0, 'spatial_jitter_fraction': 0.5},
    'jitter_fluid2': {'spatial_points': 51, 'gradient_weight': 0.01, 'curvature_weight': 0.0, 'curvature_channel_weights': None, 'field_channel_weights': [1.0, 2.0], 'curvature_start_fraction': 1.0, 'spatial_jitter_fraction': 0.5},
    'jitter_fluid2_entry': {'spatial_points': 51, 'gradient_weight': 0.01, 'curvature_weight': 0.0, 'curvature_channel_weights': None, 'field_channel_weights': [1.0, 2.0], 'curvature_start_fraction': 1.0, 'spatial_jitter_fraction': 0.5, 'entry_time_weight': 1.0, 'entry_field_weight': 1.0},
    'jitter_fluid3_balance': {'spatial_points': 51, 'gradient_weight': 0.01, 'curvature_weight': 0.0, 'curvature_channel_weights': None, 'field_channel_weights': [1.0, 3.0], 'curvature_start_fraction': 1.0, 'spatial_jitter_fraction': 0.5},
    'dense_curvature': {'spatial_points': 51, 'gradient_weight': 0.01, 'curvature_weight': 0.001, 'curvature_channel_weights': None, 'field_channel_weights': None, 'curvature_start_fraction': 0.0},
    'dense_curvature_light': {'spatial_points': 51, 'gradient_weight': 0.01, 'curvature_weight': 1e-6, 'curvature_channel_weights': None, 'field_channel_weights': None, 'curvature_start_fraction': 0.0},
    'dense_curvature_mid': {'spatial_points': 51, 'gradient_weight': 0.01, 'curvature_weight': 1e-5, 'curvature_channel_weights': None, 'field_channel_weights': None, 'curvature_start_fraction': 0.0},
    'dense_curvature_balanced': {'spatial_points': 51, 'gradient_weight': 0.01, 'curvature_weight': 0.01, 'curvature_channel_weights': None, 'field_channel_weights': None, 'curvature_start_fraction': 0.0},
    'dense_curvature_strong': {'spatial_points': 51, 'gradient_weight': 0.01, 'curvature_weight': 0.05, 'curvature_channel_weights': None, 'field_channel_weights': None, 'curvature_start_fraction': 0.0},
    'solid_curvature_strong': {'spatial_points': 51, 'gradient_weight': 0.01, 'curvature_weight': 0.05, 'curvature_channel_weights': [1.0, 0.0], 'field_channel_weights': None, 'curvature_start_fraction': 0.0},
    'solid_curvature_fluid_weighted': {'spatial_points': 51, 'gradient_weight': 0.01, 'curvature_weight': 0.05, 'curvature_channel_weights': [1.0, 0.0], 'field_channel_weights': [1.0, 2.0], 'curvature_start_fraction': 0.0},
    'solid_curvature_fluid4': {'spatial_points': 51, 'gradient_weight': 0.01, 'curvature_weight': 0.05, 'curvature_channel_weights': [1.0, 0.0], 'field_channel_weights': [1.0, 4.0], 'curvature_start_fraction': 0.0},
    'staged_solid_curvature': {'spatial_points': 51, 'gradient_weight': 0.01, 'curvature_weight': 0.05, 'curvature_channel_weights': [1.0, 0.0], 'field_channel_weights': [1.0, 2.0], 'curvature_start_fraction': 0.5},
}


def _aggregate(rows, seeds):
    result = {}
    for variant, variant_rows in rows.items():
        result[variant] = {}
        for cohort in ('all_test', 'fixed_three_cycle'):
            result[variant][cohort] = {}
            for category, groups in variant_rows[0][cohort].items():
                result[variant][cohort][category] = {}
                for key, metrics in groups.items():
                    result[variant][cohort][category][key] = {'case_count': metrics['case_count']}
                    for name in metrics:
                        if name == 'case_count':
                            continue
                        values = [row[cohort][category][key][name] for row in variant_rows]
                        result[variant][cohort][category][key][name] = {
                            'mean': float(np.mean(values)),
                            'sd': float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                            'seeds': list(seeds),
                            'values': values,
                        }
    return result


def _coordinates(spatial_points, time_indices):
    if spatial_points < 3:
        raise ValueError('At least three spatial points are required')
    indices = np.linspace(0, 100, spatial_points, dtype=int)
    if len(np.unique(indices)) != spatial_points:
        raise ValueError('Spatial query indices must be unique')
    return np.asarray(time_indices, dtype=int), indices


def _query_positions(spatial_indices, jitter_fraction, rng):
    positions = np.asarray(spatial_indices, dtype=float) / 100.
    if jitter_fraction:
        jitter = rng.uniform(-.5, .5, size=len(positions))
        jitter *= jitter_fraction / 100.
        jitter[[0, -1]] = 0.
        positions = positions + jitter
    if not np.all(np.diff(positions) > 0) or positions[0] != 0. or positions[-1] != 1.:
        raise ValueError('Query positions must be strictly increasing with exact endpoints')
    return torch.tensor(positions, dtype=torch.float64)


def _train_one(output, arrays, ids, hashes, variant, seed, updates, batch_cases, lr, settings_override=None):
    settings = VARIANTS[variant] if settings_override is None else settings_override
    output.mkdir(parents=True)
    protocol = {
        'variant': variant,
        'seed': seed,
        'updates': updates,
        'batch_cases': batch_cases,
        'optimizer': 'AdamW',
        'lr': lr,
        'weight_decay': 1e-4,
        'gradient_clip': 1.0,
        'schedule': 'linear warmup then cosine decay to 10% of initial lr',
        'query_sampling': {
            'time': 'time zero + endpoint + six random interior indices',
            'spatial_points': settings['spatial_points'],
            'spatial_indices': np.linspace(0, 100, settings['spatial_points'], dtype=int).tolist(),
            'spatial_jitter_fraction_of_native_spacing': settings.get('spatial_jitter_fraction', 0.0),
        },
        'loss': {
            'state_blocks': 'eight equally weighted normalized blocks',
            'field_weight': 1.0,
            'spatial_gradient_weight': settings['gradient_weight'],
            'spatial_curvature_weight': settings['curvature_weight'],
            'spatial_curvature_channel_weights': settings['curvature_channel_weights'],
            'field_channel_weights': settings['field_channel_weights'],
            'curvature_start_fraction': settings['curvature_start_fraction'],
            'entry_time_weight': settings.get('entry_time_weight', 0.0),
            'entry_field_weight': settings.get('entry_field_weight', 0.0),
            'state_block_weights': settings.get('state_block_weights'),
            'state_stage_weights': settings.get('state_stage_weights'),
            'material_field_channel_weights': settings.get('material_field_channel_weights'),
        },
        'training_case_ids': ids,
        'training_cycles': [1, 2],
        'supervision': 'existing training-case states and fields only',
        'fresh_solver_calls': 0,
        'source_sha256': {
            'SciML/models/v13/hierarchical_operator.py': sha(ROOT / 'SciML/models/v13/hierarchical_operator.py'),
            'examples/hro_spatial_ablation.py': sha(ROOT / 'examples/hro_spatial_ablation.py'),
        },
        'data_sha256': hashes,
        'baseline_summary_sha256': sha(FORMAL / 'summary.json'),
    }
    write(output / 'protocol.json', protocol)

    with torch.random.fork_rng(devices=[]), threadpool_limits(limits=1):
        torch.manual_seed(seed)
        base = HistoricalModel.load(ROOT / 'checkpoints/historical-rollout-r1/s20260910/model.pt')
        model = HierarchicalStateOperator(base, OperatorConfig())
        field_delta = arrays['fields'] - arrays['fields'][:, :, :, 0:1]
        model.decoder.field_scale.copy_(field_delta.square().mean(dim=(0, 1, 3, 4)).sqrt().clamp_min(.1))
        frozen = {name: value.clone() for name, value in model.named_buffers()}
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        warmup = min(20, max(1, updates // 10))

        def factor(step):
            if step < warmup:
                return (step + 1) / warmup
            progress = (step - warmup) / max(1, updates - warmup)
            return .1 + .9 * .5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
        rng = np.random.default_rng(seed)
        source_grid = torch.linspace(0, 1, 101, dtype=torch.float64)
        history = []
        started = time.perf_counter()
        model.train()
        for step in range(updates):
            chosen = np.sort(rng.choice(len(ids), batch_cases, replace=False))
            time_indices = np.r_[0, np.sort(rng.choice(np.arange(1, 49), 6, replace=False)), 49]
            time_indices, spatial_indices = _coordinates(settings['spatial_points'], time_indices)
            times = torch.tensor(time_indices / 49.)
            positions = _query_positions(spatial_indices, settings.get('spatial_jitter_fraction', 0.0), rng)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(arrays['initial'][chosen], arrays['control'][chosen], 2,
                               times=times, positions=positions)
            fields_at_times = arrays['fields'][chosen][:, :, :, time_indices]
            fields = (interpolate(fields_at_times, source_grid, positions)
                      if settings.get('spatial_jitter_fraction', 0.0)
                      else fields_at_times[:, :, :, :, spatial_indices])
            sample_field_weights = None
            if settings.get('material_field_channel_weights') is not None:
                sample_field_weights = torch.tensor([
                    settings['material_field_channel_weights']['recAS' if '_recas_' in ids[i].lower() else 'recA']
                    for i in chosen
                ], dtype=torch.float64)
            curvature_weight = (settings['curvature_weight'] if
                                (step + 1) > int(updates * settings['curvature_start_fraction']) else 0.0)
            losses = joint_loss(
                prediction,
                arrays['states'][chosen],
                model.feature_scale,
                field_truth=fields,
                field_scale=model.decoder.field_scale,
                positions=positions,
                gradient_weight=settings['gradient_weight'],
                curvature_weight=curvature_weight,
                curvature_channel_weights=settings['curvature_channel_weights'],
                field_channel_weights=settings['field_channel_weights'],
                times=times,
                entry_time_weight=settings.get('entry_time_weight', 0.0),
                entry_field_weight=settings.get('entry_field_weight', 0.0),
                state_block_weights=settings.get('state_block_weights'),
                state_stage_weights=settings.get('state_stage_weights'),
                field_sample_channel_weights=sample_field_weights,
            )
            losses['total'].backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            history.append({
                'update': step + 1,
                'train_case_ids': [ids[i] for i in chosen],
                'time_indices': time_indices.tolist(),
                'position_indices': spatial_indices.tolist(),
                'lr': float(optimizer.param_groups[0]['lr']),
                'gradient_norm_before_clip': float(norm),
                'curvature_weight_active': curvature_weight,
                **{key: float(value.detach()) for key, value in losses.items()},
            })
            optimizer.step()
            scheduler.step()

        training_seconds = time.perf_counter() - started
        for name, value in model.named_buffers():
            if not torch.equal(value, frozen[name]):
                raise ValueError(f'Frozen buffer changed: {name}')

        model.eval()
        times = torch.linspace(0, 1, 10, dtype=torch.float64)
        positions = torch.linspace(0, 1, 51, dtype=torch.float64)
        with torch.no_grad():
            execution = model(arrays['initial'][:1], arrays['control'][:1], 3,
                              times=times, positions=positions)
        model.save(output / 'model.pt', metadata={
            'status': 'trained_spatial_supervision_ablation',
            'variant': variant,
            'updates': updates,
            'seed': seed,
            'data_sha256': hashes,
        })
        loaded = HierarchicalStateOperator.load(output / 'model.pt')
        with torch.no_grad():
            replay = loaded(arrays['initial'][:1], arrays['control'][:1], 3,
                            times=times, positions=positions)
        for key in ('states', 'fields', 'final_memory'):
            torch.testing.assert_close(execution[key], replay[key], rtol=0, atol=0)
        with (output / 'training_state.pt').open('xb') as handle:
            torch.save({
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'torch_rng': torch.get_rng_state(),
                'numpy_rng_json': json.dumps(rng.bit_generator.state),
                'completed_updates': updates,
            }, handle)
        np.savez_compressed(
            output / 'execution_arrays.npz',
            states=execution['states'].numpy(),
            fields=execution['fields'].numpy(),
            initial=arrays['initial'][:1].numpy(),
            control=arrays['control'][:1].numpy(),
            times=times.numpy(),
            positions=positions.numpy(),
        )
        write(output / 'history.json', history)
        return {
            'training_seconds': training_seconds,
            'parameter_report': model.parameter_report(),
            'checkpoint_replay_exact': True,
            'frozen_prior_and_scalers_unchanged': True,
            'history': history,
        }


def run(output, variants, seeds, updates=60, batch_cases=2, lr=.0003):
    if output.exists():
        raise FileExistsError('Use a new experiment directory')
    if not variants or any(v not in VARIANTS for v in variants):
        raise ValueError(f'Unknown variants: {variants}')
    if not seeds or updates < 1 or not 1 <= batch_cases <= 8 or lr <= 0:
        raise ValueError('Invalid fixed budget')
    torch.set_num_threads(1)
    arrays, ids, hashes = training_data()
    output.mkdir(parents=True)
    protocol = {
        'experiment': 'hro_spatial_ablation_20261123',
        'written_before_training': True,
        'variants': {name: VARIANTS[name] for name in variants},
        'seeds': list(seeds),
        'updates_per_fit': updates,
        'batch_cases': batch_cases,
        'fresh_solver_calls': 0,
        'data_sha256': hashes,
        'baseline_summary_sha256': sha(FORMAL / 'summary.json'),
        'environment': {'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__},
    }
    write(output / 'protocol.json', protocol)
    started = time.perf_counter()
    rows = {variant: [] for variant in variants}
    training_seconds = {}
    d = __import__('examples.multicycle_multistep', fromlist=['data']).data()[0]
    with threadpool_limits(limits=1):
        for variant in variants:
            for seed in seeds:
                destination = output / variant / f's{seed}'
                fit = _train_one(destination, arrays, ids, hashes, variant, seed, updates, batch_cases, lr)
                model = HierarchicalStateOperator.load(destination / 'model.pt')
                prediction = infer(model, d)
                np.savez_compressed(destination / 'test_predictions.npz', **prediction)
                report = {'variant': variant, 'seed': seed, **cohorts(prediction, d)}
                write(destination / 'evaluation.json', report)
                rows[variant].append(report)
                training_seconds[f'{variant}/s{seed}'] = fit['training_seconds']
    baseline = json.loads((FORMAL / 'summary.json').read_text())['new']['fixed_three_cycle']
    summary = {
        'status': 'completed_hro_spatial_ablation',
        'fresh_solver_calls': 0,
        'variants': _aggregate(rows, seeds),
        'baseline_formal_fixed_three_cycle': baseline,
        'training_seconds_by_fit': training_seconds,
        'total_training_seconds': float(sum(training_seconds.values())),
        'total_seconds': float(time.perf_counter() - started),
        'selection': 'Exploratory screening only. No test-based checkpoint was promoted.',
        'artifacts': {
            path.relative_to(output).as_posix(): sha(path)
            for path in sorted(output.rglob('*')) if path.is_file() and path.name != 'summary.json'
        },
    }
    write(output / 'summary.json', summary)
    return {
        'status': summary['status'],
        'fresh_solver_calls': 0,
        'training_seconds': summary['total_training_seconds'],
        'total_seconds': summary['total_seconds'],
        'c3s4': {
            variant: summary['variants'][variant]['fixed_three_cycle']['state']['c3s4']
            for variant in variants
        },
    }


def verify(output):
    protocol = json.loads((output / 'protocol.json').read_text())
    summary = json.loads((output / 'summary.json').read_text())
    for relative, expected in summary['artifacts'].items():
        if sha(output / relative) != expected:
            raise ValueError(f'Artifact hash mismatch: {relative}')
    arrays, ids, hashes = training_data()
    if protocol['data_sha256'] != hashes:
        raise ValueError('Training snapshot checksum changed')
    d = __import__('examples.multicycle_multistep', fromlist=['data']).data()[0]
    rows = {variant: [] for variant in protocol['variants']}
    maximum = 0.0
    with threadpool_limits(limits=1):
        for variant in protocol['variants']:
            for seed in protocol['seeds']:
                destination = output / variant / f's{seed}'
                report = json.loads((destination / 'evaluation.json').read_text())
                rows[variant].append(report)
                model = HierarchicalStateOperator.load(destination / 'model.pt')
                replay = infer(model, d)
                with np.load(destination / 'test_predictions.npz', allow_pickle=False) as saved:
                    for key in replay:
                        maximum = max(maximum, float(np.abs(saved[key] - replay[key]).max()))
                        np.testing.assert_allclose(saved[key], replay[key], rtol=0, atol=1e-8)
                if cohorts(replay, d) != {key: report[key] for key in ('all_test', 'fixed_three_cycle', 'per_three_cycle_case')}:
                    raise ValueError(f'Score mismatch: {variant}/s{seed}')
    if _aggregate(rows, protocol['seeds']) != summary['variants']:
        raise ValueError('Aggregate mismatch')
    return {
        'status': 'verified_hro_spatial_ablation',
        'checkpoint_replays': sum(len(protocol['seeds']) for _ in protocol['variants']),
        'scores_recomputed': True,
        'max_replay_absolute_difference': maximum,
        'fresh_solver_calls': 0,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    parser.add_argument('--variants', default=','.join(VARIANTS))
    parser.add_argument('--seeds', default=','.join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument('--updates', type=int, default=60)
    parser.add_argument('--batch-cases', type=int, default=2)
    parser.add_argument('--lr', type=float, default=.0003)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if args.verify_only:
        result = verify(args.output)
    else:
        variants = tuple(value.strip() for value in args.variants.split(',') if value.strip())
        seeds = tuple(int(value.strip()) for value in args.seeds.split(',') if value.strip())
        result = run(args.output, variants, seeds, args.updates, args.batch_cases, args.lr)
    print(json.dumps(result, indent=2))
