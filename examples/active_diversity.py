"""Fixed-budget diversity extension on existing labels; no fresh physics simulations."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from SciML.active_learning.learning_pool_loop import (
    ArchivedLabelOracle, acquisition_components, choose_batch, initial_design,
)
from SciML.models.v12.learning_ensemble import LearningEnsemble

ROOT = Path(__file__).resolve().parents[1]
STRATEGIES = ('random', 'uncertainty', 'composite', 'composite_diversity')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2), encoding='utf-8', newline='\n')


def diverse_batch(ids, candidate_z, known_z, composite, count):
    """Greedy equal weighting of composite score and distance rank; inputs only.

    All coordinates use the model's scaler fitted on revealed cases. Distance
    is to the union of revealed points and the batch selected so far. No target
    values, test metrics or calibration examples enter this function.
    """
    z, known, scores = map(lambda a: np.asarray(a, float), (candidate_z, known_z, composite))
    if (z.ndim != 2 or known.ndim != 2 or z.shape[1] != known.shape[1]
            or len(known) == 0 or len(ids) != len(z) or len(set(ids)) != len(ids)
            or scores.shape != (len(ids),) or not 0 < count <= len(ids)
            or not all(np.isfinite(v).all() for v in (z, known, scores))):
        raise ValueError('Invalid finite input-only acquisition data')
    remaining = list(range(len(ids)))
    selected, trace = [], []
    anchors = known.copy()
    for _ in range(count):
        distance = np.sqrt(((z[remaining, None, :] - anchors[None, :, :]) ** 2).sum(2)).min(1)
        distance_rank = (rankdata(distance, method='average') - .5) / len(remaining)
        combined = .5 * scores[remaining] + .5 * distance_rank
        order = np.lexsort((np.asarray(ids)[remaining], -combined))
        local = int(order[0]); chosen = remaining[local]
        trace.append({'candidate_ids': [ids[i] for i in remaining],
                      'min_distance': distance.tolist(), 'distance_rank': distance_rank.tolist(),
                      'combined_score': combined.tolist(), 'selected_id': ids[chosen]})
        selected.append(ids[chosen]); anchors = np.vstack((anchors, z[chosen]))
        remaining.remove(chosen)
    return selected, trace


def experiment(x, ids, labels, test_x, test_y, initial, strategy, seed, out):
    oracle = ArchivedLabelOracle(ids, labels)
    lookup = {case: i for i, case in enumerate(ids)}
    training_ids = list(initial)
    training_y = oracle.reveal(training_ids, 'initial input-only design')
    history, predictions = [], []
    for round_index in range(3):
        training_x = x[[lookup[case] for case in training_ids]]
        model = LearningEnsemble(members=3, epochs=200, seed=10000 + seed * 10 + round_index)
        model.fit(training_x, training_y, training_ids)
        record = {'round': round_index, 'revealed_count': len(training_ids),
                  'training_ids': list(training_ids)}
        if round_index < 2:
            candidates = sorted(set(ids) - set(training_ids))
            candidate_x = x[[lookup[case] for case in candidates]]
            mean, std = model.predict(candidate_x)
            scores = acquisition_components(mean, std, training_y, model.y_scaler.mean_, model.y_scaler.scale_)
            if strategy == 'composite_diversity':
                chosen, trace = diverse_batch(candidates, model.x_scaler.transform(candidate_x),
                                             model.x_scaler.transform(training_x), scores['composite'], 5)
                record['diversity_trace'] = trace
            else:
                chosen = choose_batch(candidates, scores, strategy, 5, seed + round_index)
            record.update(candidate_ids=candidates, chosen_ids=chosen,
                          candidate_components={k: v.tolist() for k, v in scores.items()})
            # Decisions are persisted before either querying pool labels or evaluating the test set.
            write_json(out / f'round_{round_index}_prequery.json', record)
        mean, _ = model.predict(test_x)
        predictions.append(mean)
        record['test_target_rmse'] = np.sqrt(np.mean((mean - test_y) ** 2, axis=0)).tolist()
        history.append(record)
        if round_index < 2:
            training_y = np.vstack((training_y, oracle.reveal(chosen, 'recorded fixed-budget decision')))
            training_ids += chosen
        else:
            model.save(out / 'final_ensemble.pt')
            loaded = LearningEnsemble.load(out / 'final_ensemble.pt')
            np.testing.assert_allclose(loaded.predict(test_x)[0], mean, rtol=0, atol=1e-10)
    np.savez_compressed(out / 'test_predictions.npz', predictions=np.stack(predictions), targets=test_y)
    report = {'strategy': strategy, 'seed': seed, 'history': history, 'oracle_queries': oracle.events,
              'final_revealed_count': len(oracle.revealed), 'test_metrics_used_for_acquisition': False}
    write_json(out / 'report.json', report)
    return report


def run(out):
    if out.exists():
        raise FileExistsError('Use a fresh experiment directory; preserve prior evidence')
    with np.load(ROOT / 'data/scalar63.npz', allow_pickle=False) as data:
        x, y, ids, split = [data[k].copy() for k in ('x', 'y', 'case_ids', 'split')]
        targets = data['target_names'].tolist()
    train, test = split == 'train', split == 'test'
    pool_ids, test_ids = ids[train].tolist(), ids[test].tolist()
    original = json.loads((ROOT / 'evidence/archive/wp6-active-r1/protocol.json').read_text())
    assert pool_ids == original['pool_case_ids'] and test_ids == original['test_case_ids']
    assert (int(train.sum()), int(test.sum()), int((split == 'calibration').sum())) == (37, 13, 13)
    assert np.isfinite(x).all() and np.isfinite(y).all()
    table = pd.DataFrame({'case_id': pool_ids, 'material_id': np.where(x[train, 5] == 1, 'recA', 'recAs')})
    out.mkdir(parents=True)
    protocol = {'name': 'composite_plus_input_diversity_v1', 'offline_archived_label_reveal': True,
                'seeds': [701, 702, 703], 'strategies': list(STRATEGIES), 'budgets': [12, 17, 22],
                'members': 3, 'epochs': 200, 'pool_case_ids': pool_ids, 'test_case_ids': test_ids,
                'protected_calibration_ids': ids[split == 'calibration'].tolist(), 'target_names': targets,
                'diversity_weight': .5, 'distance': 'Euclidean; scaler fitted on revealed training cases only',
                'batch_rule': 'greedy 0.5 composite percentile + 0.5 rank(distance to revealed and selected inputs)',
                'selection_policy': 'fixed once before training; no performance-based hyperparameter search',
                'test_policy': 'previously examined exploratory split; test data used only for evaluation',
                'fresh_solver_calls': 0, 'data_sha256': digest(ROOT / 'data/scalar63.npz'),
                'implementation_sha256': digest(Path(__file__)),
                'ensemble_sha256': digest(ROOT / 'SciML/models/v12/learning_ensemble.py')}
    write_json(out / 'protocol.json', protocol)
    start = time.monotonic(); reports = []
    for seed in protocol['seeds']:
        initial = initial_design(table, seed)
        for strategy in STRATEGIES:
            destination = out / f's{seed}_{strategy}'; destination.mkdir()
            reports.append(experiment(x[train], pool_ids, y[train], x[test], y[test], initial, strategy, seed, destination))
    curves = {}
    for strategy in STRATEGIES:
        values = np.asarray([[r['test_target_rmse'] for r in report['history']]
                             for report in reports if report['strategy'] == strategy])
        curves[strategy] = {'seed_mean_rmse': values.mean(0).tolist(), 'seed_std_rmse': values.std(0, ddof=1).tolist()}
    archived = json.loads((ROOT / 'evidence/archive/wp6-active-r1/summary.json').read_text())
    for strategy in STRATEGIES[:3]:
        np.testing.assert_allclose(curves[strategy]['seed_mean_rmse'], archived['learning_curves'][strategy]['seed_mean_rmse'], rtol=0, atol=1e-10)
    for report in reports:
        assert report['final_revealed_count'] == 22
        if report['strategy'] != 'composite_diversity':
            old = json.loads((ROOT / f"evidence/archive/wp6-active-r1/s{report['seed']}_{report['strategy']}/report.json").read_text())
            for actual, expected in zip(report['oracle_queries'], old['oracle_queries']):
                assert actual['case_ids'] == expected['case_ids']
    summary = {'status': 'completed_fixed_budget_diversity_extension', 'experiment_count': len(reports),
               'network_fits': len(reports) * 3 * 3, 'budgets': protocol['budgets'], 'target_names': targets,
               'learning_curves': curves, 'three_archived_baselines_reproduced': True,
               'checkpoint_reloads_passed': True, 'fresh_solver_calls': 0,
               'elapsed_seconds': time.monotonic() - start,
               'artifacts': {p.relative_to(out).as_posix(): digest(p) for p in sorted(out.rglob('*')) if p.is_file()}}
    write_json(out / 'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps({k: v for k, v in result.items() if k != 'artifacts'}, indent=2))
