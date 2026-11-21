"""Fixed-budget v13 training and autonomous historical comparison; no solver calls."""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .multicycle_multistep import ROOT, SEEDS as OLD_SEEDS, MODES, data, score, write
from .train_hierarchical import run as train, training_data
from SciML.models.v12.learning_historical_rollout import HistoricalModel, BLOCKS, rollout
from SciML.models.v12.multistep_historical import sha
from SciML.models.v13.hierarchical_operator import HierarchicalStateOperator, joint_loss

SEEDS = (20261121, 20261122, 20261123)
DEFAULT = ROOT/'evidence/experiments/v13_formal_v1'
OLD = ROOT/'evidence/experiments/multicycle_multistep_v1'


def infer(model, d):
    """Read initial conditions, controls, horizons and indexing metadata only."""
    state_map, field_map = {}, {}
    times = torch.linspace(0, 1, 50, dtype=torch.float64)
    positions = torch.linspace(0, 1, 101, dtype=torch.float64)
    model.eval()
    with torch.no_grad():
        for i in np.flatnonzero(d['split']=='test'):
            result = model(torch.tensor(d['initial'][i:i+1]), torch.tensor(d['control'][i:i+1]),
                           int(d['cycles'][i]), times=times, positions=positions)
            for key, value in zip(result['keys'], result['states'][0].numpy()):
                state_map[(i, *key)] = value
            for key, value in zip(result['field_keys'], result['fields'][0].numpy()):
                field_map[(i, *key)] = value
    test = d['split'][d['case_index']]=='test'
    state_keys = list(zip(d['case_index'][test], d['cycle'][test], d['stage'][test]))
    field_keys = list(zip(d['field_case_index'], d['field_cycle'], d['field_stage']))
    assert set(state_keys)==set(state_map) and set(field_keys)==set(field_map)
    result = {'state':np.stack([state_map[k] for k in state_keys]),
              'field':np.stack([field_map[k] for k in field_keys])}
    assert all(np.isfinite(v).all() for v in result.values())
    return result


def cohorts(arrays, d):
    test = np.flatnonzero(d['split']=='test'); fixed = test[d['cycles'][test]==3]
    return {'all_test':score(arrays,d,test), 'fixed_three_cycle':score(arrays,d,fixed),
            'per_three_cycle_case':{str(d['case_ids'][i]):score(arrays,d,[i]) for i in fixed}}


def aggregate(rows, seeds):
    result = {}
    for cohort in ('all_test','fixed_three_cycle'):
        result[cohort] = {}
        for category, groups in rows[0][cohort].items():
            result[cohort][category] = {}
            for key, metrics in groups.items():
                dest = {'case_count':metrics['case_count']}
                for name in metrics:
                    if name=='case_count':continue
                    values = [r[cohort][category][key][name] for r in rows]
                    dest[name] = {'mean':float(np.mean(values)), 'sd':float(np.std(values,ddof=1)),
                                  'seeds':list(seeds), 'values':values}
                result[cohort][category][key] = dest
    return result


def old_results(d):
    summary = json.loads((OLD/'summary.json').read_text())
    result = {}
    for mode in MODES:
        rows = []
        for seed in OLD_SEEDS:
            rel = f's{seed}/{mode}.npz'
            assert sha(OLD/rel)==summary['artifacts'][rel]
            with np.load(OLD/rel,allow_pickle=False) as z:arrays = {k:z[k] for k in ('state','field')}
            row = cohorts(arrays,d)
            recorded = json.loads((OLD/f's{seed}/report.json').read_text())['modes'][mode]
            for key in row:assert row[key]==recorded[key]
            rows.append(row)
        result[mode] = aggregate(rows,OLD_SEEDS)
    return result


def affine_results(d):
    test = np.flatnonzero(d['split']=='test'); fixed = test[d['cycles'][test]==3]
    selected = d['split'][d['case_index']]=='test'
    ci,cy,st = [d[k][selected] for k in ('case_index','cycle','stage')]
    # The affine prior is seed-independent. Verify this before reporting one reference.
    models = [HistoricalModel.load(ROOT/f'checkpoints/historical-rollout-r1/s{s}/model.pt') for s in OLD_SEEDS]
    states = [np.concatenate([rollout(m,d['initial'][i],d['control'][i],int(d['cycles'][i]),neural=False) for i in test]) for m in models]
    for a in states[1:]:np.testing.assert_array_equal(a,states[0])
    result = {}
    for name,indices in [('all_test',test),('fixed_three_cycle',fixed)]:
        result[name] = {}
        for c in (1,2,3):
            for s in range(1,6):
                use = np.isin(ci,indices)&(cy==c)&(st==s)
                if use.any():
                    result[name][f'c{c}s{s}'] = {'case_count':int(use.sum()), **{
                        block:float(np.sqrt(np.mean((states[0][use,sl]-d['after'][selected][use,sl])**2)))
                        for block,sl in BLOCKS.items()}}
    return result


def train_diagnostics(model):
    arrays,ids,_ = training_data(); losses = []
    with torch.no_grad():
        for i in range(len(ids)):
            positions = torch.linspace(0,1,101,dtype=torch.float64)
            pred = model(arrays['initial'][i:i+1],arrays['control'][i:i+1],2,
                         times=torch.linspace(0,1,50,dtype=torch.float64),positions=positions)
            loss = joint_loss(pred,arrays['states'][i:i+1],model.feature_scale,
                              field_truth=arrays['fields'][i:i+1],field_scale=model.decoder.field_scale,positions=positions)
            losses.append({'case_id':ids[i],**{k:float(v) for k,v in loss.items()}})
    return {'full_grid_per_case':losses,'full_grid_mean':{k:float(np.mean([r[k] for r in losses])) for k in ('state','field','spatial_gradient','total')}}


def run(output):
    if output.exists():raise FileExistsError('Preserve old runs; use a new directory')
    d,provenance = data(); _,ids,hashes = training_data()
    fixed = np.flatnonzero((d['split']=='test')&(d['cycles']==3))
    output.mkdir(parents=True)
    sources = ['examples/formal_hierarchical.py','examples/train_hierarchical.py','examples/multicycle_multistep.py',
               'SciML/models/v13/hierarchical_operator.py']
    protocol = {'experiment':'v13_formal_v1','written_before_training':True,'seeds':list(SEEDS),
        'groups':provenance['groups'],'fixed_three_cycle_cases':d['case_ids'][fixed].tolist(),
        'training':{'updates_per_seed':200,'batch_cases':2,'total_updates':600,'optimizer':'AdamW','lr':.0003,
            'weight_decay':1e-4,'gradient_clip':1.,'warmup_updates':20,'cosine_final_fraction':.1,
            'selection':'fixed final update; no early stopping, test selection or hyperparameter search',
            'precision':'CPU float64, one BLAS/PyTorch thread','train_cases':8,'cycles':[1,2],
            'query_sampling':'8 times including zero/end, 26 spatial positions; see per-update logs',
            'initialization':'fresh neural weights; frozen training-only Ridge and scalers from historical seed 20260910',
            'loss':'eight-block state MSE + field MSE + 0.01 supervised spatial-gradient MSE; normalized with training data'},
        'evaluation':{'input':'observed first mechanical frame plus declared thermal defaults, control9 and requested horizon only',
            'autonomous':True,'teacher_forcing':False,'test_cases':4,'three_cycle_cases':2,
            'transitions':46,'exchange_segments':20,'field_grid':[50,101],'exclude_field_time_zero':True,
            'primary_endpoints':['c3s4 SMA state RMSE','c3s4 future SMA field RMSE','c3s4 future HTF field RMSE'],
            'aggregation':'pooled RMSE over fixed two cases per seed; arithmetic mean and sample SD across seeds',
            'limitations':'historical test cohort already examined; seed SD is not a confidence interval over independent cases'},
        'comparison':'same cases, input information, horizons and evaluation grids; different training budgets/supervision, not an architecture-only ablation',
        'reference_modes':list(MODES)+['affine_state_only'],'old_summary_sha256':sha(OLD/'summary.json'),
        'data_sha256':hashes,'source_sha256':{p:sha(ROOT/p) for p in sources},
        'demonstration_selection':{'case_id':str(d['case_ids'][fixed[0]]),'new_seed':SEEDS[0],'old_seed':OLD_SEEDS[0],
                                   'old_mode':'archived_one_step','rule':'first sorted case and first listed seeds; no best-seed selection'},
        'environment':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__},'fresh_solver_calls':0}
    write(output/'protocol.json',protocol)
    started = time.perf_counter(); reports = []; training_seconds = []
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        # Finish all preset fits before the first test inference.
        for seed in SEEDS:
            result = train(output/f's{seed}',200,2,seed,.0003)
            training_seconds.append(result['training_seconds'])
        old = old_results(d); affine = affine_results(d)
        for seed in SEEDS:
            dest = output/f's{seed}';model = HierarchicalStateOperator.load(dest/'model.pt')
            arrays = infer(model,d)
            np.savez_compressed(dest/'test_predictions.npz',**arrays)
            report = {'seed':seed,**cohorts(arrays,d),'training_diagnostics':train_diagnostics(model)}
            write(dest/'evaluation.json',report);reports.append(report)
    result = {'status':'completed_fixed_budget_v13_comparison','fresh_solver_calls':0,
        'new':aggregate(reports,SEEDS),'references':old,'affine_state_only':affine,
        'training_seconds_by_seed':training_seconds,'total_training_seconds':sum(training_seconds),
        'training_and_evaluation_seconds':time.perf_counter()-started,
        'model_selection':'all fixed final checkpoints; no test-based choice',
        'artifacts':{p.relative_to(output).as_posix():sha(p) for p in sorted(output.rglob('*')) if p.is_file()}}
    write(output/'summary.json',result)
    return {'status':result['status'],'training_seconds':result['total_training_seconds'],
            'total_seconds':result['training_and_evaluation_seconds'],'third_cycle':{
                k:result['new']['fixed_three_cycle'][k]['c3s4'] for k in ('state','field')},'fresh_solver_calls':0}


def verify(output):
    d,_ = data(); torch.set_num_threads(1)
    summary = json.loads((output/'summary.json').read_text())
    protocol = json.loads((output/'protocol.json').read_text())
    for rel,value in summary['artifacts'].items():assert sha(output/rel)==value,rel
    assert protocol['old_summary_sha256']==sha(OLD/'summary.json')
    _,ids,hashes = training_data();assert protocol['data_sha256']==hashes
    reports = [];maximum = 0.
    with threadpool_limits(limits=1):
        for seed in SEEDS:
            dest = output/f's{seed}';report = json.loads((dest/'evaluation.json').read_text());reports.append(report)
            training = json.loads((dest/'protocol.json').read_text());history = json.loads((dest/'history.json').read_text())
            assert training['training_case_ids']==ids and training['updates']==200 and len(history)==200
            assert all(set(row['train_case_ids'])<=set(ids) for row in history)
            replay = infer(HierarchicalStateOperator.load(dest/'model.pt'),d)
            with np.load(dest/'test_predictions.npz',allow_pickle=False) as z:arrays = {k:z[k] for k in z.files}
            for k in arrays:
                maximum = max(maximum,float(np.abs(arrays[k]-replay[k]).max()))
                np.testing.assert_allclose(arrays[k],replay[k],rtol=0,atol=1e-8)
            for k,v in cohorts(arrays,d).items():assert v==report[k]
        assert aggregate(reports,SEEDS)==summary['new']
        assert old_results(d)==summary['references']
        # Affine computations can differ by roundoff across BLAS implementations.
        affine = affine_results(d)
        for cohort,groups in affine.items():
            for key,metrics in groups.items():
                for name,value in metrics.items():
                    np.testing.assert_allclose(value,summary['affine_state_only'][cohort][key][name],rtol=0,atol=1e-8)
    return {'status':'verified_v13_formal_comparison','checkpoints_replayed':3,'test_cases_per_checkpoint':4,
        'test_scores_recomputed':True,'old_prediction_sets_checked':9,'training_ids_checked':True,
        'max_replay_absolute_difference':maximum,'replay_absolute_tolerance':1e-8,'fresh_solver_calls':0}


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=DEFAULT)
    p.add_argument('--verify-only',action='store_true');a = p.parse_args()
    print(json.dumps(verify(a.output) if a.verify_only else run(a.output),indent=2))
