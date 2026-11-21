"""Fixed-budget continuation of archived historical multi-cycle checkpoints.

Training uses eight whole two-cycle cases; evaluation replays the four original
test cases, including two held-out three-cycle cases. No physical solver calls.
"""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from SciML.models.v12.learning_historical_rollout import HistoricalModel, BLOCKS, rollout, mask
from SciML.models.v12.learning_historical_fields import cascade_inputs
from SciML.models.v12.learning_coordinate_operator import CoordinateOperator
from SciML.models.v12.multistep_historical import DifferentiableHistorical, continuation, sha

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (20260910, 20260911, 20260912)
MODES = ('archived_one_step', 'single_step_continue', 'multistep')
STAGES = [1, 2, 3, 4, 5, 1, 2, 3, 4]
DEFAULT = ROOT/'evidence/experiments/multicycle_multistep_v1'


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def data():
    path = ROOT/'data/historical12_state.npz'
    provenance = json.loads(path.with_suffix('.json').read_text())
    if sha(path) != provenance['snapshot_sha256']:
        raise ValueError('Historical snapshot checksum mismatch')
    with np.load(path, allow_pickle=False) as z:
        d = {k:z[k] for k in z.files}
    assert len(d['case_ids']) == 12 and len(d['sample_ids']) == 118
    assert provenance['all_targets_equal_archived_predictions']
    for part, ids in provenance['groups'].items():
        assert d['case_ids'][d['split']==part].tolist() == ids
    train = np.flatnonzero(d['split']=='train')
    assert len(train) == 8 and np.all(d['cycles'][train]==2)
    for i in train:
        use = d['case_index']==i
        np.testing.assert_array_equal(d['stage'][use], STAGES)
        np.testing.assert_array_equal(d['before'][use][1:], d['after'][use][:-1])
        np.testing.assert_array_equal(d['initial'][i], d['before'][use][0])
    return d, provenance


def infer(model, operator, d):
    """Only starts, controls and requested horizons enter either predictor."""
    ids = np.flatnonzero(d['split']=='test')
    starts = {d['case_ids'][i]:{'initial':d['initial'][i], 'control':d['control'][i],
                              'cycles':int(d['cycles'][i])} for i in ids}
    state = np.concatenate([rollout(model, **starts[d['case_ids'][i]]) for i in ids])
    entries = cascade_inputs(model, starts, [d['case_ids'][i] for i in ids])
    pairs = [entries[(d['case_ids'][i], int(c), int(s))]
             for i,c,s in zip(d['field_case_index'],d['field_cycle'],d['field_stage'])]
    branch = np.stack([p[0] for p in pairs]); initial = np.stack([p[1] for p in pairs])
    field = operator.predict(branch, initial)
    np.testing.assert_array_equal(field[:,:,0], initial)
    assert np.isfinite(state).all() and np.isfinite(field).all()
    return {'state':state, 'field':field, 'forecast_initial':initial}


def score(arrays, d, selected_cases):
    test = d['split'][d['case_index']]=='test'
    ci,cy,st = (d[k][test] for k in ('case_index','cycle','stage'))
    truth = d['after'][test]
    result = {'state':{}, 'field':{}}
    for c in (1,2,3):
        for s in range(1,6):
            use = np.isin(ci,selected_cases)&(cy==c)&(st==s)
            if use.any():
                result['state'][f'c{c}s{s}'] = {'case_count':int(use.sum()),
                    **{name:float(np.sqrt(np.mean((arrays['state'][use,sl]-truth[use,sl])**2)))
                       for name,sl in BLOCKS.items()}}
        for s in (2,4):
            use = np.isin(d['field_case_index'],selected_cases)&(d['field_cycle']==c)&(d['field_stage']==s)
            if use.any():
                result['field'][f'c{c}s{s}'] = {'case_count':int(use.sum()),
                    **{name:float(np.sqrt(np.mean((arrays['field'][use,j,1:]-d['field_truth'][use,j,1:])**2)))
                       for j,name in enumerate(('sma_K','htf_K'))}}
    return result


def compare_archived(scores, seed):
    state = json.loads((ROOT/f'evidence/archive/historical-rollout-r1/s{seed}/report.json').read_text())
    field = json.loads((ROOT/f'evidence/archive/historical-fields-r1/s{seed}/report.json').read_text())
    diffs=[]
    for key, row in scores['state'].items():
        c,s=int(key[1]),int(key[3])
        for block in BLOCKS:
            diffs.append(abs(row[block]-state['groups']['test']['autonomous_neural'][f'cycle_{c}'][str(s)][block]))
    for key, row in scores['field'].items():
        c,s=int(key[1]),int(key[3])
        for name in ('sma','htf'):
            diffs.append(abs(row[name+'_K']-field['test']['autonomous_boundary'][f'cycle{c}_stage{s}'][name]['future_rmse_K']))
    if max(diffs)>1e-8:
        raise ValueError(f'Archived baseline replay mismatch: {max(diffs)}')
    return max(diffs)


def aggregate(reports):
    result={}
    for cohort in ('all_test', 'fixed_three_cycle'):
        result[cohort]={}
        for mode in MODES:
            first=reports[0]['modes'][mode][cohort]
            result[cohort][mode]={}
            for category, groups in first.items():
                result[cohort][mode][category]={}
                for key, metrics in groups.items():
                    dest={'case_count':metrics['case_count']}
                    for metric in metrics:
                        if metric=='case_count': continue
                        values=[r['modes'][mode][cohort][category][key][metric] for r in reports]
                        dest[metric]={'mean':float(np.mean(values)), 'sd':float(np.std(values,ddof=1)),
                                      'seeds':list(SEEDS),'values':values}
                    result[cohort][mode][category][key]=dest
    return result


def run(output):
    if output.exists(): raise FileExistsError('Preserve previous runs; choose a new output directory')
    d, provenance=data()
    train=np.flatnonzero(d['split']=='train'); test=np.flatnonzero(d['split']=='test')
    fixed=test[d['cycles'][test]==3]
    train_rows=d['split'][d['case_index']]=='train'
    training_ids=d['sample_ids'][train_rows].tolist()
    checkpoints={}
    manifest=json.loads((ROOT/'evidence/bundle_manifest.json').read_text())['files']
    for seed in SEEDS:
        for folder, name in [('historical-rollout-r1','model.pt'),('historical-fields-r1','operator.pt')]:
            rel=f'checkpoints/{folder}/s{seed}/{name}'
            checkpoints[rel]=sha(ROOT/rel)
            assert checkpoints[rel]==manifest[rel]
    output.mkdir(parents=True)
    protocol={'experiment':'multicycle_multistep_v1', 'written_before_training':True,
        'seeds':list(SEEDS),'modes':list(MODES),'groups':provenance['groups'],
        'fixed_three_cycle_cases':d['case_ids'][fixed].tolist(),
        'training_transitions':72,'test_transitions':46,'field_test_samples':20,
        'state_dimension':515,'feature_dimension':529,
        'active_outputs_by_stage':{str(s):int(mask(s).sum()) for s in range(1,6)},
        'training_stage_chain':STAGES, 'training_cycles':[1,2],
        'continuation':{'updates':300,'optimizer':'AdamW','lr':.0005,'weight_decay':1e-4,'gradient_clip_norm':10.,
            'full_batch_cases':8,'dtype':'float64','device':'cpu',
            'loss':'mean of eight block MSEs after division by frozen training feature scales; equal case/time weights',
            'trainable':'five neural residual maps only','frozen':'Ridge maps, input scaler, residual amplitude scales, projection, all field decoders',
            'teacher_forced_control':'same budget and loss; true before-states from training cases only',
            'multistep':'nine transitions; model-fed intermediate states; full backpropagation without detach',
            'selection':'fixed final update; no test-based stopping or hyperparameter selection'},
        'primary_endpoints':['c3s4 SMA boundary RMSE','c3s4 future SMA field RMSE','c3s4 future HTF field RMSE'],
        'evaluation':'autonomous inference from first initial state, control9 and horizon only; field errors omit time zero',
        'aggregation':'per-seed pooled RMSE across fixed two cases; arithmetic mean and sample SD across three seeds; per-case scores also saved',
        'scope':'held-out cases in an already studied historical corpus; not a new untouched benchmark or current63 evidence',
        'fresh_solver_calls':0,'source_snapshot_sha256':provenance['snapshot_sha256'],
        'source_fingerprints':provenance['fingerprints'],'checkpoints_sha256':checkpoints,
        'source_sha256':{p:sha(ROOT/p) for p in ['examples/multicycle_multistep.py','SciML/models/v12/multistep_historical.py']},
        'environment':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__}}
    write(output/'protocol.json',protocol)
    before=np.stack([d['before'][d['case_index']==i] for i in train])
    after=np.stack([d['after'][d['case_index']==i] for i in train])
    reports=[];torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        for seed in SEEDS:
            dest=output/f's{seed}';dest.mkdir()
            base_rel=f'checkpoints/historical-rollout-r1/s{seed}/model.pt'
            base=HistoricalModel.load(ROOT/base_rel)
            assert base.training_ids==training_ids
            operator=CoordinateOperator.load(ROOT/f'checkpoints/historical-fields-r1/s{seed}/operator.pt')
            report={'seed':seed,'modes':{}}
            original=infer(base,operator,d)
            adapter=infer(DifferentiableHistorical(base),operator,d)
            report['torch_adapter_max_abs_difference']=max(float(np.abs(original[k]-adapter[k]).max()) for k in original)
            for k in original: np.testing.assert_allclose(original[k],adapter[k],rtol=0,atol=1e-8)
            for mode in MODES:
                start=time.perf_counter(); history=[]; diagnostics={}
                if mode=='archived_one_step': arrays=original
                else:
                    model=DifferentiableHistorical(base)
                    buffers={k:v.clone() for k,v in model.named_buffers()}
                    history,diagnostics=continuation(model,d['initial'][train],d['control'][train],before,after,STAGES,mode)
                    for k,v in model.named_buffers():
                        if not torch.equal(v,buffers[k]): raise ValueError('Frozen affine/scaler buffer changed')
                    arrays=infer(model,operator,d)
                    weight_path=dest/(mode+'.pt')
                    model.save_residual(weight_path,base_rel,checkpoints[base_rel],mode,history)
                    replay=infer(DifferentiableHistorical.load_residual(weight_path,ROOT),operator,d)
                    for k in arrays: np.testing.assert_array_equal(arrays[k],replay[k])
                np.savez_compressed(dest/(mode+'.npz'),**arrays)
                score_all=score(arrays,d,test)
                if mode=='archived_one_step':
                    report['archived_score_max_abs_difference']=compare_archived(score_all,seed)
                report['modes'][mode]={'training_history':history,'training_final_loss':diagnostics,
                    'elapsed_seconds':time.perf_counter()-start,'all_test':score_all,
                    'fixed_three_cycle':score(arrays,d,fixed),
                    'per_three_cycle_case':{str(d['case_ids'][i]):score(arrays,d,[i]) for i in fixed}}
            write(dest/'report.json',report);reports.append(report)
    # Original checkpoints and snapshot must remain untouched.
    for rel,expected in checkpoints.items(): assert sha(ROOT/rel)==expected
    assert sha(ROOT/'data/historical12_state.npz')==provenance['snapshot_sha256']
    summary={'status':'completed_fixed_budget_multicycle_comparison','fresh_solver_calls':0,
        'all_checkpoint_replays_exact':True,'frozen_buffers_unchanged':True,
        'archived_baseline_scores_match':True,'aggregate':aggregate(reports),
        'artifacts':{p.relative_to(output).as_posix():sha(p) for p in sorted(output.rglob('*')) if p.is_file()}}
    write(output/'summary.json',summary)
    return {'status':summary['status'],'fresh_solver_calls':0,'third_cycle':{
        mode:{category:summary['aggregate']['fixed_three_cycle'][mode][category]['c3s4'] for category in ('state','field')}
        for mode in MODES}}


def verify(output):
    """Recompute all recorded test scores and replay saved final checkpoints."""
    d,_=data();torch.set_num_threads(1)
    summary=json.loads((output/'summary.json').read_text())
    for rel,expected in summary['artifacts'].items():
        assert sha(output/rel)==expected, rel
    reports=[];test=np.flatnonzero(d['split']=='test');fixed=test[d['cycles'][test]==3]
    with threadpool_limits(limits=1):
        for seed in SEEDS:
            dest=output/f's{seed}';r=json.loads((dest/'report.json').read_text());reports.append(r)
            operator=CoordinateOperator.load(ROOT/f'checkpoints/historical-fields-r1/s{seed}/operator.pt')
            for mode in MODES:
                model=(HistoricalModel.load(ROOT/f'checkpoints/historical-rollout-r1/s{seed}/model.pt') if mode==MODES[0]
                       else DifferentiableHistorical.load_residual(dest/(mode+'.pt'),ROOT))
                replay=infer(model,operator,d)
                with np.load(dest/(mode+'.npz'),allow_pickle=False) as z:
                    for k in replay: np.testing.assert_allclose(z[k],replay[k],rtol=0,atol=1e-8)
                    arrays={k:z[k] for k in z.files}
                for cohort,cases in [('all_test',test),('fixed_three_cycle',fixed)]:
                    assert score(arrays,d,cases)==r['modes'][mode][cohort]
                for i in fixed:
                    assert score(arrays,d,[i])==r['modes'][mode]['per_three_cycle_case'][str(d['case_ids'][i])]
    assert aggregate(reports)==summary['aggregate']
    return {'status':'verified_multicycle_comparison','checkpoints_replayed':9,'saved_scores_recomputed':True,
            'replay_absolute_tolerance':1e-8,'fresh_solver_calls':0}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=DEFAULT);p.add_argument('--verify-only',action='store_true');a=p.parse_args()
    print(json.dumps(verify(a.output) if a.verify_only else run(a.output),indent=2))
