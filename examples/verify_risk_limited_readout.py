"""Replay all seeds, refit training-only calibration, and verify both internal folds."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .multicycle_multistep import ROOT,write,data
from .train_hierarchical import training_data
from .mean_readout_study import cache
from .stage_readout_study import SEEDS
from .risk_limited_readout_study import fit_response
from .formal_risk_limited_readout import verify_development
from .formal_relaxation_readout import PRIOR,evaluate_predictions
from .formal_hierarchical import infer,cohorts,aggregate
from .verify_relaxation_readout import run as verify_legacy
from SciML.models.v12.multistep_historical import sha
from SciML.models.v13.hierarchical_operator import HierarchicalStateOperator
from SciML.models.v13.risk_limited_readout import RiskLimitedRelaxationOperator
from SciML.models.v13.stage_thermal_readout import StageThermalReadoutOperator


def compare_tree(actual,expected):
    if isinstance(actual,dict):
        assert set(actual)==set(expected)
        for key,value in actual.items():compare_tree(value,expected[key])
    elif isinstance(actual,list):
        assert len(actual)==len(expected)
        for a,b in zip(actual,expected):compare_tree(a,b)
    elif isinstance(actual,float):np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-10)
    else:assert actual==expected,(actual,expected)


def check_field_entries(prediction, dataset):
    """Bind each decoded initial frame to the rollout's entering state."""
    test_rows = dataset['split'][dataset['case_index']] == 'test'
    state_keys = zip(*(dataset[key][test_rows] for key in ('case_index', 'cycle', 'stage')))
    state_indices = {key: index for index, key in enumerate(state_keys)}
    case_indices = dataset['field_case_index']
    cycles = dataset['field_cycle']
    for stage in (2, 4):
        field_rows = np.flatnonzero(dataset['field_stage'] == stage)
        entry_rows = [state_indices[(case_indices[i], cycles[i], stage - 1)] for i in field_rows]
        entries = torch.from_numpy(prediction['state'][entry_rows])
        expected = HierarchicalStateOperator.initial_fields(entries, stage)
        np.testing.assert_array_equal(prediction['field'][field_rows, :, 0], expected.numpy())
    return len(case_indices)


def summarize_field_rmse(predictions, references, dataset):
    """Report all-case and cycle-3 results with the same per-seed pooling."""
    error = {
        name: np.stack([run['field'][:, 0, 1:] - dataset['field_truth'][:, 0, 1:] for run in runs])
        for name, runs in (('before', references), ('after', predictions))
    }

    def score(mask):
        statistics = {}
        for name, values in error.items():
            rmse = np.sqrt(np.mean(values[:, mask] ** 2, axis=(1, 2, 3)))
            statistics[name] = {'mean': float(rmse.mean()), 'sd': float(rmse.std(ddof=1)),
                                'values': rmse.tolist()}
        return statistics

    case_indices = dataset['field_case_index']
    by_case = [
        {'case_id': str(dataset['case_ids'][index]), **score(case_indices == index)}
        for index in np.unique(case_indices)
    ]
    cycle3 = dataset['field_cycle'] == 3
    stages = dataset['field_stage']
    return {
        'case_pooled_sma_field_rmse_K': by_case,
        'worst_case_id_before': max(by_case, key=lambda row: row['before']['mean'])['case_id'],
        'worst_case_id_after': max(by_case, key=lambda row: row['after']['mean'])['case_id'],
        'third_cycle_sma_field_rmse_K': {
            'both_stages': score(cycle3),
            'stage2': score(cycle3 & (stages == 2)),
            'stage4': score(cycle3 & (stages == 4)),
        },
    }


def compare_forecast(archived_path, prediction):
    """Allow replay roundoff while checking every stored component."""
    max_gap = 0.
    with np.load(archived_path, allow_pickle=False) as archived:
        assert set(prediction) == set(archived.files)
        for key, value in prediction.items():
            np.testing.assert_allclose(value, archived[key], rtol=0, atol=1e-10)
            max_gap = max(max_gap, float(np.max(np.abs(value - archived[key]))))
    return max_gap


def run(root):
    archive=json.loads((root/'archive_manifest.json').read_text())
    for folder,row in archive['runs'].items():
        for name,digest in row['copied_files'].items():assert sha(root/folder/name)==digest,(folder,name)
    for name,row in archive['internal_backbones'].items():assert sha(root/'internal_backbones'/name)==row['sha256']
    internal=verify_development(root/'development',root/'internal_backbones')
    arrays,ids,hashes=training_data();protocol=json.loads((root/'formal/protocol.json').read_text())
    assert protocol['fit_ids']==ids and protocol['data_sha256']==hashes and protocol['mean_scale']==.5
    assert protocol['error_scale_multiplier']==1. and protocol['seeds']==list(SEEDS)
    assert sha(root/'development/summary.json')==protocol['development_sha256']
    for name,digest in protocol['checkpoint_sha256'].items():assert sha(ROOT/name)==digest
    for folder in ('development','formal'):
        p=json.loads((root/folder/'protocol.json').read_text())
        for name,digest in p['source_sha256'].items():assert sha(root/folder/'source'/name)==digest
    d,_=data();predictions=[];references=[];cohort_rows=[];reports=[]
    for seed in SEEDS:
        dest=root/f'formal/s{seed}';model=RiskLimitedRelaxationOperator.load(dest/'model.pt')
        assert float(model.mean_scale)==.5
        reference=StageThermalReadoutOperator.load(PRIOR/f's{seed}/model.pt')
        assert set(model.reference.state_dict())==set(reference.state_dict())
        for key,value in reference.state_dict().items():
            torch.testing.assert_close(value,model.reference.state_dict()[key],rtol=0,atol=0)
        fit=cache(reference.reference.backbone,arrays,list(range(len(ids))))
        head,scale,info,members=fit_response(fit)
        compare_tree(info,json.loads((dest/'fit.json').read_text()))
        torch.testing.assert_close(head.rate,model.response.rate,rtol=0,atol=1e-10)
        torch.testing.assert_close(scale,model.error_scale,rtol=0,atol=1e-10)
        saved=torch.load(dest/'calibration.pt',weights_only=True)['members']
        assert len(members)==len(saved)==len(ids)
        for member,weights in zip(members,saved):
            for key,value in member.state_dict().items():torch.testing.assert_close(value,weights[key],rtol=0,atol=1e-10)
        pred = infer(model, d)
        reference_pred = infer(reference, d)
        entries_matched = check_field_entries(pred, d)
        gap = compare_forecast(dest/'test_predictions.npz', pred)
        reference_gap = compare_forecast(PRIOR/f's{seed}'/'test_predictions.npz', reference_pred)
        score=cohorts(pred,d);compare_tree(score,json.loads((dest/'evaluation.json').read_text()))
        cohort_rows.append(score);predictions.append(pred)
        references.append(reference_pred)
        reports.append({'seed':seed,'prediction_max_difference_K':gap,
            'reference_replay_max_difference':reference_gap,
            'field_entries_equal_to_entering_state':entries_matched,
            'rate_refit_difference':float((head.rate-model.response.rate).abs()),
            'error_scale_refit_max_difference_K':float((scale-model.error_scale).abs().max()),
            'deletion_members_verified':len(members),'baseline_weights_exact':True})
    result=evaluate_predictions(predictions,references,d);saved=json.loads((root/'formal/summary.json').read_text())
    for key,value in result.items():compare_tree(value,saved[key])
    compare_tree(aggregate(cohort_rows,SEEDS),saved['adapted'])
    assert result['candidate_promoted'] and all(result['promotion_gate'].values())
    # Verify inherited inference refactoring against the previous, unmodified
    # checkpoints and all their archived scores, including failed experiments.
    legacy=verify_legacy(ROOT/'evidence/local_studies/sma_relaxation_readout')
    return {'status':'verified_risk_limited_readout','internal_public_inference':internal,'seeds':reports,
        'promotion_gate':result['promotion_gate'],'candidate_promoted':True,
        'individual_seed_regressions':result['individual_seed_regressions'],
        **summarize_field_rmse(predictions, references, d),
        'state_fluid_initial_stage4_exact':True,'spatial_difference_tolerance_K':1e-12,
        'legacy_response_verification':legacy,'fresh_solver_calls':0,'neural_training_updates':0}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT/'evidence/local_studies/sma_risk_limited_readout')
    p.add_argument('--output',type=Path);a=p.parse_args();torch.set_num_threads(1)
    with torch.no_grad(),threadpool_limits(limits=1):result=run(a.root)
    if a.output:write(a.output,result)
    print(json.dumps(result,indent=2))
