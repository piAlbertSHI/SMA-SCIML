"""Fixed empirical-error shrinkage followed by historical regression evaluation."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .multicycle_multistep import ROOT,data,write
from .train_hierarchical import training_data
from .mean_readout_study import cache,current_from_source,current_prediction
from .risk_limited_readout_study import fit_response,predict
from .stage_readout_study import FOLDS,SEEDS,metrics,gate
from .formal_relaxation_readout import PRIOR,evaluate_predictions
from .formal_hierarchical import infer,cohorts,aggregate
from .verify_mean_development import compare_metrics
from SciML.models.v12.multistep_historical import sha
from SciML.models.v13.modal_operator import ModalStateOperator
from SciML.models.v13.stage_thermal_readout import StageThermalReadoutOperator
from SciML.models.v13.relaxation_readout import VolumeRelaxationMeanReadout
from SciML.models.v13.risk_limited_readout import RiskLimitedRelaxationOperator


def verify_development(folder,checkpoint_root=None):
    summary=json.loads((folder/'summary.json').read_text());protocol=json.loads((folder/'protocol.json').read_text())
    if not summary['passed_both_folds']:raise ValueError('Both development gates must pass first')
    assert protocol['mean_scale']==.5 and protocol['error_scale_multiplier']==1.
    manifest=json.loads((folder/'manifest.json').read_text())
    for p,digest in manifest['artifacts'].items():assert sha(folder/p)==digest,p
    arrays,ids,hashes=training_data();assert hashes==protocol['data_sha256'];reports=[]
    assert len(summary['folds'])==len(protocol['folds'])==len(FOLDS)
    for (path,selected,validation),record,binding in zip(FOLDS,summary['folds'],protocol['folds']):
        source_path=ROOT/path if checkpoint_root is None else checkpoint_root/f"fold{record['fold']}_model.pt"
        assert binding['checkpoint']==path and sha(source_path)==binding['checkpoint_sha256']
        assert binding['fit_ids']==[ids[i] for i in selected] and binding['validation_ids']==[ids[i] for i in validation]
        source=ModalStateOperator.load(source_path)
        assert {v.split(':')[0] for v in source.training_ids}==set(binding['fit_ids'])
        ref,fit=current_from_source(source,arrays,selected);val=cache(ref.reference.backbone,arrays,validation)
        saved=torch.load(folder/f"fold{record['fold']}"/'readout.pt',weights_only=True)
        head=VolumeRelaxationMeanReadout();head.load_state_dict(saved['response']);scale=saved['error_scale']
        refit,refit_scale,_,members=fit_response(fit)
        torch.testing.assert_close(head.rate,refit.rate,rtol=0,atol=1e-10)
        torch.testing.assert_close(scale,refit_scale,rtol=0,atol=1e-10)
        assert len(members)==len(saved['members'])==len(selected)
        for member,weights in zip(members,saved['members']):
            for key,value in member.state_dict().items():
                torch.testing.assert_close(value,weights[key],rtol=0,atol=1e-10)
        model=RiskLimitedRelaxationOperator(ref,head,error_scale=scale);fields=[];reference_fields=[]
        with torch.no_grad():
            for i in validation:
                result=model(arrays['initial'][i:i+1],arrays['control'][i:i+1],2,
                    times=head.time_grid,positions=ref.stage2_readout.space_grid)
                fields.append(result['fields'][0,:,0])
                reference_result=ref(arrays['initial'][i:i+1],arrays['control'][i:i+1],2,
                    times=head.time_grid,positions=ref.stage2_readout.space_grid)
                reference_fields.append(reference_result['fields'][0,:,0])
        live=torch.stack(fields,dim=1).flatten(0,1);cached=predict(ref,head,scale,val)
        torch.testing.assert_close(live,cached,rtol=0,atol=1e-12)
        # Compare identical batching/inference paths, especially when shrinkage
        # makes the correction exactly zero. Keep the strict scientific gate.
        live_reference=torch.stack(reference_fields,dim=1).flatten(0,1)
        cached_reference=current_prediction(ref,val)
        torch.testing.assert_close(live_reference,cached_reference,rtol=0,atol=1e-12)
        before=metrics(live_reference,val);after=metrics(live,val)
        assert gate(after,before);compare_metrics(after,record['candidate'])
        reports.append({'fold':record['fold'],'public_inference_gate_passed':True,
                        'comparison_path':'per-case batch1 live reference and candidate',
                        'max_cached_live_reference_K':float((live_reference-cached_reference).abs().max()),
                        'max_cached_live_difference_K':float((live-cached).abs().max())})
    return reports


def run(output,development):
    checks=verify_development(development);arrays,ids,hashes=training_data();output.mkdir(parents=True,exist_ok=False)
    sources=['SciML/models/v13/risk_limited_readout.py','SciML/models/v13/relaxation_readout.py',
        'examples/risk_limited_readout_study.py','examples/formal_risk_limited_readout.py','examples/formal_relaxation_readout.py']
    protocol={'candidate':'volume-scaled response with training-case RMS shrinkage and fixed half blend',
        'mean_scale':.5,'error_scale_multiplier':1.,'seeds':list(SEEDS),'fit_ids':ids,'data_sha256':hashes,
        'development_sha256':sha(development/'summary.json'),'public_inference_checks':checks,
        'checkpoint_sha256':{(PRIOR/f's{s}/model.pt').relative_to(ROOT).as_posix():sha(PRIOR/f's{s}/model.pt') for s in SEEDS},
        'source_sha256':{p:sha(ROOT/p) for p in sources},
        'promotion_rules':['overall mean field RMSE decreases','every case/cycle/stage three-seed mean nonregresses',
            'worst seed overall RMSE nonregresses','case A cycles 1 and 2 recover modal-only reference accuracy',
            'state, fluid, initial field and stage4 exactly unchanged'],
        'calibration_scope':'case deletion only for response rates; frozen backbone saw fit cases; no coverage guarantee',
        'evaluation_scope':'historical development regression; no coefficient changes after historical scoring',
        'fresh_solver_calls':0,'neural_training_updates':0}
    write(output/'protocol.json',protocol)
    for p in sources:
        dest=output/'source'/p;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/p).read_bytes())
    started=time.perf_counter();fits=[]
    for seed in SEEDS:
        ref=StageThermalReadoutOperator.load(PRIOR/f's{seed}/model.pt')
        values=cache(ref.reference.backbone,arrays,list(range(8)));response,scale,info,members=fit_response(values)
        if info['central']['rate_at_numerical_bound'] or any(r['fit']['rate_at_numerical_bound'] for r in info['case_deletions']):
            raise ValueError('Response rate hit numerical bounds')
        model=RiskLimitedRelaxationOperator(ref,response,error_scale=scale);dest=output/f's{seed}';dest.mkdir()
        model.save(dest/'model.pt',metadata={'fit':info,'recipe':protocol['candidate']});write(dest/'fit.json',info)
        torch.save({'members':[m.state_dict() for m in members]},dest/'calibration.pt');fits.append(info)
    fit_seconds=time.perf_counter()-started;d,_=data();predictions=[];references=[];cohort_rows=[]
    for seed in SEEDS:
        dest=output/f's{seed}';p=infer(RiskLimitedRelaxationOperator.load(dest/'model.pt'),d)
        np.savez_compressed(dest/'test_predictions.npz',**p);predictions.append(p)
        score=cohorts(p,d);cohort_rows.append(score);write(dest/'evaluation.json',score)
        with np.load(PRIOR/f's{seed}/test_predictions.npz',allow_pickle=False) as z:references.append({k:z[k] for k in z.files})
    result={'status':'completed_fixed_risk_limited_readout','fit_seconds':fit_seconds,'fits':fits,
        'adapted':aggregate(cohort_rows,SEEDS),**evaluate_predictions(predictions,references,d)}
    write(output/'summary.json',result)
    write(output/'manifest.json',{'artifacts':{p.relative_to(output).as_posix():sha(p) for p in output.rglob('*') if p.is_file()}})
    return {k:result[k] for k in ('status','fit_seconds','overall_before_K','overall_after_K','promotion_gate',
                               'candidate_promoted','groups','individual_seed_regressions')}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--development',required=True,type=Path);a=p.parse_args();torch.set_num_threads(1)
    with threadpool_limits(limits=1):result=run(a.output,a.development)
    print(json.dumps(result,indent=2))
