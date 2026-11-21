"""Freeze the selected response recipe before three-seed historical regression."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .multicycle_multistep import ROOT,write,data
from .train_hierarchical import training_data
from .mean_readout_study import cache,current_from_source,current_prediction
from .stage_readout_study import FOLDS,SEEDS,metrics,gate
from .relaxation_readout_study import fit_head,predict
from .verify_mean_development import compare_metrics
from .formal_hierarchical import infer,cohorts,aggregate
from SciML.models.v12.multistep_historical import sha
from SciML.models.v13.modal_operator import ModalStateOperator
from SciML.models.v13.stage_thermal_readout import StageThermalReadoutOperator
from SciML.models.v13.relaxation_readout import VolumeRelaxationMeanReadout,RelaxationReadoutOperator

PRIOR=ROOT/'evidence/local_studies/sma_stage_readout/formal'


def verify_development(folder):
    protocol=json.loads((folder/'protocol.json').read_text());summary=json.loads((folder/'summary.json').read_text())
    if not summary['passed_both_folds']:raise ValueError('Both development gates must pass before historical evaluation')
    assert protocol['geometry_scaling']=='duration/volume' and protocol['mean_scale']==.5
    candidate=summary['selected_candidate'];assert candidate=='autonomous_thermal_response'
    manifest=json.loads((folder/'manifest.json').read_text())
    for p,digest in manifest['artifacts'].items():assert sha(folder/p)==digest,p
    arrays,ids,hashes=training_data();assert protocol['data_sha256']==hashes;reports=[]
    for (path,selected,validation),record in zip(FOLDS,summary['folds']):
        binding=protocol['folds'][record['fold']-1]
        assert binding['checkpoint']==path and sha(ROOT/path)==binding['checkpoint_sha256']
        assert binding['fit_ids']==[ids[i] for i in selected] and binding['validation_ids']==[ids[i] for i in validation]
        ref,_=current_from_source(ModalStateOperator.load(ROOT/path),arrays,selected)
        val=cache(ref.reference.backbone,arrays,validation)
        value=torch.load(folder/f"fold{record['fold']}"/candidate/'readout.pt',weights_only=True)
        head=VolumeRelaxationMeanReadout();head.load_state_dict(value['weights'])
        model=RelaxationReadoutOperator(ref,head,.5)
        fields=[]
        with torch.no_grad():
            for i in validation:
                result=model(arrays['initial'][i:i+1],arrays['control'][i:i+1],2,
                    times=head.time_grid,positions=ref.stage2_readout.space_grid)
                fields.append(result['fields'][0,:,0])
        live=torch.stack(fields,dim=1).flatten(0,1)
        cached=predict(ref,head,val,.5);gap=float((live-cached).abs().max())
        torch.testing.assert_close(live,cached,rtol=0,atol=1e-12)
        before=metrics(current_prediction(ref,val),val);after=metrics(live,val)
        assert gate(after,before);compare_metrics(after,record['candidates'][0]['metrics'])
        reports.append({'fold':record['fold'],'public_inference_gate_passed':True,'max_cached_live_difference_K':gap})
    return reports


def evaluate_predictions(predictions,references,d):
    comparisons=[];overall_before=[];overall_after=[]
    for seed,p,old in zip(SEEDS,predictions,references):
        for a,b in ((p['state'],old['state']),(p['field'][:,1],old['field'][:,1]),
                    (p['field'][:,:,0],old['field'][:,:,0]),
                    (p['field'][d['field_stage']==4],old['field'][d['field_stage']==4])):
            np.testing.assert_array_equal(a,b)
        np.testing.assert_allclose(np.diff(p['field'][:,0],axis=-1),np.diff(old['field'][:,0],axis=-1),rtol=0,atol=1e-12)
        for dest,z in ((overall_before,old),(overall_after,p)):
            dest.append(float(np.sqrt(np.mean((z['field'][:,0,1:]-d['field_truth'][:,0,1:])**2))))
        for i in range(len(d['field_stage'])):
            entry={'seed':seed,'case_id':str(d['case_ids'][d['field_case_index'][i]]),
                'cycle':int(d['field_cycle'][i]),'stage':int(d['field_stage'][i])}
            for name,z in (('before',old),('after',p)):
                e=z['field'][i,0,1:]-d['field_truth'][i,0,1:]
                entry[name+'_K']=float(np.sqrt(np.mean(e*e)))
                entry[name+'_bias_K']=float(e.mean())
            comparisons.append(entry)
    groups=[]
    for case,cycle,stage in sorted({(r['case_id'],r['cycle'],r['stage']) for r in comparisons}):
        selected=[r for r in comparisons if (r['case_id'],r['cycle'],r['stage'])==(case,cycle,stage)]
        groups.append({'case_id':case,'cycle':cycle,'stage':stage,**{name:{'mean':float(np.mean([r[name+'_K'] for r in selected])),
            'sd':float(np.std([r[name+'_K'] for r in selected],ddof=1)),'values':[r[name+'_K'] for r in selected]} for name in ('before','after')}})
    prior=json.loads((ROOT/'assets/sma_stage_readout/analysis.json').read_text())['four_segments']
    target=[r for r in prior if r['cycle'] in (1,2) and '_reca_' in r['case_id']]
    recovered=[]
    for r in target:
        row=next(g for g in groups if (g['case_id'],g['cycle'],g['stage'])==(r['case_id'],r['cycle'],r['stage']))
        recovered.append(row['after']['mean']<r['modal_metrics']['rmse_K'])
    gate_results={'all_test_mean_improves':np.mean(overall_after)<np.mean(overall_before),
        'every_segment_seed_mean_nonregressing':all(r['after']['mean']<=r['before']['mean'] for r in groups),
        'worst_seed_all_test_nonregressing':max(overall_after)<=max(overall_before),
        'case_A_first_two_cycles_recover_modal_accuracy':len(recovered)==2 and all(recovered),
        'state_fluid_initial_stage4_exact':True}
    return {'overall_before_K':{'mean':float(np.mean(overall_before)),'sd':float(np.std(overall_before,ddof=1)),'values':overall_before},
        'overall_after_K':{'mean':float(np.mean(overall_after)),'sd':float(np.std(overall_after,ddof=1)),'values':overall_after},
        'groups':groups,'comparisons':comparisons,'promotion_gate':{k:bool(v) for k,v in gate_results.items()},
        'candidate_promoted':all(gate_results.values()),
        'individual_seed_regressions':[r for r in comparisons if r['after_K']>r['before_K']]}


def run(output,development):
    checks=verify_development(development);arrays,ids,hashes=training_data()
    output.mkdir(parents=True,exist_ok=False)
    sources=['SciML/models/v13/relaxation_readout.py','examples/formal_relaxation_readout.py',
        'examples/relaxation_readout_study.py','examples/stage_readout_study.py']
    protocol={'candidate':'volume-scaled autonomous thermal response, fixed half blend','seeds':list(SEEDS),
        'mean_scale':.5,'fit_ids':ids,'data_sha256':hashes,'development_sha256':sha(development/'summary.json'),
        'internal_public_inference_checks':checks,'checkpoint_sha256':{
            (PRIOR/f's{s}/model.pt').relative_to(ROOT).as_posix():sha(PRIOR/f's{s}/model.pt') for s in SEEDS},
        'source_sha256':{p:sha(ROOT/p) for p in sources},
        'promotion_rules':['overall mean field RMSE decreases','every case/cycle/stage three-seed mean nonregresses',
            'worst seed overall RMSE nonregresses','case A cycles 1 and 2 recover modal-only reference accuracy',
            'state, fluid, initial field and stage4 exactly unchanged'],
        'individual_seed_segment_errors':'all reported; no post-evaluation parameter changes',
        'evaluation_status':'historical development regression, not untouched independent confirmation',
        'fresh_solver_calls':0,'neural_training_updates':0}
    write(output/'protocol.json',protocol)
    for p in sources:
        dest=output/'source'/p;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/p).read_bytes())
    started=time.perf_counter();fits=[]
    for seed in SEEDS:
        ref=StageThermalReadoutOperator.load(PRIOR/f's{seed}/model.pt')
        fit=cache(ref.reference.backbone,arrays,list(range(8)))
        head,info=fit_head(fit,'autonomous_thermal_response',True)
        if info['rate_at_numerical_bound']:raise ValueError('Response rate hit numerical boundary')
        dest=output/f's{seed}';dest.mkdir();model=RelaxationReadoutOperator(ref,head,.5)
        model.save(dest/'model.pt',metadata={'fit':info,'recipe':protocol['candidate']});write(dest/'fit.json',info);fits.append(info)
    fit_seconds=time.perf_counter()-started
    d,_=data();predictions=[];references=[];cohort_rows=[]
    for seed in SEEDS:
        dest=output/f's{seed}';pred=infer(RelaxationReadoutOperator.load(dest/'model.pt'),d)
        np.savez_compressed(dest/'test_predictions.npz',**pred);predictions.append(pred)
        score=cohorts(pred,d);cohort_rows.append(score);write(dest/'evaluation.json',score)
        with np.load(PRIOR/f's{seed}/test_predictions.npz',allow_pickle=False) as z:references.append({k:z[k] for k in z.files})
    result={'status':'completed_fixed_relaxation_readout','fits':fits,'fit_seconds':fit_seconds,
        'adapted':aggregate(cohort_rows,SEEDS),**evaluate_predictions(predictions,references,d)}
    write(output/'summary.json',result)
    write(output/'manifest.json',{'artifacts':{p.relative_to(output).as_posix():sha(p) for p in output.rglob('*') if p.is_file()}})
    return {k:result[k] for k in ('status','fit_seconds','fits','overall_before_K','overall_after_K','promotion_gate',
        'candidate_promoted','individual_seed_regressions','groups')}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--development',required=True,type=Path);a=p.parse_args();torch.set_num_threads(1)
    with threadpool_limits(limits=1):result=run(a.output,a.development)
    print(json.dumps(result,indent=2))
