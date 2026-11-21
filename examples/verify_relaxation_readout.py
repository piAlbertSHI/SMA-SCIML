"""Replay fixed response checkpoints, refit train-only rates and audit failures."""
import argparse
import json
import runpy
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .multicycle_multistep import ROOT,write,data
from .train_hierarchical import training_data
from .mean_readout_study import cache,current_from_source,current_prediction
from .relaxation_readout_study import fit_head,predict
from .stage_readout_study import FOLDS,SEEDS,metrics,gate
from .formal_relaxation_readout import PRIOR,evaluate_predictions,verify_development
from .formal_hierarchical import infer
from .verify_mean_development import compare_metrics
from SciML.models.v12.multistep_historical import sha
from SciML.models.v13.modal_operator import ModalStateOperator
from SciML.models.v13.relaxation_readout import RelaxationReadoutOperator,RelaxationMeanReadout,VolumeRelaxationMeanReadout


def run(root):
    archive=json.loads((root/'archive_manifest.json').read_text())
    for folder,row in archive['runs'].items():
        for name,digest in row['copied_files'].items():assert sha(root/folder/name)==digest,(folder,name)
    arrays,ids,hashes=training_data();cached=[]
    for path,selected,validation in FOLDS:
        if not (ROOT/path).is_file():cached.append(None);continue
        ref,fit=current_from_source(ModalStateOperator.load(ROOT/path),arrays,selected)
        cached.append((ref,fit,cache(ref.reference.backbone,arrays,validation)))
    replay_count=0;rate_gap=0.;score_gap=0.;legacy_replayed=False
    for folder in ('thickness_development','volume_development','damped_development'):
        p=json.loads((root/folder/'protocol.json').read_text());s=json.loads((root/folder/'summary.json').read_text())
        assert p['data_sha256']==hashes
        if 'geometry_scaling' not in p:
            # The first archived protocol predates the volume-scaled subclass.
            # Bind its interpretation to that exact source, not a generic fallback.
            assert folder=='thickness_development'
            old_source=(root/folder/'source/examples/relaxation_readout_study.py').read_text()
            assert 'head=RelaxationMeanReadout()' in old_source
            volume=False
        else:volume=p['geometry_scaling']=='duration/volume'
        scale=p.get('mean_scale',1.)
        for fold in s['folds']:
            if cached[fold['fold']-1] is None:continue
            ref,fit,val=cached[fold['fold']-1];binding=p['folds'][fold['fold']-1]
            assert sha(ROOT/binding['checkpoint'])==binding['checkpoint_sha256']
            assert set(binding['fit_ids'])=={x.split(':')[0] for x in ref.training_ids}
            baseline=metrics(current_prediction(ref,val),val)
            for row in fold['candidates']:
                head=VolumeRelaxationMeanReadout() if volume else RelaxationMeanReadout()
                value=torch.load(root/folder/f"fold{fold['fold']}"/row['candidate']/'readout.pt',weights_only=True)
                head.load_state_dict(value['weights'])
                if folder=='thickness_development':
                    # Preserve the first experiment's NumPy division order in
                    # scalar optimization. The later tensor geometry path is
                    # algebraically equivalent but moves the optimum ~3e-9.
                    snapshot=root/folder/'source/SciML/models/v13/relaxation_readout.py'
                    assert sha(snapshot)==p['source_sha256']['SciML/models/v13/relaxation_readout.py']
                    legacy=runpy.run_path(str(snapshot))['RelaxationMeanReadout'];refit=legacy();use=fit['stage']==2
                    observed=row['candidate']=='observed_thermal_response'
                    initial=fit['truth'][use,0,0].mean(-1) if observed else fit['initial'][use,0].mean(-1)
                    fluid=fit['truth'][use,1].mean(-1) if observed else fit['base'][use,1].mean(-1)
                    refit.fit(initial,fluid,fit['control'][use],fit['truth'][use,0].mean(-1))
                    legacy_replayed=True
                else:refit,_=fit_head(fit,row['candidate'],volume)
                rate_gap=max(rate_gap,abs(float(head.rate-refit.rate)))
                torch.testing.assert_close(head.rate,refit.rate,rtol=0,atol=1e-10)
                after=metrics(predict(ref,head,val,scale),val)
                score_gap=max(score_gap,compare_metrics(after,row['metrics']))
                assert (gate(after,baseline) and not row['fit']['rate_at_numerical_bound'])==row['passed'];replay_count+=1
    protocol=json.loads((root/'formal/protocol.json').read_text())
    assert protocol['fit_ids']==ids and protocol['data_sha256']==hashes and protocol['mean_scale']==.5
    for name,digest in protocol['checkpoint_sha256'].items():assert sha(ROOT/name)==digest,name
    d,_=data();predictions=[];references=[];max_gap=0.
    for seed in SEEDS:
        dest=root/f'formal/s{seed}';model=RelaxationReadoutOperator.load(dest/'model.pt')
        fit=cache(model.reference.reference.backbone,arrays,list(range(8)));refit,_=fit_head(fit,'autonomous_thermal_response',True)
        rate_gap=max(rate_gap,abs(float(refit.rate-model.response.rate)))
        torch.testing.assert_close(refit.rate,model.response.rate,rtol=0,atol=1e-10)
        reference=RelaxationReadoutOperator.load(dest/'model.pt').reference
        from SciML.models.v13.stage_thermal_readout import StageThermalReadoutOperator
        retained=StageThermalReadoutOperator.load(PRIOR/f's{seed}/model.pt')
        for k,v in reference.state_dict().items():torch.testing.assert_close(v,retained.state_dict()[k],rtol=0,atol=0)
        replay=infer(model,d)
        with np.load(dest/'test_predictions.npz',allow_pickle=False) as z:
            for key in replay:
                max_gap=max(max_gap,float(np.max(np.abs(replay[key]-z[key]))))
                np.testing.assert_allclose(replay[key],z[key],rtol=0,atol=1e-10)
        predictions.append(replay)
        with np.load(PRIOR/f's{seed}/test_predictions.npz',allow_pickle=False) as z:references.append({k:z[k] for k in z.files})
    result=evaluate_predictions(predictions,references,d);saved=json.loads((root/'formal/summary.json').read_text())
    assert result['promotion_gate']==saved['promotion_gate'] and not result['candidate_promoted']
    assert len(result['individual_seed_regressions'])==len(saved['individual_seed_regressions'])
    for key in ('overall_before_K','overall_after_K'):
        np.testing.assert_allclose(result[key]['values'],saved[key]['values'],rtol=0,atol=1e-10)
    return {'status':'verified_relaxation_candidate_with_disclosed_regression','checkpoints_replayed':3,
        'internal_scores_replayed':replay_count,'local_inner_backbones_available':sum(v is not None for v in cached),
        'legacy_thickness_fit_replayed_from_bound_source':legacy_replayed,
        'max_prediction_replay_difference_K':max_gap,'max_rate_refit_difference':rate_gap,'max_internal_metric_difference_K':score_gap,
        'baseline_checkpoint_weights_unchanged':True,'state_fluid_initial_stage4_exact':True,
        'candidate_promoted':False,'promotion_gate':result['promotion_gate'],
        'individual_seed_regressions':len(result['individual_seed_regressions']),
        'fresh_solver_calls':0,'neural_training_updates':0}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT/'evidence/local_studies/sma_relaxation_readout')
    p.add_argument('--output',type=Path);a=p.parse_args();torch.set_num_threads(1)
    with threadpool_limits(limits=1):result=run(a.root)
    if a.output:write(a.output,result)
    print(json.dumps(result,indent=2))
