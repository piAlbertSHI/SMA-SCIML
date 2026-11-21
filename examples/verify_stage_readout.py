"""Audit stage-specific checkpoints, train-only refits and immutable forecasts."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .multicycle_multistep import ROOT, data, write
from .train_hierarchical import training_data
from .formal_hierarchical import infer, cohorts, aggregate
from .stage_readout_study import cache, fit_head, FOLDS, SEEDS, PRIOR, candidate_prediction, metrics, gate
from .fit_thermal_readout import fit_readout
from SciML.models.v12.multistep_historical import sha
from SciML.models.v13.modal_operator import ModalStateOperator
from SciML.models.v13.thermal_readout import ThermalReadoutOperator
from SciML.models.v13.stage_thermal_readout import StageThermalReadoutOperator


def verify(root):
    archive=json.loads((root/'archive_manifest.json').read_text())
    for folder,record in archive['runs'].items():
        for name,digest in record['copied_files'].items():assert sha(root/folder/name)==digest,(folder,name)
    protocol=json.loads((root/'formal/protocol.json').read_text())
    for p,digest in protocol['source_sha256'].items():assert sha(root/'formal/source'/p)==digest
    for p,digest in protocol['checkpoint_sha256'].items():assert sha(ROOT/p)==digest
    assert sha(root/'shape_development/summary.json')==protocol['development_sha256']
    development=json.loads((root/'shape_development/summary.json').read_text())
    candidate=protocol['candidate'];assert candidate==development['selected_candidate']
    assert development['passed_both_folds'] and candidate['preserve_spatial_mean']
    arrays,ids,hashes=training_data();assert protocol['data_sha256']==hashes and protocol['fit_ids']==ids
    d,_=data();max_replay=0.;max_refit=0.;max_mean=0.;rows=[]
    for seed in SEEDS:
        dest=root/f'formal/s{seed}';model=StageThermalReadoutOperator.load(dest/'model.pt')
        reference=ThermalReadoutOperator.load(PRIOR/f's{seed}/model.pt')
        for k,v in reference.state_dict().items():assert torch.equal(v,model.reference.state_dict()[k]),k
        assert model.preserve_spatial_mean and float(model.scale)==1.
        replay=infer(model,d)
        with np.load(dest/'test_predictions.npz',allow_pickle=False) as z:saved={k:z[k] for k in z.files}
        for k in replay:
            max_replay=max(max_replay,float(np.max(np.abs(replay[k]-saved[k]))))
            np.testing.assert_allclose(replay[k],saved[k],rtol=0,atol=1e-8)
        with np.load(PRIOR/f's{seed}/test_predictions.npz',allow_pickle=False) as old:
            for a,b in [(saved['state'],old['state']),(saved['field'][:,1],old['field'][:,1]),
                        (saved['field'][:,:,0],old['field'][:,:,0]),(saved['field'][d['field_stage']==4],old['field'][d['field_stage']==4])]:
                np.testing.assert_array_equal(a,b)
            mean_delta=saved['field'][:,0].mean(-1)-old['field'][:,0].mean(-1)
            max_mean=max(max_mean,float(np.abs(mean_delta).max()))
            np.testing.assert_allclose(mean_delta,0.,rtol=0,atol=1e-12)
        fit=cache(reference.backbone,arrays,list(range(8)));head,_=fit_head(reference,fit,candidate)
        for k,v in head.state_dict().items():
            gap=float((v-model.stage2_readout.state_dict()[k]).abs().max());max_refit=max(max_refit,gap)
            torch.testing.assert_close(v,model.stage2_readout.state_dict()[k],rtol=0,atol=1e-9)
        score=cohorts(saved,d);assert score==json.loads((dest/'evaluation.json').read_text());rows.append(score)
    assert aggregate(rows,SEEDS)==json.loads((root/'formal/summary.json').read_text())['adapted']
    folds=[]
    dev_protocol=json.loads((root/'shape_development/protocol.json').read_text())
    for (p,selected,validation),record in zip(FOLDS,dev_protocol['folds']):
        if not (ROOT/p).exists():
            folds.append({'runtime_refit':'local intermediate unavailable','checkpoint_sha256':record['checkpoint_sha256']});continue
        assert sha(ROOT/p)==record['checkpoint_sha256']
        source=ModalStateOperator.load(ROOT/p);ref,_=fit_readout(source,arrays,selected,'residual')
        fit=cache(source,arrays,selected);val=cache(source,arrays,validation);head,_=fit_head(ref,fit,candidate)
        # Use the public inference path, not just cached arithmetic, for the gate.
        model=StageThermalReadoutOperator(ref,head,preserve_spatial_mean=True)
        def live(m):
            fields=[]
            with torch.no_grad():
                for i in validation:
                    out=m(arrays['initial'][i:i+1],arrays['control'][i:i+1],2,
                        times=head.time_grid,positions=head.space_grid)
                    fields.append(out['fields'][0,:,0])
            return torch.stack(fields,dim=1).flatten(0,1)
        baseline=metrics(live(ref),val);after=metrics(live(model),val)
        assert gate(after,baseline)
        cached=candidate_prediction(ref,head,val,candidate)
        torch.testing.assert_close(live(model),cached,rtol=0,atol=1e-12)
        folds.append({'runtime_refit':'passed','before_K':baseline['stage2_mean_rmse_K'],
                      'after_K':after['stage2_mean_rmse_K'],'all_stage2_segments_nonregressing':True})
    return {'status':'verified_stage_thermal_readout','checkpoints_replayed':3,'cases_per_checkpoint':4,
        'max_replay_difference':max_replay,'max_training_only_refit_difference':max_refit,
        'max_spatial_mean_difference_K':max_mean,'stage4_state_fluid_initial_exact':True,
        'inner_folds':folds,'fresh_solver_calls':0,'neural_training_updates':0}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT/'evidence/local_studies/sma_stage_readout')
    p.add_argument('--output',type=Path);a=p.parse_args();torch.set_num_threads(1)
    with threadpool_limits(limits=1):result=verify(a.root)
    if a.output:write(a.output,result)
    print(json.dumps(result,indent=2))
