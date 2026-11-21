"""Replay final corrections, refit their training-only linear readouts and audit invariants."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .multicycle_multistep import ROOT, data
from .train_hierarchical import training_data
from .fit_thermal_readout import fit_readout
from .adapt_sma import field_metrics
from .formal_hierarchical import infer, cohorts, aggregate
from SciML.models.v13.modal_operator import ModalStateOperator
from SciML.models.v13.thermal_readout import ThermalReadoutOperator
from SciML.models.v12.multistep_historical import sha

DEFAULT=ROOT/'evidence/local_studies/sma_readout'
SEEDS=(20261121,20261122,20261123)


def verify(root):
    archive=json.loads((root/'archive_manifest.json').read_text())
    for folder,record in archive['runs'].items():
        for rel,digest in record['copied_files'].items():assert sha(root/folder/rel)==digest,(folder,rel)
    protocol=json.loads((root/'formal/protocol.json').read_text())
    for rel,digest in protocol['source_sha256'].items():assert sha(root/'formal/source'/rel)==digest
    for rel,digest in protocol['checkpoint_sha256'].items():assert sha(ROOT/rel)==digest
    assert protocol['development_sha256']==sha(root/'readout_development/summary.json')
    assert protocol['cross_check_sha256']==sha(root/'second_fold/summary.json')
    arrays,ids,hashes=training_data();assert protocol['data_sha256']==hashes and protocol['fit_case_ids']==ids
    d,_=data();rows=[];maximum=0.;refit_max=0.
    for seed in SEEDS:
        dest=root/f'formal/s{seed}';model=ThermalReadoutOperator.load(dest/'model.pt')
        base=ModalStateOperator.load(ROOT/f'evidence/experiments/hro_modal/formal/s{seed}/refined/model.pt')
        assert set(model.training_ids)==set(base.training_ids)
        for k,v in model.backbone.state_dict().items():assert torch.equal(v,base.state_dict()[k]),k
        replay=infer(model,d)
        with np.load(dest/'test_predictions.npz',allow_pickle=False) as z:
            saved={k:z[k] for k in z.files}
            for k in replay:
                maximum=max(maximum,float(np.abs(replay[k]-saved[k]).max()))
                np.testing.assert_allclose(replay[k],saved[k],rtol=0,atol=1e-8)
        with np.load(ROOT/f'evidence/experiments/hro_modal/formal/s{seed}/test_predictions.npz') as old:
            for a,b in [(saved['state'],old['state']),(saved['field'][:,1],old['field'][:,1]),
                        (saved['field'][:,:,0],old['field'][:,:,0])]:
                np.testing.assert_array_equal(a,b)
        refitted,_=fit_readout(base,arrays,list(range(8)),'residual')
        for k,v in model.readout.state_dict().items():
            gap=float((v-refitted.readout.state_dict()[k]).abs().max());refit_max=max(refit_max,gap)
            torch.testing.assert_close(v,refitted.readout.state_dict()[k],rtol=0,atol=1e-9)
        row=cohorts(saved,d);rows.append(row)
        assert row==json.loads((dest/'evaluation.json').read_text())
    summary=json.loads((root/'formal/summary.json').read_text())
    assert aggregate(rows,SEEDS)==summary['adapted']
    # Recheck both inner folds with the finalized cache execution mode, reusing
    # their existing backbones; no neural retraining and no test-case fitting.
    fold_rows=[]
    for path,selected,validation,recorded in [
        ('run-output-hro-modal-refinement/refined/model.pt',[0,1,2,4,5,6],[3,7],root/'readout_development/summary.json'),
        ('run-output-sma-physical-fold2/refined/model.pt',[0,2,3,4,6,7],[1,5],root/'second_fold/summary.json')]:
        file=ROOT/path
        if not file.exists():
            fold_rows.append({'runtime_refit':'local intermediate checkpoint unavailable','record_sha256':sha(recorded)})
            continue
        m=ModalStateOperator.load(file);m.eval().requires_grad_(False)
        before=field_metrics(m,arrays,validation);after,_=fit_readout(m,arrays,selected,'residual')
        result=field_metrics(after,arrays,validation)
        assert result['mean']['sma_rmse_K']<before['mean']['sma_rmse_K']
        assert result['mean']['c2s4_sma_rmse_K']<before['mean']['c2s4_sma_rmse_K']
        assert all(a['sma_rmse_K']<=1.05*b['sma_rmse_K'] and a['fluid_rmse_K']==b['fluid_rmse_K']
                   for a,b in zip(result['per_case'],before['per_case']))
        fold_rows.append({'runtime_refit':'passed','baseline':before['mean'],'corrected':result['mean']})
    return {'status':'verified_conditional_thermal_readout','checkpoints_replayed':3,'cases_per_checkpoint':4,
        'max_replay_difference':maximum,'max_training_only_refit_difference':refit_max,
        'state_fluid_initial_fields_exactly_preserved':True,'both_inner_folds':fold_rows,'fresh_solver_calls':0}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=DEFAULT)
    p.add_argument('--output',type=Path,help='Write verification JSON to a new file');a=p.parse_args()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):result=verify(a.root)
    if a.output:
        with a.output.open('x',encoding='utf-8',newline='\n') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result,indent=2))
