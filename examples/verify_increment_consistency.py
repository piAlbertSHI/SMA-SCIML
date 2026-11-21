"""Verify the physical identity audit and replay the rejected projection."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .multicycle_multistep import ROOT, write
from .train_hierarchical import training_data
from .stage_readout_study import metrics, gate
from .audit_thermal_means import describe, reconstructed
from .verify_mean_development import compare_metrics
from SciML.models.v12.multistep_historical import sha
from SciML.models.v13.thermal_geometry import strip_weights, radial_weights, weighted_mean
from SciML.models.v13.increment_consistency import project_heat_increment


def run():
    geometry=ROOT/'evidence/local_studies/sma_mean_geometry'
    study=ROOT/'evidence/local_studies/sma_increment_consistency'
    for root in (geometry,study):
        manifest=json.loads((root/'manifest.json').read_text())
        for path,digest in manifest['artifacts'].items():assert sha(root/path)==digest,path
    audit=json.loads((geometry/'analysis.json').read_text());arrays,ids,hashes=training_data()
    assert hashes==audit['data_sha256']
    ws,wr=strip_weights(),radial_weights();max_identity=0.
    assert len(audit['training_rows'])==32
    for row in audit['training_rows']:
        i=ids.index(row['case_id']);j=2*(row['cycle']-1)+(row['stage']==4);step=(1,3,6,8)[j]
        before=arrays['states'][i,step-1].numpy();after=arrays['states'][i,step].numpy()
        field=arrays['fields'][i,j,0].numpy()
        entry=reconstructed(before);np.testing.assert_allclose(entry,field[0],rtol=0,atol=1e-12)
        delta=float(weighted_mean(after[102:153]-before[102:153],wr)-weighted_mean(field[-1]-field[0],ws))
        np.testing.assert_allclose(delta,row['radial_minus_strip_delta_K'],rtol=0,atol=1e-12)
        max_identity=max(max_identity,abs(delta))
    assert max_identity<1e-12
    record=json.loads((study/'summary.json').read_text());diagnostic=[];replayed=0
    for fold in record['folds']:
        values=torch.load(study/f"fold{fold['fold']}.pt",weights_only=True);use=values['stage']==2
        replay=values['before'].clone()
        replay[use],_=project_heat_increment(values['before'][use],values['entry_state'][use],values['exit_state'][use])
        torch.testing.assert_close(replay,values['after'],rtol=0,atol=0)
        cache={'truth':values['truth'][:,None],**{k:values[k] for k in ('case','cycle','stage')}}
        before,after=metrics(values['before'],cache),metrics(replay,cache)
        compare_metrics(before,fold['reference']);compare_metrics(after,fold['candidate'])
        assert gate(after,before)==fold['passed']
        for ix in torch.nonzero(use).flatten().tolist():
            ci=int(values['case'][ix]);cycle=int(values['cycle'][ix]);step=(1,6)[cycle-1]
            pred_delta=weighted_mean((values['exit_state'][ix,102:153]-values['entry_state'][ix,102:153]).numpy(),wr)
            true_delta=weighted_mean((arrays['states'][ci,step,102:153]-arrays['states'][ci,step-1,102:153]).numpy(),wr)
            entry_error=weighted_mean((values['before'][ix,0]-values['truth'][ix,0]).numpy(),ws)
            before_end_error=weighted_mean((values['before'][ix,-1]-values['truth'][ix,-1]).numpy(),ws)
            after_end_error=weighted_mean((values['after'][ix,-1]-values['truth'][ix,-1]).numpy(),ws)
            np.testing.assert_allclose(after_end_error,entry_error+pred_delta-true_delta,rtol=0,atol=1e-12)
            e=values['before'][ix,1:]-values['truth'][ix,1:];delta=values['after'][ix,1:]-values['before'][ix,1:]
            diagnostic.append({'fold':fold['fold'],'case_index':ci,'cycle':cycle,'stage':2,
                'predicted_increment_error_K':float(pred_delta-true_delta),'entry_capacity_bias_K':float(entry_error),
                'before_endpoint_capacity_bias_K':float(before_end_error),'after_endpoint_capacity_bias_K':float(after_end_error),
                'correction_square_K2':float(delta.square().mean()),'error_correction_cross_K2':float(2*(e*delta).mean())})
        replayed+=1
    assert record['passed_both_folds']==all(f['passed'] for f in record['folds'])
    return {'status':'verified_increment_identity_and_rejected_projection','training_identity_rows':32,
        'max_increment_identity_error_K':max_identity,'folds_replayed':replayed,'max_prediction_replay_error_K':0.,
        'projection_diagnosis':diagnostic,'candidate_promoted':False,'fresh_solver_calls':0}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path);a=p.parse_args();result=run()
    if a.output:write(a.output,result)
    print(json.dumps(result,indent=2))
