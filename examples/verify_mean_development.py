"""Replay every accepted/rejected internal score and verify that the baseline is unchanged."""
import argparse
import json
from pathlib import Path

import torch
import numpy as np
from threadpoolctl import threadpool_limits

from .multicycle_multistep import ROOT, write
from .train_hierarchical import training_data
from .mean_readout_study import cache, current_from_source, current_prediction, mean_prediction, fit_mean, candidates_for
from .stage_readout_study import FOLDS, metrics, gate
from SciML.models.v12.multistep_historical import sha
from SciML.models.v13.modal_operator import ModalStateOperator
from SciML.models.v13.mean_thermal_readout import make_mean_readout


def compare_metrics(actual,saved):
    assert len(actual['segments'])==len(saved['segments'])
    differences=[abs(actual['stage2_mean_rmse_K']-saved['stage2_mean_rmse_K'])]
    for a,b in zip(actual['segments'],saved['segments']):
        assert all(a[k]==b[k] for k in ('case_index','cycle','stage'))
        differences.extend(abs(a[k]-b[k]) for k in ('rmse_K','bias_K'))
    assert max(differences)<1e-10
    return max(differences)


def run(root):
    manifest=json.loads((root/'archive_manifest.json').read_text())
    for name,record in manifest['runs'].items():
        for path,digest in record['copied_files'].items():assert sha(root/name/path)==digest,(name,path)
    summary=json.loads((root/'summary.json').read_text())
    for path,digest in summary['active_baseline']['checkpoints'].items():assert sha(ROOT/path)==digest,path
    arrays,ids,hashes=training_data();loaded=[]
    for path,selected,validation in FOLDS:
        if not (ROOT/path).exists():loaded.append(None);continue
        ref,fit=current_from_source(ModalStateOperator.load(ROOT/path),arrays,selected)
        val=cache(ref.reference.backbone,arrays,validation);loaded.append((ref,fit,val))
    checked=0;max_gap=0.;refit_gap=0.
    for record in summary['development']:
        folder=root/record['profile'];p=json.loads((folder/'protocol.json').read_text());s=json.loads((folder/'summary.json').read_text())
        assert candidates_for(record['profile'])==p['candidates'] and p['data_sha256']==hashes
        assert s['historical_test_scored'] is False and s['passed_both_folds'] is False
        for rel,digest in p['source_sha256'].items():assert sha(folder/'source'/rel)==digest
        for fold in s['folds']:
            if loaded[fold['fold']-1] is None:continue
            ref,fit,val=loaded[fold['fold']-1]
            binding=p['folds'][fold['fold']-1]
            assert sha(ROOT/binding['checkpoint'])==binding['checkpoint_sha256']
            assert {s.split(':')[0] for s in ref.training_ids}==set(binding['fit_ids'])
            baseline=metrics(current_prediction(ref,val),val)
            max_gap=max(max_gap,compare_metrics(baseline,fold['reference']))
            for row in fold['candidates']:
                c=row['candidate'];file=folder/f"fold{fold['fold']}"/c['name']/'readout.pt'
                value=torch.load(file,weights_only=True);assert value['candidate']==c
                config={'kind':c['kind'],'rank':c['rank'],'samples':int((fit['stage']==2).sum()),'latent_width':ref.reference.backbone.config.width}
                head=make_mean_readout(config);head.load_state_dict(value['weights'])
                members=[]
                for weights in value.get('members',[]):
                    member=make_mean_readout({**config,'samples':config['samples']-2});member.load_state_dict(weights);members.append(member)
                after=metrics(mean_prediction(ref,head,val,c,members),val)
                gap=compare_metrics(after,row['metrics'])
                max_gap=max(max_gap,gap)
                assert gap<1e-10 and gate(after,baseline)==row['passed']
                refit,_=fit_mean(ref,fit,c)
                for k,v in head.state_dict().items():
                    gap=float((v-refit.state_dict()[k]).abs().max());refit_gap=max(refit_gap,gap)
                    torch.testing.assert_close(v,refit.state_dict()[k],rtol=0,atol=1e-9)
                checked+=1
    return {'status':'verified_mean_development_without_promotion','internal_scores_replayed':checked,
        'max_score_replay_difference_K':max_gap,'max_readout_refit_difference':refit_gap,
        'local_backbone_folds_available':sum(v is not None for v in loaded),
        'baseline_checkpoint_hashes_unchanged':True,'historical_test_candidates_scored':0,'fresh_solver_calls':0}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT/'evidence/local_studies/sma_mean_readout')
    p.add_argument('--output',type=Path);a=p.parse_args();torch.set_num_threads(1)
    with threadpool_limits(limits=1):result=run(a.root)
    if a.output:write(a.output,result)
    print(json.dumps(result,indent=2))
