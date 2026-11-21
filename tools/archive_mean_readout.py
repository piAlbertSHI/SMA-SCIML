"""Preserve all mean-readout development evidence without promoting a candidate."""
import json
import shutil

from examples.multicycle_multistep import ROOT, write
from SciML.models.v12.multistep_historical import sha

RUNS={'diagnosis':'run-output-sma-mean-diagnosis','direction':'run-output-sma-mean-direction-diagnosis',
      'residual':'run-output-sma-mean-development'}
for profile,part in [('compact','compact'),('evolution','evolution'),('observed_blend','observed-blend'),
    ('interaction','interaction'),('profile_evolution','profile-evolution'),('context','context'),
    ('reliable','reliable'),('pointwise','pointwise'),('latent','latent'),('kernel','kernel')]:
    RUNS[profile]=f'run-output-sma-mean-{part}-development'


def run():
    dest=ROOT/'evidence/local_studies/sma_mean_readout';dest.mkdir(parents=True,exist_ok=False);records={};summaries=[]
    for name,folder in RUNS.items():
        source=ROOT/folder;target=dest/name;files={}
        for p in sorted(source.rglob('*')):
            if not p.is_file():continue
            rel=p.relative_to(source).as_posix();out=target/rel;out.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(p,out);assert sha(out)==sha(p);files[rel]=sha(p)
        records[name]={'source_run':folder,'copied_files':files}
        if (target/'summary.json').exists():
            s=json.loads((target/'summary.json').read_text())
            summaries.append({'profile':name,'selected_candidate':s['selected_candidate'],
                'passed_both_folds':s['passed_both_folds'],'seconds':s['seconds'],
                'folds':[{'fold':f['fold'],'reference':f['reference']['stage2_mean_rmse_K'],
                    'candidates':[{'name':c['candidate']['name'],'rmse_K':c['metrics']['stage2_mean_rmse_K'],'passed':c['passed']}
                                  for c in f['candidates']]} for f in s['folds']]})
    write(dest/'archive_manifest.json',{'status':'local_development_archive','runs':records,'github_uploaded':False})
    result={'status':'no_mean_candidate_promoted','development':summaries,'total_recorded_development_seconds':sum(s['seconds'] for s in summaries),
        'historical_test_candidates_scored':0,'fresh_solver_calls':0,'neural_training_updates':0,
        'active_baseline':{'commit':'3f4f4b6','checkpoints':{f'evidence/local_studies/sma_stage_readout/formal/s{s}/model.pt':
            sha(ROOT/f'evidence/local_studies/sma_stage_readout/formal/s{s}/model.pt') for s in (20261121,20261122,20261123)}}}
    write(dest/'summary.json',result)
    return {k:v for k,v in result.items() if k not in ('development','active_baseline')}


if __name__=='__main__':print(json.dumps(run(),indent=2))
