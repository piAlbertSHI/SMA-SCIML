"""Preserve every response candidate, diagnostic and fixed historical evaluation."""
import json
from pathlib import Path
import shutil

from examples.multicycle_multistep import ROOT,write
from SciML.models.v12.multistep_historical import sha

RUNS={'thickness_development':'run-output-sma-relaxation-development',
      'failure_diagnosis':'run-output-sma-relaxation-diagnosis',
      'volume_development':'run-output-sma-volume-relaxation-development',
      'damped_development':'run-output-sma-damped-relaxation-development',
      'formal':'run-output-sma-relaxation-formal',
      'capacity_audit':'run-output-sma-exchange-capacity-audit'}


def run():
    root=ROOT/'evidence/local_studies/sma_relaxation_readout';root.mkdir(parents=True,exist_ok=False);records={}
    for key,relative in RUNS.items():
        source=ROOT/relative;assert source.is_dir();dest=root/key;shutil.copytree(source,dest)
        copied={p.relative_to(dest).as_posix():sha(p) for p in dest.rglob('*') if p.is_file()}
        for name,digest in copied.items():assert sha(source/name)==digest
        records[key]={'source':relative,'copied_files':copied}
    write(root/'archive_manifest.json',{'runs':records,'complete_run_copies':True,'sources_retained':True})
    result=json.loads((root/'formal/summary.json').read_text())
    write(root/'stage_checkpoint.json',{'goal_status':'active','candidate_promoted':result['candidate_promoted'],
        'numerical_objective_achieved':False,'retained_baseline_commit':'3f4f4b6',
        'candidate_before_K':result['overall_before_K'],'candidate_after_K':result['overall_after_K'],
        'promotion_gate':result['promotion_gate'],'formal_summary_sha256':sha(root/'formal/summary.json'),
        'remaining_issue':'One stage2 segment three-seed mean regresses; six individual seed-segments regress. '
            'Keep this fixed candidate as evidence; do not retune it on historical scores.',
        'fresh_solver_calls':0,'neural_training_updates':0,'github_pushes':0})
    return {'status':'archived_relaxation_study','runs':len(records),'candidate_promoted':result['candidate_promoted']}


if __name__=='__main__':print(json.dumps(run(),indent=2))
