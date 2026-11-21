"""Archive completed runs and the exact internal-fold backbones, without moving them."""
import json
import shutil

from examples.multicycle_multistep import ROOT,write
from examples.stage_readout_study import FOLDS
from SciML.models.v12.multistep_historical import sha


def run():
    root=ROOT/'evidence/local_studies/sma_risk_limited_readout'
    root.mkdir(parents=True,exist_ok=False);records={}
    runs={'development':'run-output-sma-risk-limited-development',
          'formal':'run-output-sma-risk-limited-formal',
          'comparator_diagnosis':'run-output-sma-risk-public-diagnosis'}
    for name,path in runs.items():
        source=ROOT/path;dest=root/name;shutil.copytree(source,dest)
        copied={p.relative_to(dest).as_posix():sha(p) for p in dest.rglob('*') if p.is_file()}
        for relative,digest in copied.items():assert sha(source/relative)==digest
        records[name]={'source':path,'copied_files':copied}
    checkpoints=root/'internal_backbones';checkpoints.mkdir();bindings={}
    for fold,(path,_,_) in enumerate(FOLDS,1):
        dest=checkpoints/f'fold{fold}_model.pt';shutil.copy2(ROOT/path,dest)
        assert sha(dest)==sha(ROOT/path)
        bindings[dest.name]={'original':path,'sha256':sha(dest)}
    write(root/'archive_manifest.json',{'runs':records,'internal_backbones':bindings,
          'complete_run_copies':True,'sources_retained':True})
    return {'status':'archived_risk_limited_readout','runs':len(records),'internal_backbones':len(bindings)}


if __name__=='__main__':print(json.dumps(run(),indent=2))
