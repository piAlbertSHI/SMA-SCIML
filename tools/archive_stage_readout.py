"""Archive stage-readout evidence locally without overwriting previous studies."""
import json
import shutil

from examples.multicycle_multistep import ROOT, write
from SciML.models.v12.multistep_historical import sha


def run():
    target=ROOT/'evidence/local_studies/sma_stage_readout';target.mkdir(parents=True,exist_ok=False)
    mapping={'diagnosis':'run-output-sma-stage-diagnosis','initial_development':'run-output-sma-stage-development',
             'shape_development':'run-output-sma-stage-shape-development','formal':'run-output-sma-stage-formal'}
    records={}
    for label,path in mapping.items():
        source=ROOT/path;dest=target/label;files={}
        for p in sorted(source.rglob('*')):
            if not p.is_file():continue
            rel=p.relative_to(source).as_posix();out=dest/rel;out.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(p,out);assert sha(out)==sha(p)
            files[rel]=sha(p)
        records[label]={'source_run':path,'copied_files':files}
    write(target/'archive_manifest.json',{'status':'local_git_archive','github_uploaded':False,'runs':records})
    return {'status':'local_git_archive','files':sum(len(r['copied_files']) for r in records.values())}


if __name__=='__main__':print(json.dumps(run(),indent=2))
