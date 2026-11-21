"""Create a local Git evidence bundle without changing complete run directories."""
import hashlib
import json
from pathlib import Path
import shutil

from examples.multicycle_multistep import ROOT, write
from SciML.models.v12.multistep_historical import sha


def run():
    dest=ROOT/'evidence/local_studies/sma_readout'
    dest.mkdir(parents=True,exist_ok=False)
    mapping={'diagnosis':'run-output-sma-diagnosis','decoder_development':'run-output-sma-adaptation-development',
        'readout_development':'run-output-sma-physical-development','second_fold':'run-output-sma-physical-fold2',
        'formal':'run-output-sma-physical-formal'}
    records={}
    for name,folder in mapping.items():
        source=ROOT/folder;target=dest/name;target.mkdir();files={};omitted={}
        for p in sorted(source.rglob('*')):
            if not p.is_file():continue
            rel=p.relative_to(source).as_posix();digest=sha(p)
            if p.suffix=='.pt' and name!='formal':
                omitted[rel]=digest;continue
            out=target/rel;out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,out)
            if sha(out)!=digest:raise ValueError('Archive copy mismatch')
            files[rel]=digest
        record={'source_run':folder,'copied_files':files,'omitted_intermediate_checkpoints_retained_locally':omitted}
        write(target/'local_archive_manifest.json',record);records[name]=record
    write(dest/'archive_manifest.json',{'status':'local_git_archive','github_uploaded':False,'runs':records})
    return {'status':'local_git_archive','files':sum(p.is_file() for p in dest.rglob('*')),
            'bytes':sum(p.stat().st_size for p in dest.rglob('*') if p.is_file())}


if __name__=='__main__':print(json.dumps(run(),indent=2))
