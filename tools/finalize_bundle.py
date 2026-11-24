"""Maintainer command: create a new manifest only after reviewing intentional edits."""
import hashlib,json,platform,importlib.metadata as md
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def included(p):
    rel=p.relative_to(ROOT)
    return p.is_file() and not any(part in ('.git','.venv','__pycache__','.batch_runtime') or part.startswith(('demo-output','run-output','learning-exercises','hro_spatial_ablation_','hro_internal_validation_','hro_protected_state_','hro_material_conditioned_','hro_temporal_')) or part == 'hro_field_entry_consistency_20261123' for part in rel.parts) and p.suffix not in ('.pyc','.pyo')
def main():
    # Bind LF bytes so Windows worktrees and Linux GitHub checkouts agree.
    for p in ROOT.rglob('*'):
        if included(p) and (p.suffix in ('.py','.md','.json','.yml','.yaml','.txt') or p.name in ('.gitattributes','.gitignore')):
            body=p.read_text(encoding='utf-8')
            p.write_text(body,encoding='utf-8',newline='\n')
    p=ROOT/'evidence/export_provenance.json'
    provenance=json.loads(p.read_text(encoding='utf-8'))
    for row in provenance['source_records']:
        if 'export_relative' in row:row['export_sha256']=sha(ROOT/row['export_relative'])
    p.write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding='utf-8',newline='\n')
    env={'python':platform.python_version(),'platform':platform.system(),'packages':{v:md.version(v) for v in ['numpy','scipy','pandas','scikit-learn','torch','h5py','pyarrow','matplotlib','pillow','threadpoolctl']},'note':'observed environment, not a tested fresh installation lock'}
    (ROOT/'evidence/current_environment.json').write_text(json.dumps(env,indent=2),encoding='utf-8',newline='\n')
    files={}
    for p in sorted(ROOT.rglob('*')):
        rel=p.relative_to(ROOT)
        if not included(p):continue
        if rel.as_posix()=='evidence/bundle_manifest.json':continue
        if p.suffix in ('.pyc','.pyo'):continue
        files[rel.as_posix()]=sha(p)
    (ROOT/'evidence/bundle_manifest.json').write_text(json.dumps({'scope':'current curated private distribution','files':files},indent=2),encoding='utf-8',newline='\n')
    print(json.dumps({'status':'manifest_written','files':len(files)}))
if __name__=='__main__':main()
