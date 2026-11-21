"""Recover an isolated historical solver bundle without running MATLAB.

The source root is an explicit local argument. Original requests are preserved
locally; the portable audit contains relative paths, hashes and parameter values.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root, output, report):
    if output.exists() or report.exists(): raise FileExistsError('Preserve previous audit and recovery bundles')
    package = root/'Transfer/gx101r1_sciml_paper220_remote_20260822_final'
    table = pd.read_parquet(root/'SciML/datasets/v1.2-confirm-temporal12/case_index.parquet')
    bindings = {}; materials = {}; requests = []; sources = {}
    material_keys = ['E_A','E_M','nu','H_max','rho','c_A','c_M','k_A','k_M','alpha_A','alpha_M',
                     'ds0','hc','Hfor','M_s','M_f','A_s','A_f']
    for row in table.itertuples(index=False):
        request = Path(row.source_artifact).parent/'sciml_case_request.json'
        if sha(request) != row.source_request_sha256: raise ValueError('Changed historical request')
        value = json.loads(request.read_text())
        requests.append({'case_id': row.case_id, 'request_sha256': sha(request),
                         'raw_mat_available': Path(row.source_artifact).is_file(),
                         'case_spec': value['case_spec']})
        for item in value['implementation_fingerprint']:
            relative = Path(item['path']).relative_to(root)
            rel = relative.as_posix()
            if rel in bindings and bindings[rel] != item['sha256']: raise ValueError('Mixed implementation profiles')
            bindings[rel] = item['sha256']
            choices = [package/relative, root/relative]
            matching = [p for p in choices if p.is_file() and sha(p) == item['sha256']]
            if not matching: raise FileNotFoundError('No exact historical source: '+rel)
            sources[rel] = matching[0]
        solver = value['run_options']['solver_options']
        params = {k: solver[k] for k in material_keys}
        material = row.material_id
        if material in materials and materials[material] != params: raise ValueError('Within-profile parameter drift')
        materials[material] = params
    output.mkdir(parents=True)
    shutil.copytree(package, output/'runtime')
    for rel, source in sources.items():
        target = output/'runtime'/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if sha(target) != bindings[rel]: raise ValueError('Recovery hash mismatch')
    (output/'requests').mkdir()
    for row in table.itertuples(index=False):
        shutil.copy2(Path(row.source_artifact).parent/'sciml_case_request.json', output/'requests'/f'{row.case_id}.json')
    result = {'status': 'historical_instrumented_sources_and_parameters_recovered', 'fresh_solver_calls': 0,
        'request_count': len(requests), 'matched_instrumented_files': len(bindings),
        'physics_fingerprint': str(table.physics_implementation_fingerprint_sha256.iloc[0]),
        'pipeline_fingerprint': str(table.pipeline_implementation_fingerprint_sha256.iloc[0]),
        'source_bindings': {rel: {'sha256': digest, 'recovered_from': sources[rel].relative_to(root).as_posix()}
                            for rel, digest in bindings.items()},
        'material_parameters_SI': materials, 'cases': requests,
        'dependency_package': package.relative_to(root).as_posix(),
        'runtime_executed': False,
        'limitations': ['Nine instrumented sources match original request hashes; other packaged dependencies lack per-case historical hashes.',
                       'Original absolute output/cache paths require an isolated workspace override before any run.',
                       'A one-case runtime/dependency check and cost estimate are still required before data expansion.',
                       'No new simulation data may be merged merely because the material names match.'],
        'bundle_files': {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob('*')) if p.is_file()}}
    text = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n'
    (output/'recovery_manifest.json').write_text(text, encoding='utf-8')
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(text, encoding='utf-8')
    return {k: result[k] for k in ('status','fresh_solver_calls','request_count','matched_instrumented_files','material_parameters_SI')}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--physics-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(audit(a.physics_root.resolve(), a.output.resolve(), a.report.resolve()), indent=2))
