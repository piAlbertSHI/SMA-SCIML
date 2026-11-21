"""Export final checkpoints and audit records; retain optimizer bundles locally."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import torch


def tensor_digest(weights):
    digest = hashlib.sha256()
    for name, value in sorted(weights.items()):
        if name.startswith('decoder.'): continue
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode()); digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode()); digest.update(array.tobytes())
    return digest.hexdigest()


def export(source, output, formal=False):
    if output.exists(): raise FileExistsError('Preserve released evidence')
    summary = json.loads((source/'summary.json').read_text())
    for rel, digest in summary['artifacts'].items():
        if hashlib.sha256((source/rel).read_bytes()).hexdigest() != digest: raise ValueError('Source changed: '+rel)
    output.mkdir(parents=True)
    omitted = {}; audits = {}
    for rel in [*summary['artifacts'], 'summary.json']:
        keep = not rel.endswith('.pt') or (formal and rel.endswith('/refined/model.pt'))
        if not keep:
            omitted[rel] = {'sha256': summary['artifacts'][rel], 'reason': 'retained in complete local experiment bundle'}
            continue
        target = output/rel; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source/rel, target)
    if formal:
        for seed in [20261121, 20261122, 20261123]:
            a = torch.load(source/f's{seed}/joint/model.pt', weights_only=True)
            b = torch.load(source/f's{seed}/refined/model.pt', weights_only=True)
            original = tensor_digest(a['weights']); refined = tensor_digest(b['weights'])
            if original != refined: raise ValueError('Decoder refinement changed state backbone')
            audits[str(seed)] = {'joint_backbone_sha256': original, 'refined_backbone_sha256': refined,
                'joint_checkpoint_sha256': summary['artifacts'][f's{seed}/joint/model.pt']}
    record = {'status': 'portable_evidence_export', 'original_summary_sha256': hashlib.sha256((source/'summary.json').read_bytes()).hexdigest(),
              'omitted_local_artifacts': omitted, 'frozen_backbone_audit': audits,
              'scope': 'Final checkpoints, logs, predictions and source snapshots are portable. Intermediate checkpoints and optimizer/RNG bundles remain local; this export alone is not a training-resume bundle.'}
    (output/'export_manifest.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    return {'files': sum(p.is_file() for p in output.rglob('*')), 'omitted_local_files': len(omitted)}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--formal', action='store_true'); a = p.parse_args()
    print(json.dumps(export(a.source, a.output, a.formal)))
