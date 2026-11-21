"""Replay modal HRO evidence and audit frozen-state decoder refinement."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .multicycle_multistep import ROOT, data
from .train_hierarchical import training_data
from .formal_hierarchical import infer, cohorts, aggregate
from .modal_hierarchical import SEEDS
from SciML.models.v12.multistep_historical import sha
from SciML.models.v13.modal_operator import ModalStateOperator
from tools.export_modal_evidence import tensor_digest

DEFAULT = ROOT/'evidence/experiments/hro_modal/formal'


def verify(output):
    summary = json.loads((output/'summary.json').read_text())
    protocol = json.loads((output/'protocol.json').read_text())
    _, ids, hashes = training_data()
    assert protocol['data_sha256'] == hashes and protocol['fit_case_ids'] == ids
    export_path = output/'export_manifest.json'
    export = json.loads(export_path.read_text()) if export_path.exists() else None
    if export: assert sha(output/'summary.json') == export['original_summary_sha256']
    for rel, digest in summary['artifacts'].items():
        if (output/rel).exists(): assert sha(output/rel) == digest, rel
        else:
            assert export and export['omitted_local_artifacts'][rel]['sha256'] == digest, rel
            assert rel.endswith('.pt'), rel
    for rel, digest in protocol['source_sha256'].items(): assert sha(output/'source'/rel) == digest, rel
    assert sha(output/'source/examples/refine_modal.py') == protocol['decoder_source_sha256']
    d, _ = data(); rows = []; maximum = 0.
    for seed in SEEDS:
        dest = output/f's{seed}'
        model = ModalStateOperator.load(dest/'refined/model.pt')
        if (dest/'joint/model.pt').exists():
            joint = ModalStateOperator.load(dest/'joint/model.pt')
            assert model.training_ids == joint.training_ids
            assert tensor_digest(joint.state_dict()) == tensor_digest(model.state_dict())
        else:
            audit = export['frozen_backbone_audit'][str(seed)]
            assert audit['joint_checkpoint_sha256'] == summary['artifacts'][f's{seed}/joint/model.pt']
            assert audit['joint_backbone_sha256'] == audit['refined_backbone_sha256'] == tensor_digest(model.state_dict())
        assert {s.split(':')[0] for s in model.training_ids} == set(ids)
        history = json.loads((dest/'joint/history.json').read_text())
        assert len(history) == 200 and all(set(r['case_ids']) <= set(ids) for r in history)
        refined = json.loads((dest/'refined/history.json').read_text())
        assert len(refined) == 320 and all(all(0 <= i < 32 for i in r['cached_training_rows']) for r in refined)
        prediction = infer(model, d)
        with np.load(dest/'test_predictions.npz', allow_pickle=False) as saved:
            for name, value in prediction.items():
                maximum = max(maximum, float(np.abs(value-saved[name]).max()))
                np.testing.assert_allclose(value, saved[name], rtol=0, atol=1e-8)
            rows.append(cohorts({k: saved[k] for k in saved.files}, d))
        report = json.loads((dest/'evaluation.json').read_text())
        for key, value in rows[-1].items(): assert value == report[key]
    assert aggregate(rows, SEEDS) == summary['modal']
    return {'status': 'verified_two_phase_modal_evidence', 'checkpoint_pairs': 3,
            'test_cases_per_checkpoint': 4, 'maximum_replay_difference': maximum,
            'state_backbone_exactly_unchanged_by_refinement': True, 'training_ids_checked': True,
            'intermediate_checkpoint_audit': 'recorded exact tensor hashes; intermediate checkpoints local' if export else 'both checkpoints compared directly',
            'original_training_sources_verified': True, 'fresh_solver_calls': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT); args = parser.parse_args()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1): result = verify(args.output)
    print(json.dumps(result, indent=2))
