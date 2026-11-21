"""Export only existing historical training fields for joint architecture checks."""
import argparse
import json
from pathlib import Path

import numpy as np

from SciML.models.v11.spatiotemporal_operator import build_operator_corpus
from SciML.models.v12.multistep_historical import sha
from SciML.models.v13.hierarchical_operator import HierarchicalStateOperator
from .multicycle_multistep import data, write
import torch


def export(dataset,output):
    if output.exists() or output.with_suffix('.json').exists():raise FileExistsError('Preserve previous snapshots')
    d,source=data()
    manifest=json.loads((dataset/'artifact_manifest.json').read_text())
    assert sha(dataset/'artifact_manifest.json')==source['dataset_manifest_sha256']
    for artifact in manifest['artifacts']:
        assert sha(dataset/artifact['path'])==artifact['sha256']
    samples=build_operator_corpus(dataset)
    train_ids=source['groups']['train']
    selected=[s for s in samples if s.case_id in train_ids]
    assert len(selected)==32 and {s.case_id for s in selected}==set(train_ids)
    # Cross-check the common test fields against the previously sealed portable snapshot.
    for s in samples:
        if s.case_id in source['groups']['test']:
            case_index=d['case_ids'].tolist().index(s.case_id)
            use=(d['field_case_index']==case_index)&(d['field_cycle']==s.cycle_index)&(d['field_stage']==s.stage_index)
            assert use.sum()==1
            np.testing.assert_array_equal(d['field_truth'][use][0],np.stack((s.sma_history,s.htf_history)))
    maximum=0.
    for s in selected:
        case_index=d['case_ids'].tolist().index(s.case_id)
        use=(d['case_index']==case_index)&(d['cycle']==s.cycle_index)&(d['stage']==s.stage_index)
        reconstructed=HierarchicalStateOperator.initial_fields(torch.tensor(d['before'][use]),s.stage_index)[0].numpy()
        observed=np.stack((s.sma_history[0],s.htf_history[0]))
        maximum=max(maximum,float(np.abs(observed-reconstructed).max()))
        np.testing.assert_allclose(reconstructed,observed,atol=1e-9,rtol=0)
    np.savez_compressed(output,case_ids=np.array([s.case_id for s in selected]),
        cycle=np.array([s.cycle_index for s in selected]),stage=np.array([s.stage_index for s in selected]),
        truth=np.stack([np.stack((s.sma_history,s.htf_history)) for s in selected]))
    report={'status':'verified_training_fields_only','case_count':8,'field_samples':32,
        'train_case_ids':train_ids,'test_labels_exported':False,'initial_reconstruction_max_abs_K':maximum,
        'snapshot_sha256':sha(output),'source_dataset_manifest_sha256':source['dataset_manifest_sha256'],
        'source_hdf5_sha256':sha(dataset/'fields.h5'),'state_snapshot_sha256':source['snapshot_sha256'],
        'raw_mat_reaudit':False,'fresh_solver_calls':0}
    write(output.with_suffix('.json'),report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(json.dumps(export(a.dataset,a.output),indent=2))
