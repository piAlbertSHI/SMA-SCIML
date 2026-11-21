"""Export a sealed historical HDF5 snapshot for the multi-step comparison.

Original raw MAT files may no longer be present. This exporter verifies the
sealed dataset and requests, and checks every derived target against the prior
saved training/test arrays. It does not claim a new raw-MAT provenance audit.
"""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from SciML.models.v12.learning_historical_rollout import metadata, odd, mask, split_cases
from SciML.models.v12.learning_boundary_state import update_mechanical, update_side
from SciML.models.v12.run_learning_existing import STRICT_COLUMNS
from SciML.evaluation.inspect_learning_state_contract import request_initial_state
from SciML.models.v11.state_space import STAGE_NAMES
from SciML.models.v11.spatiotemporal_operator import build_operator_corpus

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(dataset, campaign, output):
    if output.exists() or output.with_suffix('.json').exists():
        raise FileExistsError('Preserve prior portable snapshots')
    manifest = json.loads((dataset/'artifact_manifest.json').read_text())
    for item in manifest['artifacts']:
        if sha(dataset/item['path']) != item['sha256']:
            raise ValueError('Sealed dataset file mismatch: '+item['path'])
    table = pd.read_parquet(dataset/'case_index.parquet').sort_values('case_id')
    assert len(table) == 12 and sorted(table.cycle_count) == [2]*10+[3]*2
    for key in STRICT_COLUMNS:
        assert table[key].map(lambda v: isinstance(v, (bool, np.bool_)) and bool(v)).all(), key
    protocol = json.loads((ROOT/'evidence/archive/historical-rollout-r1/protocol.json').read_text())
    groups = split_cases(table); assert groups == protocol['groups']
    for key, expected in protocol['fingerprints'].items():
        assert table[key].unique().tolist() == [expected]
    case_ids = table.case_id.tolist(); initials=[]; controls=[]; records=[]; requests=[]
    with h5py.File(dataset/'fields.h5', 'r') as h:
        for ci, row in enumerate(table.itertuples(index=False)):
            source = Path(row.source_artifact)
            request_file = source.parent/'sciml_case_request.json'
            assert sha(request_file) == row.source_request_sha256
            if source.is_file(): assert sha(source) == row.source_artifact_sha256
            requests.append({'case_id': row.case_id, 'request_sha256': sha(request_file),
                             'historical_raw_sha256': row.source_artifact_sha256,
                             'raw_mat_available_for_recheck': source.is_file()})
            request=json.loads(request_file.read_text()); cycles=h[row.hdf5_group+'/cycles']
            state=np.r_[update_mechanical(request_initial_state(request), cycles['cycle_001/stage1_loading'], 0), np.zeros(101)]
            initials.append(state.copy());controls.append(metadata(request))
            def append(after, cycle, stage, sample_id):
                nonlocal state
                assert np.isfinite(after).all()
                np.testing.assert_array_equal(state[~mask(stage)], after[~mask(stage)])
                records.append((sample_id, ci, cycle, stage, state.copy(), after.copy()))
                state=after.copy()
            for cycle in range(1, int(row.cycle_count)+1):
                s1,s2,s3,s4=[cycles[f'cycle_{cycle:03}'][name] for name in STAGE_NAMES]
                if cycle>1:
                    after=state.copy();after[:414]=update_mechanical(state[:414],s1,0)
                    append(after,cycle-1,5,f'{row.case_id}:c{cycle-1}:restart')
                after=state.copy();after[:414]=update_mechanical(after[:414],s1,-1)
                after[102:153]=np.asarray(s2['sma_T_in_51_K']);after[204:305]=np.asarray(s2['htf_internal_T_K_t'])[0]
                after[:414]=update_side(after[:414],s2,0,2);after[414:]=odd(np.asarray(s2['sma_strip_T_K_t'])[0])
                append(after,cycle,1,f'{row.case_id}:c{cycle}:stage1')
                after=state.copy();after[:414]=update_mechanical(after[:414],s3,0)
                after[204:305]=np.asarray(s2['htf_internal_T_K_t'])[-1]
                after[:414]=update_side(after[:414],s2,1,2);after[414:]=odd(np.asarray(s2['sma_strip_T_K_t'])[-1])
                append(after,cycle,2,f'{row.case_id}:c{cycle}:stage2')
                after=state.copy();after[:414]=update_mechanical(after[:414],s3,-1)
                after[102:153]=np.asarray(s4['sma_T_in_51_K']);after[305:406]=np.asarray(s4['htf_internal_T_K_t'])[0]
                after[:414]=update_side(after[:414],s4,0,4);after[414:]=odd(np.asarray(s4['sma_strip_T_K_t'])[0])
                append(after,cycle,3,f'{row.case_id}:c{cycle}:stage3')
                after=state.copy();after[102:153]=np.asarray(s4['sma_T_out_51_K']);after[305:406]=np.asarray(s4['htf_internal_T_K_t'])[-1]
                after[:414]=update_side(after[:414],s4,1,4);after[414:]=odd(np.asarray(s4['sma_strip_T_K_t'])[-1])
                append(after,cycle,4,f'{row.case_id}:c{cycle}:stage4')
    arrays={'case_ids':np.array(case_ids),'initial':np.stack(initials),'control':np.stack(controls),
            'cycles':table.cycle_count.to_numpy(int),'split':np.array(['train' if c in groups['train'] else 'test' for c in case_ids]),
            'sample_ids':np.array([r[0] for r in records]),'case_index':np.array([r[1] for r in records]),
            'cycle':np.array([r[2] for r in records]),'stage':np.array([r[3] for r in records]),
            'before':np.stack([r[4] for r in records]),'after':np.stack([r[5] for r in records])}
    archived_prediction=campaign/'historical-rollout-r1/s20260910/predictions.npz'
    original_summary=json.loads((campaign/'historical-rollout-r1/summary.json').read_text())
    expected=original_summary['artifacts'].get('s20260910\\predictions.npz',original_summary['artifacts'].get('s20260910/predictions.npz'))
    assert sha(archived_prediction)==expected
    with np.load(archived_prediction,allow_pickle=False) as z:
        for split in ['train','test']:
            use=arrays['split'][arrays['case_index']]==split
            np.testing.assert_array_equal(arrays['sample_ids'][use],z[split+'_ids'])
            np.testing.assert_array_equal(arrays['after'][use],z[split+'_truth'])
    samples=[s for s in build_operator_corpus(dataset) if s.case_id in groups['test']]
    arrays.update(field_case_index=np.array([case_ids.index(s.case_id) for s in samples]),
                  field_cycle=np.array([s.cycle_index for s in samples]),field_stage=np.array([s.stage_index for s in samples]),
                  field_truth=np.stack([np.stack([s.sma_history,s.htf_history]) for s in samples]))
    output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(output,**arrays)
    result={'status':'verified_sealed_historical_snapshot','groups':groups,'fingerprints':protocol['fingerprints'],
            'state_transitions':len(records),'train_transitions':72,'test_transitions':46,'field_test_samples':len(samples),
            'dataset_manifest_sha256':sha(dataset/'artifact_manifest.json'),
            'dataset_artifacts':{v['path']:v['sha256'] for v in manifest['artifacts']},'requests':requests,
            'archived_prediction_sha256':sha(archived_prediction),'all_targets_equal_archived_predictions':True,
            'snapshot_sha256':sha(output),'source_scope':'sealed HDF5 and request hashes reverified; raw MAT availability recorded separately',
            'fresh_solver_calls':0}
    output.with_suffix('.json').write_text(json.dumps(result,indent=2),encoding='utf-8',newline='\n')
    return {k:result[k] for k in ['status','state_transitions','train_transitions','test_transitions','field_test_samples','all_targets_equal_archived_predictions','fresh_solver_calls']}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(json.dumps(export(a.dataset,a.campaign,a.output),indent=2))
