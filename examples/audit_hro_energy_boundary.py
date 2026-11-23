"""Machine-check the information boundary for HRO field-to-energy claims.

The audit inspects existing snapshots and the archived field-energy report. It
does not infer missing energy terms from temperature values and does not call a
physical solver.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .multicycle_multistep import ROOT
from SciML.models.v12.multistep_historical import sha


DEFAULT = ROOT / 'evidence/local_studies/hro_energy_boundary_20261123/audit.json'
REQUIRED_TERMS = (
    'realized_signed_mass_flow',
    'pressure_command_trajectory',
    'port_temperature_ownership',
    'mechanical_work',
    'capacity_and_latent_restart_convention',
    'environment_heat',
    'internal_pipe_fluid_method',
    'independent_closure_residual',
)


def run(output=DEFAULT):
    output = Path(output).resolve()
    state_path = ROOT / 'data/historical12_state.npz'
    field_path = ROOT / 'data/historical12_training_fields.npz'
    state_meta = json.loads(state_path.with_suffix('.json').read_text())
    field_meta = json.loads(field_path.with_suffix('.json').read_text())
    with np.load(state_path, allow_pickle=False) as state:
        state_shapes = {key: list(state[key].shape) for key in state.files}
        state_keys = list(state.files)
    with np.load(field_path, allow_pickle=False) as fields:
        field_shapes = {key: list(fields[key].shape) for key in fields.files}
        field_keys = list(fields.files)
    archived = json.loads((ROOT / 'evidence/archive/field-energy-information-r1/summary.json').read_text())
    available = set(state_keys + field_keys)
    missing = list(REQUIRED_TERMS)
    result = {
        'status': 'completed_hro_energy_boundary_audit',
        'state_snapshot_sha256': sha(state_path),
        'field_snapshot_sha256': sha(field_path),
        'state_snapshot_provenance_sha256': state_meta['snapshot_sha256'],
        'field_snapshot_provenance_sha256': field_meta['snapshot_sha256'],
        'state_keys': state_keys,
        'state_shapes': state_shapes,
        'field_keys': field_keys,
        'field_shapes': field_shapes,
        'required_energy_terms': REQUIRED_TERMS,
        'missing_energy_terms': missing,
        'archived_information_gaps': archived['information_gaps'],
        'export_identity_residuals': archived['max_export_identity_abs_residuals'],
        'endpoint_port_gaps_K': archived['max_internal_endpoint_port_gaps_K'],
        'independent_closure_established': False,
        'autonomous_field_to_COP_claim': False,
        'fresh_solver_calls': 0,
        'interpretation': 'Temperature state and field snapshots support state/field prediction audits. They do not contain the signed flow, pressure, work, storage ownership and independent closure terms required for an autonomous field-to-energy claim.'
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8', newline='\n')
    return {'status': result['status'], 'missing_energy_terms': len(missing), 'fresh_solver_calls': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))
