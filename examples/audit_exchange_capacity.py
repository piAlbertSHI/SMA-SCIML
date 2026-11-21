"""Read the sealed Simulink archive to distinguish dynamic and bridge weights."""
import argparse
import json
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET

from .multicycle_multistep import ROOT,write
from .train_hierarchical import training_data
from SciML.models.v12.multistep_historical import sha
from SciML.models.v13.relaxation_readout import RelaxationMeanReadout,VolumeRelaxationMeanReadout


def run(output,legacy_root):
    output.mkdir(parents=True,exist_ok=False)
    binding=json.loads((ROOT/'evidence/hro_legacy_physics.json').read_text())['source_bindings']
    model=legacy_root/'nixunhuan_solver101_internal.slx';assert sha(model)==binding[model.name]['sha256']
    params=legacy_root/'Build_Simulink_Exchange_Params.m';assert sha(params)==binding[params.name]['sha256']
    masses=[];pipes=[];members={}
    with zipfile.ZipFile(model) as z:
        for name in z.namelist():
            if not name.startswith('simulink/systems/') or not name.endswith('.xml'):continue
            raw=z.read(name);xml=ET.fromstring(raw)
            for block in xml.iter('Block'):
                props={e.get('Name'):e.text for e in block.iter('P')}
                if props.get('mass')=='solver51_node_mass':
                    masses.append({'member':name,'name':block.get('Name'),'sid':block.get('SID'),
                        **{k:props[k] for k in ('mass','mass_unit','sp_heat','sp_heat_unit')}})
                if block.get('Name','').startswith(('HoldPipe_','CooldownPipe_')):
                    pipes.append({'member':name,'name':block.get('Name'),
                        **{k:props.get(k) for k in ('length','area','Dh','length_add','Nu_lam')}})
            if any(r['member']==name for r in masses+pipes):
                import hashlib
                members[name]=hashlib.sha256(raw).hexdigest()
    assert len(masses)==101 and all(m['sp_heat']=='solver51_node_cp' for m in masses)
    text=params.read_text(encoding='utf-8')
    assert 'sim_params.solver51_node_mass = total_mass / thermal_model_node_count;' in text
    assert 'sim_params.solver51_node_cp = node_cp;' in text
    arrays,ids,hashes=training_data();rates=[]
    for i,case in enumerate(ids):
        field=arrays['fields'][i,[0,2]];control=arrays['control'][i:i+1].repeat(2,1)
        fits={}
        for kind,cls in (('thickness',RelaxationMeanReadout),('volume',VolumeRelaxationMeanReadout)):
            head=cls();fits[kind]=head.fit(field[:,0,0].mean(-1),field[:,1].mean(-1),control,field[:,0].mean(-1))
        rates.append({'case_id':case,'radius_m':float(control[0,0]),'fits':fits})
    result={'status':'verified_sealed_dynamic_capacity_layout','model_sha256':sha(model),'parameter_source_sha256':sha(params),
        'xml_member_sha256':members,'equal_mass_sma_nodes':len(masses),'exchange_pipe_blocks':len(pipes),
        'dynamic_node_mass':'total_mass / thermal_model_node_count','dynamic_node_cp':'node_cp',
        'dynamic_capacity_mean':'arithmetic mean of 101 equal-mass, equal-cp nodes',
        'bridge_capacity_mean':'separate chord-area/annular weights in compression and state reconstruction',
        'capacity_claim':'the verified bridge increment identity is not a complete Simscape heat balance',
        'masses':masses,'pipes':pipes,'diagnostic_observed_response_fits':rates,'data_sha256':hashes,
        'per_case_fit_scope':'diagnostic only; never deployed as case-specific prediction parameters',
        'source_sha256':sha(Path(__file__)),'fresh_solver_calls':0}
    write(output/'analysis.json',result)
    return {k:result[k] for k in ('status','equal_mass_sma_nodes','exchange_pipe_blocks','dynamic_capacity_mean','capacity_claim')}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--legacy-root',required=True,type=Path);a=p.parse_args()
    print(json.dumps(run(a.output,a.legacy_root),indent=2))
