"""Replay the full v13 checkpoint and verify the engineering execution evidence."""
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .multicycle_multistep import ROOT
from SciML.models.v12.multistep_historical import sha
from SciML.models.v13.hierarchical_operator import HierarchicalStateOperator


def verify():
    root=ROOT/'evidence/architecture/v13_execution_v1'
    summary=json.loads((root/'summary.json').read_text())
    for rel,expected in summary['artifacts'].items():
        if sha(root/rel)!=expected:raise ValueError('Execution artifact changed: '+rel)
    torch.set_num_threads(1)
    with threadpool_limits(limits=1),torch.no_grad(),np.load(root/'execution_arrays.npz',allow_pickle=False) as z:
        model=HierarchicalStateOperator.load(root/'model.pt')
        result=model(torch.tensor(z['initial']),torch.tensor(z['control']),3,times=torch.tensor(z['times']),positions=torch.tensor(z['positions']))
        maximum=max(float(np.abs(result[k].numpy()-z[k]).max()) for k in ('states','fields'))
        for k in ('states','fields'):np.testing.assert_allclose(result[k],z[k],rtol=0,atol=1e-8)
        assert model.parameter_report()==summary['parameter_report']
        assert all(summary['major_module_gradient_coverage'].values())
    return {'status':'verified_v13_architecture_execution','trainable_parameters':summary['parameter_report']['trainable'],
        'max_replay_absolute_difference':maximum,'replay_absolute_tolerance':1e-8,'accuracy_evaluated':False,'fresh_solver_calls':0}


if __name__=='__main__':print(json.dumps(verify(),indent=2))
