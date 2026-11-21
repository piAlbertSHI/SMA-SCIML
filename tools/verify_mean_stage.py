"""Record tests and evidence replay for the mean-temperature development stage."""
import argparse
import json
from pathlib import Path
import sys
import time
import unittest

import torch
from threadpoolctl import threadpool_limits

from examples.multicycle_multistep import ROOT, write
from examples.verify_mean_development import run as verify_means
from examples.verify_increment_consistency import run as verify_increment
from SciML.models.v12.multistep_historical import sha

TEST_MODULES=[f'SciML.tests.test_{name}' for name in (
    'learning_coordinate_operator','learning_ensemble','learning_pinn_inverse','learning_pool_loop',
    'learning_design_loop','learning_exercises','acquisition_diversity','multistep_historical',
    'hierarchical_operator','formal_hierarchical','modal_operator','selective_sma','thermal_readout',
    'stage_thermal_readout','mean_thermal_readout','thermal_geometry')]


def run(output):
    output.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    with (output/'unittest.log').open('x',encoding='utf-8',newline='\n') as log:
        suite=unittest.defaultTestLoader.loadTestsFromNames(TEST_MODULES)
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    tests={'modules':TEST_MODULES,'tests_run':result.testsRun,'failures':len(result.failures),
           'errors':len(result.errors),'skipped':len(result.skipped),'passed':result.wasSuccessful()}
    write(output/'tests.json',tests)
    if not result.wasSuccessful():return {'status':'failed_tests','tests':tests}
    means=verify_means(ROOT/'evidence/local_studies/sma_mean_readout');write(output/'mean_replay.json',means)
    increment=verify_increment();write(output/'increment_replay.json',increment)
    baseline_path=ROOT/'evidence/local_studies/sma_mean_readout/retained_baseline_verification.json'
    baseline=json.loads(baseline_path.read_text())
    assert baseline['max_replay_difference']==0 and baseline['stage4_state_fluid_initial_exact']
    result={'status':'verified_mean_development_stage','goal_status':'active','candidate_promoted':False,
        'numerical_objective_achieved':False,'tests':tests,
        'replay':{k:v for k,v in means.items() if k!='status'},
        'physical_increment_identity_max_abs_K':increment['max_increment_identity_error_K'],
        'baseline_verification_sha256':sha(baseline_path),'seconds':time.perf_counter()-started,
        'remaining_work':'Reduce stage2 mean bias while retaining third-cycle and segment-level improvements; '
            'investigate calibrated increment conditioning or soft consistency using training cases only.',
        'fresh_solver_calls':0,'neural_training_updates':0,'github_pushes':0,
        'artifacts':{p.name:sha(p) for p in output.iterdir() if p.is_file()}}
    write(output/'summary.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):result=run(a.output)
    print(json.dumps(result,indent=2));sys.exit(0 if result['status']=='verified_mean_development_stage' else 1)
