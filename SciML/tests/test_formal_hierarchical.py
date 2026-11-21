"""Evaluate row alignment and the target-free boundary of the benchmark adapter."""
import unittest
import numpy as np
import torch
from examples.formal_hierarchical import infer, data


class EchoOperator:
    def eval(self):return self

    def __call__(self, initial, control, cycles, *, times, positions):
        keys = [(c-1 if s==5 else c,s) for c in range(1,cycles+1)
                for s in (([5] if c>1 else [])+[1,2,3,4])]
        field_keys = [(c,s) for c,s in keys if s in (2,4)]
        return {'keys':keys,'field_keys':field_keys,
            'states':torch.stack([initial*0+10*c+s for c,s in keys],1),
            'fields':torch.stack([torch.full((1,2,len(times),len(positions)),10*c+s,dtype=torch.float64)
                                  for c,s in field_keys],1)}


class TestBenchmarkBoundary(unittest.TestCase):
    def setUp(self):
        source,_ = data()
        # No before/after states, target fields or sample labels are accessible.
        self.d = {k:source[k].copy() for k in ('initial','control','cycles','split','case_index','cycle','stage',
                                               'field_case_index','field_cycle','field_stage')}

    def test_target_free_and_keyed_restart_alignment(self):
        d = self.d
        for keys in [('case_index','cycle','stage'),('field_case_index','field_cycle','field_stage')]:
            for k in keys:d[k] = d[k][::-1]
        result = infer(EchoOperator(),d)
        selected = d['split'][d['case_index']]=='test'
        np.testing.assert_array_equal(result['state'][:,0],(10*d['cycle']+d['stage'])[selected])
        np.testing.assert_array_equal(result['field'][:,0,0,0],10*d['field_cycle']+d['field_stage'])
        self.assertEqual(result['state'].shape,(46,515))
        self.assertEqual(result['field'].shape,(20,2,50,101))

    def test_missing_exchange_metadata_is_rejected(self):
        for k in ('field_case_index','field_cycle','field_stage'):self.d[k] = self.d[k][1:]
        with self.assertRaises(AssertionError):infer(EchoOperator(),self.d)


if __name__=='__main__':unittest.main()
