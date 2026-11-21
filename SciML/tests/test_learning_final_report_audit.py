from pathlib import Path
import tempfile
import unittest

from SciML.evaluation.audit_learning_final_report import requirement_ids, require, verify_digests
from SciML.models.v12.run_learning_existing import sha


class FinalReportAuditTests(unittest.TestCase):
    def test_all_sixteen_requirements_exactly_once(self):
        body = '\n'.join(f'| R{i:02d} | 满足学习交付 | evidence |' for i in range(16))
        self.assertEqual(len(requirement_ids(body)), 16)
        for altered in (body+'\n| R00 | 满足学习交付 | duplicate |', body.replace('R15', 'R16'), body.replace('满足学习交付', '待核验', 1)):
            with self.assertRaises(ValueError):
                requirement_ids(altered)

    def test_digest_binding_detects_changed_content(self):
        with tempfile.TemporaryDirectory(prefix='report_hash_test_') as directory:
            root = Path(directory)
            path = root/'source.txt'
            path.write_text('first', encoding='utf-8')
            mapping = {'source.txt': sha(path)}
            verify_digests(root, mapping)
            path.write_text('changed', encoding='utf-8')
            with self.assertRaises(ValueError):
                verify_digests(root, mapping)

    def test_empty_evidence_is_not_pass(self):
        with self.assertRaises(ValueError):
            verify_digests(Path('.'), {})

    def test_required_checks_do_not_depend_on_python_assert(self):
        with self.assertRaises(ValueError):
            require(False, 'must fail even under optimization')
        require(True, 'valid')


if __name__ == '__main__':
    unittest.main()
