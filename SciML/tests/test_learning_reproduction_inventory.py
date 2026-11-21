import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from SciML.evaluation.learning_reproduction_inventory import (
    parse_recipe, read_help, verify_help_options,
)
from SciML.evaluation.audit_learning_wp5 import verify_artifacts
from SciML.models.v12.run_learning_existing import write_json, sha


class ReproductionInventoryTests(unittest.TestCase):
    def test_artifact_reader_uses_writer_utf8_with_chinese_names(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            payload = root/"中文练习.json"
            write_json(payload, {"答案": "有限区间不可得"})
            expected = {"status": "参考答案", "artifacts": {payload.name: sha(payload)}}
            write_json(root/"summary.json", expected)
            original = Path.read_text

            def enforce_explicit_utf8(path, *args, **kwargs):
                if path == root/"summary.json":
                    self.assertEqual(kwargs.get("encoding"), "utf-8")
                return original(path, *args, **kwargs)

            with patch.object(Path, "read_text", enforce_explicit_utf8):
                self.assertEqual(verify_artifacts(root), expected)

    def test_unicode_fix_does_not_relax_digest_check(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            write_json(root/"中文练习.json", {"value": 1})
            write_json(root/"summary.json", {"artifacts": {"中文练习.json": "0"*64}})
            with self.assertRaisesRegex(ValueError, "changed artifact"):
                verify_artifacts(root)

    def test_module_options_quotes_and_flag(self):
        module, options = parse_recipe('python -u -m SciML.evaluation.example --dataset-root "a b/data" --seed=-7 --check-help')
        self.assertEqual(module, "SciML.evaluation.example")
        self.assertEqual(options, {"--dataset-root": "a b/data", "--seed": "-7", "--check-help": None})

    def test_bad_module_recipe_and_duplicate_rejected(self):
        for command in ("", "python file.py", "python -m", "python -m --bad", "python -m os", "python -m SciML.evaluation.example --seed 1 --seed 2"):
            with self.subTest(command=command), self.assertRaises(ValueError):
                parse_recipe(command)

    def test_help_uses_exact_options(self):
        verify_help_options({"--seed": "1"}, "usage: example [--seed SEED]")
        with self.assertRaises(ValueError):
            verify_help_options({"--seed": "1"}, "usage: example [--seeds SEEDS]")

    def test_read_help_only_and_restores_argv(self):
        previous = sys.argv

        def fake_module(module, run_name):
            self.assertEqual(sys.argv, ["SciML.evaluation.example", "--help"])
            self.assertEqual(run_name, "__main__")
            print("usage: example [--seed SEED]")
            raise SystemExit(0)

        with patch("SciML.evaluation.learning_reproduction_inventory.runpy.run_module", side_effect=fake_module) as runner:
            self.assertIn("--seed", read_help("SciML.evaluation.example")["stdout"])
            runner.assert_called_once()
        self.assertIs(sys.argv, previous)

    def test_help_failure_and_unittest_never_run(self):
        previous = sys.argv
        with patch("SciML.evaluation.learning_reproduction_inventory.runpy.run_module", side_effect=SystemExit(2)):
            with self.assertRaises(RuntimeError):
                read_help("SciML.evaluation.example")
        self.assertIs(sys.argv, previous)
        with patch("SciML.evaluation.learning_reproduction_inventory.runpy.run_module") as runner:
            with self.assertRaises(ValueError):
                read_help("unittest")
            runner.assert_not_called()
            with self.assertRaises(RuntimeError):
                read_help("SciML.evaluation.example")
        self.assertIs(sys.argv, previous)


if __name__ == "__main__":
    unittest.main()
