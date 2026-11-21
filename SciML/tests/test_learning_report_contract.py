from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from SciML.evaluation.inspect_learning_report_contract import inspect_chapter, local_target


class ReportContractTests(unittest.TestCase):
    def test_math_and_code_are_not_links(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            chapter = root/"chapter.md"
            target = root/"source.md"
            target.write_text("source", encoding="utf-8")
            chapter.write_text("# 标题\n\\[\nI[T0](q)\n\\]\n\\(I[T0](q)\\)\n`[not](missing)`\n"
                "```python\nvalue = '[also_not](missing)'\n```\n[真实链接](source.md)\n", encoding="utf-8")
            result = inspect_chapter(chapter)
            self.assertEqual(len(result["links"]), 1)
            self.assertTrue(result["links"][0]["exists"])
            self.assertEqual(result["links"][0]["line"], 10)
            self.assertTrue(result["snippets"][0]["syntax_valid"])

    def test_broken_link_and_invalid_syntax_are_reported_not_hidden(self):
        with TemporaryDirectory() as folder:
            chapter = Path(folder)/"chapter.md"
            chapter.write_text("[missing](missing.md)\n```python\nif:\n```\n", encoding="utf-8")
            result = inspect_chapter(chapter)
            self.assertFalse(result["links"][0]["exists"])
            self.assertFalse(result["snippets"][0]["syntax_valid"])
            self.assertTrue(result["review_required"])

    def test_remote_not_fetched_and_relative_target_resolution(self):
        chapter = Path("C:/project/report/chapter.md")
        self.assertIsNone(local_target(chapter, "https://example.com/paper"))
        self.assertEqual(local_target(chapter, "#section"), chapter)
        self.assertEqual(local_target(chapter, "../code.py:7"), chapter.parent/"../code.py")


if __name__ == "__main__":
    unittest.main()
