"""Inventory report links, fenced code, section structure and source bindings.

An inventory is not a readability or scientific-completeness verdict. Python
snippets are parsed, not executed; commands and links are never shell-evaluated.
"""
import argparse
import ast
import json
from pathlib import Path
import re
from urllib.parse import unquote

from SciML.models.v12.run_learning_existing import sha, write_json


def local_target(chapter, target):
    target = target.strip()
    if re.match(r"^[a-zA-Z][\w+.-]*://", target) or target.startswith("mailto:"):
        return None
    if target.startswith("#"):
        return chapter
    target = unquote(target.split("#", 1)[0])
    target = re.sub(r":\d+$", "", target)
    if re.match(r"^/[A-Za-z]:/", target):
        target = target[1:]
    path = Path(target)
    return path if path.is_absolute() else chapter.parent/path


def inspect_chapter(chapter):
    body = chapter.read_text(encoding="utf-8")
    # Preserve character/line offsets while masking code and LaTeX. Expressions
    # such as I[T0](q) inside math are not Markdown links.
    prose = body
    for pattern in (r"^```[^\n]*\n.*?^```[ \t]*$", r"`+[^`]*`+", r"\\\[.*?\\\]", r"\\\(.*?\\\)"):
        prose = re.sub(pattern, lambda match: "".join("\n" if c == "\n" else " " for c in match.group()), prose, flags=re.M | re.S)
    links = []
    for match in re.finditer(r"!?\[([^\]\n]*)\]\((?:<([^>]+)>|([^\)]+))\)", prose):
        target = match.group(2) or match.group(3)
        local = local_target(chapter, target)
        links.append({"label": match.group(1), "target": target, "local": local is not None,
            "exists": local.is_file() if local is not None else None,
            "resolved": str(local.resolve()) if local is not None else None,
            "sha256": sha(local) if local is not None and local.is_file() else None,
            "line": body[:match.start()].count("\n")+1})
    snippets = []
    for match in re.finditer(r"^```([^\n]*)\n(.*?)^```[ \t]*$", body, re.M | re.S):
        language, code = match.group(1).strip(), match.group(2)
        item = {"language": language, "line": body[:match.start()].count("\n")+1,
            "line_count": len(code.splitlines()), "executed": False,
            "scope": "syntax/structure inventory only; pedagogical fragments may need chapter context"}
        if language == "python":
            try:
                ast.parse(code)
            except SyntaxError as error:
                item["syntax_valid"] = False
                item["syntax_error"] = str(error)
            else:
                item["syntax_valid"] = True
        snippets.append(item)
    return {"chapter": chapter.name, "sha256": sha(chapter), "characters": len(body),
        "line_count": len(body.splitlines()), "headings": re.findall(r"^#{1,6} .+$", body, re.M),
        "fence_markers": len(re.findall(r"^```", body, re.M)), "links": links, "snippets": snippets,
        "python_module_command_count": len(re.findall(r"^python .* -m |^python -m ", body, re.M)),
        "review_required": True}


def run(report_root, out):
    if out.exists():
        raise FileExistsError("preserve prior report structural inventory")
    chapters = [inspect_chapter(p) for p in sorted(report_root.glob("*.md"))]
    if not chapters:
        raise ValueError("no report chapters")
    bad_links = [{"chapter": c["chapter"], **link} for c in chapters for link in c["links"] if link["local"] and not link["exists"]]
    invalid_python = [{"chapter": c["chapter"], **snippet} for c in chapters for snippet in c["snippets"] if snippet.get("syntax_valid") is False]
    odd_fences = [c["chapter"] for c in chapters if c["fence_markers"] % 2]
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/"chapter_inventory.json", chapters)
    summary = {"status": "report_structure_inventory_requires_semantic_review", "whole_report_complete": False,
        "chapter_files": len(chapters), "characters": sum(c["characters"] for c in chapters),
        "local_links": sum(link["local"] for c in chapters for link in c["links"]),
        "remote_links_not_fetched": sum(not link["local"] for c in chapters for link in c["links"]),
        "fenced_code_blocks": sum(len(c["snippets"]) for c in chapters),
        "python_blocks_parsed": sum(s["language"] == "python" for c in chapters for s in c["snippets"]),
        "broken_local_links": bad_links, "python_syntax_errors": invalid_python, "odd_fence_chapters": odd_fences,
        "semantic_scope": "links and syntax cannot prove formulas, scientific assertions, numerical tables, readability or learner mastery",
        "implementation_sha256": sha(Path(__file__)), "artifacts": {"chapter_inventory.json": sha(out/"chapter_inventory.json")}}
    write_json(out/"summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.report_root, args.output_root), indent=2))
