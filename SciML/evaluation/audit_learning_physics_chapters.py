"""Independent observed-port quadratures and narrow chapters07A1/A2 audit."""
import argparse
import ast
import json
from pathlib import Path
import re

import h5py
import numpy as np
import pandas as pd

from .audit_learning_operator_chapter import table_cells
from .audit_learning_wp5 import verify_artifacts
from SciML.models.v12.run_learning_existing import sha, write_json


def run(dataset, campaign, report_root, out):
    if out.exists():
        raise FileExistsError("preserve physics teaching review")
    energy = verify_artifacts(campaign/"field-energy-information-r1")
    physics = verify_artifacts(campaign/"staged-physics-r1")
    derivative = verify_artifacts(campaign/"staged-physics-audit-r1")
    for path, digest in energy["sources_sha256"].items():
        if sha(Path(path)) != digest:
            raise ValueError("readout source changed")
    manifest = json.loads((dataset/"artifact_manifest.json").read_text())
    for entry in manifest["artifacts"]:
        if sha(dataset/entry["path"]) != entry["sha256"]:
            raise ValueError("source dataset changed")
    table = pd.read_parquet(dataset/"case_index.parquet")
    stage_rows = json.loads((campaign/"field-energy-information-r1/stage_quadrature_diagnostics.json").read_text())
    stage_rows = {(r["case_id"], r["stage"]): r for r in stage_rows}
    differences, gaps = [], []
    with h5py.File(dataset/"fields.h5", "r") as handle:
        for row in table.itertuples(index=False):
            request_file = Path(row.source_artifact).parent/"sciml_case_request.json"
            if sha(request_file) != row.source_request_sha256 or sha(Path(row.source_artifact)) != row.source_artifact_sha256:
                raise ValueError("readout raw source changed")
            request = json.loads(request_file.read_text())
            opts = request["run_options"]["exchange_options"]
            cp, rho = opts.get("cp_water", 4180.), opts.get("rho_water", 998.)
            cycle = handle[row.hdf5_group+"/cycles/cycle_001"]
            for stage, name in ((2, "stage2_hold_exchange"), (4, "stage4_cooldown_exchange")):
                g = cycle[name]
                t = np.asarray(g["time_local_s"])
                m = np.asarray(g["mass_flow_t_kgps"])
                tin, tout = np.asarray(g["htf_T_in_t_K"]), np.asarray(g["htf_T_out_t_K"])
                pressure = np.asarray(g["pressure_pi_command_MPa_t"])
                heat_integrand = m*cp*(tin-tout)
                work_integrand = np.abs(pressure*1e6*m/rho)
                # Explicit trapezoid sum: no production integration helper.
                heat = float(np.sum(np.diff(t)*(heat_integrand[:-1]+heat_integrand[1:])/2))
                work = float(np.sum(np.diff(t)*(work_integrand[:-1]+work_integrand[1:])/2))
                stored = stage_rows[(row.case_id, stage)]
                np.testing.assert_allclose([heat, work], [stored["observed_port_heat_50point_J"], stored["observed_pump_work_50point_J"]], atol=1e-14, rtol=1e-14)
                qtrue = float(g["energy/Q_sma_htf_J"][()])
                wtrue = float(cycle[f"performance/Wpump_stage{stage}_J"][()])
                differences.append([heat-qtrue, work-wtrue])
                internal = np.asarray(g["htf_internal_T_K_t"])
                values = [float(np.max(np.abs(a-b))) for a, b in ((internal[:, 0], tin), (internal[:, -1], tout), (internal[:, -1], tin), (internal[:, 0], tout))]
                np.testing.assert_array_equal(values, list(stored["internal_endpoint_minus_port_max_K"].values()))
                gaps.append(values)
    differences, gaps = np.array(differences), np.array(gaps)
    np.testing.assert_allclose(np.abs(differences).max(axis=0),
        [v["max_abs_J"] for v in energy["observed_quadrature_diagnostic"].values()], atol=1e-14, rtol=1e-14)
    chapter1 = report_root/"07A1_温度场到能量的读出契约.md"
    chapter2 = report_root/"07A2_物理损失的实现与消融.md"
    a, b = chapter1.read_text(encoding="utf-8"), chapter2.read_text(encoding="utf-8")
    cells = 0
    for label, value in zip(("internal first vs inlet", "internal last vs outlet", "internal last vs inlet", "internal first vs outlet"), gaps.max(axis=0)):
        if table_cells(a, label) != [f"{value:.6f}"]:
            raise ValueError("energy port table mismatch")
        cells += 1
    metric_section = b.split("## 9.", 1)[1].split("## 10.", 1)[0]
    keys = ("unobserved_field_rmse_K", "pde_rms_K_per_s", "stage_jump_rms_K", "cycle_jump_rms_K", "energy_rate_residual_rms_W")
    for variant in ("data", "pde", "pde_interfaces", "full"):
        expected = [f"{physics['aggregate'][variant][k]['mean']:.6f}" for k in keys]
        if table_cells(metric_section, variant) != expected:
            raise ValueError("physics metric table mismatch")
        cells += 5
    weights = b.split("## 8.", 1)[1].split("## 9.", 1)[0]
    for variant, expected in (("data", [10, 0, 0, 0, 0]), ("pde", [10, 1, 0, 0, 0]),
                              ("pde_interfaces", [10, 1, 1, 1, 0]), ("full", [10, 1, 1, 1, 1])):
        if table_cells(weights, variant) != list(map(str, expected)):
            raise ValueError("loss-weight table mismatch")
        cells += 5
    links = re.findall(r"\]\(<([^>]+)>\)", a+b)
    for path in links:
        path = re.sub(r":\d+$", "", path)
        if re.match(r"^/[A-Za-z]:/", path):
            path = path[1:]
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    commands = re.findall(r"^python .*", a+b, re.M)
    for command in commands:
        words = command.split()
        module = words[words.index("-m")+1]
        if module == "unittest":
            continue
        tree = ast.parse(Path(module.replace(".", "/")+".py").read_text(encoding="utf-8"))
        flags = {v.value for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and n.func.attr == "add_argument" for v in n.args if isinstance(v, ast.Constant)}
        if any(word.startswith("--") and word not in flags for word in words):
            raise ValueError("unknown documented option")
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/"independent_port_quadrature.json", {"stage_count": len(differences), "differences_J": differences.tolist(), "endpoint_gaps_K": gaps.tolist()})
    report = {"status": "passed_physics_chapters_narrow_review", "whole_report_complete": False,
        "independent_stage_quadratures": len(differences), "checked_metric_and_weight_cells": cells,
        "source_and_dataset_hashes_verified": True, "local_links_checked": len(links), "documented_commands": len(commands),
        "command_scope": "AST module/option inventory; existing actual runs separately audited, user replay output directories not rerun",
        "readability_scope": "author walkthrough; symbols/units precede loss formulas; data permissions and negative results explicit; independent human comprehension not tested",
        "scientific_boundaries": ["archivalCOPnotnetrecovery", "internalfieldendpointsnotports", "readoutdiagnosticusesobservedfutureinputs",
            "manufacturedlossbenchmarknotSMA", "bothcyclesobservednotrollout", "energyintegralconsequenceofPDE", "hardconditionsineveryvariant"],
        "fresh_training_audit_fits": derivative["fresh_fits_exact"],
        "chapter_sha256": {p.name: sha(p) for p in (chapter1, chapter2)}, "implementation_sha256": sha(Path(__file__)),
        "artifacts": {"independent_port_quadrature.json": sha(out/"independent_port_quadrature.json")}}
    write_json(out/"summary.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.dataset_root, args.campaign_root, args.report_root, args.output_root), indent=2))
