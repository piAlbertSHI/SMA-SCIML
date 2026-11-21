"""Show why a material one-hot unseen in training cannot learn material response."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .audit_learning_wp5 import verify_artifacts
from SciML.models.v12.learning_ensemble import LearningEnsemble, design_matrix
from SciML.models.v12.run_learning_existing import TARGETS, sha, validate_dataset, write_json


def material_support(training_materials, query_materials):
    """Input-domain abstention rule, not a learned OOD detector or accuracy guarantee."""
    known = set(map(str, training_materials))
    if not known:
        raise ValueError("training material support must be nonempty")
    return [{"material": str(value), "supported_category": str(value) in known,
             "action": "eligible_for_further_checks" if str(value) in known else "abstain_unseen_material"}
            for value in query_materials]


def run(dataset, campaign, out):
    if out.exists():
        raise FileExistsError("preserve material diagnostic")
    table, quality, _ = validate_dataset(dataset)
    root = campaign/"ood-material-r1"
    verify_artifacts(root)
    protocol = json.loads((root/"protocol.json").read_text())
    train = table.condition_id.isin(protocol["condition_groups"]["train"]).to_numpy()
    ood = table.condition_id.isin(protocol["condition_groups"]["ood_material"]).to_numpy()
    x, y = design_matrix(table), table[list(TARGETS)].to_numpy(float)
    support = material_support(table.material_id[train], table.material_id)
    decisions = [{"case_id": str(case), **decision} for case, decision in zip(table.case_id, support)]
    if any(decision["supported_category"] for decision, use in zip(support, ood) if use):
        raise ValueError("unseen material was not rejected")
    diagnostics = []
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        for seed in (20260910, 20260911, 20260912):
            model = LearningEnsemble.load(root/f"s{seed}/scalar.pt")
            tx = torch.from_numpy(model.x_scaler.transform(x[train]))
            ty = torch.from_numpy(model.y_scaler.transform(y[train]))
            if not torch.equal(tx[:, 5:7], torch.zeros_like(tx[:, 5:7])):
                raise ValueError("material channels were not constant in training")
            gradients = []
            before = model.predict_members(x)
            for network in model.networks:
                network.zero_grad(set_to_none=True)
                loss = torch.mean((network(tx)-ty)**2)
                loss.backward()
                maximum = float(network[0].weight.grad[:, 5:7].abs().max())
                if maximum != 0:
                    raise ValueError("unexpected data gradient in constant material channels")
                gradients.append(maximum)
                network.zero_grad(set_to_none=True)
            # This is an artificial metadata change, not a valid material prediction.
            # It measures dependence on one-hot weights unconstrained by train data.
            as_seen = x.copy()
            as_seen[ood, 5], as_seen[ood, 6] = 1., 0.
            hypothetical = model.predict_members(as_seen)
            np.testing.assert_array_equal(before, model.predict_members(x))
            diagnostics.append({"seed": seed, "max_data_gradient_material_weight_by_member": gradients,
                "mean_absolute_prediction_shift_if_unseen_material_recoded_as_seen": np.abs(before[:, ood]-hypothetical[:, ood]).mean(axis=(0, 1)).tolist(),
                "no_optimizer_update_performed": True})
    finally:
        torch.set_num_threads(threads)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/"material_support_decisions.json", decisions)
    result = {"status": "completed_material_representation_diagnostic", "learning_only": True,
        "fresh_solver_calls": 0, "training_materials": sorted(set(table.material_id[train])),
        "unseen_material_cases_rejected": int(ood.sum()), "target_names": list(TARGETS),
        "diagnostics": diagnostics, "scope": "zero data gradient in two standardized one-hot columns; weight decay may still alter weights, no learned material contrast",
        "abstention_scope": "known-category rule only; does not certify predictions on seen categories or solve material transfer",
        "counterfactual_scope": "recoding test category is diagnostic ONLY, not a repaired deployed prediction",
        "source_content_sha256": quality["source_content_sha256"], "upstream_summary_sha256": sha(root/"summary.json"),
        "implementation_sha256": sha(Path(__file__)),
        "artifacts": {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    write_json(out/"summary.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.dataset_root, args.campaign_root, args.output_root), indent=2))
