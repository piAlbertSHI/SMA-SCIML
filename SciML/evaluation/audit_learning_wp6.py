"""Recompute UQ and replay both acquisition rounds from REVEALED labels only."""
from pathlib import Path
import argparse
import json

import numpy as np
from sklearn.metrics import roc_auc_score
import torch

from SciML.evaluation.audit_learning_wp5 import verify_artifacts
from SciML.models.v12.run_learning_existing import validate_dataset,sha,write_json
from SciML.models.v12.learning_ensemble import (
    LearningEnsemble,ENSEMBLE_TARGETS,design_matrix,calibrated_intervals,
)
from SciML.active_learning.learning_pool_loop import acquisition_components,choose_batch,initial_design


def run(campaign,dataset,original_protocol,out):
    if out.exists():
        raise FileExistsError("preserve previous WP6 audits")
    uq=campaign/"wp6-uq-r1"
    active=campaign/"wp6-active-r1"
    us,als=verify_artifacts(uq),verify_artifacts(active)
    table,quality,_=validate_dataset(dataset)
    original=json.loads(original_protocol.read_text())
    if original["dataset_source_content_sha256"]!=quality["source_content_sha256"]:
        raise ValueError("dataset changed")
    x,y=design_matrix(table),table[list(ENSEMBLE_TARGETS)].to_numpy(float)
    splits=original["condition_splits"]
    masks={k:table.condition_id.isin(ids).to_numpy() for k,ids in splits.items()}
    data=np.load(uq/"predictions.npz",allow_pickle=False)
    ensemble=LearningEnsemble.load(uq/"ensemble.pt")
    expected_ids=table.case_id[masks["train"]].tolist()
    if list(ensemble.training_ids)!=expected_ids:
        raise ValueError("UQ ensemble training split differs")
    prediction=ensemble.predict_members(x)
    np.testing.assert_array_equal(prediction,data["member_predictions"])
    np.testing.assert_array_equal(data["targets"],y)
    mean,std=prediction.mean(axis=0),prediction.std(axis=0,ddof=1)
    for coverage,recorded in zip((.9,.95),us["intervals"]):
        actual,width=calibrated_intervals(y[masks["calibration"]],mean[masks["calibration"]],std[masks["calibration"]],
            y[masks["test"]],mean[masks["test"]],std[masks["test"]],ensemble.y_scaler.scale_,coverage)
        if actual!=recorded:
            raise ValueError("calibration report mismatch")
        if width is not None:
            np.testing.assert_array_equal(width,data[f"width_{coverage}"])
    detector=LearningEnsemble.load(uq/"material_ood_ensemble.pt")
    known_materials=set(table.loc[table.case_id.isin(detector.training_ids),"material_id"])
    oid=table.set_index("case_id").loc[data["ood_case_ids"]]
    simple_unknown=(~oid.material_id.isin(known_materials)).to_numpy(int)
    simple_auc=float(roc_auc_score(data["ood_labels"],simple_unknown))
    eval_ids=set(data["ood_case_ids"].tolist())
    if set(detector.training_ids)&eval_ids:
        raise ValueError("OOD detector trained on evaluated condition")
    pool=table[masks["train"]]
    lookup={case:index for index,case in enumerate(table.case_id)}
    pool_ids=set(pool.case_id)
    test_x,test_y=x[masks["test"]],y[masks["test"]]
    replays=[]
    for seed in (701,702,703):
        initial_prediction=None
        for strategy in ("random","uncertainty","composite"):
            root=active/f"s{seed}_{strategy}"
            report=json.loads((root/"report.json").read_text())
            saved=np.load(root/"test_predictions.npz",allow_pickle=False)
            np.testing.assert_array_equal(saved["targets"],test_y)
            ids=initial_design(pool,seed)
            if initial_prediction is None:
                initial_prediction=saved["predictions"][0]
            else:
                np.testing.assert_array_equal(initial_prediction,saved["predictions"][0])
            for round_index in range(3):
                record=report["history"][round_index]
                if ids!=record["training_ids"] or len(ids)!=(12+5*round_index) or not set(ids)<=pool_ids:
                    raise ValueError("active-learning budget or label separation violated")
                rows=[lookup[case] for case in ids]
                if round_index<2:
                    # Deliberate independent replay, not production retry: reconstruct
                    # exactly what the learner knew at this round, then re-fit.
                    model=LearningEnsemble(members=3,epochs=200,seed=10000+seed*10+round_index).fit(x[rows],y[rows],ids)
                    candidates=sorted(pool_ids-set(ids))
                    candidate_rows=[lookup[case] for case in candidates]
                    cm,cs=model.predict(x[candidate_rows])
                    components=acquisition_components(cm,cs,y[rows],model.y_scaler.mean_,model.y_scaler.scale_)
                    if candidates!=record["candidate_ids"]:
                        raise ValueError("candidate pool differs")
                    for key,values in components.items():
                        np.testing.assert_allclose(values,record["candidate_components"][key],rtol=1e-12,atol=1e-12)
                    chosen=choose_batch(candidates,components,strategy,5,seed+round_index)
                    if chosen!=record["chosen_ids"]:
                        raise ValueError("acquisition choice cannot be replayed from revealed labels")
                    pre=json.loads((root/f"round_{round_index}_prequery.json").read_text())
                    if pre!=record:
                        raise ValueError("decision-before-label snapshot differs from final log")
                    ids+=chosen
                else:
                    model=LearningEnsemble.load(root/"final_ensemble.pt")
                    if list(model.training_ids)!=ids:
                        raise ValueError("final ensemble training IDs differ")
                actual=model.predict(test_x)[0]
                np.testing.assert_allclose(actual,saved["predictions"][round_index],rtol=1e-12,atol=1e-12)
                errors=np.sqrt(np.mean((actual-test_y)**2,axis=0))
                np.testing.assert_allclose(errors,record["test_target_rmse"],rtol=1e-12,atol=1e-12)
            queries=[case for event in report["oracle_queries"] for case in event["case_ids"]]
            if queries!=ids or len(set(queries))!=22 or report["fresh_solver_calls"]!=0:
                raise ValueError("oracle ledger inconsistency")
            replays.append({"seed":seed,"strategy":strategy,"rounds_replayed":2,"budget":22})
    # Risk/coverage is a diagnostic; points are sorted by UQ, never by their errors.
    uncertainty=np.mean(std[masks["test"]]/ensemble.y_scaler.scale_,axis=1)
    error=np.mean(((mean[masks["test"]]-test_y)/ensemble.y_scaler.scale_)**2,axis=1)
    order=np.argsort(uncertainty,kind="stable")
    risk=[{"retained_count":count,"coverage":count/len(order),
           "normalized_mse":float(error[order[:count]].mean())} for count in range(1,len(order)+1)]
    out.mkdir(parents=True,exist_ok=False)
    report={"status":"passed_wp6_uq_active_learning_audit","learning_only":True,
        "uq_checkpoint_and_calibration_replayed":True,"acquisition_replays":replays,
        "material_ood_unseen_category_rule_auroc":simple_auc,
        "material_ood_interpretation":"Ensemble AUROC=1 does not prove incremental detection skill: known unseen-category rule is also perfect here",
        "risk_coverage":risk,"active_summary_sha256":sha(active/"summary.json"),
        "uq_summary_sha256":sha(uq/"summary.json"),
        "no_new_solver_calls":True,"no_claim_of_universal_active_learning_superiority":True}
    write_json(out/"audit.json",report)
    make_figure(als,out/"active_learning_curves.png")
    return report


def make_figure(summary,path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(12,7),layout="constrained")
    labels=["Qcool (J)","Wpump (J)","COP","SCP (W/kg)","Temperature span (K)","COP denominator work (J)"]
    colors={"random":"#6c788d","uncertainty":"#007d91","composite":"#b97520"}
    for strategy,record in summary["learning_curves"].items():
        mean,std=np.asarray(record["seed_mean_rmse"]),np.asarray(record["seed_std_rmse"])
        for j,axis in enumerate(axes.ravel()):
            axis.plot(summary["budgets"],mean[:,j],"o-",label=strategy,color=colors[strategy])
            axis.fill_between(summary["budgets"],np.maximum(0,mean[:,j]-std[:,j]),mean[:,j]+std[:,j],alpha=.12,color=colors[strategy])
            axis.set(title=labels[j],xlabel="Revealed training conditions",ylabel="Test RMSE",xticks=summary["budgets"])
    axes[0,0].legend(frameon=False)
    fig.suptitle("Two-round OFFLINE active learning | mean +/- one SD over 3 seeds | 13 fixed exploratory test conditions",fontsize=12)
    fig.savefig(path,dpi=150)
    plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root",type=Path,required=True)
    parser.add_argument("--dataset-root",type=Path,required=True)
    parser.add_argument("--original-protocol",type=Path,required=True)
    parser.add_argument("--output-root",type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(run(args.campaign_root,args.dataset_root,args.original_protocol,args.output_root),indent=2))
