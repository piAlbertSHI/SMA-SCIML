"""Two-round archived-label active-learning experiment, not fresh solver sampling.

The selector receives candidate inputs/predictions and already revealed targets,
never unrevealed pool targets. Test and calibration IDs cannot be queried.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import rankdata
import torch

from SciML.models.v12.learning_ensemble import LearningEnsemble,ENSEMBLE_TARGETS,design_matrix
from SciML.models.v12.run_learning_existing import sha,validate_dataset,write_json


class ArchivedLabelOracle:
    """API-level experimental separation, not a security boundary against Python introspection."""
    def __init__(self,ids,labels):
        values=np.asarray(labels,float)
        if len(ids)!=len(values) or len(set(ids))!=len(ids) or not np.isfinite(values).all():
            raise ValueError("invalid archived oracle data")
        self._labels={str(key):value.copy() for key,value in zip(ids,values)}
        self.revealed=set()
        self.events=[]

    def reveal(self,ids,reason):
        if (not ids or len(set(ids))!=len(ids) or any(key not in self._labels for key in ids)
            or any(key in self.revealed for key in ids)):
            raise ValueError("oracle refuses protected, duplicate or already revealed IDs")
        self.revealed.update(ids)
        self.events.append({"query_index":len(self.events),"reason":str(reason),"case_ids":list(ids)})
        return np.stack([self._labels[key] for key in ids])


def pareto_mask(values):
    values=np.asarray(values,float)
    if values.ndim!=2 or values.shape[1]!=2 or not np.isfinite(values).all():
        raise ValueError("finite 2-D maximization objectives required")
    return np.array([not any(np.all(other>=point) and np.any(other>point) for other in values)
                     for point in values])


def hypervolume(values,reference):
    values=np.asarray(values,float).reshape(-1,2)
    reference=np.asarray(reference,float)
    if reference.shape!=(2,) or not np.isfinite(reference).all():
        raise ValueError("finite reference required")
    if not np.isfinite(values).all():
        raise ValueError("non-finite Pareto objective")
    values=values[np.all(values>reference,axis=1)]
    if not len(values):
        return 0.
    front=values[pareto_mask(values)]
    front=front[np.argsort(front[:,0])]
    total,previous=0.,reference[0]
    for x,y in front:
        total+=(x-previous)*(y-reference[1])
        previous=x
    return float(total)


def acquisition_components(mean,std,known_targets,target_mean,target_scale):
    mean,std,known=np.asarray(mean),np.asarray(std),np.asarray(known_targets)
    scale=np.asarray(target_scale)
    if (mean.ndim!=2 or mean.shape!=std.shape or mean.shape[1]!=6 or known.ndim!=2 or known.shape[1]!=6
        or scale.shape!=(6,) or np.any(scale<=0) or np.any(std<0)
        or not all(np.isfinite(v).all() for v in (mean,std,known,scale,target_mean))):
        raise ValueError("invalid acquisition inputs")
    uncertainty=np.mean(std/scale,axis=1)
    # An algebraic scalar-head consistency diagnostic, NOT a PDE/energy residual.
    identity=np.abs(mean[:,0]-mean[:,2]*mean[:,5])/scale[0]
    objectives=[2,3]  # maximize COP and SCP
    front=(known[:,objectives]-np.asarray(target_mean)[objectives])/scale[objectives]
    optimistic=(mean[:,objectives]+std[:,objectives]-np.asarray(target_mean)[objectives])/scale[objectives]
    reference=front.min(axis=0)-1.
    base=hypervolume(front,reference)
    improvement=np.array([max(0.,hypervolume(np.vstack((front,point)),reference)-base) for point in optimistic])
    normalize=lambda v: (rankdata(v,method="average")-.5)/len(v)
    composite=(normalize(uncertainty)+normalize(identity)+normalize(improvement))/3
    return {"uncertainty":uncertainty,"performance_identity_residual":identity,
            "optimistic_hypervolume_gain":improvement,"composite":composite}


def choose_batch(ids,components,strategy,batch_size,seed):
    if len(ids)!=len(set(ids)) or not 0<batch_size<=len(ids):
        raise ValueError("invalid candidate IDs/budget")
    if strategy=="random":
        order=np.random.default_rng(seed).permutation(len(ids))
    elif strategy in ("uncertainty","composite"):
        values=np.asarray(components[strategy])
        if values.shape!=(len(ids),) or not np.isfinite(values).all():
            raise ValueError("invalid acquisition score")
        order=np.lexsort((np.asarray(ids),-values))
    else:
        raise ValueError("unknown acquisition strategy")
    return [ids[i] for i in order[:batch_size]]


def initial_design(table,seed,count=12):
    rank=lambda value:hashlib.sha256(f"initial:{seed}:{value}".encode()).hexdigest()
    ordered=sorted(table.case_id.tolist(),key=rank)
    selected=[]
    for material in sorted(table.material_id.unique()):
        allowed=set(table.loc[table.material_id==material,"case_id"])
        selected.append(next(case for case in ordered if case in allowed))
    selected+= [case for case in ordered if case not in selected][:count-len(selected)]
    if len(selected)!=count:
        raise ValueError("not enough initial design points")
    return selected


def one_experiment(pool_x,pool_ids,oracle,initial_ids,test_x,test_y,strategy,seed,out):
    lookup={case:index for index,case in enumerate(pool_ids)}
    training_ids=list(initial_ids)
    training_y=oracle.reveal(training_ids,"initial input-only design")
    history,predictions=[],[]
    for round_index in range(3):  # initial fit + TWO query/retrain rounds
        training_x=pool_x[[lookup[case] for case in training_ids]]
        model=LearningEnsemble(members=3,epochs=200,seed=10000+seed*10+round_index).fit(training_x,training_y,training_ids)
        test_mean,test_std=model.predict(test_x)
        predictions.append(test_mean)
        record={"round":round_index,"revealed_count":len(training_ids),"training_ids":training_ids.copy(),
            "test_target_rmse":np.sqrt(np.mean((test_mean-test_y)**2,axis=0)).tolist()}
        # Evaluation is logged; no test metric enters stopping, ranking or fitting.
        if round_index<2:
            candidates=sorted(set(pool_ids)-set(training_ids))
            mean,std=model.predict(pool_x[[lookup[case] for case in candidates]])
            components=acquisition_components(mean,std,training_y,model.y_scaler.mean_,model.y_scaler.scale_)
            chosen=choose_batch(candidates,components,strategy,5,seed+round_index)
            record["candidate_ids"]=candidates
            record["candidate_components"]={key:value.tolist() for key,value in components.items()}
            record["chosen_ids"]=chosen
            # Save decisions BEFORE labels are revealed; x-only acquisition replayable.
            write_json(out/f"round_{round_index}_prequery.json",record)
            added=oracle.reveal(chosen,"fixed-budget acquisition after recorded decision")
            training_y=np.vstack((training_y,added))
            training_ids+=chosen
        else:
            model.save(out/"final_ensemble.pt")
        history.append(record)
    np.savez_compressed(out/"test_predictions.npz",predictions=np.stack(predictions),targets=test_y)
    report={"strategy":strategy,"seed":seed,"history":history,"oracle_queries":oracle.events,
        "final_revealed_count":len(oracle.revealed),"final_unrevealed_count":len(pool_ids)-len(oracle.revealed),
        "test_metrics_used_for_acquisition":False,"fresh_solver_calls":0}
    write_json(out/"report.json",report)
    return report


def run(root,original_protocol,out):
    if out.exists():
        raise FileExistsError("preserve previous active-learning evidence")
    table,quality,_=validate_dataset(root)
    original=json.loads(original_protocol.read_text())
    if original["dataset_source_content_sha256"]!=quality["source_content_sha256"]:
        raise ValueError("dataset/split mismatch")
    pool=table[table.condition_id.isin(original["condition_splits"]["train"])].copy()
    test=table[table.condition_id.isin(original["condition_splits"]["test"])].copy()
    pool_ids=pool.case_id.tolist()
    pool_x,test_x=design_matrix(pool),design_matrix(test)
    pool_y,test_y=pool[list(ENSEMBLE_TARGETS)].to_numpy(float),test[list(ENSEMBLE_TARGETS)].to_numpy(float)
    out.mkdir(parents=True,exist_ok=False)
    protocol={"learning_only":True,"offline_archived_label_reveal":True,"fresh_solver_calls":0,
        "dataset_source_content_sha256":quality["source_content_sha256"],"pool_case_ids":pool_ids,
        "test_case_ids":test.case_id.tolist(),"calibration_cases_never_queried":True,
        "seeds":[701,702,703],"strategies":["random","uncertainty","composite"],
        "budgets":[12,17,22],"members":3,"epochs":200,"target_names":ENSEMBLE_TARGETS,
        "composite":"equal percentile ranks of ensemble disagreement, Q-COP*Wtotal identity and optimistic COP/SCP hypervolume gain",
        "residual_boundary":"performance identity is NOT physical energy balance or PDE residual",
        "test_policy":"Fixed-budget exploratory evaluation; previous test history disclosed; no score-based tuning/stopping",
        "implementation_sha256":sha(Path(__file__)),
        "ensemble_implementation_sha256":sha(Path(__file__).parents[1]/"models/v12/learning_ensemble.py")}
    write_json(out/"protocol.json",protocol)
    start=time.monotonic()
    reports=[]
    old_threads=torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        for seed in protocol["seeds"]:
            initial_ids=initial_design(pool,seed)
            for strategy in protocol["strategies"]:
                destination=out/f"s{seed}_{strategy}"
                destination.mkdir()
                oracle=ArchivedLabelOracle(pool_ids,pool_y)
                reports.append(one_experiment(pool_x,pool_ids,oracle,initial_ids,test_x,test_y,strategy,seed,destination))
    finally:
        torch.set_num_threads(old_threads)
    curves={}
    for strategy in protocol["strategies"]:
        values=np.array([[r["test_target_rmse"] for r in report["history"]]
                         for report in reports if report["strategy"]==strategy])
        curves[strategy]={"seed_mean_rmse":values.mean(axis=0).tolist(),
                          "seed_std_rmse":values.std(axis=0,ddof=1).tolist()}
    summary={"status":"completed_two_round_offline_active_learning","learning_only":True,
        "fresh_solver_calls":0,"experiment_count":len(reports),"budgets":protocol["budgets"],
        "target_names":ENSEMBLE_TARGETS,"learning_curves":curves,"elapsed_seconds":time.monotonic()-start,
        "artifacts":{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}}
    write_json(out/"summary.json",summary)
    return summary


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root",type=Path,required=True)
    parser.add_argument("--original-protocol",type=Path,required=True)
    parser.add_argument("--output-root",type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(run(args.dataset_root,args.original_protocol,args.output_root),indent=2))
