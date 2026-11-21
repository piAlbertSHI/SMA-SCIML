"""Small transparent deep ensemble for the existing-data WP6 teaching loop."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn

from .run_learning_existing import FEATURES,TARGETS,sha,write_json,validate_dataset

ENSEMBLE_TARGETS=tuple(TARGETS)+("Wtotal_for_COP_J",)


def design_matrix(table):
    if not table.material_id.isin(["recA","recAs"]).all():
        raise ValueError("unknown material encoding")
    return np.column_stack((table[list(FEATURES)].to_numpy(float),
        (table.material_id=="recA").to_numpy(float),(table.material_id=="recAs").to_numpy(float)))


class LearningEnsemble:
    """Independently initialized networks; train-only common scalers; no bootstrapping.

    Between-member dispersion is a heuristic epistemic indicator, NOT a complete
    posterior or independently identifiable aleatoric/epistemic decomposition.
    """
    def __init__(self,members=5,epochs=300,seed=20260910):
        if type(members) is not int or members<2 or type(epochs) is not int or epochs<1:
            raise ValueError("at least two members and positive integer epochs required")
        self.members,self.epochs,self.seed=members,epochs,seed
        self.networks=[]
        self.training_ids=()
        self.history=[]

    def fit(self,x,y,ids):
        x,y=np.asarray(x,float),np.asarray(y,float)
        if (x.ndim!=2 or y.ndim!=2 or len(x)!=len(y) or len(x)<3 or
            len(ids)!=len(x) or len(set(ids))!=len(ids) or
            not np.isfinite(x).all() or not np.isfinite(y).all()):
            raise ValueError("invalid finite/unique training data")
        self.x_scaler,self.y_scaler=StandardScaler().fit(x),StandardScaler().fit(y)
        tx=torch.from_numpy(self.x_scaler.transform(x))
        ty=torch.from_numpy(self.y_scaler.transform(y))
        self.networks=[]
        self.history=[]
        old_threads=torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            with torch.random.fork_rng(devices=[]):
                for member in range(self.members):
                    torch.manual_seed(self.seed+1009*member)
                    network=nn.Sequential(nn.Linear(x.shape[1],32),nn.SiLU(),nn.Linear(32,32),
                                          nn.SiLU(),nn.Linear(32,y.shape[1])).double()
                    optimizer=torch.optim.AdamW(network.parameters(),lr=.002,weight_decay=1e-4)
                    losses=[]
                    for epoch in range(self.epochs):
                        optimizer.zero_grad(set_to_none=True)
                        loss=torch.mean((network(tx)-ty)**2)
                        if not torch.isfinite(loss):
                            raise RuntimeError("non-finite ensemble training loss")
                        loss.backward()
                        nn.utils.clip_grad_norm_(network.parameters(),10.,error_if_nonfinite=True)
                        optimizer.step()
                        if epoch%50==0 or epoch==self.epochs-1:
                            losses.append({"epoch":epoch+1,"normalized_mse":float(loss.detach())})
                    network.eval()
                    self.networks.append(network)
                    self.history.append({"member":member,"seed":self.seed+1009*member,"losses":losses})
        except Exception:
            self.networks=[]
            self.training_ids=()
            raise
        finally:
            torch.set_num_threads(old_threads)
        self.training_ids=tuple(map(str,ids))
        return self

    def predict_members(self,x):
        x=np.asarray(x,float)
        if len(self.networks)!=self.members or not self.training_ids:
            raise RuntimeError("ensemble is not fitted")
        if x.ndim!=2 or not np.isfinite(x).all():
            raise ValueError("non-finite/wrong-shaped query")
        tx=torch.from_numpy(self.x_scaler.transform(x))
        with torch.no_grad():
            pred=np.stack([self.y_scaler.inverse_transform(net(tx).numpy()) for net in self.networks])
        if not np.isfinite(pred).all():
            raise RuntimeError("non-finite ensemble prediction")
        return pred

    def predict(self,x):
        predictions=self.predict_members(x)
        return predictions.mean(axis=0),predictions.std(axis=0,ddof=1)

    def save(self,path):
        if path.exists():
            raise FileExistsError("preserve existing ensemble checkpoint")
        payload={"learning_only":True,"members":self.members,"epochs":self.epochs,"seed":self.seed,
            "training_ids":list(self.training_ids),"input_dim":self.x_scaler.n_features_in_,
            "output_dim":self.y_scaler.n_features_in_,"history":self.history,
            "x_mean":torch.tensor(self.x_scaler.mean_),"x_scale":torch.tensor(self.x_scaler.scale_),
            "y_mean":torch.tensor(self.y_scaler.mean_),"y_scale":torch.tensor(self.y_scaler.scale_),
            "state_dicts":[network.state_dict() for network in self.networks]}
        with path.open("xb") as stream:
            torch.save(payload,stream)
        return sha(path)

    @classmethod
    def load(cls,path):
        payload=torch.load(path,weights_only=True)
        if payload.get("learning_only") is not True:
            raise ValueError("unexpected checkpoint authority")
        model=cls(payload["members"],payload["epochs"],payload["seed"])
        for prefix in ("x","y"):
            scaler=StandardScaler()
            scaler.mean_=payload[prefix+"_mean"].numpy()
            scaler.scale_=payload[prefix+"_scale"].numpy()
            scaler.var_=scaler.scale_**2
            scaler.n_features_in_=len(scaler.mean_)
            scaler.n_samples_seen_=len(payload["training_ids"])
            setattr(model,prefix+"_scaler",scaler)
        with torch.random.fork_rng(devices=[]):
            for state in payload["state_dicts"]:
                network=nn.Sequential(nn.Linear(payload["input_dim"],32),nn.SiLU(),nn.Linear(32,32),
                                      nn.SiLU(),nn.Linear(32,payload["output_dim"])).double()
                network.load_state_dict(state)
                network.eval()
                model.networks.append(network)
        model.training_ids=tuple(payload["training_ids"])
        model.history=payload["history"]
        return model


def rank_correlation(x,y):
    if np.ptp(x)==0 or np.ptp(y)==0:
        return None
    value=float(spearmanr(x,y).statistic)
    return value if np.isfinite(value) else None


def calibrated_intervals(cal_truth,cal_mean,cal_std,test_truth,test_mean,test_std,target_scale,coverage):
    """Simultaneous split conformal, with a train-scale floor fixed before calibration."""
    if not 0<coverage<1:
        raise ValueError("coverage must lie strictly inside (0,1)")
    floor=.01*np.asarray(target_scale)
    cal_width=np.maximum(cal_std,floor)
    test_width=np.maximum(test_std,floor)
    scores=np.max(np.abs(cal_truth-cal_mean)/cal_width,axis=1)
    if not np.isfinite(scores).all():
        raise ValueError("non-finite conformal calibration score")
    order=int(np.ceil((len(scores)+1)*coverage))
    if order>len(scores):
        return {"nominal_coverage":coverage,"calibration_count":len(scores),
            "order_statistic_rank":order,"status":"unbounded_interval_required",
            "finite_interval_available":False,"quantile":None,
            "reason":"Finite-sample rank exceeds available calibration examples; do not replace with a smaller quantile"},None
    q=float(np.sort(scores)[order-1])
    width=q*test_width
    inside=np.abs(test_truth-test_mean)<=width
    return {"nominal_coverage":coverage,"calibration_count":len(scores),
        "order_statistic_rank":order,"status":"finite_calibrated_interval",
        "finite_interval_available":True,"quantile":q,
        "test_simultaneous_coverage":float(inside.all(axis=1).mean()),
        "per_target_coverage":inside.mean(axis=0).tolist(),
        "mean_full_width":(2*width).mean(axis=0).tolist()},width


def run(root,original_protocol,out):
    if out.exists():
        raise FileExistsError("preserve previous UQ run")
    table,quality,_=validate_dataset(root)
    previous=json.loads(original_protocol.read_text())
    if previous["dataset_source_content_sha256"]!=quality["source_content_sha256"]:
        raise ValueError("source mismatch")
    splits=previous["condition_splits"]
    masks={k:table.condition_id.isin(ids).to_numpy() for k,ids in splits.items()}
    x,y=design_matrix(table),table[list(ENSEMBLE_TARGETS)].to_numpy(float)
    out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    protocol={"learning_only":True,"source_content_sha256":quality["source_content_sha256"],
        "condition_splits":splits,"members":5,"epochs":300,"target_names":ENSEMBLE_TARGETS,
        "implementation_sha256":sha(Path(__file__)),
        "epistemic_indicator":"sample std of independently initialized network predictions",
        "aleatoric_variance":"not identifiable from deterministic single-observation numerical data",
        "test_history":"Previously inspected WP2-WP4 split; exploratory new-model comparison, not untouched confirmation",
        "ood_protocol":"Train only recA excluding fixed ID test; compare ID held-out recA and all recAs",
        "no_solver_run":True}
    write_json(out/"protocol.json",protocol)
    old_threads=torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        model=LearningEnsemble().fit(x[masks["train"]],y[masks["train"]],table.case_id[masks["train"]].tolist())
        members=model.predict_members(x)
        mean,std=members.mean(axis=0),members.std(axis=0,ddof=1)
        digest=model.save(out/"ensemble.pt")
        replay=LearningEnsemble.load(out/"ensemble.pt").predict_members(x)
        np.testing.assert_array_equal(members,replay)
        test=masks["test"]
        report={"per_target":{},"checkpoint_sha256":digest,"checkpoint_roundtrip_exact":True,
                "history":model.history,"intervals":[]}
        for j,name in enumerate(ENSEMBLE_TARGETS):
            error=np.abs(mean[test,j]-y[test,j])
            report["per_target"][name]={"ensemble_test_rmse":float(np.sqrt(np.mean(error**2))),
                "member_test_rmse":[float(np.sqrt(np.mean((p[test,j]-y[test,j])**2))) for p in members],
                "error_uncertainty_spearman":rank_correlation(error,std[test,j]),
                "mean_ensemble_std":float(std[test,j].mean())}
        saved={"case_ids":table.case_id.to_numpy(str),"targets":y,"member_predictions":members}
        for coverage in (.9,.95):
            info,width=calibrated_intervals(y[masks["calibration"]],mean[masks["calibration"]],std[masks["calibration"]],
                y[test],mean[test],std[test],model.y_scaler.scale_,coverage)
            report["intervals"].append(info)
            if width is not None:
                saved[f"width_{coverage}"]=width
        # Material OOD discrimination is separate from physical failure detection.
        id_test=masks["test"]&(table.material_id=="recA").to_numpy()
        ood=(table.material_id=="recAs").to_numpy()
        ood_train=(table.material_id=="recA").to_numpy()&~id_test
        detector=LearningEnsemble(seed=20260911).fit(x[ood_train],y[ood_train],table.case_id[ood_train].tolist())
        evaluated=id_test|ood
        op,os=detector.predict(x[evaluated])
        labels=ood[evaluated].astype(int)
        disagreement=np.mean(os/detector.y_scaler.scale_,axis=1)
        report["material_ood"]={"training_count":int(ood_train.sum()),"id_test_count":int(id_test.sum()),
            "ood_test_count":int(ood.sum()),"ood_definition":"recAs, absent from this detector's fit",
            "uncertainty_auroc":float(roc_auc_score(labels,disagreement)),
            "uncertainty_auprc":float(average_precision_score(labels,disagreement)),
            "not_a_failure_classifier":True,
            "id_target_rmse":np.sqrt(np.mean((op[labels==0]-y[evaluated][labels==0])**2,axis=0)).tolist(),
            "ood_target_rmse":np.sqrt(np.mean((op[labels==1]-y[evaluated][labels==1])**2,axis=0)).tolist(),
            "training_ids":list(detector.training_ids)}
        detector.save(out/"material_ood_ensemble.pt")
        saved.update(ood_case_ids=table.case_id[evaluated].to_numpy(str),ood_labels=labels,
                     ood_prediction=op,ood_std=os)
        np.savez_compressed(out/"predictions.npz",**saved)
        write_json(out/"uq_report.json",report)
    finally:
        torch.set_num_threads(old_threads)
    summary={"status":"completed_uq_learning_experiment","learning_only":True,
        "member_count":5,"epochs":300,"elapsed_seconds":time.monotonic()-start,
        "intervals":report["intervals"],"material_ood":report["material_ood"],
        "aleatoric_decomposition_verified":False,
        "artifacts":{p.name:sha(p) for p in out.iterdir() if p.is_file()}}
    write_json(out/"summary.json",summary)
    return summary


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root",type=Path,required=True)
    parser.add_argument("--original-protocol",type=Path,required=True)
    parser.add_argument("--output-root",type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(run(args.dataset_root,args.original_protocol,args.output_root),indent=2))
