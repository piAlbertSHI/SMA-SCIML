"""WP5 fair-comparison audit: shared PDE coefficient, unseen interior, baseline repair."""
from pathlib import Path
import argparse
import json
import math

import numpy as np
import torch

from SciML.models.v12.learning_pinn_inverse import HeatInversePINN,predict
from SciML.models.v12.run_learning_existing import sha,write_json


def verify_artifacts(root):
    summary=json.loads((root/"summary.json").read_text(encoding="utf-8"))
    for name,digest in summary["artifacts"].items():
        if sha(root/name)!=digest:
            raise ValueError(f"changed artifact {root/name}")
    return summary


def run(campaign,out):
    if out.exists():
        raise FileExistsError("preserve prior audit")
    manufactured=campaign/"wp5-pinn-r1"
    summary=verify_artifacts(manufactured)
    current=campaign/"wp5-local-thermal-r2"
    corrected=verify_artifacts(current)
    old=campaign/"wp5-local-thermal-r1"
    verify_artifacts(old)
    data=np.load(manufactured/"predictions.npz",allow_pickle=False)
    grid,truth,observed=data["grid"],data["truth"],data["observation_coordinates"]
    mask=(grid[:,0]>0)&(grid[:,0]<1)&(grid[:,1]>0)
    for point in observed:
        mask &= ~np.all(np.isclose(grid,point,rtol=0,atol=1e-12),axis=1)
    reports=[]
    for record in summary["runs"]:
        tag=record["tag"]
        model=HeatInversePINN()
        payload=torch.load(manufactured/(tag+".pt"),weights_only=True)
        model.load_state_dict(payload["state_dict"])
        prediction=predict(model,grid)
        np.testing.assert_array_equal(prediction,data[tag+"_prediction"])
        # Data-only coefficient never trained: its r1 native residual is not a
        # fair PDE comparison. Set the SAME known coefficient for EVALUATION only.
        with torch.no_grad():
            model.log_diffusivity.copy_(torch.tensor(math.log(.12),dtype=torch.float64))
        residual=model.residual(torch.tensor(grid[mask][::7],dtype=torch.float64,requires_grad=True)).detach().numpy()
        reports.append({"tag":tag,"unobserved_interior_count":int(mask.sum()),
            "unobserved_interior_rmse":float(np.sqrt(np.mean((prediction[mask]-truth[mask])**2))),
            "common_true_diffusivity_pde_rms":float(np.sqrt(np.mean(residual**2))),
            "coefficient_used_for_fair_residual":.12})
    unchanged=[]
    for record in corrected["results"]:
        name=record["tag"]+".pt"
        if sha(old/name)!=sha(current/name):
            raise ValueError("baseline repair unexpectedly changed neural fit")
        unchanged.append(name)
    out.mkdir(parents=True,exist_ok=False)
    report={"status":"passed_wp5_learning_audit","learning_only":True,
        "manufactured_results":reports,"local_neural_checkpoints_unchanged":unchanged,
        "baseline_correction":"Exact recurrence replaces noisy adaptive integration inside finite-difference least squares; same forcing/model/observations",
        "fairness_note":"Original data-only residuals use untrained initial coefficient; use common-coefficient residuals here for PDE comparisons",
        "local_model_scope":"Effective rate under observed forcing, not physical h or autonomous forecasting",
        "local_corrected_summary_sha256":sha(current/"summary.json"),
        "manufactured_summary_sha256":sha(manufactured/"summary.json")}
    write_json(out/"audit.json",report)
    make_figures(campaign,data,reports,out)
    return report


def make_figures(campaign,data,reports,out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.spines.top":False,"axes.spines.right":False,"font.size":10})
    fig,axes=plt.subplots(1,3,figsize=(14,4.4),layout="constrained")
    levels=np.linspace(0,.08,21)
    for axis,tag,title in zip(axes[:2],("s410_pinn","s410_data_only"),("PINN | unseen-field reconstruction","Data only | same 12 sensors")):
        error=np.abs(data[tag+"_prediction"]-data["truth"]).reshape(51,51)
        im=axis.contourf(np.linspace(0,1,51),np.linspace(0,1,51),error,levels=levels,extend="max",cmap="magma")
        axis.scatter(data["observation_coordinates"][:,0],data["observation_coordinates"][:,1],s=12,c="cyan",edgecolors="black",linewidths=.3)
        axis.set(xlabel="x (dimensionless)",ylabel="time (dimensionless)",title=title)
    fig.colorbar(im,ax=axes[:2],label="Absolute normalized temperature error")
    for tag,label,color in (("s410_pinn","PINN","#007d91"),("s410_data_only","Data only","#9a5b32")):
        r=json.loads((campaign/"wp5-pinn-r1"/(tag+".json")).read_text())
        h=[v for v in r["training"]["loss_history"] if v["phase"]=="adam"]
        axes[2].semilogy([v["step"] for v in h],[v["data"] for v in h],label=label,color=color)
    axes[2].set(xlabel="Adam step",ylabel="Sparse-observation MSE",title="Small sensor loss is not full-field proof")
    axes[2].legend(frameon=False)
    fig.suptitle("WP5 manufactured heat equation | known a = 0.12 | not SMA validation",fontsize=14)
    fig.savefig(out/"pinn_inverse.png",dpi=150)
    plt.close(fig)
    curves=np.load(campaign/"wp5-local-thermal-r2/curves.npz",allow_pickle=False)
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout="constrained")
    for axis,tag in zip(axes.ravel(),curves.files):
        v=curves[tag]
        axis.plot(v[:,0],v[:,1],color="black",label="SMA numerical record")
        axis.plot(v[:,0],v[:,3],"--",color="#007d91",label="PINN")
        axis.plot(v[:,0],v[:,4],":",color="#bd7131",label="Classical (corrected)")
        ix=[0,5,11,17,23,29,35,39]
        axis.scatter(v[ix,0],v[ix,1],s=12,c="black")
        axis.axvspan(v[40,0],v[-1,0],alpha=.15,color="#007d91")
        axis.set(xlabel="Time (s)",ylabel="Mean SMA temperature (K)",title=tag)
    axes[0,0].legend(frameon=False,fontsize=8)
    fig.suptitle("WP5 existing project data | reduced node equation, observed HTF forcing | shaded tail held out",fontsize=13)
    fig.savefig(out/"local_thermal_inverse.png",dpi=150)
    plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root",type=Path,required=True)
    parser.add_argument("--output-root",type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(run(args.campaign_root,args.output_root),indent=2))
