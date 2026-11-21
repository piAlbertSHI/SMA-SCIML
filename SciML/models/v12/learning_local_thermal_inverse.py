"""WP5 extension: a deliberately reduced thermal-node inverse model on real fields.

T_s'(t)=k*(mean(T_HTF)(t)-T_s(t)), k>0. This is a model-discrepancy exercise:
k is NOT an identified physical heat-transfer coefficient of the SMA solver.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import h5py
import numpy as np
from scipy.optimize import least_squares
import torch
from torch import nn

from .run_learning_existing import sha, validate_dataset, write_json


class ThermalNodePINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(1,24), nn.Tanh(), nn.Linear(24,24),
                                     nn.Tanh(), nn.Linear(24,1)).double()
        self.log_rate = nn.Parameter(torch.tensor(math.log(.5), dtype=torch.float64))

    def forward(self, tau):
        return tau*self.network(2*tau-1)

    def residual(self, tau, forcing_normalized, duration):
        y = self(tau)
        derivative = torch.autograd.grad(y,tau,torch.ones_like(y),create_graph=True)[0]
        return derivative-duration*torch.exp(self.log_rate)*(forcing_normalized-y)


def ode_prediction(time_s, forcing_K, initial_K, rate):
    """Exact node recurrence for the SAME piecewise-linear observed forcing.

    Adaptive integration inside a finite-difference least-squares objective
    produced numerical-gradient noise in r1. No approximation to the forcing
    contract is changed here: each linear segment is now integrated exactly.
    """
    time_s,forcing_K = np.asarray(time_s,float),np.asarray(forcing_K,float)
    if rate <= 0 or not np.isfinite(rate):
        raise ValueError("rate must be positive/finite")
    if (time_s.ndim != 1 or len(time_s)<2 or forcing_K.shape != time_s.shape or
        np.any(np.diff(time_s)<=0) or not np.isfinite(time_s).all() or
        not np.isfinite(forcing_K).all() or not np.isfinite(initial_K)):
        raise ValueError("invalid node forcing/time data")
    prediction = np.empty_like(time_s)
    prediction[0] = initial_K
    for index,dt in enumerate(np.diff(time_s)):
        z = rate*dt
        one_minus_decay = -np.expm1(-z)
        # Avoid cancellation in dt-(1-exp(-rate*dt))/rate for tiny rates.
        ramp_integral = (dt*(z/2-z*z/6+z**3/24-z**4/120)
                         if z<1e-3 else dt-one_minus_decay/rate)
        slope = (forcing_K[index+1]-forcing_K[index])/dt
        prediction[index+1] = (prediction[index]*np.exp(-z)+
            forcing_K[index]*one_minus_decay+slope*ramp_integral)
    return prediction


def fit(time_s, forcing_K, observed_indices, observed_K, *, seed=20260910, epochs=1200):
    time_s, forcing_K = np.asarray(time_s,float), np.asarray(forcing_K,float)
    indices, observations = np.asarray(observed_indices,int), np.asarray(observed_K,float)
    if (time_s.ndim != 1 or forcing_K.shape != time_s.shape or
        observations.shape != indices.shape or len(indices) < 3 or
        len(set(indices)) != len(indices) or indices[0] != 0 or
        np.any(indices < 0) or np.any(indices >= len(time_s)) or np.any(np.diff(time_s) <= 0) or
        not all(np.isfinite(v).all() for v in (time_s,forcing_K,observations))):
        raise ValueError("invalid time/forcing/sparse observations")
    duration, initial = float(time_s[-1]-time_s[0]), float(observations[0])
    scale = max(float(np.max(np.abs(observations-initial))), 1.)
    tau = (time_s-time_s[0])/duration
    old_threads = torch.get_num_threads()
    history = []
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = ThermalNodePINN()
            x = torch.tensor(tau[indices,None],dtype=torch.float64)
            y = torch.tensor((observations[:,None]-initial)/scale,dtype=torch.float64)
            c_np = np.linspace(0,1,128)
            collocation = torch.tensor(c_np[:,None],dtype=torch.float64,requires_grad=True)
            force = torch.tensor(((np.interp(c_np,tau,forcing_K)-initial)/scale)[:,None],dtype=torch.float64)
            optimizer = torch.optim.Adam(model.parameters(),lr=.003)

            def loss():
                data = torch.mean((model(x)-y)**2)
                physics = torch.mean(model.residual(collocation,force,duration)**2)
                return 10*data+physics,data,physics

            for step in range(epochs):
                optimizer.zero_grad(set_to_none=True)
                collocation.grad = None
                total,data,physics = loss()
                if not torch.isfinite(total):
                    raise RuntimeError("non-finite local inverse training")
                total.backward()
                nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True)
                optimizer.step()
                if step % 100 == 0:
                    history.append({"step":step+1,"data":float(data.detach()),"physics":float(physics.detach())})
            optimizer = torch.optim.LBFGS(model.parameters(),max_iter=100,line_search_fn="strong_wolfe",
                                         tolerance_grad=1e-10,tolerance_change=1e-12)

            def closure():
                optimizer.zero_grad(set_to_none=True)
                collocation.grad = None
                total,_,_ = loss()
                if not torch.isfinite(total):
                    raise RuntimeError("non-finite local inverse L-BFGS")
                total.backward()
                return total

            optimizer.step(closure)
            model.eval()
            with torch.no_grad():
                prediction = initial+scale*model(torch.tensor(tau[:,None],dtype=torch.float64)).numpy()[:,0]
            _,data,physics = loss()
            fitted_rate = float(torch.exp(model.log_rate).detach())
    finally:
        torch.set_num_threads(old_threads)
    classical = least_squares(lambda z: ode_prediction(time_s,forcing_K,initial,np.exp(z[0]))[indices]-observations,
                              [math.log(.5)], bounds=([-10.],[5.]))
    if not classical.success:
        raise RuntimeError("classical reduced inverse fit failed")
    classical_rate = float(np.exp(classical.x[0]))
    return model,prediction,ode_prediction(time_s,forcing_K,initial,classical_rate),{
        "effective_rate_per_s":fitted_rate,"classical_rate_per_s":classical_rate,
        "normalization":{"initial_K":initial,"temperature_scale_K":scale,"duration_s":duration},
        "observed_indices":indices.tolist(),"epochs":epochs,"lbfgs_max_iterations":100,
        "data_weight":10.,"physics_weight":1.,"seed":seed,"history":history,
        "final_normalized_data_mse":float(data.detach()),"final_normalized_physics_mse":float(physics.detach()),
        "physics_residual_rms_K_per_s":float(torch.sqrt(physics).detach())*scale/duration}


def run(root, original_protocol, out):
    if out.exists():
        raise FileExistsError("preserve previous inverse runs")
    table,quality,_ = validate_dataset(root)
    protocol = json.loads(original_protocol.read_text())
    if protocol["dataset_source_content_sha256"] != quality["source_content_sha256"]:
        raise ValueError("dataset/protocol mismatch")
    train = table[table.condition_id.isin(protocol["condition_splits"]["train"])].sort_values("case_id")
    selected = [group.iloc[0].case_id for _,group in train.groupby("material_id",sort=True)]
    selected += [case for case in train.case_id if case not in selected][:3-len(selected)]
    out.mkdir(parents=True,exist_ok=False)
    start = time.monotonic()
    write_json(out/"protocol.json",{"learning_only":True,"selected_training_case_ids":selected,
        "dataset_source_content_sha256":quality["source_content_sha256"],"implementation_sha256":sha(Path(__file__)),
        "interpretation":"Effective positive relaxation rate in a reduced node equation, NOT true SMA h or diffusivity",
        "forcing":"Observed spatial-mean HTF temperature over whole interval, including evaluation interval",
        "split":"8 SMA sensors at indices0,5,11,17,23,29,35,39; last10 SMA values held out",
        "forecast_boundary":"Conditional reconstruction with observed forcing; not autonomous unknown-input forecasting"})
    records,saved = [],{}
    with h5py.File(root/"fields.h5","r") as handle:
        for case in selected:
            row = table[table.case_id==case].iloc[0]
            cycle = handle[row.hdf5_group+"/cycles/cycle_001"]
            for stage,name in ((2,"stage2_hold_exchange"),(4,"stage4_cooldown_exchange")):
                field = cycle[name]
                times = np.asarray(field["time_local_s"]).ravel()
                target = np.asarray(field["sma_strip_T_K_t"]).mean(axis=1)
                forcing = np.asarray(field["htf_internal_T_K_t"]).mean(axis=1)
                if times.shape != (50,) or target.shape != (50,):
                    raise ValueError("unexpected temporal grid")
                observed = np.array([0,5,11,17,23,29,35,39])
                model,pred,base,info = fit(times,forcing,observed,target[observed])
                tag = f"case{len(records)//2+1}_stage{stage}"
                checkpoint = out/(tag+".pt")
                torch.save({"state_dict":model.state_dict(),"normalization":info["normalization"]},checkpoint)
                restored = ThermalNodePINN()
                restored.load_state_dict(torch.load(checkpoint,weights_only=True)["state_dict"])
                tau = (times-times[0])/(times[-1]-times[0])
                with torch.no_grad():
                    replay = info["normalization"]["initial_K"]+info["normalization"]["temperature_scale_K"]*restored(
                        torch.tensor(tau[:,None],dtype=torch.float64)).numpy()[:,0]
                if not np.allclose(pred,replay,rtol=0,atol=1e-12):
                    raise RuntimeError("thermal checkpoint replay mismatch")
                result = {"tag":tag,"case_id":case,"stage":stage,**info,
                    "full_curve_rmse_K":float(np.sqrt(np.mean((pred-target)**2))),
                    "held_tail_rmse_K":float(np.sqrt(np.mean((pred[40:]-target[40:])**2))),
                    "classical_held_tail_rmse_K":float(np.sqrt(np.mean((base[40:]-target[40:])**2))),
                    "neural_vs_classical_rate_relative_difference":abs(info["effective_rate_per_s"]-info["classical_rate_per_s"])/info["classical_rate_per_s"],
                    "identified_physical_h":None,"checkpoint_sha256":sha(checkpoint)}
                records.append(result)
                write_json(out/(tag+".json"),result)
                saved[tag] = np.column_stack((times,target,forcing,pred,base))
    np.savez_compressed(out/"curves.npz",**saved)
    summary = {"status":"completed_reduced_model_diagnostic","learning_only":True,
        "physical_parameter_validation":False,"case_count":len(selected),"stage_count":len(records),
        "elapsed_seconds":time.monotonic()-start,
        "results":[{k:r[k] for k in ("tag","effective_rate_per_s","classical_rate_per_s","held_tail_rmse_K","classical_held_tail_rmse_K")} for r in records],
        "artifacts":{p.name:sha(p) for p in out.iterdir() if p.is_file()}}
    write_json(out/"summary.json",summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root",type=Path,required=True)
    parser.add_argument("--original-protocol",type=Path,required=True)
    parser.add_argument("--output-root",type=Path,required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.dataset_root,args.original_protocol,args.output_root),indent=2))
