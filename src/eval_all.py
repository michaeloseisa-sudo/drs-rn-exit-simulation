import pickle, os, json, numpy as np
from pools import load_P
from sim import WEIGHTS, ACT_NAMES
from sac import drs_fn_from
from evaluate import run_policy, cvi, summarize
P = load_P(); pools = pickle.load(open("pools.pkl", "rb")); ev = pools["eval"]
SEED = 9001
out = {}
def keep(S): return {k: v for k, v in S.items()}
res = {}
for pol in ["behavioral", "expert"]:
    res[pol] = run_policy(P, ev, pol, SEED)
for f in sorted(os.listdir("actors")):
    net = pickle.load(open("actors/" + f, "rb"))
    res["drs:" + f[:-4]] = run_policy(P, ev, "drs", SEED, drs_fn_from(net))
    print("eval", f, flush=True)
# structural perturbations (no retraining, no recalibration)
PERT = {"switch_cost_x2": {"switch_cost": 1.0}, "no_switch_premium": {"switch_prem": 0.0},
        "slower_burnout_recovery": {"kB": 1 / 18}}
pert = {}
for name, ch in PERT.items():
    P2 = dict(P); P2.update(ch); pert[name] = {}
    for pol in ["behavioral", "expert"]:
        pert[name][pol] = run_policy(P2, ev, pol, SEED)
    for s in range(1, 6):
        net = pickle.load(open(f"actors/baseline_{s}.pkl", "rb"))
        pert[name][f"drs:{s}"] = run_policy(P2, ev, "drs", SEED, drs_fn_from(net))
    print("pert", name, flush=True)
pickle.dump(dict(res=res, pert=pert, PERT=PERT), open("eval_results.pkl", "wb"))
print("ALL DONE", flush=True)
