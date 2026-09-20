import pickle, time, sys, numpy as np
from pools import load_P
from sim import WEIGHTS
from sac import train, drs_fn_from
from evaluate import run_policy, summarize
P = load_P(); pools = pickle.load(open("pools.pkl", "rb"))
t0 = time.time()
actor = train(P, pools["train"], WEIGHTS["baseline"], seed=0, iters=int(sys.argv[1]), log=sys.stdout, target_frac=float(sys.argv[2]))
print("train time", time.time() - t0)
dev = pools["dev"]
for pol in ["behavioral", "expert", "drs", "drsq"]:
    fn = drs_fn_from(actor, "actor" if pol == "drs" else "q") if pol.startswith("drs") else None
    S = run_policy(P, dev, "drs" if fn else pol, 77, fn)
    s = summarize(P, S)
    print(pol, {k: round(float(np.mean(v)), 2) for k, v in s.items()}, np.round(S["act_share"], 3))
