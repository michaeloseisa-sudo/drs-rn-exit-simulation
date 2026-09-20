import pickle, os, sys, time, numpy as np
from pools import load_P
from sim import WEIGHTS
from sac import train
P = load_P(); pools = pickle.load(open("pools.pkl", "rb"))
os.makedirs("actors", exist_ok=True)
runs = [("baseline", s) for s in [1, 2, 3, 4, 5]] + [(w, s) for w in ["equal", "wellbeing", "compensation", "career"] for s in [1, 2, 3]]
log = open("train.log", "a")
for w, s in runs:
    f = f"actors/{w}_{s}.pkl"
    if os.path.exists(f): continue
    t0 = time.time()
    net = train(P, pools["train"], WEIGHTS[w], seed=s, iters=40000, target_frac=0.05, log=log)
    pickle.dump(net, open(f, "wb"))
    print(f"DONE {w} {s} {time.time()-t0:.0f}s", file=log, flush=True)
