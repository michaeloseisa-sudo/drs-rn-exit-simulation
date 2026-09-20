import json, numpy as np, pickle
from sim import BASE, Pop
from burnin import burnin

def load_P():
    P = dict(BASE); P.update(json.load(open("calibrated.json"))); return P

def make_pool(P, n_draw, seed, n_keep):
    pop, _ = burnin(P, n_draw, seed, collect=False)
    h = np.where(pop.sector == 0)[0][:n_keep]
    sub = pop.subset(h)
    sub.exp0 = pop.exp_years0[h]
    sub.B0 = sub.B.copy()
    return sub

if __name__ == "__main__":
    P = load_P()
    ev = make_pool(P, 13500, 5001, 10000); print("eval", ev.n)
    tr = make_pool(P, 40000, 7001, 30000); print("train", tr.n)
    dv = make_pool(P, 4000, 6001, 3000); print("dev", dv.n)
    pickle.dump(dict(eval=ev, train=tr, dev=dv), open("pools.pkl", "wb"))
    print("early-career share (eval):", (ev.exp0 <= 5).mean(), " high burnout:", (ev.B0 >= 50).mean(), " thin:", ev.thin.mean())
