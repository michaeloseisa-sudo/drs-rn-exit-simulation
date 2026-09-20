import numpy as np
from scipy.stats import norm
from sim import *

def run_policy(P, pool, policy, seed, drs_fn=None):
    """policy in {'behavioral','expert','drs'}; returns per-career outcome dict (common random numbers)."""
    pop = pool.copy(); n = pop.n
    S = dict(comp=np.zeros((n, 5)), pen=np.zeros(n), sw=np.zeros(n), months_burn=np.zeros(n),
             sep_h=np.zeros(n), vol=np.zeros(n), out_months=np.zeros(n))
    act_counts = np.zeros(N_ACT)
    for t in range(H):
        r = draws(np.random.default_rng([seed, 100000 + t]), n)
        obs, info = observe(P, pop, r, H - t)
        m = masks(pop)
        if policy == "behavioral": a = behavioral(P, pop, r)
        elif policy == "expert": a = expert(P, pop, info)
        else: a = drs_fn(obs, m)
        a = np.where(m[np.arange(n), a], a, 0)
        act_counts += np.bincount(a, minlength=N_ACT)
        dest, ev = step(P, pop, a, r)
        WL = np.clip(0.6 * (1 - pop.OT / 20) + 0.4 * pop.sj, 0, 1)
        S["comp"] += np.stack([pop.F, 1 - pop.B / 100, norm.cdf(pop.c), pop.K / 100, WL], 1)
        S["pen"] += (pop.D >= P["dep_months"]); S["sw"] += ev["switched"]
        S["months_burn"] += ev["burned"]; S["sep_h"] += ev["sep_h"]; S["vol"] += ev["vol"]
        S["out_months"] += pop.sector > 0
        if t == 59:
            S["comp_y5"] = 100 * norm.cdf(pop.c); S["K_y5"] = pop.K.copy()
        apply_moves(P, pop, dest, r)
    S["burn_y10"] = (pop.B >= P["burn_thr"]).astype(float)
    S["act_share"] = act_counts / act_counts.sum()
    return S

def cvi(P, S, w):
    return 100 / H * (S["comp"] @ w - P["lam_burn"] * S["pen"] - P["switch_cost"] * S["sw"])

def summarize(P, S, w=WEIGHTS["baseline"]):
    return dict(CVI=cvi(P, S, w), fit=100 * S["comp"][:, 0] / H, burn_y10=100 * S["burn_y10"],
                months_burn=S["months_burn"], comp_y5=S["comp_y5"], K_y5=S["K_y5"],
                job_changes=S["sw"], vol_changes=S["vol"], sep_h=S["sep_h"])
