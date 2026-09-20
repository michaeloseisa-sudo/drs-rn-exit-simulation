import numpy as np
from sim import *

T_BURN = 300
WIN = 36

def burnin(P, n, seed, collect=True):
    """Simulate careers from staggered entry (0-25 y experience at t=0) under behavioral policy."""
    rng0 = np.random.default_rng([seed, 999999])
    X0 = rng0.uniform(0, 25, n)
    start = T_BURN - np.round(12 * X0).astype(int)
    pop = Pop(n)
    pop.thin = (rng0.random(n) < P["thin_share"]).astype(float)
    active = np.zeros(n, bool)
    st = dict(rn_months=0, sep=0, sep_burn=0, to_out=0, sep_ten=[], coh_n=0, coh_left=0, burn_n=0, burn_m=0)
    hire_t = np.full(n, -10**6)
    for t in range(T_BURN):
        r = draws(np.random.default_rng([seed, t]), n)
        newc = np.where(start == t)[0]
        if len(newc):
            init_careers(P, pop, newc, np.random.default_rng([seed, t, 7]))
            active[newc] = True
            hire_t[newc] = t
        a = behavioral(P, pop, r)
        h_pre = active & (pop.sector == 0)
        xq = (P["a0"] + P["aF"] * (pop.F - 0.6) / 0.15 + P["aC"] * (-pop.c) + P["aT"] * (pop.ten < 12) + P["aT2"] * ((pop.ten >= 12) & (pop.ten < 24)) + P["aT5"] * (pop.ten >= 60))
        p_q = 1 / (1 + np.exp(-(xq + P["aB"] * (pop.B - 50) / 10)))
        p_cf = 1 / (1 + np.exp(-(xq + P["aB"] * (np.minimum(pop.B, 49.9) - 50) / 10)))
        a[~active] = 0
        dest, ev = step(P, pop, a, r)
        dest[~active] = -1
        for k in ev: 
            if ev[k].dtype == bool: ev[k] &= active
        if collect and t >= T_BURN - WIN:
            h = active & (pop.sector == 0)
            st["rn_months"] += h.sum()
            st["burn_n"] += (h & ev["burned"]).sum(); st["burn_m"] += h.sum()
            st["sep"] += ev["sep_h"].sum()
            st["sep_burn"] += (ev["sep_h"] & ev["burned"]).sum()
            st["to_out"] += ev["to_out"].sum()
            st["sep_ten"].append(ev["ten_at_sep"][ev["sep_h"]])
            st.setdefault("af_num", 0.0); st.setdefault("af_den", 0.0)
            hb = h_pre
            st["af_num"] += (p_q - p_cf)[hb].sum(); st["af_den"] += (p_q + P["h_exo"])[hb].sum()
        if collect:
            # first-year cohort: hospital hires in [T-60, T-13], followed 12 months
            left = ev["sep_h"] & (hire_t >= T_BURN - 60) & (hire_t <= T_BURN - 13) & (t - hire_t < 12)
            st["coh_left"] += left.sum()
        apply_moves(P, pop, dest, r)
        moved_h = (dest == 0)
        hire_t[dest >= 0] = t + 1
        if collect:
            st["coh_n"] += (moved_h & (t + 1 >= T_BURN - 60) & (t + 1 <= T_BURN - 13)).sum()
        if collect and len(newc):
            st["coh_n"] += ((start[newc] >= T_BURN - 60) & (start[newc] <= T_BURN - 13)).sum()
    pop.exp_years0 = X0
    return pop, st


def targets(st, pop):
    yrs = st["rn_months"] / 12
    ten = np.concatenate(st["sep_ten"]) if st["sep_ten"] else np.array([])
    bins = [0, 12, 24, 60, 120, 1e9]
    hist = np.histogram(ten, bins=bins)[0] / max(len(ten), 1)
    h = pop.sector == 0
    return dict(
        turnover=st["sep"] / yrs,
        first_year=st["coh_left"] / max(st["coh_n"], 1),
        burnout_prev=st["burn_n"] / st["burn_m"],
        sep_in_burnout=st["sep_burn"] / max(st["sep"], 1),
        burnout_attrib=st["af_num"] / st["af_den"],
        exit_sector=st["to_out"] / yrs,
        ten_lt1=hist[0], ten_1_2=hist[1], ten_2_5=hist[2], ten_5_10=hist[3], ten_gt10=hist[4],
        mean_tenure_yrs=pop.ten[h].mean() / 12,
        share_in_hospital=h.mean(),
    )

if __name__ == "__main__":
    import time
    t0 = time.time()
    pop, st = burnin(BASE, 6000, 1)
    print(time.time() - t0)
    for k, v in targets(st, pop).items(): print(k, round(float(v), 4))
