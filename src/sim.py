"""Publicly calibrated career simulator for U.S. hospital staff RNs (monthly epochs).

Every parameter is tagged in PARAMS_SOURCES as CALIBRATED (fitted to a public target),
INPUT (taken directly from a public source) or ASSUMPTION (modeller's choice, no public source).
"""
import numpy as np
from scipy.stats import norm

H = 120                     # evaluation horizon, months
SIG_LOGW = np.log(135320 / 66030) / (2 * 1.2816)   # BLS OEWS May-2024 RN P10/P90 -> sd of log wage ~0.28
N_ACT = 5                   # 0 stay, 1 renegotiate, 2 exit->other healthcare org, 3 exit->other sector, 4 self-employment
ACT_NAMES = ["Stay", "Renegotiate", "Exit: other healthcare org", "Exit: other sector", "Exit: self-employment"]

BASE = dict(
    # ---- calibrated (initial values; overwritten by calibrate.py) ----
    a0=-5.0, aB=0.35, aT=0.9, aT2=0.0, aT5=0.0, ds0=-1.6, b0=44.0,
    # ---- inputs from public sources ----
    stayer_lag=0.002,        # Atlanta Fed WGT Jul-2026: overall 3.8% vs stayers 3.6% -> stayers lose 0.2 pp/yr vs market
    switch_prem=0.008,       # Atlanta Fed WGT Jul-2026: switchers 4.4% vs stayers 3.6% -> +0.8 pp on a switch
    burn_thr=50.0,           # burnout cut-point (dichotomous burnout definition, VHA AES / MBI items)
    turnover_cost=60090.0,   # NSI 2026: cost per staff-RN separation (secondary endpoint only)
    # ---- assumptions ----
    h_exo=0.004,             # non-work separations (relocation, personal, involuntary): ~4.7%/yr
    fit_mu=0.60, fit_sd=0.15, kF=0.15, sF=0.03, p_mgr=1 / 30, mgr_sd=0.12,
    kB=1 / 9, sB=3.0, bF=60.0, bOT=1.5, bS=12.0, bDep=8.0, dep_months=6,
    switch_sd=0.06, raise_neg=0.04, p_neg=0.35, neg_fail_fit=0.02,
    sector_pay=-0.05, sector_pay_sd=0.10, self_pay_sd=0.20,
    g_K=0.30, g_K_early=0.45,
    p_offer=0.85, p_offer_thin=0.35, p_offer_sector=0.60, p_offer_self=0.50,
    p_return=0.008, p_breneg=0.01, p_self=0.03, aF=-0.4, aC=0.2,
    thin_share=0.20, lam_burn=0.05, switch_cost=0.5,
)

WEIGHTS = {
    "baseline":      np.array([0.30, 0.25, 0.20, 0.15, 0.10]),
    "equal":         np.array([0.20, 0.20, 0.20, 0.20, 0.20]),
    "wellbeing":     np.array([0.20, 0.40, 0.15, 0.10, 0.15]),
    "compensation":  np.array([0.20, 0.15, 0.40, 0.15, 0.10]),
    "career":        np.array([0.20, 0.15, 0.15, 0.40, 0.10]),
}


class Pop:
    """Struct-of-arrays population state."""
    FIELDS = ["sector", "fj", "F", "d", "sj", "OT", "B", "D", "c", "K", "ten", "exp", "negcd",
              "thin", "c12", "Dobs", "hire_t"]

    def __init__(self, n):
        for f in self.FIELDS:
            setattr(self, f, np.zeros(n))
        self.c_hist = np.zeros((n, 12))
        self.n = n

    def subset(self, idx):
        p = Pop(len(idx))
        for f in self.FIELDS:
            setattr(p, f, getattr(self, f)[idx].copy())
        p.c_hist = self.c_hist[idx].copy()
        return p

    def copy(self):
        return self.subset(np.arange(self.n))


def draws(rng, n):
    """All random numbers for one month (common random numbers across policies)."""
    return dict(
        eF=rng.standard_normal(n), eB=rng.standard_normal(n), eOT=rng.standard_normal(n),
        ec=rng.standard_normal(n), umgr=rng.random(n), zmgr=rng.standard_normal(n),
        uexo=rng.random(n), uoff=rng.random(n), uneg=rng.random(n), uquit=rng.random(n),
        udest=rng.random(n), ubr=rng.random(n), uret=rng.random(n),
        nj=rng.standard_normal((n, 5)), ob=rng.standard_normal((n, 10)),
    )


def new_job(P, pop, idx, dest, z):
    """Start a new job for rows idx in destination sector dest (0 hospital, 1 other sector, 2 self)."""
    if len(idx) == 0:
        return
    z = z[idx]
    mu_f = np.where(dest == 0, P["fit_mu"], np.where(dest == 1, P["fit_mu"] - 0.02, P["fit_mu"] + 0.02))
    sd_f = np.where(dest == 2, 0.20, P["fit_sd"])
    pop.fj[idx] = np.clip(mu_f + sd_f * z[:, 0], 0.05, 0.95)
    pop.F[idx] = np.clip(pop.fj[idx] + 0.08 * z[:, 4], 0.0, 1.0)
    pop.d[idx] = np.where(dest == 0, z[:, 1], np.where(dest == 1, -0.7 + 0.8 * z[:, 1], 1.2 * z[:, 1]))
    pop.sj[idx] = np.clip(np.where(dest == 2, 0.7, 0.5) + 0.2 * z[:, 2], 0, 1)
    old = pop.sector[idx]
    same = (dest == 0) & (old == 0)
    dc = np.where(same, P["switch_prem"] + P["switch_sd"] * z[:, 3],
         np.where(dest == 0, P["switch_sd"] * z[:, 3],
         np.where(dest == 1, P["sector_pay"] + P["sector_pay_sd"] * z[:, 3], P["self_pay_sd"] * z[:, 3])))
    pop.c[idx] += dc / SIG_LOGW
    pop.K[idx] = np.where(dest == 0, pop.K[idx] - 2.0, np.where(dest == 1, pop.K[idx] * 0.90, pop.K[idx] * 0.95))
    pop.sector[idx] = dest
    pop.ten[idx] = 0
    pop.negcd[idx] = 12


def init_careers(P, pop, idx, rng):
    n = len(idx)
    z = rng.standard_normal((pop.n, 5))
    pop.sector[idx] = 0
    pop.c[idx] = 0.7 * rng.standard_normal(n)
    pop.K[idx] = 15 + 3 * rng.standard_normal(n)
    pop.B[idx] = 35 + 8 * rng.standard_normal(n)
    pop.D[idx] = 0; pop.Dobs[idx] = 0; pop.exp[idx] = 0
    pop.c_hist[idx] = pop.c[idx][:, None]
    new_job(P, pop, idx, np.zeros(n), z)
    pop.K[idx] += 2.0  # undo first-job capital loss


def masks(pop):
    m = np.ones((pop.n, N_ACT), bool)
    m[:, 1] = (pop.sector != 2) & (pop.negcd >= 12)
    m[:, 3] = pop.sector != 1
    m[:, 4] = pop.sector != 2
    return m


def observe(P, pop, r, t_rem):
    """Noisy Table-1 indicators + context -> feature vector in ~[0,1]."""
    ob = r["ob"]
    F, K = pop.F, pop.K
    lik = lambda x, e, s=0.6: np.clip(np.round(1 + 4 * x + s * e), 1, 5)
    task = np.clip(np.round(10 * F + 1.5 * ob[:, 0]), 0, 10)
    skill, auton = lik(F, ob[:, 1]), lik(F, ob[:, 2])
    sup, just = lik(F, ob[:, 3]), lik(F, ob[:, 4])
    career = lik(0.5 * F + 0.5 * K / 100, ob[:, 5])
    sched = lik(pop.sj, ob[:, 6])
    Bobs = np.clip(pop.B + 6 * ob[:, 7], 0, 100)
    comp_pct = 100 * norm.cdf(pop.c)
    growth = (pop.c - pop.c_hist[:, 0]) * SIG_LOGW + 0.036
    Kobs = np.clip(K + 3 * ob[:, 8], 0, 100)
    fit_hat = (task / 10 + (skill - 1) / 4 + (auton - 1) / 4 + (sup - 1) / 4 + (just - 1) / 4) / 5
    feats = np.stack([
        task / 10, (skill - 1) / 4, (auton - 1) / 4, comp_pct / 100, np.clip(growth, -0.2, 0.4) * 2.5,
        pop.OT / 40, (sched - 1) / 4, Bobs / 100, (sup - 1) / 4, (career - 1) / 4, (just - 1) / 4,
        pop.sector == 0, pop.sector == 1, pop.sector == 2, np.minimum(pop.exp / 300, 1),
        np.minimum(pop.ten / 120, 1), np.minimum(pop.negcd / 12, 1), np.minimum(pop.Dobs / 12, 1),
        np.broadcast_to(t_rem / H, (pop.n,)), Kobs / 100, pop.thin, pop.exp < 60,
    ], 1).astype(np.float32)
    return feats, dict(Bobs=Bobs, fit_hat=fit_hat, comp_pct=comp_pct)


def utility(P, pop, w, switched):
    WL = np.clip(0.6 * (1 - pop.OT / 20) + 0.4 * pop.sj, 0, 1)
    comps = np.stack([pop.F, 1 - pop.B / 100, norm.cdf(pop.c), pop.K / 100, WL], 1)
    u = comps @ w - P["lam_burn"] * (pop.D >= P["dep_months"]) - P["switch_cost"] * switched
    return u


def step(P, pop, a, r, rng_init=None):
    """Advance one month. a: (n,) actions (assumed feasible). Returns event dict."""
    n = pop.n
    in_h = pop.sector == 0
    # --- renegotiation ---
    neg = (a == 1) & (pop.negcd >= 12) & (pop.sector != 2)
    succ = neg & (r["uneg"] < P["p_neg"])
    pop.c[succ] += P["raise_neg"] / SIG_LOGW
    fail = neg & ~succ
    pop.fj[fail] = np.clip(pop.fj[fail] - P["neg_fail_fit"], 0.05, 0.95)
    pop.negcd[neg] = 0
    # --- exit attempts (take effect at end of month: one month notice worked) ---
    dest = np.full(n, -1)
    p_off = np.where(a == 2, np.where(pop.thin > 0, P["p_offer_thin"], P["p_offer"]),
            np.where(a == 3, P["p_offer_sector"], P["p_offer_self"]))
    ok = (a >= 2) & (r["uoff"] < p_off)
    dest[ok] = a[ok] - 2
    exo = in_h & (r["uexo"] < P["h_exo"])
    dest[exo] = 0
    # --- monthly dynamics ---
    mgr = r["umgr"] < P["p_mgr"]
    pop.fj[mgr] = np.clip(pop.fj[mgr] + P["mgr_sd"] * r["zmgr"][mgr], 0.05, 0.95)
    pop.F = np.clip(pop.F + P["kF"] * (pop.fj - pop.F) + P["sF"] * r["eF"], 0, 1)
    pop.OT = np.clip(4 + 4 * pop.d + 2 * r["eOT"], 0, 40)
    Bstar = (P["b0"] + P["bF"] * (0.6 - pop.F) + P["bOT"] * (pop.OT - 4) + P["bS"] * (0.5 - pop.sj)
             + P["bDep"] * (pop.D >= P["dep_months"]))
    pop.B = np.clip(pop.B + P["kB"] * (Bstar - pop.B) + P["sB"] * r["eB"], 0, 100)
    burned = pop.B >= P["burn_thr"]
    pop.D = np.where(burned, pop.D + 1, 0)
    Bobs = pop.B + 6 * r["ob"][:, 9]
    pop.Dobs = np.where(Bobs >= P["burn_thr"], pop.Dobs + 1, 0)
    pop.c += -P["stayer_lag"] / SIG_LOGW / 12 + 0.01 * r["ec"]
    g = np.where(pop.exp < 60, P["g_K_early"], P["g_K"]) * np.where(pop.sector == 0, 1.0, np.where(pop.sector == 1, 0.6, 0.8))
    pop.K = np.clip(pop.K + g * (0.4 + pop.F) * (1 - pop.K / 100), 0, 100)
    pop.ten += 1; pop.exp += 1; pop.negcd += 1
    pop.c_hist = np.roll(pop.c_hist, -1, 1); pop.c_hist[:, -1] = pop.c
    switched = dest >= 0
    ev = dict(switched=switched, sep_h=in_h & switched, ten_at_sep=pop.ten.copy(), burned=burned.copy(),
              to_out=in_h & switched & (dest > 0), exo=exo, vol=switched & ~exo)
    return dest, ev


def apply_moves(P, pop, dest, r):
    idx = np.where(dest >= 0)[0]
    new_job(P, pop, idx, dest[idx], r["nj"])


# ------------------------------------------------------------------ policies
def behavioral(P, pop, r):
    a = np.zeros(pop.n, int)
    h = pop.sector == 0
    x = (P["a0"] + P["aB"] * (pop.B - 50) / 10 + P["aF"] * (pop.F - 0.6) / 0.15 + P["aC"] * (-pop.c)
         + P["aT"] * (pop.ten < 12) + P["aT2"] * ((pop.ten >= 12) & (pop.ten < 24)) + P["aT5"] * (pop.ten >= 60))
    quit = h & (r["uquit"] < 1 / (1 + np.exp(-x)))
    p_sec = 1 / (1 + np.exp(-(P["ds0"] + 1.0 * (pop.B >= 50))))
    ud = r["udest"]
    a[quit] = np.where(ud[quit] < P["p_self"], 4, np.where(ud[quit] < P["p_self"] + p_sec[quit], 3, 2))
    br = h & ~quit & (pop.c < -0.25) & (pop.negcd >= 12) & (r["ubr"] < P["p_breneg"])
    a[br] = 1
    ret = (pop.sector > 0) & (r["uret"] < P["p_return"])
    a[ret] = 2
    return a


def expert(P, pop, info):
    a = np.zeros(pop.n, int)
    h = pop.sector == 0
    ex = h & (pop.Dobs >= 6) & (info["fit_hat"] < 0.50)
    ng = h & ~ex & (info["comp_pct"] < 40) & (info["fit_hat"] >= 0.65) & (pop.negcd >= 12)
    a[ex] = 2
    a[ng] = 1
    return a
