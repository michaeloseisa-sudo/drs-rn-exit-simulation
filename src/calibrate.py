import numpy as np, json
from scipy.optimize import minimize
from sim import BASE
from burnin import burnin, targets

CAL = {  # name: (target, tolerance, source)
 "turnover":       (0.176, 0.015, "NSI 2026 report (CY2025): national staff RN turnover 17.6%"),
 "first_year":     (0.227, 0.025, "NSI 2026 report: first-year RN turnover 22.7%"),
 "burnout_prev":   (0.340, 0.030, "Mohr et al. 2025 (VHA AES 2023): RN burnout 34.0% (levels I-II; range 32.6-37.5%)"),
 "burnout_attrib": (0.315, 0.050, "Shah et al. 2021 (2018 NSSRN): 31.5% of RNs who left cited burnout (burnout-attributable share of separations)"),
 "exit_sector":    (0.0315, 0.0065, "Fusilier et al. 2026 (IPUMS CPS): labor-force exits 2.5-3.8%/yr"),
 "ten_lt1":  (0.290, 0.05, "NSI 2026: share of RN separations with <1 y tenure 29.0%"),
 "ten_1_2":  (0.219, 0.05, "NSI 2026: 1-2 y 21.9%"),
 "ten_2_5":  (0.277, 0.05, "NSI 2026: 2-5 y 27.7%"),
 "ten_5_10": (0.105, 0.05, "NSI 2026: 5-10 y 10.5%"),
 "ten_gt10": (0.108, 0.05, "NSI 2026: >10 y 10.8%"),
}
VAL = {
 "mean_tenure_yrs": (6.8, 1.0, "NSI 2026: average RN tenure 6.8 y (definition not fully specified)"),
}
KEYS = ["a0", "aB", "aT", "aT2", "aT5", "ds0", "b0"]

def loss(x, n=6000, seed=101):
    P = dict(BASE); P.update(dict(zip(KEYS, x))); P["aB"] = abs(P["aB"])  # sign constraint aB>=0
    pop, st = burnin(P, n, seed)
    m = targets(st, pop)
    return sum(((m[k] - v[0]) / v[1]) ** 2 for k, v in CAL.items()), m

if __name__ == "__main__":
    x0 = np.array([-4.45, 0.57, 0.5, 0.5, -0.8, -1.84, 47.5])
    best = [1e9, None]
    def f(x):
        L, m = loss(x)
        if L < best[0]: best[0], best[1] = L, (x.copy(), m)
        return L
    res = minimize(f, x0, method="Nelder-Mead", options=dict(maxfev=450, xatol=1e-3, fatol=1e-3,
                   initial_simplex=np.vstack([x0] + [x0 + np.eye(7)[i] * s for i, s in enumerate([0.4, 0.15, 0.4, 0.4, 0.5, 0.4, 2.0])])))
    x, m = best[1]; x = x.copy(); x[1] = abs(x[1])
    print("loss", best[0]); print(dict(zip(KEYS, np.round(x, 4))))
    for k in CAL: print(k, round(m[k], 4), CAL[k][0])
    json.dump(dict(zip(KEYS, [float(v) for v in x])), open("calibrated.json", "w"), indent=1)
