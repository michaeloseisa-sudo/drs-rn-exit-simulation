import json, numpy as np
from sim import BASE
from burnin import burnin, targets
from calibrate import CAL, VAL
P = dict(BASE); P.update(json.load(open("calibrated.json")))
rows = []
for seed in [202, 203, 204]:
    pop, st = burnin(P, 15000, seed)
    rows.append(targets(st, pop))
out = {}
for k in list(CAL) + list(VAL) + ["sep_in_burnout", "share_in_hospital"]:
    vals = np.array([r[k] for r in rows]); out[k] = (vals.mean(), vals.std())
    src = CAL.get(k) or VAL.get(k)
    tag = "CAL" if k in CAL else ("VAL" if k in VAL else "info")
    if src:
        ok = abs(vals.mean() - src[0]) <= src[1]
        print(f"{tag} {k:16s} sim={vals.mean():.4f} (sd {vals.std():.4f}) target={src[0]} tol={src[1]} {'PASS' if ok else 'FAIL'}")
    else:
        print(f"{tag} {k:16s} sim={vals.mean():.4f}")
json.dump({k: [float(a), float(b)] for k, (a, b) in out.items()}, open("calibration_heldout.json", "w"), indent=1)
