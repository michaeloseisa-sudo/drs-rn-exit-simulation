import pickle, json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from pools import load_P
from sim import WEIGHTS, ACT_NAMES
from evaluate import cvi, summarize
from calibrate import CAL, VAL
P = load_P(); ev = pickle.load(open("pools.pkl", "rb"))["eval"]
R = pickle.load(open("eval_results.pkl", "rb")); res, pert = R["res"], R["pert"]
rng = np.random.default_rng(0)
COMP = ["behavioral", "expert"]
def tci(x):
    x = np.asarray(x); m = x.mean()
    if len(x) < 2: return m, m, m
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)); return m, m - h, m + h
def boot(x, B=2000):
    idx = rng.integers(0, len(x), (B, len(x))); bm = x[idx].mean(1)
    return x.mean(), np.percentile(bm, 2.5), np.percentile(bm, 97.5)
out = {}
wb = WEIGHTS["baseline"]
seeds = [1, 2, 3, 4, 5]
drs = {s: res[f"drs:baseline_{s}"] for s in seeds}
# ---------- primary
out["cvi_abs"] = {c: float(cvi(P, res[c], wb).mean()) for c in COMP}
out["cvi_abs"]["drs_seeds"] = [float(cvi(P, drs[s], wb).mean()) for s in seeds]
out["primary"] = {}
for c in COMP:
    d = [float((cvi(P, drs[s], wb) - cvi(P, res[c], wb)).mean()) for s in seeds]
    within = [boot(cvi(P, drs[s], wb) - cvi(P, res[c], wb)) for s in seeds]
    m, lo, hi = tci(d)
    out["primary"][c] = dict(per_seed=d, mean=m, lo=lo, hi=hi, within=[list(map(float, w)) for w in within],
                             C1_pass=bool(min(d) > 0 and lo > 0))
# ---------- secondary
def sec(S): return {k: float(np.mean(v)) for k, v in summarize(P, S).items()}
out["secondary"] = {c: sec(res[c]) for c in COMP}
ds = [sec(drs[s]) for s in seeds]
out["secondary"]["drs"] = {k: float(np.mean([d[k] for d in ds])) for k in ds[0]}
out["secondary_diff"] = {}
for c in COMP:
    out["secondary_diff"][c] = {}
    for k in ds[0]:
        m, lo, hi = tci([d[k] - out["secondary"][c][k] for d in ds]); out["secondary_diff"][c][k] = [m, lo, hi]
cost = P["turnover_cost"]
out["employer_cost_per_100_RN_10y"] = {c: float(res[c]["sep_h"].mean() * 100 * cost) for c in COMP}
out["employer_cost_per_100_RN_10y"]["drs"] = float(np.mean([drs[s]["sep_h"].mean() for s in seeds]) * 100 * cost)
out["action_share"] = {c: list(map(float, res[c]["act_share"])) for c in COMP}
out["action_share"]["drs"] = list(map(float, np.mean([drs[s]["act_share"] for s in seeds], 0)))
# ---------- reward weights
out["weights"] = {}
for w in ["baseline", "equal", "wellbeing", "compensation", "career"]:
    ss = seeds if w == "baseline" else [1, 2, 3]; W = WEIGHTS[w]; out["weights"][w] = {}
    for c in COMP:
        d = [float((cvi(P, res[f"drs:{w}_{s}"], W) - cvi(P, res[c], W)).mean()) for s in ss]
        m, lo, hi = tci(d)
        out["weights"][w][c] = dict(per_seed=d, mean=m, lo=lo, hi=hi, all_positive=bool(min(d) > 0))
out["C2_pass"] = {c: all(out["weights"][w][c]["all_positive"] for w in out["weights"]) for c in COMP}
# ---------- subgroups (prespecified)
groups = {"All RNs": np.ones(ev.n, bool), "Early-career (0-5 y)": ev.exp0 <= 5, "Mid/late-career (>5 y)": ev.exp0 > 5,
          "High baseline burnout": ev.B0 >= 50, "Thin labor market": ev.thin > 0}
out["subgroups"] = {}
cvi_d = {s: cvi(P, drs[s], wb) for s in seeds}
for g, msk in groups.items():
    out["subgroups"][g] = dict(n=int(msk.sum()))
    for c in COMP:
        cc = cvi(P, res[c], wb)
        per = np.mean([cvi_d[s] - cc for s in seeds], 0)[msk]
        bd = np.mean([drs[s]["burn_y10"] for s in seeds], 0)[msk] * 100 - res[c]["burn_y10"][msk] * 100
        m, lo, hi = boot(per); bm, blo, bhi = boot(bd)
        per_seed = [float((cvi_d[s] - cc)[msk].mean()) for s in seeds]
        worse = bool((m < -1.0 and hi < 0) or (bm > 3 and blo > 0))
        out["subgroups"][g][c] = dict(cvi=[m, lo, hi], burn_y10_pp=[bm, blo, bhi], per_seed=per_seed, materially_worse=worse)
out["C3_pass"] = not any(out["subgroups"][g][c]["materially_worse"] for g in groups for c in COMP)
# early-career full table
em = groups["Early-career (0-5 y)"]
def secm(S, msk): return {k: float(np.mean(v[msk])) for k, v in summarize(P, S).items()}
out["early_table"] = {c: secm(res[c], em) for c in COMP}
eds = [secm(drs[s], em) for s in seeds]
out["early_table"]["drs"] = {k: float(np.mean([d[k] for d in eds])) for k in eds[0]}
out["early_table"]["drs_seed_range"] = {k: [float(min(d[k] for d in eds)), float(max(d[k] for d in eds))] for k in eds[0]}
# ---------- perturbations
out["pert"] = {}
for name, pr in pert.items():
    P2 = dict(P); P2.update(R["PERT"][name]); out["pert"][name] = {}
    for c in COMP:
        d = [float((cvi(P2, pr[f"drs:{s}"], wb) - cvi(P2, pr[c], wb)).mean()) for s in seeds]
        m, lo, hi = tci(d); out["pert"][name][c] = dict(per_seed=d, mean=m, lo=lo, hi=hi)
# ---------- calibration
ho = json.load(open("calibration_heldout.json"))
out["calibration"] = []
for k, (t, tol, src) in list(CAL.items()) + list(VAL.items()):
    sim = ho[k][0]; out["calibration"].append(dict(k=k, target=t, tol=tol, sim=sim, sd=ho[k][1], src=src,
        role="calibration" if k in CAL else "validation", pass_=bool(abs(sim - t) <= tol)))
out["C4_pass"] = all(r["pass_"] for r in out["calibration"] if r["role"] == "calibration")
json.dump(out, open("results.json", "w"), indent=1, default=float)

# ================= figures
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15, "axes.spines.top": False, "axes.spines.right": False})
INK, TEAL, SAND, RUST, GREY = "#1f2a36", "#1b7a6e", "#c9a227", "#b5452f", "#8a939c"
# calibration
lab = {"turnover": "Annual RN turnover", "first_year": "First-year RN turnover", "burnout_prev": "Burnout prevalence",
       "burnout_attrib": "Burnout-attributable exits", "exit_sector": "Exits out of sector", "ten_lt1": "Exits, tenure <1 y",
       "ten_1_2": "Exits, tenure 1-2 y", "ten_2_5": "Exits, tenure 2-5 y", "ten_5_10": "Exits, tenure 5-10 y",
       "ten_gt10": "Exits, tenure >10 y", "mean_tenure_yrs": "Mean RN tenure (validation)"}
cal = out["calibration"]
fig, ax = plt.subplots(figsize=(11, 6.2), dpi=200)
y = np.arange(len(cal))[::-1]
z = [(r["sim"] - r["target"]) / r["tol"] for r in cal]
ax.axvspan(-1, 1, color=TEAL, alpha=0.10, lw=0); ax.axvline(0, color=GREY, lw=1)
ax.scatter(z, y, s=90, color=[TEAL if r["role"] == "calibration" else SAND for r in cal], zorder=3, edgecolor=INK, lw=0.6)
ax.set_yticks(y, [lab[r["k"]] for r in cal]); ax.set_xlim(-1.6, 1.6)
ax.set_xlabel("Held-out error ÷ prespecified tolerance  (inside ±1 = pass)")
fig.tight_layout(); fig.savefig("fig_calibration.png"); plt.close()
# primary per seed
fig, ax = plt.subplots(figsize=(11, 5.4), dpi=200)
for i, c in enumerate(COMP):
    p = out["primary"][c]
    ax.scatter([i] * 5, p["per_seed"], s=80, color=GREY, zorder=2, label="Independent training seed" if i == 0 else None)
    ax.errorbar(i + 0.18, p["mean"], yerr=[[p["mean"] - p["lo"]], [p["hi"] - p["mean"]]], fmt="o", ms=11, color=TEAL if p["lo"] > 0 else RUST, capsize=8, lw=2.5, label="Mean, 95% CI across seeds" if i == 0 else None)
ax.axhline(0, color=INK, lw=1.2)
ax.set_xticks([0, 1], ["DRS − behavioral policy", "DRS − expert-guided policy"]); ax.set_xlim(-0.6, 1.8)
ax.set_ylabel("Difference in Career Value Index (points)"); ax.legend(frameon=False, loc="upper right")
fig.tight_layout(); fig.savefig("fig_primary.png"); plt.close()
# weights
names = ["baseline", "equal", "wellbeing", "compensation", "career"]
fig, ax = plt.subplots(figsize=(11, 5.4), dpi=200)
for j, c in enumerate(COMP):
    for i, w in enumerate(names):
        d = out["weights"][w][c]["per_seed"]; x = i + (j - 0.5) * 0.3
        ax.scatter([x] * len(d), d, s=55, color=[GREY, SAND][j], alpha=0.9, zorder=2)
        ax.scatter([x], [np.mean(d)], s=160, marker="_", color=[INK, RUST][j], lw=3, zorder=3)
ax.axhline(0, color=INK, lw=1.2)
ax.set_xticks(range(5), ["Baseline", "Equal", "Wellbeing-\npriority", "Compensation-\npriority", "Career-\npriority"])
ax.set_ylabel("DRS advantage (CVI points)")
from matplotlib.lines import Line2D
ax.legend([Line2D([], [], marker="o", ls="", color=GREY), Line2D([], [], marker="o", ls="", color=SAND)],
          ["vs behavioral", "vs expert-guided"], frameon=False, loc="best")
fig.tight_layout(); fig.savefig("fig_weights.png"); plt.close()
# subgroups forest
gl = list(groups)
fig, axs = plt.subplots(1, 2, figsize=(12, 5.4), dpi=200, sharey=True)
for j, c in enumerate(COMP):
    ax = axs[j]
    for i, g in enumerate(gl):
        m, lo, hi = out["subgroups"][g][c]["cvi"]; yy = len(gl) - 1 - i
        ax.plot([lo, hi], [yy, yy], color=INK, lw=2.5); ax.scatter([m], [yy], s=90, color=TEAL if lo > 0 else (RUST if hi < 0 else SAND), zorder=3, edgecolor=INK)
    ax.axvline(0, color=GREY, lw=1.2); ax.axvline(-1.0, color=RUST, lw=1, ls="--")
    ax.set_title(["vs behavioral policy", "vs expert-guided policy"][j], fontsize=15)
    ax.set_xlabel("DRS − comparator, CVI points (95% CI)")
axs[0].set_yticks(range(len(gl))[::-1], [f"{g}  (n={out['subgroups'][g]['n']:,})" for g in gl])
fig.tight_layout(); fig.savefig("fig_subgroups.png"); plt.close()
print(json.dumps({k: out[k] for k in ["cvi_abs", "C2_pass", "C3_pass", "C4_pass"]}, indent=1, default=float))
for c in COMP: print(c, out["primary"][c]["per_seed"], out["primary"][c]["mean"], out["primary"][c]["lo"], out["primary"][c]["hi"], out["primary"][c]["C1_pass"])
