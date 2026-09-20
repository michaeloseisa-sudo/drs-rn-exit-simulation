import sys, json, numpy as np, pandas as pd
sys.path.insert(0, '.')
from load import load
W = ["rkrnwgta"] + [f"rkrnwgta{i}" for i in range(1, 81)]
cols = ["pn_empset_comb_puf", "pn_travel", "pn_lftwrk", "le_lve_brnout", "le_wrknurs", "le_lve_retire",
        "cv_burnt_2019", "cv_burnt_2020", "cv_burnt_2021", "pn_howlong", "re_cnsrdlv", "re_clvyear",
        "re_lve_brnout", "nh_emppy", "nh_postn", "pn_satisfd", "pn_burnout", "yrsnc_gp_puf"] + W
d = load(cols)
wt = d[W].astype(float).values
def est(num, den):
    """Weighted proportion with SDR standard error (Fay & Train 1995; 4/80 factor)."""
    num = num.values.astype(float); den = den.values.astype(float)
    th = (wt * num[:, None]).sum(0) / (wt * den[:, None]).sum(0)
    se = np.sqrt(4 / 80 * ((th[1:] - th[0]) ** 2).sum())
    return th[0], se, int(den.sum())
hosp = (d.pn_empset_comb_puf == "1") & (d.pn_travel == "2")
out = {}
def add(name, num, den, desc):
    p, se, n = est(num & den, den); out[name] = dict(p=p, se=se, n=n, desc=desc)
    print(f"{name:28s} {100*p:6.1f}% (SE {100*se:.1f}; 95% CI {100*(p-1.96*se):.1f}-{100*(p+1.96*se):.1f}; n={n:,})  {desc}")
worked = lambda c: d[c].isin(["2", "3", "4", "5", "6"])
for y in ["2019", "2020", "2021"]:
    c = f"cv_burnt_{y}"
    add(f"burnout_weekly_{y}", d[c].isin(["5", "6"]), hosp & worked(c), f"Hospital RNs: burned out a few times a week or more in {y}")
add("left_position", d.pn_lftwrk == "1", hosp & d.pn_lftwrk.isin(["1", "2"]), "Hospital RNs who later left their Dec-31-2021 primary position (by survey response)")
lv = hosp & (d.pn_lftwrk == "1") & d.le_lve_brnout.isin(["1", "2"])
add("leavers_cite_burnout", d.le_lve_brnout == "1", lv, "Hospital RN leavers citing burnout as a reason (Section D1)")
lv2 = lv & (d.le_lve_retire != "1")
add("leavers_cite_burnout_noretire", d.le_lve_brnout == "1", lv2, "Same, excluding leavers citing retirement")
add("leavers_left_nursing_noretire", d.le_wrknurs == "2", lv2 & d.le_wrknurs.isin(["1", "2"]), "Non-retiring hospital RN leavers who did not continue in nursing")
lva = hosp & (d.pn_lftwrk == "1") & d.le_wrknurs.isin(["1", "2"])
add("leavers_left_nursing_all", d.le_wrknurs == "2", lva, "All hospital RN leavers who did not continue in nursing")
ten = hosp & d.pn_howlong.isin(["1", "2", "3"])
for code, lab in [("1", "lt1"), ("2", "1_5"), ("3", "gt5")]:
    add(f"tenure_{lab}", d.pn_howlong == code, ten, f"Hospital RNs: time with employer, category {lab}")
add("considered_leaving_pastyear", d.re_clvyear == "1", hosp & (d.pn_lftwrk == "2") & d.re_clvyear.isin(["1", "2", "L"]) & d.re_cnsrdlv.isin(["1", "2"]), "Hospital RN stayers who considered leaving in the past year")
add("changed_job_2021", d.nh_postn == "2", hosp & d.nh_postn.isin(["1", "2"]), "Hospital RNs whose Dec-2020 position/employer differed from Dec-2021")
json.dump(out, open("nssrn_targets.json", "w"), indent=1)
print("weighted hospital non-traveler RNs:", round((wt[:, 0] * hosp.values).sum()))
