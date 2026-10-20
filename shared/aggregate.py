"""Aggregate eval.json files and apply the pre-specified protocol section 4 criteria -> results.json, verdict.json.
usage: python aggregate.py"""
import json

import numpy as np

from common import CFG, TK, ROOT, w2_1d, save_json

SEEDS = range(5)
CLASSES = ["MMN", "obs-TD", "SB"]
TAGS = {  # (class, setting) -> tag prefix
    ("MMN", "mu0"): "A0_mmn2", ("MMN", "mu1"): "A1_mmn2", ("MMN", "sh"): "A01_mmn2", ("MMN", "c"): "A01_mmnc2",
    ("obs-TD", "mu0"): "A0_td", ("obs-TD", "mu1"): "A1_td", ("obs-TD", "sh"): "A01_tdsh", ("obs-TD", "c"): "A01_tdlab",
    ("SB", "mu0"): "A0_sb", ("SB", "mu1"): "A1_sb", ("SB", "sh"): "A01_sbsh", ("SB", "c"): "A01_sb",
}
KEYS = ["M0_0", "epsC_0", "dB_0", "M0_1", "epsC_1", "dB_1"]


def load(prefix):
    out = {k: [] for k in KEYS}; out["seeds"] = []; out["nparams"] = None
    for s in SEEDS:
        f = ROOT / "runs" / f"{prefix}_s{s}" / "eval.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text()); out["seeds"].append(s); out["nparams"] = d["nparams"]
        for k in KEYS:
            if k in d:
                out[k].append(d[k])
    for k in KEYS:
        v = np.array(out[k], float)
        if len(v):
            out[k + "_mean"] = float(np.mean(v)); out[k + "_sd"] = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
    return out


R = {f"{c}|{s}": load(TAGS[(c, s)]) for (c, s) in TAGS}

# noise floors (independent truth samples) -- same definitions as evaluate.py
Z = np.load(ROOT / "data" / "truth.npz")
dt, T = CFG["dt_model"], CFG["T"]
idx = lambda t: int(round(t / dt))
import evaluate_helpers as H  # noqa: E402
nf = dict(
    epsC_0=H.relfro(H.cov2(Z["XA0b"]), H.cov2(Z["XA0"])), epsC_1=H.relfro(H.cov2(Z["XA1b"]), H.cov2(Z["XA1"])),
    W2_0=float(np.mean([w2_1d(Z["XA0b"][idx(t)], Z["XA0"][idx(t)]) for t in TK[1:]])),
    W2_1=float(np.mean([w2_1d(Z["XA1b"][idx(t)], Z["XA1"][idx(t)]) for t in TK[1:]])),
    W2_0_max=float(max(w2_1d(Z["XA0b"][idx(t)], Z["XA0"][idx(t)]) for t in TK)),
    W2_A0_B0_max=float(max(w2_1d(Z["XA0"][idx(t)], Z["XB0"][idx(t)]) for t in TK)),
    W2_A1_B1_max=float(max(w2_1d(Z["XA1"][idx(t)], Z["XB1"][idx(t)]) for t in TK)),
    W2_1_max=float(max(w2_1d(Z["XA1b"][idx(t)], Z["XA1"][idx(t)]) for t in TK)),
    epsC_B0_vs_A0=H.relfro(H.cov2(Z["XB0"]), H.cov2(Z["XA0"])),
)

V = {"noise": nf}
# ---- per joint model: fit-comparable, Delta, t
rows = {}
for c in CLASSES:
    r0, r1 = R[f"{c}|mu0"], R[f"{c}|mu1"]
    for fam in ("sh", "c"):
        r = R[f"{c}|{fam}"]
        if "epsC_0_mean" not in r:
            continue
        fc = (r["M0_0_mean"] <= 1.5 * r0["M0_0_mean"]) and (r["M0_1_mean"] <= 1.5 * r1["M0_1_mean"])
        D = r0["epsC_0_mean"] - r["epsC_0_mean"]
        se = np.sqrt(r0["epsC_0_sd"] ** 2 / len(r0["epsC_0"]) + r["epsC_0_sd"] ** 2 / len(r["epsC_0"]))
        rows[f"{c}|{fam}"] = dict(
            n=len(r["seeds"]), nparams=r["nparams"],
            M0_0_ratio=r["M0_0_mean"] / r0["M0_0_mean"], M0_1_ratio=r["M0_1_mean"] / r1["M0_1_mean"],
            fit_comparable=bool(fc), Delta=float(D), t=float(D / se) if se > 0 else float("nan"),
            dB0_change=r["dB_0_mean"] - r0["dB_0_mean"],
            epsC1_change=r1["epsC_1_mean"] - r["epsC_1_mean"],
        )
V["joint"] = rows


def spread(keys):
    v = [R[k]["epsC_0_mean"] for k in keys if k in R and "epsC_0_mean" in R[k]]
    return (float(max(v) - min(v)) if len(v) >= 2 else None), len(v)


S0, _ = spread([f"{c}|mu0" for c in CLASSES])
fc_sh = [f"{c}|sh" for c in CLASSES if rows.get(f"{c}|sh", {}).get("fit_comparable")]
fc_c = [f"{c}|c" for c in CLASSES if rows.get(f"{c}|c", {}).get("fit_comparable")]
S_sh, n_sh = spread(fc_sh); S_c, n_c = spread(fc_c)
S_sh_all, _ = spread([f"{c}|sh" for c in CLASSES]); S_c_all, _ = spread([f"{c}|c" for c in CLASSES])
S1_mu0only_vs_mu1only = spread([f"{c}|mu1" for c in CLASSES])[0]
V["spreads"] = dict(S0=S0, S_sh=S_sh, n_sh=n_sh, S_c=S_c, n_c=n_c, S_sh_all=S_sh_all, S_c_all=S_c_all,
                    members_sh=fc_sh, members_c=fc_c)
V["spreads_mu1test"] = dict(  # same with epsC_1 (information only)
    mu1only=float(np.ptp([R[f"{c}|mu1"]["epsC_1_mean"] for c in CLASSES])),
    sh_all=float(np.ptp([R[f"{c}|sh"]["epsC_1_mean"] for c in CLASSES])),
    c_all=float(np.ptp([R[f"{c}|c"]["epsC_1_mean"] for c in CLASSES])))

# ---- criteria
a = all(rows[k]["fit_comparable"] for k in ("MMN|sh", "MMN|c", "obs-TD|c", "SB|c"))
b = (nf["W2_A0_B0_max"] <= 2 * nf["W2_0_max"]) and S0 >= 0.5
c1 = rows["MMN|sh"]["Delta"] >= 0.10 and rows["MMN|sh"]["t"] >= 2.0
c2 = (S_sh is not None) and S_sh <= 0.5 * S0
d = (S_c is not None) and S_c >= 0.5 * S0 and S_c >= 0.20
V["criteria"] = dict(
    a=dict(pass_=bool(a), detail={k: (rows[k]["M0_0_ratio"], rows[k]["M0_1_ratio"], rows[k]["fit_comparable"]) for k in rows}),
    b=dict(pass_=bool(b), W2_A0_B0_max=nf["W2_A0_B0_max"], W2_noise_max=nf["W2_0_max"], S0=S0),
    c=dict(pass_=bool(c1 or c2), c1=bool(c1), c2=bool(c2), Delta_MMN_sh=rows["MMN|sh"]["Delta"], t_MMN_sh=rows["MMN|sh"]["t"],
           S_sh=S_sh, n_sh=n_sh, thr_c2=0.5 * S0),
    d=dict(pass_=bool(d), S_c=S_c, n_c=n_c, thr=max(0.5 * S0, 0.20)),
)
V["claim_version"] = "strong" if (a and b and (c1 or c2) and d) else "conservative"
save_json(R, ROOT / "results.json"); save_json(V, ROOT / "verdict.json")

print("noise", {k: round(v, 4) for k, v in nf.items()})
print(f"{'model':16s} n  M0_0           M0_1           epsC_0         dB_0           epsC_1")
for k, r in R.items():
    if "epsC_0_mean" in r or "epsC_1_mean" in r:
        f = lambda q: f"{r.get(q+'_mean', np.nan):.3f}+-{r.get(q+'_sd', np.nan):.3f}"
        print(f"{k:16s} {len(r['seeds'])}  {f('M0_0'):14s} {f('M0_1'):14s} {f('epsC_0'):14s} {f('dB_0'):14s} {f('epsC_1')}")
for k, r in rows.items():
    print(k, {q: (round(v, 3) if isinstance(v, float) else v) for q, v in r.items()})
print("spreads", V["spreads"], V["spreads_mu1test"])
print("criteria", json.dumps(V["criteria"], indent=0, default=str)[:2000])
print("claim:", V["claim_version"])
