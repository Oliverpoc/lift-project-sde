"""Evaluate world-C-trained runs and add dC to the reused A0/B0 runs of duffing experiment C.
usage: python evaluate_C.py            -> runs/<tag>/eval.json (C runs), results.json
Rollout exactly as duffing/evaluate_run.py: x0 = XA0[0] (4000), generator seed 999, dt 0.1, horizon 3T.
"""
import json

import numpy as np
import torch

from worldc_common import DUFF, HERE, TK, T, DT, cov2, relfro, idx
import common as dc

torch.set_num_threads(2)
Z = np.load(DUFF / "data" / "truth.npz"); ZC = np.load(HERE / "data" / "truth_C.npz")
XA0, XB0, XC0 = Z["XA0"], Z["XB0"], ZC["XC0"]
CA, CB, CC = cov2(XA0), cov2(XB0), cov2(XC0)
HO_T = list(np.arange(0.5, T, 1.0))
CLS = ["mmn2", "td", "cfm", "sb"]
SEEDS = range(5)


def roll(model, x0, t_end):
    gen = torch.Generator().manual_seed(999)
    with torch.no_grad():
        return model.rollout(torch.tensor(x0, dtype=torch.float32), t_end, DT, e=0, gen=gen).numpy()


def fin(x):
    return float(x) if np.isfinite(x) else float("inf")


runs = {}
for c in CLS:
    for s in SEEDS:
        tag = f"C0_{c}_s{s}"
        d = HERE / "runs" / tag
        if not (d / "model.pt").exists():
            print("MISSING", tag); continue
        info = json.loads((d / "train.json").read_text())
        model = dc.build(info["method"], info["dh"], info["n_exp"])
        model.load_state_dict(torch.load(d / "model.pt")); model.eval()
        X = roll(model, XA0[0], 3 * T)
        C = cov2(X)
        m = dict(tag=tag, cls=c, seed=s, train_world="C", nparams=info["nparams"], wall_s=info["wall_s"],
                 M0=fin(np.mean([dc.w2_1d(X[idx(t)], XC0[idx(t)]) for t in TK[1:]])),
                 M0_ho=fin(np.mean([dc.w2_1d(X[idx(t)], XC0[idx(t)]) for t in HO_T])),
                 dA=relfro(C, CA), dB=relfro(C, CB), dC=relfro(C, CC), nonfinite=bool(not np.isfinite(X).all()))
        (d / "eval.json").write_text(json.dumps(m, indent=1))
        np.savez_compressed(d / "store.npz", C0=C, mean0=X.mean(1))
        runs[tag] = m
        print(tag, {k: round(v, 4) for k, v in m.items() if isinstance(v, float)}, flush=True)
# reused duffing runs (A- and B-trained): dA, dB as stored; dC from the stored covariance C0 (same rollout)
for w in "AB":
    for c in CLS:
        for s in SEEDS:
            tag = f"{w}0_{c}_s{s}"
            e = json.loads((DUFF / "runs" / tag / "eval.json").read_text())
            C = np.load(DUFF / "runs" / tag / "store.npz")["C0"]
            runs[tag] = dict(tag=tag, cls=c, seed=s, train_world=w, M0=e["M0"], M0_ho=e["M0_ho"],
                             dA=relfro(C, CA), dB=relfro(C, CB), dC=relfro(C, CC),
                             dA_stored=e["M2_in"], dB_stored=e["M2_in_vsB"], nonfinite=e["nonfinite"])

# ------------------------------------------------ aggregation + decision rule (PROTOCOL section 5) ----
PRIOR = dict(mmn2="A", td="B", cfm="B", sb="B")


def st(v):
    a = np.array(v, float)
    return dict(per_seed=a.tolist(), mean=float(a.mean()), sd=float(a.std(ddof=1)), median=float(np.median(a)))


agg = {}
for w in "ABC":
    for c in CLS:
        rs = [runs[f"{w}0_{c}_s{s}"] for s in SEEDS if f"{w}0_{c}_s{s}" in runs]
        g = {k: st([r[k] for r in rs]) for k in ("M0", "M0_ho", "dA", "dB", "dC")}
        means = {X: g["d" + X]["mean"] for X in "ABC"}
        g["landing"] = min(means, key=means.get)
        g["per_seed_landing"] = [min("ABC", key=lambda X: r["d" + X]) for r in rs]
        g["prior_world"] = PRIOR[c]
        g["closer_to_prior_than_C"] = bool(means[PRIOR[c]] < means["C"])
        g["per_seed_closer_to_prior_than_C"] = [bool(r["d" + PRIOR[c]] < r["dC"]) for r in rs]
        g["n"] = len(rs)
        agg[f"{w}0_{c}"] = g
best = {w: min(agg[f"{w}0_{c}"]["M0"]["mean"] for c in CLS) for w in "ABC"}
for w in "ABC":
    for c in CLS:
        g = agg[f"{w}0_{c}"]
        g["competent"] = bool(g["M0"]["mean"] <= 3 * best[w])
for c in CLS:
    g = agg[f"C0_{c}"]
    g["shift_vs_A_data"] = {X: abs(g["d" + X]["mean"] - agg[f"A0_{c}"]["d" + X]["mean"]) for X in "ABC"}
    g["shift_vs_B_data"] = {X: abs(g["d" + X]["mean"] - agg[f"B0_{c}"]["d" + X]["mean"]) for X in "ABC"}
    v = "thesis holds (lands at prior world, closer to it than to C)" if (g["landing"] == g["prior_world"] and g["closer_to_prior_than_C"]) \
        else ("learns C" if g["landing"] == "C" else "intermediate")
    g["verdict"] = v
    print(f"{c:5s} C-data: M0 {g['M0']['mean']:.4f} dA {g['dA']['mean']:.3f} dB {g['dB']['mean']:.3f} "
          f"dC {g['dC']['mean']:.3f} -> {g['landing']} | prior {g['prior_world']} closer-than-C {g['closer_to_prior_than_C']} "
          f"| seeds {g['per_seed_landing']} | {v} | competent {g['competent']}")
for w in "AB":
    for c in CLS:
        g = agg[f"{w}0_{c}"]
        print(f"{c:5s} {w}-data: dA {g['dA']['mean']:.3f} dB {g['dB']['mean']:.3f} dC {g['dC']['mean']:.3f} -> {g['landing']}")
check = max(abs(r["dA"] - r["dA_stored"]) + abs(r["dB"] - r["dB_stored"]) for r in runs.values() if "dA_stored" in r)
ver = json.loads((HERE / "data" / "verify_C.json").read_text())
out = dict(world_C=dict(tau=ver["tau"], Omega=ver["Omega"], verification={k: v for k, v in ver.items()}),
           select_C=json.loads((HERE / "data" / "select_C.json").read_text()),
           consistency_check_reused_dA_dB_max_abs_diff=check, best_M0=best, groups=agg, runs=runs)
(HERE / "results.json").write_text(json.dumps(out, indent=1))
print("reused-run consistency (recomputed vs stored dA,dB):", check)
