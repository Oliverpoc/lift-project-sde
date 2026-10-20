"""Duffing gate criteria K1, K2, K4, K5 exactly as in the pre-specified protocol section 5.
usage: python gate.py -> verdict.json"""
import json

import numpy as np

from common import ROOT, save_json

R = json.loads((ROOT / "results.json").read_text())
G = R["groups"]
mean = lambda g, m: G[g][m]["mean"] if g in G and m in G[g] else None
V = {}


def competent(classes, two_exp=False):
    """classes: list of group names in one setting. Returns dict name -> bool (+ the numbers)."""
    def m0(g):
        return (mean(g, "M0") + mean(g, "mu1_M0")) / 2 if two_exp else mean(g, "M0")
    def m0h(g):
        return (mean(g, "M0_ho") + mean(g, "mu1_M0_ho")) / 2 if two_exp else mean(g, "M0_ho")
    cl = [c for c in classes if c in G]
    b0, bh = min(m0(c) for c in cl), min(m0h(c) for c in cl)
    return {c: dict(competent=bool(m0(c) <= 3 * b0 and m0h(c) <= 3 * bh), M0=m0(c), M0_ho=m0h(c),
                    ratio_M0_to_best=m0(c) / b0, ratio_M0ho_to_best=m0h(c) / bh) for c in cl}


P = "A0_mmn2"
# ---------------------------------------------------------------- K1 / K2 (setting A)
BASE_A = ["A0_mmn0", "A0_td", "A0_cfm", "A0_sb"]
compA = competent([P] + BASE_A)
V["competence_A"] = compA
cb = [b for b in BASE_A if compA.get(b, {}).get("competent")]
k1 = {}
for stat in ("M2_in", "M1", "M3"):
    mm = mean(P, stat)
    if not cb:
        k1[stat] = dict(verdict="not evaluable")
        continue
    bv = {b: (mean(b, "M1_best") if (stat == "M1" and "M1_best" in G[b]) else mean(b, stat)) for b in cb}
    best = min(bv, key=bv.get)
    k1[stat] = dict(mmn=mm, best_competent_baseline=best, baseline=bv[best], ratio=mm / bv[best],
                    passes=bool(mm <= 0.5 * bv[best]))
V["K1"] = dict(detail=k1, verdict="PASS" if any(d.get("passes") for d in k1.values()) else "FAIL")
nb = sum(compA.get(b, {}).get("competent", False) for b in ["A0_td", "A0_cfm", "A0_sb"])
k2ok = compA[P]["competent"] and compA.get("A0_td", {}).get("competent", False) and nb >= 2
ideal = {b: dict(M0_ratio_to_MMN=mean(b, "M0") / mean(P, "M0"), M0ho_ratio_to_MMN=mean(b, "M0_ho") / mean(P, "M0_ho"))
         for b in ["A0_td", "A0_cfm", "A0_sb", "A0_mmn0"] if b in G}
V["K2"] = dict(n_competent_among_td_cfm_sb=nb, ratios_to_MMN=ideal, verdict="PASS" if k2ok else "FAIL")

# ---------------------------------------------------------------- K4 (settings A and C)
k4 = {}
classes = ["mmn2", "mmn1", "mmn0", "td", "cfm", "sb"]
sel = {}
for W in "AB":
    comp = competent([f"{W}0_{c}" for c in classes if c != "mmn1"])
    rows = {}
    for c in classes:
        g = f"{W}0_{c}"
        if g not in G:
            continue
        dA, dB = mean(g, "M2_in"), mean(g, "M2_in_vsB")
        rows[c] = dict(dA=dA, dB=dB, selects="A" if dA < dB else "B", M0=mean(g, "M0"),
                       competent=comp.get(g, {}).get("competent"))
        sel[(W, c)] = rows[c]["selects"]
    k4[f"data_world_{W}"] = rows
ok_worlds = {}
partners = []
for b in ["td", "cfm", "sb", "mmn0"]:
    good = True
    for W in "AB":
        r = k4[f"data_world_{W}"]
        if b not in r or "mmn2" not in r:
            good = False; break
        good &= bool(r[b]["competent"] and r["mmn2"]["competent"] and r[b]["selects"] != r["mmn2"]["selects"]
                     and abs(r[b]["dA"] - r["mmn2"]["dA"]) >= 0.5)
    good &= all(((("A", c) in sel) and sel[("A", c)] == sel.get(("B", c))) for c in ("mmn2", b))
    if good:
        partners.append(b)
k4["baselines_satisfying"] = partners
k4["selection_invariant_to_data_world"] = {c: (sel.get(("A", c)), sel.get(("B", c))) for c in classes}
V["K4"] = dict(detail=k4, verdict="PASS" if partners else "FAIL")

# ---------------------------------------------------------------- K5 (setting D vs A)
dist = R["distinguishability"]
floor = max(dist["W2_A0_vs_A0b_noise_max"], dist["W2_A1_vs_A1b_noise_max"])
a_ok = dist["W2_A1_vs_B1_max"] >= 10 * floor
S0c = [P, "A0_td", "A0_cfm", "A0_sb"]
S1c = ["A01_mmn2", "A01_tdlab", "A01_tdsh", "A01_cfm", "A01_sb"]
k5 = dict(distinguishable=dict(W2_A1_vs_B1_max=dist["W2_A1_vs_B1_max"], noise_floor_max=floor, passes=bool(a_ok)))
if all(c in G for c in S1c):
    cA = competent(S0c); cD = competent(S1c, two_exp=True)
    v0 = {c: mean(c, "M2_in") for c in S0c if cA[c]["competent"]}
    v1 = {c: mean(c, "M2_in") for c in S1c if cD[c]["competent"]}
    v1m = {c: mean(c, "mu1_M2_in") for c in S1c if cD[c]["competent"]}
    S0 = max(v0.values()) - min(v0.values()); S1 = max(v1.values()) - min(v1.values())
    k5.update(competence_A=cA, competence_D=cD, M2_in_A=v0, M2_in_D=v1, mu1_M2_in_D=v1m, S0=S0, S1=S1,
              S1_mu1=max(v1m.values()) - min(v1m.values()), ratio=S1 / S0,
              all_classes_D={c: dict(M2_in=mean(c, "M2_in"), mu1_M2_in=mean(c, "mu1_M2_in"),
                                     M0=mean(c, "M0"), mu1_M0=mean(c, "mu1_M0")) for c in S1c})
    V["K5"] = dict(detail=k5, verdict="PASS" if (a_ok and S1 <= 0.5 * S0) else "FAIL")
else:
    V["K5"] = dict(detail=k5, verdict="NOT RUN")
save_json(V, ROOT / "verdict.json")
for k in ("K1", "K2", "K4", "K5"):
    print(k, V[k]["verdict"])
print(json.dumps(V["K1"]["detail"], indent=0)[:1500])
