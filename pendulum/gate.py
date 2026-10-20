"""Apply the the pre-specified protocol rules (pendulum side of the pre-specified criteria K1-K4) to results.json -> gate.json.
usage: python gate.py"""
import json

import numpy as np

from common import ROOT, save_json

R = json.loads((ROOT / "results.json").read_text())
runs = R["runs"]
GROUPS = {
    "MMN_dh1": "mmn_dh1_", "MMN_dh2": "mmn_dh2_", "MMN_dh4": "mmn_dh4_",
    "obsTH(dh0)": "mmn_dh0_", "obsTD": "td_s", "obsTD_warm": "tdw_s", "OT-CFM": "cfm_s",
    "obsTD_clip": "tdclip_s", "obsTD_warm_clip": "tdwclip_s", "OT-CFM_clip": "cfmclip_s",
    "REF_delay(privileged)": "ref_s",
}
BASELINES = ["obsTH(dh0)", "obsTD", "obsTD_warm", "OT-CFM", "obsTD_clip", "obsTD_warm_clip", "OT-CFM_clip"]
CLASS_OF = {"obsTD_clip": "obsTD", "obsTD_warm_clip": "obsTD_warm", "OT-CFM_clip": "OT-CFM"}
METRICS = ["M0", "M0_held", "M1", "M2_in", "M2_out", "M3", "M2_in_vs_worldB", "M2_out_vs_worldB"]
TEMPORAL = ["M1", "M2_in", "M2_out", "M3"]
P = "MMN_dh2"          # pre-registered primary entry


def members(prefix):
    return [v for k, v in sorted(runs.items()) if k.startswith(prefix)]


def stat(vals):
    a = np.array(vals, float)
    fin = a[np.isfinite(a)]
    return dict(per_seed=[float(x) for x in a], median=float(np.median(a)),
                mean=float(a.mean()), std=float(a.std(ddof=1)) if len(a) > 1 else 0.0,
                n_nonfinite=int((~np.isfinite(a)).sum()), min=float(fin.min()) if len(fin) else None,
                max=float(fin.max()) if len(fin) else None)


S = {}
for g, p in GROUPS.items():
    ms = members(p)
    if not ms:
        continue
    S[g] = {m: stat([r[m] for r in ms]) for m in METRICS}
    S[g]["selects"] = [r["selects"] for r in ms]
    S[g]["nparams"] = sorted({r["nparams"] for r in ms})
    S[g]["seeds"] = [r["seed"] for r in ms]
    for m in ("R2_lin_x", "R2_mlp_x", "R2_lin_xh", "R2_mlp_xh", "final_train_loss"):
        if m in ms[0]:
            S[g][m] = stat([r[m] for r in ms])
    for m in ("M1_per_t", "M3_per_t", "M0_per_t"):
        S[g][m + "_median"] = np.median([r[m] for r in ms], 0).tolist()
med = lambda g, m: S[g][m]["median"]

G = {}
# competence (pre-registered): baseline median M0 and M0_held each <= 3x MMN_dh2 median
comp = [b for b in BASELINES if med(b, "M0") <= 3 * med(P, "M0") and med(b, "M0_held") <= 3 * med(P, "M0_held")]
G["competent_baselines"] = comp

# K1
k1 = {}
for m in TEMPORAL:
    pool_c = comp if comp else []
    bc = min(pool_c, key=lambda b: med(b, m)) if pool_c else None
    ba = min(BASELINES, key=lambda b: med(b, m))
    k1[m] = dict(mmn=med(P, m), best_competent=bc, best_competent_val=med(bc, m) if bc else None,
                 ratio_competent=med(P, m) / med(bc, m) if bc else None,
                 best_any=ba, best_any_val=med(ba, m), ratio_any=med(P, m) / med(ba, m))
pass_c = [m for m in TEMPORAL if k1[m]["ratio_competent"] is not None and k1[m]["ratio_competent"] <= 0.5]
pass_a = [m for m in TEMPORAL if k1[m]["ratio_any"] <= 0.5]
G["K1"] = dict(detail=k1, metrics_passing_vs_competent=pass_c, metrics_passing_vs_all_baselines=pass_a,
               verdict="PASS" if pass_c else "FAIL")

# K2
bestM0 = min(BASELINES, key=lambda b: med(b, "M0"))
bestH = min(BASELINES, key=lambda b: med(b, "M0_held"))
r0 = med(P, "M0") / med(bestM0, "M0"); rh = med(P, "M0_held") / med(bestH, "M0_held")
G["K2"] = dict(mmn_M0=med(P, "M0"), best_baseline_M0=(bestM0, med(bestM0, "M0")), ratio_M0=r0,
               mmn_M0_held=med(P, "M0_held"), best_baseline_M0_held=(bestH, med(bestH, "M0_held")), ratio_M0_held=rh,
               within_30pct=bool(r0 <= 1.3 and rh <= 1.3 and r0 >= 1 / 1.3 and rh >= 1 / 1.3),
               n_competent=len(comp),
               verdict="PASS" if (r0 <= 3 and rh <= 3 and len(comp) > 0) else "FAIL")

# K3
lad = {0: "obsTH(dh0)", 1: "MMN_dh1", 2: "MMN_dh2", 4: "MMN_dh4"}
k3 = {}
for m in ("M0", "M2_in"):
    v = {d: med(g, m) for d, g in lad.items()}
    k3[m] = dict(medians=v, improve_0_to_1=v[1] / v[0],
                 sat_2_over_1=v[2] / v[1], sat_4_over_1=v[4] / v[1])
imp = all(k3[m]["improve_0_to_1"] <= 0.5 for m in k3)
sat = all(k3[m]["sat_2_over_1"] >= 0.67 and k3[m]["sat_4_over_1"] >= 0.67 for m in k3)
for m in ("M1", "M3", "M2_out"):
    k3[m] = dict(medians={d: med(g, m) for d, g in lad.items()})
G["K3"] = dict(detail=k3, clear_improvement_0_to_1=imp, saturation_beyond_1=sat,
               verdict="PASS" if (imp and sat) else "FAIL")

# K4
def world(g):
    sel = S[g]["selects"]; nA = sel.count("A")
    by_median = "A" if med(g, "M2_in") < med(g, "M2_in_vs_worldB") else "B"
    maj = "A" if nA * 2 > len(sel) else "B"
    return dict(n_select_A=nA, n=len(sel), by_median=by_median, majority=maj,
                consistent=(by_median if by_median == maj else "mixed"),
                M2_in_vsA_median=med(g, "M2_in"), M2_in_vsB_median=med(g, "M2_in_vs_worldB"))
W = {g: world(g) for g in S}
sepAB = R["worldB"]["M2_in_truthB_vs_truthA"]; nfl = R["noise_floor"]["M2_in"]
mmn_A = W[P]["consistent"] == "A"
comp_classes = sorted({CLASS_OF.get(b, b) for b in comp})
other_B = [b for b in comp_classes if W[b]["consistent"] == "B"]
G["K4"] = dict(worlds=W, AB_separation_M2_in=sepAB, noise_floor_M2_in=nfl, separated=bool(sepAB >= 5 * nfl),
               mmn_selects_A=mmn_A, competent_baselines_selecting_B=other_B,
               verdict="PASS" if (mmn_A and other_B and sepAB >= 5 * nfl) else "FAIL")

# M4 summary (not a gate criterion)
ctr = R.get("probe_control_untrained", {})
m4 = {}
for dh in (1, 2, 4):
    g = f"MMN_dh{dh}"
    c = ctr.get(f"dh{dh}", {})
    m4[g] = dict(R2_mlp_xh=S[g]["R2_mlp_xh"], R2_lin_xh=S[g]["R2_lin_xh"],
                 R2_mlp_x_only=S[g]["R2_mlp_x"]["median"],
                 untrained_R2_mlp_xh=[v["R2_mlp_xh"] for v in c.values()],
                 untrained_R2_lin_xh=[v["R2_lin_xh"] for v in c.values()],
                 cross_seed_cca=R["latent_consistency_cca"][f"dh{dh}"])
m4["theta_window_reference"] = R["probe_reference_theta_window"]
G["M4"] = m4

save_json(dict(summary=S, gate=G), ROOT / "gate.json")

print(f"{'group':22s} {'params':>10s}" + "".join(f"{m:>16s}" for m in METRICS[:6]) + "  selects")
for g in S:
    print(f"{g:22s} {str(S[g]['nparams']):>10s}" + "".join(f"{med(g, m):16.4f}" for m in METRICS[:6]),
          "".join(S[g]["selects"]))
print("noise floor", {m: round(R["noise_floor"][m], 4) for m in ("M0", "M0_held", "M1", "M2_in", "M2_out", "M3")})
print("worldB", R["worldB"])
print("competent", comp)
for k in ("K1", "K2", "K3", "K4"):
    print(k, G[k]["verdict"])
print(json.dumps({k: G[k] for k in ("K1", "K2", "K3")}, indent=1, default=str)[:6000])
print(json.dumps({g: W[g] for g in W}, indent=0))
print(json.dumps(m4, indent=0, default=str)[:4000])
