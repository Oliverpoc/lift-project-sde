"""Aggregate results_<ds>.json (seed mean / sd) and apply the pre-registered gate rules (the pre-specified protocol section 6).
usage: python gate.py ds1 [ds2 ...]   -> prints tables, writes summary_<ds>.json and gate.json"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CLASSES = ["mmn2", "mmn1", "mmn0", "td", "cfm", "sot", "entot"]
NAMES = dict(mmn2="MMN d_h=2", mmn1="MMN d_h=1", mmn0="obs-TH", td="obs-TD", cfm="OT-CFM", sot="static OT",
             entot="entropic OT")
BASE = ["mmn0", "td", "cfm", "sot", "entot"]
KEYS = ["par_w2", "par_mmd", "fit_w2", "phase_mae", "phase_corr", "phase_corr_FL", "phase_corr_FL", "phase_mae_hold", "within_bin_spearman",
        "T1_dir_err", "T2_cycle", "T3_shortcut", "T4_track", "M2t_err", "diverged_frac"]


def agg(res):
    out = {}
    for c in CLASSES:
        runs = [v for k, v in res.items() if k.startswith(c + "_s")]
        if not runs:
            continue
        out[c] = {"n": len(runs)}
        for key in KEYS:
            vals = [r[key] for r in runs if key in r]
            if vals:
                a = np.array(vals, float)
                out[c][key] = dict(mean=float(np.mean(a)), sd=float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
                                   vals=[float(v) for v in a])
    for r in ("REF_pca_angle", "REF_centroid_polygon", "TRUTH_cohort"):
        if r in res:
            out[r] = {k: dict(mean=float(v), sd=0.0, vals=[float(v)]) for k, v in res[r].items()}
    return out


def gate(S):
    m = lambda c, k: S[c][k]["mean"]
    sd = lambda c, k: S[c][k]["sd"]
    g = {}
    bb = min(BASE, key=lambda c: m(c, "par_w2"))
    g["K1"] = dict(mmn=m("mmn2", "par_w2"), best=bb, best_val=m(bb, "par_w2"),
                   ratio=m("mmn2", "par_w2") / m(bb, "par_w2"))
    g["K1"]["pass"] = g["K1"]["ratio"] <= 1.2
    bb = min(BASE, key=lambda c: m(c, "phase_mae"))
    g["K2"] = dict(mmn=m("mmn2", "phase_mae"), best=bb, best_val=m(bb, "phase_mae"),
                   ratio=m("mmn2", "phase_mae") / m(bb, "phase_mae"))
    g["K2"]["pass"] = g["K2"]["ratio"] <= 0.75
    refmin = min(m("REF_pca_angle", "phase_mae"), m("REF_centroid_polygon", "phase_mae"))
    g["K4"] = dict(mmn=m("mmn2", "phase_mae"), transcriptome_only_best=refmin,
                   beats_transcriptome=m("mmn2", "phase_mae") < refmin, K2_pass=g["K2"]["pass"])
    g["K4"]["pass"] = bool(g["K4"]["beats_transcriptome"] and g["K2"]["pass"])
    k5 = {}
    for key, hib in (("T1_dir_err", False), ("T2_cycle", True), ("T3_shortcut", False), ("T4_track", False)):
        if hib:
            val = lambda c: 1 - m(c, key)
        else:
            val = lambda c: m(c, key)
        bb = min(BASE, key=val)
        rel = 1 - val("mmn2") / max(val(bb), 1e-12)
        if hib:
            sep = m("mmn2", key) - sd("mmn2", key) > m(bb, key) + sd(bb, key)
        else:
            sep = m("mmn2", key) + sd("mmn2", key) < m(bb, key) - sd(bb, key)
        k5[key] = dict(mmn=m("mmn2", key), best=bb, best_val=m(bb, key), rel_improvement=rel,
                       sd_separated=bool(sep), clearly=bool(rel >= 0.25 and sep))
    g["K5"] = dict(items=k5, pass_=any(v["clearly"] for v in k5.values()))
    return g


def fmt(S, keys):
    rows = []
    hdr = "| class | " + " | ".join(keys) + " |"
    rows.append(hdr); rows.append("|" + "---|" * (len(keys) + 1))
    for c in CLASSES + ["REF_pca_angle", "REF_centroid_polygon", "TRUTH_cohort"]:
        if c not in S:
            continue
        cells = []
        for k in keys:
            if k in S[c]:
                cells.append(f"{S[c][k]['mean']:.3f}+-{S[c][k]['sd']:.3f}" if S[c][k]["sd"] > 0 else f"{S[c][k]['mean']:.3f}")
            else:
                cells.append("-")
        rows.append(f"| {NAMES.get(c, c)} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


if __name__ == "__main__":
    allg = {}
    for ds in sys.argv[1:]:
        res = json.loads((ROOT / f"results_{ds}.json").read_text())
        S = agg(res)
        (ROOT / f"summary_{ds}.json").write_text(json.dumps(S, indent=1))
        print(f"\n=== {ds} ===")
        print(fmt(S, ["par_w2", "par_mmd", "fit_w2", "phase_mae", "phase_corr", "phase_corr_FL", "phase_mae_hold", "within_bin_spearman"]))
        print(fmt(S, ["T1_dir_err", "T2_cycle", "T3_shortcut", "T4_track", "M2t_err", "diverged_frac"]))
        if all(c in S for c in ["mmn2"] + BASE):
            g = gate(S); allg[ds] = g
            print(json.dumps(g, indent=1))
    if len(allg) >= 1:
        real = [d for d in allg if d != "synth"]
        if len(real) == 2:
            a, b2 = real
            allg["K3"] = dict(first=a, second=b2, K2_ratio_first=allg[a]["K2"]["ratio"], K2_ratio_second=allg[b2]["K2"]["ratio"],
                              pass_=bool((allg[a]["K2"]["pass"] and allg[b2]["K2"]["ratio"] <= 0.85) or
                                         (allg[b2]["K2"]["pass"] and allg[a]["K2"]["ratio"] <= 0.85)))
        (ROOT / "gate.json").write_text(json.dumps(allg, indent=1, default=bool))
