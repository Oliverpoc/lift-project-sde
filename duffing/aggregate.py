"""Collect runs/*/eval.json into results.json (per group: per-seed, mean, sd, median) + cross-seed CCA.
usage: python aggregate.py"""
import itertools
import json
import re
from collections import defaultdict

import numpy as np

from common import ROOT, save_json

METRICS = ["M0", "M0_ho", "M1", "clip_M1", "M2_in", "M2_out", "clip_M2_out", "M2_in_vsB", "M3",
           "mu1_M0", "mu1_M0_ho", "mu1_M2_in", "mu1_M2_in_vsB", "R2_mlp_xh", "R2_mlp_x", "R2_lin_xh",
           "wall_s", "nparams"]
groups = defaultdict(list)
for f in sorted((ROOT / "runs").glob("*/eval.json")):
    r = json.loads(f.read_text())
    g = re.sub(r"_s\d+$", "", r["tag"])
    groups[g].append(r)


def stats(vals):
    a = np.array(vals, dtype=float)
    fin = a[np.isfinite(a)]
    return dict(per_seed=[float(v) for v in a], mean=float(a.mean()) if len(a) else None,
                sd=float(a.std(ddof=1)) if len(a) > 1 else 0.0, median=float(np.median(a)),
                n=int(len(a)), n_nonfinite=int(len(a) - len(fin)))


out = dict(groups={})
for g, rs in sorted(groups.items()):
    rs = sorted(rs, key=lambda r: r["seed"])
    d = dict(seeds=[r["seed"] for r in rs], nonfinite_seeds=[r["seed"] for r in rs if r["nonfinite"]],
             frac_absx_gt5_t45=[r["frac_absx_gt5"]["45"] for r in rs])
    for m in METRICS:
        if m in rs[0]:
            d[m] = stats([r.get(m, np.nan) for r in rs])
    if "clip_M1" in d:   # better continuation per seed (baseline-favouring)
        d["M1_best"] = stats([min(r["M1"], r["clip_M1"]) for r in rs])
    out["groups"][g] = d


def first_cca(A, B):
    A = A - A.mean(0); B = B - B.mean(0)
    def wh(M):
        U_, S_, _ = np.linalg.svd(M, full_matrices=False)
        return U_[:, S_ > 1e-8 * S_[0]]
    return float(np.linalg.svd(wh(A).T @ wh(B), compute_uv=False)[0])


cca = {}
for dh in (1, 2, 4):
    H = {s: np.load(p) for s in range(5) if (p := ROOT / "runs" / f"A0_mmn{dh}_s{s}" / "h_inf.npy").exists()}
    H = {s: h for s, h in H.items() if np.isfinite(h).all()}
    pairs = {f"{a}-{b}": float(np.mean([first_cca(H[a][p], H[b][p]) for p in range(H[a].shape[0])]))
             for a, b in itertools.combinations(sorted(H), 2)}
    cca[f"dh{dh}"] = dict(pairs=pairs, mean=float(np.mean(list(pairs.values()))) if pairs else None,
                          sd=float(np.std(list(pairs.values()), ddof=1)) if len(pairs) > 1 else None)
out["latent_cca_cross_seed"] = cca
out["distinguishability"] = json.loads((ROOT / "data" / "distinguishability.json").read_text())
save_json(out, ROOT / "results.json")
for g, d in out["groups"].items():
    print(f"{g:14s} n={d['M0']['n']} " + " ".join(
        f"{m}={d[m]['mean']:.3g}" for m in ("M0", "M0_ho", "M1", "M2_in", "M2_in_vsB", "M3", "mu1_M0", "mu1_M2_in") if m in d))
print(cca)
