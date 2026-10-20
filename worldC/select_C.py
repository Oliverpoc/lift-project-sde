"""Pre-declared choice of world C's rank-dynamics parameters (tau, Omega).  Run BEFORE any C training.

Criterion (fixed before running this scan): world C must be informative about the reviewer's thesis,
i.e. distinguishable from (i) world A, (ii) world B, and (iii) the coupling that each model class
ALREADY selects on A/B data (its "native" position; stored two-time covariances of the 40 A0/B0 runs of
duffing experiment C).  Otherwise "lands near C" and "lands at its native preference" cannot be told apart.
Score(tau, Omega) = min{ eps(C vs A), eps(C vs B), min_class mean_seeds,worlds eps(model vs C) },
eps = relative Frobenius two-time covariance error (M2_in definition). Choose the argmax over the grid
  Omega = 0 (scalar OU):  tau in {0.5, 1, 2, 3, 5, 8}
  Omega in {0.5, 1.0, 1.5, 2.0} (oscillatory rank correlation, period >= pi > 3 snapshot gaps): tau in {5, 10, 20}
Ties within 0.02: prefer Omega = 0 (non-oscillatory), then the larger tau.
usage: python select_C.py  -> data/select_C.json
"""
import json

import numpy as np

from worldc_common import DUFF, HERE, cov2, relfro, world_C_paths

Z = np.load(DUFF / "data" / "truth.npz")
XA0, XB0 = Z["XA0"], Z["XB0"]
CA, CB = cov2(XA0), cov2(XB0)
CLS = ["mmn2", "td", "cfm", "sb"]
native = {c: [np.load(DUFF / "runs" / f"{w}0_{c}_s{s}" / "store.npz")["C0"] for w in "AB" for s in range(5)] for c in CLS}
grid = [(t, 0.0) for t in (0.5, 1, 2, 3, 5, 8)] + [(t, o) for o in (0.5, 1.0, 1.5, 2.0) for t in (5, 10, 20)]
rows = []
for tau, Om in grid:
    CC = cov2(world_C_paths(XA0, tau, Om, seed=2024))
    r = dict(tau=tau, Omega=Om, C_vs_A=relfro(CC, CA), C_vs_B=relfro(CC, CB),
             native_vs_C={c: float(np.mean([relfro(m, CC) for m in native[c]])) for c in CLS})
    r["score"] = min(r["C_vs_A"], r["C_vs_B"], min(r["native_vs_C"].values()))
    rows.append(r)
    print(f"tau={tau:5.1f} Om={Om:3.1f}  C~A {r['C_vs_A']:.3f}  C~B {r['C_vs_B']:.3f}  native->C " +
          " ".join(f"{c}:{v:.2f}" for c, v in r["native_vs_C"].items()) + f"  score {r['score']:.3f}")
best = max(r["score"] for r in rows)
cand = [r for r in rows if r["score"] >= best - 0.02]
cand.sort(key=lambda r: (r["Omega"] != 0.0, -r["tau"]))
choice = cand[0]
print("CHOICE", choice["tau"], choice["Omega"], choice["score"])
(HERE / "data").mkdir(exist_ok=True)
(HERE / "data" / "select_C.json").write_text(json.dumps(dict(grid=rows, choice=choice), indent=1))
