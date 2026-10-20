"""Optional neural overlay: evaluate lifted (d_h=2) and obs-TD trained (unchanged duffing train.py, Fermi) on the
linear-Gaussian snapshots against the EXACT covariances of lg_exact.py.
usage: python evaluate_L.py   (runs/L0_*/model.pt synced back from Fermi)  -> results.json["neural"]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "duffing"))
import common as dc  # noqa: E402

torch.set_num_threads(2)
E = np.load(HERE / "data" / "exact_cov.npz")
CA, CB, MASK, C_T, C_D = E["CA"], E["CB"], E["mask"], E["C_T"], E["C_D"]
XL = np.load(HERE / "data" / "truth_L.npz")["XL"]
DT, T = 0.1, 15.0
idx = lambda t: int(round(t / DT))


def cov(X):
    C = np.full(CA.shape, np.nan)
    for i, t in enumerate(C_T):
        for j, d in enumerate(C_D):
            if MASK[i, j]:
                a, b = X[idx(t)], X[idx(t + d)]
                C[i, j] = np.mean(a * b) - a.mean() * b.mean()
    return C


def eps(C, R):
    return float(np.linalg.norm((C - R)[MASK]) / np.linalg.norm(R[MASK]))


out = {}
for tag in sorted(p.name for p in (HERE / "runs").glob("L0_*") if (p / "model.pt").exists()):
    d = HERE / "runs" / tag
    info = json.loads((d / "train.json").read_text())
    m = dc.build(info["method"], info["dh"], info["n_exp"]); m.load_state_dict(torch.load(d / "model.pt")); m.eval()
    with torch.no_grad():
        X = m.rollout(torch.tensor(XL[0], dtype=torch.float32), T, DT, gen=torch.Generator().manual_seed(999)).numpy()
    C = cov(X)
    out[tag] = dict(M0=float(np.mean([dc.w2_1d(X[idx(t)], XL[idx(t)]) for t in dc.TK[1:]])),
                    eps_vs_true=eps(C, CA), eps_vs_comonotone=eps(C, CB), C=C.tolist(), wall_s=info["wall_s"])
    print(tag, {k: round(v, 4) for k, v in out[tag].items() if isinstance(v, float)})
# noise floor of the sampled covariance itself (4000-particle simulation of the true system vs exact)
out["noise_floor_sampled_truth_vs_exact"] = eps(cov(XL), CA)
print("noise floor", out["noise_floor_sampled_truth_vs_exact"])
r = json.loads((HERE / "results.json").read_text()); r["neural"] = out
(HERE / "results.json").write_text(json.dumps(r, indent=1))
