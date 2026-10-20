"""Evaluate one run of the shared-vs-conditioned experiment.  usage: python evaluate.py TAG
For every experiment e the run was trained on: M0_e (training-window W2 vs truth A under mu_e),
epsC_e (two-time covariance rel. Frobenius error vs truth A under mu_e, t+d <= T), dB_e (same vs world B).
Definitions identical to duffing/evaluate_run.py (M0, M2_in, M2_in_vsB, mu1_*)."""
import json
import sys
import time

import numpy as np
import torch

from common import CFG, TK, ROOT, build, w2_1d, save_json

torch.set_num_threads(2)
tag = sys.argv[1]
run = ROOT / "runs" / tag
t0w = time.time()
info = json.loads((run / "train.json").read_text())
model = build(info["method"], info["dh"], info["n_exp"])
model.load_state_dict(torch.load(run / "model.pt")); model.eval()
Z = np.load(ROOT / "data" / "truth.npz")
T, dt = CFG["T"], CFG["dt_model"]
idx = lambda t: int(round(t / dt))
C_T = np.arange(0, 2 * T + 1e-9, 1.0)
C_D = [0.5, 1, 1.5, 2, 3, 4, 6, 8]
TT, DD = np.meshgrid(C_T, C_D, indexing="ij")
MASK_IN = (TT + DD <= T + 1e-9)


def cov2(X):
    C = np.full((len(C_T), len(C_D)), np.nan)
    for i, t in enumerate(C_T):
        for j, d in enumerate(C_D):
            if t + d <= 2 * T + 1e-9 and idx(t + d) < X.shape[0]:
                a, b = X[idx(t)], X[idx(t + d)]
                with np.errstate(all="ignore"):
                    C[i, j] = np.mean(a * b) - a.mean() * b.mean()
    return C


def relfro(C, Cref):
    v = (C - Cref)[MASK_IN]
    if not np.isfinite(v).all():
        return float("inf")
    return float(np.linalg.norm(v) / np.linalg.norm(Cref[MASK_IN]))


TRUTH = {0: (Z["XA0"], Z["XB0"]), 1: (Z["XA1"], Z["XB1"])}
E = [int(c) for c in info["exps"]]
m = dict(tag=tag, **{k: info[k] for k in ("world", "exps", "method", "seed", "dh", "n_exp", "nparams", "wall_s")})
for ei, e in enumerate(E):
    XA, XB = TRUTH[e]
    lab = e if info["n_exp"] > 1 else 0           # conditioned / two-experiment models get label e
    gen = torch.Generator().manual_seed(999)
    with torch.no_grad():
        X = model.rollout(torch.tensor(XA[0], dtype=torch.float32), T, dt, e=lab, gen=gen).numpy()
    m[f"M0_{e}_per_t"] = [w2_1d(X[idx(t)], XA[idx(t)]) for t in TK[1:]]
    M0 = float(np.mean(m[f"M0_{e}_per_t"]))
    m[f"M0_{e}"] = M0 if np.isfinite(M0) else float("inf")
    C = cov2(X)
    m[f"epsC_{e}"] = relfro(C, cov2(XA))
    m[f"dB_{e}"] = relfro(C, cov2(XB))
    np.savez_compressed(run / f"store_e{e}.npz", C=C, mean=X.mean(1), q=np.quantile(X, [0.05, 0.5, 0.95], axis=1))
m["eval_wall_s"] = time.time() - t0w
save_json(m, run / "eval.json")
print(tag, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items() if not isinstance(v, (list, dict))})
