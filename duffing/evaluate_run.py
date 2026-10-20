"""Evaluate one trained run on fresh simulator data (data/truth.npz).
usage: python evaluate_run.py TAG   -> runs/TAG/eval.json, runs/TAG/store.npz
Metrics (same definitions as the earlier Duffing study, plus held-out-time marginal W2 and mu1 metrics):
  M0     mean W2 over training times t_k (k=1..15) vs the training world's marginals
  M0_ho  mean W2 over held-out times t = 0.5,1.5,...,14.5 (inside the window, never trained on)
  M1     mean W2 at t = 1.5T, 2T, 3T (extrapolation) vs world-A marginals
  M2_in  relative Frobenius error of the two-time covariance C(t,t+d), t+d <= T, vs truth A
  M2_out same for T < t+d <= 2T
  M2_in_vsB  same as M2_in but vs world B
  M3     mean W2 after the intervention x -> x+0.6 at T/2 vs truth A
  M4     R^2(v | x,h) by MLP probe (MMN, world A, mu0-only runs)
"""
import json
import sys
import time

import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

from common import CFG, TK, ROOT, build, w2_1d, save_json

torch.set_num_threads(2)
tag = sys.argv[1]
run = ROOT / "runs" / tag
t0w = time.time()
info = json.loads((run / "train.json").read_text())
model = build(info["method"], info["dh"], info["n_exp"])
model.load_state_dict(torch.load(run / "model.pt")); model.eval()
Z = np.load(ROOT / "data" / "truth.npz")
XA0, VA0, XAi, XA1, XB0, XB1 = Z["XA0"], Z["VA0"], Z["XAi"], Z["XA1"], Z["XB0"], Z["XB1"]
T, dt, delta = CFG["T"], CFG["dt_model"], CFG["delta"]
idx = lambda t: int(round(t / dt))
EXT_T = [1.5 * T, 2 * T, 3 * T]
INT_T = [T / 2 + s for s in (0.5, 1, 2, 4, 7.5)]
HO_T = list(np.arange(0.5, T, 1.0))
C_T = np.arange(0, 2 * T + 1e-9, 1.0)
C_D = [0.5, 1, 1.5, 2, 3, 4, 6, 8]
TT, DD = np.meshgrid(C_T, C_D, indexing="ij")
MASK_IN = (TT + DD <= T + 1e-9)
MASK_OUT = (TT + DD > T + 1e-9) & (TT + DD <= 2 * T + 1e-9)


def cov2(X):
    C = np.full((len(C_T), len(C_D)), np.nan)
    for i, t in enumerate(C_T):
        for j, d in enumerate(C_D):
            if t + d <= 2 * T + 1e-9 and idx(t + d) < X.shape[0]:
                a, b = X[idx(t)], X[idx(t + d)]
                with np.errstate(all="ignore"):
                    C[i, j] = np.mean(a * b) - a.mean() * b.mean()
    return C


def relfro(C, Cref, mask):
    v = (C - Cref)[mask]
    if not np.isfinite(v).all():
        return float("inf")
    return float(np.linalg.norm(v) / np.linalg.norm(Cref[mask]))


C_A0, C_B0, C_A1, C_B1 = cov2(XA0), cov2(XB0), cov2(XA1), cov2(XB1)
XW0 = XA0 if info["world"] == "A" else XB0
XW1 = XA1 if info["world"] == "A" else XB1


def roll(e, x0, t_end, intervene=None):
    gen = torch.Generator().manual_seed(999)
    x0 = torch.tensor(x0, dtype=torch.float32)
    with torch.no_grad():
        if info["method"] == "ref":
            xa = torch.tensor(XA0[1], dtype=torch.float32)
            X = torch.cat([x0[None], model.rollout(x0, xa, t_end, dt, gen=gen, t_start=dt, intervene=intervene)])
        else:
            X = model.rollout(x0, t_end, dt, e=e, gen=gen, intervene=intervene)
    return X.numpy()


def finite(x):
    return float(x) if np.isfinite(x) else float("inf")


m = dict(tag=tag, **{k: info[k] for k in ("world", "exps", "method", "seed", "dh", "n_exp", "nparams", "wall_s")})
X = roll(0, XA0[0], 3 * T)
Xi = roll(0, XA0[0], T, intervene=(T / 2, delta))
m["M0_per_t"] = [w2_1d(X[idx(t)], XW0[idx(t)]) for t in TK[1:]]
m["M0"] = finite(np.mean(m["M0_per_t"]))
m["M0_ho"] = finite(np.mean([w2_1d(X[idx(t)], XW0[idx(t)]) for t in HO_T]))
m["M1_per_t"] = [w2_1d(X[idx(t)], XA0[idx(t)]) for t in EXT_T]
m["M1"] = finite(np.mean(m["M1_per_t"]))
C = cov2(X)
m["M2_in"] = relfro(C, C_A0, MASK_IN)
m["M2_out"] = relfro(C, C_A0, MASK_OUT)
m["M2_in_vsB"] = relfro(C, C_B0, MASK_IN)
m["M3_per_t"] = [w2_1d(Xi[idx(t)], XAi[idx(t)]) for t in INT_T]
m["M3"] = finite(np.mean(m["M3_per_t"]))
m["nonfinite"] = bool(not np.isfinite(X).all())
m["frac_absx_gt5"] = {str(t): float(np.mean(~(np.abs(X[idx(t)]) <= 5))) for t in (15, 22.5, 30, 45)}
store = dict(mean0=X.mean(1), q0=np.quantile(X, [0.05, 0.25, 0.5, 0.75, 0.95], axis=1), C0=C,
             qi=np.quantile(Xi, [0.05, 0.5, 0.95], axis=1))
if getattr(model, "time_dep", False) and info["method"] != "ref":
    model.tclip = T
    Xc = roll(0, XA0[0], 3 * T)
    model.tclip = float("inf")
    m["clip_M1"] = finite(np.mean([w2_1d(Xc[idx(t)], XA0[idx(t)]) for t in EXT_T]))
    m["clip_M2_out"] = relfro(cov2(Xc), C_A0, MASK_OUT)
    store["mean0_clip"] = Xc.mean(1)
if info["n_exp"] > 1:
    X1 = roll(1, XA1[0], 2 * T)
    m["mu1_M0"] = finite(np.mean([w2_1d(X1[idx(t)], XW1[idx(t)]) for t in TK[1:]]))
    m["mu1_M0_ho"] = finite(np.mean([w2_1d(X1[idx(t)], XW1[idx(t)]) for t in HO_T]))
    C1 = cov2(X1)
    m["mu1_M2_in"] = relfro(C1, C_A1, MASK_IN)
    m["mu1_M2_in_vsB"] = relfro(C1, C_B1, MASK_IN)
    store.update(mean1=X1.mean(1), q1=np.quantile(X1, [0.05, 0.25, 0.5, 0.75, 0.95], axis=1), C1=C1)

# ------------------------------------------------ latent probe (MMN, world A, mu0 only) ---
PROBE_T = [2.0, 5.0, 8.0, 11.0]; NP = 2000; WIN = 10
if info["method"] == "mmn" and info["dh"] > 0 and info["world"] == "A" and info["exps"] == "0":
    Hs = []
    for tp in PROBE_T:
        i0 = idx(tp)
        xobs = torch.tensor(XA0[i0:i0 + WIN + 1, :NP], dtype=torch.float32)
        torch.manual_seed(0)
        h = torch.zeros(NP, model.dh, requires_grad=True)
        opt = torch.optim.Adam([h], lr=0.05)
        for _ in range(300):
            z = torch.cat([xobs[0][:, None], h], 1)
            err = 0.0
            for s in range(1, WIN + 1):
                z = model.step(z, None, dt, 0, None, False)
                err = err + ((z[:, 0] - xobs[s]) ** 2).sum()
            opt.zero_grad(); err.backward(); opt.step()
        Hs.append(h.detach().numpy().copy())
    H = np.stack(Hs)
    np.save(run / "h_inf.npy", H)
    xs = np.concatenate([XA0[idx(t), :NP] for t in PROBE_T])
    vs = np.concatenate([VA0[idx(t), :NP] for t in PROBE_T])
    perm = np.random.default_rng(0).permutation(len(xs)); tr, te = perm[:len(xs) // 2], perm[len(xs) // 2:]
    for name, F in (("x", xs[:, None]), ("xh", np.concatenate([xs[:, None], H.reshape(-1, H.shape[-1])], 1))):
        F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
        m[f"R2_lin_{name}"] = float(LinearRegression().fit(F[tr], vs[tr]).score(F[te], vs[te]))
        m[f"R2_mlp_{name}"] = float(MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=600, random_state=0)
                                    .fit(F[tr], vs[tr]).score(F[te], vs[te]))
m["eval_wall_s"] = time.time() - t0w
np.savez_compressed(run / "store.npz", **store)
save_json(m, run / "eval.json")
print(tag, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items() if not isinstance(v, (list, dict))})
