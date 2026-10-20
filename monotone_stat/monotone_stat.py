"""Monotone temporal statistics for the stored L4DC models (reviewer request: a non-covariance statistic).

Proposition 2 (strengthened): for a scalar diffusion that is Markov in x alone, Cov(phi(X_s), psi(X_t)) >= 0
for all non-decreasing phi, psi and s <= t.  Two such statistics are evaluated here on the SAME (t, d) grid
as eps_C (t in {0,...,15}, d in {0.5,1,1.5,2,3,4,6,8}, t+d <= T):
  rho_S(t, t+d)       lagged Spearman rank correlation (phi = F_s, psi = F_t, empirical ranks)
  Ct(t, t+d)          Cov(tanh x_t, tanh x_{t+d})
For the truth (world A), world B, world C, and every stored trained model (all seeds), simulated exactly as in
duffing/evaluate_run.py / pendulum/evaluate.py (x0 = truth sample at t=0, torch generator seed 999, dt 0.1).
No training.  Outputs: results_<system>.json.

usage: python monotone_stat.py duffing|pendulum
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SYS = sys.argv[1]
ROOT = HERE.parent / SYS
sys.path.insert(0, str(ROOT))
import common as cm  # noqa: E402  (duffing/common.py or pendulum/common.py)

torch.set_num_threads(2)
T, DT = cm.CFG["T"], cm.CFG["dt_model"]
C_T = np.arange(0, T + 1e-9, 1.0)
C_D = [0.5, 1, 1.5, 2, 3, 4, 6, 8]
PAIRS = [(t, d) for t in C_T for d in C_D if t + d <= T + 1e-9]   # == MASK_IN of the eps_C grid
idx = lambda t: int(round(t / DT))


def stats(X):
    """X: (steps, N) observed paths. Returns rho_S, Cov(tanh,tanh), Cov(x,x) on PAIRS."""
    used = sorted({idx(t) for t, d in PAIRS} | {idx(t + d) for t, d in PAIRS})
    R, Th = {}, {}
    for i in used:
        r = np.argsort(np.argsort(X[i])).astype(float)
        R[i] = (r - r.mean()) / r.std()
        Th[i] = np.tanh(X[i])
    rho = np.array([np.mean(R[idx(t)] * R[idx(t + d)]) for t, d in PAIRS])
    ct = np.array([np.cov(Th[idx(t)], Th[idx(t + d)])[0, 1] for t, d in PAIRS])
    cx = np.array([np.mean(X[idx(t)] * X[idx(t + d)]) - X[idx(t)].mean() * X[idx(t + d)].mean() for t, d in PAIRS])
    return rho, ct, cx


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def summary(s):
    rho, ct, _ = s
    return dict(rho_frac_neg=float(np.mean(rho < 0)), rho_frac_below_m005=float(np.mean(rho < -0.05)),
                rho_min=float(rho.min()), tanh_frac_neg=float(np.mean(ct < 0)), tanh_min=float(ct.min()))


def lower_bound(v):
    return float(np.linalg.norm(np.minimum(v, 0)) / np.linalg.norm(v))


# ------------------------------------------------------------------ truths ---
if SYS == "duffing":
    Z = np.load(ROOT / "data" / "truth.npz")
    ZC = np.load(HERE.parent / "worldC" / "data" / "truth_C.npz")
    worlds = dict(A=Z["XA0"], B=Z["XB0"], C=ZC["XC0"], A_indep=Z["XA0b"])
    x0 = Z["XA0"][0]
else:
    Z = np.load(ROOT / "truth.npz")
    worlds = dict(A=Z["Xt"], B=Z["XB"], A_indep=Z["Xt2"])
    x0 = Z["Xt"][0]
W = {k: stats(v) for k, v in worlds.items()}
out = dict(system=SYS, pairs=[list(p) for p in PAIRS], n_pairs=len(PAIRS), worlds={}, runs={})
for k, s in W.items():
    e = summary(s)
    e.update(lower_bound_rho=lower_bound(s[0]), lower_bound_tanh=lower_bound(s[1]), lower_bound_cov=lower_bound(s[2]),
             rho=s[0].tolist(), tanh=s[1].tolist())
    for r in ("A", "B", "C"):
        if r in W and r != k:
            e[f"eps_rho_vs{r}"] = rel(s[0], W[r][0]); e[f"eps_C_vs{r}"] = rel(s[2], W[r][2])
    out["worlds"][k] = e
print({k: {kk: round(vv, 4) for kk, vv in v.items() if isinstance(vv, float)} for k, v in out["worlds"].items()},
      flush=True)

# ------------------------------------------------------------------ models ---
if SYS == "duffing":
    groups = {f"{w}0_{c}": ROOT / "runs" for w in "A" for c in ("mmn0", "mmn1", "mmn2", "mmn4", "td", "sb", "cfm")}
    groups.update({f"B0_{c}": ROOT / "runs" for c in ("mmn2", "td", "sb", "cfm")})
    groups.update({f"C0_{c}": HERE.parent / "worldC" / "runs" for c in ("mmn2", "td", "sb", "cfm")})
else:
    groups = {g: ROOT / "runs" for g in ("mmn_dh0", "mmn_dh1", "mmn_dh2", "mmn_dh4", "td", "tdw", "cfm")}


def roll(info, model):
    gen = torch.Generator().manual_seed(999)
    x = torch.tensor(x0, dtype=torch.float32)
    with torch.no_grad():
        if SYS == "duffing":
            X = model.rollout(x, 3 * T, DT, e=0, gen=gen)
        elif info["method"] == "cfm":
            X = model.rollout(x, 3 * T, DT)
        else:
            X = model.rollout(x, 3 * T, DT, gen=gen)
    return X.numpy()[: idx(T) + 1]


t0 = time.time()
for g, base in groups.items():
    for s in range(5):
        tag = f"{g}_s{s}"
        d = base / tag
        if not (d / "model.pt").exists():
            print("MISSING", tag); continue
        info = json.loads((d / "train.json").read_text())
        if SYS == "duffing":
            model = cm.build(info["method"], info["dh"], info["n_exp"])
        else:
            model = cm.build(info["method"], info["dh"], width=info["width"])
        model.load_state_dict(torch.load(d / "model.pt")); model.eval()
        X = roll(info, model)
        st = stats(X)
        e = dict(group=g, seed=s, nonfinite=bool(not np.isfinite(X).all()), **summary(st))
        for r in W:
            if r == "A_indep":
                continue
            e[f"eps_rho_vs{r}"] = rel(st[0], W[r][0])
            e[f"eps_tanh_vs{r}"] = rel(st[1], W[r][1])
            e[f"eps_C_vs{r}"] = rel(st[2], W[r][2])   # check against stored M2_in
        out["runs"][tag] = e
        print(tag, {k: round(v, 4) for k, v in e.items() if isinstance(v, float)}, flush=True)
        del model, X

# ------------------------------------------------------------- group medians ---
agg = {}
for g in groups:
    rs = [v for v in out["runs"].values() if v["group"] == g]
    if not rs:
        continue
    agg[g] = {k: dict(median=float(np.median([r[k] for r in rs])), min=float(np.min([r[k] for r in rs])),
                      max=float(np.max([r[k] for r in rs])), per_seed=[r[k] for r in rs])
              for k in rs[0] if isinstance(rs[0][k], float)}
out["groups"] = agg
out["wall_s"] = time.time() - t0
(HERE / f"results_{SYS}.json").write_text(json.dumps(out, indent=1))
print("done", SYS, f"{out['wall_s']:.0f}s")
