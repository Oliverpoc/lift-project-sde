"""World C data: training snapshots, SB pairs, evaluation truth, and verification of the marginals.
usage: python prep_C.py   (after select_C.py)
Writes data/snap_C0.npy, data/sb_pairs_C0.npy (consumed UNCHANGED by duffing/train.py with WORLD=C),
data/truth_C.npz, data/verify_C.json.
"""
import json
import time

import numpy as np
from scipy.stats import norm

from processc_common import DUFF, HERE, TK, CFG, DT, T, cov2, relfro, world_C_paths, idx
import common as dc   # duffing/common.py (path set in processc_common)

t0 = time.time()
D = HERE / "data"; D.mkdir(exist_ok=True)
ch = json.loads((D / "select_C.json").read_text())["choice"]
TAU, OM = ch["tau"], ch["Omega"]
print("world C: tau", TAU, "Omega", OM)

# ---------------- training snapshots: K unpaired snapshots, 2000 fresh particles each ----------------
# The rank process is stationary (Y_t ~ N(0, I) for all t), so an unpaired snapshot of world C at t_k is
# x = Q_k(Phi(xi)), xi ~ N(0,1) fresh, with Q_k the quantile of the 40000-particle world-A mu0 ensemble
# (the same ensemble that defines world B's training data, duffing/data/B_ens.npy).
Ens = np.load(DUFF / "data" / "B_ens.npy")
snap = np.zeros((CFG["K"], CFG["N_train"]))
for k in range(CFG["K"]):
    rng = np.random.default_rng(72345 + 7 * k + 1)
    snap[k] = np.quantile(Ens[k], norm.cdf(rng.standard_normal(CFG["N_train"])))
np.save(D / "snap_C0.npy", snap)


# ---------------- SB pairs: verbatim copy of duffing/prep.py sinkhorn_pairs, same eps / P / seeds ----------------
def sinkhorn_pairs(a, b, eps, P, seed, iters=3000):
    C = 0.5 * (a[:, None] - b[None, :]) ** 2
    Kmat = np.exp(-C / eps)
    n, m = len(a), len(b)
    u = np.ones(n); v = np.ones(m)
    for i in range(iters):
        u = (1.0 / n) / (Kmat @ v + 1e-300)
        v = (1.0 / m) / (Kmat.T @ u + 1e-300)
        if i % 100 == 0:
            err = np.abs(u * (Kmat @ v) - 1.0 / n).sum()
            if err < 1e-6:
                break
    Pm = u[:, None] * Kmat * v[None, :]
    Pm = Pm / Pm.sum()
    rng = np.random.default_rng(seed)
    idx_ = rng.choice(n * m, size=P, p=Pm.ravel())
    return np.stack([a[idx_ // m], b[idx_ % m]], 1), err


eps = CFG["sb_sigma"] ** 2 * (TK[1] - TK[0])
out = []
for k in range(CFG["K"] - 1):
    pr, err = sinkhorn_pairs(snap[k], snap[k + 1], eps, 100000, 900 + k)
    out.append(pr)
np.save(D / "sb_pairs_C0.npy", np.stack(out).astype(np.float32))
print("sb pairs done, last sinkhorn err", err, f"{time.time()-t0:.0f}s", flush=True)

# ---------------- evaluation truth on the 0.1 grid, t in [0, 3T] (as duffing truth.npz) ----------------
Z = np.load(DUFF / "data" / "truth.npz")
XA0, XA0b, XB0 = Z["XA0"], Z["XA0b"], Z["XB0"]
XC0 = world_C_paths(XA0, TAU, OM, seed=5)          # world B used U from seed 5 on XA0 quantiles; C likewise
XC0b = world_C_paths(XA0b, TAU, OM, seed=6)        # independent replica (noise floor)
np.savez_compressed(D / "truth_C.npz", XC0=XC0, XC0b=XC0b, tau=TAU, Omega=OM)

# ---------------- verification ----------------
CA, CAb, CB, CC, CCb = cov2(XA0), cov2(XA0b), cov2(XB0), cov2(XC0), cov2(XC0b)
grid01 = np.arange(0, T + 1e-9, DT)
v = dict(tau=TAU, Omega=OM,
         eval_W2_C_vs_A_at_tk=[dc.w2_1d(XC0[idx(t)], XA0[idx(t)]) for t in TK],
         eval_W2_C_vs_A_max_window_0p1grid=float(max(dc.w2_1d(XC0[idx(t)], XA0[idx(t)]) for t in grid01)),
         eval_W2_B_vs_A_at_tk=[dc.w2_1d(XB0[idx(t)], XA0[idx(t)]) for t in TK],
         eval_W2_A_vs_Ab_noise_at_tk=[dc.w2_1d(XA0[idx(t)], XA0b[idx(t)]) for t in TK],
         train_W2_C0data_vs_A0data_at_tk=[dc.w2_1d(a, b) for a, b in zip(snap, dc.load_data("A", 0))],
         train_W2_B0data_vs_A0data_at_tk=[dc.w2_1d(a, b) for a, b in zip(dc.load_data("B", 0), dc.load_data("A", 0))],
         uniformity_KS_U=None)
for k in [k for k in v if k.endswith("_at_tk")]:
    v[k.replace("_at_tk", "_max")] = float(np.max(v[k]))
# eps_C between worlds (row = covariance compared, column = reference world)
v["epsC"] = {"C_vs_A": relfro(CC, CA), "C_vs_B": relfro(CC, CB), "A_vs_C": relfro(CA, CC), "B_vs_C": relfro(CB, CC),
             "A_vs_B": relfro(CA, CB), "B_vs_A": relfro(CB, CA),
             "noise_Ab_vs_A": relfro(CAb, CA), "noise_Cb_vs_C": relfro(CCb, CC)}
# rank correlation check: empirical corr of Phi^{-1}(U) at lag d vs exp(-d/tau) cos(Omega d)
from processc_common import rank_paths  # noqa: E402
U = rank_paths(20000, idx(8.0), DT, TAU, OM, seed=11)
Y = norm.ppf(np.clip(U, 1e-12, 1 - 1e-12))
v["rank_corr_check"] = [dict(d=d, empirical=float(np.corrcoef(Y[0], Y[idx(d)])[0, 1]),
                             exact=float(np.exp(-d / TAU) * np.cos(OM * d))) for d in (0.5, 1, 2, 3, 4, 6, 8)]
from scipy.stats import kstest  # noqa: E402
v["uniformity_KS_U"] = dict(t0=float(kstest(U[0], "uniform").pvalue), t8=float(kstest(U[-1], "uniform").pvalue))
(D / "verify_C.json").write_text(json.dumps(v, indent=1))
print(json.dumps({k: x for k, x in v.items() if not k.endswith("_at_tk")}, indent=1))
print("prep_C done", f"{time.time()-t0:.0f}s")
