"""Generate all training data, Schrodinger-bridge pair samples, and the evaluation truth store.
usage: python prep.py"""
import time

import numpy as np

from common import (CFG, TK, ROOT, load_data, simulate, B_quantile_ensemble, B_map, mu_params,
                    build, nparams, w2_1d, save_json)

t0 = time.time()
(ROOT / "data").mkdir(exist_ok=True)
for w in "AB":
    for e in (0, 1):
        d = load_data(w, e)
        print("data", w, e, d.shape, np.round(d.mean(1)[:6], 3), flush=True)


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
    idx = rng.choice(n * m, size=P, p=Pm.ravel())
    return np.stack([a[idx // m], b[idx % m]], 1), err


eps = CFG["sb_sigma"] ** 2 * (TK[1] - TK[0])
for w in "AB":
    for e in (0, 1):
        f = ROOT / "data" / f"sb_pairs_{w}{e}.npy"
        if f.exists():
            continue
        d = load_data(w, e)
        out = []
        for k in range(CFG["K"] - 1):
            pr, err = sinkhorn_pairs(d[k], d[k + 1], eps, 100000, 900 + k)
            out.append(pr)
        np.save(f, np.stack(out).astype(np.float32))
        print("sb pairs", w, e, "last sinkhorn err", err, f"{time.time()-t0:.0f}s", flush=True)

# ------------------------------------------------------------ evaluation truth ---
NE, T, dt, delta = CFG["N_eval"], CFG["T"], CFG["dt_model"], CFG["delta"]
tg, XA0, VA0 = simulate(NE, 3 * T, CFG["eval_seed"], save_dt=dt)
_, XA0b, _ = simulate(NE, 3 * T, CFG["eval_seed"] + 1, save_dt=dt)
_, XAi, _ = simulate(NE, T, CFG["eval_seed"], save_dt=dt, intervene=(T / 2, delta))
_, XAib, _ = simulate(NE, T, CFG["eval_seed"] + 1, save_dt=dt, intervene=(T / 2, delta))
_, XA1, _ = simulate(NE, 2 * T, CFG["eval_seed"] + 2, save_dt=dt, params=mu_params(1))
_, XA1b, _ = simulate(NE, 2 * T, CFG["eval_seed"] + 3, save_dt=dt, params=mu_params(1))
# world B on the eval grid: rank-preserving process built from the mu0 eval marginals (as earlier Duffing study)
U = np.random.default_rng(5).random(NE)
XB0 = np.stack([np.quantile(XA0[i], U) for i in range(XA0.shape[0])])
# world B under mu1: x0 ~ x-law of mu1, x_t = T_{0,t}(x0) using the mu0 marginals
XB1 = np.stack([B_map(XA1[0], XA0[0], XA0[i]) for i in range(int(round(2 * T / dt)) + 1)])
np.savez_compressed(ROOT / "data" / "truth.npz", tg=tg, XA0=XA0, VA0=VA0, XA0b=XA0b, XAi=XAi, XAib=XAib,
                    XA1=XA1, XA1b=XA1b, XB0=XB0, XB1=XB1)

# distinguishability of A and B under mu0 / mu1 (snapshot marginals only)
Ens = B_quantile_ensemble()
idx = lambda t: int(round(t / dt))
dist = dict(
    W2_A0_vs_B0_at_tk=[w2_1d(XA0[idx(t)], XB0[idx(t)]) for t in TK],
    W2_A1_vs_B1_at_tk=[w2_1d(XA1[idx(t)], XB1[idx(t)]) for t in TK],
    W2_A1_vs_A1b_noise_at_tk=[w2_1d(XA1[idx(t)], XA1b[idx(t)]) for t in TK],
    W2_A0_vs_A0b_noise_at_tk=[w2_1d(XA0[idx(t)], XA0b[idx(t)]) for t in TK],
)
for k in list(dist):
    dist[k.replace("_at_tk", "_max")] = float(np.max(dist[k]))
    dist[k.replace("_at_tk", "_mean")] = float(np.mean(dist[k][1:]))
# data-level check: B training snapshots vs A training snapshots
dist["train_W2_A0data_vs_B0data_max"] = float(max(w2_1d(a, b) for a, b in zip(load_data("A", 0), load_data("B", 0))))
dist["train_W2_A1data_vs_B1data_max"] = float(max(w2_1d(a, b) for a, b in zip(load_data("A", 1), load_data("B", 1))))
dist["train_W2_A0data_vs_A1data_max"] = float(max(w2_1d(a, b) for a, b in zip(load_data("A", 0), load_data("A", 1))))
pc = {}
for meth, dh, ne in [("mmn", 0, 1), ("mmn", 1, 1), ("mmn", 2, 1), ("mmn", 4, 1), ("td", 0, 1), ("cfm", 0, 1),
                     ("sb", 0, 1), ("ref", 0, 1), ("mmn", 1, 2), ("mmn", 2, 2), ("tdlab", 0, 2), ("tdsh", 0, 2),
                     ("cfm", 0, 2), ("sb", 0, 2)]:
    pc[f"{meth}{dh if meth=='mmn' else ''}_nexp{ne}"] = nparams(build(meth, dh, ne))
dist["nparams"] = pc
save_json(dist, ROOT / "data" / "distinguishability.json")
print({k: v for k, v in dist.items() if not isinstance(v, list)})
print("prep done", f"{time.time()-t0:.0f}s")
